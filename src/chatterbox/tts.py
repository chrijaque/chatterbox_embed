from dataclasses import dataclass
from pathlib import Path
import os
import tempfile
import logging
import re
import time
import unicodedata
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple, Dict, Union
from datetime import datetime
from enum import Enum

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
import torchaudio
import numpy as np

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen, VoiceProfile
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

# Configure logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import nltk
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    # Force download and setup NLTK punkt tokenizer
    try:
        nltk.download('punkt', quiet=True)
        nltk.data.find('tokenizers/punkt')
        NLTK_AVAILABLE = True
        logger.info("✅ NLTK punkt tokenizer available")
    except Exception as e:
        logger.warning(f"⚠️ NLTK setup failed: {e}")
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("⚠️ nltk not available - will use simple text splitting")

try:
    from pydub import AudioSegment, effects
    PYDUB_AVAILABLE = True
    logger.info("✅ pydub available for audio processing")
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("⚠️ pydub not available - will use torchaudio for audio processing")

REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


# ================== Smart Chunking and Adaptive Parameters ==================

class ContentType(Enum):
    DIALOGUE = "dialogue"
    NARRATIVE = "narrative" 
    DESCRIPTIVE = "descriptive"
    TRANSITION = "transition"


@dataclass
class ChunkInfo:
    """Information about a text chunk with context analysis"""
    id: int
    text: str
    content_type: ContentType
    char_count: int
    word_count: int
    is_first_chunk: bool
    is_last_chunk: bool
    ending_punctuation: str
    paragraph_break_after: bool
    dialogue_ratio: float
    complexity_score: float


class SmartChunker:
    """Intelligent text chunking with content awareness"""
    
    def __init__(self):
        self.paragraph_markers = ['\n\n', '---', '***', '\n \n']
        self.dialogue_markers = ['"', "'", '"', '"', '«', '»']
        self.narrative_indicators = ['suddenly', 'meanwhile', 'then', 'next', 'after', 'before', 'during', 'while']
        self.transition_words = ['however', 'therefore', 'nevertheless', 'furthermore', 'moreover', 'consequently']
        
        # Punctuation weights for optimal break points
        self.punctuation_weights = {
            '.': 1.0,   # Strong break - perfect for chunking
            '!': 1.0,   # Strong break - perfect for chunking  
            '?': 1.0,   # Strong break - perfect for chunking
            ';': 0.7,   # Medium break - good for chunking
            ':': 0.5,   # Weak break - use if needed
            ',': 0.3,   # Very weak break - avoid if possible
            '—': 0.6,   # Medium break - dialogue/emphasis
            '–': 0.6    # Medium break - dialogue/emphasis
        }
    
    def analyze_content_type(self, text: str) -> ContentType:
        """Analyze text to determine its content type"""
        text_lower = text.lower()
        
        # Calculate dialogue ratio
        dialogue_count = sum(1 for c in text if c in self.dialogue_markers)
        dialogue_ratio = dialogue_count / len(text) if text else 0
        
        # Check for dialogue content
        if dialogue_ratio > 0.02 or text.count('"') >= 2:
            return ContentType.DIALOGUE
        
        # Check for narrative indicators
        narrative_score = sum(1 for word in self.narrative_indicators if word in text_lower)
        if narrative_score >= 2:
            return ContentType.NARRATIVE
        
        # Check for transition words
        transition_score = sum(1 for word in self.transition_words if word in text_lower)
        if transition_score >= 1:
            return ContentType.TRANSITION
            
        # Default to descriptive
        return ContentType.DESCRIPTIVE
    
    def calculate_complexity_score(self, text: str) -> float:
        """Calculate text complexity based on various factors"""
        if not text:
            return 0.0
            
        words = text.split()
        if not words:
            return 0.0
            
        # Average word length
        avg_word_length = sum(len(word.strip('.,!?;:"')) for word in words) / len(words)
        
        # Sentence count
        sentence_endings = sum(1 for c in text if c in '.!?')
        avg_sentence_length = len(words) / max(sentence_endings, 1)
        
        # Punctuation density
        punctuation_count = sum(1 for c in text if c in '.,!?;:"-')
        punctuation_density = punctuation_count / len(text)
        
        # Complex punctuation (colons, semicolons, dashes)
        complex_punct = sum(1 for c in text if c in ';:—–')
        complex_punct_ratio = complex_punct / len(text)
        
        # Normalize and combine factors
        complexity = (
            (avg_word_length - 4) * 0.3 +  # Word complexity
            (avg_sentence_length - 10) * 0.2 + # Sentence complexity  
            punctuation_density * 50 * 0.3 +   # Punctuation complexity
            complex_punct_ratio * 100 * 0.2    # Advanced punctuation
        )
        
        return max(0, min(10, complexity))  # Clamp between 0-10
    
    def find_optimal_break_point(self, text: str, start_pos: int, max_chars: int) -> Tuple[int, float]:
        """Find the optimal position to break text based on punctuation weights"""
        if start_pos + max_chars >= len(text):
            return len(text), 1.0
        
        # Look for break points in the target range
        search_start = start_pos + max_chars // 2  # Prefer middle to end of range
        search_end = min(start_pos + max_chars, len(text))
        
        best_pos = search_end
        best_score = 0.0
        
        for i in range(search_start, search_end):
            char = text[i]
            if char in self.punctuation_weights:
                # Calculate score based on punctuation weight and position preference
                punct_weight = self.punctuation_weights[char]
                
                # Prefer positions closer to the ideal target
                ideal_pos = start_pos + int(max_chars * 0.8)  # 80% through the chunk
                position_preference = 1.0 - abs(i - ideal_pos) / max_chars
                
                # Bonus for space after punctuation (cleaner breaks)
                space_bonus = 0.1 if i + 1 < len(text) and text[i + 1] == ' ' else 0
                
                total_score = punct_weight * 0.7 + position_preference * 0.2 + space_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_pos = i + 1  # Include the punctuation in the chunk
        
        return best_pos, best_score
    
    def smart_chunk(self, text: str, target_chars: int = 400, max_chars: int = 600) -> List[ChunkInfo]:
        """Create intelligent chunks with content awareness"""
        if not text or not text.strip():
            return []
        
        # Clean text first
        text = text.strip()
        
        # Handle paragraph breaks
        paragraphs = []
        current_paragraph = ""
        
        for line in text.split('\n'):
            line = line.strip()
            if line:
                if current_paragraph:
                    current_paragraph += " " + line
                else:
                    current_paragraph = line
            else:
                if current_paragraph:
                    paragraphs.append(current_paragraph)
                    current_paragraph = ""
        
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        # Create chunks from paragraphs
        chunks = []
        chunk_id = 0
        
        for para_idx, paragraph in enumerate(paragraphs):
            para_chunks = self._chunk_paragraph(paragraph, target_chars, max_chars, chunk_id)
            
            # Mark paragraph breaks
            for i, chunk_info in enumerate(para_chunks):
                chunk_info.paragraph_break_after = (i == len(para_chunks) - 1 and para_idx < len(paragraphs) - 1)
                chunks.append(chunk_info)
                chunk_id += 1
        
        # Mark first and last chunks
        if chunks:
            chunks[0].is_first_chunk = True
            chunks[-1].is_last_chunk = True
        
        logger.info(f"🧠 Smart chunking: {len(text)} chars → {len(chunks)} chunks")
        logger.info(f"   Content types: {self._get_content_type_distribution(chunks)}")
        
        return chunks
    
    def _chunk_paragraph(self, paragraph: str, target_chars: int, max_chars: int, start_id: int) -> List[ChunkInfo]:
        """Chunk a single paragraph intelligently"""
        if len(paragraph) <= max_chars:
            # Single chunk
            return [self._create_chunk_info(start_id, paragraph, False, False)]
        
        chunks = []
        current_pos = 0
        chunk_id = start_id
        
        while current_pos < len(paragraph):
            # Find optimal break point
            break_pos, break_score = self.find_optimal_break_point(paragraph, current_pos, target_chars)
            
            # Extract chunk text
            chunk_text = paragraph[current_pos:break_pos].strip()
            
            if chunk_text:
                chunk_info = self._create_chunk_info(chunk_id, chunk_text, 
                                                   len(chunks) == 0, break_pos >= len(paragraph))
                chunks.append(chunk_info)
                chunk_id += 1
            
            current_pos = break_pos
        
        return chunks
    
    def _create_chunk_info(self, chunk_id: int, text: str, is_first: bool, is_last: bool) -> ChunkInfo:
        """Create a ChunkInfo object with full analysis"""
        content_type = self.analyze_content_type(text)
        complexity = self.calculate_complexity_score(text)
        
        # Get ending punctuation
        ending_punct = text.rstrip()[-1] if text.rstrip() else '.'
        
        # Calculate dialogue ratio
        dialogue_count = sum(1 for c in text if c in self.dialogue_markers)
        dialogue_ratio = dialogue_count / len(text) if text else 0
        
        return ChunkInfo(
            id=chunk_id,
            text=text,
            content_type=content_type,
            char_count=len(text),
            word_count=len(text.split()),
            is_first_chunk=is_first,
            is_last_chunk=is_last,
            ending_punctuation=ending_punct,
            paragraph_break_after=False,  # Set by caller
            dialogue_ratio=dialogue_ratio,
            complexity_score=complexity
        )
    
    def _get_content_type_distribution(self, chunks: List[ChunkInfo]) -> Dict[str, int]:
        """Get distribution of content types for logging"""
        distribution = {}
        for chunk in chunks:
            content_type = chunk.content_type.value
            distribution[content_type] = distribution.get(content_type, 0) + 1
        return distribution


class AdaptiveParameterManager:
    """Manages adaptive parameters based on content analysis"""
    
    # Base parameter profiles for different content types
    CONTENT_PROFILES = {
        ContentType.DIALOGUE: {
            "temperature": 0.7,         # Lower for more consistent speech patterns
            "exaggeration": 0.7,        # Higher for emotional expression
            "cfg_weight": 0.6,          # Higher for clarity in speech
            "repetition_penalty": 1.3,  # Higher to avoid repetitive dialogue
            "min_p": 0.05,
            "top_p": 0.95,              # Slightly lower for more focused generation
        },
        ContentType.NARRATIVE: {
            "temperature": 0.8,         # Balanced for natural storytelling
            "exaggeration": 0.5,        # Neutral for story flow
            "cfg_weight": 0.5,          # Balanced
            "repetition_penalty": 1.2,  # Standard
            "min_p": 0.05,
            "top_p": 1.0,               # Full range for natural variation
        },
        ContentType.DESCRIPTIVE: {
            "temperature": 0.9,         # Higher for natural variation in descriptions
            "exaggeration": 0.3,        # Lower for calm, descriptive delivery
            "cfg_weight": 0.4,          # Lower for more creative interpretation
            "repetition_penalty": 1.1,  # Lower - descriptions can have repetitive structure
            "min_p": 0.04,              # Slightly lower for more variety
            "top_p": 1.0,
        },
        ContentType.TRANSITION: {
            "temperature": 0.75,        # Moderate for smooth transitions
            "exaggeration": 0.4,        # Lower for transitional calm
            "cfg_weight": 0.55,         # Slightly higher for clarity
            "repetition_penalty": 1.25, # Moderate
            "min_p": 0.05,
            "top_p": 0.98,
        }
    }
    
    def __init__(self):
        self.complexity_adjustments = {
            "temperature": {
                "high_complexity": -0.1,    # Lower temp for complex text
                "low_complexity": 0.05      # Higher temp for simple text
            },
            "exaggeration": {
                "high_complexity": -0.1,    # Less exaggeration for complex content
                "low_complexity": 0.1       # More exaggeration for simple content
            },
            "cfg_weight": {
                "high_complexity": 0.1,     # Higher CFG for complex text clarity
                "low_complexity": -0.05     # Lower CFG for simple text creativity
            }
        }
    
    def get_adaptive_parameters(self, chunk_info: ChunkInfo) -> Dict:
        """Get optimized parameters for a specific chunk"""
        
        # Start with base profile for content type
        params = self.CONTENT_PROFILES[chunk_info.content_type].copy()
        
        # Apply complexity adjustments
        if chunk_info.complexity_score > 6:  # High complexity
            params["temperature"] += self.complexity_adjustments["temperature"]["high_complexity"]
            params["exaggeration"] += self.complexity_adjustments["exaggeration"]["high_complexity"] 
            params["cfg_weight"] += self.complexity_adjustments["cfg_weight"]["high_complexity"]
        elif chunk_info.complexity_score < 3:  # Low complexity
            params["temperature"] += self.complexity_adjustments["temperature"]["low_complexity"]
            params["exaggeration"] += self.complexity_adjustments["exaggeration"]["low_complexity"]
            params["cfg_weight"] += self.complexity_adjustments["cfg_weight"]["low_complexity"]
        
        # Apply positional adjustments
        if chunk_info.is_first_chunk:
            params["temperature"] *= 0.95      # Slightly more stable start
            params["cfg_weight"] *= 1.05       # Slightly higher clarity for beginning
        
        if chunk_info.is_last_chunk:
            params["exaggeration"] *= 0.9      # Slightly calmer ending
        
        # Apply length-based adjustments
        if chunk_info.char_count > 500:
            params["repetition_penalty"] *= 1.05   # Prevent repetition in long chunks
        elif chunk_info.char_count < 200:
            params["temperature"] *= 1.05          # More variation in short chunks
        
        # Apply dialogue-specific adjustments
        if chunk_info.dialogue_ratio > 0.1:
            params["exaggeration"] = min(0.8, params["exaggeration"] * 1.1)  # More expression
            params["temperature"] = max(0.6, params["temperature"] * 0.95)   # More consistency
        
        # Clamp all parameters to safe ranges
        params = self._clamp_parameters(params)
        
        logger.debug(f"🎛️ Chunk {chunk_info.id} ({chunk_info.content_type.value}): "
                    f"temp={params['temperature']:.2f}, "
                    f"exag={params['exaggeration']:.2f}, "
                    f"cfg={params['cfg_weight']:.2f}")
        
        return params
    
    def _clamp_parameters(self, params: Dict) -> Dict:
        """Clamp parameters to safe ranges"""
        clamps = {
            "temperature": (0.5, 1.2),
            "exaggeration": (0.1, 1.0),
            "cfg_weight": (0.2, 0.8),
            "repetition_penalty": (1.0, 1.5),
            "min_p": (0.01, 0.1),
            "top_p": (0.8, 1.0)
        }
        
        for param, (min_val, max_val) in clamps.items():
            if param in params:
                params[param] = max(min_val, min(max_val, params[param]))
        
        return params


class AdvancedTextSanitizer:
    """Comprehensive text cleaning and normalization for optimal TTS quality"""
    
    def __init__(self):
        # Problematic character mappings
        self.unicode_replacements = {
            # Unicode punctuation
            '…': '...',
            '–': '-', 
            '—': ' - ',
            ''': "'",
            ''': "'", 
            '"': '"',
            '"': '"',
            '«': '"',
            '»': '"',
            '„': '"',
            '"': '"',
            
            # Mathematical symbols
            '×': ' times ',
            '÷': ' divided by ',
            '±': ' plus or minus ',
            '≤': ' less than or equal to ',
            '≥': ' greater than or equal to ',
            '≠': ' not equal to ',
            '≈': ' approximately ',
            '∞': ' infinity ',
            
            # Currency symbols
            '€': ' euros',
            '£': ' pounds',
            '¥': ' yen',
            '₽': ' rubles',
            '₹': ' rupees',
            '₿': ' bitcoin',
            
            # Special symbols
            '©': ' copyright ',
            '®': ' registered ',
            '™': ' trademark ',
            '§': ' section ',
            '¶': ' paragraph ',
            '†': '',  # Remove dagger
            '‡': '',  # Remove double dagger
            '°': ' degrees ',
            '%': ' percent',
            '‰': ' per mille',
            
            # Arrows and symbols
            '→': ' arrow ',
            '←': ' arrow ',
            '↑': ' up arrow ',
            '↓': ' down arrow ',
            '⇒': ' implies ',
            '⇔': ' if and only if ',
            
            # Fractions
            '½': ' half',
            '⅓': ' one third',
            '⅔': ' two thirds',
            '¼': ' one quarter',
            '¾': ' three quarters',
            '⅛': ' one eighth',
            '⅜': ' three eighths',
            '⅝': ' five eighths',
            '⅞': ' seven eighths',
        }
        
        # Common abbreviations
        self.abbreviations = {
            'etc.': 'etcetera',
            'e.g.': 'for example',
            'i.e.': 'that is',
            'vs.': 'versus',
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'Dr.': 'Doctor',
            'Prof.': 'Professor',
            'St.': 'Saint',
            'Ave.': 'Avenue',
            'Blvd.': 'Boulevard',
            'Rd.': 'Road',
            'Ct.': 'Court',
            'Ln.': 'Lane',
            'Pkwy.': 'Parkway',
            'Inc.': 'Incorporated',
            'Corp.': 'Corporation',
            'Ltd.': 'Limited',
            'Co.': 'Company',
            'Jr.': 'Junior',
            'Sr.': 'Senior',
            'Ph.D.': 'PhD',
            'M.D.': 'MD',
            'B.A.': 'BA',
            'M.A.': 'MA',
            'CEO': 'C E O',
            'CFO': 'C F O',
            'CTO': 'C T O',
            'USA': 'U S A',
            'UK': 'U K',
            'EU': 'E U',
            'FBI': 'F B I',
            'CIA': 'C I A',
            'NASA': 'N A S A',
            'GPS': 'G P S',
            'AI': 'A I',
            'API': 'A P I',
            'URL': 'U R L',
            'HTML': 'H T M L',
            'CSS': 'C S S',
            'JS': 'JavaScript',
            'iOS': 'i O S',
            'macOS': 'mac O S',
            'WiFi': 'Wi-Fi',
        }
        
        # Number patterns
        self.number_patterns = [
            (r'\b(\d{1,3}),(\d{3})\b', r'\1\2'),  # Remove commas in numbers
            (r'\$(\d+)', r'\1 dollars'),          # Currency
            (r'(\d+)%', r'\1 percent'),           # Percentages
            (r'(\d+)°([CF]?)', self._temperature_replace),  # Temperature
            (r'(\d+):\d{2}', self._time_replace), # Time format
        ]
        
        # Problematic sequences that cause TTS issues
        self.problematic_sequences = {
            '...': '. ',
            '!!': '!',
            '??': '?',
            '.,': '.',
            '.?': '?',
            '.!': '!',
            '!?': '!',
            '?!': '?',
            ';;': ';',
            '::': ':',
            '--': ' - ',
            '---': ' - ',
            '____': '',  # Remove long underscores
            '****': '',  # Remove asterisk patterns
            '####': '',  # Remove hash patterns
        }
    
    def _temperature_replace(self, match):
        """Convert temperature format"""
        num = match.group(1)
        unit = match.group(2) or 'F'
        unit_name = 'Fahrenheit' if unit == 'F' else 'Celsius' if unit == 'C' else 'degrees'
        return f"{num} degrees {unit_name}"
    
    def _time_replace(self, match):
        """Convert time format"""
        time_str = match.group(0)
        try:
            hour, minute = time_str.split(':')
            hour = int(hour)
            minute = int(minute)
            
            if hour == 0:
                return f"twelve {minute:02d} AM"
            elif hour < 12:
                return f"{hour} {minute:02d} AM"
            elif hour == 12:
                return f"twelve {minute:02d} PM"
            else:
                return f"{hour-12} {minute:02d} PM"
        except:
            return time_str
    
    def normalize_numbers(self, text: str) -> str:
        """Convert numbers to spoken form"""
        # Handle ordinals
        text = re.sub(r'\b(\d+)(st|nd|rd|th)\b', r'\1', text)
        
        # Handle years (1900-2099)
        text = re.sub(r'\b(19|20)(\d{2})\b', 
                     lambda m: f"{m.group(1)} {m.group(2)}" if int(m.group(2)) < 10 
                     else f"{m.group(1)}{m.group(2)[:1]} {m.group(2)[1:]}", text)
        
        # Handle phone numbers (basic format)
        text = re.sub(r'\b(\d{3})-(\d{3})-(\d{4})\b', 
                     r'\1 \2 \3', text)
        
        # Apply number patterns
        for pattern, replacement in self.number_patterns:
            if callable(replacement):
                text = re.sub(pattern, replacement, text)
            else:
                text = re.sub(pattern, replacement, text)
        
        return text
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations"""
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Check for exact matches first
            if word in self.abbreviations:
                expanded_words.append(self.abbreviations[word])
            else:
                # Check for case-insensitive matches
                word_lower = word.lower()
                if word_lower in self.abbreviations:
                    expanded_words.append(self.abbreviations[word_lower])
                else:
                    expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def clean_spacing_and_punctuation(self, text: str) -> str:
        """Fix spacing and punctuation issues"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix space before punctuation
        text = re.sub(r'\s+([.!?,:;])', r'\1', text)
        
        # Fix space after punctuation
        text = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', text)
        text = re.sub(r'([,:;])([A-Za-z])', r'\1 \2', text)
        
        # Remove problematic sequences
        for problematic, replacement in self.problematic_sequences.items():
            text = text.replace(problematic, replacement)
        
        # Fix quotes spacing
        text = re.sub(r'\s*"\s*([^"]*)\s*"\s*', r' "\1" ', text)
        text = re.sub(r"\s*'\s*([^']*)\s*'\s*", r" '\1' ", text)
        
        return text.strip()
    
    def remove_markup_and_formatting(self, text: str) -> str:
        """Remove HTML, markdown, and other markup"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'__(.*?)__', r'\1', text)      # Bold
        text = re.sub(r'_(.*?)_', r'\1', text)        # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
        
        # Remove links but keep text
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # [text](url)
        text = re.sub(r'https?://[^\s]+', '', text)           # URLs
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        return text
    
    def deep_clean(self, text: str) -> str:
        """Comprehensive text cleaning pipeline"""
        if not text or not text.strip():
            return "You need to add some text for me to talk."
        
        logger.debug(f"🧹 Starting deep clean for {len(text)} characters")
        
        # 1. Unicode normalization
        text = unicodedata.normalize('NFKD', text)
        
        # 2. Remove markup and formatting
        text = self.remove_markup_and_formatting(text)
        
        # 3. Replace problematic Unicode characters
        for old_char, new_char in self.unicode_replacements.items():
            text = text.replace(old_char, new_char)
        
        # 4. Normalize numbers and special formats
        text = self.normalize_numbers(text)
        
        # 5. Expand abbreviations
        text = self.expand_abbreviations(text)
        
        # 6. Clean spacing and punctuation
        text = self.clean_spacing_and_punctuation(text)
        
        # 7. Final normalization
        # Capitalize first letter
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        
        # Ensure proper sentence ending
        text = text.rstrip()
        sentence_enders = {'.', '!', '?'}
        if not any(text.endswith(p) for p in sentence_enders):
            text += '.'
        
        # Remove any remaining problematic characters
        # Keep only printable ASCII plus common extended characters
        text = ''.join(char for char in text if ord(char) < 127 or char in 'áéíóúàèìòùâêîôûäëïöüñç')
        
        logger.debug(f"🧹 Deep clean completed: {len(text)} characters")
        
        return text.strip()


@dataclass 
class QualityScore:
    """Quality assessment results for audio chunks"""
    overall_score: float  # 0-100 score
    issues: List[str]     # List of detected issues
    duration: float       # Audio duration in seconds
    silence_ratio: float  # Ratio of silence to total audio
    peak_db: float        # Peak audio level in dB
    rms_db: float         # RMS audio level in dB
    should_regenerate: bool = False


class ChunkQualityAnalyzer:
    """Analyzes individual chunk quality before stitching"""
    
    def __init__(self):
        self.min_duration = 0.3      # Minimum acceptable duration (seconds)
        self.max_duration = 120.0    # Fallback maximum acceptable duration (seconds)
        self.silence_threshold = -30  # dB threshold for silence detection
        self.max_silence_ratio = 0.5  # Maximum acceptable silence ratio
        self.min_peak_db = -25       # Minimum peak level (too quiet)
        self.max_peak_db = -1        # Maximum peak level (too loud, risk of clipping)
        self.min_rms_db = -35        # Minimum RMS level
        # Relaxed speaking rate bounds to avoid over-penalizing fast/slow but acceptable delivery
        self.chars_per_second_range = (3, 35)  # Expected characters per second range
    
    def detect_silence_segments(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[float, List[Tuple[float, float]]]:
        """Detect silence segments in audio"""
        try:
            # Convert to dB
            audio_db = librosa.amplitude_to_db(np.abs(audio_data + 1e-9))
            
            # Find silence segments
            silence_mask = audio_db < self.silence_threshold
            
            # Convert to time segments
            frame_duration = len(audio_data) / sample_rate / len(audio_db)
            silence_segments = []
            
            in_silence = False
            silence_start = 0
            
            for i, is_silent in enumerate(silence_mask):
                if is_silent and not in_silence:
                    # Start of silence
                    silence_start = i * frame_duration
                    in_silence = True
                elif not is_silent and in_silence:
                    # End of silence
                    silence_end = i * frame_duration
                    silence_segments.append((silence_start, silence_end))
                    in_silence = False
            
            # Handle case where audio ends in silence
            if in_silence:
                silence_end = len(audio_db) * frame_duration
                silence_segments.append((silence_start, silence_end))
            
            # Calculate total silence ratio
            total_silence = sum(end - start for start, end in silence_segments)
            silence_ratio = total_silence / (len(audio_data) / sample_rate)
            
            return silence_ratio, silence_segments
            
        except Exception as e:
            logger.warning(f"Silence detection failed: {e}")
            return 0.0, []
    
    def analyze_audio_levels(self, audio_data: np.ndarray) -> Tuple[float, float]:
        """Analyze peak and RMS levels"""
        try:
            # Peak level in dB
            peak_linear = np.max(np.abs(audio_data))
            peak_db = librosa.amplitude_to_db(peak_linear) if peak_linear > 0 else -np.inf
            
            # RMS level in dB
            rms_linear = np.sqrt(np.mean(audio_data ** 2))
            rms_db = librosa.amplitude_to_db(rms_linear) if rms_linear > 0 else -np.inf
            
            return peak_db, rms_db
            
        except Exception as e:
            logger.warning(f"Audio level analysis failed: {e}")
            return -np.inf, -np.inf
    
    def analyze_chunk_quality(self, audio_path: str, chunk_info: ChunkInfo) -> QualityScore:
        """Comprehensive quality analysis of an audio chunk"""
        quality_issues = []
        
        try:
            # Load audio
            audio_data, sample_rate = librosa.load(audio_path, sr=None)
            duration = len(audio_data) / sample_rate
            
            # 1. Duration validation (dynamic based on text length)
            # Compute expected duration bounds from characters-per-second range
            expected_min_duration = max(0.2, chunk_info.char_count / self.chars_per_second_range[1])
            expected_max_duration = chunk_info.char_count / self.chars_per_second_range[0]
            # Allow generous headroom for expressive pacing but cap at fallback max
            dynamic_max_duration = min(max(15.0, expected_max_duration * 1.5), self.max_duration)
            dynamic_min_duration = max(self.min_duration, expected_min_duration * 0.5)

            if duration < dynamic_min_duration:
                quality_issues.append("too_short")
            elif duration > dynamic_max_duration:
                quality_issues.append("too_long")
            
            # 2. Silence analysis
            silence_ratio, silence_segments = self.detect_silence_segments(audio_data, sample_rate)
            if silence_ratio > self.max_silence_ratio:
                quality_issues.append("excessive_silence")
            
            # Check for silence at beginning or end (>0.5s)
            if silence_segments:
                if silence_segments[0][0] == 0 and silence_segments[0][1] > 0.5:
                    quality_issues.append("silence_at_start")
                if silence_segments[-1][1] >= duration - 0.1 and silence_segments[-1][1] - silence_segments[-1][0] > 0.5:
                    quality_issues.append("silence_at_end")
            
            # 3. Audio level analysis
            peak_db, rms_db = self.analyze_audio_levels(audio_data)
            
            if peak_db < self.min_peak_db:
                quality_issues.append("too_quiet")
            elif peak_db > self.max_peak_db:
                quality_issues.append("too_loud")
            
            if rms_db < self.min_rms_db:
                quality_issues.append("low_energy")
            
            # 4. Duration vs text length consistency
            expected_chars_per_sec = chunk_info.char_count / duration
            if not (self.chars_per_second_range[0] <= expected_chars_per_sec <= self.chars_per_second_range[1]):
                if expected_chars_per_sec < self.chars_per_second_range[0]:
                    quality_issues.append("too_slow")
                else:
                    quality_issues.append("too_fast")
            
            # 5. Check for audio artifacts
            if len(silence_segments) > duration * 2:  # Too many silence gaps
                quality_issues.append("fragmented_audio")
            
            # Calculate overall quality score
            base_score = 100
            score_penalties = {
                "too_short": 30,
                "too_long": 20,
                "excessive_silence": 25,
                "silence_at_start": 15,
                "silence_at_end": 15,
                "too_quiet": 20,
                "too_loud": 25,
                "low_energy": 15,
                "too_slow": 20,
                "too_fast": 20,
                "fragmented_audio": 15,
            }
            
            for issue in quality_issues:
                base_score -= score_penalties.get(issue, 10)
            
            overall_score = max(0, base_score)
            
            # Do not trigger regeneration based on quality metrics; allow any score.
            # Retries should only occur if audio generation itself fails (exceptions in generate/save).
            should_regenerate = False
            
            quality_score = QualityScore(
                overall_score=overall_score,
                issues=quality_issues,
                duration=duration,
                silence_ratio=silence_ratio,
                peak_db=peak_db,
                rms_db=rms_db,
                should_regenerate=should_regenerate
            )
            
            logger.debug(f"🔍 Chunk {chunk_info.id} quality: {overall_score:.1f}/100, issues: {quality_issues}")
            
            return quality_score
            
        except Exception as e:
            logger.error(f"❌ Quality analysis failed for chunk {chunk_info.id}: {e}")
            # Still do not request regeneration; treat as non-blocking QA failure
            return QualityScore(
                overall_score=30,
                issues=["analysis_failed"],
                duration=0,
                silence_ratio=1.0,
                peak_db=-np.inf,
                rms_db=-np.inf,
                should_regenerate=False
            )


class AdvancedStitcher:
    """Advanced audio stitching with fade transitions and smart pauses"""
    
    def __init__(self):
        # Punctuation-based pause durations (in milliseconds)
        self.punctuation_pauses = {
            '.': 270,   # Period: longer pause for sentence end
            '!': 250,   # Exclamation: medium-long pause with emotion
            '?': 250,   # Question: medium-long pause with inflection
            ',': 80,   # Comma: short pause for breath
            ';': 120,   # Semicolon: medium pause for clause separation
            ':': 175,   # Colon: medium-short pause for introduction
            '-': 100,   # Dash: short pause for quick aside
            '—': 200,   # Em dash: medium pause for emphasis
            '\n': 320,  # Paragraph: longest pause for topic change
        }
        
        # Content type pause modifiers
        self.content_type_modifiers = {
            ContentType.DIALOGUE: 0.8,     # Faster pacing for conversation
            ContentType.NARRATIVE: 1.0,    # Standard pacing for storytelling
            ContentType.DESCRIPTIVE: 1.2,  # Slower pacing for descriptions
            ContentType.TRANSITION: 0.9,   # Slightly faster for transitions
        }
        
        # Fade settings
        self.fade_in_duration = 50   # ms
        self.fade_out_duration = 50  # ms
        self.crossfade_duration = 25 # ms for overlapping chunks
    
    def calculate_smart_pause(self, chunk_info: ChunkInfo, next_chunk_info: Optional[ChunkInfo] = None) -> int:
        """Calculate optimal pause duration based on context"""
        
        # Get base pause from ending punctuation
        ending_punct = chunk_info.ending_punctuation
        base_pause = self.punctuation_pauses.get(ending_punct, 200)
        
        # Apply content type modifier
        content_modifier = self.content_type_modifiers.get(chunk_info.content_type, 1.0)
        pause_duration = base_pause * content_modifier
        
        # Adjust based on paragraph breaks
        if chunk_info.paragraph_break_after:
            pause_duration *= 1.4  # Longer pause between paragraphs
        
        # Adjust based on content type transitions
        if next_chunk_info and chunk_info.content_type != next_chunk_info.content_type:
            # Transitioning between content types needs extra pause
            pause_duration *= 1.2
            
            # Special case: dialogue to non-dialogue needs more pause
            if chunk_info.content_type == ContentType.DIALOGUE:
                pause_duration *= 1.1
        
        # Adjust based on complexity
        if chunk_info.complexity_score > 7:
            pause_duration *= 1.1  # Longer pause after complex content
        
        # Ensure reasonable bounds
        pause_duration = max(100, min(800, pause_duration))
        
        return int(pause_duration)
    
    def apply_smart_fades(self, segment, is_first: bool, is_last: bool, 
                         prev_chunk_info: Optional[ChunkInfo] = None,
                         next_chunk_info: Optional[ChunkInfo] = None):
        """Apply intelligent fade in/out based on content context"""
        
        processed_segment = segment
        
        # Apply fade in (except for first chunk)
        if not is_first:
            fade_in_duration = self.fade_in_duration
            
            # Longer fade for content type transitions
            if prev_chunk_info and prev_chunk_info.content_type == ContentType.DIALOGUE:
                fade_in_duration = int(fade_in_duration * 1.2)
            
            processed_segment = processed_segment.fade_in(fade_in_duration)
        
        # Apply fade out (except for last chunk)
        if not is_last:
            fade_out_duration = self.fade_out_duration
            
            # Longer fade for dialogue endings
            if next_chunk_info and next_chunk_info.content_type == ContentType.DIALOGUE:
                fade_out_duration = int(fade_out_duration * 1.2)
                
            processed_segment = processed_segment.fade_out(fade_out_duration)
        
        return processed_segment
    
    def normalize_segment_levels(self, segment, target_lufs: float = -23.0):
        """Normalize audio segment to target LUFS level"""
        try:
            # Use pydub's normalize function as a starting point
            normalized = effects.normalize(segment)
            
            # Additional RMS-based normalization for consistency
            current_rms = segment.rms
            if current_rms > 0:
                # Target RMS based on LUFS approximation
                target_rms = current_rms * (10 ** (target_lufs / 20))
                adjustment_db = 20 * np.log10(target_rms / current_rms) if current_rms > 0 else 0
                
                # Limit adjustment to prevent distortion
                adjustment_db = max(-12, min(6, adjustment_db))
                
                if abs(adjustment_db) > 0.5:  # Only adjust if meaningful difference
                    normalized = normalized + adjustment_db
            
            return normalized
            
        except Exception as e:
            logger.warning(f"Level normalization failed: {e}, using basic normalize")
            return effects.normalize(segment)
    
    def advanced_stitch(self, wav_paths: List[str], chunk_infos: List[ChunkInfo], 
                       output_path: str) -> Tuple[torch.Tensor, int, float]:
        """Advanced stitching with smart pauses, fades, and normalization"""
        
        if not PYDUB_AVAILABLE:
            logger.warning("⚠️ pydub not available, falling back to basic stitching")
            return self._fallback_stitch(wav_paths, output_path)
        
        logger.info(f"🎼 Advanced stitching {len(wav_paths)} chunks with smart transitions")
        
        try:
            combined = AudioSegment.empty()
            processing_stats = []
            
            for i, (wav_path, chunk_info) in enumerate(zip(wav_paths, chunk_infos)):
                # Load and normalize individual segment
                segment = AudioSegment.from_wav(wav_path)
                original_duration = len(segment)
                
                # Individual normalization first
                segment = self.normalize_segment_levels(segment)
                
                # Apply smart fades
                prev_chunk = chunk_infos[i-1] if i > 0 else None
                next_chunk = chunk_infos[i+1] if i < len(chunk_infos) - 1 else None
                
                segment = self.apply_smart_fades(
                    segment, 
                    is_first=(i == 0), 
                    is_last=(i == len(wav_paths) - 1),
                    prev_chunk_info=prev_chunk,
                    next_chunk_info=next_chunk
                )
                
                # Add to combined audio
                combined += segment
                
                # Add smart pause (except after last chunk)
                if i < len(wav_paths) - 1:
                    pause_duration = self.calculate_smart_pause(chunk_info, next_chunk)
                    combined += AudioSegment.silent(pause_duration)
                    
                    processing_stats.append({
                        "chunk_id": chunk_info.id,
                        "content_type": chunk_info.content_type.value,
                        "original_duration_ms": original_duration,
                        "processed_duration_ms": len(segment),
                        "pause_after_ms": pause_duration,
                        "ending_punctuation": chunk_info.ending_punctuation
                    })
                else:
                    processing_stats.append({
                        "chunk_id": chunk_info.id,
                        "content_type": chunk_info.content_type.value,
                        "original_duration_ms": original_duration,
                        "processed_duration_ms": len(segment),
                        "pause_after_ms": 0,
                        "ending_punctuation": chunk_info.ending_punctuation
                    })
            
            # Final global normalization for consistency
            normalized_combined = self.normalize_segment_levels(combined)
            
            # Export to file
            logger.info(f"🎼 Exporting stitched audio to: {output_path}")
            normalized_combined.export(output_path, format="wav")
            
            # Load back as tensor for return
            audio_tensor, sample_rate = torchaudio.load(output_path)
            duration = len(normalized_combined) / 1000.0
            
            # Log stitching statistics
            total_pause_time = sum(stat["pause_after_ms"] for stat in processing_stats) / 1000.0
            avg_pause = total_pause_time / max(len(processing_stats) - 1, 1)
            content_transitions = sum(1 for i in range(len(chunk_infos) - 1) 
                                   if chunk_infos[i].content_type != chunk_infos[i+1].content_type)
            
            logger.info(f"🎼 Advanced stitching completed:")
            logger.info(f"   - Total duration: {duration:.2f}s")
            logger.info(f"   - Total pause time: {total_pause_time:.2f}s")
            logger.info(f"   - Average pause: {avg_pause:.2f}s")
            logger.info(f"   - Content transitions: {content_transitions}")
            
            return audio_tensor, sample_rate, duration
            
        except Exception as e:
            logger.error(f"❌ Advanced stitching failed: {e}")
            logger.info("🔄 Falling back to basic stitching")
            return self._fallback_stitch(wav_paths, output_path)
    
    def _fallback_stitch(self, wav_paths: List[str], output_path: str, pause_ms: int = 200) -> Tuple[torch.Tensor, int, float]:
        """Fallback stitching method"""
        audio_chunks = []
        sample_rate = None
        
        for wav_path in wav_paths:
            audio_tensor, sr = torchaudio.load(wav_path)
            if sample_rate is None:
                sample_rate = sr
            audio_chunks.append(audio_tensor)
            
            # Add silence between chunks
            silence_duration = int(pause_ms * sample_rate / 1000)
            silence = torch.zeros(1, silence_duration)
            audio_chunks.append(silence)
        
        # Remove last silence
        if audio_chunks:
            audio_chunks.pop()
        
        # Concatenate all chunks
        if audio_chunks:
            final_audio = torch.cat(audio_chunks, dim=-1)
            torchaudio.save(output_path, final_audio, sample_rate)
            duration = final_audio.shape[-1] / sample_rate
            return final_audio, sample_rate, duration
        else:
            raise RuntimeError("No audio chunks to concatenate")


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()
        
        # Initialize smart chunking and adaptive parameters
        self.smart_chunker = SmartChunker()
        self.param_manager = AdaptiveParameterManager()
        
        # Initialize advanced features
        self.text_sanitizer = AdvancedTextSanitizer()
        self.quality_analyzer = ChunkQualityAnalyzer()
        self.advanced_stitcher = AdvancedStitcher()
        
        # Parallel processing settings
        self.max_parallel_workers = 2  # Conservative for GPU memory
        self.enable_parallel_processing = True
        
        logger.info(f"✅ ChatterboxTTS initialized successfully")
        logger.info(f"  - Available methods: {[m for m in dir(self) if not m.startswith('_')]}")
        
        # Debug: Check for specific methods
        expected_tts_methods = [
            'generate_tts_story',
            'generate_long_text',
            'chunk_text',
            'generate_chunks',
            'stitch_and_normalize',
            'cleanup_chunks',
            'tensor_to_mp3_bytes',
            'tensor_to_audiosegment',
            'tensor_to_wav_bytes',
            'upload_to_firebase'
        ]
        
        available_methods = [m for m in dir(self) if not m.startswith('_')]
        missing_methods = [m for m in expected_tts_methods if m not in available_methods]
        
        logger.info(f"🔍 TTS Method Check:")
        logger.info(f"  - Expected methods: {expected_tts_methods}")
        logger.info(f"  - Available methods: {available_methods}")
        logger.info(f"  - Missing methods: {missing_methods}")
        
        if missing_methods:
            logger.error(f"❌ MISSING METHODS: {missing_methods}")
        else:
            logger.info(f"✅ All expected methods are available!")

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    def save_voice_clone(self, audio_file_path: str, save_path: str):
        """Save a voice embedding from an audio file for fast reuse"""
        import librosa
        ref_wav, sr = librosa.load(audio_file_path, sr=None)
        ref_wav = torch.from_numpy(ref_wav).float()
        self.s3gen.save_voice_clone(ref_wav, sr, save_path)
        print(f"Voice clone saved to {save_path}")

    def save_voice_profile(self, audio_file_path: str, save_path: str):
        """Save a complete voice profile including embedding, prompt features, and tokens for more accurate TTS"""
        import librosa
        ref_wav, sr = librosa.load(audio_file_path, sr=None)
        ref_wav = torch.from_numpy(ref_wav).float()
        
        # Get the full reference dictionary from s3gen
        ref_dict = self.s3gen.embed_ref(ref_wav, sr, device=self.device)
        
        # Also compute voice encoder embedding for T3
        ref_16k_wav = librosa.resample(ref_wav.numpy(), orig_sr=sr, target_sr=S3_SR)
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)
        
        # Create and save the complete voice profile
        profile = VoiceProfile(
            embedding=ref_dict["embedding"],  # CAMPPlus embedding for S3Gen
            prompt_feat=ref_dict["prompt_feat"],
            prompt_feat_len=ref_dict.get("prompt_feat_len"),
            prompt_token=ref_dict["prompt_token"],
            prompt_token_len=ref_dict["prompt_token_len"],
        )
        
        # Add voice encoder embedding to the profile data
        profile_data = {
            "embedding": profile.embedding.detach().cpu().numpy(),
            "ve_embedding": ve_embed.detach().cpu().numpy(),  # Voice encoder embedding for T3
        }
        if profile.prompt_feat is not None:
            profile_data["prompt_feat"] = profile.prompt_feat.detach().cpu().numpy()
        if profile.prompt_feat_len is not None:
            profile_data["prompt_feat_len"] = profile.prompt_feat_len
        if profile.prompt_token is not None:
            profile_data["prompt_token"] = profile.prompt_token.detach().cpu().numpy()
        if profile.prompt_token_len is not None:
            profile_data["prompt_token_len"] = profile.prompt_token_len.detach().cpu().numpy()
        
        import numpy as np
        np.save(save_path, profile_data)
        print(f"✅ Full voice profile saved to {save_path}")

    def load_voice_clone(self, path: str):
        """Load a pre-saved voice embedding"""
        return self.s3gen.load_voice_clone(path)

    def load_voice_profile(self, path: str):
        """Load a complete voice profile with custom format including voice encoder embedding"""
        import numpy as np
        data = np.load(path, allow_pickle=True).item()
        
        # Create VoiceProfile object
        profile = VoiceProfile(
            embedding=torch.tensor(data["embedding"]).to(self.device),
            prompt_feat=torch.tensor(data["prompt_feat"]).to(self.device) if "prompt_feat" in data else None,
            prompt_feat_len=data.get("prompt_feat_len"),
            prompt_token=torch.tensor(data["prompt_token"]).to(self.device) if "prompt_token" in data else None,
            prompt_token_len=torch.tensor(data["prompt_token_len"]).to(self.device) if "prompt_token_len" in data else None,
        )
        
        # Add voice encoder embedding as an attribute
        if "ve_embedding" in data:
            profile.ve_embedding = torch.tensor(data["ve_embedding"]).to(self.device)
        else:
            # Fallback for old profiles without voice encoder embedding
            profile.ve_embedding = None
            
        return profile

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def prepare_conditionals_with_saved_voice(self, saved_voice_path: str, prompt_audio_path: str, exaggeration=0.5):
        """Prepare conditionals using a pre-saved voice embedding for faster processing"""
        ## Load saved voice embedding
        saved_embedding = self.s3gen.load_voice_clone(saved_voice_path)
        
        ## Load prompt reference wav for tokens and features (still needed)
        s3gen_ref_wav, _sr = librosa.load(prompt_audio_path, sr=S3GEN_SR)
        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        
        # Generate mel features and tokens from prompt audio
        ref_wav_24 = torch.from_numpy(s3gen_ref_wav).float().to(self.device)
        if len(ref_wav_24.shape) == 1:
            ref_wav_24 = ref_wav_24.unsqueeze(0)
        ref_mels_24 = self.s3gen.mel_extractor(ref_wav_24).transpose(1, 2).to(self.device)
        
        # Tokenize 16khz reference for prompt tokens
        ref_16k_wav_tensor = torch.from_numpy(ref_16k_wav).float().to(self.device)[None, ]
        ref_speech_tokens, ref_speech_token_lens = self.s3gen.tokenizer(ref_16k_wav_tensor)
        
        # Make sure mel_len = 2 * stoken_len
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            ref_speech_tokens = ref_speech_tokens[:, :ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]
        
        # Create s3gen ref_dict with saved embedding
        s3gen_ref_dict = dict(
            prompt_token=ref_speech_tokens.to(self.device),
            prompt_token_len=ref_speech_token_lens,
            prompt_feat=ref_mels_24,
            prompt_feat_len=None,
            embedding=saved_embedding,  # Use the pre-saved embedding!
        )

        # Speech cond prompt tokens for T3
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding for T3 (different from CAMPPlus)
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)
        print(f"Conditionals prepared using saved voice from {saved_voice_path}")

    def prepare_conditionals_with_voice_profile(self, voice_profile_path: str, exaggeration=0.5):
        """Prepare conditionals using a complete voice profile for most accurate TTS"""
        # Load the voice profile
        profile = self.load_voice_profile(voice_profile_path)
        
        # Create s3gen ref_dict from the loaded profile
        s3gen_ref_dict = dict(
            prompt_token=profile.prompt_token.to(self.device),
            prompt_token_len=profile.prompt_token_len.to(self.device),
            prompt_feat=profile.prompt_feat.to(self.device),
            prompt_feat_len=profile.prompt_feat_len,
            embedding=profile.embedding.to(self.device),
        )
        
        # For T3, we can use the prompt tokens from the profile for conditioning
        if plen := self.t3.hp.speech_cond_prompt_len:
            # Use the prompt tokens from the profile, but limit to the required length
            t3_cond_prompt_tokens = profile.prompt_token[:, :plen].to(self.device)
        else:
            t3_cond_prompt_tokens = None
        
        # Use the voice encoder embedding from the profile for T3 conditioning
        if hasattr(profile, 've_embedding') and profile.ve_embedding is not None:
            ve_embed = profile.ve_embedding.to(self.device)
        else:
            # Fallback to random embedding if no voice encoder embedding
            ve_embed = torch.randn(1, 256).to(self.device)
        
        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)
        print(f"✅ Conditionals prepared using voice profile from {voice_profile_path}")

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        audio_prompt_path=None,
        saved_voice_path=None,
        voice_profile_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    ):
        if voice_profile_path:
            # Use complete voice profile for most accurate TTS
            self.prepare_conditionals_with_voice_profile(voice_profile_path, exaggeration=exaggeration)
        elif saved_voice_path and audio_prompt_path:
            # Use saved voice embedding with fresh prompt audio for prosody
            self.prepare_conditionals_with_saved_voice(saved_voice_path, audio_prompt_path, exaggeration=exaggeration)
        elif audio_prompt_path:
            # Traditional method: compute everything fresh
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first, specify `audio_prompt_path`, or provide `voice_profile_path`, or provide both `saved_voice_path` and `audio_prompt_path`"

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=self.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,  # TODO: use the value in config
                temperature=temperature,
                cfg_weight=cfg_weight,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
            )
            # Extract only the conditional batch.
            speech_tokens = speech_tokens[0]

            # TODO: output becomes 1D
            speech_tokens = drop_invalid_tokens(speech_tokens)
            
            speech_tokens = speech_tokens[speech_tokens < 6561]

            speech_tokens = speech_tokens.to(self.device)

            wav, _ = self.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=self.conds.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)

    def chunk_text(self, text: str, max_chars: int = 500) -> List[ChunkInfo]:
        """
        Smart text chunking with advanced sanitization and content awareness.

        :param text: Full story text
        :param max_chars: Maximum number of characters per chunk
        :return: List of ChunkInfo objects with full analysis
        """
        # Step 1: Advanced text sanitization
        logger.info(f"🧹 Applying advanced text sanitization to {len(text)} characters")
        sanitized_text = self.text_sanitizer.deep_clean(text)
        
        if len(sanitized_text) != len(text):
            logger.info(f"🧹 Text sanitization: {len(text)} → {len(sanitized_text)} characters")
        
        # Step 2: Smart chunking with content analysis
        target_chars = int(max_chars * 0.8)  # Target 80% of max for better quality
        chunk_infos = self.smart_chunker.smart_chunk(sanitized_text, target_chars, max_chars)
        
        # Log detailed chunk analysis
        if chunk_infos:
            total_chars = sum(chunk.char_count for chunk in chunk_infos)
            avg_chars = total_chars / len(chunk_infos)
            complexity_scores = [chunk.complexity_score for chunk in chunk_infos]
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            
            logger.info(f"📊 Advanced Chunk Analysis:")
            logger.info(f"   - Total chunks: {len(chunk_infos)}")
            logger.info(f"   - Avg chars/chunk: {avg_chars:.1f}")
            logger.info(f"   - Avg complexity: {avg_complexity:.1f}/10")
            logger.info(f"   - Content distribution: {self.smart_chunker._get_content_type_distribution(chunk_infos)}")
            
            # Sanitization impact analysis
            original_problematic = sum(1 for c in text if ord(c) > 127)
            sanitized_problematic = sum(1 for c in sanitized_text if ord(c) > 127)
            if original_problematic > sanitized_problematic:
                logger.info(f"   - Problematic chars removed: {original_problematic - sanitized_problematic}")
        
        return chunk_infos
    
    def simple_sentence_split(self, text: str) -> List[str]:
        """
        Simple sentence splitting using punctuation marks.
        
        :param text: Text to split
        :return: List of sentences
        """
        logger.info(f"📝 Simple sentence splitting for text length: {len(text)}")
        
        # Simple sentence endings
        sentence_endings = ['.', '!', '?', '\n']
        
        sentences = []
        current_sentence = ""
        
        for char in text:
            current_sentence += char
            if char in sentence_endings:
                # Clean up the sentence
                sentence = current_sentence.strip()
                if sentence:
                    sentences.append(sentence)
                current_sentence = ""
        
        # Add any remaining text as a sentence
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        logger.info(f"✅ Split into {len(sentences)} sentences")
        return sentences

    def _generate_single_chunk_with_quality(self, chunk_info: ChunkInfo, voice_profile_path: str) -> Tuple[str, QualityScore]:
        """Generate a single chunk with quality analysis and retry logic"""
        adaptive_params = self.param_manager.get_adaptive_parameters(chunk_info)
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Clear GPU cache before each attempt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Generate audio tensor using adaptive parameters
                audio_tensor = self.generate(
                    text=chunk_info.text,
                    voice_profile_path=voice_profile_path,
                    temperature=adaptive_params["temperature"],
                    exaggeration=adaptive_params["exaggeration"],
                    cfg_weight=adaptive_params["cfg_weight"],
                    repetition_penalty=adaptive_params["repetition_penalty"],
                    min_p=adaptive_params["min_p"],
                    top_p=adaptive_params["top_p"]
                )
                
                # Save to temporary file
                temp_wav = tempfile.NamedTemporaryFile(suffix=f"_chunk_{chunk_info.id}_attempt_{attempt}.wav", delete=False)
                torchaudio.save(temp_wav.name, audio_tensor, self.sr)
                
                # Quality analysis
                quality_score = self.quality_analyzer.analyze_chunk_quality(temp_wav.name, chunk_info)
                
                # Check if regeneration is needed
                if not quality_score.should_regenerate or attempt == max_retries - 1:
                    return temp_wav.name, quality_score
                else:
                    # Adjust parameters for retry
                    os.unlink(temp_wav.name)  # Clean up failed attempt
                    adaptive_params["temperature"] *= 0.95
                    adaptive_params["cfg_weight"] *= 1.05
                    logger.info(f"🔄 Chunk {chunk_info.id} quality score {quality_score.overall_score:.1f}/100, retrying...")
                    
            except Exception as e:
                logger.warning(f"⚠️ Chunk {chunk_info.id} attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
        
        raise RuntimeError(f"Failed to generate acceptable quality for chunk {chunk_info.id}")
    
    def generate_chunks_parallel(self, chunk_infos: List[ChunkInfo], voice_profile_path: str) -> List[Tuple[str, QualityScore]]:
        """Generate chunks in parallel with quality analysis"""
        logger.info(f"🚀 Starting parallel chunk generation ({self.max_parallel_workers} workers)")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self._generate_single_chunk_with_quality, chunk_info, voice_profile_path): chunk_info
                for chunk_info in chunk_infos
            }
            
            # Collect results as they complete
            for future in future_to_chunk:
                chunk_info = future_to_chunk[future]
                try:
                    wav_path, quality_score = future.result()
                    results.append((chunk_info.id, wav_path, quality_score))
                    
                    logger.info(f"✅ Chunk {chunk_info.id + 1} | "
                               f"Quality: {quality_score.overall_score:.1f}/100 | "
                               f"Duration: {quality_score.duration:.2f}s | "
                               f"Issues: {len(quality_score.issues)}")
                    
                except Exception as e:
                    logger.error(f"❌ Chunk {chunk_info.id + 1} failed completely: {e}")
                    raise
        
        # Sort results by chunk ID to maintain order
        results.sort(key=lambda x: x[0])
        return [(wav_path, quality_score) for _, wav_path, quality_score in results]
    
    def generate_chunks(self, chunk_infos: List[ChunkInfo], voice_profile_path: str, 
                       base_temperature: float = 0.8, base_exaggeration: float = 0.5, 
                       base_cfg_weight: float = 0.5) -> List[str]:
        """
        Advanced chunk generation with parallel processing and quality analysis.

        :param chunk_infos: List of ChunkInfo objects with full analysis
        :param voice_profile_path: Path to the voice profile (.npy)
        :param base_temperature: Base temperature (will be adapted per chunk)
        :param base_exaggeration: Base exaggeration (will be adapted per chunk)
        :param base_cfg_weight: Base CFG weight (will be adapted per chunk)
        :return: List of file paths to temporary WAV files
        """
        generation_start = time.time()
        
        if self.enable_parallel_processing and len(chunk_infos) > 1:
            # Use parallel processing with quality analysis
            logger.info(f"🚀 Using parallel processing for {len(chunk_infos)} chunks")
            chunk_results = self.generate_chunks_parallel(chunk_infos, voice_profile_path)
            wav_paths = [wav_path for wav_path, _ in chunk_results]
            quality_scores = [quality_score for _, quality_score in chunk_results]
        else:
            # Use sequential processing (fallback or single chunk)
            logger.info(f"🔄 Using sequential processing for {len(chunk_infos)} chunks")
            wav_paths = []
            quality_scores = []
            
            for chunk_info in chunk_infos:
                wav_path, quality_score = self._generate_single_chunk_with_quality(chunk_info, voice_profile_path)
                wav_paths.append(wav_path)
                quality_scores.append(quality_score)
        
        # Log comprehensive quality analysis
        total_generation_time = time.time() - generation_start
        self._log_quality_analysis(chunk_infos, quality_scores, total_generation_time)
        
        return wav_paths
    
    def _log_quality_analysis(self, chunk_infos: List[ChunkInfo], quality_scores: List[QualityScore], total_time: float):
        """Log comprehensive quality analysis results"""
        if not quality_scores:
            return
        
        # Overall quality metrics
        avg_quality = np.mean([qs.overall_score for qs in quality_scores])
        min_quality = min(qs.overall_score for qs in quality_scores)
        max_quality = max(qs.overall_score for qs in quality_scores)
        
        # Issue analysis
        all_issues = []
        for qs in quality_scores:
            all_issues.extend(qs.issues)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        # Duration analysis
        total_audio_duration = sum(qs.duration for qs in quality_scores)
        avg_duration = total_audio_duration / len(quality_scores)
        
        # Content type quality breakdown
        content_quality = {}
        for chunk_info, quality_score in zip(chunk_infos, quality_scores):
            content_type = chunk_info.content_type.value
            if content_type not in content_quality:
                content_quality[content_type] = []
            content_quality[content_type].append(quality_score.overall_score)
        
        avg_content_quality = {
            ct: np.mean(scores) for ct, scores in content_quality.items()
        }
        
        # Log comprehensive analysis
        logger.info(f"🔍 Comprehensive Quality Analysis:")
        logger.info(f"   - Overall quality: {avg_quality:.1f}/100 (range: {min_quality:.1f}-{max_quality:.1f})")
        logger.info(f"   - Total audio duration: {total_audio_duration:.2f}s")
        logger.info(f"   - Average chunk duration: {avg_duration:.2f}s")
        logger.info(f"   - Generation time: {total_time:.2f}s")
        logger.info(f"   - Audio/Generation ratio: {total_audio_duration/total_time:.2f}x")
        
        if issue_counts:
            logger.info(f"   - Quality issues: {dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True))}")
        
        if len(avg_content_quality) > 1:
            logger.info(f"   - Content type quality: {avg_content_quality}")
        
        # Quality warnings
        poor_quality_chunks = [i for i, qs in enumerate(quality_scores) if qs.overall_score < 70]
        if poor_quality_chunks:
            logger.warning(f"⚠️ {len(poor_quality_chunks)} chunks have quality scores < 70: {poor_quality_chunks}")
        
        excellent_quality_chunks = [i for i, qs in enumerate(quality_scores) if qs.overall_score >= 90]
        if excellent_quality_chunks:
            logger.info(f"🌟 {len(excellent_quality_chunks)} chunks have excellent quality (≥90): {excellent_quality_chunks}")

    def stitch_and_normalize(self, wav_paths: List[str], chunk_infos: List[ChunkInfo], 
                           output_path: str, pause_ms: int = 100) -> Tuple[torch.Tensor, int, float]:
        """
        Advanced stitching with smart pauses, fades, and quality-aware normalization.

        :param wav_paths: List of temporary WAV file paths
        :param chunk_infos: List of ChunkInfo objects for context-aware stitching
        :param output_path: Final path to export the combined WAV file
        :param pause_ms: Base pause duration (will be adjusted by advanced stitcher)
        :return: Tuple of (audio_tensor, sample_rate, duration_seconds)
        """
        logger.info(f"🎼 Advanced stitching with context-aware transitions")
        
        # Use advanced stitcher for intelligent combining
        return self.advanced_stitcher.advanced_stitch(wav_paths, chunk_infos, output_path)

    def cleanup_chunks(self, wav_paths: List[str]):
        """
        Deletes temporary WAV files to clean up disk space.

        :param wav_paths: List of paths to temporary WAV files
        """
        for path in wav_paths:
            try:
                os.remove(path)
                logger.debug(f"🧹 Cleaned up temporary file: {path}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to delete {path} — {e}")

    def generate_long_text(self, text: str, voice_profile_path: str, output_path: str, 
                          max_chars: int = 500, pause_ms: int = 100, temperature: float = 0.8,
                          exaggeration: float = 0.5, cfg_weight: float = 0.5) -> Tuple[torch.Tensor, int, Dict]:
        """
        Full TTS pipeline for long texts: chunk → generate → stitch → clean.

        :param text: Input text to synthesize
        :param voice_profile_path: Path to the voice profile (.npy)
        :param output_path: Path to save the final audio file
        :param max_chars: Maximum number of characters per chunk
        :param pause_ms: Milliseconds of silence between chunks
        :param temperature: Generation temperature
        :param exaggeration: Voice exaggeration factor
        :param cfg_weight: CFG weight for generation
        :return: Tuple of (audio_tensor, sample_rate, metadata_dict)
        """
        logger.info(f"🎵 Starting TTS processing for {len(text)} characters")
        logger.info(f"🔍 Output path: {output_path}")
        
        # Safety check for extremely long texts
        if len(text) > 13000:
            logger.warning(f"⚠️ Very long text ({len(text)} chars) - truncating to safe length")
            text = text[:13000] + "... [truncated]"
            logger.info(f"📝 Truncated text to {len(text)} characters")
        
        chunk_infos = self.chunk_text(text, max_chars)
        logger.info(f"📦 Split into {len(chunk_infos)} intelligent chunks")
        
        wav_paths = self.generate_chunks(chunk_infos, voice_profile_path, temperature, exaggeration, cfg_weight)
        if not wav_paths:
            raise RuntimeError("Failed to generate any audio chunks")
        
        logger.info(f"🔗 Advanced stitching {len(wav_paths)} audio chunks...")
        audio_tensor, sample_rate, total_duration = self.stitch_and_normalize(wav_paths, chunk_infos, output_path, pause_ms)
        
        self.cleanup_chunks(wav_paths)
        
        logger.info(f"✅ TTS processing completed | Duration: {total_duration:.2f}s")
        logger.info(f"🔍 Final output path: {output_path}")
        logger.info(f"🔍 Output file exists: {Path(output_path).exists()}")
        
        # Enhanced metadata with advanced features analysis
        if chunk_infos:
            content_type_dist = self.smart_chunker._get_content_type_distribution(chunk_infos)
            avg_complexity = sum(c.complexity_score for c in chunk_infos) / len(chunk_infos)
            avg_chunk_chars = sum(c.char_count for c in chunk_infos) / len(chunk_infos)
            dialogue_chunks = sum(1 for c in chunk_infos if c.content_type == ContentType.DIALOGUE)
            paragraph_breaks = sum(1 for c in chunk_infos if c.paragraph_break_after)
        else:
            content_type_dist = {}
            avg_complexity = 0
            avg_chunk_chars = 0
            dialogue_chunks = 0
            paragraph_breaks = 0
        
        metadata = {
            # Basic metrics
            "chunk_count": len(chunk_infos),
            "output_path": output_path,
            "duration_sec": total_duration,
            "successful_chunks": len(wav_paths),
            "sample_rate": sample_rate,
            "text_length": len(text),
            "max_chars_per_chunk": max_chars,
            "pause_ms": pause_ms,
            
            # Smart chunking analysis
            "avg_chunk_chars": round(avg_chunk_chars, 1),
            "avg_complexity_score": round(avg_complexity, 2),
            "content_type_distribution": content_type_dist,
            "dialogue_chunk_count": dialogue_chunks,
            "paragraph_breaks": paragraph_breaks,
            "chunking_method": "smart_content_aware",
            
            # Advanced features
            "text_sanitization": "advanced_unicode_normalization",
            "parallel_processing": self.enable_parallel_processing,
            "max_parallel_workers": self.max_parallel_workers if self.enable_parallel_processing else 1,
            "quality_analysis": "comprehensive_audio_validation",
            "stitching_method": "advanced_context_aware_transitions",
            
            # Performance metrics
            "audio_chars_per_second": round(len(text) / max(total_duration, 0.1), 1),  # Characters per second of audio
            "audio_efficiency_ratio": round(total_duration / max(len(text) * 0.08, 1), 2),  # Audio duration vs expected
        }
        
        return audio_tensor, sample_rate, metadata

    # ------------------------------------------------------------------
    # Complete TTS Pipeline with Firebase Upload
    # ------------------------------------------------------------------
    def tensor_to_mp3_bytes(self, audio_tensor: torch.Tensor, sample_rate: int, bitrate: str = "96k") -> bytes:
        """
        Convert audio tensor directly to MP3 bytes.
        
        :param audio_tensor: PyTorch audio tensor
        :param sample_rate: Audio sample rate
        :param bitrate: MP3 bitrate (e.g., "96k", "128k", "160k")
        :return: MP3 bytes
        """
        if PYDUB_AVAILABLE:
            try:
                # Convert tensor to AudioSegment
                audio_segment = self.tensor_to_audiosegment(audio_tensor, sample_rate)
                # Export to MP3 bytes
                mp3_file = audio_segment.export(format="mp3", bitrate=bitrate)
                # Read the bytes from the file object
                mp3_bytes = mp3_file.read()
                return mp3_bytes
            except Exception as e:
                logger.warning(f"Direct MP3 conversion failed: {e}, falling back to WAV")
                return self.tensor_to_wav_bytes(audio_tensor, sample_rate)
        else:
            logger.warning("pydub not available, falling back to WAV")
            return self.tensor_to_wav_bytes(audio_tensor, sample_rate)

    def tensor_to_audiosegment(self, audio_tensor: torch.Tensor, sample_rate: int):
        """
        Convert PyTorch audio tensor to pydub AudioSegment.
        
        :param audio_tensor: PyTorch audio tensor
        :param sample_rate: Audio sample rate
        :return: pydub AudioSegment
        """
        if not PYDUB_AVAILABLE:
            raise ImportError("pydub is required for audio conversion")
        
        # Convert tensor to numpy array
        if audio_tensor.dim() == 2:
            # Stereo: (channels, samples)
            audio_np = audio_tensor.numpy()
        else:
            # Mono: (samples,) -> (1, samples)
            audio_np = audio_tensor.unsqueeze(0).numpy()
        
        # Convert to int16 for pydub
        audio_np = (audio_np * 32767).astype(np.int16)
        
        # Create AudioSegment
        audio_segment = AudioSegment(
            audio_np.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit
            channels=audio_np.shape[0]
        )
        
        return audio_segment

    def tensor_to_wav_bytes(self, audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
        """
        Convert audio tensor to WAV bytes (fallback).
        
        :param audio_tensor: PyTorch audio tensor
        :param sample_rate: Audio sample rate
        :return: WAV bytes
        """
        # Save to temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        torchaudio.save(temp_wav.name, audio_tensor, sample_rate)
        
        # Read WAV bytes
        with open(temp_wav.name, 'rb') as f:
            wav_bytes = f.read()
        
        # Clean up temp file
        os.unlink(temp_wav.name)
        
        return wav_bytes

    def upload_to_firebase(self, data: bytes, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None) -> Optional[str]:
        """
        Upload data directly to Firebase Storage with metadata
        
        :param data: Binary data to upload
        :param destination_blob_name: Destination path in Firebase
        :param content_type: MIME type of the file
        :param metadata: Optional metadata to store with the file
        :return: Public URL
        """
        try:
            from google.cloud import storage
            
            # Initialize Firebase storage client
            storage_client = storage.Client()
            bucket = storage_client.bucket("godnathistorie-a25fa.firebasestorage.app")
            
            logger.info(f"🔍 Starting Firebase upload: {destination_blob_name} ({len(data)} bytes)")
            
            # Create blob and upload (create-only to avoid requiring delete on overwrite)
            blob = bucket.blob(destination_blob_name)
            
            # Set metadata if provided
            if metadata:
                blob.metadata = metadata
            
            # Upload the data with precondition: create only if object doesn't exist
            try:
                blob.upload_from_string(data, content_type=content_type, if_generation_match=0)
            except Exception as precond_err:
                # If the object exists or precondition failed, surface the error to caller
                logger.error(f"❌ Precondition for create-only upload failed: {precond_err}")
                return None
            
            # Try to make the blob publicly accessible (non-fatal if not permitted)
            public_url: Optional[str]
            try:
                blob.make_public()
                public_url = blob.public_url
            except Exception as e:
                logger.warning(f"⚠️ make_public failed (object may still be uploaded): {e}")
                # Fallback to GCS URL (may still require auth depending on bucket policy)
                public_url = f"https://storage.googleapis.com/{bucket.name}/{destination_blob_name}"

            logger.info(f"✅ Uploaded to Firebase: {destination_blob_name} -> {public_url}")
            return public_url
            
        except Exception as e:
            logger.error(f"❌ Failed to upload to Firebase: {e}")
            return None

    def generate_tts_story(self, text: str, voice_id: str, profile_base64: str, 
                          language: str = 'en', story_type: str = 'user', 
                          is_kids_voice: bool = False, metadata: Dict = None) -> Dict:
        """
        Generate TTS story with voice profile from base64.
        
        Args:
            text: Text to synthesize
            voice_id: Unique voice identifier
            profile_base64: Voice profile as base64 string
            language: Language code
            story_type: Type of story
            is_kids_voice: Whether it's a kids voice
            metadata: Optional metadata (for API compatibility)
            
        Returns:
            Dict with status, audio_data, firebase_url, firebase_path, story_type, and generation_time
        """
        import time
        import base64
        import tempfile
        import os
        
        start_time = time.time()
        
        logger.info(f"📚 ChatterboxTTS.generate_tts_story called")
        logger.info(f"  - text length: {len(text)}")
        logger.info(f"  - voice_id: {voice_id}")
        logger.info(f"  - language: {language}")
        logger.info(f"  - story_type: {story_type}")
        logger.info(f"  - is_kids_voice: {is_kids_voice}")
        logger.info(f"  - metadata: {metadata}")
        
        try:
            # Step 1: Load voice profile from base64
            logger.info(f"  - Step 1: Loading voice profile from base64...")
            profile_bytes = base64.b64decode(profile_base64)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp_file:
                temp_file.write(profile_bytes)
                temp_profile_path = temp_file.name
            
            logger.info(f"    - Voice profile loaded from base64")
            
            # Step 2: Generate TTS audio
            logger.info(f"  - Step 2: Generating TTS audio...")
            audio_tensor, sample_rate, generation_metadata = self.generate_long_text(
                text=text,
                voice_profile_path=temp_profile_path,
                output_path="./temp_tts_output.wav",
                max_chars=500,
                pause_ms=150,
                temperature=0.8,
                exaggeration=0.5,
                cfg_weight=0.5
            )
            
            # Step 3: Convert to MP3 bytes
            logger.info(f"  - Step 3: Converting to MP3...")
            mp3_bytes = self.tensor_to_mp3_bytes(audio_tensor, sample_rate, "96k")
            
            # Step 4: Upload to Firebase Storage
            logger.info(f"  - Step 4: Uploading to Firebase Storage...")
            
            # Extract story_type from metadata or use direct parameter
            final_story_type = story_type  # Start with direct parameter
            if metadata and isinstance(metadata, dict) and 'story_type' in metadata:
                final_story_type = metadata['story_type']  # Override with metadata if available
            
            # Ensure story_type is valid (user or app)
            if final_story_type not in ['user', 'app']:
                logger.warning(f"Invalid story_type '{final_story_type}', defaulting to 'user'")
                final_story_type = 'user'
            
            # Generate Firebase path based on story_type and language
            firebase_path = f"audio/stories/{language}/{final_story_type}/{voice_id}.mp3"
            logger.info(f"    - Firebase path: {firebase_path}")
            logger.info(f"    - Story type: {final_story_type}")
            
            # Upload to Firebase
            try:
                firebase_url = self.upload_to_firebase(
                    data=mp3_bytes,
                    destination_blob_name=firebase_path,
                    content_type="audio/mpeg",
                    metadata={
                        "voice_id": voice_id,
                        "language": language,
                        "story_type": final_story_type,
                        "text_length": len(text),
                        "generation_time": time.time() - start_time,
                        "audio_size": len(mp3_bytes)
                    }
                )
                if firebase_url:
                    logger.info(f"    - Uploaded successfully: {firebase_url}")
                else:
                    logger.warning("⚠️ Upload did not complete (object may already exist and overwrite is not permitted). Consider using unique filenames or granting delete permission.")
            except Exception as upload_error:
                logger.error(f"❌ Firebase upload failed: {upload_error}")
                firebase_url = None
            
            # Convert to base64
            audio_base64 = base64.b64encode(mp3_bytes).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(temp_profile_path)
            
            generation_time = time.time() - start_time
            
            # Return result with Firebase URL
            result = {
                "status": "success",
                "audio_data": audio_base64,
                "firebase_url": firebase_url,
                "firebase_path": firebase_path,
                "story_type": final_story_type,
                "generation_time": generation_time
            }
            
            logger.info(f"✅ TTS story generated successfully!")
            logger.info(f"  - Audio size: {len(mp3_bytes)} bytes")
            logger.info(f"  - Generation time: {generation_time:.2f}s")
            
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"❌ ChatterboxTTS.generate_tts_story failed: {e}")
            logger.error(f"  - Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"  - Full traceback: {traceback.format_exc()}")
            
            return {
                "status": "error",
                "error": str(e),
                "generation_time": generation_time
            }