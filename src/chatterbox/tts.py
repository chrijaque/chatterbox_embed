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
# Note: resolve_bucket_name is imported lazily inside methods to avoid circular import

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


def _get_git_sha() -> str:
    """Return current git commit SHA if available, else 'unknown'."""
    try:
        import subprocess
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        if sha:
            return sha
    except Exception:
        pass
    # Try common env vars set by CI/CD
    for key in ("GIT_COMMIT", "SOURCE_COMMIT", "COMMIT_SHA", "VERCEL_GIT_COMMIT_SHA"):
        val = os.environ.get(key)
        if val:
            return val
    return "unknown"


def _peak_rms_dbfs_from_np(x: np.ndarray) -> Tuple[float, float]:
    try:
        x = x.astype(np.float64)
        peak = float(np.max(np.abs(x)) + 1e-12)
        rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        return 20.0 * np.log10(peak), 20.0 * np.log10(rms)
    except Exception:
        return float("nan"), float("nan")


def _levels_from_tensor(tensor: torch.Tensor) -> Tuple[float, float]:
    try:
        if tensor.is_cuda or (hasattr(torch.backends, 'mps') and tensor.device.type == 'mps'):
            tensor = tensor.to('cpu')
        npy = tensor.squeeze(0).detach().numpy().astype(np.float32)
        return _peak_rms_dbfs_from_np(npy)
    except Exception:
        return float("nan"), float("nan")


def _maybe_log_seg_levels(tag: str, seg) -> None:
    try:
        if PYDUB_AVAILABLE and seg is not None:
            logger.info(f"🔊 {tag}: peak={seg.max_dBFS:.2f} dBFS, avg={seg.dBFS:.2f} dBFS")
    except Exception:
        pass


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
        ("...", ". "),
        ("…", ". "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
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
    has_story_break: bool = False  # Whether this chunk should have a story break pause


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
            complexity_score=complexity,
            has_story_break=False  # Set by caller if needed
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
            "temperature": 0.8,         # Slightly higher for dialogue expressiveness
            "exaggeration": 0.7,        # More expression for dialogue
            "cfg_weight": 0.55,          # Slightly lower for more natural flow
            "repetition_penalty": 1.2,   # Standard
            "min_p": 0.05,
            "top_p": 0.9,                # Slightly lower for more focused sampling
        },
        ContentType.NARRATIVE: {
            "temperature": 0.7,          # Balanced for narrative flow
            "exaggeration": 0.5,         # Moderate expression for storytelling
            "cfg_weight": 0.6,           # Good adherence for consistency
            "repetition_penalty": 1.2,
            "min_p": 0.05,
            "top_p": 0.92,               # Slightly higher for narrative variety
        },
        ContentType.DESCRIPTIVE: {
            "temperature": 0.68,         # Slightly higher for descriptive richness
            "exaggeration": 0.45,        # Moderate expression for descriptions
            "cfg_weight": 0.58,          # Balanced adherence
            "repetition_penalty": 1.15,  # Slightly lower to allow more natural flow
            "min_p": 0.05,
            "top_p": 0.94,               # Higher for descriptive variety
        },
        ContentType.TRANSITION: {
            "temperature": 0.72,         # Slightly higher for smooth transitions
            "exaggeration": 0.5,         # Moderate expression
            "cfg_weight": 0.55,          # Slightly lower for natural flow
            "repetition_penalty": 1.18,  # Lower for smoother transitions
            "min_p": 0.05,
            "top_p": 0.93,               # Higher for transition variety
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
        # Intro boost to make the first sentence more engaging
        self.enable_intro_boost = True
        self.intro_exaggeration_boost = 0.2   # +0.2 on first chunk
        self.intro_temperature_boost = 0.05   # +0.05 on first chunk
        self.intro_cfg_weight_factor = 0.9    # reduce CFG a bit for more expression (not below min)
        self.intro_boost_max_words = 35       # apply full boost only if first chunk is short
        # New safety guards for first-chunk stability
        self.intro_min_words_for_boost = 12   # disable boost for extremely short openers
        self.first_chunk_exaggeration_cap = 0.7
        self.first_chunk_min_cfg_weight = 0.5

        # Opener preset (stability over creativity)
        self.enable_opener_preset = True
        self.opener_temperature = 0.62
        self.opener_cfg_weight = 0.7
        self.opener_exaggeration = 0.35
        self.opener_top_p = 0.9
        self.opener_min_p = 0.05
        self.opener_repetition_penalty = 1.18
    
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
            # Original behavior favored stability; instead, apply a gentle intro boost for engagement.
            # For long first chunks, avoid raising temperature to prevent over-energizing long passages.
            if self.enable_intro_boost:
                # No boost for extremely short openers
                if chunk_info.word_count < self.intro_min_words_for_boost:
                    params["temperature"] = params.get("temperature", 0.8)
                    params["exaggeration"] = min(params.get("exaggeration", 0.5), self.first_chunk_exaggeration_cap)
                    params["cfg_weight"] = max(self.first_chunk_min_cfg_weight, params.get("cfg_weight", 0.5))
                elif chunk_info.word_count <= self.intro_boost_max_words:
                    params["temperature"] = max(0.5, min(1.2, params.get("temperature", 0.8) + self.intro_temperature_boost))
                    params["exaggeration"] = max(0.1, min(self.first_chunk_exaggeration_cap, params.get("exaggeration", 0.5) + self.intro_exaggeration_boost))
                    # Do not allow CFG to fall below the minimum for stability
                    boosted_cfg = params.get("cfg_weight", 0.5) * self.intro_cfg_weight_factor
                    params["cfg_weight"] = max(self.first_chunk_min_cfg_weight, boosted_cfg)
                else:
                    # Long first chunk: keep temperature unchanged, only a light expressiveness bump
                    params["exaggeration"] = max(0.1, min(self.first_chunk_exaggeration_cap, params.get("exaggeration", 0.5) + min(0.1, self.intro_exaggeration_boost * 0.5)))
                    params["cfg_weight"] = max(self.first_chunk_min_cfg_weight, params.get("cfg_weight", 0.5))

            # Apply opener preset ONLY for short openers; avoid clamping a long first chunk
            if self.enable_opener_preset and (chunk_info.word_count <= self.intro_boost_max_words or chunk_info.char_count <= 220):
                params["temperature"] = min(params.get("temperature", 0.8), self.opener_temperature)
                params["cfg_weight"] = max(params.get("cfg_weight", 0.5), self.opener_cfg_weight)
                params["exaggeration"] = min(params.get("exaggeration", 0.5), self.opener_exaggeration)
                params["top_p"] = min(params.get("top_p", 1.0), self.opener_top_p)
                params["min_p"] = max(params.get("min_p", 0.05), self.opener_min_p)
                params["repetition_penalty"] = max(params.get("repetition_penalty", 1.2), self.opener_repetition_penalty)
        else:
            # Gentle easing for the second chunk to avoid sudden jump after opener
            # Chunk IDs are sequential; id==1 typically indicates the chunk after opener.
            if chunk_info.id == 1:
                params["temperature"] = min(params.get("temperature", 0.8), max(0.58, self.opener_temperature + 0.05))
                params["exaggeration"] = min(params.get("exaggeration", 0.5), self.first_chunk_exaggeration_cap - 0.1)
                params["cfg_weight"] = max(params.get("cfg_weight", 0.5), max(self.first_chunk_min_cfg_weight, self.opener_cfg_weight - 0.02))
        
        if chunk_info.is_last_chunk:
            params["exaggeration"] *= 0.9      # Slightly calmer ending
        
        # Apply length-based adjustments
        if chunk_info.char_count > 500:
            params["repetition_penalty"] *= 1.05   # Prevent repetition in long chunks
        elif chunk_info.char_count < 200:
            params["temperature"] *= 1.05          # More variation in short chunks
        
        # Apply dialogue-specific adjustments
        if chunk_info.dialogue_ratio > 0.1:
            params["exaggeration"] = min(0.8, params["exaggeration"] * 1.15)  # More expression for dialogue
            params["temperature"] = max(0.6, params["temperature"] * 0.98)   # Slightly more consistency
        
        # Apply descriptive-specific adjustments for slower, more narrative delivery
        if chunk_info.content_type == ContentType.DESCRIPTIVE:
            params["temperature"] = max(0.65, params["temperature"] * 0.95)  # Slightly more consistent for descriptions
            params["cfg_weight"] = min(0.7, params["cfg_weight"] * 1.05)     # Slightly stronger adherence
            params["repetition_penalty"] = max(1.1, params["repetition_penalty"] * 0.98)  # Allow more natural repetition
        
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
            
            # Story section breaks
            '⁂': ' <STORY_BREAK> ',  # Special marker for longer pauses between story sections
            
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
        
        # 7. Remove story break markers (they're handled separately for pause timing)
        text = text.replace('<STORY_BREAK>', '')
        
        # 8. Final normalization
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
        """Detect silence segments in audio using frame-based RMS energy."""
        try:
            # Frame parameters (25ms window, 10ms hop)
            win_ms = 25.0
            hop_ms = 10.0
            win = max(1, int(sample_rate * (win_ms / 1000.0)))
            hop = max(1, int(sample_rate * (hop_ms / 1000.0)))

            # Compute frame-wise RMS in dB
            num_frames = 1 + max(0, (len(audio_data) - win) // hop)
            if num_frames <= 0:
                return 0.0, []

            rms_values = []
            for f in range(num_frames):
                start = f * hop
                end = start + win
                frame = audio_data[start:end]
                if len(frame) == 0:
                    rms_values.append(-np.inf)
                    continue
                rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2) + 1e-12)
                rms_db = 20.0 * np.log10(max(rms, 1e-12))
                rms_values.append(rms_db)

            silence_mask = np.array(rms_values) < self.silence_threshold

            silence_segments: List[Tuple[float, float]] = []
            in_silence = False
            silence_start = 0.0
            frame_time = hop / float(sample_rate)

            for i, is_silent in enumerate(silence_mask):
                if is_silent and not in_silence:
                    silence_start = i * frame_time
                    in_silence = True
                elif not is_silent and in_silence:
                    silence_end = i * frame_time
                    silence_segments.append((silence_start, silence_end))
                    in_silence = False

            if in_silence:
                silence_end = len(silence_mask) * frame_time
                silence_segments.append((silence_start, silence_end))

            total_silence = sum(end - start for start, end in silence_segments)
            duration = len(audio_data) / float(sample_rate)
            silence_ratio = 0.0 if duration <= 0 else total_silence / duration

            return float(silence_ratio), silence_segments
            
        except Exception as e:
            logger.warning(f"Silence detection failed: {e}")
            return 0.0, []
    
    def analyze_audio_levels(self, audio_data: np.ndarray) -> Tuple[float, float]:
        """Analyze peak and RMS levels"""
        try:
            # Peak level in dB
            peak_linear = np.max(np.abs(audio_data))
            peak_db = 20.0 * np.log10(max(peak_linear, 1e-12)) if peak_linear > 0 else -np.inf
            
            # RMS level in dB
            rms_linear = np.sqrt(np.mean(audio_data ** 2))
            rms_db = 20.0 * np.log10(max(rms_linear, 1e-12)) if rms_linear > 0 else -np.inf
            
            return peak_db, rms_db
            
        except Exception as e:
            logger.warning(f"Audio level analysis failed: {e}")
            return -np.inf, -np.inf
    
    def analyze_chunk_quality(self, audio_path: str, chunk_info: ChunkInfo) -> QualityScore:
        """Comprehensive quality analysis of an audio chunk"""
        quality_issues = []
        
        try:
            # Load audio
            audio_tensor, sample_rate = torchaudio.load(audio_path)
            audio_data = audio_tensor.squeeze(0).numpy()
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
            '?': 300,   # Question: medium-long pause with inflection
            ',': 80,   # Comma: short pause for breath
            ';': 120,   # Semicolon: medium pause for clause separation
            ':': 175,   # Colon: medium-short pause for introduction
            '-': 100,   # Dash: short pause for quick aside
            '—': 200,   # Em dash: medium pause for emphasis
            '\n': 320,  # Paragraph: longest pause for topic change
            '<STORY_BREAK>': 550,  # Story section break: very long pause for dramatic effect
        }
        
        # Content type pause modifiers
        self.content_type_modifiers = {
            ContentType.DIALOGUE: 0.85,    # Faster pacing for conversation
            ContentType.NARRATIVE: 1.15,   # Slower pacing for storytelling
            ContentType.DESCRIPTIVE: 1.25, # Even slower pacing for descriptions
            ContentType.TRANSITION: 0.95,  # Slightly faster for transitions
        }
        
        # Fade settings
        self.fade_in_duration = 90   # ms
        self.fade_out_duration = 70  # ms
        self.crossfade_duration = 25 # ms for overlapping chunks

        # Global pause scaling to control narration pace (1.0 = baseline)
        self.global_pause_factor = 1.2  # Increase global pauses for more narrative pacing

        # Loudness normalization disabled
        self.enable_loudness_normalization = False
        self.enable_per_chunk_normalization = False
        # Gentle fade-in for the very first chunk to avoid abrupt start (ms)
        self.fade_in_first_chunk_ms = 130

        # Add an extra pause after the first chunk to let the opener land (ms)
        self.extra_first_pause_ms = 60
        # Loudness target/method removed
    
    def calculate_smart_pause(self, chunk_info: ChunkInfo, next_chunk_info: Optional[ChunkInfo] = None) -> int:
        """Calculate optimal pause duration based on context"""
        
        # Consistent baseline: only paragraph/story breaks get extended pauses.
        # All other punctuation use a short, consistent pause.
        if chunk_info.has_story_break or chunk_info.paragraph_break_after:
            base_pause = 600  # clear separation
        else:
            base_pause = 250  # consistent short pause for sentence ends/commas/etc.

        # Apply global pace factor
        pause_duration = base_pause * max(0.5, min(2.0, self.global_pause_factor))

        # Gentle extra air after the first chunk only
        if chunk_info.is_first_chunk:
            pause_duration += max(0, int(self.extra_first_pause_ms))

        # Clamp to reasonable bounds
        pause_duration = max(120, min(900, pause_duration))
        return int(pause_duration)
    
    def apply_smart_fades(self, segment, is_first: bool, is_last: bool, 
                         prev_chunk_info: Optional[ChunkInfo] = None,
                         next_chunk_info: Optional[ChunkInfo] = None):
        """Apply intelligent fade in/out based on content context"""
        
        processed_segment = segment
        
        # Apply fade in
        if is_first:
            fade_in_duration = max(0, int(self.fade_in_first_chunk_ms))
            if fade_in_duration > 0:
                processed_segment = processed_segment.fade_in(fade_in_duration)
        else:
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

            # Remove additional RMS/LUFS-based attenuation to avoid reducing output volume
            # Returning the peak-normalized segment preserves dynamics while preventing clipping
            return normalized
            
        except Exception as e:
            logger.warning(f"Level normalization failed: {e}, using basic normalize")
            return effects.normalize(segment)

    def _ffmpeg_available(self) -> bool:
        try:
            import shutil
            return shutil.which("ffmpeg") is not None
        except Exception:
            return False

    def _run_ffmpeg_loudnorm(self, input_path: str, output_path: str) -> bool:
        """Removed: loudness normalization disabled."""
        return False

    def _fallback_simple_loudness(self, input_path: str, output_path: str) -> bool:
        """Removed: loudness normalization disabled."""
        return False

    def apply_loudness_normalization_file(self, input_path: str) -> str:
        """Removed: loudness normalization disabled."""
        return input_path
    
    def advanced_stitch(self, wav_paths: List[str], chunk_infos: List[ChunkInfo], 
                       output_path: str) -> Tuple[torch.Tensor, int, float]:
        """Advanced stitching with smart pauses, fades, and normalization"""
        
        if not PYDUB_AVAILABLE:
            logger.warning("⚠️ pydub not available, falling back to basic stitching")
            return self._fallback_stitch(wav_paths, output_path)
        
        logger.info(f"🎼 Advanced stitching {len(wav_paths)} chunks with smart transitions")
        try:
            logger.info(
                "🧪 Stitch cfg | loudnorm_enabled=%s, method=%s, ffmpeg_available=%s",
                getattr(self, "enable_loudness_normalization", False),
                getattr(self, "loudness_method", ""),
                self._ffmpeg_available() if hasattr(self, "_ffmpeg_available") else False,
            )
        except Exception:
            pass
        
        try:
            combined = AudioSegment.empty()
            processing_stats = []
            
            for i, (wav_path, chunk_info) in enumerate(zip(wav_paths, chunk_infos)):
                # Load and normalize individual segment
                segment = AudioSegment.from_wav(wav_path)
                original_duration = len(segment)
                
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

            # Final loudness normalization to target LUFS/TP/LRA
            logger.info("🎚️ Loudness normalization disabled; using original export")
            ln_path = output_path

            # Load back as tensor for return (use loudnorm output if available)
            audio_tensor, sample_rate = torchaudio.load(ln_path)
            duration = float(audio_tensor.shape[-1]) / float(sample_rate)

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
        try:
            kwargs = torch.load(fpath, map_location=map_location, weights_only=True)  # torch>=2.0
        except TypeError:
            kwargs = torch.load(fpath, map_location=map_location)
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
        self.max_parallel_workers = 1  # Disabled for single-user processing
        self.enable_parallel_processing = False  # Disabled as requested
        # Quality Analysis toggle (disable to remove overhead)
        self.enable_quality_analysis = False
        
        # Phase 1: Conditional Caching
        self._cached_conditionals = None
        self._cached_voice_profile_path = None
        self._cached_exaggeration = None
        self._cached_saved_voice_path = None
        self._cached_audio_prompt_path = None
        self._conditional_cache_hits = 0
        self._conditional_cache_misses = 0
        
        logger.info(f"✅ ChatterboxTTS initialized successfully with conditional caching")
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
            'upload_to_storage'
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

        # Startup environment diagnostics
        try:
            logger.info(
                "🧪 Audio env | PYDUB_AVAILABLE=%s, loudnorm_enabled=%s, loudnorm_method=%s, ffmpeg_available=%s, git_sha=%s",
                PYDUB_AVAILABLE,
                getattr(self.advanced_stitcher, "enable_loudness_normalization", False),
                getattr(self.advanced_stitcher, "loudness_method", ""),
                self.advanced_stitcher._ffmpeg_available() if hasattr(self.advanced_stitcher, "_ffmpeg_available") else False,
                _get_git_sha(),
            )
        except Exception:
            pass

    def _get_or_prepare_conditionals(self, voice_profile_path: str = None, 
                                   saved_voice_path: str = None, 
                                   audio_prompt_path: str = None,
                                   exaggeration: float = 0.5) -> Conditionals:
        """
        Get cached conditionals or prepare new ones.
        This is the core optimization that eliminates redundant conditional preparation.
        
        Args:
            voice_profile_path: Path to voice profile (.npy)
            saved_voice_path: Path to saved voice embedding
            audio_prompt_path: Path to audio prompt file
            exaggeration: Voice exaggeration factor
            
        Returns:
            Conditionals object ready for use
        """
        # Determine cache key based on input parameters
        if voice_profile_path:
            cache_key = ('voice_profile', voice_profile_path, exaggeration)
            current_path = voice_profile_path
            current_type = 'voice_profile'
        elif saved_voice_path and audio_prompt_path:
            cache_key = ('saved_voice', saved_voice_path, audio_prompt_path, exaggeration)
            current_path = f"{saved_voice_path}+{audio_prompt_path}"
            current_type = 'saved_voice'
        elif audio_prompt_path:
            cache_key = ('audio_prompt', audio_prompt_path, exaggeration)
            current_path = audio_prompt_path
            current_type = 'audio_prompt'
        else:
            raise ValueError("Must provide one of: voice_profile_path, (saved_voice_path + audio_prompt_path), or audio_prompt_path")
        
        # Check if we have valid cached conditionals
        if (self._cached_conditionals is not None and 
            cache_key == self._get_cache_key()):
            
            self._conditional_cache_hits += 1
            logger.debug(f"♻️ Conditional cache HIT ({self._conditional_cache_hits} hits, {self._conditional_cache_misses} misses)")
            logger.debug(f"   - Reusing conditionals for: {current_type} = {current_path}")
            return self._cached_conditionals
        
        # Cache miss - prepare new conditionals
        self._conditional_cache_misses += 1
        logger.info(f"🔄 Conditional cache MISS - preparing new conditionals")
        logger.info(f"   - Type: {current_type}")
        logger.info(f"   - Path: {current_path}")
        logger.info(f"   - Exaggeration: {exaggeration}")
        
        # Prepare conditionals based on type
        if voice_profile_path:
            self.prepare_conditionals_with_voice_profile(voice_profile_path, exaggeration)
        elif saved_voice_path and audio_prompt_path:
            self.prepare_conditionals_with_saved_voice(saved_voice_path, audio_prompt_path, exaggeration)
        elif audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration)
        
        # Cache the prepared conditionals (deep copy to avoid reference issues)
        import copy
        self._cached_conditionals = copy.deepcopy(self.conds)
        self._cached_voice_profile_path = voice_profile_path
        self._cached_saved_voice_path = saved_voice_path
        self._cached_audio_prompt_path = audio_prompt_path
        self._cached_exaggeration = exaggeration
        
        logger.info(f"✅ Conditionals prepared and cached successfully")
        return self._cached_conditionals
    
    def _get_cache_key(self):
        """Get the current cache key for comparison"""
        if self._cached_voice_profile_path:
            return ('voice_profile', self._cached_voice_profile_path, self._cached_exaggeration)
        elif self._cached_saved_voice_path and self._cached_audio_prompt_path:
            return ('saved_voice', self._cached_saved_voice_path, self._cached_audio_prompt_path, self._cached_exaggeration)
        elif self._cached_audio_prompt_path:
            return ('audio_prompt', self._cached_audio_prompt_path, self._cached_exaggeration)
        return None
    
    def clear_conditional_cache(self):
        """Clear the conditional cache to free memory"""
        logger.info("🧹 Clearing conditional cache")
        self._cached_conditionals = None
        self._cached_voice_profile_path = None
        self._cached_saved_voice_path = None
        self._cached_audio_prompt_path = None
        self._cached_exaggeration = None
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("🧹 GPU cache cleared")
    
    def get_conditional_cache_stats(self):
        """Get conditional cache statistics"""
        total_requests = self._conditional_cache_hits + self._conditional_cache_misses
        hit_rate = (self._conditional_cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self._conditional_cache_hits,
            'misses': self._conditional_cache_misses,
            'total_requests': total_requests,
            'hit_rate_percent': hit_rate,
            'cache_size': 1 if self._cached_conditionals is not None else 0
        }

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
        try:
            ve.to(device).eval()
        except Exception as e:
            logger.warning(f"Failed to move VoiceEncoder to device '{device}': {e}. Falling back to CPU.")
            device = "cpu"
            ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        try:
            t3.to(device).eval()
        except Exception as e:
            logger.warning(f"Failed to move T3 to device '{device}': {e}. Falling back to CPU.")
            device = "cpu"
            t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        try:
            s3gen.to(device).eval()
        except Exception as e:
            logger.warning(f"Failed to move S3Gen to device '{device}': {e}. Falling back to CPU.")
            device = "cpu"
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
        # Resolve/validate requested device, fallback to CPU when unavailable
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available or visible. Falling back to CPU.")
            device = "cpu"
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
        t3_cond_prompt_tokens = None
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
            # Ensure tensor is not in inference mode before in-place modification
            ref_speech_token_lens = ref_speech_token_lens.clone().detach()
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
        t3_cond_prompt_tokens = None
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
        t3_cond_prompt_tokens = None
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
        # Use conditional caching/preparation only when necessary. If conditionals
        # are already prepared, reuse them to avoid heavy recomputation per chunk.
        if self.conds is None:
            if voice_profile_path:
                self.prepare_conditionals_with_voice_profile(voice_profile_path, exaggeration=exaggeration)
            elif saved_voice_path and audio_prompt_path:
                self.prepare_conditionals_with_saved_voice(saved_voice_path, audio_prompt_path, exaggeration=exaggeration)
            elif audio_prompt_path:
                self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
            else:
                raise RuntimeError("Conditionals are not prepared. Provide voice_profile_path, (saved_voice_path + audio_prompt_path), or audio_prompt_path.")
        
        logger.info("✅ Conditionals prepared for TTS generation")

        # Norm and tokenize text
        # Text is assumed sanitized earlier in the pipeline; avoid re-normalizing here
        # to prevent double processing and preserve pacing.
        text = text
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
        return torch.from_numpy(wav).unsqueeze(0)

    def _generate_with_prepared_conditionals(
        self,
        text,
        conditionals: Conditionals,
        exaggeration=None,
        repetition_penalty=1.2,
        min_p=0.05,
        top_p=1.0,
        cfg_weight=0.5,
        temperature=0.8,
        max_new_tokens_override: Optional[int] = None,
    ):
        """
        Generate audio with provided conditionals, optionally overriding exaggeration per chunk.
        """
        # Validate conditionals
        if conditionals is None:
            raise RuntimeError("Conditionals must be provided to _generate_with_prepared_conditionals.")
        
        # Create a deep copy of conditionals to avoid modifying the original
        import copy
        chunk_conditionals = copy.deepcopy(conditionals)
        
        # Optional per-chunk emotion override without re-prepping conditionals
        # This allows AdaptiveParameterManager to vary 'exaggeration' per chunk.
        if exaggeration is not None and hasattr(chunk_conditionals, "t3") and hasattr(chunk_conditionals.t3, "emotion_adv"):
            try:
                chunk_conditionals.t3.emotion_adv = (torch.ones(1, 1, 1, device=self.device) * float(exaggeration))
            except Exception as e:
                logger.warning(f"Could not set per-chunk emotion_adv: {e}")

        # Text is assumed sanitized earlier in the pipeline
        text = text
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(
                t3_cond=chunk_conditionals.t3,
                text_tokens=text_tokens,
                max_new_tokens=max_new_tokens_override or 1000,  # dynamic cap for long chunks
                show_progress=True,
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
                ref_dict=chunk_conditionals.gen,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
        return torch.from_numpy(wav).unsqueeze(0)

    def chunk_text(self, text: str, max_chars: int = 500) -> List[ChunkInfo]:
        """
        Smart text chunking with advanced sanitization and content awareness.

        :param text: Full story text
        :param max_chars: Maximum number of characters per chunk
        :return: List of ChunkInfo objects with full analysis
        """
        # Step 1: Detect story breaks before sanitization
        story_break_positions = []
        if '⁂' in text:
            # Find all story break positions in the original text
            start_pos = 0
            while True:
                pos = text.find('⁂', start_pos)
                if pos == -1:
                    break
                story_break_positions.append(pos)
                start_pos = pos + 1
            logger.info(f"🎭 Found {len(story_break_positions)} story break(s) at positions: {story_break_positions}")
        
        # Step 2: Advanced text sanitization
        logger.debug(f"🧹 Applying advanced text sanitization to {len(text)} characters")
        sanitized_text = self.text_sanitizer.deep_clean(text)
        
        if len(sanitized_text) != len(text):
            logger.debug(f"🧹 Text sanitization: {len(text)} → {len(sanitized_text)} characters")
        
        # Step 3: Smart chunking with optional lead-sentence split
        target_chars = int(max_chars * 0.8)  # Target 80% of max for better quality

        # Simple chunking without complex opener logic
        chunk_infos = self.smart_chunker.smart_chunk(sanitized_text, target_chars, max_chars)
        
        # Step 4: Mark chunks that should have story break pauses
        if story_break_positions:
            self._mark_story_break_chunks(chunk_infos, story_break_positions, text)

        # Ensure chunk_infos is always defined and not empty
        if not chunk_infos:
            logger.warning("⚠️ No chunks generated, falling back to basic chunking")
            chunk_infos = self.smart_chunker.smart_chunk(sanitized_text, target_chars, max_chars)
        
        # Log detailed chunk analysis
        if chunk_infos:
            total_chars = sum(chunk.char_count for chunk in chunk_infos)
            avg_chars = total_chars / len(chunk_infos)
            complexity_scores = [chunk.complexity_score for chunk in chunk_infos]
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            
            logger.debug(f"📊 Advanced Chunk Analysis:")
            logger.debug(f"   - Total chunks: {len(chunk_infos)}")
            logger.debug(f"   - Avg chars/chunk: {avg_chars:.1f}")
            logger.debug(f"   - Avg complexity: {avg_complexity:.1f}/10")
            logger.debug(f"   - Content distribution: {self.smart_chunker._get_content_type_distribution(chunk_infos)}")
            
            # Sanitization impact analysis
            original_problematic = sum(1 for c in text if ord(c) > 127)
            sanitized_problematic = sum(1 for c in sanitized_text if ord(c) > 127)
            if original_problematic > sanitized_problematic:
                logger.debug(f"   - Problematic chars removed: {original_problematic - sanitized_problematic}")
        
        return chunk_infos
    
    def _mark_story_break_chunks(self, chunk_infos: List[ChunkInfo], story_break_positions: List[int], original_text: str) -> None:
        """
        Mark chunks that should have story break pauses based on original text positions.
        
        :param chunk_infos: List of chunk info objects
        :param story_break_positions: Positions of '⁂' symbols in original text
        :param original_text: Original text before sanitization
        """
        if not story_break_positions or not chunk_infos:
            return
        
        # Calculate cumulative character positions for each chunk in original text
        cumulative_chars = 0
        chunk_boundaries = []
        
        for chunk in chunk_infos:
            # Find this chunk's text in the original text (approximate)
            chunk_start = cumulative_chars
            chunk_end = cumulative_chars + chunk.char_count
            chunk_boundaries.append((chunk_start, chunk_end))
            cumulative_chars += chunk.char_count
        
        # Mark chunks that contain story breaks
        for break_pos in story_break_positions:
            for i, (start, end) in enumerate(chunk_boundaries):
                if start <= break_pos <= end:
                    chunk_infos[i].has_story_break = True
                    logger.info(f"🎭 Marked chunk {i} for story break pause")
                    break
        
        # Log summary
        story_break_chunks = sum(1 for chunk in chunk_infos if chunk.has_story_break)
        logger.info(f"🎭 Story break analysis: {len(story_break_positions)} breaks → {story_break_chunks} chunks marked")
    
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

    def _generate_single_chunk_with_quality(self, chunk_info: ChunkInfo, voice_profile_path: str, 
                                          pre_prepared_conditionals: Conditionals = None) -> Tuple[str, QualityScore]:
        """Generate a single chunk with quality analysis and retry logic"""
        adaptive_params = self.param_manager.get_adaptive_parameters(chunk_info)
        max_retries = 3
        
        # Fast path: QA disabled -> single attempt, no analysis
        if not getattr(self, "enable_quality_analysis", True):
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if pre_prepared_conditionals is not None:
                    audio_tensor = self._generate_with_prepared_conditionals(
                        text=chunk_info.text,
                        conditionals=pre_prepared_conditionals,
                        exaggeration=adaptive_params.get("exaggeration"),
                        temperature=adaptive_params.get("temperature", 0.8),
                        cfg_weight=adaptive_params.get("cfg_weight", 0.5),
                        repetition_penalty=adaptive_params.get("repetition_penalty", 1.2),
                        min_p=adaptive_params.get("min_p", 0.05),
                        top_p=adaptive_params.get("top_p", 1.0),
                    )
                else:
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

                temp_wav = tempfile.NamedTemporaryFile(suffix=f"_chunk_{chunk_info.id}.wav", delete=False)
                temp_wav_path = temp_wav.name
                temp_wav.close()
                torchaudio.save(temp_wav_path, audio_tensor, self.sr)
                return temp_wav_path, quality_score
            except Exception as e:
                logger.warning(f"⚠️ Chunk {chunk_info.id} generation failed with QA disabled: {e}")
                raise

        for attempt in range(max_retries):
            try:
                # Clear GPU cache before each attempt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Prefer using pre-prepared conditionals to avoid recomputation.
                if pre_prepared_conditionals is not None:
                    audio_tensor = self._generate_with_prepared_conditionals(
                        text=chunk_info.text,
                        conditionals=pre_prepared_conditionals,
                        exaggeration=adaptive_params.get("exaggeration"),
                        temperature=adaptive_params.get("temperature", 0.8),
                        cfg_weight=adaptive_params.get("cfg_weight", 0.5),
                        repetition_penalty=adaptive_params.get("repetition_penalty", 1.2),
                        min_p=adaptive_params.get("min_p", 0.05),
                        top_p=adaptive_params.get("top_p", 1.0),
                    )
                else:
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
                
                # Save to temporary file (close handle first for safety)
                temp_wav = tempfile.NamedTemporaryFile(suffix=f"_chunk_{chunk_info.id}_attempt_{attempt}.wav", delete=False)
                temp_wav_path = temp_wav.name
                temp_wav.close()
                torchaudio.save(temp_wav_path, audio_tensor, self.sr)

                # Validate audio file was created and has content
                if not Path(temp_wav_path).exists():
                    raise RuntimeError(f"Audio file was not created: {temp_wav_path}")

                # Check file size (should be > 44 bytes for WAV header + some data)
                file_size = Path(temp_wav_path).stat().st_size
                if file_size < 100:  # Very small file indicates likely silence/empty
                    logger.warning(f"⚠️ Chunk {chunk_info.id} generated very small audio file ({file_size} bytes)")
                    os.unlink(temp_wav_path)
                    raise RuntimeError(f"Generated audio file too small ({file_size} bytes)")

                # Load and validate audio tensor has actual content
                validation_tensor, _ = torchaudio.load(temp_wav_path)
                if validation_tensor.numel() == 0 or validation_tensor.abs().max().item() < 1e-6:
                    logger.warning(f"⚠️ Chunk {chunk_info.id} generated silent or empty audio")
                    os.unlink(temp_wav_path)
                    raise RuntimeError("Generated audio appears to be silent")

                # Quality analysis
                quality_score = self.quality_analyzer.analyze_chunk_quality(temp_wav_path, chunk_info)
                
                # Check if regeneration is needed
                if not quality_score.should_regenerate or attempt == max_retries - 1:
                    return temp_wav_path, quality_score
                else:
                    # Adjust parameters for retry
                    os.unlink(temp_wav_path)  # Clean up failed attempt
                    adaptive_params["temperature"] *= 0.95
                    adaptive_params["cfg_weight"] *= 1.05
                    logger.info(f"🔄 Chunk {chunk_info.id} quality score {quality_score.overall_score:.1f}/100, retrying...")
                    
            except Exception as e:
                logger.warning(f"⚠️ Chunk {chunk_info.id} attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
        
        raise RuntimeError(f"Failed to generate acceptable quality for chunk {chunk_info.id}")
    
    def generate_chunks_parallel(self, chunk_infos: List[ChunkInfo], voice_profile_path: str, 
                               pre_prepared_conditionals: Conditionals = None) -> List[Tuple[str, QualityScore]]:
        """Generate chunks in parallel with quality analysis"""
        logger.info(f"🚀 Starting parallel chunk generation ({self.max_parallel_workers} workers)")
        
        results = []
        with ThreadPoolExecutor(max_workers=self.max_parallel_workers) as executor:
            # Submit all tasks
            future_to_chunk = {
                executor.submit(self._generate_single_chunk_with_quality, chunk_info, voice_profile_path, pre_prepared_conditionals): chunk_info
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
        Now optimized with conditional caching for improved performance.

        :param chunk_infos: List of ChunkInfo objects with full analysis
        :param voice_profile_path: Path to the voice profile (.npy)
        :param base_temperature: Base temperature (will be adapted per chunk)
        :param base_exaggeration: Base exaggeration (will be adapted per chunk)
        :param base_cfg_weight: Base CFG weight (will be adapted per chunk)
        :return: List of file paths to temporary WAV files
        """
        generation_start = time.time()
        
        # Prepare conditionals once and reuse for all chunks to avoid recomputation
        logger.info(f"🎯 Preparing conditionals once for {len(chunk_infos)} chunks")
        try:
            # If a full voice profile (.npy) is provided, use the dedicated loader
            if isinstance(voice_profile_path, str) and voice_profile_path.lower().endswith('.npy'):
                self.prepare_conditionals_with_voice_profile(voice_profile_path, exaggeration=base_exaggeration)
            else:
                # Otherwise treat it as an audio prompt path
                self.prepare_conditionals(voice_profile_path, exaggeration=base_exaggeration)
        except Exception:
            logger.exception("❌ Failed to prepare conditionals")
            raise

        logger.info(f"🔄 Using sequential processing with cached conditionals for {len(chunk_infos)} chunks")
        wav_paths = []

        for i, chunk_info in enumerate(chunk_infos):
            logger.info(f"🎵 Generating chunk {i+1}/{len(chunk_infos)}: {chunk_info.text[:50]}...")

            # Use fixed parameters for speed (original working approach)
            audio_tensor = self._generate_with_prepared_conditionals(
                text=chunk_info.text,
                conditionals=self.conds,
                exaggeration=base_exaggeration,
                temperature=base_temperature,
                cfg_weight=base_cfg_weight,
                repetition_penalty=1.2,
                min_p=0.05,
                top_p=1.0,
            )

            # Save chunk to temporary file
            temp_wav = tempfile.NamedTemporaryFile(suffix=f"_chunk_{chunk_info.id}.wav", delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()
            torchaudio.save(temp_wav_path, audio_tensor, self.sr)

            wav_paths.append(temp_wav_path)

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
                          exaggeration: float = 0.5, cfg_weight: float = 0.5, pause_scale: float = 1.0) -> Tuple[torch.Tensor, int, Dict]:
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
        
        # Apply global pace factor to advanced stitcher
        try:
            self.advanced_stitcher.global_pause_factor = max(0.5, min(2.0, float(pause_scale)))
            logger.info(f"🕰️ Global pause scale set to {self.advanced_stitcher.global_pause_factor}")
        except Exception:
            logger.warning("⚠️ Failed to apply pause_scale; using default 1.0")
        
        wav_paths = self.generate_chunks(chunk_infos, voice_profile_path, temperature, exaggeration, cfg_weight)
        if not wav_paths:
            raise RuntimeError("Failed to generate any audio chunks")
        
        logger.info(f"🔗 Advanced stitching {len(wav_paths)} audio chunks...")
        audio_tensor, sample_rate, total_duration = self.stitch_and_normalize(wav_paths, chunk_infos, output_path, pause_ms)

        # Apply watermark once on final audio
        try:
            final_np = audio_tensor.squeeze(0).detach().cpu().numpy()
            final_np = self.watermarker.apply_watermark(final_np, sample_rate=sample_rate)
            audio_tensor = torch.from_numpy(final_np).unsqueeze(0)
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply final watermark: {e}")
        
        self.cleanup_chunks(wav_paths)
        
        logger.info(f"✅ TTS processing completed | Duration: {total_duration:.2f}s")
        logger.info(f"🔍 Final output path: {output_path}")
        logger.info(f"🔍 Output file exists: {Path(output_path).exists()}")
        
        # Enhanced metadata with cache performance
        cache_stats = self.get_conditional_cache_stats()
        metadata = self._create_generation_metadata(chunk_infos, total_duration, sample_rate, text, max_chars, 
                                                  pause_ms, pause_scale, cache_stats)
        metadata["output_path"] = output_path
        metadata["successful_chunks"] = len(wav_paths)
        
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
                _maybe_log_seg_levels("mp3 pre-export", audio_segment)
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
        
        # Move to CPU and convert tensor to numpy array
        if audio_tensor.is_cuda or (hasattr(torch.backends, 'mps') and audio_tensor.device.type == 'mps'):
            audio_tensor = audio_tensor.to('cpu')
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

    def upload_to_storage(self, data: bytes, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None) -> Optional[str]:
        """
        Upload data directly to R2 storage with metadata.
        Only R2 storage is supported.
        
        :param data: Binary data to upload
        :param destination_blob_name: Destination path in R2 (e.g., "private/users/{user_id}/stories/audio/{lang}/{story_id}/{version_id}.mp3")
        :param content_type: MIME type of the file
        :param metadata: Optional metadata to store with the file
        :return: Public URL
        """
        try:
            # Lazy import to avoid circular dependency
            from .vc import resolve_bucket_name, is_r2_bucket
            
            # Resolve region-aware bucket from metadata hints
            bucket_hint = (metadata or {}).get('bucket_name') if isinstance(metadata, dict) else None
            country_hint = (metadata or {}).get('country_code') if isinstance(metadata, dict) else None
            resolved_bucket = resolve_bucket_name(bucket_hint, country_hint)

            # Basic destination path sanitization
            dest_name = str(destination_blob_name or "").lstrip("/")

            # Only R2 is supported - verify bucket is R2
            if not is_r2_bucket(resolved_bucket):
                error_msg = f"Only R2 storage is supported. Bucket '{resolved_bucket}' is not an R2 bucket. Expected 'daezend-public-content'."
                logger.error(f"❌ {error_msg}")
                raise ValueError(error_msg)
            
            logger.info(f"✅ Using R2 upload for bucket: {resolved_bucket}")
            return self._upload_to_r2(data, dest_name, content_type, metadata)
            
        except Exception as e:
            logger.error(f"❌ Failed to upload: {e}")
            return None
    
    def _upload_to_r2(self, data: bytes, destination_key: str, content_type: str = "application/octet-stream", metadata: dict = None) -> Optional[str]:
        """
        Upload data to Cloudflare R2 using boto3 S3 client.
        
        :param data: Binary data to upload
        :param destination_key: Destination key/path in R2
        :param content_type: MIME type of the file
        :param metadata: Optional metadata dict (will be stored as R2 metadata)
        :return: Public URL or None if failed
        """
        try:
            import boto3
            import os
            
            # Get R2 credentials from environment
            r2_account_id = os.getenv('R2_ACCOUNT_ID')
            r2_access_key_id = os.getenv('R2_ACCESS_KEY_ID')
            r2_secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
            r2_endpoint = os.getenv('R2_ENDPOINT')
            r2_bucket_name = os.getenv('R2_BUCKET_NAME', 'daezend-public-content')
            r2_public_url = os.getenv('NEXT_PUBLIC_R2_PUBLIC_URL') or os.getenv('R2_PUBLIC_URL')
            
            if not all([r2_account_id, r2_access_key_id, r2_secret_access_key, r2_endpoint]):
                logger.error("❌ R2 credentials not configured")
                return None
            
            # Create S3 client for R2
            s3_client = boto3.client(
                's3',
                endpoint_url=r2_endpoint,
                aws_access_key_id=r2_access_key_id,
                aws_secret_access_key=r2_secret_access_key,
                region_name='auto'
            )
            
            # Prepare metadata for R2
            extra_args = {
                'ContentType': content_type,
            }
            if metadata:
                # R2 metadata must be strings
                extra_args['Metadata'] = {str(k): str(v) for k, v in metadata.items()}
            
            # Upload to R2
            s3_client.put_object(
                Bucket=r2_bucket_name,
                Key=destination_key,
                Body=data,
                **extra_args
            )
            
            logger.info(f"✅ Uploaded to R2: {destination_key} ({len(data)} bytes)")
            
            # Return public URL if available
            if r2_public_url:
                public_url = f"{r2_public_url.rstrip('/')}/{destination_key}"
                return public_url
            
            # Fallback: return R2 path
            return destination_key
            
        except Exception as e:
            logger.error(f"❌ R2 upload failed: {e}")
            import traceback
            logger.error(f"❌ R2 upload traceback: {traceback.format_exc()}")
            return None
    
    def _download_from_r2(self, source_key: str) -> Optional[bytes]:
        """
        Download data from Cloudflare R2 using boto3 S3 client.
        
        :param source_key: Source key/path in R2
        :return: Binary data or None if failed
        """
        try:
            import boto3
            import os
            
            # Get R2 credentials from environment
            r2_account_id = os.getenv('R2_ACCOUNT_ID')
            r2_access_key_id = os.getenv('R2_ACCESS_KEY_ID')
            r2_secret_access_key = os.getenv('R2_SECRET_ACCESS_KEY')
            r2_endpoint = os.getenv('R2_ENDPOINT')
            r2_bucket_name = os.getenv('R2_BUCKET_NAME', 'daezend-public-content')
            
            if not all([r2_account_id, r2_access_key_id, r2_secret_access_key, r2_endpoint]):
                logger.error("❌ R2 credentials not configured")
                return None
            
            # Create S3 client for R2
            s3_client = boto3.client(
                's3',
                endpoint_url=r2_endpoint,
                aws_access_key_id=r2_access_key_id,
                aws_secret_access_key=r2_secret_access_key,
                region_name='auto'
            )
            
            # Download from R2
            response = s3_client.get_object(
                Bucket=r2_bucket_name,
                Key=source_key
            )
            
            data = response['Body'].read()
            logger.info(f"✅ Downloaded from R2: {source_key} ({len(data)} bytes)")
            return data
            
        except Exception as e:
            logger.error(f"❌ R2 download failed: {e}")
            import traceback
            logger.error(f"❌ R2 download traceback: {traceback.format_exc()}")
            return None

    def generate_tts_story(self, text: str, voice_id: str, profile_base64: str = "", 
                           language: str = 'en', story_type: str = 'user', 
                           is_kids_voice: bool = False, metadata: Dict = None, pause_scale: float = 1.15,
                           *, user_id: str = "", story_id: str = "", profile_path: str = "", voice_name: str = "") -> Dict:
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
            Dict with status, audio_data, storage_url (R2), storage_path (R2), story_type, and generation_time
        """
        import time
        import base64
        import tempfile
        import os
        
        start_time = time.time()
        
        # Extract voice_name from metadata if not provided directly
        if not voice_name and metadata and isinstance(metadata, dict) and 'voice_name' in metadata:
            voice_name = metadata['voice_name']
        
        # Fallback to voice_id if no voice_name provided
        if not voice_name:
            voice_name = voice_id
        
        logger.info(f"📚 ChatterboxTTS.generate_tts_story called")
        logger.info(f"  - text length: {len(text)}")
        logger.info(f"  - voice_id: {voice_id}")
        logger.info(f"  - voice_name: {voice_name}")
        logger.info(f"  - language: {language}")
        logger.info(f"  - story_type: {story_type}")
        logger.info(f"  - is_kids_voice: {is_kids_voice}")
        logger.info(f"  - metadata: {metadata}")
        
        try:
            # Step 1: Load voice profile from base64 or R2 storage path
            if profile_base64:
                logger.info(f"  - Step 1: Loading voice profile from base64...")
                profile_bytes = base64.b64decode(profile_base64)
            elif profile_path:
                logger.info(f"  - Step 1: Loading voice profile from storage path: {profile_path}")
                try:
                    # Lazy import to avoid circular dependency
                    from .vc import resolve_bucket_name, is_r2_bucket
                    
                    # Resolve bucket from metadata hints (non-R2 bucket names are ignored)
                    bucket_hint = (metadata or {}).get('bucket_name') if isinstance(metadata, dict) else None
                    country_hint = (metadata or {}).get('country_code') if isinstance(metadata, dict) else None
                    resolved_bucket = resolve_bucket_name(bucket_hint, country_hint)
                    
                    # resolve_bucket_name() always returns an R2 bucket (ignores non-R2 bucket names)
                    logger.info(f"    - Using R2 download (bucket={resolved_bucket})")
                    profile_bytes = self._download_from_r2(profile_path)
                    if not profile_bytes:
                        raise ValueError(f"Failed to download profile from R2: {profile_path}")
                    logger.info(f"    - Voice profile downloaded from R2 (bucket={resolved_bucket})")
                except Exception as storage_e:
                    logger.error(f"❌ Failed to download profile from storage: {storage_e}")
                    raise
            else:
                raise ValueError("Either profile_base64 or profile_path must be provided")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp_file:
                temp_file.write(profile_bytes)
                temp_profile_path = temp_file.name
            
            logger.info(f"    - Voice profile loaded")
            
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
                cfg_weight=0.5,
                pause_scale=pause_scale
            )
            
            # Step 3: Convert to MP3 bytes
            logger.info(f"  - Step 3: Converting to MP3...")
            mp3_bytes = self.tensor_to_mp3_bytes(audio_tensor, sample_rate, "96k")
            
            # Step 4: Upload directly to R2
            logger.info(f"  - Step 4: Uploading directly to R2...")
            
            # Extract story_type from metadata or use direct parameter
            final_story_type = story_type  # Start with direct parameter
            if metadata and isinstance(metadata, dict) and 'story_type' in metadata:
                final_story_type = metadata['story_type']  # Override with metadata if available
            
            # Ensure story_type is valid (user or app)
            if final_story_type not in ['user', 'app']:
                logger.warning(f"Invalid story_type '{final_story_type}', defaulting to 'user'")
                final_story_type = 'user'
            
            # Determine storage path based on admin generation flag
            is_admin_gen = (metadata or {}).get('is_admin_generation', False) if isinstance(metadata, dict) else False
            storage_path_hint = (metadata or {}).get('storage_path', '') if isinstance(metadata, dict) else ''
            
            if is_admin_gen and storage_path_hint:
                # Admin generation: use provided storage path (R2)
                import random
                suffix = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=4))
                unique_filename = f"{voice_id}_{suffix}.mp3"
                r2_path = f"{storage_path_hint.rstrip('/')}/{unique_filename}"
                version_id = suffix  # Use suffix as version_id for admin generation
            else:
                # Regular user story generation: use new R2 path structure
                # Generate versionId (timestamp-based)
                version_id = f"{int(time.time() * 1000)}"
                
                # Validate required fields for R2 path
                if not user_id or not story_id or not language:
                    raise ValueError(f"Missing required fields for R2 path: user_id={user_id}, story_id={story_id}, language={language}")
                
                # Use new R2 path structure: private/users/{userId}/stories/audio/{language}/{storyId}/{versionId}.mp3
                r2_path = f"private/users/{user_id}/stories/audio/{language}/{story_id}/{version_id}.mp3"
            
            logger.info(f"    - R2 path: {r2_path}")
            logger.info(f"    - Story type: {final_story_type}")
            logger.info(f"    - Is admin generation: {is_admin_gen}")
            logger.info(f"    - Version ID: {version_id}")
            
            # Upload directly to R2
            r2_url = None
            try:
                # Always use R2 bucket for user stories
                r2_url = self.upload_to_storage(
                    data=mp3_bytes,
                    destination_blob_name=r2_path,
                    content_type="audio/mpeg",
                    metadata={
                        "bucket_name": "daezend-public-content",  # Always R2 for user stories
                        "user_id": user_id,
                        "story_id": story_id,
                        "voice_id": voice_id,
                        "voice_name": voice_name,
                        "language": language,
                        "story_type": final_story_type,
                        "text_length": len(text),
                        "generation_time": time.time() - start_time,
                        "audio_size": len(mp3_bytes),
                        "duration": generation_metadata.get("duration_sec", 0),
                    }
                )
                if r2_url:
                    logger.info(f"    - Uploaded successfully to R2: {r2_url}")
                else:
                    logger.warning("⚠️ R2 upload did not return URL")
            except Exception as upload_error:
                logger.error(f"❌ R2 upload failed: {upload_error}")
                import traceback
                logger.error(f"❌ R2 upload traceback: {traceback.format_exc()}")
                r2_url = None
                raise  # Re-raise to fail the generation
            
            # Convert to base64
            audio_base64 = base64.b64encode(mp3_bytes).decode('utf-8')
            
            # Clean up temporary file
            os.unlink(temp_profile_path)
            
            generation_time = time.time() - start_time
            
            # Return result with R2 URL and path
            result = {
                "status": "success",
                "audio_data": audio_base64,
                "storage_url": r2_url,  # R2 public URL (primary)
                "storage_path": r2_path,  # R2 path (primary)
                "r2_path": r2_path,  # Explicit R2 path for callback validation
                "r2_url": r2_url,  # Explicit R2 URL
                "audio_url": r2_url,  # Alias for compatibility
                # Backward compatibility keys (deprecated, use storage_url/storage_path)
                "firebase_url": r2_url,
                "firebase_path": r2_path,
                "version_id": version_id,
                "story_type": final_story_type,
                "generation_time": generation_time,
                "duration": generation_metadata.get("duration_sec", 0)
            }

            # Firestore write for stories/{story_id}
            try:
                from google.cloud import firestore
                client = firestore.Client()
                from google.cloud.firestore import SERVER_TIMESTAMP  # type: ignore
                if story_id:
                    doc = client.collection("stories").document(story_id)
                    # Build new audio version entry
                    # Use version_id from path generation (already generated above)
                    new_version = {
                        "id": version_id,
                        "voiceId": voice_id,
                        "voiceName": voice_name,
                        "audioUrl": r2_url or "",
                        "url": r2_url or "",
                        "createdAt": SERVER_TIMESTAMP,
                        "updatedAt": SERVER_TIMESTAMP,
                        "service": "chatterbox",
                        "metadata": {
                            "format": "mp3",
                            "size": len(mp3_bytes),
                            "duration": generation_metadata.get("duration_sec", 0) if 'generation_metadata' in locals() else 0,
                            "voiceName": voice_name,
                            "r2Path": r2_path,  # Store R2 path in metadata
                        },
                    }
                    # Update atomically: set status and push new version
                    doc.set({
                        "audioStatus": "ready",
                        "audioUrl": r2_url or "",
                        "updatedAt": SERVER_TIMESTAMP,
                    }, merge=True)
                    # Firestore array union append (fallback to read-modify-write if needed)
                    try:
                        from google.cloud.firestore_v1 import ArrayUnion
                        doc.update({"audioVersions": ArrayUnion([new_version])})
                    except Exception:
                        # Read-modify-write fallback
                        snap = doc.get()
                        existing = []
                        if snap.exists and isinstance(snap.to_dict().get("audioVersions"), list):
                            existing = snap.to_dict()["audioVersions"]
                        existing.append(new_version)
                        doc.set({"audioVersions": existing}, merge=True)
                    result["firestore_story_id"] = story_id
            except Exception as fe:
                logger.warning(f"⚠️ Firestore update for story failed: {fe}")
            
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

    def generate_chunks_with_saved_voice(self, chunk_infos: List[ChunkInfo], saved_voice_path: str, 
                                       audio_prompt_path: str, base_temperature: float = 0.8, 
                                       base_exaggeration: float = 0.5, base_cfg_weight: float = 0.5) -> List[str]:
        """
        Generate chunks using saved voice + audio prompt with conditional caching optimization.
        
        :param chunk_infos: List of ChunkInfo objects with full analysis
        :param saved_voice_path: Path to saved voice embedding
        :param audio_prompt_path: Path to audio prompt file
        :param base_temperature: Base temperature (will be adapted per chunk)
        :param base_exaggeration: Base exaggeration (will be adapted per chunk)
        :param base_cfg_weight: Base CFG weight (will be adapted per chunk)
        :return: List of file paths to temporary WAV files
        """
        generation_start = time.time()
        
        # TEMPORARILY DISABLED: Conditional caching to isolate voice issues
        # Phase 1: Prepare conditionals once for all chunks
        # logger.info(f"🎯 Preparing conditionals once for {len(chunk_infos)} chunks (saved voice + audio prompt)")
        # pre_prepared_conditionals = self._get_or_prepare_conditionals(
        #     saved_voice_path=saved_voice_path,
        #     audio_prompt_path=audio_prompt_path,
        #     exaggeration=base_exaggeration
        # )
        #
        # # Log cache performance
        # cache_stats = self.get_conditional_cache_stats()
        # logger.info(f"📊 Conditional cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate_percent']:.1f}% hit rate)")
        #
        # # Use the same generation logic as the main method
        # return self._generate_chunks_with_prepared_conditionals(
        #     chunk_infos, pre_prepared_conditionals, generation_start
        # )
        
        # Use original sequential generation logic
        logger.info(f"🔄 Using original sequential processing for {len(chunk_infos)} chunks (saved voice + audio prompt)")
        wav_paths = []
        quality_scores = []
        
        for chunk_info in chunk_infos:
            wav_path, quality_score = self._generate_single_chunk_with_quality(
                chunk_info, None, None  # No pre-prepared conditionals
            )
            wav_paths.append(wav_path)
            quality_scores.append(quality_score)
        
        # Log comprehensive quality analysis
        total_generation_time = time.time() - generation_start
        self._log_quality_analysis(chunk_infos, quality_scores, total_generation_time)
        
        return wav_paths
    
    def generate_chunks_with_audio_prompt(self, chunk_infos: List[ChunkInfo], audio_prompt_path: str,
                                        base_temperature: float = 0.8, base_exaggeration: float = 0.5, 
                                        base_cfg_weight: float = 0.5) -> List[str]:
        """
        Generate chunks using audio prompt with conditional caching optimization.
        
        :param chunk_infos: List of ChunkInfo objects with full analysis
        :param audio_prompt_path: Path to audio prompt file
        :param base_temperature: Base temperature (will be adapted per chunk)
        :param base_exaggeration: Base exaggeration (will be adapted per chunk)
        :param base_cfg_weight: Base CFG weight (will be adapted per chunk)
        :return: List of file paths to temporary WAV files
        """
        generation_start = time.time()
        
        # TEMPORARILY DISABLED: Conditional caching to isolate voice issues
        # Phase 1: Prepare conditionals once for all chunks
        # logger.info(f"🎯 Preparing conditionals once for {len(chunk_infos)} chunks (audio prompt)")
        # pre_prepared_conditionals = self._get_or_prepare_conditionals(
        #     audio_prompt_path=audio_prompt_path,
        #     exaggeration=base_exaggeration
        # )
        #
        # # Log cache performance
        # cache_stats = self.get_conditional_cache_stats()
        # logger.info(f"📊 Conditional cache stats: {cache_stats['hits']} hits, {cache_stats['misses']} misses ({cache_stats['hit_rate_percent']:.1f}% hit rate)")
        #
        # # Use the same generation logic as the main method
        # return self._generate_chunks_with_prepared_conditionals(
        #     chunk_infos, pre_prepared_conditionals, generation_start
        # )
        
        # Use original sequential generation logic
        logger.info(f"🔄 Using original sequential processing for {len(chunk_infos)} chunks (audio prompt)")
        wav_paths = []
        quality_scores = []
        
        for chunk_info in chunk_infos:
            wav_path, quality_score = self._generate_single_chunk_with_quality(
                chunk_info, None, None  # No pre-prepared conditionals
            )
            wav_paths.append(wav_path)
            quality_scores.append(quality_score)
        
        # Log comprehensive quality analysis
        total_generation_time = time.time() - generation_start
        self._log_quality_analysis(chunk_infos, quality_scores, total_generation_time)
        
        return wav_paths
    
    def _generate_chunks_with_prepared_conditionals(self, chunk_infos: List[ChunkInfo], 
                                                  pre_prepared_conditionals: Conditionals,
                                                  generation_start: float) -> List[str]:
        """
        Internal method to generate chunks with pre-prepared conditionals.
        This centralizes the generation logic for all conditional types.
        
        :param chunk_infos: List of ChunkInfo objects
        :param pre_prepared_conditionals: Pre-prepared conditionals
        :param generation_start: Start time for performance tracking
        :return: List of file paths to temporary WAV files
        """
        if self.enable_parallel_processing and len(chunk_infos) > 1:
            # Use parallel processing with quality analysis
            logger.info(f"🚀 Using parallel processing for {len(chunk_infos)} chunks")
            chunk_results = self.generate_chunks_parallel(chunk_infos, None, pre_prepared_conditionals)
            wav_paths = [wav_path for wav_path, _ in chunk_results]
            quality_scores = [quality_score for _, quality_score in chunk_results]
        else:
            # Use sequential processing (fallback or single chunk)
            logger.info(f"🔄 Using sequential processing for {len(chunk_infos)} chunks")
            wav_paths = []
            quality_scores = []
            
            for chunk_info in chunk_infos:
                wav_path, quality_score = self._generate_single_chunk_with_quality(
                    chunk_info, None, pre_prepared_conditionals
                )
                wav_paths.append(wav_path)
                quality_scores.append(quality_score)
        
        # Log comprehensive quality analysis
        total_generation_time = time.time() - generation_start
        self._log_quality_analysis(chunk_infos, quality_scores, total_generation_time)
        
        return wav_paths

    def generate_long_text_with_saved_voice(self, text: str, saved_voice_path: str, audio_prompt_path: str, 
                                          output_path: str, max_chars: int = 500, pause_ms: int = 100, 
                                          temperature: float = 0.8, exaggeration: float = 0.5, 
                                          cfg_weight: float = 0.5, pause_scale: float = 1.0) -> Tuple[torch.Tensor, int, Dict]:
        """
        Generate long text TTS using saved voice + audio prompt with conditional caching optimization.
        
        :param text: Input text to synthesize
        :param saved_voice_path: Path to saved voice embedding
        :param audio_prompt_path: Path to audio prompt file
        :param output_path: Path to save the final audio file
        :param max_chars: Maximum number of characters per chunk
        :param pause_ms: Milliseconds of silence between chunks
        :param temperature: Generation temperature
        :param exaggeration: Voice exaggeration factor
        :param cfg_weight: CFG weight for generation
        :param pause_scale: Global pause scaling factor
        :return: Tuple of (audio_tensor, sample_rate, metadata_dict)
        """
        logger.info(f"🎵 Starting optimized TTS processing for {len(text)} characters (saved voice + audio prompt)")
        logger.info(f"🔍 Output path: {output_path}")
        
        # Safety check for extremely long texts
        if len(text) > 13000:
            logger.warning(f"⚠️ Very long text ({len(text)} chars) - truncating to safe length")
            text = text[:13000] + "... [truncated]"
            logger.info(f"📝 Truncated text to {len(text)} characters")
        
        chunk_infos = self.chunk_text(text, max_chars)
        logger.info(f"📦 Split into {len(chunk_infos)} intelligent chunks")
        
        # Apply global pace factor to advanced stitcher
        try:
            self.advanced_stitcher.global_pause_factor = max(0.5, min(2.0, float(pause_scale)))
            logger.info(f"🕰️ Global pause scale set to {self.advanced_stitcher.global_pause_factor}")
        except Exception:
            logger.warning("⚠️ Failed to apply pause_scale; using default 1.0")
        
        # Prepare conditionals once for saved voice + prompt
        self.prepare_conditionals_with_saved_voice(saved_voice_path, audio_prompt_path, exaggeration=exaggeration)

        # Generate using centralized prepared-conditional path
        generation_start = time.time()
        wav_paths: List[str] = []
        quality_scores: List[QualityScore] = []
        for chunk_info in chunk_infos:
            wav_path, quality_score = self._generate_single_chunk_with_quality(
                chunk_info, None, self.conds
            )
            wav_paths.append(wav_path)
            quality_scores.append(quality_score)
        if not wav_paths:
            raise RuntimeError("Failed to generate any audio chunks")
        
        logger.info(f"🔗 Advanced stitching {len(wav_paths)} audio chunks...")
        audio_tensor, sample_rate, total_duration = self.stitch_and_normalize(wav_paths, chunk_infos, output_path, pause_ms)

        # Apply watermark once on final audio
        try:
            final_np = audio_tensor.squeeze(0).detach().cpu().numpy()
            final_np = self.watermarker.apply_watermark(final_np, sample_rate=sample_rate)
            audio_tensor = torch.from_numpy(final_np).unsqueeze(0)
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply final watermark: {e}")
        
        self.cleanup_chunks(wav_paths)
        
        logger.info(f"✅ Optimized TTS processing completed | Duration: {total_duration:.2f}s")
        logger.info(f"🔍 Final output path: {output_path}")
        logger.info(f"🔍 Output file exists: {Path(output_path).exists()}")
        
        # Enhanced metadata with cache performance
        cache_stats = self.get_conditional_cache_stats()
        metadata = self._create_generation_metadata(chunk_infos, total_duration, sample_rate, text, max_chars, 
                                                  pause_ms, pause_scale, cache_stats)
        
        return audio_tensor, sample_rate, metadata
    
    def generate_long_text_with_audio_prompt(self, text: str, audio_prompt_path: str, output_path: str,
                                           max_chars: int = 500, pause_ms: int = 100, temperature: float = 0.8,
                                           exaggeration: float = 0.5, cfg_weight: float = 0.5, 
                                           pause_scale: float = 1.0) -> Tuple[torch.Tensor, int, Dict]:
        """
        Generate long text TTS using audio prompt with conditional caching optimization.
        
        :param text: Input text to synthesize
        :param audio_prompt_path: Path to audio prompt file
        :param output_path: Path to save the final audio file
        :param max_chars: Maximum number of characters per chunk
        :param pause_ms: Milliseconds of silence between chunks
        :param temperature: Generation temperature
        :param exaggeration: Voice exaggeration factor
        :param cfg_weight: CFG weight for generation
        :param pause_scale: Global pause scaling factor
        :return: Tuple of (audio_tensor, sample_rate, metadata_dict)
        """
        logger.info(f"🎵 Starting optimized TTS processing for {len(text)} characters (audio prompt)")
        logger.info(f"🔍 Output path: {output_path}")
        
        # Safety check for extremely long texts
        if len(text) > 13000:
            logger.warning(f"⚠️ Very long text ({len(text)} chars) - truncating to safe length")
            text = text[:13000] + "... [truncated]"
            logger.info(f"📝 Truncated text to {len(text)} characters")
        
        chunk_infos = self.chunk_text(text, max_chars)
        logger.info(f"📦 Split into {len(chunk_infos)} intelligent chunks")
        
        # Apply global pace factor to advanced stitcher
        try:
            self.advanced_stitcher.global_pause_factor = max(0.5, min(2.0, float(pause_scale)))
            logger.info(f"🕰️ Global pause scale set to {self.advanced_stitcher.global_pause_factor}")
        except Exception:
            logger.warning("⚠️ Failed to apply pause_scale; using default 1.0")
        
        # Prepare conditionals once for audio prompt
        self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)

        # Generate using centralized prepared-conditional path
        wav_paths: List[str] = []
        quality_scores: List[QualityScore] = []
        for chunk_info in chunk_infos:
            wav_path, quality_score = self._generate_single_chunk_with_quality(
                chunk_info, None, self.conds
            )
            wav_paths.append(wav_path)
            quality_scores.append(quality_score)
        if not wav_paths:
            raise RuntimeError("Failed to generate any audio chunks")
        
        logger.info(f"🔗 Advanced stitching {len(wav_paths)} audio chunks...")
        audio_tensor, sample_rate, total_duration = self.stitch_and_normalize(wav_paths, chunk_infos, output_path, pause_ms)

        # Apply watermark once on final audio
        try:
            final_np = audio_tensor.squeeze(0).detach().cpu().numpy()
            final_np = self.watermarker.apply_watermark(final_np, sample_rate=sample_rate)
            audio_tensor = torch.from_numpy(final_np).unsqueeze(0)
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply final watermark: {e}")
        
        self.cleanup_chunks(wav_paths)
        
        logger.info(f"✅ Optimized TTS processing completed | Duration: {total_duration:.2f}s")
        logger.info(f"🔍 Final output path: {output_path}")
        logger.info(f"🔍 Output file exists: {Path(output_path).exists()}")
        
        # Enhanced metadata with cache performance
        cache_stats = self.get_conditional_cache_stats()
        metadata = self._create_generation_metadata(chunk_infos, total_duration, sample_rate, text, max_chars, 
                                                  pause_ms, pause_scale, cache_stats)
        
        return audio_tensor, sample_rate, metadata
    
    def _create_generation_metadata(self, chunk_infos: List[ChunkInfo], total_duration: float, 
                                  sample_rate: int, text: str, max_chars: int, pause_ms: int, 
                                  pause_scale: float, cache_stats: Dict) -> Dict:
        """Create comprehensive metadata for generation results"""
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
        
        return {
            # Basic metrics
            "chunk_count": len(chunk_infos),
            "duration_sec": total_duration,
            "sample_rate": sample_rate,
            "text_length": len(text),
            "max_chars_per_chunk": max_chars,
            "pause_ms": pause_ms,
            "pause_scale": pause_scale,
            
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
            "audio_chars_per_second": round(len(text) / max(total_duration, 0.1), 1),
            "audio_efficiency_ratio": round(total_duration / max(len(text) * 0.08, 1), 2),
            
            # Conditional caching performance
            "conditional_cache_hits": cache_stats['hits'],
            "conditional_cache_misses": cache_stats['misses'],
            "conditional_cache_hit_rate": round(cache_stats['hit_rate_percent'], 1),
            "conditional_cache_total_requests": cache_stats['total_requests'],
            "optimization_enabled": True,
        }