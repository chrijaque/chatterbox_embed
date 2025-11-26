"""Smart text chunking with content awareness."""
import logging
from typing import List, Tuple, Dict

from .types import ContentType, ChunkInfo

logger = logging.getLogger(__name__)


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

