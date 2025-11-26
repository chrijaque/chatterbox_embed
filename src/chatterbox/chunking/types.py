"""Types for text chunking."""
from dataclasses import dataclass
from enum import Enum


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

