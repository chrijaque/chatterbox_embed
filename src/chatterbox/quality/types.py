"""Types for quality analysis."""
from dataclasses import dataclass
from typing import List


@dataclass
class QualityScore:
    """Quality assessment results for audio chunks"""
    overall_score: float  # 0-100 score
    issues: List[str]     # List of detected issues
    duration: float       # Audio duration in seconds
    silence_ratio: float  # Ratio of silence to total audio
    peak_db: float        # Peak audio level in dB
    rms_db: float        # RMS audio level in dB
    should_regenerate: bool = False

