"""Quality analyzer for audio chunks."""
import logging
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torchaudio

from .types import QualityScore
from ..chunking.types import ChunkInfo

logger = logging.getLogger(__name__)


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
        self.regen_mode = str(os.getenv("CHATTERBOX_QA_REGEN_MODE", "silence_only")).strip().lower()
        if self.regen_mode not in {"silence_only", "broad", "off"}:
            logger.warning("Invalid CHATTERBOX_QA_REGEN_MODE=%s, defaulting to silence_only", self.regen_mode)
            self.regen_mode = "silence_only"
        logger.info("🧪 QA regen mode: %s", self.regen_mode)
    
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
            
            # Decide whether we should regenerate this chunk.
            #
            # Rationale: In production we sometimes get "valid" waveforms that are mostly silence
            # (or have long leading silence). The generation call doesn't throw, so without this
            # gate the final stitched audio contains multi-second silent spans.
            #
            if self.regen_mode == "off":
                regen_triggers = set()
            elif self.regen_mode == "broad":
                regen_triggers = {
                    "excessive_silence",
                    "silence_at_start",
                    "silence_at_end",
                    "too_short",
                    "too_quiet",
                    "low_energy",
                    "fragmented_audio",
                }
            else:
                # We only request regeneration for true silence-type failures.
                regen_triggers = {
                    "excessive_silence",
                    "silence_at_start",
                    "silence_at_end",
                }
            should_regenerate = any(issue in regen_triggers for issue in quality_issues)
            
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

