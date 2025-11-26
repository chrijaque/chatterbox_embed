"""Advanced audio stitching with fade transitions and smart pauses."""
import logging
from typing import List, Optional, Tuple

import torchaudio

from ..chunking.types import ContentType, ChunkInfo
from ..utils import PYDUB_AVAILABLE

logger = logging.getLogger(__name__)

if PYDUB_AVAILABLE:
    from pydub import AudioSegment, effects


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
        if not PYDUB_AVAILABLE:
            return segment
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

