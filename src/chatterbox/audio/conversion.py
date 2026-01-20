"""Audio conversion utilities for PyTorch tensors."""
import os
import tempfile
import logging
from typing import Optional

import torch
import numpy as np
import torchaudio

from ..utils import PYDUB_AVAILABLE, _maybe_log_seg_levels

logger = logging.getLogger(__name__)


def tensor_to_mp3_bytes(audio_tensor: torch.Tensor, sample_rate: int, bitrate: str = "96k") -> bytes:
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
            audio_segment = tensor_to_audiosegment(audio_tensor, sample_rate)
            _maybe_log_seg_levels("mp3 pre-export", audio_segment)
            # Export to MP3 bytes
            mp3_file = audio_segment.export(format="mp3", bitrate=bitrate)
            # Read the bytes from the file object
            mp3_bytes = mp3_file.read()
            return mp3_bytes
        except Exception as e:
            logger.warning(f"Direct MP3 conversion failed: {e}, falling back to WAV")
            return tensor_to_wav_bytes(audio_tensor, sample_rate)
    else:
        logger.warning("pydub not available, falling back to WAV")
        return tensor_to_wav_bytes(audio_tensor, sample_rate)


def tensor_to_audiosegment(audio_tensor: torch.Tensor, sample_rate: int):
    """
    Convert PyTorch audio tensor to pydub AudioSegment.
    
    :param audio_tensor: PyTorch audio tensor
    :param sample_rate: Audio sample rate
    :return: pydub AudioSegment
    """
    if not PYDUB_AVAILABLE:
        raise ImportError("pydub is required for audio conversion")
    
    from pydub import AudioSegment
    
    # Move to CPU and convert tensor to numpy array
    if audio_tensor.is_cuda or (hasattr(torch.backends, 'mps') and audio_tensor.device.type == 'mps'):
        audio_tensor = audio_tensor.to('cpu')
    if audio_tensor.dim() == 2:
        # Stereo: (channels, samples)
        audio_np = audio_tensor.numpy()
    else:
        # Mono: (samples,) -> (1, samples)
        audio_np = audio_tensor.unsqueeze(0).numpy()
    
    # CRITICAL FIX: Clamp audio to [-1, 1] range BEFORE conversion to prevent clipping
    audio_np = np.clip(audio_np, -1.0, 1.0)
    
    # Apply headroom to prevent clipping during int16 conversion
    # Use -0.3 dBFS (0.966) to leave headroom for MP3 encoding
    headroom_factor = 0.966  # ~-0.3 dBFS
    audio_np = audio_np * headroom_factor
    
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


def tensor_to_wav_bytes(audio_tensor: torch.Tensor, sample_rate: int) -> bytes:
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


def convert_audio_file_to_mp3(input_path: str, output_path: Optional[str] = None, bitrate: str = "96k") -> str:
    """
    Convert an audio file on disk to MP3 format.

    :param input_path: Path to an existing audio file readable by pydub/ffmpeg.
    :param output_path: Optional destination path. Defaults to <stem>.mp3 in same dir.
    :param bitrate: Target MP3 bitrate string (e.g. "96k").
    :return: Path to the written MP3 file.
    """
    if not PYDUB_AVAILABLE:
        raise ImportError("pydub/ffmpeg required for convert_audio_file_to_mp3")

    from pydub import AudioSegment

    if output_path is None:
        base, _ = os.path.splitext(input_path)
        output_path = f"{base}.mp3"

    audio_seg = AudioSegment.from_file(input_path)
    _maybe_log_seg_levels("file->mp3 pre-export", audio_seg)
    audio_seg.export(output_path, format="mp3", bitrate=bitrate)
    return output_path

