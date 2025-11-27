"""Watermarking interface and implementations."""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

__all__ = ['BaseWatermarker', 'PerthWatermarker', 'NoOpWatermarker', 'MetadataWatermarker']


class BaseWatermarker(ABC):
    """Abstract base class for audio watermarkers."""
    
    @abstractmethod
    def apply_watermark(self, audio: np.ndarray, sample_rate: int, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Apply watermark to audio.
        
        :param audio: Audio samples as numpy array (mono or multi-channel)
        :param sample_rate: Sample rate of the audio
        :param metadata: Optional metadata to embed in watermark (e.g., user_id, story_id, timestamp)
        :return: Watermarked audio as numpy array
        """
        pass
    
    @abstractmethod
    def get_watermark(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Extract watermark from audio (if supported).
        
        :param audio: Audio samples as numpy array
        :param sample_rate: Sample rate of the audio
        :return: Watermark value (e.g., 0.0 for no watermark, 1.0 for watermarked)
        """
        pass
    
    def extract_metadata(self, audio: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        """
        Extract metadata from watermarked audio (if supported).
        
        :param audio: Audio samples as numpy array
        :param sample_rate: Sample rate of the audio
        :return: Extracted metadata dictionary, or None if not supported
        """
        return None


class PerthWatermarker(BaseWatermarker):
    """Wrapper for Resemble AI's Perth watermarker."""
    
    def __init__(self):
        try:
            import perth
            self._watermarker = perth.PerthImplicitWatermarker()
        except ImportError:
            raise ImportError("perth library not installed. Install with: pip install resemble-perth")
    
    def apply_watermark(self, audio: np.ndarray, sample_rate: int, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Apply Perth watermark to audio.
        
        Note: Perth watermarker doesn't support custom metadata embedding.
        Metadata parameter is accepted for API compatibility but ignored.
        """
        return self._watermarker.apply_watermark(audio, sample_rate=sample_rate)
    
    def get_watermark(self, audio: np.ndarray, sample_rate: int) -> float:
        """Extract Perth watermark from audio."""
        return self._watermarker.get_watermark(audio, sample_rate=sample_rate)


class NoOpWatermarker(BaseWatermarker):
    """No-op watermarker that returns audio unchanged."""
    
    def apply_watermark(self, audio: np.ndarray, sample_rate: int, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """Return audio unchanged (no watermarking)."""
        return audio
    
    def get_watermark(self, audio: np.ndarray, sample_rate: int) -> float:
        """Always return 0.0 (no watermark detected)."""
        return 0.0


class MetadataWatermarker(BaseWatermarker):
    """
    Watermarker that embeds custom metadata into audio.
    
    Uses spread spectrum technique to embed metadata as imperceptible frequency components.
    Metadata is encoded as a binary sequence and embedded across multiple frequency bands.
    """
    
    def __init__(self, strength: float = 0.0005, base_freq: float = 18000):
        """
        Initialize metadata watermarker.
        
        :param strength: Watermark strength (0.0 to 1.0). Lower = more imperceptible
        :param base_freq: Base frequency for embedding (Hz). Should be above human hearing (~18kHz)
        """
        self.strength = max(0.0, min(1.0, strength))
        self.base_freq = base_freq
        self._metadata_cache = {}  # Cache for extraction
    
    def _encode_metadata(self, metadata: Dict[str, Any]) -> np.ndarray:
        """Encode metadata dictionary into binary sequence."""
        import json
        import hashlib
        
        # Convert metadata to JSON string
        metadata_str = json.dumps(metadata, sort_keys=True)
        
        # Create hash for consistency
        metadata_hash = hashlib.md5(metadata_str.encode()).hexdigest()
        
        # Convert to binary sequence (use first 64 bits of hash)
        binary_str = bin(int(metadata_hash[:16], 16))[2:].zfill(64)
        binary_array = np.array([int(b) for b in binary_str], dtype=np.float32)
        
        # Store for extraction
        self._metadata_cache[metadata_hash] = metadata
        
        return binary_array, metadata_hash
    
    def _decode_metadata(self, audio: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        """Extract metadata from audio."""
        # FFT to frequency domain
        fft_audio = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
        
        # Extract bits from high-frequency components
        extracted_bits = []
        num_bits = 64
        
        for i in range(num_bits):
            # Check frequency bin corresponding to this bit
            freq_idx = int((self.base_freq + i * 100) * len(audio) / sample_rate)
            if 0 <= freq_idx < len(fft_audio):
                # Extract bit based on phase/amplitude
                phase = np.angle(fft_audio[freq_idx])
                bit = 1 if phase > 0 else 0
                extracted_bits.append(bit)
            else:
                extracted_bits.append(0)
        
        # Convert binary to hex
        binary_str = ''.join(str(b) for b in extracted_bits)
        try:
            hex_str = hex(int(binary_str, 2))[2:].zfill(16)
            # Check cache
            if hex_str in self._metadata_cache:
                return self._metadata_cache[hex_str]
        except ValueError:
            pass
        
        return None
    
    def apply_watermark(self, audio: np.ndarray, sample_rate: int, metadata: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Apply watermark with embedded metadata.
        
        :param audio: Audio samples as numpy array
        :param sample_rate: Sample rate of the audio
        :param metadata: Dictionary of metadata to embed (e.g., {'user_id': '123', 'story_id': 'abc'})
        :return: Watermarked audio
        """
        if metadata is None or len(metadata) == 0:
            # No metadata to embed, return audio unchanged
            return audio
        
        # Encode metadata to binary
        binary_data, metadata_hash = self._encode_metadata(metadata)
        
        # FFT to frequency domain
        fft_audio = np.fft.fft(audio)
        freqs = np.fft.fftfreq(len(audio), 1/sample_rate)
        
        # Embed bits in high-frequency components (above human hearing)
        for i, bit in enumerate(binary_data):
            freq = self.base_freq + i * 100  # Spread across frequencies
            freq_idx = int(freq * len(audio) / sample_rate)
            
            if 0 <= freq_idx < len(fft_audio):
                # Modify phase/amplitude based on bit value
                current_phase = np.angle(fft_audio[freq_idx])
                current_amp = np.abs(fft_audio[freq_idx])
                
                # Embed bit in phase
                if bit > 0.5:
                    new_phase = current_phase + np.pi * self.strength
                else:
                    new_phase = current_phase - np.pi * self.strength
                
                # Update FFT coefficient
                fft_audio[freq_idx] = current_amp * np.exp(1j * new_phase)
        
        # Convert back to time domain
        watermarked = np.real(np.fft.ifft(fft_audio))
        
        # Ensure no clipping
        watermarked = np.clip(watermarked, -1.0, 1.0)
        
        return watermarked
    
    def get_watermark(self, audio: np.ndarray, sample_rate: int) -> float:
        """Check if watermark is present."""
        metadata = self.extract_metadata(audio, sample_rate)
        return 1.0 if metadata is not None else 0.0
    
    def extract_metadata(self, audio: np.ndarray, sample_rate: int) -> Optional[Dict[str, Any]]:
        """Extract embedded metadata from audio."""
        return self._decode_metadata(audio, sample_rate)

