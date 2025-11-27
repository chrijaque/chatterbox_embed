"""
Example: Creating a Custom Watermarker

This example shows how to create a custom watermarker for ChatterboxTTS.
"""

import numpy as np
from chatterbox.watermarking import BaseWatermarker


class SimpleAmplitudeWatermarker(BaseWatermarker):
    """
    Simple example watermarker that adds imperceptible amplitude modulation.
    
    This is a basic example - for production use, consider more sophisticated
    watermarking techniques like spread spectrum or perceptual hashing.
    """
    
    def __init__(self, strength: float = 0.001):
        """
        Initialize the watermarker.
        
        :param strength: Watermark strength (0.0 to 1.0). Lower values are more imperceptible.
        """
        self.strength = max(0.0, min(1.0, strength))
        self.watermark_key = 12345  # Simple key for demonstration
    
    def apply_watermark(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply watermark by adding imperceptible amplitude modulation.
        
        :param audio: Audio samples as numpy array
        :param sample_rate: Sample rate of the audio
        :return: Watermarked audio
        """
        # Create a watermark signal based on audio characteristics
        np.random.seed(self.watermark_key)
        watermark_signal = np.random.randn(len(audio)) * self.strength
        
        # Apply watermark (very subtle)
        watermarked = audio + watermark_signal * np.abs(audio) * 0.01
        
        # Ensure no clipping
        watermarked = np.clip(watermarked, -1.0, 1.0)
        
        return watermarked
    
    def get_watermark(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Attempt to detect watermark (simplified - not robust).
        
        :param audio: Audio samples as numpy array
        :param sample_rate: Sample rate of the audio
        :return: Confidence score (0.0 to 1.0)
        """
        # This is a simplified detection - real watermarking would be more robust
        # For demonstration, we'll return a low confidence
        return 0.3  # Low confidence detection


class MetadataWatermarker(BaseWatermarker):
    """
    Example watermarker that embeds metadata as inaudible tones.
    
    This embeds a simple ID as frequency-shifted tones that are
    outside the normal hearing range or masked by the audio.
    """
    
    def __init__(self, metadata_id: str = "default"):
        """
        Initialize with metadata to embed.
        
        :param metadata_id: String identifier to embed
        """
        self.metadata_id = metadata_id
        # Convert string to numeric key
        self.key = hash(metadata_id) % 10000
    
    def apply_watermark(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Embed metadata as inaudible frequency components.
        
        :param audio: Audio samples as numpy array
        :param sample_rate: Sample rate of the audio
        :return: Watermarked audio
        """
        # Generate watermark signal at frequencies above 18kHz (inaudible to most)
        duration = len(audio) / sample_rate
        t = np.linspace(0, duration, len(audio))
        
        # Embed metadata as phase modulation at high frequency
        watermark_freq = 19000  # Above human hearing range
        watermark_signal = np.sin(2 * np.pi * watermark_freq * t + self.key * 0.01)
        
        # Add very subtle watermark
        watermarked = audio + watermark_signal * 0.0001
        
        # Ensure no clipping
        watermarked = np.clip(watermarked, -1.0, 1.0)
        
        return watermarked
    
    def get_watermark(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Extract metadata from audio.
        
        :param audio: Audio samples as numpy array
        :param sample_rate: Sample rate of the audio
        :return: Confidence score
        """
        # Simplified detection - check for high-frequency components
        # Real implementation would use FFT and pattern matching
        return 0.5


# Example usage:
if __name__ == "__main__":
    from chatterbox import ChatterboxTTS
    
    # Option 1: Use custom watermarker
    custom_watermarker = SimpleAmplitudeWatermarker(strength=0.001)
    
    # Initialize TTS with custom watermarker
    # tts = ChatterboxTTS.from_pretrained(
    #     device='cuda',
    #     watermarker=custom_watermarker
    # )
    
    # Option 2: Disable watermarking entirely
    from chatterbox.watermarking import NoOpWatermarker
    
    # no_watermark = NoOpWatermarker()
    # tts = ChatterboxTTS.from_pretrained(
    #     device='cuda',
    #     watermarker=no_watermark
    # )
    
    # Option 3: Use default Perth watermarker (default behavior)
    # tts = ChatterboxTTS.from_pretrained(device='cuda')
    # # watermarker is automatically PerthWatermarker()

