# Watermarking in ChatterboxTTS

## Current Watermark

ChatterboxTTS uses **Perth (Perceptual Threshold) Watermarker** from Resemble AI by default. This is an imperceptible neural watermark that:

- ✅ Survives MP3 compression
- ✅ Survives audio editing and common manipulations  
- ✅ Maintains nearly 100% detection accuracy
- ✅ Is imperceptible to human listeners

The watermark is a **binary watermark** - it either detects presence (1.0) or absence (0.0) of the watermark.

## How It Works

The watermark is applied automatically to all generated audio:

1. **During TTS generation**: After audio chunks are stitched together
2. **During voice cloning**: After voice conversion
3. **Before final export**: Right before converting to MP3/WAV

## Customizing Watermarking

### Option 1: Use Default Perth Watermarker (Default)

```python
from chatterbox import ChatterboxTTS

# Default behavior - uses Perth watermarker
tts = ChatterboxTTS.from_pretrained(device='cuda')
```

### Option 2: Disable Watermarking

```python
from chatterbox import ChatterboxTTS
from chatterbox.watermarking import NoOpWatermarker

# Disable watermarking entirely
no_watermark = NoOpWatermarker()
tts = ChatterboxTTS.from_pretrained(
    device='cuda',
    watermarker=no_watermark
)
```

### Option 3: Use Custom Watermarker

Create a custom watermarker by inheriting from `BaseWatermarker`:

```python
from chatterbox.watermarking import BaseWatermarker
import numpy as np

class MyCustomWatermarker(BaseWatermarker):
    """Your custom watermarker implementation."""
    
    def apply_watermark(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply your watermark to the audio.
        
        :param audio: Audio samples as numpy array (mono or multi-channel)
        :param sample_rate: Sample rate of the audio
        :return: Watermarked audio as numpy array
        """
        # Your watermarking logic here
        # Example: add imperceptible signal
        watermark_signal = np.random.randn(len(audio)) * 0.0001
        return audio + watermark_signal
    
    def get_watermark(self, audio: np.ndarray, sample_rate: int) -> float:
        """
        Extract watermark from audio (if supported).
        
        :param audio: Audio samples as numpy array
        :param sample_rate: Sample rate of the audio
        :return: Watermark value (0.0 to 1.0 confidence)
        """
        # Your detection logic here
        return 1.0  # Watermark detected

# Use your custom watermarker
custom_watermarker = MyCustomWatermarker()
tts = ChatterboxTTS.from_pretrained(
    device='cuda',
    watermarker=custom_watermarker
)
```

## Watermarker Interface

All watermarkers must implement the `BaseWatermarker` interface:

```python
class BaseWatermarker(ABC):
    @abstractmethod
    def apply_watermark(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply watermark to audio."""
        pass
    
    @abstractmethod
    def get_watermark(self, audio: np.ndarray, sample_rate: int) -> float:
        """Extract watermark from audio."""
        pass
```

## Available Watermarkers

- **`PerthWatermarker`**: Default Perth watermarker (requires `resemble-perth` package)
- **`NoOpWatermarker`**: No-op watermarker that returns audio unchanged
- **`BaseWatermarker`**: Abstract base class for custom implementations

## Examples

See `examples/custom_watermarker.py` for complete examples of:
- Simple amplitude modulation watermarker
- Metadata embedding watermarker
- How to use custom watermarkers

## Watermark Detection

To detect watermarks in generated audio:

```python
import librosa
from chatterbox.watermarking import PerthWatermarker

# Load watermarked audio
audio, sr = librosa.load("watermarked_audio.wav", sr=None)

# Initialize watermarker
watermarker = PerthWatermarker()

# Extract watermark
watermark_value = watermarker.get_watermark(audio, sample_rate=sr)
print(f"Watermark detected: {watermark_value}")  # 0.0 or 1.0
```

## Embedding Custom Metadata

To embed custom data (like user_id, story_id, timestamps) in watermarks:

### Option 1: Use MetadataWatermarker

```python
from chatterbox import ChatterboxTTS
from chatterbox.watermarking import MetadataWatermarker

# Initialize with metadata watermarker
metadata_watermarker = MetadataWatermarker(strength=0.0005)
tts = ChatterboxTTS.from_pretrained(
    device='cuda',
    watermarker=metadata_watermarker
)

# Generate audio with embedded metadata
audio, sr, _ = tts.generate_long_text(
    text="Your story text",
    voice_profile_path="voice.npy",
    output_path="output.wav",
    watermark_metadata={
        "user_id": "user_123",
        "story_id": "story_456",
        "voice_id": "voice_789",
        "timestamp": "2024-01-01T00:00:00Z",
        "custom_field": "custom_value"
    }
)
```

### Option 2: Automatic Metadata in generate_tts_story

When using `generate_tts_story`, metadata is automatically built and embedded:

```python
result = tts.generate_tts_story(
    text="Your story",
    voice_id="voice_123",
    profile_base64=profile_data,
    user_id="user_456",      # Automatically embedded
    story_id="story_789",     # Automatically embedded
    language="en",            # Automatically embedded
    story_type="user",        # Automatically embedded
    voice_name="My Voice"     # Automatically embedded
)
```

### Extracting Metadata

```python
from chatterbox.watermarking import MetadataWatermarker
import librosa

# Initialize the same watermarker (must match embedding watermarker)
watermarker = MetadataWatermarker(strength=0.0005)

# Load watermarked audio
audio, sr = librosa.load("watermarked_audio.wav", sr=None)

# Extract metadata
metadata = watermarker.extract_metadata(audio, sr)

if metadata:
    print(f"User ID: {metadata.get('user_id')}")
    print(f"Story ID: {metadata.get('story_id')}")
    print(f"Voice ID: {metadata.get('voice_id')}")
else:
    print("No metadata found")
```

## Notes

- Watermarking happens **after** DC offset removal to prevent artifacts
- The watermark is applied to the final stitched audio, not individual chunks
- Custom watermarkers should ensure they don't introduce audible artifacts
- **MetadataWatermarker** uses spread spectrum technique to embed data imperceptibly
- Metadata is embedded in high-frequency components (above human hearing range)
- For production use, consider robust watermarking techniques (spread spectrum, perceptual hashing, etc.)
- **Perth watermarker** does NOT support custom metadata - use MetadataWatermarker for that

