"""
Example: Embedding Metadata in Watermarks

This example shows how to embed custom metadata (user_id, story_id, etc.)
into audio watermarks and extract it later.
"""

import numpy as np
from chatterbox import ChatterboxTTS
from chatterbox.watermarking import MetadataWatermarker


def example_embed_metadata():
    """Example of embedding metadata in watermarks."""
    
    # Initialize TTS with metadata-capable watermarker
    metadata_watermarker = MetadataWatermarker(strength=0.0005)
    
    # Note: You would initialize TTS like this:
    # tts = ChatterboxTTS.from_pretrained(
    #     device='cuda',
    #     watermarker=metadata_watermarker
    # )
    
    # When generating audio, metadata is automatically embedded:
    # audio, sr, metadata = tts.generate_long_text(
    #     text="Your story text here",
    #     voice_profile_path="voice.npy",
    #     output_path="output.wav",
    #     watermark_metadata={
    #         "user_id": "user_123",
    #         "story_id": "story_456",
    #         "voice_id": "voice_789",
    #         "timestamp": "2024-01-01T00:00:00Z",
    #         "language": "en"
    #     }
    # )
    
    print("✅ Metadata watermarker initialized")
    print("   - Metadata will be embedded in audio during generation")
    print("   - Can extract metadata later using extract_metadata()")


def example_extract_metadata():
    """Example of extracting metadata from watermarked audio."""
    import librosa
    
    # Initialize the same watermarker (must match the one used for embedding)
    metadata_watermarker = MetadataWatermarker(strength=0.0005)
    
    # Load watermarked audio
    # audio, sr = librosa.load("watermarked_audio.wav", sr=None)
    
    # Extract metadata
    # extracted_metadata = metadata_watermarker.extract_metadata(audio, sr)
    
    # if extracted_metadata:
    #     print(f"✅ Extracted metadata: {extracted_metadata}")
    #     print(f"   - User ID: {extracted_metadata.get('user_id')}")
    #     print(f"   - Story ID: {extracted_metadata.get('story_id')}")
    #     print(f"   - Voice ID: {extracted_metadata.get('voice_id')}")
    # else:
    #     print("❌ No metadata found in audio")
    
    print("✅ Metadata extraction example ready")


def example_automatic_metadata():
    """Example showing automatic metadata embedding in generate_tts_story."""
    
    # When using generate_tts_story, metadata is automatically built from parameters:
    # result = tts.generate_tts_story(
    #     text="Your story",
    #     voice_id="voice_123",
    #     profile_base64=profile_data,
    #     user_id="user_456",
    #     story_id="story_789",
    #     language="en",
    #     story_type="user",
    #     voice_name="My Voice"
    # )
    
    # The watermark will automatically contain:
    # - user_id
    # - story_id
    # - voice_id
    # - voice_name
    # - language
    # - story_type
    # - Any additional metadata from the metadata parameter
    
    print("✅ Automatic metadata embedding in generate_tts_story")
    print("   - Metadata is built from function parameters")
    print("   - Passed to watermarker automatically")


def example_custom_metadata_fields():
    """Example of adding custom metadata fields."""
    
    # You can add custom fields to watermark metadata:
    custom_metadata = {
        "user_id": "user_123",
        "story_id": "story_456",
        "custom_field_1": "custom_value_1",
        "custom_field_2": 42,
        "custom_field_3": True,
        "timestamp": "2024-01-01T00:00:00Z",
        "generation_id": "gen_abc123"
    }
    
    # When generating:
    # audio, sr, _ = tts.generate_long_text(
    #     text="Your text",
    #     voice_profile_path="voice.npy",
    #     output_path="output.wav",
    #     watermark_metadata=custom_metadata
    # )
    
    print("✅ Custom metadata fields can be embedded")
    print("   - Supports strings, integers, floats, booleans")
    print("   - All metadata is embedded in the watermark")


if __name__ == "__main__":
    print("=" * 60)
    print("Watermark Metadata Examples")
    print("=" * 60)
    print()
    
    example_embed_metadata()
    print()
    
    example_extract_metadata()
    print()
    
    example_automatic_metadata()
    print()
    
    example_custom_metadata_fields()
    print()
    
    print("=" * 60)
    print("Note: MetadataWatermarker uses spread spectrum technique")
    print("      to embed data imperceptibly in high-frequency components.")
    print("=" * 60)


