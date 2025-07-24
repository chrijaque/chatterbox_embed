#!/usr/bin/env python3
"""
Simple Voice Profile TTS Example
This shows the EXACT code your app should use.
"""

from chatterbox import ChatterboxTTS
import torch
import torchaudio

def generate_tts_with_voice_profile(text, voice_profile_path):
    """
    Generate TTS using a voice profile.
    
    Args:
        text (str): Text to convert to speech
        voice_profile_path (str): Path to the voice profile (.npy file)
    
    Returns:
        torch.Tensor: Audio tensor
    """
    
    # 1. Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = ChatterboxTTS.from_pretrained(device)
    
    # 2. Load voice profile and prepare conditionals
    # ✅ THIS IS THE KEY LINE - use this method, not load_voice_profile directly
    tts.prepare_conditionals_with_voice_profile(voice_profile_path, exaggeration=0.5)
    
    # 3. Generate TTS
    audio = tts.generate(text, temperature=0.7)
    
    return audio

def main():
    """Example usage"""
    
    # Your voice profile file
    voice_profile_path = "voice_chrisrepo1.npy"
    
    # Test text
    test_text = "Hello, this is a test of voice cloning with the voice profile method. This should sound like the original voice."
    
    print("=== Simple Voice Profile TTS Example ===")
    print(f"Voice profile: {voice_profile_path}")
    print(f"Text: {test_text}")
    print(f"Text length: {len(test_text)} characters")
    
    # Generate TTS
    print("\nGenerating TTS...")
    audio = generate_tts_with_voice_profile(test_text, voice_profile_path)
    
    # Save result
    output_path = "simple_voice_profile_example.wav"
    torchaudio.save(output_path, audio, 24000)  # 24kHz sample rate
    
    print(f"✅ Generated: {output_path}")
    print(f"Audio duration: {audio.shape[1] / 24000:.2f} seconds")
    
    return output_path

if __name__ == "__main__":
    main() 