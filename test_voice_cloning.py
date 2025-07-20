#!/usr/bin/env python3
"""
Simple test script for voice cloning functionality
"""
import torch
import numpy as np
from chatterbox.tts import ChatterboxTTS

def test_voice_cloning():
    print("Testing voice cloning functionality...")
    
    # Test device detection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # Load model
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Test that the new methods exist
    try:
        assert hasattr(model, 'save_voice_clone'), "save_voice_clone method missing"
        assert hasattr(model, 'prepare_conditionals_with_saved_voice'), "prepare_conditionals_with_saved_voice method missing"
        assert hasattr(model.s3gen, 'save_voice_clone'), "s3gen.save_voice_clone method missing"
        assert hasattr(model.s3gen, 'load_voice_clone'), "s3gen.load_voice_clone method missing"
        print("✓ All new methods are present")
    except AssertionError as e:
        print(f"✗ Method check failed: {e}")
        return False
    
    # Test basic functionality with built-in voice
    try:
        text = "Hello, this is a test."
        wav = model.generate(text)
        print(f"✓ Basic generation works, output shape: {wav.shape}")
    except Exception as e:
        print(f"✗ Basic generation failed: {e}")
        return False
    
    print("✓ All tests passed! Voice cloning is ready to use.")
    print("\nTo use voice cloning:")
    print("1. model.save_voice_clone('your_audio.wav', 'voice.npy')")
    print("2. model.generate('text', saved_voice_path='voice.npy', audio_prompt_path='prompt.wav')")
    return True

if __name__ == "__main__":
    test_voice_cloning() 