#!/usr/bin/env python3
"""
Quick Voice Cloning Test
========================

A simple test to verify the T3TextEncoder fix works.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_t3_text_encoder():
    """Test the T3TextEncoder wrapper"""
    print("🧪 Testing T3TextEncoder wrapper...")
    
    try:
        from chatterbox.vc import ChatterboxVC
        
        # Initialize VC model
        print("📦 Initializing ChatterboxVC...")
        vc_model = ChatterboxVC.from_pretrained(device='cpu')
        print("✅ ChatterboxVC initialized")
        
        # Check if text encoder is attached
        if hasattr(vc_model.s3gen, 'text_encoder'):
            print(f"✅ Text encoder attached: {type(vc_model.s3gen.text_encoder)}")
            
            # Test the encode method
            if hasattr(vc_model.s3gen.text_encoder, 'encode'):
                print("✅ Text encoder has encode method")
                
                # Test encoding
                test_text = "Hello world"
                print(f"🎤 Testing encode with: '{test_text}'")
                
                try:
                    tokens = vc_model.s3gen.text_encoder.encode(test_text)
                    print(f"✅ Encode successful, tokens shape: {tokens.shape}")
                    return True
                except Exception as e:
                    print(f"❌ Encode failed: {e}")
                    return False
            else:
                print("❌ Text encoder missing encode method")
                return False
        else:
            print("❌ No text encoder attached")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_voice_clone():
    """Test voice cloning with a simple audio file"""
    print("\n🧪 Testing voice cloning...")
    
    try:
        from chatterbox.vc import ChatterboxVC
        import torch
        import torchaudio
        
        # Create a simple test audio file
        print("🎵 Creating test audio...")
        sample_rate = 44100
        duration = 2.0
        frequency = 440
        
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * torch.pi * frequency * t) * 0.3
        
        test_audio_path = "quick_test_audio.wav"
        torchaudio.save(test_audio_path, audio.unsqueeze(0), sample_rate)
        print(f"✅ Test audio created: {test_audio_path}")
        
        # Initialize VC model
        print("📦 Initializing ChatterboxVC...")
        vc_model = ChatterboxVC.from_pretrained(device='cpu')
        
        # Test voice cloning
        print("🎤 Creating voice clone...")
        result = vc_model.create_voice_clone(
            audio_file_path=test_audio_path,
            voice_id="quick_test_voice",
            voice_name="Quick Test Voice",
            metadata={
                "language": "en", 
                "is_kids_voice": False,
                "profile_filename": "quick_test_voice.npy",
                "sample_filename": "quick_test_voice.mp3",
                "recorded_filename": "recording_quick_test_voice.wav",
                "storage_metadata": {
                    "user_id": "test_user",
                    "voice_id": "quick_test_voice",
                    "voice_name": "Quick Test Voice",
                    "language": "en",
                    "is_kids_voice": "false",
                    "model_type": "chatterbox",
                    "file_kind": "profile"
                }
            }
        )
        
        if result.get('status') == 'success':
            print("✅ Voice clone created successfully!")
            print(f"   - Sample audio size: {result.get('sample_audio_size', 0):,} bytes")
            print(f"   - Recorded audio size: {result.get('recorded_audio_size', 0):,} bytes")
            return True
        else:
            print(f"❌ Voice clone failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Voice clone test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Quick Voice Cloning Test")
    print("=" * 40)
    
    # Test 1: T3TextEncoder
    test1_passed = test_t3_text_encoder()
    
    # Test 2: Voice Cloning
    test2_passed = test_voice_clone()
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 RESULTS:")
    print(f"T3TextEncoder Test: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    print(f"Voice Cloning Test: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS PASSED! The fix is working.")
        print("💡 You can deploy to RunPod with confidence.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please fix before deploying.")
        sys.exit(1) 