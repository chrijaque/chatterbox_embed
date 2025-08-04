#!/usr/bin/env python3
"""
Complete Voice Cloning Pipeline Test
====================================

This script tests the complete voice cloning pipeline to ensure everything works
before deploying to RunPod. It simulates what the API app would do.

Usage:
    python test_vc_pipeline.py

Requirements:
    - A test audio file (WAV format)
    - All model dependencies installed
    - Firebase credentials (optional, for upload testing)
"""

import os
import sys
import logging
import tempfile
import base64
from pathlib import Path
from typing import Dict, Any

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_model_initialization():
    """Test 1: Model Initialization"""
    logger.info("🧪 ===== TEST 1: Model Initialization =====")
    
    try:
        from chatterbox.vc import ChatterboxVC
        from chatterbox.tts import ChatterboxTTS
        
        # Test TTS model initialization
        logger.info("📦 Initializing ChatterboxTTS...")
        tts_model = ChatterboxTTS.from_pretrained(device='cpu')
        logger.info("✅ ChatterboxTTS initialized successfully")
        
        # Test VC model initialization
        logger.info("📦 Initializing ChatterboxVC...")
        vc_model = ChatterboxVC.from_pretrained(device='cpu')
        logger.info("✅ ChatterboxVC initialized successfully")
        
        # Check if text encoder is properly attached
        if hasattr(vc_model.s3gen, 'text_encoder'):
            logger.info("✅ Text encoder is attached to s3gen")
            logger.info(f"   - Text encoder type: {type(vc_model.s3gen.text_encoder)}")
        else:
            logger.error("❌ Text encoder is NOT attached to s3gen")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_voice_clone_creation(audio_file_path: str):
    """Test 2: Voice Clone Creation"""
    logger.info("🧪 ===== TEST 2: Voice Clone Creation =====")
    
    try:
        from chatterbox.vc import ChatterboxVC
        
        # Initialize VC model
        vc_model = ChatterboxVC.from_pretrained(device='cpu')
        
        # Test parameters (similar to what API app would send)
        voice_id = "test_voice_clone"
        voice_name = "Test Voice Clone"
        metadata = {
            "language": "en",
            "is_kids_voice": False,
            "template_message": "Hello, this is a test voice clone."
        }
        
        logger.info(f"🎤 Creating voice clone...")
        logger.info(f"   - Audio file: {audio_file_path}")
        logger.info(f"   - Voice ID: {voice_id}")
        logger.info(f"   - Voice name: {voice_name}")
        logger.info(f"   - Metadata: {metadata}")
        
        # Call the voice cloning method
        result = vc_model.create_voice_clone(
            audio_file_path=audio_file_path,
            voice_id=voice_id,
            voice_name=voice_name,
            metadata=metadata
        )
        
        # Check the result
        logger.info("📊 Voice clone result:")
        logger.info(f"   - Status: {result.get('status')}")
        logger.info(f"   - Voice ID: {result.get('voice_id')}")
        logger.info(f"   - Profile path: {result.get('profile_path')}")
        logger.info(f"   - Sample audio size: {result.get('sample_audio_size', 0):,} bytes")
        logger.info(f"   - Recorded audio size: {result.get('recorded_audio_size', 0):,} bytes")
        logger.info(f"   - Audio tensor shape: {result.get('audio_tensor_shape')}")
        logger.info(f"   - Sample rate: {result.get('sample_rate')}")
        
        if result.get('status') == 'success':
            logger.info("✅ Voice clone creation successful!")
            
            # Save the generated files for inspection
            output_dir = Path("test_output")
            output_dir.mkdir(exist_ok=True)
            
            # Save sample audio
            sample_path = output_dir / f"{voice_id}_sample.mp3"
            with open(sample_path, 'wb') as f:
                f.write(result['sample_audio_bytes'])
            logger.info(f"💾 Sample audio saved to: {sample_path}")
            
            # Save recorded audio
            recorded_path = output_dir / f"{voice_id}_recorded.mp3"
            with open(recorded_path, 'wb') as f:
                f.write(result['recorded_audio_bytes'])
            logger.info(f"💾 Recorded audio saved to: {recorded_path}")
            
            # Save voice profile
            profile_path = output_dir / f"{voice_id}.npy"
            import numpy as np
            np.save(profile_path, np.load(result['profile_path']))
            logger.info(f"💾 Voice profile saved to: {profile_path}")
            
            return True
        else:
            logger.error(f"❌ Voice clone creation failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Voice clone creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tts_generation(voice_profile_path: str):
    """Test 3: TTS Generation with Voice Profile"""
    logger.info("🧪 ===== TEST 3: TTS Generation =====")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        
        # Initialize TTS model
        tts_model = ChatterboxTTS.from_pretrained(device='cpu')
        
        # Test text
        test_text = "This is a test of text-to-speech generation using the created voice profile."
        
        logger.info(f"📝 Generating TTS...")
        logger.info(f"   - Text: {test_text}")
        logger.info(f"   - Voice profile: {voice_profile_path}")
        
        # Call the TTS generation method
        result = tts_model.generate_tts_story(
            text=test_text,
            voice_profile_path=voice_profile_path,
            voice_id="test_voice_clone",
            language="en",
            story_type="user",
            is_kids_voice=False,
            metadata={"test": True}
        )
        
        # Check the result
        logger.info("📊 TTS generation result:")
        logger.info(f"   - Status: {result.get('status')}")
        logger.info(f"   - Audio size: {result.get('audio_size', 0):,} bytes")
        logger.info(f"   - Duration: {result.get('duration', 0):.2f} seconds")
        logger.info(f"   - Firebase URL: {result.get('firebase_url', 'N/A')}")
        
        if result.get('status') == 'success':
            logger.info("✅ TTS generation successful!")
            
            # Save the generated audio
            output_dir = Path("test_output")
            output_dir.mkdir(exist_ok=True)
            
            tts_path = output_dir / "tts_generated.mp3"
            with open(tts_path, 'wb') as f:
                f.write(result['audio_bytes'])
            logger.info(f"💾 TTS audio saved to: {tts_path}")
            
            return True
        else:
            logger.error(f"❌ TTS generation failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"❌ TTS generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoint_simulation(audio_file_path: str):
    """Test 4: Simulate Complete API Endpoint"""
    logger.info("🧪 ===== TEST 4: API Endpoint Simulation =====")
    
    try:
        from chatterbox.vc import ChatterboxVC
        from chatterbox.tts import ChatterboxTTS
        
        # Simulate API request data
        api_request = {
            "name": "Test Voice Clone",
            "language": "en",
            "is_kids_voice": False,
            "audio_data": "base64_encoded_audio_data",  # In real API, this would be base64
            "audio_format": "wav"
        }
        
        logger.info("🔄 Simulating API endpoint...")
        logger.info(f"   - Request data: {api_request}")
        
        # Step 1: Initialize models (what API handler would do)
        logger.info("📦 Step 1: Initializing models...")
        tts_model = ChatterboxTTS.from_pretrained(device='cpu')
        vc_model = ChatterboxVC.from_pretrained(device='cpu')
        logger.info("✅ Models initialized")
        
        # Step 2: Create voice clone (what API handler would call)
        logger.info("🎤 Step 2: Creating voice clone...")
        voice_id = "api_test_voice"
        voice_name = api_request["name"]
        
        vc_result = vc_model.create_voice_clone(
            audio_file_path=audio_file_path,
            voice_id=voice_id,
            voice_name=voice_name,
            metadata={
                "language": api_request["language"],
                "is_kids_voice": api_request["is_kids_voice"]
            }
        )
        
        if vc_result.get('status') != 'success':
            logger.error(f"❌ Voice clone failed: {vc_result.get('error')}")
            return False
            
        logger.info("✅ Voice clone created successfully")
        
        # Step 3: Generate TTS sample (optional, what API might do)
        logger.info("📝 Step 3: Generating TTS sample...")
        tts_result = tts_model.generate_tts_story(
            text="Hello, this is a test of the voice cloning system.",
            voice_profile_path=vc_result['profile_path'],
            voice_id=voice_id,
            language=api_request["language"],
            story_type="user",
            is_kids_voice=api_request["is_kids_voice"]
        )
        
        if tts_result.get('status') == 'success':
            logger.info("✅ TTS sample generated successfully")
        else:
            logger.warning(f"⚠️ TTS sample generation failed: {tts_result.get('error')}")
        
        # Step 4: Simulate API response
        api_response = {
            "status": "success",
            "voice_id": voice_id,
            "voice_name": voice_name,
            "profile_path": vc_result['profile_path'],
            "sample_audio_size": vc_result['sample_audio_size'],
            "recorded_audio_size": vc_result['recorded_audio_size'],
            "tts_status": tts_result.get('status', 'not_attempted')
        }
        
        logger.info("📤 API Response:")
        for key, value in api_response.items():
            logger.info(f"   - {key}: {value}")
        
        logger.info("✅ API endpoint simulation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ API endpoint simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_test_audio():
    """Create a test audio file if none exists"""
    test_audio_path = Path("test_audio.wav")
    
    if test_audio_path.exists():
        logger.info(f"✅ Test audio file exists: {test_audio_path}")
        return str(test_audio_path)
    
    logger.info("🎵 Creating test audio file...")
    
    try:
        import torch
        import torchaudio
        
        # Create a simple sine wave
        sample_rate = 44100
        duration = 3.0  # 3 seconds
        frequency = 440  # A4 note
        
        t = torch.linspace(0, duration, int(sample_rate * duration))
        audio = torch.sin(2 * torch.pi * frequency * t)
        
        # Add some variation to make it more realistic
        audio = audio * 0.3  # Reduce volume
        audio = audio + torch.randn_like(audio) * 0.01  # Add noise
        
        # Save as WAV
        torchaudio.save(test_audio_path, audio.unsqueeze(0), sample_rate)
        
        logger.info(f"✅ Test audio created: {test_audio_path}")
        return str(test_audio_path)
        
    except Exception as e:
        logger.error(f"❌ Failed to create test audio: {e}")
        return None

def main():
    """Run all tests"""
    logger.info("🚀 Starting Voice Cloning Pipeline Tests")
    logger.info("=" * 50)
    
    # Create test audio if needed
    audio_file_path = create_test_audio()
    if not audio_file_path:
        logger.error("❌ Cannot proceed without test audio file")
        return False
    
    # Run all tests
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Voice Clone Creation", lambda: test_voice_clone_creation(audio_file_path)),
        ("API Endpoint Simulation", lambda: test_api_endpoint_simulation(audio_file_path)),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("📊 TEST RESULTS SUMMARY")
    logger.info("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! The voice cloning pipeline is working correctly.")
        logger.info("💡 You can now deploy to RunPod with confidence.")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix the issues before deploying to RunPod.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 