#!/usr/bin/env python3
"""
Test script to verify the response structure from create_voice_clone method.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatterbox.vc import ChatterboxVC
import tempfile
import json

def test_response_structure():
    """Test the response structure from create_voice_clone."""
    
    print("🧪 Testing create_voice_clone response structure...")
    
    try:
        # Initialize VC model (CPU for testing)
        print("  - Initializing ChatterboxVC...")
        vc_model = ChatterboxVC.from_pretrained(device='cpu')
        print("  ✅ VC model initialized")
        
        # Create a dummy audio file for testing
        print("  - Creating dummy audio file...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            # Create a simple WAV file (1 second of silence)
            import wave
            import struct
            
            # WAV header for 1 second of silence at 22050 Hz
            sample_rate = 22050
            duration = 1
            num_samples = sample_rate * duration
            
            with wave.open(temp_file.name, 'w') as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                
                # Write silence
                for _ in range(num_samples):
                    wav_file.writeframes(struct.pack('<h', 0))
            
            temp_audio_path = temp_file.name
        
        print(f"  ✅ Dummy audio created: {temp_audio_path}")
        
        # Test metadata
        test_metadata = {
            'user_id': 'test_user_123',
            'project_id': 'test_project_456',
            'voice_type': 'regular',
            'quality': 'high',
            'language': 'en',
            'is_kids_voice': False
        }
        
        print("  - Testing create_voice_clone with metadata...")
        print(f"    - Metadata: {test_metadata}")
        
        # Call create_voice_clone
        result = vc_model.create_voice_clone(
            audio_file_path=temp_audio_path,
            voice_id="test_voice_789",
            voice_name="Test Voice",
            metadata=test_metadata
        )
        
        print("  ✅ create_voice_clone completed")
        
        # Analyze the response
        print("\n📊 RESPONSE ANALYSIS:")
        print(f"  - Result type: {type(result)}")
        print(f"  - Result keys: {list(result.keys())}")
        print(f"  - Number of keys: {len(result.keys())}")
        
        print("\n🔍 DETAILED RESPONSE:")
        for key, value in result.items():
            print(f"  - {key}: {type(value)} = {value}")
        
        # Test JSON serialization
        print("\n🧪 JSON SERIALIZATION TEST:")
        try:
            json_str = json.dumps(result, indent=2)
            print("  ✅ Response is JSON serializable")
            print(f"  - JSON length: {len(json_str)} characters")
        except Exception as e:
            print(f"  ❌ JSON serialization failed: {e}")
        
        # Clean up
        os.unlink(temp_audio_path)
        print(f"\n🧹 Cleaned up: {temp_audio_path}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    test_response_structure() 