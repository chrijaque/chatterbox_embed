#!/usr/bin/env python3
"""
Simple test script to verify voice profile functionality.
This script tests the basic functionality without requiring actual audio files.
"""

import torch
import numpy as np
from chatterbox.models.s3gen import VoiceProfile

def test_voice_profile_basic():
    """Test basic VoiceProfile functionality"""
    print("Testing VoiceProfile basic functionality...")
    
    # Create dummy data
    embedding = torch.randn(1, 80)  # CAMPPlus embedding
    prompt_feat = torch.randn(1, 80, 100)  # Mel features
    prompt_token = torch.randint(0, 1000, (1, 50))  # Speech tokens
    prompt_token_len = torch.tensor([50])
    
    # Create voice profile
    profile = VoiceProfile(
        embedding=embedding,
        prompt_feat=prompt_feat,
        prompt_feat_len=100,
        prompt_token=prompt_token,
        prompt_token_len=prompt_token_len,
    )
    
    print(f"✅ VoiceProfile created successfully")
    print(f"   - Embedding shape: {profile.embedding.shape}")
    print(f"   - Prompt features shape: {profile.prompt_feat.shape}")
    print(f"   - Prompt tokens shape: {profile.prompt_token.shape}")
    print(f"   - Prompt token length: {profile.prompt_token_len}")
    
    # Test save and load
    test_path = "test_voice_profile.npy"
    try:
        profile.save(test_path)
        print(f"✅ VoiceProfile saved to {test_path}")
        
        loaded_profile = VoiceProfile.load(test_path, device="cpu")
        print(f"✅ VoiceProfile loaded from {test_path}")
        
        # Verify data integrity
        assert torch.allclose(profile.embedding, loaded_profile.embedding)
        assert torch.allclose(profile.prompt_feat, loaded_profile.prompt_feat)
        assert torch.allclose(profile.prompt_token, loaded_profile.prompt_token)
        assert torch.allclose(profile.prompt_token_len, loaded_profile.prompt_token_len)
        print("✅ Data integrity verified")
        
        # Clean up
        import os
        os.remove(test_path)
        print(f"✅ Test file cleaned up")
        
    except Exception as e:
        print(f"❌ Error in save/load test: {e}")
        return False
    
    return True

def test_voice_profile_partial():
    """Test VoiceProfile with partial data"""
    print("\nTesting VoiceProfile with partial data...")
    
    # Create profile with only embedding
    embedding = torch.randn(1, 80)
    profile = VoiceProfile(embedding=embedding)
    
    print(f"✅ VoiceProfile with partial data created")
    print(f"   - Embedding: {profile.embedding is not None}")
    print(f"   - Prompt features: {profile.prompt_feat is None}")
    print(f"   - Prompt tokens: {profile.prompt_token is None}")
    
    # Test save and load
    test_path = "test_partial_profile.npy"
    try:
        profile.save(test_path)
        loaded_profile = VoiceProfile.load(test_path, device="cpu")
        
        assert torch.allclose(profile.embedding, loaded_profile.embedding)
        assert loaded_profile.prompt_feat is None
        assert loaded_profile.prompt_token is None
        print("✅ Partial data integrity verified")
        
        # Clean up
        import os
        os.remove(test_path)
        
    except Exception as e:
        print(f"❌ Error in partial data test: {e}")
        return False
    
    return True

def main():
    print("=== Voice Profile Test Suite ===")
    
    success = True
    success &= test_voice_profile_basic()
    success &= test_voice_profile_partial()
    
    if success:
        print("\n🎉 All tests passed! Voice profile functionality is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
    
    return success

if __name__ == "__main__":
    main() 