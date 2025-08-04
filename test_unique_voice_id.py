#!/usr/bin/env python3
"""
Test script to demonstrate unique voice ID generation with alphanumeric characters and Firebase checking.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatterbox.vc import generate_unique_voice_id

def test_unique_voice_id_generation():
    """Test the unique voice ID generation function."""
    
    print("🧪 Testing unique voice ID generation with alphanumeric characters...")
    
    # Test 1: Basic generation
    print("\n📋 Test 1: Basic generation")
    voice_id_1 = generate_unique_voice_id("christestclone")
    print(f"  - Generated: {voice_id_1}")
    
    # Test 2: Multiple generations (should be different)
    print("\n📋 Test 2: Multiple generations")
    voice_ids = []
    for i in range(5):
        voice_id = generate_unique_voice_id("christestclone")
        voice_ids.append(voice_id)
        print(f"  - Generation {i+1}: {voice_id}")
    
    # Test 3: Check uniqueness
    print("\n📋 Test 3: Uniqueness check")
    unique_ids = set(voice_ids)
    if len(unique_ids) == len(voice_ids):
        print("  ✅ All generated IDs are unique!")
    else:
        print("  ❌ Some IDs are not unique!")
    
    # Test 4: Different voice names
    print("\n📋 Test 4: Different voice names")
    voice_id_alice = generate_unique_voice_id("alice")
    voice_id_bob = generate_unique_voice_id("bob")
    print(f"  - Alice: {voice_id_alice}")
    print(f"  - Bob: {voice_id_bob}")
    
    # Test 5: Custom length
    print("\n📋 Test 5: Custom length")
    voice_id_custom = generate_unique_voice_id("custom", length=12)
    print(f"  - Custom length (12): {voice_id_custom}")
    
    # Test 6: Show pattern analysis
    print("\n📋 Test 6: Pattern analysis")
    print("  - Pattern: voice_{name}_{alphanumeric}")
    print("  - Alphanumeric characters: A-Z, a-z, 0-9")
    print("  - Default length: 8 characters")
    print("  - Total combinations: 62^8 = ~218 trillion")
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    test_unique_voice_id_generation() 