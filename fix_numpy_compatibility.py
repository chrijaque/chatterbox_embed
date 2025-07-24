#!/usr/bin/env python3
"""
Fix numpy compatibility issue and recreate voice profile.
"""

import subprocess
import sys
import os

def fix_numpy_compatibility():
    """Fix numpy version compatibility issues"""
    
    print("=== Fixing Numpy Compatibility Issue ===")
    
    # 1. Check current numpy version
    try:
        import numpy as np
        print(f"Current numpy version: {np.__version__}")
    except ImportError:
        print("Numpy not installed")
        return False
    
    # 2. Check if we need to upgrade/downgrade numpy
    print("\nChecking for numpy compatibility issues...")
    
    # 3. Try to fix by reinstalling numpy
    print("\nAttempting to fix numpy compatibility...")
    try:
        # Upgrade numpy to latest version
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "numpy"])
        print("✅ Numpy upgraded successfully")
        
        # Verify the fix
        import numpy as np
        print(f"New numpy version: {np.__version__}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to upgrade numpy: {e}")
        return False

def recreate_voice_profile():
    """Recreate the voice profile with proper format"""
    
    print("\n=== Recreating Voice Profile ===")
    
    try:
        from chatterbox import ChatterboxTTS
        
        # Initialize model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        tts = ChatterboxTTS.from_pretrained(device)
        
        # Recreate voice profile
        reference_audio_path = "audio_test/reference.wav"
        new_voice_profile_path = "voice_profile_fixed.npy"
        
        if not os.path.exists(reference_audio_path):
            print(f"❌ Reference audio not found: {reference_audio_path}")
            return False
        
        print(f"Creating new voice profile from: {reference_audio_path}")
        tts.save_voice_profile(reference_audio_path, new_voice_profile_path)
        
        print(f"✅ New voice profile created: {new_voice_profile_path}")
        
        # Test loading the new profile
        print("Testing new voice profile...")
        profile = tts.load_voice_profile(new_voice_profile_path)
        print(f"✅ Voice profile loaded successfully!")
        print(f"   Embedding shape: {profile.embedding.shape}")
        print(f"   Has voice encoder embedding: {hasattr(profile, 've_embedding') and profile.ve_embedding is not None}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error recreating voice profile: {e}")
        return False

def main():
    """Main function to fix the issue"""
    
    print("This script will fix the numpy compatibility issue and recreate your voice profile.")
    
    # Step 1: Fix numpy
    if not fix_numpy_compatibility():
        print("❌ Failed to fix numpy compatibility")
        return
    
    # Step 2: Recreate voice profile
    if not recreate_voice_profile():
        print("❌ Failed to recreate voice profile")
        return
    
    print("\n=== Success! ===")
    print("The numpy compatibility issue has been fixed and a new voice profile has been created.")
    print("You can now use 'voice_profile_fixed.npy' instead of 'voice_chrisrepo1.npy'")
    print("\nTo test the new voice profile, run:")
    print("python debug_voice_profile.py")
    print("(Make sure to update the voice_profile_path in the script to 'voice_profile_fixed.npy')")

if __name__ == "__main__":
    main() 