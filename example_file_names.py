#!/usr/bin/env python3
"""
Example script showing file names with new alphanumeric voice ID generation.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chatterbox.vc import generate_unique_voice_id

def show_file_name_examples():
    """Show examples of file names with the new voice ID generation."""
    
    print("📁 File Name Examples with New Alphanumeric Voice ID Generation")
    print("=" * 70)
    
    # Generate some example voice IDs
    voice_names = ["christestclone", "alice", "bob", "sarah", "mike"]
    
    for voice_name in voice_names:
        voice_id = generate_unique_voice_id(voice_name)
        
        print(f"\n🎤 Voice Name: '{voice_name}'")
        print(f"🆔 Generated Voice ID: '{voice_id}'")
        print("📁 Generated Files:")
        print(f"  - Profile: {voice_id}.npy")
        print(f"  - Recorded Audio: {voice_id}_recorded.mp3")
        print(f"  - Sample Audio: {voice_id}_sample.mp3")
        
        print("🌐 Firebase Paths:")
        print(f"  - Profile: audio/voices/en/profiles/{voice_id}.npy")
        print(f"  - Recorded: audio/voices/en/recorded/{voice_id}_recorded.mp3")
        print(f"  - Sample: audio/voices/en/samples/{voice_id}_sample.mp3")
    
    print("\n" + "=" * 70)
    print("✅ Benefits:")
    print("  - Alphanumeric characters (A-Z, a-z, 0-9)")
    print("  - 8-character random suffix by default")
    print("  - ~218 trillion possible combinations")
    print("  - Firebase uniqueness checking")
    print("  - Collision prevention across all languages")
    print("  - Fallback to timestamp if needed")

if __name__ == "__main__":
    show_file_name_examples() 