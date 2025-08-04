#!/usr/bin/env python3
"""
Debug script to check available methods on ChatterboxVC.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def debug_methods():
    """Debug what methods are available on ChatterboxVC."""
    
    print("🔍 Debugging ChatterboxVC methods...")
    
    try:
        # Import the module
        import chatterbox.vc
        print("✅ chatterbox.vc imported successfully")
        
        # Check the class
        from chatterbox.vc import ChatterboxVC
        print("✅ ChatterboxVC imported successfully")
        
        # Get all methods
        all_methods = [m for m in dir(ChatterboxVC) if not m.startswith('_')]
        print(f"📋 All ChatterboxVC methods ({len(all_methods)}):")
        for method in sorted(all_methods):
            print(f"  - {method}")
        
        # Check for specific methods
        expected_methods = [
            'create_voice_clone',
            'save_voice_profile',
            'load_voice_profile', 
            'set_voice_profile',
            'tensor_to_mp3_bytes',
            'tensor_to_audiosegment',
            'tensor_to_wav_bytes',
            'convert_audio_file_to_mp3',
            'upload_to_firebase'
        ]
        
        print(f"\n🔍 Checking expected methods:")
        for method in expected_methods:
            has_method = hasattr(ChatterboxVC, method)
            print(f"  - {method}: {'✅' if has_method else '❌'}")
        
        # Check the source code
        print(f"\n🔍 Checking source code for create_voice_clone:")
        with open('src/chatterbox/vc.py', 'r') as f:
            content = f.read()
            if 'def create_voice_clone' in content:
                print("  ✅ create_voice_clone found in source code")
                # Find the line number
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if 'def create_voice_clone' in line:
                        print(f"  - Line {i+1}: {line.strip()}")
                        break
            else:
                print("  ❌ create_voice_clone NOT found in source code")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    debug_methods() 