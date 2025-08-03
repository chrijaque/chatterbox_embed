#!/usr/bin/env python3
"""
Debug script to check what's actually in the files that RunPod is using.
"""

import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_file_contents():
    """
    Check what's actually in the files that RunPod is using.
    """
    logger.info("🔍 Checking file contents...")
    
    # Check VC file
    vc_file = "/workspace/chatterbox_embed/src/chatterbox/vc.py"
    if os.path.exists(vc_file):
        logger.info(f"✅ VC file exists: {vc_file}")
        
        with open(vc_file, 'r') as f:
            content = f.read()
            
        # Check for specific methods
        methods_to_check = [
            'def create_voice_clone',
            'def save_voice_profile',
            'def load_voice_profile',
            'def set_voice_profile',
            'def tensor_to_mp3_bytes',
            'def convert_audio_file_to_mp3'
        ]
        
        logger.info("🔍 Checking VC methods:")
        for method in methods_to_check:
            if method in content:
                logger.info(f"  ✅ Found: {method}")
            else:
                logger.error(f"  ❌ Missing: {method}")
        
        # Show last few lines to see what's actually there
        lines = content.split('\n')
        logger.info(f"📄 VC file has {len(lines)} lines")
        logger.info("📄 Last 20 lines of VC file:")
        for i, line in enumerate(lines[-20:]):
            logger.info(f"  {len(lines)-20+i+1}: {line}")
            
    else:
        logger.error(f"❌ VC file does not exist: {vc_file}")
    
    # Check TTS file
    tts_file = "/workspace/chatterbox_embed/src/chatterbox/tts.py"
    if os.path.exists(tts_file):
        logger.info(f"✅ TTS file exists: {tts_file}")
        
        with open(tts_file, 'r') as f:
            content = f.read()
            
        # Check for specific methods
        methods_to_check = [
            'def generate_tts_story',
            'def generate_long_text',
            'def chunk_text',
            'def generate_chunks',
            'def stitch_and_normalize',
            'def cleanup_chunks'
        ]
        
        logger.info("🔍 Checking TTS methods:")
        for method in methods_to_check:
            if method in content:
                logger.info(f"  ✅ Found: {method}")
            else:
                logger.error(f"  ❌ Missing: {method}")
        
        # Show last few lines to see what's actually there
        lines = content.split('\n')
        logger.info(f"📄 TTS file has {len(lines)} lines")
        logger.info("📄 Last 20 lines of TTS file:")
        for i, line in enumerate(lines[-20:]):
            logger.info(f"  {len(lines)-20+i+1}: {line}")
            
    else:
        logger.error(f"❌ TTS file does not exist: {tts_file}")

if __name__ == "__main__":
    check_file_contents() 