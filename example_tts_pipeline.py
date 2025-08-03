#!/usr/bin/env python3
"""
Example script demonstrating the complete TTS pipeline.
This shows how the API app should call the new generate_tts_story method.
"""

import torch
import logging
import base64
import tempfile
from pathlib import Path
from chatterbox.tts import ChatterboxTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Example of using the complete TTS pipeline"""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA is required but not available")
        return
    
    logger.info("🚀 Initializing ChatterboxTTS...")
    
    try:
        # Initialize the TTS model
        model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ Model initialized successfully")
        
        # Example voice profile (you'll need to provide your own)
        voice_profile_path = "path/to/your/voice_profile.npy"
        
        if not Path(voice_profile_path).exists():
            logger.error(f"❌ Voice profile not found: {voice_profile_path}")
            logger.info("💡 To create a voice profile, use:")
            logger.info("   model.save_voice_profile('path/to/audio.wav', 'path/to/voice_profile.npy')")
            return
        
        # Example TTS request (what API app would send)
        tts_request = {
            "text": """
            Once upon a time, in a magical forest filled with talking animals and enchanted trees, 
            there lived a wise old owl named Professor Hoot. He was known throughout the forest for 
            his incredible knowledge and his ability to solve any problem that came his way.
            
            One day, a young rabbit named Thumper came to Professor Hoot with a very special request. 
            "Professor Hoot," said Thumper, "I want to learn how to tell stories that make everyone smile. 
            Can you teach me your secrets?"
            
            Professor Hoot smiled warmly and said, "My dear Thumper, the secret to telling great stories 
            is to speak from the heart and to believe in the magic of your own words. Every story you tell 
            should carry a piece of your soul and a spark of wonder."
            
            From that day forward, Thumper practiced telling stories with passion and joy, and soon he 
            became the most beloved storyteller in the entire forest. His stories brought happiness to 
            all who heard them, proving that the best stories come from the heart.
            """,
            "voice_id": "voice_professor_hoot",
            "language": "en",
            "story_type": "user",
            "is_kids_voice": False,
            "metadata": {
                "user_id": "user_123",
                "project_id": "story_project_456",
                "story_title": "Professor Hoot and Thumper",
                "genre": "children_fantasy"
            }
        }
        
        logger.info("🎵 Starting TTS story generation...")
        logger.info(f"📝 Text length: {len(tts_request['text'])} characters")
        logger.info(f"🎯 Voice ID: {tts_request['voice_id']}")
        logger.info(f"🌍 Language: {tts_request['language']}")
        logger.info(f"📚 Story type: {tts_request['story_type']}")
        
        # Call the complete TTS pipeline
        result = model.generate_tts_story(
            text=tts_request["text"],
            voice_profile_path=voice_profile_path,
            voice_id=tts_request["voice_id"],
            language=tts_request["language"],
            story_type=tts_request["story_type"],
            is_kids_voice=tts_request["is_kids_voice"],
            metadata=tts_request.get("metadata", {})
        )
        
        if result["status"] == "success":
            logger.info("✅ TTS story generation completed!")
            logger.info(f"📊 Results:")
            logger.info(f"  - Voice ID: {result['voice_id']}")
            logger.info(f"  - Audio Path: {result['audio_path']}")
            logger.info(f"  - Audio URL: {result['audio_url']}")
            logger.info(f"  - Generation Time: {result['generation_time']:.2f} seconds")
            logger.info(f"  - Model: {result['model']}")
            
            # Show detailed metadata
            metadata = result["metadata"]
            logger.info(f"📋 Detailed Metadata:")
            logger.info(f"  - Chunk Count: {metadata.get('chunk_count', 'N/A')}")
            logger.info(f"  - Duration: {metadata.get('duration_sec', 'N/A'):.2f} seconds")
            logger.info(f"  - Text Length: {metadata.get('text_length', 'N/A')} characters")
            logger.info(f"  - Sample Rate: {metadata.get('sample_rate', 'N/A')} Hz")
            logger.info(f"  - Language: {metadata.get('language', 'N/A')}")
            logger.info(f"  - Story Type: {metadata.get('story_type', 'N/A')}")
            logger.info(f"  - Is Kids Voice: {metadata.get('is_kids_voice', 'N/A')}")
            logger.info(f"  - Firebase Path: {metadata.get('firebase_path', 'N/A')}")
            logger.info(f"  - Timestamp: {metadata.get('timestamp', 'N/A')}")
            
        else:
            logger.error(f"❌ TTS story generation failed: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        logger.error(f"❌ Error during TTS generation: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 