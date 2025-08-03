#!/usr/bin/env python3
"""
Example script demonstrating long text TTS with chunking and stitching.
This shows how to use the new generate_long_text method in ChatterboxTTS.
"""

import torch
import logging
from pathlib import Path
from chatterbox.tts import ChatterboxTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Example of using long text TTS with chunking and stitching"""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA is required but not available")
        return
    
    logger.info("🚀 Initializing ChatterboxTTS...")
    
    try:
        # Initialize the TTS model
        model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ Model initialized successfully")
        
        # Example long text (you can replace this with your own text)
        long_text = """
        Once upon a time, in a land far, far away, there lived a wise old wizard named Merlin. 
        He had spent his entire life studying the ancient arts of magic and had become one of the 
        most powerful sorcerers in the realm. His knowledge was vast, spanning from the simplest 
        spells to the most complex enchantments that could change the very fabric of reality.
        
        Merlin lived in a magnificent tower that reached high into the clouds, where he could 
        observe the stars and planets to better understand the cosmic forces that governed magic. 
        The tower was filled with countless books, scrolls, and magical artifacts that he had 
        collected throughout his many years of travel and study. Each item held a story, and each 
        story held a lesson that had helped shape him into the wise wizard he had become.
        
        One day, a young apprentice named Arthur arrived at Merlin's tower, seeking to learn the 
        ways of magic. Arthur was bright and eager, but he was also impatient and often tried to 
        rush through his studies. Merlin saw great potential in the young man, but he also knew 
        that true wisdom comes not from haste, but from careful study and deep understanding.
        
        "Magic is not about power," Merlin would often tell Arthur, "it's about understanding. 
        Understanding the world around you, understanding yourself, and understanding how all things 
        are connected. When you truly understand something, you can work with it rather than against it."
        
        Arthur struggled with this concept at first. He wanted to cast powerful spells immediately, 
        to see results right away. But Merlin was patient and continued to guide him, showing him 
        how to observe, how to question, and how to think deeply about the nature of magic and reality.
        
        As the years passed, Arthur began to understand. He learned to see the patterns in nature, 
        to recognize the subtle energies that flowed through all things, and to work with these forces 
        rather than trying to force his will upon them. His magic became more elegant, more effective, 
        and more in harmony with the world around him.
        
        Eventually, Arthur became a great wizard in his own right, not because he had learned to 
        cast the most powerful spells, but because he had learned to understand the fundamental 
        principles that governed all magic. He had learned that true power comes from wisdom, 
        and wisdom comes from understanding.
        
        The lesson that Arthur learned, and that we can all learn from this story, is that success 
        in any field comes not from rushing to achieve quick results, but from taking the time to 
        truly understand the principles and fundamentals that govern that field. Whether it's magic, 
        science, art, or any other pursuit, the path to mastery lies in deep understanding rather 
        than superficial knowledge.
        """
        
        # Create a voice profile from an audio file (you'll need to provide your own audio file)
        voice_profile_path = "path/to/your/voice_profile.npy"
        
        # Check if voice profile exists
        if not Path(voice_profile_path).exists():
            logger.error(f"❌ Voice profile not found: {voice_profile_path}")
            logger.info("💡 To create a voice profile, use:")
            logger.info("   model.save_voice_profile('path/to/audio.wav', 'path/to/voice_profile.npy')")
            return
        
        # Output path for the generated audio
        output_path = "long_text_output.wav"
        
        logger.info("🎵 Starting long text TTS generation...")
        logger.info(f"📝 Text length: {len(long_text)} characters")
        logger.info(f"🎯 Voice profile: {voice_profile_path}")
        logger.info(f"📁 Output path: {output_path}")
        
        # Generate the long text with chunking and stitching
        audio_tensor, sample_rate, metadata = model.generate_long_text(
            text=long_text,
            voice_profile_path=voice_profile_path,
            output_path=output_path,
            max_chars=500,      # Maximum characters per chunk
            pause_ms=150,        # Pause between chunks in milliseconds
            temperature=0.8,     # Generation temperature
            exaggeration=0.5,    # Voice exaggeration
            cfg_weight=0.5       # CFG weight
        )
        
        logger.info("✅ Long text TTS generation completed!")
        logger.info(f"📊 Results:")
        logger.info(f"  - Audio tensor shape: {audio_tensor.shape}")
        logger.info(f"  - Sample rate: {sample_rate} Hz")
        logger.info(f"  - Duration: {metadata['duration_sec']:.2f} seconds")
        logger.info(f"  - Chunks processed: {metadata['chunk_count']}")
        logger.info(f"  - Output file: {metadata['output_path']}")
        logger.info(f"  - File exists: {Path(output_path).exists()}")
        
        if Path(output_path).exists():
            file_size = Path(output_path).stat().st_size
            logger.info(f"  - File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
        
    except Exception as e:
        logger.error(f"❌ Error during TTS generation: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 