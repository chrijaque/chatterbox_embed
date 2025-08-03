#!/usr/bin/env python3
"""
Example script demonstrating voice cloning with ChatterboxVC.
This shows how to use the new voice cloning methods in ChatterboxVC.
"""

import torch
import logging
from pathlib import Path
from chatterbox.vc import ChatterboxVC
from chatterbox.tts import ChatterboxTTS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Example of using voice cloning with ChatterboxVC"""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.error("❌ CUDA is required but not available")
        return
    
    logger.info("🚀 Initializing ChatterboxVC...")
    
    try:
        # Initialize the TTS model first (needed for S3Gen)
        tts_model = ChatterboxTTS.from_pretrained(device='cuda')
        logger.info("✅ ChatterboxTTS model initialized")
        
        # Initialize ChatterboxVC with the S3Gen from TTS model
        vc_model = ChatterboxVC(
            s3gen=tts_model.s3gen,
            device=tts_model.device
        )
        logger.info("✅ ChatterboxVC initialized successfully")
        
        # Attach the T3 text encoder to S3Gen for TTS functionality
        if hasattr(tts_model, "t3"):
            vc_model.s3gen.text_encoder = tts_model.t3
            logger.info("📌 Attached text encoder to S3Gen")
        
        # Example 1: Create a voice clone from an audio file
        logger.info("🎤 ===== EXAMPLE 1: Creating Voice Clone =====")
        
        # You'll need to provide your own audio file
        audio_file_path = "path/to/your/reference_audio.wav"
        voice_id = "example_voice"
        
        if not Path(audio_file_path).exists():
            logger.error(f"❌ Audio file not found: {audio_file_path}")
            logger.info("💡 Please provide a valid audio file path")
            return
        
        # Create voice clone (complete pipeline)
        result = vc_model.create_voice_clone(
            audio_file_path=audio_file_path,
            voice_id=voice_id,
            output_dir="./voice_clones"
        )
        
        if result["status"] == "success":
            logger.info("✅ Voice clone created successfully!")
            logger.info(f"📊 Results:")
            logger.info(f"  - Voice ID: {result['voice_id']}")
            logger.info(f"  - Profile path: {result['profile_path']}")
            logger.info(f"  - Sample audio size: {result['sample_audio_size']:,} bytes")
            logger.info(f"  - Recorded audio size: {result['recorded_audio_size']:,} bytes")
            logger.info(f"  - Audio tensor shape: {result['audio_tensor_shape']}")
            logger.info(f"  - Sample rate: {result['sample_rate']} Hz")
            
            # Save the sample audio to a file
            sample_output_path = f"./voice_clones/{voice_id}_sample.mp3"
            with open(sample_output_path, 'wb') as f:
                f.write(result['sample_audio_bytes'])
            logger.info(f"💾 Sample audio saved to: {sample_output_path}")
            
            # Save the recorded audio to a file
            recorded_output_path = f"./voice_clones/{voice_id}_recorded.mp3"
            with open(recorded_output_path, 'wb') as f:
                f.write(result['recorded_audio_bytes'])
            logger.info(f"💾 Recorded audio saved to: {recorded_output_path}")
            
        else:
            logger.error(f"❌ Voice clone creation failed: {result.get('error', 'Unknown error')}")
            return
        
        # Example 2: Generate additional voice samples using the saved profile
        logger.info("🎵 ===== EXAMPLE 2: Generating Additional Samples =====")
        
        profile_path = result['profile_path']
        
        # Generate a custom voice sample
        custom_text = "This is a custom voice sample generated using the saved voice profile."
        audio_tensor, mp3_bytes = vc_model.generate_voice_sample(
            voice_profile_path=profile_path,
            text=custom_text
        )
        
        # Save the custom sample
        custom_sample_path = f"./voice_clones/{voice_id}_custom_sample.mp3"
        with open(custom_sample_path, 'wb') as f:
            f.write(mp3_bytes)
        logger.info(f"💾 Custom sample saved to: {custom_sample_path}")
        
        # Example 3: Voice conversion (voice cloning)
        logger.info("🔄 ===== EXAMPLE 3: Voice Conversion =====")
        
        # Load a different audio file to convert to the target voice
        source_audio_path = "path/to/your/source_audio.wav"
        
        if Path(source_audio_path).exists():
            # Convert the source audio to the target voice
            converted_audio = vc_model.generate(
                audio=source_audio_path,
                target_voice_path=audio_file_path  # Use the original reference
            )
            
            # Save the converted audio
            converted_output_path = f"./voice_clones/{voice_id}_converted.wav"
            import torchaudio
            torchaudio.save(converted_output_path, converted_audio, vc_model.sr)
            logger.info(f"💾 Converted audio saved to: {converted_output_path}")
        else:
            logger.warning(f"⚠️ Source audio file not found: {source_audio_path}")
            logger.info("💡 Skipping voice conversion example")
        
        # Example 4: TTS with voice profile
        logger.info("📝 ===== EXAMPLE 4: TTS with Voice Profile =====")
        
        # Set the voice profile for TTS
        vc_model.set_voice_profile(profile_path)
        
        # Generate TTS with the voice profile
        tts_text = "Hello, this is text-to-speech using the voice profile. The voice cloning technology allows us to synthesize speech in any voice."
        tts_audio = vc_model.tts(tts_text)
        
        # Save the TTS audio
        tts_output_path = f"./voice_clones/{voice_id}_tts.wav"
        import torchaudio
        torchaudio.save(tts_output_path, tts_audio, vc_model.sr)
        logger.info(f"💾 TTS audio saved to: {tts_output_path}")
        
        logger.info("🎉 All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Error during voice cloning: {e}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 