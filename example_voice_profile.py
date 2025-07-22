#!/usr/bin/env python3
"""
Example demonstrating the new voice profile functionality in ChatterboxTTS.

This example shows how to:
1. Save a complete voice profile (embedding + prompt features + tokens)
2. Load a voice profile for TTS generation
3. Compare with the old voice cloning method
"""

import torch
from chatterbox import ChatterboxTTS

def main():
    # Initialize the TTS model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tts = ChatterboxTTS.from_pretrained(device)
    
    # Example audio file path (replace with your own)
    reference_audio_path = "path/to/your/reference/audio.wav"
    voice_profile_path = "voice_profile.npy"
    voice_embedding_path = "voice_embedding.npy"
    
    print("=== Voice Profile Example ===")
    
    # 1. Save a complete voice profile
    print("\n1. Saving complete voice profile...")
    try:
        tts.save_voice_profile(reference_audio_path, voice_profile_path)
        print("✅ Voice profile saved successfully!")
    except Exception as e:
        print(f"❌ Error saving voice profile: {e}")
        return
    
    # 2. Save just the embedding (old method for comparison)
    print("\n2. Saving voice embedding (old method)...")
    try:
        tts.save_voice_clone(reference_audio_path, voice_embedding_path)
        print("✅ Voice embedding saved successfully!")
    except Exception as e:
        print(f"❌ Error saving voice embedding: {e}")
    
    # 3. Generate speech using the complete voice profile
    print("\n3. Generating speech with complete voice profile...")
    text_to_speak = "Hello! This is a test of the new voice profile functionality."
    
    try:
        # Method 1: Using voice profile (most accurate)
        audio_profile = tts.generate(
            text=text_to_speak,
            voice_profile_path=voice_profile_path,
            exaggeration=0.5,
            temperature=0.8
        )
        print("✅ Speech generated using voice profile!")
        
        # Save the audio
        import torchaudio
        torchaudio.save("output_voice_profile.wav", audio_profile, tts.sr)
        print("✅ Audio saved as 'output_voice_profile.wav'")
        
    except Exception as e:
        print(f"❌ Error generating speech with voice profile: {e}")
    
    # 4. Alternative: Load voice profile and prepare conditionals manually
    print("\n4. Loading voice profile and preparing conditionals manually...")
    try:
        # Load the voice profile
        profile = tts.load_voice_profile(voice_profile_path)
        print(f"✅ Voice profile loaded!")
        print(f"   - Embedding shape: {profile.embedding.shape}")
        print(f"   - Prompt features shape: {profile.prompt_feat.shape}")
        print(f"   - Prompt tokens shape: {profile.prompt_token.shape}")
        print(f"   - Prompt token length: {profile.prompt_token_len}")
        
        # Prepare conditionals manually
        tts.prepare_conditionals_with_voice_profile(voice_profile_path, exaggeration=0.5)
        
        # Generate speech
        audio_manual = tts.generate(text_to_speak, temperature=0.8)
        print("✅ Speech generated using manually prepared conditionals!")
        
        # Save the audio
        import torchaudio
        torchaudio.save("output_manual_profile.wav", audio_manual, tts.sr)
        print("✅ Audio saved as 'output_manual_profile.wav'")
        
    except Exception as e:
        print(f"❌ Error in manual voice profile usage: {e}")
    
    # 5. Generate speech using the old method (for comparison)
    print("\n5. Generating speech with voice embedding (old method)...")
    try:
        # Method 2: Using voice embedding + prompt audio (requires both)
        audio_embedding = tts.generate(
            text=text_to_speak,
            saved_voice_path=voice_embedding_path,
            audio_prompt_path=reference_audio_path,  # Still need prompt audio
            exaggeration=0.5,
            temperature=0.8
        )
        print("✅ Speech generated using voice embedding!")
        
        # Save the audio
        import torchaudio
        torchaudio.save("output_voice_embedding.wav", audio_embedding, tts.sr)
        print("✅ Audio saved as 'output_voice_embedding.wav'")
        
    except Exception as e:
        print(f"❌ Error generating speech with voice embedding: {e}")
    
    # 6. Load and inspect a voice profile
    print("\n6. Loading and inspecting voice profile...")
    try:
        profile = tts.load_voice_profile(voice_profile_path)
        print(f"✅ Voice profile loaded!")
        print(f"   - Embedding shape: {profile.embedding.shape}")
        print(f"   - Prompt features shape: {profile.prompt_feat.shape}")
        print(f"   - Prompt tokens shape: {profile.prompt_token.shape}")
        print(f"   - Prompt token length: {profile.prompt_token_len}")
        
    except Exception as e:
        print(f"❌ Error loading voice profile: {e}")
    
    print("\n=== Summary ===")
    print("Voice profiles provide:")
    print("✅ Complete voice information (embedding + features + tokens)")
    print("✅ More accurate TTS generation")
    print("✅ No need for separate prompt audio during generation")
    print("✅ Faster generation (no need to recompute features)")
    print("✅ Better voice consistency")

if __name__ == "__main__":
    main() 