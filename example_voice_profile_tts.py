#!/usr/bin/env python3
"""
Example: How to correctly use voice profiles for TTS generation.

This shows the proper workflow and common mistakes to avoid.
"""

import torch
import torchaudio
from chatterbox import ChatterboxTTS

def example_correct_voice_profile_usage():
    """Example of correct voice profile usage for TTS"""
    
    print("=== Correct Voice Profile TTS Example ===")
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tts = ChatterboxTTS.from_pretrained(device)
    
    # Paths
    voice_profile_path = "voice_chrisrepo1.npy"
    test_text = "Hello, this is a test of voice cloning with the voice profile method."
    
    print(f"\n1. Loading voice profile from: {voice_profile_path}")
    
    # ✅ CORRECT METHOD: Use prepare_conditionals_with_voice_profile
    print("\n2. Preparing conditionals with voice profile...")
    tts.prepare_conditionals_with_voice_profile(voice_profile_path, exaggeration=0.5)
    print("✅ Conditionals prepared successfully!")
    
    # Check what was loaded
    print(f"   T3 speaker embedding shape: {tts.conds.t3.speaker_emb.shape}")
    print(f"   S3Gen embedding shape: {tts.conds.gen['embedding'].shape}")
    print(f"   Prompt tokens shape: {tts.conds.gen['prompt_token'].shape}")
    print(f"   Prompt features shape: {tts.conds.gen['prompt_feat'].shape}")
    
    # Generate TTS
    print(f"\n3. Generating TTS for text: '{test_text}'")
    audio = tts.generate(test_text, temperature=0.8)
    
    # Save result
    output_path = "example_voice_profile_output.wav"
    torchaudio.save(output_path, audio, tts.sr)
    print(f"✅ Generated: {output_path}")
    
    return output_path

def example_incorrect_voice_profile_usage():
    """Example of INCORRECT voice profile usage (what might be causing issues)"""
    
    print("\n=== INCORRECT Voice Profile Usage (Common Mistakes) ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = ChatterboxTTS.from_pretrained(device)
    
    voice_profile_path = "voice_chrisrepo1.npy"
    test_text = "This is what happens when you use voice profiles incorrectly."
    
    print(f"\n❌ MISTAKE 1: Loading voice profile but not setting up conditionals properly")
    
    try:
        # ❌ WRONG: Just loading the profile without setting up conditionals
        profile = tts.load_voice_profile(voice_profile_path)
        print(f"   Profile loaded: embedding shape {profile.embedding.shape}")
        
        # ❌ WRONG: Trying to generate without proper conditionals
        print("   Trying to generate without proper conditionals...")
        audio = tts.generate(test_text, temperature=0.8)
        torchaudio.save("example_incorrect_output.wav", audio, tts.sr)
        print("   ❌ This will likely sound wrong!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print(f"\n❌ MISTAKE 2: Manually setting up conditionals incorrectly")
    
    try:
        # ❌ WRONG: Manual conditional setup without proper structure
        profile = tts.load_voice_profile(voice_profile_path)
        
        # This is what the prepare_conditionals_with_voice_profile does internally
        # but doing it manually can lead to errors
        from chatterbox.models.s3gen.s3gen import T3Cond, Conditionals
        
        # ❌ WRONG: Incorrect manual setup
        t3_cond = T3Cond(
            speaker_emb=profile.embedding,  # Wrong! Should use ve_embedding
            cond_prompt_speech_tokens=profile.prompt_token,
            emotion_adv=torch.ones(1, 1, 1)
        ).to(device=tts.device)
        
        s3gen_ref_dict = dict(
            embedding=profile.embedding,
            prompt_token=profile.prompt_token,
            prompt_token_len=profile.prompt_token_len,
            prompt_feat=profile.prompt_feat,
            prompt_feat_len=profile.prompt_feat_len,
        )
        
        tts.conds = Conditionals(t3_cond, s3gen_ref_dict)
        
        audio = tts.generate(test_text, temperature=0.8)
        torchaudio.save("example_manual_incorrect_output.wav", audio, tts.sr)
        print("   ❌ This will likely sound wrong due to embedding mismatch!")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")

def example_comparison():
    """Compare different methods to show the difference"""
    
    print("\n=== Method Comparison ===")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = ChatterboxTTS.from_pretrained(device)
    
    test_text = "This is a comparison test between different voice cloning methods."
    
    # Method 1: Voice Profile (Correct)
    print("\n1. Voice Profile Method (Correct):")
    tts.prepare_conditionals_with_voice_profile("voice_chrisrepo1.npy", exaggeration=0.5)
    audio1 = tts.generate(test_text, temperature=0.8)
    torchaudio.save("comparison_voice_profile.wav", audio1, tts.sr)
    print("   ✅ Generated: comparison_voice_profile.wav")
    
    # Method 2: Traditional Method
    print("\n2. Traditional Method:")
    tts.prepare_conditionals("audio_test/reference.wav", exaggeration=0.5)
    audio2 = tts.generate(test_text, temperature=0.8)
    torchaudio.save("comparison_traditional.wav", audio2, tts.sr)
    print("   ✅ Generated: comparison_traditional.wav")
    
    # Method 3: Voice Embedding Method
    print("\n3. Voice Embedding Method:")
    tts.prepare_conditionals_with_saved_voice("audio_test/reference_voice_clone.npy", "audio_test/reference.wav", exaggeration=0.5)
    audio3 = tts.generate(test_text, temperature=0.8)
    torchaudio.save("comparison_voice_embedding.wav", audio3, tts.sr)
    print("   ✅ Generated: comparison_voice_embedding.wav")

def main():
    """Main function demonstrating correct voice profile usage"""
    
    print("Voice Profile TTS Generation Examples")
    print("=" * 50)
    
    # 1. Correct usage
    correct_output = example_correct_voice_profile_usage()
    
    # 2. Incorrect usage examples
    example_incorrect_voice_profile_usage()
    
    # 3. Comparison
    example_comparison()
    
    print("\n" + "=" * 50)
    print("SUMMARY:")
    print("✅ CORRECT: Use tts.prepare_conditionals_with_voice_profile(profile_path)")
    print("❌ WRONG: Just load_voice_profile() without setting up conditionals")
    print("❌ WRONG: Manual conditional setup without proper embedding handling")
    print("\nGenerated files:")
    print("  - example_voice_profile_output.wav (correct method)")
    print("  - comparison_voice_profile.wav (voice profile vs others)")
    print("  - comparison_traditional.wav (traditional method)")
    print("  - comparison_voice_embedding.wav (voice embedding method)")
    
    print(f"\n🎯 The key is using prepare_conditionals_with_voice_profile() method!")
    print(f"This method handles all the complex conditional setup internally.")

if __name__ == "__main__":
    main() 