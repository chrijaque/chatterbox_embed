#!/usr/bin/env python3
"""
Diagnostic script to debug voice profile issues.
This script will help identify why voice profiles aren't producing the expected voice quality.
"""

import torch
import torchaudio
import numpy as np
import os
from chatterbox import ChatterboxTTS

def debug_voice_profile(voice_profile_path, reference_audio_path, voice_embedding_path):
    """Debug voice profile functionality"""
    
    print("=== Voice Profile Debug Analysis ===")
    
    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tts = ChatterboxTTS.from_pretrained(device)
    
    # 1. Check if files exist
    print("\n1. File existence check:")
    print(f"   Voice profile: {os.path.exists(voice_profile_path)} ({voice_profile_path})")
    print(f"   Reference audio: {os.path.exists(reference_audio_path)} ({reference_audio_path})")
    print(f"   Voice embedding: {os.path.exists(voice_embedding_path)} ({voice_embedding_path})")
    
    # 2. Analyze voice profile content
    print("\n2. Voice profile content analysis:")
    try:
        import numpy as np
        data = np.load(voice_profile_path, allow_pickle=True).item()
        print(f"   Keys in profile: {list(data.keys())}")
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"   {key}: {type(value)} = {value}")
                
    except Exception as e:
        print(f"   ❌ Error loading voice profile: {e}")
        print(f"   This might be a numpy version compatibility issue.")
        print(f"   Current numpy version: {np.__version__}")
        print(f"   Trying alternative loading method...")
        
        # Try alternative loading method
        try:
            import pickle
            with open(voice_profile_path, 'rb') as f:
                data = pickle.load(f)
            print(f"   ✅ Successfully loaded with pickle")
            print(f"   Keys in profile: {list(data.keys())}")
            
            for key, value in data.items():
                if hasattr(value, 'shape'):
                    print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                else:
                    print(f"   {key}: {type(value)} = {value}")
                    
        except Exception as e2:
            print(f"   ❌ Alternative loading also failed: {e2}")
            print(f"   The voice profile file might be corrupted or incompatible.")
            return
    
    # 3. Load and compare embeddings
    print("\n3. Embedding comparison:")
    try:
        # Load voice profile
        profile = tts.load_voice_profile(voice_profile_path)
        print(f"   Voice profile embedding shape: {profile.embedding.shape}")
        print(f"   Voice profile has ve_embedding: {hasattr(profile, 've_embedding') and profile.ve_embedding is not None}")
        if hasattr(profile, 've_embedding') and profile.ve_embedding is not None:
            print(f"   Voice encoder embedding shape: {profile.ve_embedding.shape}")
        
        # Load voice embedding
        voice_embedding = tts.load_voice_clone(voice_embedding_path)
        print(f"   Voice embedding shape: {voice_embedding.shape}")
        
        # Compare embeddings
        if profile.embedding.shape == voice_embedding.shape:
            diff = torch.norm(profile.embedding - voice_embedding).item()
            print(f"   Embedding difference: {diff:.6f}")
            if diff < 1e-6:
                print("   ✅ Embeddings are identical")
            else:
                print("   ⚠️  Embeddings are different")
        else:
            print("   ❌ Embedding shapes don't match")
            
    except Exception as e:
        print(f"   ❌ Error comparing embeddings: {e}")
    
    # 4. Test TTS generation with different methods
    print("\n4. TTS generation test:")
    test_text = "Hello, this is a test of voice cloning functionality."
    
    # Method 1: Voice profile
    print("   Testing voice profile method...")
    try:
        tts.prepare_conditionals_with_voice_profile(voice_profile_path, exaggeration=0.5)
        print("   ✅ Voice profile conditionals prepared")
        
        # Check conditionals
        print(f"   T3 speaker embedding shape: {tts.conds.t3.speaker_emb.shape}")
        print(f"   S3Gen embedding shape: {tts.conds.gen['embedding'].shape}")
        print(f"   Prompt tokens shape: {tts.conds.gen['prompt_token'].shape}")
        print(f"   Prompt features shape: {tts.conds.gen['prompt_feat'].shape}")
        
        # Generate audio
        audio_profile = tts.generate(test_text, temperature=0.8)
        torchaudio.save("debug_voice_profile_output.wav", audio_profile, tts.sr)
        print("   ✅ Voice profile audio generated: debug_voice_profile_output.wav")
        
    except Exception as e:
        print(f"   ❌ Error with voice profile: {e}")
    
    # Method 2: Voice embedding + prompt audio
    print("   Testing voice embedding method...")
    try:
        tts.prepare_conditionals_with_saved_voice(voice_embedding_path, reference_audio_path, exaggeration=0.5)
        print("   ✅ Voice embedding conditionals prepared")
        
        # Check conditionals
        print(f"   T3 speaker embedding shape: {tts.conds.t3.speaker_emb.shape}")
        print(f"   S3Gen embedding shape: {tts.conds.gen['embedding'].shape}")
        print(f"   Prompt tokens shape: {tts.conds.gen['prompt_token'].shape}")
        print(f"   Prompt features shape: {tts.conds.gen['prompt_feat'].shape}")
        
        # Generate audio
        audio_embedding = tts.generate(test_text, temperature=0.8)
        torchaudio.save("debug_voice_embedding_output.wav", audio_embedding, tts.sr)
        print("   ✅ Voice embedding audio generated: debug_voice_embedding_output.wav")
        
    except Exception as e:
        print(f"   ❌ Error with voice embedding: {e}")
    
    # Method 3: Traditional method
    print("   Testing traditional method...")
    try:
        tts.prepare_conditionals(reference_audio_path, exaggeration=0.5)
        print("   ✅ Traditional conditionals prepared")
        
        # Check conditionals
        print(f"   T3 speaker embedding shape: {tts.conds.t3.speaker_emb.shape}")
        print(f"   S3Gen embedding shape: {tts.conds.gen['embedding'].shape}")
        print(f"   Prompt tokens shape: {tts.conds.gen['prompt_token'].shape}")
        print(f"   Prompt features shape: {tts.conds.gen['prompt_feat'].shape}")
        
        # Generate audio
        audio_traditional = tts.generate(test_text, temperature=0.8)
        torchaudio.save("debug_traditional_output.wav", audio_traditional, tts.sr)
        print("   ✅ Traditional audio generated: debug_traditional_output.wav")
        
    except Exception as e:
        print(f"   ❌ Error with traditional method: {e}")
    
    print("\n=== Debug Summary ===")
    print("Generated files:")
    print("  - debug_voice_profile_output.wav (voice profile method)")
    print("  - debug_voice_embedding_output.wav (voice embedding method)")
    print("  - debug_traditional_output.wav (traditional method)")
    print("\nCompare these audio files to identify which method works best.")

def main():
    # Update these paths to match your files
    voice_profile_path = "voice_chrisrepo1.npy"  # Your voice profile
    reference_audio_path = "audio_test/reference.wav"  # Original reference audio
    voice_embedding_path = "audio_test/reference_voice_clone.npy"  # Voice embedding
    
    debug_voice_profile(voice_profile_path, reference_audio_path, voice_embedding_path)

if __name__ == "__main__":
    main() 