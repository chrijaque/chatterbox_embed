import torchaudio as ta
import torch
import os
import sys
import numpy as np
from pathlib import Path

# Add the src directory to the path so we can import chatterbox
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load the model
model = ChatterboxTTS.from_pretrained(device=device)

# Test if load_voice_clone method exists
print(f"\n🔍 Checking model capabilities:")
print(f"  - has load_voice_clone: {hasattr(model, 'load_voice_clone')}")
print(f"  - has generate: {hasattr(model, 'generate')}")

if not hasattr(model, 'load_voice_clone'):
    print("❌ load_voice_clone method is NOT available!")
    print("This suggests you might be using an older version or different installation.")
    sys.exit(1)

print("✅ load_voice_clone method is available!")

# Path to your saved voice embedding
SAVED_VOICE_PATH = "audio_test/reference_voice_clone.npy"

# Check if the voice file exists
if not os.path.exists(SAVED_VOICE_PATH):
    print(f"❌ Voice embedding file not found: {SAVED_VOICE_PATH}")
    sys.exit(1)

print(f"✅ Found voice embedding: {SAVED_VOICE_PATH}")

# Test loading the voice embedding directly
print(f"\n📂 Testing direct voice embedding loading...")
try:
    loaded_embedding = model.load_voice_clone(SAVED_VOICE_PATH)
    print(f"✅ Successfully loaded voice embedding")
    print(f"  - Shape: {loaded_embedding.shape}")
    print(f"  - Dtype: {loaded_embedding.dtype}")
    print(f"  - Device: {loaded_embedding.device}")
    
except Exception as e:
    print(f"❌ Error loading voice embedding: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create a custom method to use only the saved voice embedding
def generate_with_saved_voice_only(model, text, saved_voice_path, exaggeration=0.5, cfg_weight=0.5):
    """Generate TTS using only a saved voice embedding without requiring reference audio"""
    
    # Load the saved voice embedding
    saved_embedding = model.s3gen.load_voice_clone(saved_voice_path)
    
    # Create minimal prosody tokens and features (using zeros as default)
    # This is a workaround since the model requires these for the architecture
    device = model.device
    
    # Create minimal prompt tokens (just a few tokens)
    prompt_tokens = torch.zeros(1, 10, dtype=torch.long, device=device)  # Minimal prompt
    prompt_token_len = torch.tensor([10], device=device)
    
    # Create minimal mel features
    prompt_feat = torch.zeros(1, 20, 80, device=device)  # Minimal mel features
    
    # Create s3gen ref_dict with saved embedding and minimal prosody
    s3gen_ref_dict = dict(
        prompt_token=prompt_tokens,
        prompt_token_len=prompt_token_len,
        prompt_feat=prompt_feat,
        prompt_feat_len=None,
        embedding=saved_embedding,  # Use the pre-saved embedding!
    )
    
    # Create minimal T3 conditionals
    from chatterbox.models.t3.modules.cond_enc import T3Cond
    
    # Use a simple speaker embedding (you could also load this from the saved embedding)
    speaker_emb = torch.zeros(1, 256, device=device)  # Default speaker embedding size
    
    # Create minimal prompt tokens for T3
    t3_cond_prompt_tokens = torch.zeros(1, 10, dtype=torch.long, device=device)
    
    t3_cond = T3Cond(
        speaker_emb=speaker_emb,
        cond_prompt_speech_tokens=t3_cond_prompt_tokens,
        emotion_adv=exaggeration * torch.ones(1, 1, 1),
    ).to(device=device)
    
    # Set the conditionals
    from chatterbox.tts import Conditionals
    model.conds = Conditionals(t3_cond, s3gen_ref_dict)
    
    # Now generate using the standard method
    return model.generate(
        text,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight
    )

# Test text to synthesize
text = "Hello, this is a test of text-to-speech using only the saved voice embedding. No reference audio file needed!"

print(f"\n🎵 Generating TTS with saved voice embedding only (no reference audio)...")
print(f"Text: {text}")

try:
    # Generate TTS using only the saved voice embedding
    wav = generate_with_saved_voice_only(
        model,
        text,
        SAVED_VOICE_PATH,
        exaggeration=0.5,
        cfg_weight=0.5
    )
    
    # Save the output
    output_path = "test_output_with_saved_voice_only.wav"
    ta.save(output_path, wav, model.sr)
    print(f"✅ Generated audio saved to: {output_path}")
    
except Exception as e:
    print(f"❌ Error generating TTS: {e}")
    import traceback
    traceback.print_exc()

print("\n🏁 TTS generation completed!") 