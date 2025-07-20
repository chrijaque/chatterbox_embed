#!/usr/bin/env python3
"""
Practical test of voice cloning using the reference.wav file in audio_test directory
"""
import os
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

def main():
    print("🎙️ Practical Voice Cloning Test")
    print("=" * 50)
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    
    # File paths
    audio_dir = "audio_test"
    reference_audio = os.path.join(audio_dir, "reference.wav")
    voice_clone_path = os.path.join(audio_dir, "reference_voice_clone.npy")
    
    # Check if reference file exists
    if not os.path.exists(reference_audio):
        print(f"❌ Reference audio not found: {reference_audio}")
        return
    
    print(f"📁 Reference audio: {reference_audio}")
    print(f"💾 Voice clone will be saved to: {voice_clone_path}")
    
    # Load TTS model
    print("\n🤖 Loading ChatterboxTTS model...")
    try:
        model = ChatterboxTTS.from_pretrained(device=device)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Step 1: Create and save voice clone
    print(f"\n🎯 Step 1: Creating voice clone from {reference_audio}...")
    try:
        model.save_voice_clone(reference_audio, voice_clone_path)
        print(f"✅ Voice clone saved to: {voice_clone_path}")
        
        # Check file size
        file_size = os.path.getsize(voice_clone_path)
        print(f"📊 Voice clone file size: {file_size} bytes ({file_size/1024:.2f} KB)")
        
    except Exception as e:
        print(f"❌ Failed to create voice clone: {e}")
        return
    
    # Step 2: Test TTS generation with saved voice clone
    print(f"\n🗣️ Step 2: Testing TTS generation with saved voice clone...")
    test_texts = [
        "Hello! This is a test of the voice cloning system using the saved embedding.",
        "The voice clone should sound like the original reference audio.",
        "This demonstrates significant performance improvements for repeated synthesis."
    ]
    
    output_dir = audio_dir
    
    for i, text in enumerate(test_texts, 1):
        try:
            print(f"   Generating text {i}: '{text[:50]}...'")
            
            # Generate using saved voice clone (fast method!)
            wav = model.generate(
                text,
                saved_voice_path=voice_clone_path,  # Pre-saved voice identity
                audio_prompt_path=reference_audio,  # Fresh audio for prosody
                temperature=0.7,
                exaggeration=0.6
            )
            
            # Save output
            output_path = os.path.join(output_dir, f"voice_clone_test_{i}.wav")
            ta.save(output_path, wav, model.sr)
            print(f"   ✅ Saved: {output_path}")
            
        except Exception as e:
            print(f"   ❌ Failed to generate text {i}: {e}")
    
    # Step 3: Compare with traditional method (for reference)
    print(f"\n🔄 Step 3: Generating with traditional method (for comparison)...")
    try:
        traditional_wav = model.generate(
            "This is generated using the traditional method without saved embeddings.",
            audio_prompt_path=reference_audio,  # Computes embedding fresh each time
            temperature=0.7,
            exaggeration=0.6
        )
        
        traditional_path = os.path.join(output_dir, "traditional_method_comparison.wav")
        ta.save(traditional_path, traditional_wav, model.sr)
        print(f"✅ Traditional method saved: {traditional_path}")
        
    except Exception as e:
        print(f"❌ Traditional method failed: {e}")
    
    # Summary
    print(f"\n🎉 Voice cloning test completed!")
    print(f"📂 All outputs saved in: {audio_dir}/")
    print(f"💾 Voice clone embedding: {voice_clone_path}")
    print(f"\n🚀 Benefits of saved voice clone:")
    print(f"   • Skip expensive speaker encoding on repeated use")
    print(f"   • Faster TTS generation for same voice")
    print(f"   • Reusable across sessions")
    print(f"\n📝 Usage pattern:")
    print(f"   model.generate(text, saved_voice_path='{voice_clone_path}', audio_prompt_path='prompt.wav')")

if __name__ == "__main__":
    main() 