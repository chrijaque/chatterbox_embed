import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Automatically detect the best available device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Load the TTS model
model = ChatterboxTTS.from_pretrained(device=device)

# Text to synthesize
text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."

# Example 1: Traditional method (computes everything fresh each time)
print("Example 1: Traditional voice cloning")
REFERENCE_AUDIO = "YOUR_REFERENCE_VOICE.wav"  # Replace with your audio file
wav = model.generate(text, audio_prompt_path=REFERENCE_AUDIO)
ta.save("traditional-method.wav", wav, model.sr)

# Example 2: Save a voice clone for fast reuse
print("Example 2: Saving voice clone...")
VOICE_CLONE_PATH = "my_saved_voice.npy"
model.save_voice_clone(REFERENCE_AUDIO, VOICE_CLONE_PATH)

# Example 3: Use saved voice clone (much faster!)
print("Example 3: Using saved voice clone")
# Note: You still need a prompt audio for prosody/style, but the expensive speaker encoding is skipped
PROMPT_AUDIO = REFERENCE_AUDIO  # Can be the same file or different clip for varied prosody
wav_fast = model.generate(
    text, 
    saved_voice_path=VOICE_CLONE_PATH,  # Pre-saved voice identity
    audio_prompt_path=PROMPT_AUDIO      # Fresh audio for prosody/style
)
ta.save("fast-voice-clone.wav", wav_fast, model.sr)

# Example 4: Generate multiple texts with same voice (main benefit!)
texts = [
    "Hello, this is a test of the voice cloning system.",
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning has revolutionized artificial intelligence."
]

print("Example 4: Generating multiple texts with saved voice...")
for i, text in enumerate(texts):
    wav = model.generate(
        text,
        saved_voice_path=VOICE_CLONE_PATH,
        audio_prompt_path=PROMPT_AUDIO,
        temperature=0.7,  # You can still adjust generation parameters
        exaggeration=0.6
    )
    ta.save(f"voice-clone-{i+1}.wav", wav, model.sr)

print("All examples completed! Check the generated audio files.")
print(f"\nSaved voice embedding: {VOICE_CLONE_PATH}")
print("You can reuse this .npy file for future generations with the same voice!") 