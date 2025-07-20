
<img width="1200" alt="cb-big2" src="https://github.com/user-attachments/assets/bd8c5f03-e91d-4ee5-b680-57355da204d1" />

# Chatterbox TTS

[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

We're excited to introduce Chatterbox, [Resemble AI's](https://resemble.ai) first production-grade open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. It's also the first open source TTS model to support **emotion exaggeration control**, a powerful feature that makes your voices stand out. Try it now on our [Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox)

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

# Key Details
- SoTA zeroshot TTS
- 0.5B Llama backbone
- Unique exaggeration/intensity control
- Ultra-stable with alignment-informed inference
- Trained on 0.5M hours of cleaned data
- Watermarked outputs
- Easy voice conversion script
- [Outperforms ElevenLabs](https://podonos.com/resembleai/chatterbox)

# Tips
- **General Use (TTS and Voice Agents):**
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts.
  - If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
  - Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
  - Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.


# Installation
```shell
pip install chatterbox-tts
```

Alternatively, you can install from source:
```shell
# conda create -yn chatterbox python=3.11
# conda activate chatterbox

git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .
```
We developed and tested Chatterbox on Python 3.11 on Debain 11 OS; the versions of the dependencies are pinned in `pyproject.toml` to ensure consistency. You can modify the code or dependencies in this installation mode.


# Usage

## Basic TTS Generation
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")

text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-1.wav", wav, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
```

## Voice Embedding for Fast Voice Cloning

Chatterbox supports **voice embedding** - a powerful feature that allows you to save and reuse voice characteristics for significantly faster TTS generation. This is especially beneficial when generating multiple texts with the same voice.

### How Voice Embeddings Work

Voice embeddings capture the unique speaker characteristics from a reference audio file and save them as a compact `.npy` file (typically <1KB). This eliminates the need to recompute expensive speaker encodings on every generation, resulting in **substantial performance improvements** for repeated use.

### Save a Voice Clone
```python
# One-time setup: Save voice embedding from reference audio
model.save_voice_clone("reference_speaker.wav", "my_voice.npy")
```

### Use Saved Voice Clone
```python
# Fast generation using pre-saved voice embedding
wav = model.generate(
    "Any text you want to synthesize!",
    saved_voice_path="my_voice.npy",        # Pre-computed voice identity (fast!)
    audio_prompt_path="prosody_sample.wav", # Fresh audio for prosody/style
    exaggeration=0.6,
    cfg_weight=0.5
)
ta.save("output.wav", wav, model.sr)
```

### Complete Voice Cloning Example
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Load model
model = ChatterboxTTS.from_pretrained(device="cuda")

# Step 1: Create and save voice clone (do this once)
reference_audio = "speaker_sample.wav"
voice_clone_path = "speaker_voice.npy"
model.save_voice_clone(reference_audio, voice_clone_path)
print(f"Voice clone saved to {voice_clone_path}")

# Step 2: Generate multiple texts with the same voice (much faster!)
texts = [
    "Hello, this is a demonstration of voice cloning.",
    "The saved embedding makes repeated synthesis much faster.",
    "You can reuse this voice across different sessions."
]

for i, text in enumerate(texts):
    wav = model.generate(
        text,
        saved_voice_path=voice_clone_path,
        audio_prompt_path=reference_audio,  # Can use same or different audio for prosody
        temperature=0.7
    )
    ta.save(f"output_{i+1}.wav", wav, model.sr)
```

### Benefits
- **⚡ Performance**: Skip expensive speaker encoding on repeated generations
- **💾 Efficiency**: Voice embeddings are tiny (~1KB) compared to audio files
- **🔄 Reusable**: Save once, use across different sessions and applications
- **🎯 Quality**: Maintains the same voice quality as traditional method

See `example_tts.py`, `example_vc.py`, and `example_tts_with_voice_cloning.py` for more examples.

# Supported Lanugage
Currenlty only English.

# Acknowledgements
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

# Built-in PerTh Watermarking for Responsible AI

Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.


## Watermark extraction

You can look for the watermark using the following script.

```python
import perth
import librosa

AUDIO_PATH = "YOUR_FILE.wav"

# Load the watermarked audio
watermarked_audio, sr = librosa.load(AUDIO_PATH, sr=None)

# Initialize watermarker (same as used for embedding)
watermarker = perth.PerthImplicitWatermarker()

# Extract watermark
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"Extracted watermark: {watermark}")
# Output: 0.0 (no watermark) or 1.0 (watermarked)
```


# Official Discord

👋 Join us on [Discord](https://discord.gg/rJq9cRJBJ6) and let's build something awesome together!

# Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.
