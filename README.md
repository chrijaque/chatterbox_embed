
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

## Voice Profiles for Enhanced Voice Cloning

For even more accurate voice cloning, Chatterbox now supports **voice profiles** - complete voice representations that include not just the speaker embedding, but also prompt features and tokens. This provides the most accurate voice reproduction and eliminates the need for separate prompt audio during generation.

### Save a Complete Voice Profile
```python
# Save complete voice profile (embedding + features + tokens)
model.save_voice_profile("reference_speaker.wav", "my_voice_profile.npy")
```

### Use Voice Profile for TTS
```python
# Generate speech using complete voice profile (most accurate)
wav = model.generate(
    "Any text you want to synthesize!",
    voice_profile_path="my_voice_profile.npy",  # Complete voice profile
    exaggeration=0.6,
    cfg_weight=0.5
)
ta.save("output.wav", wav, model.sr)
```

### Voice Profile vs Voice Embedding

| Feature | Voice Embedding | Voice Profile |
|---------|----------------|---------------|
| **Size** | ~1KB | ~50-100KB |
| **Accuracy** | Good | Excellent |
| **Prompt Audio Required** | Yes | No |
| **Generation Speed** | Fast | Fastest |
| **Use Case** | General purpose | High-quality production |

### Complete Voice Profile Example
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Load model
model = ChatterboxTTS.from_pretrained(device="cuda")

# Step 1: Create and save voice profile (do this once)
reference_audio = "speaker_sample.wav"
voice_profile_path = "speaker_profile.npy"
model.save_voice_profile(reference_audio, voice_profile_path)
print(f"Voice profile saved to {voice_profile_path}")

# Step 2: Generate multiple texts with the same voice (fastest method!)
texts = [
    "Hello, this is a demonstration of voice profiling.",
    "The complete profile provides the most accurate voice reproduction.",
    "No need for separate prompt audio during generation."
]

for i, text in enumerate(texts):
    wav = model.generate(
        text,
        voice_profile_path=voice_profile_path,  # Complete profile - no prompt audio needed!
        temperature=0.7
    )
    ta.save(f"output_{i+1}.wav", wav, model.sr)
```

### Benefits of Voice Profiles
- **🎯 Maximum Accuracy**: Complete voice representation for best quality
- **⚡ Fastest Generation**: No need to recompute features or provide prompt audio
- **🔄 Self-Contained**: Everything needed for TTS is in one file
- **🎵 Consistent Quality**: Maintains exact voice characteristics across generations

See `example_voice_profile.py` for a complete demonstration.

## Long Text TTS with Chunking and Stitching

For processing very long texts (stories, articles, etc.), Chatterbox now supports automatic text chunking and audio stitching. This feature intelligently splits long text into manageable chunks, generates audio for each chunk, and then stitches them back together with natural pauses.

### Basic Long Text Generation
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# Load model
model = ChatterboxTTS.from_pretrained(device="cuda")

# Long text (e.g., a story or article)
long_text = """
Once upon a time, in a land far, far away, there lived a wise old wizard named Merlin. 
He had spent his entire life studying the ancient arts of magic and had become one of the 
most powerful sorcerers in the realm. His knowledge was vast, spanning from the simplest 
spells to the most complex enchantments that could change the very fabric of reality.

Merlin lived in a magnificent tower that reached high into the clouds, where he could 
observe the stars and planets to better understand the cosmic forces that governed magic. 
The tower was filled with countless books, scrolls, and magical artifacts that he had 
collected throughout his many years of travel and study.
"""

# Generate long text with chunking and stitching
audio_tensor, sample_rate, metadata = model.generate_long_text(
    text=long_text,
    voice_profile_path="my_voice_profile.npy",
    output_path="long_story.wav",
    max_chars=500,      # Maximum characters per chunk
    pause_ms=150,        # Pause between chunks in milliseconds
    temperature=0.8,     # Generation temperature
    exaggeration=0.5,    # Voice exaggeration
    cfg_weight=0.5       # CFG weight
)

print(f"Generated {metadata['chunk_count']} chunks")
print(f"Total duration: {metadata['duration_sec']:.2f} seconds")
```

### Advanced Long Text Processing
```python
# Customize chunking and stitching parameters
audio_tensor, sample_rate, metadata = model.generate_long_text(
    text=very_long_text,
    voice_profile_path="voice_profile.npy",
    output_path="output.wav",
    max_chars=300,       # Smaller chunks for better memory management
    pause_ms=200,        # Longer pauses between chunks
    temperature=0.7,     # Lower temperature for more consistent voice
    exaggeration=0.6,    # Slightly more expressive
    cfg_weight=0.4       # Lower CFG for more natural pacing
)
```

### Chunking and Stitching Features

- **📝 Intelligent Text Chunking**: Uses NLTK sentence tokenization (with fallback) to split text at natural sentence boundaries
- **🎵 Audio Stitching**: Combines audio chunks with configurable pauses between segments
- **🔧 Audio Normalization**: Professional audio processing with pydub (fallback to torchaudio)
- **🔄 Retry Logic**: Automatic retry mechanism for failed chunks with GPU cache clearing
- **🧹 Cleanup**: Automatic cleanup of temporary files
- **📊 Detailed Metadata**: Returns comprehensive information about the generation process

### Chunking Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_chars` | 500 | Maximum characters per chunk |
| `pause_ms` | 100 | Milliseconds of silence between chunks |
| `temperature` | 0.8 | Generation temperature (0.0-1.0) |
| `exaggeration` | 0.5 | Voice exaggeration factor |
| `cfg_weight` | 0.5 | CFG weight for generation |

### Benefits of Long Text TTS

- **📚 Handle Long Content**: Process stories, articles, and other long-form content
- **💾 Memory Efficient**: Processes text in manageable chunks
- **🎯 Consistent Quality**: Maintains voice consistency across long texts
- **⚡ Robust Processing**: Automatic retry logic and error handling
- **🎵 Professional Audio**: High-quality audio stitching and normalization

See `example_long_text_tts.py` for a complete demonstration.

## Voice Cloning with ChatterboxVC

ChatterboxVC provides comprehensive voice cloning capabilities, including voice profile management, audio processing, and complete voice cloning pipelines. This functionality was previously only available in API applications but is now integrated into the core library.

### Basic Voice Cloning
```python
import torch
from chatterbox.vc import ChatterboxVC
from chatterbox.tts import ChatterboxTTS

# Initialize models
tts_model = ChatterboxTTS.from_pretrained(device="cuda")
vc_model = ChatterboxVC(
    s3gen=tts_model.s3gen,
    device=tts_model.device
)

# Attach text encoder for TTS functionality
if hasattr(tts_model, "t3"):
    vc_model.s3gen.text_encoder = tts_model.t3

# Create a voice clone from reference audio
result = vc_model.create_voice_clone(
    audio_file_path="reference_audio.wav",
    voice_id="my_voice",
    output_dir="./voice_clones"
)

if result["status"] == "success":
    print(f"Voice clone created: {result['voice_id']}")
    print(f"Profile saved to: {result['profile_path']}")
```

### Voice Profile Management
```python
# Save a voice profile
vc_model.save_voice_profile("reference_audio.wav", "voice_profile.npy")

# Load and use a voice profile
vc_model.set_voice_profile("voice_profile.npy")

# Generate TTS with the voice profile
audio_tensor = vc_model.tts("Hello, this is voice cloning!")
```

### Audio Processing Utilities
```python
# Convert audio tensor to MP3 bytes
mp3_bytes = vc_model.tensor_to_mp3_bytes(audio_tensor, sample_rate=24000, bitrate="96k")

# Convert audio file to MP3
vc_model.convert_audio_file_to_mp3("input.wav", "output.mp3", bitrate="160k")

# Generate voice sample from profile
audio_tensor, mp3_bytes = vc_model.generate_voice_sample(
    voice_profile_path="voice_profile.npy",
    text="Custom voice sample text"
)
```

### Complete Voice Cloning Pipeline
```python
# Complete voice cloning with all outputs
result = vc_model.create_voice_clone(
    audio_file_path="reference_audio.wav",
    voice_id="unique_voice_id",
    output_dir="./outputs"
)

# Access all generated data
profile_path = result["profile_path"]
sample_audio_bytes = result["sample_audio_bytes"]
recorded_audio_bytes = result["recorded_audio_bytes"]

# Save outputs
with open("sample.mp3", "wb") as f:
    f.write(sample_audio_bytes)

with open("recorded.mp3", "wb") as f:
    f.write(recorded_audio_bytes)
```

### Voice Conversion (Voice Cloning)
```python
# Convert audio to target voice
converted_audio = vc_model.generate(
    audio="source_audio.wav",
    target_voice_path="reference_voice.wav"
)

# Save converted audio
import torchaudio
torchaudio.save("converted.wav", converted_audio, vc_model.sr)
```

### Voice Cloning Features

- **🎤 Complete Voice Cloning**: Create voice profiles from reference audio
- **💾 Profile Management**: Save and load voice profiles for reuse
- **🎵 Audio Processing**: Convert between audio formats (WAV, MP3)
- **📝 TTS Integration**: Generate speech using voice profiles
- **🔄 Voice Conversion**: Convert audio to target voices
- **📊 Comprehensive Metadata**: Detailed information about all operations

### Voice Cloning Parameters

| Parameter | Description |
|-----------|-------------|
| `audio_file_path` | Path to reference audio file |
| `voice_id` | Unique identifier for the voice |
| `output_dir` | Directory to save outputs |
| `bitrate` | MP3 bitrate for audio conversion |
| `text` | Custom text for voice samples |

### Benefits of Voice Cloning

- **🎯 High Quality**: Professional voice cloning with accurate voice reproduction
- **⚡ Fast Processing**: Efficient voice profile creation and reuse
- **🔄 Reusable**: Save voice profiles once, use multiple times
- **🎵 Multiple Formats**: Support for WAV and MP3 audio formats
- **📊 Detailed Output**: Comprehensive metadata and multiple output formats

See `example_voice_cloning.py` for a complete demonstration.

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
