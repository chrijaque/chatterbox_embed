# Deep Dependency Analysis for Chatterbox TTS Repository

## Overview
This document provides a comprehensive analysis of all dependencies required to run the Chatterbox TTS repository in production environments, Docker containers, or other applications.

## Core Dependencies (From pyproject.toml)

### Primary Dependencies
```toml
dependencies = [
    "numpy>=1.26.0",           # Numerical computing
    "librosa==0.11.0",         # Audio processing
    "s3tokenizer",             # Speech tokenization
    "torch==2.6.0",            # Deep learning framework
    "torchaudio==2.6.0",       # Audio processing for PyTorch
    "transformers==4.46.3",    # Hugging Face transformers
    "diffusers==0.29.0",       # Diffusion models
    "resemble-perth==1.0.1",   # Watermarking
    "conformer==0.3.2",        # Conformer architecture
    "safetensors==0.5.3"       # Safe tensor serialization
]
```

## Detailed Dependency Breakdown

### 1. **Core ML/AI Framework**
- **torch==2.6.0** - Primary deep learning framework
  - Used for: All neural network operations, tensor operations, model inference
  - Critical for: T3 model, S3Gen model, voice encoder, all model components
  - GPU/CPU support required

- **torchaudio==2.6.0** - Audio processing extension for PyTorch
  - Used for: Audio resampling, audio I/O operations
  - Critical for: Audio preprocessing, model input/output

### 2. **Audio Processing**
- **librosa==0.11.0** - Audio and music analysis library
  - Used for: Audio loading, resampling, mel spectrogram extraction
  - Critical for: Voice profile creation, audio preprocessing
  - Dependencies: scipy, numpy, soundfile, audioread

- **s3tokenizer** - Speech tokenization library
  - Used for: Converting speech to discrete tokens
  - Critical for: S3Gen model input processing
  - Dependencies: torch, librosa, onnx

### 3. **Transformer Models**
- **transformers==4.46.3** - Hugging Face transformers library
  - Used for: Llama model backbone, tokenization, generation
  - Critical for: T3 model (text-to-speech token generation)
  - Dependencies: torch, tokenizers, huggingface-hub

- **diffusers==0.29.0** - Diffusion models library
  - Used for: Flow matching, attention mechanisms
  - Critical for: S3Gen flow matching components
  - Dependencies: torch, transformers, accelerate

### 4. **Specialized Components**
- **conformer==0.3.2** - Conformer architecture implementation
  - Used for: Audio encoder in S3Gen
  - Critical for: Speech feature processing
  - Dependencies: torch, einops

- **resemble-perth==1.0.1** - Watermarking library
  - Used for: Audio watermarking
  - Critical for: Output audio watermarking
  - Dependencies: torch, numpy

- **safetensors==0.5.3** - Safe tensor serialization
  - Used for: Model weight loading
  - Critical for: Loading pre-trained models
  - Dependencies: None (pure Python)

### 5. **Numerical Computing**
- **numpy>=1.26.0** - Numerical computing library
  - Used for: Array operations, data manipulation
  - Critical for: All data processing operations
  - Dependencies: None (core library)

## Hidden Dependencies (Not in pyproject.toml)

### 6. **Additional Required Libraries**
```python
# Found in code analysis:
import tqdm                    # Progress bars
import einops                  # Tensor operations
import scipy                   # Scientific computing
import soundfile               # Audio file I/O
import audioread               # Audio file reading
import huggingface_hub         # Model downloading
import tokenizers              # Text tokenization
import accelerate              # Distributed training
import setuptools              # Package building
```

### 7. **System Dependencies**
```bash
# Audio processing dependencies:
- libsndfile                  # Audio file format support
- ffmpeg                      # Audio/video processing
- sox                         # Audio processing utilities

# Python packages:
- setuptools>=61.0            # Package building
- importlib-metadata          # Package metadata
- typing_extensions           # Type hints
- packaging                   # Package utilities
```

## Runtime Requirements

### 8. **Hardware Requirements**
- **GPU**: CUDA-compatible GPU recommended (NVIDIA)
- **CPU**: Multi-core CPU for fallback
- **RAM**: Minimum 8GB, recommended 16GB+
- **Storage**: 2-5GB for models and dependencies

### 9. **Software Requirements**
- **Python**: >=3.9 (as specified)
- **CUDA**: 11.8+ (for GPU acceleration)
- **OS**: Linux (recommended), macOS, Windows

## Installation Strategies

### 10. **Production Installation**

#### Option A: From PyPI (Original)
```bash
pip install chatterbox-tts
# ❌ Missing voice profile functionality
```

#### Option B: From Local Source (Recommended)
```bash
git clone <your-repo>
cd chatterbox_embed
pip install -e .
# ✅ Includes voice profile functionality
```

#### Option C: From Fork
```bash
pip install git+https://github.com/YOUR_USERNAME/chatterbox_embed.git
# ✅ Includes voice profile functionality
```

### 11. **Docker Installation**
```dockerfile
# Base image with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 python3.9-dev python3-pip \
    libsndfile1 ffmpeg sox \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install chatterbox from source
COPY . /app/chatterbox_embed
WORKDIR /app/chatterbox_embed
RUN pip install -e .
```

### 12. **Requirements.txt (Complete)**
```txt
# Core dependencies
numpy>=1.26.0
librosa==0.11.0
s3tokenizer
torch==2.6.0
torchaudio==2.6.0
transformers==4.46.3
diffusers==0.29.0
resemble-perth==1.0.1
conformer==0.3.2
safetensors==0.5.3

# Hidden dependencies
tqdm
einops
scipy
soundfile
audioread
huggingface_hub
tokenizers
accelerate
setuptools>=61.0
importlib-metadata
typing_extensions
packaging

# Optional but recommended
torchaudio[all]  # Full audio support
```

## Dependency Categories

### 13. **Critical Dependencies (Must Have)**
- torch, torchaudio
- transformers
- librosa
- s3tokenizer
- numpy
- safetensors

### 14. **Important Dependencies (Should Have)**
- diffusers
- conformer
- resemble-perth
- huggingface_hub

### 15. **Optional Dependencies (Nice to Have)**
- tqdm (progress bars)
- einops (tensor operations)
- accelerate (distributed training)

## Potential Issues and Solutions

### 16. **Common Installation Issues**

#### CUDA Version Mismatch
```bash
# Solution: Install matching CUDA version
pip install torch==2.6.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

#### Audio Library Issues
```bash
# Solution: Install system audio libraries
sudo apt-get install libsndfile1 ffmpeg sox
```

#### Memory Issues
```bash
# Solution: Use CPU-only version
pip install torch==2.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

### 17. **Performance Optimizations**
- Use CUDA for GPU acceleration
- Install torch with CUDA support
- Use appropriate batch sizes
- Enable mixed precision (torch.cuda.amp)

## Summary

### 18. **Minimal Production Setup**
```bash
# 1. Install system dependencies
sudo apt-get install libsndfile1 ffmpeg sox

# 2. Install Python dependencies
pip install torch==2.6.0 torchaudio==2.6.0
pip install transformers==4.46.3 diffusers==0.29.0
pip install librosa==0.11.0 s3tokenizer conformer==0.3.2
pip install resemble-perth==1.0.1 safetensors==0.5.3

# 3. Install chatterbox from source
git clone <your-repo>
cd chatterbox_embed
pip install -e .
```

### 19. **Verification Commands**
```python
# Test installation
import torch
import torchaudio
import transformers
import librosa
import chatterbox

# Test voice profile functionality
from chatterbox import ChatterboxTTS
model = ChatterboxTTS.from_pretrained("cpu")
print("✅ All dependencies working!")
```

This analysis ensures that any application using this repository will have all necessary dependencies for both the original functionality and the new voice profile features. 