"""Shared utilities for chatterbox TTS and VC modules."""
import os
import subprocess
import logging
from typing import Tuple

import torch
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import nltk
    from nltk.tokenize.punkt import PunktSentenceTokenizer
    # Force download and setup NLTK punkt tokenizer
    try:
        nltk.download('punkt', quiet=True)
        nltk.data.find('tokenizers/punkt')
        NLTK_AVAILABLE = True
        logger.info("✅ NLTK punkt tokenizer available")
    except Exception as e:
        logger.warning(f"⚠️ NLTK setup failed: {e}")
        NLTK_AVAILABLE = False
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("⚠️ nltk not available - will use simple text splitting")

try:
    from pydub import AudioSegment, effects
    PYDUB_AVAILABLE = True
    logger.info("✅ pydub available for audio processing")
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("⚠️ pydub not available - will use torchaudio for audio processing")

REPO_ID = "ResembleAI/chatterbox"


def _get_git_sha() -> str:
    """Return current git commit SHA if available, else 'unknown'."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        if sha:
            return sha
    except Exception:
        pass
    # Try common env vars set by CI/CD
    for key in ("GIT_COMMIT", "SOURCE_COMMIT", "COMMIT_SHA", "VERCEL_GIT_COMMIT_SHA"):
        val = os.environ.get(key)
        if val:
            return val
    return "unknown"


def _peak_rms_dbfs_from_np(x: np.ndarray) -> Tuple[float, float]:
    """Calculate peak and RMS levels in dBFS from numpy array."""
    try:
        x = x.astype(np.float64)
        peak = float(np.max(np.abs(x)) + 1e-12)
        rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        return 20.0 * np.log10(peak), 20.0 * np.log10(rms)
    except Exception:
        return float("nan"), float("nan")


def _levels_from_tensor(tensor: torch.Tensor) -> Tuple[float, float]:
    """Calculate peak and RMS levels in dBFS from PyTorch tensor."""
    try:
        if tensor.is_cuda or (hasattr(torch.backends, 'mps') and tensor.device.type == 'mps'):
            tensor = tensor.to('cpu')
        npy = tensor.squeeze(0).detach().numpy().astype(np.float32)
        return _peak_rms_dbfs_from_np(npy)
    except Exception:
        return float("nan"), float("nan")


def _maybe_log_seg_levels(tag: str, seg) -> None:
    """Log audio segment levels if pydub is available."""
    try:
        if PYDUB_AVAILABLE and seg is not None:
            logger.info(f"🔊 {tag}: peak={seg.max_dBFS:.2f} dBFS, avg={seg.dBFS:.2f} dBFS")
    except Exception:
        pass

