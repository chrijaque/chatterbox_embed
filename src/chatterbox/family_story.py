"""Shared story/clone pipeline for Turbo and Multilingual Chatterbox families.

Official Turbo/MTL classes only expose short-form `generate()`. This mixin restores
the fork's long-form story path: smart chunking, stitcher pause_scale, R2 upload,
and family-specific `.npy` voice profiles that must not be shared across families.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torchaudio

from .chunking import AdvancedTextSanitizer, SmartChunker
from .models.t3.modules.cond_enc import T3Cond
from .stitching import AdvancedStitcher

logger = logging.getLogger(__name__)

PARALINGUISTIC_TAGS = ("laugh", "chuckle", "cough", "sigh", "gasp")


def _to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


def _to_tensor(value, device: str):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).to(device)
    return value


def serialize_conditionals(conds, family: str) -> dict:
    t3 = conds.t3
    return {
        "family": family,
        "t3": {
            "speaker_emb": _to_numpy(t3.speaker_emb),
            "clap_emb": _to_numpy(t3.clap_emb),
            "cond_prompt_speech_tokens": _to_numpy(t3.cond_prompt_speech_tokens),
            "cond_prompt_speech_emb": _to_numpy(t3.cond_prompt_speech_emb),
            "emotion_adv": _to_numpy(t3.emotion_adv),
        },
        "gen": {k: _to_numpy(v) for k, v in (conds.gen or {}).items()},
    }


def deserialize_conditionals(data: dict, device: str, conditionals_cls):
    t3_raw = data.get("t3") or {}
    t3 = T3Cond(
        speaker_emb=_to_tensor(t3_raw.get("speaker_emb"), device),
        clap_emb=_to_tensor(t3_raw.get("clap_emb"), device),
        cond_prompt_speech_tokens=_to_tensor(t3_raw.get("cond_prompt_speech_tokens"), device),
        cond_prompt_speech_emb=_to_tensor(t3_raw.get("cond_prompt_speech_emb"), device),
        emotion_adv=_to_tensor(t3_raw.get("emotion_adv"), device),
    ).to(device=device)
    gen = {}
    for key, value in (data.get("gen") or {}).items():
        gen[key] = _to_tensor(value, device)
    return conditionals_cls(t3, gen).to(device)


def _strip_paralinguistic_tags(text: str) -> str:
    import re
    pattern = r"\[(?:%s)\]" % "|".join(PARALINGUISTIC_TAGS)
    return re.sub(pattern, "", text, flags=re.IGNORECASE)


class FamilyStoryMixin:
    """Adds generate_tts_story / generate_long_text / family .npy profiles."""

    CHATTERBOX_FAMILY = "turbo"

    def _ensure_story_pipeline(self) -> None:
        if getattr(self, "_story_pipeline_ready", False):
            return
        self.smart_chunker = SmartChunker()
        self.text_sanitizer = AdvancedTextSanitizer()
        self.advanced_stitcher = AdvancedStitcher()
        self._story_pipeline_ready = True

    def _conditionals_cls(self):
        if self.CHATTERBOX_FAMILY == "mtl":
            from .mtl_tts import Conditionals
            return Conditionals
        from .tts_turbo import Conditionals
        return Conditionals

    def save_voice_profile(self, audio_file_path: str, save_path: str, exaggeration: float = 0.5) -> None:
        prepare = getattr(self, "prepare_conditionals")
        if self.CHATTERBOX_FAMILY == "turbo":
            prepare(audio_file_path, exaggeration=exaggeration)
        else:
            prepare(audio_file_path, exaggeration=exaggeration)
        np.save(save_path, serialize_conditionals(self.conds, self.CHATTERBOX_FAMILY))
        logger.info("Saved %s voice profile to %s", self.CHATTERBOX_FAMILY, save_path)

    def load_voice_profile(self, path: str):
        data = np.load(path, allow_pickle=True).item()
        if not isinstance(data, dict):
            raise ValueError(f"Invalid voice profile at {path}")
        family = data.get("family")
        if family is None and "ve_embedding" in data:
            raise ValueError(
                "This .npy was cloned for original Chatterbox and cannot drive Turbo or MTL. Re-clone the voice."
            )
        if family and family != self.CHATTERBOX_FAMILY:
            raise ValueError(
                f"Voice profile family '{family}' cannot be used with '{self.CHATTERBOX_FAMILY}'."
            )
        self.conds = deserialize_conditionals(data, self.device, self._conditionals_cls())
        return self.conds

    def upload_to_storage(self, data: bytes, destination_blob_name: str, content_type: str = "application/octet-stream", metadata: dict = None):
        from .tts import ChatterboxTTS
        return ChatterboxTTS.upload_to_storage(self, data, destination_blob_name, content_type, metadata)

    def generate_tts_story(self, *args, **kwargs):
        language = kwargs.get("language", "en")
        self._story_language = language
        from .tts import ChatterboxTTS
        return ChatterboxTTS.generate_tts_story(self, *args, **kwargs)

    def generate_long_text(
        self,
        text: str,
        voice_profile_path: str,
        output_path: str,
        max_chars: int = 500,
        pause_ms: int = 100,
        temperature: float = 0.6,
        exaggeration: float = 0.5,
        cfg_weight: float = 0.5,
        pause_scale: float = 1.0,
        *,
        adaptive_voice_param_blend: float = 0.2,
        language: Optional[str] = None,
    ) -> Tuple[torch.Tensor, int, Dict]:
        del pause_ms, adaptive_voice_param_blend
        self._ensure_story_pipeline()
        language = language or getattr(self, "_story_language", "en")
        logger.info("Starting %s TTS for %s characters (language=%s)", self.CHATTERBOX_FAMILY, len(text), language)

        self.load_voice_profile(voice_profile_path)
        try:
            self.advanced_stitcher.global_pause_factor = max(0.5, min(2.0, float(pause_scale)))
        except Exception:
            logger.warning("Failed to apply pause_scale; using default")

        prepared = text
        if self.CHATTERBOX_FAMILY == "mtl":
            prepared = _strip_paralinguistic_tags(prepared)
        sanitized = self.text_sanitizer.deep_clean(prepared)
        target_chars = int(max_chars * 0.8)
        chunk_infos = self.smart_chunker.smart_chunk(sanitized, target_chars, max_chars)
        if not chunk_infos:
            raise RuntimeError("Failed to chunk text for TTS")

        wav_paths = []
        for chunk_info in chunk_infos:
            wav = self._generate_chunk_audio(
                chunk_info.text,
                language=language,
                temperature=temperature,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            torchaudio.save(temp_wav_path, wav, self.sr)
            wav_paths.append(temp_wav_path)

        audio_tensor, sample_rate, total_duration = self.advanced_stitcher.advanced_stitch(
            wav_paths, chunk_infos, output_path
        )
        try:
            final_np = audio_tensor.squeeze(0).detach().cpu().numpy()
            if hasattr(self, "watermarker") and self.watermarker is not None:
                final_np = self.watermarker.apply_watermark(final_np, sample_rate=sample_rate)
            audio_tensor = torch.from_numpy(final_np).unsqueeze(0)
        except Exception as e:
            logger.warning("Failed to apply final watermark: %s", e)

        for path in wav_paths:
            try:
                os.remove(path)
            except OSError:
                pass

        metadata = {
            "duration_sec": total_duration,
            "output_path": output_path,
            "successful_chunks": len(wav_paths),
            "family": self.CHATTERBOX_FAMILY,
            "language": language,
        }
        return audio_tensor, sample_rate, metadata

    def _generate_chunk_audio(
        self,
        text: str,
        *,
        language: str,
        temperature: float,
        exaggeration: float,
        cfg_weight: float,
    ) -> torch.Tensor:
        generate_kwargs = {
            "temperature": temperature,
            "exaggeration": exaggeration,
            "cfg_weight": cfg_weight,
            "apply_watermark": False,
        }
        if self.CHATTERBOX_FAMILY == "mtl":
            generate_kwargs["language_id"] = language
            logger.info("MTL generate language_id=%s", language)
        return self.generate(text, **generate_kwargs)
