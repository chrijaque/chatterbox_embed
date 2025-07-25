# Modified from CosyVoice https://github.com/FunAudioLLM/CosyVoice
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

import numpy as np
import torch
import torchaudio as ta
from functools import lru_cache
from typing import Optional

from ..s3tokenizer import S3_SR, SPEECH_VOCAB_SIZE, S3Tokenizer
from .const import S3GEN_SR
from .flow import CausalMaskedDiffWithXvec
from .xvector import CAMPPlus
from .utils.mel import mel_spectrogram
from .f0_predictor import ConvRNNF0Predictor
from .hifigan import HiFTGenerator
from .transformer.upsample_encoder import UpsampleConformerEncoder
from .flow_matching import CausalConditionalCFM
from .decoder import ConditionalDecoder
from .configs import CFM_PARAMS


def drop_invalid_tokens(x):
    assert len(x.shape) <= 2 and x.shape[0] == 1, "only batch size of one allowed for now"
    return x[x < SPEECH_VOCAB_SIZE]


# TODO: global resampler cache
@lru_cache(100)
def get_resampler(src_sr, dst_sr, device):
    return ta.transforms.Resample(src_sr, dst_sr).to(device)


class S3Token2Mel(torch.nn.Module):
    """
    CosyVoice2's CFM decoder maps S3 speech tokens to mel-spectrograms.

    TODO: make these modules configurable?
    """
    def __init__(self):
        super().__init__()
        self.tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
        self.mel_extractor = mel_spectrogram # TODO: make it a torch module?
        self.speaker_encoder = CAMPPlus()  # use default args

        encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer='linear',
            pos_enc_layer_type='rel_pos_espnet',
            selfattention_layer_type='rel_selfattn',
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )

        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn='gelu',
        )
        cfm_params = CFM_PARAMS
        decoder = CausalConditionalCFM(
            spk_emb_dim=80,
            cfm_params=cfm_params,
            estimator=estimator,
        )

        self.flow = CausalMaskedDiffWithXvec(
            encoder=encoder,
            decoder=decoder
        )

        self.resamplers = {}

    @property
    def device(self):
        params = self.tokenizer.parameters()
        return next(params).device

    @torch.inference_mode()
    def save_voice_clone(self, ref_wav: torch.Tensor, ref_sr: int, save_path: str):
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()
        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)

        device = self.device
        ref_wav = ref_wav.to(device)
        ref_wav_16 = get_resampler(ref_sr, S3_SR, device)(ref_wav).to(device)
        
        embedding = self.speaker_encoder.inference(ref_wav_16)
        np.save(save_path, embedding.detach().cpu().numpy())

    @torch.inference_mode()
    def save_voice_profile(self, ref_wav: torch.Tensor, ref_sr: int, save_path: str):
        """Save a complete voice profile including embedding, prompt features, and tokens for more accurate TTS"""
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()
        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)

        device = self.device
        ref_wav = ref_wav.to(device)
        
        # Get the full reference dictionary using embed_ref
        ref_dict = self.embed_ref(ref_wav, ref_sr, device=device)
        
        # Create and save the complete voice profile
        profile = VoiceProfile(
            embedding=ref_dict["embedding"],
            prompt_feat=ref_dict["prompt_feat"],
            prompt_feat_len=ref_dict.get("prompt_feat_len"),
            prompt_token=ref_dict["prompt_token"],
            prompt_token_len=ref_dict["prompt_token_len"],
        )
        profile.save(save_path)

    @torch.inference_mode()
    def load_voice_clone(self, embedding_path: str) -> torch.Tensor:
        emb = np.load(embedding_path)
        return torch.from_numpy(emb).to(self.device)

    @torch.inference_mode()
    def load_voice_profile(self, profile_path: str) -> "VoiceProfile":
        """Load a complete voice profile from file"""
        return VoiceProfile.load(profile_path, device=self.device)

    def embed_ref(
        self,
        ref_wav: torch.Tensor,
        ref_sr: int,
        device="auto",
        ref_fade_out=True,
    ):
        device = self.device if device == "auto" else device
        if isinstance(ref_wav, np.ndarray):
            ref_wav = torch.from_numpy(ref_wav).float()

        if ref_wav.device != device:
            ref_wav = ref_wav.to(device)

        if len(ref_wav.shape) == 1:
            ref_wav = ref_wav.unsqueeze(0)  # (B, L)

        if ref_wav.size(1) > 10 * ref_sr:
            print("WARNING: cosydec received ref longer than 10s")

        ref_wav_24 = ref_wav
        if ref_sr != S3GEN_SR:
            ref_wav_24 = get_resampler(ref_sr, S3GEN_SR, device)(ref_wav)

        ref_mels_24 = self.mel_extractor(ref_wav_24).transpose(1, 2).to(device)
        ref_mels_24_len = None

        # Resample to 16kHz
        ref_wav_16 = get_resampler(ref_sr, S3_SR, device)(ref_wav).to(device)

        # Speaker embedding
        ref_x_vector = self.speaker_encoder.inference(ref_wav_16)

        # Tokenize 16khz reference
        ref_speech_tokens, ref_speech_token_lens = self.tokenizer(ref_wav_16)

        # Make sure mel_len = 2 * stoken_len (happens when the input is not padded to multiple of 40ms)
        if ref_mels_24.shape[1] != 2 * ref_speech_tokens.shape[1]:
            logging.warning(
                "Reference mel length is not equal to 2 * reference token length.\n"
            )
            ref_speech_tokens = ref_speech_tokens[:, :ref_mels_24.shape[1] // 2]
            ref_speech_token_lens[0] = ref_speech_tokens.shape[1]

        return dict(
            prompt_token=ref_speech_tokens.to(device),
            prompt_token_len=ref_speech_token_lens,
            prompt_feat=ref_mels_24,
            prompt_feat_len=ref_mels_24_len,
            embedding=ref_x_vector,
        )

    def forward(
        self,
        speech_tokens: torch.LongTensor,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
    ):
        """
        Generate waveforms from S3 speech tokens and a reference waveform, which the speaker timbre is inferred from.

        NOTE:
        - The speaker encoder accepts 16 kHz waveform.
        - S3TokenizerV2 accepts 16 kHz waveform.
        - The mel-spectrogram for the reference assumes 24 kHz input signal.
        - This function is designed for batch_size=1 only.

        Args
        ----
        - `speech_tokens`: S3 speech tokens [B=1, T]
        - `ref_wav`: reference waveform (`torch.Tensor` with shape=[B=1, T])
        - `ref_sr`: reference sample rate
        - `finalize`: whether streaming is finished or not. Note that if False, the last 3 tokens will be ignored.
        """
        assert (ref_wav is None) ^ (ref_dict is None), f"Must provide exactly one of ref_wav or ref_dict (got {ref_wav} and {ref_dict})"

        if ref_dict is None:
            ref_dict = self.embed_ref(ref_wav, ref_sr)
        else:
            # type/device casting (all values will be numpy if it's from a prod API call)
            for rk in list(ref_dict):
                if isinstance(ref_dict[rk], np.ndarray):
                    ref_dict[rk] = torch.from_numpy(ref_dict[rk])
                if torch.is_tensor(ref_dict[rk]):
                    ref_dict[rk] = ref_dict[rk].to(self.device)

        if len(speech_tokens.shape) == 1:
            speech_tokens = speech_tokens.unsqueeze(0)

        # assert speech_tokens.shape[0] == 1, "only batch size of one allowed for now"
        speech_token_lens = torch.LongTensor([speech_tokens.size(1)]).to(self.device)

        output_mels, _ = self.flow.inference(
            token=speech_tokens,
            token_len=speech_token_lens,
            finalize=finalize,
            **ref_dict,
        )
        return output_mels


class S3Token2Wav(S3Token2Mel):
    """
    The decoder of CosyVoice2 is a concat of token-to-mel (CFM) and a mel-to-waveform (HiFiGAN) modules.

    TODO: make these modules configurable?
    """

    def __init__(self):
        super().__init__()

        f0_predictor = ConvRNNF0Predictor()
        self.mel2wav = HiFTGenerator(
            sampling_rate=S3GEN_SR,
            upsample_rates=[8, 5, 3],
            upsample_kernel_sizes=[16, 11, 7],
            source_resblock_kernel_sizes=[7, 7, 11],
            source_resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            f0_predictor=f0_predictor,
        )

        # silence out a few ms and fade audio in to reduce artifacts
        n_trim = S3GEN_SR // 50  # 20ms = half of a frame
        trim_fade = torch.zeros(2 * n_trim)
        trim_fade[n_trim:] = (torch.cos(torch.linspace(torch.pi, 0, n_trim)) + 1) / 2
        self.register_buffer("trim_fade", trim_fade, persistent=False) # (buffers get automatic device casting)

    def forward(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor],
        ref_sr: Optional[int],
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False
    ):
        output_mels = super().forward(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize)

        # TODO jrm: ignoring the speed control (mel interpolation) and the HiFTGAN caching mechanisms for now.
        hift_cache_source = torch.zeros(1, 1, 0).to(self.device)

        output_wavs, *_ = self.mel2wav.inference(speech_feat=output_mels, cache_source=hift_cache_source)

        if not self.training:
            # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
            output_wavs[:, :len(self.trim_fade)] *= self.trim_fade

        return output_wavs

    @torch.inference_mode()
    def flow_inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        finalize: bool = False,
    ):
        return super().forward(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize)

    @torch.inference_mode()
    def hift_inference(self, speech_feat, cache_source: torch.Tensor = None):
        if cache_source is None:
            cache_source = torch.zeros(1, 1, 0).to(self.device)
        return self.mel2wav.inference(speech_feat=speech_feat, cache_source=cache_source)

    @torch.inference_mode()
    def inference(
        self,
        speech_tokens,
        # locally-computed ref embedding (mutex with ref_dict)
        ref_wav: Optional[torch.Tensor] = None,
        ref_sr: Optional[int] = None,
        # pre-computed ref embedding (prod API)
        ref_dict: Optional[dict] = None,
        cache_source: torch.Tensor = None, # NOTE: this arg is for streaming, it can probably be removed here
        finalize: bool = True,
    ):
        output_mels = self.flow_inference(speech_tokens, ref_wav=ref_wav, ref_sr=ref_sr, ref_dict=ref_dict, finalize=finalize)
        output_wavs, output_sources = self.hift_inference(output_mels, cache_source)

        # NOTE: ad-hoc method to reduce "spillover" from the reference clip.
        output_wavs[:, :len(self.trim_fade)] *= self.trim_fade


        return output_wavs, output_sources


    # ---------------------------------------------------------------------
    # NEW ‼️  High‑level helper: raw text  ➜  waveform  (profile‑based)
    # ---------------------------------------------------------------------
    @torch.inference_mode()
    def inference_from_text(
        self,
        text: str,
        ref_dict: dict,
        *,
        finalize: bool = True,
    ) -> torch.Tensor:
        """
        Convenience wrapper that lets callers supply **raw text** plus an
        in-memory voice profile (``ref_dict``) instead of an audio prompt.

        The heavy lifting (text → S3 speech-token IDs) must be performed by
        a *separate* language model that is attached to this class instance
        as ``self.text_encoder``.  That object **must** expose
        ``encode(text: str) -> torch.LongTensor`` compatible with the
        decoder's vocabulary.

        Example::

            model.text_encoder = cosy_lm          # attaches the encoder
            wav = model.inference_from_text(
                    "Hello world!",
                    ref_dict=profile_dict,
                 )

        If ``self.text_encoder`` has not been attached, this method raises
        ``RuntimeError`` so upstream code can fall back to another path.

        Returns
        -------
        torch.Tensor
            A (1, samples) waveform at ``S3GEN_SR`` ready for saving.
        """
        if not hasattr(self, "text_encoder"):
            raise RuntimeError(
                "S3Token2Wav.inference_from_text: no `text_encoder` attached "
                "(expected an object with `.encode(text) -> LongTensor`)."
            )

        # 1) Text → discrete S3 speech tokens
        # ---------------------------------------------------------
        # Accept either:
        #   • an object exposing `.encode(text) -> tensor`
        #   • a callable that can be invoked directly
        # ---------------------------------------------------------
        if hasattr(self.text_encoder, "encode"):
            speech_tokens = self.text_encoder.encode(text)
        elif callable(self.text_encoder):
            speech_tokens = self.text_encoder(text)
        else:
            raise RuntimeError(
                "S3Token2Wav.inference_from_text: `text_encoder` has neither "
                "an `.encode()` method nor is it callable.  "
                f"Got type {type(self.text_encoder)} with attrs "
                f"{dir(self.text_encoder)[:20]} …"
            )

        if not torch.is_tensor(speech_tokens):
            speech_tokens = torch.tensor(speech_tokens, dtype=torch.long)
        speech_tokens = speech_tokens.to(self.device)

        # 2) Re‑use the existing token‑level inference path
        output_wavs, _ = self.inference(
            speech_tokens=speech_tokens,
            ref_dict=ref_dict,
            finalize=finalize,
        )
        return output_wavs.squeeze(0)  # (samples,)  – drop batch dim for convenience

# ================== VoiceProfile ==================
class VoiceProfile:
    """
    Represents a full voice profile including the speaker embedding and
    reference audio features for cloning or TTS generation.
    """

    def __init__(
        self,
        embedding: torch.Tensor,
        prompt_feat: Optional[torch.Tensor] = None,
        prompt_feat_len: Optional[int] = None,
        prompt_token: Optional[torch.Tensor] = None,
        prompt_token_len: Optional[torch.Tensor] = None,
    ):
        self.embedding = embedding
        self.prompt_feat = prompt_feat
        self.prompt_feat_len = prompt_feat_len
        self.prompt_token = prompt_token
        self.prompt_token_len = prompt_token_len

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "VoiceProfile":
        data = np.load(path, allow_pickle=True).item()
        return cls(
            embedding=torch.tensor(data["embedding"]).to(device),
            prompt_feat=torch.tensor(data["prompt_feat"]).to(device) if "prompt_feat" in data else None,
            prompt_feat_len=data.get("prompt_feat_len"),
            prompt_token=torch.tensor(data["prompt_token"]).to(device) if "prompt_token" in data else None,
            prompt_token_len=torch.tensor(data["prompt_token_len"]).to(device) if "prompt_token_len" in data else None,
        )

    def save(self, path: str):
        data = {
            "embedding": self.embedding.cpu().numpy(),
        }
        if self.prompt_feat is not None:
            data["prompt_feat"] = self.prompt_feat.cpu().numpy()
        if self.prompt_feat_len is not None:
            data["prompt_feat_len"] = self.prompt_feat_len
        if self.prompt_token is not None:
            data["prompt_token"] = self.prompt_token.cpu().numpy()
        if self.prompt_token_len is not None:
            data["prompt_token_len"] = self.prompt_token_len.cpu().numpy()
        np.save(path, data)
