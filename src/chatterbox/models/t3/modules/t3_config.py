from ..llama_configs import LLAMA_CONFIGS


class T3Config:
    """Instance config with class-level defaults so original English T3() still works."""

    start_text_token = 255
    stop_text_token = 0
    text_tokens_dict_size = 704
    max_text_tokens = 2048

    start_speech_token = 6561
    stop_speech_token = 6562
    speech_tokens_dict_size = 8194
    max_speech_tokens = 4096

    llama_config_name = "Llama_520M"
    input_pos_emb = "learned"
    speech_cond_prompt_len = 150

    encoder_type = "voice_encoder"
    speaker_embed_size = 256
    use_perceiver_resampler = True
    emotion_adv = True

    def __init__(self, text_tokens_dict_size=704):
        self.start_text_token = type(self).start_text_token
        self.stop_text_token = type(self).stop_text_token
        self.text_tokens_dict_size = text_tokens_dict_size
        self.max_text_tokens = type(self).max_text_tokens

        self.start_speech_token = type(self).start_speech_token
        self.stop_speech_token = type(self).stop_speech_token
        self.speech_tokens_dict_size = type(self).speech_tokens_dict_size
        self.max_speech_tokens = type(self).max_speech_tokens

        self.llama_config_name = type(self).llama_config_name
        self.input_pos_emb = type(self).input_pos_emb
        self.speech_cond_prompt_len = type(self).speech_cond_prompt_len

        self.encoder_type = type(self).encoder_type
        self.speaker_embed_size = type(self).speaker_embed_size
        self.use_perceiver_resampler = type(self).use_perceiver_resampler
        self.emotion_adv = type(self).emotion_adv

    @property
    def n_channels(self):
        return LLAMA_CONFIGS[self.llama_config_name]["hidden_size"]

    @property
    def is_multilingual(self):
        return self.text_tokens_dict_size == 2454

    @classmethod
    def english_only(cls):
        return cls(text_tokens_dict_size=704)

    @classmethod
    def multilingual(cls):
        return cls(text_tokens_dict_size=2454)
