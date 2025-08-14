import os
import json
import time
import base64
import logging
from typing import Dict, Any

import redis

from .vc import clone_voice  # expected to write Firebase and Voice Profile doc
from .tts import ChatterboxTTS  # use class to call generate_tts_story with user_id/story_id

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RedisWorker:
    def __init__(self) -> None:
        self.redis_url = os.getenv("REDIS_URL")
        if not self.redis_url:
            raise RuntimeError("REDIS_URL not set")
        # Worker mode determines the default stream if REDIS_STREAM_NAME is not explicitly set
        self.mode = os.getenv("WORKER_MODE", "tts").lower()
        # Allow per-mode stream overrides
        stream_tts = os.getenv("REDIS_STREAM_NAME_TTS", "runpod:jobs:tts")
        stream_vc = os.getenv("REDIS_STREAM_NAME_VC", "runpod:jobs:vc")
        # Final stream resolution: explicit REDIS_STREAM_NAME wins; otherwise choose by mode
        inferred_stream = stream_tts if self.mode == "tts" else stream_vc
        self.stream = os.getenv("REDIS_STREAM_NAME", inferred_stream)
        # Allow per-mode consumer group/name via *_TTS and *_VC (helps when secret names must be unique)
        group_tts = os.getenv("REDIS_CONSUMER_GROUP_TTS")
        group_vc = os.getenv("REDIS_CONSUMER_GROUP_VC")
        name_tts = os.getenv("REDIS_CONSUMER_NAME_TTS")
        name_vc = os.getenv("REDIS_CONSUMER_NAME_VC")

        inferred_group = (group_tts if self.mode == "tts" else group_vc) or (
            "tts-consumers" if self.mode == "tts" else "vc-consumers"
        )
        inferred_name = (name_tts if self.mode == "tts" else name_vc) or f"{self.mode}-worker-1"

        self.group = os.getenv("REDIS_CONSUMER_GROUP", inferred_group)
        self.consumer = os.getenv("REDIS_CONSUMER_NAME", inferred_name)
        self.namespace = os.getenv("REDIS_NAMESPACE", "runpod")
        self.dlp_stream = os.getenv("REDIS_DLP_STREAM", "runpod:dlq")

        # Rely on rediss:// scheme in REDIS_URL to enable TLS; avoid passing ssl kwarg for wider redis-py compatibility
        self.client = redis.Redis.from_url(self.redis_url, decode_responses=True)

        # Ensure consumer group
        try:
            self.client.xgroup_create(name=self.stream, groupname=self.group, id="0-0", mkstream=True)
            logger.info(f"Created consumer group {self.group} on {self.stream} (mode={self.mode})")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group already exists on {self.stream} (mode={self.mode})")
            else:
                raise

        # Lazy-initialized TTS engine
        self._tts = None

    def _get_tts(self):
        if self._tts is None:
            self._tts = ChatterboxTTS.from_pretrained("cpu")
        return self._tts

    def _job_key(self, job_id: str) -> str:
        return f"{self.namespace}:job:{job_id}"

    def set_status(self, job_id: str, status: str, **extra: Any) -> None:
        key = self._job_key(job_id)
        mapping = {"status": status, **{k: (v if isinstance(v, str) else json.dumps(v)) for k, v in extra.items()}}
        self.client.hset(key, mapping=mapping)

    def process_message(self, message_id: str, fields: Dict[str, str]) -> None:
        job_id = fields.get("job_id") or message_id
        job_type = fields.get("type")
        # Support both flattened payload:k=v and a single JSON 'payload' field
        payload: Dict[str, Any] = {k.split(":", 1)[1]: v for k, v in fields.items() if k.startswith("payload:")}
        if not payload and "payload" in fields:
            try:
                payload_json = json.loads(fields["payload"]) if isinstance(fields["payload"], str) else fields["payload"]
                if isinstance(payload_json, dict):
                    payload.update(payload_json)
            except Exception:
                logger.warning("Failed to parse JSON payload field; continuing with flattened keys only")
        logger.info(f"Job {job_id} type={job_type} payloadKeys={list(payload.keys())}")
        self.set_status(job_id, "running")

        try:
            if job_type == "vc":
                # Expect audio_base64, audio_format, name, language, is_kids_voice
                audio_b64 = payload.get("audio_base64", "")
                audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
                # Sanitized metadata logging (no raw base64 or PII beyond identifiers)
                logger.info(
                    "VC payload snapshot: "
                    f"user_id={payload.get('user_id')} "
                    f"profile_id={payload.get('profile_id')} "
                    f"name={payload.get('name')} "
                    f"language={payload.get('language')} "
                    f"is_kids_voice={payload.get('is_kids_voice')} "
                    f"model_type={payload.get('model_type', 'chatterbox')} "
                    f"audio_format={payload.get('audio_format', 'wav')} "
                    f"audio_b64_len={len(audio_b64)}"
                )
                result = clone_voice(
                    name=payload.get("name", "voice"),
                    audio_bytes=audio_bytes,
                    audio_format=payload.get("audio_format", "wav"),
                    language=payload.get("language", "en"),
                    is_kids_voice=payload.get("is_kids_voice", "false") == "true",
                    model_type=payload.get("model_type", "chatterbox"),
                    user_id=payload.get("user_id", ""),
                    profile_id=payload.get("profile_id") or None,
                )
                self.set_status(job_id, "completed", **result)
            elif job_type == "tts":
                tts = self._get_tts()
                # Sanitized metadata logging for TTS
                text = payload.get("text", "")
                profile_b64 = payload.get("profile_base64") or ""
                profile_path = payload.get("profile_path") or ""
                logger.info(
                    "TTS payload snapshot: "
                    f"user_id={payload.get('user_id', '')} "
                    f"story_id={payload.get('story_id', '')} "
                    f"voice_id={payload.get('voice_id', '')} "
                    f"language={payload.get('language') or 'en'} "
                    f"story_type={payload.get('story_type', 'user')} "
                    f"is_kids_voice={payload.get('is_kids_voice')} "
                    f"model_type={payload.get('model_type', 'chatterbox')} "
                    f"text_len={len(text)} "
                    f"profile_base64_len={len(profile_b64)} "
                    f"profile_path_present={bool(profile_path)}"
                )
                result = tts.generate_tts_story(
                    text=text,
                    voice_id=payload.get("voice_id", ""),
                    profile_base64=profile_b64,
                    language=payload.get("language") or "en",
                    story_type=payload.get("story_type", "user"),
                    is_kids_voice=payload.get("is_kids_voice", "false") == "true",
                    metadata={"model_type": payload.get("model_type", "chatterbox")},
                    user_id=payload.get("user_id", ""),
                    story_id=payload.get("story_id", ""),
                )
                self.set_status(job_id, "completed", **result)
            else:
                self.set_status(job_id, "failed", error="unknown job type")
        except Exception as e:
            logger.exception("Job failed")
            self.set_status(job_id, "failed", error=str(e))
            # Dead letter
            self.client.xadd(self.dlp_stream, {**fields, "error": str(e)})

    def run_forever(self) -> None:
        logger.info("Redis worker started")
        while True:
            try:
                # Read one message per iteration
                entries = self.client.xreadgroup(self.group, self.consumer, {self.stream: ">"}, count=1, block=5000)
                if not entries:
                    continue
                for stream, messages in entries:
                    for message_id, fields in messages:
                        self.process_message(message_id, fields)
                        self.client.xack(self.stream, self.group, message_id)
            except Exception:
                logger.exception("Worker loop error")
                time.sleep(2)


if __name__ == "__main__":
    RedisWorker().run_forever()


