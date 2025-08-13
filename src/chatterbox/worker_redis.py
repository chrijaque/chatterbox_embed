import os
import json
import time
import base64
import logging
from typing import Dict, Any

import redis

from .vc import clone_voice  # expected to return dict with paths/ids and handle Firebase via RUNPOD_SECRET_Firebase
from .tts import generate_tts  # expected to return dict with audio path and metadata

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class RedisWorker:
    def __init__(self) -> None:
        self.redis_url = os.getenv("REDIS_URL")
        if not self.redis_url:
            raise RuntimeError("REDIS_URL not set")
        self.stream = os.getenv("REDIS_STREAM_NAME", "runpod:jobs")
        self.group = os.getenv("REDIS_CONSUMER_GROUP", "runpod-consumers")
        self.consumer = os.getenv("REDIS_CONSUMER_NAME", "worker-1")
        self.namespace = os.getenv("REDIS_NAMESPACE", "runpod")
        self.dlp_stream = os.getenv("REDIS_DLP_STREAM", "runpod:dlq")

        use_tls = os.getenv("REDIS_USE_TLS", "true").lower() == "true"
        self.client = redis.Redis.from_url(self.redis_url, decode_responses=True, ssl=use_tls)

        # Ensure consumer group
        try:
            self.client.xgroup_create(name=self.stream, groupname=self.group, id="0-0", mkstream=True)
            logger.info(f"Created consumer group {self.group} on {self.stream}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info("Consumer group already exists")
            else:
                raise

    def _job_key(self, job_id: str) -> str:
        return f"{self.namespace}:job:{job_id}"

    def set_status(self, job_id: str, status: str, **extra: Any) -> None:
        key = self._job_key(job_id)
        mapping = {"status": status, **{k: (v if isinstance(v, str) else json.dumps(v)) for k, v in extra.items()}}
        self.client.hset(key, mapping=mapping)

    def process_message(self, message_id: str, fields: Dict[str, str]) -> None:
        job_id = fields.get("job_id") or message_id
        job_type = fields.get("type")
        payload = {k.split(":", 1)[1]: v for k, v in fields.items() if k.startswith("payload:")}
        self.set_status(job_id, "running")

        try:
            if job_type == "vc":
                # Expect audio_base64, audio_format, name, language, is_kids_voice
                audio_b64 = payload.get("audio_base64", "")
                audio_bytes = base64.b64decode(audio_b64) if audio_b64 else b""
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
                result = generate_tts(
                    voice_id=payload.get("voice_id", ""),
                    text=payload.get("text", ""),
                    profile_base64=payload.get("profile_base64"),
                    language=payload.get("language"),
                    story_type=payload.get("story_type", "user"),
                    is_kids_voice=payload.get("is_kids_voice", "false") == "true",
                    model_type=payload.get("model_type", "chatterbox"),
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


