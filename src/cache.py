"""Content-addressed render cache. See docs/source.md Caching."""

from __future__ import annotations

import hashlib
from pathlib import Path


def cache_key(
    provider: str,
    model: str,
    voice: str,
    style_or_ssml: str,
    text: str,
    target_ms: int | None,
    sample_rate: int,
) -> str:
    raw = "|".join(
        [provider, model, voice, style_or_ssml, text, str(target_ms), str(sample_rate)]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Cache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, key: str) -> Path:
        return self.cache_dir / f"{key}.wav"

    def get(self, key: str) -> bytes | None:
        path = self.path_for(key)
        if path.exists():
            return path.read_bytes()
        return None

    def put(self, key: str, wav_bytes: bytes) -> None:
        self.path_for(key).write_bytes(wav_bytes)
