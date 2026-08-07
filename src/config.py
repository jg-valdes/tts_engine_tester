"""Env + CLI merge, precedence rules. See docs/source.md Configuration.

Precedence for every setting: CLI argument > .env > built-in default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

_DEFAULT_STYLE_PROMPT = (
    "Read as a neutral documentary narrator. American English. "
    "Even pacing, no emotional colour. Do not add, omit or reorder any words."
)


@dataclass
class Settings:
    provider: str = "gemini"

    gemini_api_key: str = ""
    gemini_base_url: str = ""
    gemini_model: str = "gemini-3.1-flash-tts-preview"
    gemini_voice: str = "Charon"
    gemini_style_prompt: str = _DEFAULT_STYLE_PROMPT
    gemini_timing_attempts: int = 2
    gemini_timing_tolerance_ms: int = 150

    azure_speech_key: str = ""
    azure_speech_region: str = "eastus"
    azure_voice: str = "en-US-AvaMultilingualNeural"
    azure_output_format: str = "riff-24khz-16bit-mono-pcm"
    azure_style: str = ""

    sample_rate: int = 24000
    sample_width: int = 2
    channels: int = 1

    words_per_second: float = 2.5
    max_compression_ratio: float = 1.15
    fit_mode: str = "measure"

    azure_usd_per_1m_chars: float = 16.0
    gemini_usd_per_1m_input_tokens: float = 1.0
    gemini_usd_per_1m_audio_tokens: float = 20.0

    output_dir: str = "./out"
    cache_dir: str = "./.cache"
    max_concurrency: int = 2
    log_level: str = "INFO"
    web_db_path: str = "./.webui/tts_harness.db"
    web_host: str = "127.0.0.1"
    web_port: int = 8000


_ENV_PREFIX_TYPES = {f.name: type(f.default) for f in fields(Settings)}


def _coerce(name: str, raw: str):
    target_type = _ENV_PREFIX_TYPES[name]
    if target_type is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return target_type(raw)


def load_settings(overrides: dict | None = None, env_file: str | Path = ".env") -> Settings:
    """Build Settings from defaults, then .env, then explicit overrides (CLI)."""
    load_dotenv(dotenv_path=env_file, override=False)

    values: dict = {}
    for f in fields(Settings):
        env_name = f.name.upper()
        raw = os.environ.get(env_name)
        if raw is not None and raw != "":
            values[f.name] = _coerce(f.name, raw)

    overrides = overrides or {}
    for key, value in overrides.items():
        if value is not None:
            values[key] = value

    return Settings(**values)
