"""Gemini TTS provider. See docs/source.md Provider specifics > Gemini."""

from __future__ import annotations

import base64
import json
import time

from google import genai
from google.genai import types

from ..audio import strip_riff_header
from .base import SynthesisResult, Voice

_MAX_RETRIES = 5
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Google does not provide a voice-list endpoint. This snapshot is the complete
# prebuilt voice set documented at https://ai.google.dev/gemini-api/docs/speech-generation
# and was last verified on 2026-08-07.
_GEMINI_VOICES = (
    ("Zephyr", "Bright"),
    ("Puck", "Upbeat"),
    ("Charon", "Informative"),
    ("Kore", "Firm"),
    ("Fenrir", "Excitable"),
    ("Leda", "Youthful"),
    ("Orus", "Firm"),
    ("Aoede", "Breezy"),
    ("Callirrhoe", "Easy-going"),
    ("Autonoe", "Bright"),
    ("Enceladus", "Breathy"),
    ("Iapetus", "Clear"),
    ("Umbriel", "Easy-going"),
    ("Algieba", "Smooth"),
    ("Despina", "Smooth"),
    ("Erinome", "Clear"),
    ("Algenib", "Gravelly"),
    ("Rasalgethi", "Informative"),
    ("Laomedeia", "Upbeat"),
    ("Achernar", "Soft"),
    ("Alnilam", "Firm"),
    ("Schedar", "Even"),
    ("Gacrux", "Mature"),
    ("Pulcherrima", "Forward"),
    ("Achird", "Friendly"),
    ("Zubenelgenubi", "Casual"),
    ("Vindemiatrix", "Gentle"),
    ("Sadachbia", "Lively"),
    ("Sadaltager", "Knowledgeable"),
    ("Sulafat", "Warm"),
)


class GeminiProvider:
    name = "gemini"

    def __init__(self, settings):
        self.settings = settings
        self._client = None

    @property
    def client(self):
        # Lazy: building the client eagerly would require an API key even
        # for --dry-run / build_payload, which make no network call.
        if self._client is None:
            client_kwargs: dict = {"api_key": self.settings.gemini_api_key}
            if self.settings.gemini_base_url:
                client_kwargs["http_options"] = types.HttpOptions(base_url=self.settings.gemini_base_url)
            self._client = genai.Client(**client_kwargs)
        return self._client

    def build_payload(self, text: str, target_ms: int | None = None) -> str:
        return json.dumps(
            self._request_body(text, target_ms=target_ms), ensure_ascii=False, sort_keys=True
        )

    def _request_body(
        self,
        text: str,
        *,
        target_ms: int | None = None,
        previous_ms: int | None = None,
    ) -> dict:
        # Keep direction and transcript in distinct content blocks. This makes
        # their roles unambiguous and reduces instructions leaking into speech.
        direction = self.settings.gemini_style_prompt
        if target_ms is not None:
            target_seconds = target_ms / 1000
            words_per_second = len(text.split()) / target_seconds if target_seconds else 0
            timing = (
                f"\n\nTiming is important: speak the complete transcript in approximately "
                f"{target_seconds:.2f} seconds, at about {words_per_second:.2f} words per second. "
                "Use continuous, even pacing with minimal pauses. Do not omit, add, summarize, "
                "or reorder words."
            )
            if previous_ms is not None and target_ms > 0:
                pace_factor = previous_ms / target_ms
                if pace_factor >= 1:
                    correction = f"speak about {pace_factor:.2f} times faster than that render"
                else:
                    correction = (
                        f"speak about {1 / pace_factor:.2f} times slower than that render"
                        if pace_factor > 0
                        else "use the requested target pace"
                    )
                timing += (
                    f" A previous render took {previous_ms / 1000:.2f} seconds; {correction}."
                )
            direction += timing

        return {
            "model": self.settings.gemini_model,
            "store": False,
            "input": [
                {"type": "text", "text": direction},
                {"type": "text", "text": text},
            ],
            "response_format": {"type": "audio"},
            "generation_config": {
                "speech_config": [{"voice": self.settings.gemini_voice}],
            },
        }

    def synthesize(self, text: str, *, target_ms: int | None = None) -> SynthesisResult:
        attempts = 1 if target_ms is None else max(1, self.settings.gemini_timing_attempts)
        tolerance_ms = max(0, self.settings.gemini_timing_tolerance_ms)
        previous_ms = None
        best: SynthesisResult | None = None
        total_input_tokens = 0
        total_output_audio_tokens = 0

        for _ in range(attempts):
            request_body = self._request_body(
                text, target_ms=target_ms, previous_ms=previous_ms
            )
            response = self._create_interaction(request_body)
            pcm = self._extract_pcm(response)
            self._validate_audio_metadata(response)
            input_tokens, output_audio_tokens = self._extract_usage(response)
            total_input_tokens += input_tokens
            total_output_audio_tokens += output_audio_tokens
            duration_ms = (
                len(pcm)
                * 1000
                // (
                    self.settings.sample_rate
                    * self.settings.sample_width
                    * self.settings.channels
                )
            )
            result = SynthesisResult(
                pcm=pcm,
                duration_ms=duration_ms,
                duration_constrained=False,
                billed_units={},
                request_payload=json.dumps(request_body, ensure_ascii=False, sort_keys=True),
            )
            if best is None or (
                target_ms is not None
                and abs(duration_ms - target_ms) < abs(best.duration_ms - target_ms)
            ):
                best = result
            if target_ms is None or abs(duration_ms - target_ms) <= tolerance_ms:
                break
            previous_ms = duration_ms

        if best is None:  # pragma: no cover - attempts is always at least one
            raise RuntimeError("Gemini synthesis made no attempts")

        best.billed_units = {
            "input_tokens": total_input_tokens,
            "output_audio_tokens": total_output_audio_tokens,
        }
        return best

    def _create_interaction(self, request_body: dict):
        for attempt in range(_MAX_RETRIES):
            try:
                return self.client.interactions.create(**request_body)
            except Exception as exc:  # google-genai raises provider-specific errors
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if attempt == _MAX_RETRIES - 1 or (status is not None and status not in _RETRYABLE_STATUS):
                    raise
                time.sleep(2**attempt)

    @staticmethod
    def _extract_pcm(response) -> bytes:
        audio = GeminiProvider._value(response, "output_audio")
        if audio is None:
            status = GeminiProvider._value(response, "status", "unknown")
            raise ValueError(f"Gemini interaction returned no output audio (status={status})")

        data = GeminiProvider._value(audio, "data")
        if not data:
            raise ValueError("Gemini interaction output audio contained no inline data")

        if isinstance(data, str):
            try:
                pcm = base64.b64decode(data, validate=True)
            except (ValueError, TypeError) as exc:
                raise ValueError("Gemini interaction returned invalid base64 audio data") from exc
        else:
            pcm = bytes(data)

        # The documented response is headerless PCM. Be defensive if a proxy or
        # future response uses RIFF/WAV so downstream always receives raw PCM.
        return strip_riff_header(pcm) if pcm.startswith(b"RIFF") else pcm

    def _validate_audio_metadata(self, response) -> None:
        audio = self._value(response, "output_audio")
        sample_rate = self._value(audio, "sample_rate")
        channels = self._value(audio, "channels")
        if sample_rate is not None and sample_rate != self.settings.sample_rate:
            raise ValueError(
                f"Gemini returned {sample_rate} Hz audio; configured SAMPLE_RATE is "
                f"{self.settings.sample_rate} Hz"
            )
        if channels is not None and channels != self.settings.channels:
            raise ValueError(
                f"Gemini returned {channels} audio channels; configured CHANNELS is "
                f"{self.settings.channels}"
            )

    @staticmethod
    def _extract_usage(response) -> tuple[int, int]:
        usage = GeminiProvider._value(response, "usage")
        if usage is None:
            return 0, 0

        input_tokens = GeminiProvider._value(usage, "total_input_tokens", 0) or 0
        output_audio_tokens = 0
        modality_usage = GeminiProvider._value(usage, "output_tokens_by_modality", []) or []
        for item in modality_usage:
            if GeminiProvider._value(item, "modality") == "audio":
                output_audio_tokens += GeminiProvider._value(item, "tokens", 0) or 0
        if not modality_usage:
            output_audio_tokens = GeminiProvider._value(usage, "total_output_tokens", 0) or 0
        return input_tokens, output_audio_tokens

    @staticmethod
    def _value(obj, name: str, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def list_voices(self) -> list[Voice]:
        return [Voice(name=name, description=description) for name, description in _GEMINI_VOICES]
