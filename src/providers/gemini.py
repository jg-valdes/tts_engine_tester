"""Gemini TTS provider. See docs/source.md Provider specifics > Gemini.

NOTE: implemented against the documented google-genai SDK shape for
audio-output models. `gemini-3.1-flash-tts-preview` is a preview model and
no API key was available in this environment to run Step 0's live
verification call — re-run that check against real credentials before
trusting this against production traffic, and adjust `_extract_pcm` /
`_extract_usage` if the live response shape differs.
"""

from __future__ import annotations

import time

from google import genai
from google.genai import types

from .base import SynthesisResult, Voice

_MAX_RETRIES = 5
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}


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
        # target_ms is intentionally unused: Gemini has no duration control.
        return f"[style]\n{self.settings.gemini_style_prompt}\n\n[text]\n{text}"

    def synthesize(self, text: str, *, target_ms: int | None = None) -> SynthesisResult:
        # target_ms is intentionally unused: Gemini has no duration control.
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=self.settings.gemini_style_prompt),
                    types.Part(text=text),
                ],
            )
        ]
        config = types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=self.settings.gemini_voice
                    )
                )
            ),
        )

        response = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.settings.gemini_model,
                    contents=contents,
                    config=config,
                )
                break
            except Exception as exc:  # google-genai raises provider-specific errors
                status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if attempt == _MAX_RETRIES - 1 or (status is not None and status not in _RETRYABLE_STATUS):
                    raise
                time.sleep(2**attempt)

        pcm = self._extract_pcm(response)
        input_tokens, output_audio_tokens = self._extract_usage(response)
        duration_ms = (
            len(pcm)
            * 1000
            // (self.settings.sample_rate * self.settings.sample_width * self.settings.channels)
        )

        return SynthesisResult(
            pcm=pcm,
            duration_ms=duration_ms,
            duration_constrained=False,
            billed_units={
                "input_tokens": input_tokens,
                "output_audio_tokens": output_audio_tokens,
            },
            request_payload=self.build_payload(text),
        )

    @staticmethod
    def _extract_pcm(response) -> bytes:
        part = response.candidates[0].content.parts[0]
        data = part.inline_data.data
        # google-genai decodes base64 inline_data to bytes already; guard
        # against SDK versions that instead return a base64 str.
        if isinstance(data, str):
            import base64

            return base64.b64decode(data)
        return data

    @staticmethod
    def _extract_usage(response) -> tuple[int, int]:
        usage = getattr(response, "usage_metadata", None)
        if usage is None:
            return 0, 0
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_audio_tokens = getattr(usage, "candidates_token_count", 0) or 0
        return input_tokens, output_audio_tokens

    def list_voices(self) -> list[Voice]:
        raise NotImplementedError(
            "Gemini does not expose a voice-listing API; consult provider docs for the current prebuilt voice set."
        )
