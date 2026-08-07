"""Azure AI Speech provider. See docs/source.md Provider specifics > Azure."""

from __future__ import annotations

import time
from xml.sax.saxutils import escape

import httpx

from ..audio import strip_riff_header
from .base import AttemptTrace, SynthesisResult, Voice

_MAX_RETRIES = 5
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_USER_AGENT = "tts-harness/0.1"


class AzureProvider:
    name = "azure"

    def __init__(self, settings):
        self.settings = settings

    def _base_url(self) -> str:
        return f"https://{self.settings.azure_speech_region}.tts.speech.microsoft.com"

    def build_payload(self, text: str, target_ms: int | None = None) -> str:
        escaped_text = escape(text)
        voice = escape(self.settings.azure_voice)

        duration_tag = ""
        if target_ms is not None and self.settings.fit_mode == "constrain":
            duration_tag = f'<mstts:audioduration value="{int(target_ms)}ms"/>'

        style_open, style_close = "", ""
        if self.settings.azure_style:
            style = escape(self.settings.azure_style)
            style_open = f'<mstts:express-as style="{style}">'
            style_close = "</mstts:express-as>"

        return (
            '<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" '
            'xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">'
            f'<voice name="{voice}">'
            f"{duration_tag}{style_open}{escaped_text}{style_close}"
            "</voice></speak>"
        )

    def synthesize(self, text: str, *, target_ms: int | None = None) -> SynthesisResult:
        ssml = self.build_payload(text, target_ms)
        url = f"{self._base_url()}/cognitiveservices/v1"
        headers = {
            "Ocp-Apim-Subscription-Key": self.settings.azure_speech_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": self.settings.azure_output_format,
            "User-Agent": _USER_AGENT,
        }

        response = None
        for attempt in range(_MAX_RETRIES):
            response = httpx.post(url, headers=headers, content=ssml.encode("utf-8"), timeout=30.0)
            if response.status_code == 200:
                break
            if response.status_code in _RETRYABLE_STATUS and attempt < _MAX_RETRIES - 1:
                time.sleep(2**attempt)
                continue
            response.raise_for_status()

        pcm = strip_riff_header(response.content)
        duration_ms = (
            len(pcm)
            * 1000
            // (self.settings.sample_rate * self.settings.sample_width * self.settings.channels)
        )
        constrained = target_ms is not None and self.settings.fit_mode == "constrain"

        return SynthesisResult(
            pcm=pcm,
            duration_ms=duration_ms,
            duration_constrained=constrained,
            billed_units={"characters": len(ssml)},
            request_payload=ssml,
            attempts=[
                AttemptTrace(
                    pcm=pcm,
                    duration_ms=duration_ms,
                    billed_units={"characters": len(ssml)},
                    request_payload=ssml,
                    target_ms=target_ms,
                    duration_constrained=constrained,
                )
            ],
        )

    def list_voices(self) -> list[Voice]:
        url = f"{self._base_url()}/cognitiveservices/voices/list"
        headers = {"Ocp-Apim-Subscription-Key": self.settings.azure_speech_key, "User-Agent": _USER_AGENT}
        response = httpx.get(url, headers=headers, timeout=30.0)
        response.raise_for_status()
        return [Voice(name=v["ShortName"], locale=v.get("Locale")) for v in response.json()]
