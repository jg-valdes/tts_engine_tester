"""Provider protocol and shared result types. See docs/source.md for the contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class Voice:
    name: str
    locale: str | None = None


@dataclass
class SynthesisResult:
    pcm: bytes
    duration_ms: int
    duration_constrained: bool
    billed_units: dict
    request_payload: str


class Provider(Protocol):
    name: str

    def synthesize(
        self,
        text: str,
        *,
        target_ms: int | None = None,
    ) -> SynthesisResult: ...

    def list_voices(self) -> list[Voice]: ...

    def build_payload(self, text: str, target_ms: int | None) -> str:
        """Return the exact request payload (SSML or prompt) without calling the API."""
        ...
