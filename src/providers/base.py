"""Provider protocol and shared result types. See docs/source.md for the contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ProviderRateLimitError(RuntimeError):
    def __init__(self, provider: str, message: str):
        super().__init__(message)
        self.provider = provider


@dataclass
class Voice:
    name: str
    locale: str | None = None
    description: str | None = None


@dataclass
class AttemptTrace:
    pcm: bytes
    duration_ms: int
    billed_units: dict
    request_payload: str
    target_ms: int | None = None
    duration_constrained: bool = False


@dataclass
class SynthesisResult:
    pcm: bytes
    duration_ms: int
    duration_constrained: bool
    billed_units: dict
    request_payload: str
    attempts: list[AttemptTrace] | None = None


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
