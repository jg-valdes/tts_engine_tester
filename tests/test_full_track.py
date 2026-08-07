import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from src.runner import Segment, run_synth
from src.config import Settings
from src.providers.base import ProviderRateLimitError
from src.providers.base import SynthesisResult, Voice


class FakeProvider:
    name = "gemini"

    def __init__(self, settings):
        self.settings = settings

    def build_payload(self, text: str, target_ms: int | None = None) -> str:
        return text

    def synthesize(self, text: str, *, target_ms: int | None = None) -> SynthesisResult:
        if text == "fails":
            raise RuntimeError("synthetic provider failure")
        if text == "overflow":
            duration_ms = 210 if target_ms is not None else 300
            pcm = b"\x03\x00" * duration_ms
            return SynthesisResult(pcm, duration_ms, False, {"input_tokens": 1}, text)
        sample = 1 if text == "first" else 2
        duration_ms = 50 if text == "first" else 100
        pcm = sample.to_bytes(2, "little", signed=True) * duration_ms
        return SynthesisResult(pcm, duration_ms, False, {}, text)

    def list_voices(self) -> list[Voice]:
        return []


class RateLimitedProvider(FakeProvider):
    def synthesize(self, text: str, *, target_ms: int | None = None) -> SynthesisResult:
        raise ProviderRateLimitError("gemini", "Gemini rate limit reached (HTTP 429)")


class FullTrackTests(unittest.TestCase):
    def test_timestamped_sentences_are_placed_on_one_absolute_timeline(self):
        settings = Settings(sample_rate=1000, sample_width=2, channels=1)
        segments = [
            Segment("s1", 100, 200, "first"),
            Segment("s2", 500, 700, "second"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "run"
            with (
                patch("src.runner.get_provider", return_value=FakeProvider(settings)),
                patch("src.runner.log.exception"),
            ):
                report = run_synth(settings, "gemini", segments, out_dir, False, True)

            with wave.open(str(out_dir / "track.wav"), "rb") as track:
                self.assertEqual(track.getframerate(), 1000)
                self.assertEqual(track.getnframes(), 700)
                pcm = track.readframes(track.getnframes())

            self.assertEqual(report["totals"]["trackDurationMs"], 700)
            self.assertEqual(pcm[: 100 * 2], bytes(100 * 2))
            self.assertEqual(pcm[100 * 2 : 150 * 2], b"\x01\x00" * 50)
            self.assertEqual(pcm[500 * 2 : 600 * 2], b"\x02\x00" * 100)

    def test_failed_final_sentence_does_not_shorten_the_timeline(self):
        settings = Settings(sample_rate=1000, sample_width=2, channels=1)
        segments = [
            Segment("s1", 0, 100, "first"),
            Segment("s2", 500, 900, "fails"),
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "run"
            with (
                patch("src.runner.get_provider", return_value=FakeProvider(settings)),
                patch("src.runner.log.exception"),
            ):
                report = run_synth(settings, "gemini", segments, out_dir, False, True)

            with wave.open(str(out_dir / "track.wav"), "rb") as track:
                self.assertEqual(track.getnframes(), 900)

            self.assertEqual(report["totals"]["trackDurationMs"], 900)
            self.assertEqual(report["totals"]["failed"], 1)

    def test_constrain_mode_uses_closer_best_effort_render_for_overflow(self):
        settings = Settings(
            sample_rate=1000,
            sample_width=2,
            channels=1,
            fit_mode="constrain",
        )
        segments = [Segment("s1", 0, 200, "overflow")]

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "run"
            with patch("src.runner.get_provider", return_value=FakeProvider(settings)):
                report = run_synth(settings, "gemini", segments, out_dir, False, True)

        segment = report["segments"][0]
        self.assertEqual(segment["naturalMs"], 300)
        self.assertEqual(segment["finalMs"], 210)
        self.assertTrue(segment["timingAdjusted"])
        self.assertFalse(segment["durationConstrained"])
        self.assertEqual(segment["billedUnits"], {"input_tokens": 2})
        self.assertEqual(segment["finalFit"], "TIGHT")
        self.assertEqual(report["totals"]["tight"], 1)
        self.assertEqual(report["totals"]["overflow"], 0)

    def test_rate_limited_segment_gets_placeholder_metrics_and_run_continues(self):
        settings = Settings(sample_rate=1000, sample_width=2, channels=1)
        segments = [Segment("s1", 0, 200, "first")]

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "run"
            with patch("src.runner.get_provider", return_value=RateLimitedProvider(settings)):
                report = run_synth(settings, "gemini", segments, out_dir, False, True)

        segment = report["segments"][0]
        self.assertTrue(segment["rateLimited"])
        self.assertEqual(segment["fit"], "RATE_LIMIT")
        self.assertEqual(segment["finalFit"], "RATE_LIMIT")
        self.assertIsNone(segment["naturalMs"])
        self.assertIsNone(segment["finalMs"])
        self.assertEqual(report["totals"]["rateLimited"], 1)
        self.assertEqual(report["totals"]["failed"], 1)


if __name__ == "__main__":
    unittest.main()
