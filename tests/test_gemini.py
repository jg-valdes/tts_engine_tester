import base64
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.config import Settings
from src.providers.base import ProviderRateLimitError
from src.providers.gemini import GeminiProvider


class GeminiProviderTests(unittest.TestCase):
    def setUp(self):
        self.provider = GeminiProvider(Settings())

    def test_payload_uses_interactions_shape_and_separate_text_blocks(self):
        payload = json.loads(self.provider.build_payload("Exact transcript."))

        self.assertEqual(payload["model"], "gemini-3.1-flash-tts-preview")
        self.assertFalse(payload["store"])
        self.assertEqual(payload["response_format"], {"type": "audio"})
        self.assertEqual(
            payload["generation_config"], {"speech_config": [{"voice": "Charon"}]}
        )
        self.assertEqual(len(payload["input"]), 2)
        self.assertEqual(payload["input"][0]["type"], "text")
        self.assertEqual(payload["input"][1], {"type": "text", "text": "Exact transcript."})

    def test_extract_pcm_decodes_interactions_output_audio(self):
        pcm = b"\x01\x02\x03\x04"
        response = SimpleNamespace(
            output_audio=SimpleNamespace(data=base64.b64encode(pcm).decode("ascii"))
        )

        self.assertEqual(self.provider._extract_pcm(response), pcm)

    def test_extract_usage_counts_audio_modality(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(
                total_input_tokens=17,
                total_output_tokens=200,
                output_tokens_by_modality=[
                    SimpleNamespace(modality="text", tokens=3),
                    SimpleNamespace(modality="audio", tokens=197),
                ],
            )
        )

        self.assertEqual(self.provider._extract_usage(response), (17, 197))

    def test_synthesize_makes_one_stateless_sentence_request(self):
        pcm = b"\x01\x00" * 24
        interaction = SimpleNamespace(
            status="completed",
            output_audio=SimpleNamespace(
                data=base64.b64encode(pcm).decode("ascii"),
                sample_rate=24000,
                channels=1,
            ),
            usage=SimpleNamespace(
                total_input_tokens=5,
                output_tokens_by_modality=[SimpleNamespace(modality="audio", tokens=25)],
            ),
        )
        create = Mock(return_value=interaction)
        self.provider._client = SimpleNamespace(interactions=SimpleNamespace(create=create))

        result = self.provider.synthesize("Exact transcript.")

        create.assert_called_once()
        request = create.call_args.kwargs
        self.assertFalse(request["store"])
        self.assertEqual(request["input"][1]["text"], "Exact transcript.")
        self.assertEqual(result.pcm, pcm)
        self.assertEqual(result.duration_ms, 1)
        self.assertEqual(result.billed_units, {"input_tokens": 5, "output_audio_tokens": 25})

    def test_targeted_synthesis_corrects_pace_and_keeps_closest_attempt(self):
        settings = Settings(
            sample_rate=1000,
            sample_width=2,
            channels=1,
            gemini_timing_attempts=3,
            gemini_timing_tolerance_ms=10,
        )
        provider = GeminiProvider(settings)

        def interaction(duration_ms: int):
            pcm = b"\x01\x00" * duration_ms
            return SimpleNamespace(
                status="completed",
                output_audio=SimpleNamespace(
                    data=base64.b64encode(pcm).decode("ascii"),
                    sample_rate=1000,
                    channels=1,
                ),
                usage=SimpleNamespace(
                    total_input_tokens=4,
                    output_tokens_by_modality=[
                        SimpleNamespace(modality="audio", tokens=duration_ms)
                    ],
                ),
            )

        create = Mock(side_effect=[interaction(260), interaction(205)])
        provider._client = SimpleNamespace(interactions=SimpleNamespace(create=create))

        result = provider.synthesize("four exact transcript words", target_ms=200)

        self.assertEqual(create.call_count, 2)
        self.assertEqual(result.duration_ms, 205)
        self.assertFalse(result.duration_constrained)
        self.assertEqual(result.billed_units, {"input_tokens": 8, "output_audio_tokens": 465})
        correction = create.call_args_list[1].kwargs["input"][0]["text"]
        self.assertIn("1.30 times faster", correction)

    def test_list_voices_returns_documented_snapshot(self):
        voices = self.provider.list_voices()

        self.assertEqual(len(voices), 30)
        self.assertEqual(voices[0].name, "Zephyr")
        self.assertEqual(voices[0].description, "Bright")
        self.assertIn("Charon", {voice.name for voice in voices})
        self.assertEqual(voices[-1].name, "Sulafat")

    def test_rate_limit_is_not_retried(self):
        class RateLimitError(Exception):
            status_code = 429

        create = Mock(side_effect=RateLimitError("too many requests"))
        self.provider._client = SimpleNamespace(interactions=SimpleNamespace(create=create))

        with self.assertRaises(ProviderRateLimitError):
            self.provider.synthesize("Exact transcript.")

        create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
