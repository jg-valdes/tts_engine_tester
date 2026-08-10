import tempfile
import time
import unittest
import asyncio
from pathlib import Path
from unittest.mock import patch

import httpx

from src.config import Settings
from src.providers.base import SynthesisResult, Voice
from src.store import RunStore
from src.web import JobManager, create_app


class FakeProvider:
    def __init__(self, settings, name="gemini"):
        self.settings = settings
        self.name = name

    def build_payload(self, text: str, target_ms: int | None = None) -> str:
        return text

    def synthesize(self, text: str, *, target_ms: int | None = None) -> SynthesisResult:
        time.sleep(0.01)
        duration_ms = target_ms or 80
        pcm = b"\x01\x00" * duration_ms
        return SynthesisResult(
            pcm=pcm,
            duration_ms=duration_ms,
            duration_constrained=target_ms is not None,
            billed_units={"input_tokens": 1},
            request_payload=text,
        )

    def list_voices(self) -> list[Voice]:
        return [Voice(name="Charon", description="Informative")]


class FakeAzureProvider(FakeProvider):
    calls = 0

    def list_voices(self) -> list[Voice]:
        type(self).calls += 1
        return [
            Voice(name="en-US-AvaNeural", locale="en-US"),
            Voice(name="es-UY-ValentinaNeural", locale="es-UY"),
        ]


class WebJobTests(unittest.TestCase):
    def test_job_manager_records_progress_and_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                output_dir=str(Path(temp_dir) / "out"),
                cache_dir=str(Path(temp_dir) / "cache"),
                web_db_path=str(Path(temp_dir) / "web.sqlite3"),
                sample_rate=1000,
                sample_width=2,
                channels=1,
            )
            store = RunStore(settings.web_db_path)
            manager = JobManager(settings, store)

            def fake_get_provider(name, provider_settings):
                return FakeProvider(provider_settings, name=name)

            with (
                patch("src.web.get_provider", side_effect=fake_get_provider),
                patch("src.runner.get_provider", side_effect=fake_get_provider),
            ):
                run_id = manager.submit_synth(
                    {
                        "provider": "gemini",
                        "segments": [
                            {
                                "id": "s1",
                                "startTime": 0,
                                "endTime": 100,
                                "description": "hello world",
                            }
                        ],
                    }
                )

                progress = None
                for _ in range(100):
                    progress = manager.progress(run_id)
                    if progress and progress["state"] == "completed":
                        break
                    time.sleep(0.02)

                self.assertIsNotNone(progress)
                self.assertEqual(progress["state"], "completed")

                run = store.get_run(run_id)
                self.assertIsNotNone(run)
                assert run is not None
                self.assertEqual(run["status"], "completed")
                self.assertTrue((Path(run["outputDir"]) / "track.wav").exists())
                self.assertEqual(run["segments"][0]["file"], "segments/s1.wav")
                self.assertTrue(run["segments"][0]["attempts"])
                event_states = {event["state"] for event in run["events"]}
                self.assertIn("rendering_segment", event_states)
                self.assertIn("writing_report", event_states)
                self.assertIn("completed", event_states)

    def test_create_app_exposes_dashboard_route(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                output_dir=str(Path(temp_dir) / "out"),
                cache_dir=str(Path(temp_dir) / "cache"),
                web_db_path=str(Path(temp_dir) / "web.sqlite3"),
            )
            app = create_app(settings=settings, store=RunStore(settings.web_db_path))
            paths = {route.path for route in app.routes}
            self.assertIn("/", paths)
            self.assertIn("/api/runs/{run_id}/status", paths)

    def test_dashboard_route_renders(self):
        asyncio.run(self._dashboard_route_renders())

    async def _dashboard_route_renders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                output_dir=str(Path(temp_dir) / "out"),
                cache_dir=str(Path(temp_dir) / "cache"),
                web_db_path=str(Path(temp_dir) / "web.sqlite3"),
                azure_speech_key="test-azure-key",
            )
            app = create_app(settings=settings, store=RunStore(settings.web_db_path))

            def fake_get_provider(name, provider_settings):
                return FakeProvider(provider_settings, name=name)

            with patch("src.web.get_provider", side_effect=fake_get_provider):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.get("/")
                    self.assertEqual(response.status_code, 200)
                    self.assertIn("tts-harness web", response.text)
                    self.assertIn("test-azure-key", response.text)

    def test_azure_voices_endpoint_uses_cache(self):
        asyncio.run(self._azure_voices_endpoint_uses_cache())

    async def _azure_voices_endpoint_uses_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                output_dir=str(Path(temp_dir) / "out"),
                cache_dir=str(Path(temp_dir) / "cache"),
                web_db_path=str(Path(temp_dir) / "web.sqlite3"),
                azure_speech_key="test-azure-key",
                azure_speech_region="eastus",
            )
            app = create_app(settings=settings, store=RunStore(settings.web_db_path))
            FakeAzureProvider.calls = 0

            def fake_get_provider(name, provider_settings):
                if name == "azure":
                    return FakeAzureProvider(provider_settings, name=name)
                return FakeProvider(provider_settings, name=name)

            with patch("src.web.get_provider", side_effect=fake_get_provider):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    first = await client.post(
                        "/api/providers/azure/voices",
                        json={
                            "azure_speech_key": "test-azure-key",
                            "azure_speech_region": "eastus",
                            "force_refresh": True,
                        },
                    )
                    self.assertEqual(first.status_code, 200)
                    self.assertEqual(first.json()["source"], "live")
                    self.assertEqual(FakeAzureProvider.calls, 1)

                    second = await client.post(
                        "/api/providers/azure/voices",
                        json={"azure_speech_key": "test-azure-key", "azure_speech_region": "eastus"},
                    )
                    self.assertEqual(second.status_code, 200)
                    self.assertEqual(second.json()["source"], "cache")
                    self.assertEqual(FakeAzureProvider.calls, 1)

    def test_preset_api_persists_dashboard_fields(self):
        asyncio.run(self._preset_api_persists_dashboard_fields())

    async def _preset_api_persists_dashboard_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                output_dir=str(Path(temp_dir) / "out"),
                cache_dir=str(Path(temp_dir) / "cache"),
                web_db_path=str(Path(temp_dir) / "web.sqlite3"),
            )
            app = create_app(settings=settings, store=RunStore(settings.web_db_path))
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.post(
                    "/api/presets",
                    json={
                        "name": "Tester preset",
                        "config": {
                            "provider": "azure",
                            "fit_mode": "constrain",
                            "output_dir": "/tmp/out",
                            "web_db_path": "/tmp/web.sqlite3",
                            "mode": "compare",
                            "input_type": "single_text",
                            "single_text": "should not persist",
                        },
                    },
                )
                self.assertEqual(response.status_code, 200)
                payload = response.json()
                self.assertEqual(payload["name"], "Tester preset")
                self.assertEqual(
                    payload["config"],
                    {
                        "provider": "azure",
                        "fit_mode": "constrain",
                        "output_dir": "/tmp/out",
                        "web_db_path": "/tmp/web.sqlite3",
                        "mode": "compare",
                        "input_type": "single_text",
                    },
                )

    def test_single_text_form_creates_one_segment_run(self):
        asyncio.run(self._single_text_form_creates_one_segment_run())

    async def _single_text_form_creates_one_segment_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                output_dir=str(Path(temp_dir) / "out"),
                cache_dir=str(Path(temp_dir) / "cache"),
                web_db_path=str(Path(temp_dir) / "web.sqlite3"),
                sample_rate=1000,
                sample_width=2,
                channels=1,
            )
            store = RunStore(settings.web_db_path)
            app = create_app(settings=settings, store=store)

            def fake_get_provider(name, provider_settings):
                return FakeProvider(provider_settings, name=name)

            with (
                patch("src.web.get_provider", side_effect=fake_get_provider),
                patch("src.runner.get_provider", side_effect=fake_get_provider),
            ):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.post(
                        "/runs",
                        data={
                            "mode": "synth",
                            "input_type": "single_text",
                            "provider": "gemini",
                            "single_text": "One short sample line.",
                            "single_duration_ms": "2500",
                            "segments_json": '{"segments":[]}',
                        },
                        follow_redirects=False,
                    )
                    self.assertEqual(response.status_code, 303)
                    run_id = response.headers["location"].rsplit("/", 1)[-1]

                progress = None
                for _ in range(100):
                    progress = store.get_progress(run_id)
                    if progress and progress["state"] == "completed":
                        break
                    time.sleep(0.02)

                self.assertIsNotNone(progress)
                self.assertEqual(progress["state"], "completed")
                run = store.get_run(run_id)
                assert run is not None
                self.assertEqual(len(run["segments"]), 1)
                self.assertEqual(run["segments"][0]["startTime"], 0)
                self.assertEqual(run["segments"][0]["targetMs"], 2500)

    def test_single_text_form_without_duration_uses_natural_timing(self):
        asyncio.run(self._single_text_form_without_duration_uses_natural_timing())

    async def _single_text_form_without_duration_uses_natural_timing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = Settings(
                output_dir=str(Path(temp_dir) / "out"),
                cache_dir=str(Path(temp_dir) / "cache"),
                web_db_path=str(Path(temp_dir) / "web.sqlite3"),
                sample_rate=1000,
                sample_width=2,
                channels=1,
            )
            store = RunStore(settings.web_db_path)
            app = create_app(settings=settings, store=store)

            def fake_get_provider(name, provider_settings):
                return FakeProvider(provider_settings, name=name)

            with (
                patch("src.web.get_provider", side_effect=fake_get_provider),
                patch("src.runner.get_provider", side_effect=fake_get_provider),
            ):
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                    response = await client.post(
                        "/runs",
                        data={
                            "mode": "synth",
                            "input_type": "single_text",
                            "provider": "gemini",
                            "single_text": "Untimed sample line.",
                            "single_duration_ms": "",
                            "segments_json": '{"segments":[]}',
                        },
                        follow_redirects=False,
                    )
                    self.assertEqual(response.status_code, 303)
                    run_id = response.headers["location"].rsplit("/", 1)[-1]

                progress = None
                for _ in range(100):
                    progress = store.get_progress(run_id)
                    if progress and progress["state"] == "completed":
                        break
                    time.sleep(0.02)

                self.assertIsNotNone(progress)
                self.assertEqual(progress["state"], "completed")
                run = store.get_run(run_id)
                assert run is not None
                self.assertEqual(len(run["segments"]), 1)
                self.assertIsNone(run["segments"][0]["targetMs"])
                self.assertEqual(run["segments"][0]["finalFit"], "NATURAL")


if __name__ == "__main__":
    unittest.main()
