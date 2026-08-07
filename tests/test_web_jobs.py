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


if __name__ == "__main__":
    unittest.main()
