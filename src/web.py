"""Local FastAPI UI for configuring and testing runs."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import fields
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .config import Settings, load_settings
from .report import new_run_id
from .runner import (
    build_comparison,
    get_provider,
    load_segments_payload,
    resolved_config,
    run_synth,
    segments_to_payload,
)
from .store import RunStore

_SECRET_FIELDS = {"gemini_api_key", "azure_speech_key"}
_PRESET_FIELDS = {
    "provider",
    "gemini_base_url",
    "gemini_model",
    "gemini_voice",
    "gemini_style_prompt",
    "gemini_timing_attempts",
    "gemini_timing_tolerance_ms",
    "azure_speech_region",
    "azure_voice",
    "azure_output_format",
    "azure_style",
    "sample_rate",
    "sample_width",
    "channels",
    "words_per_second",
    "max_compression_ratio",
    "fit_mode",
}
_FIELD_TYPES = {field.name: type(field.default) for field in fields(Settings)}


def _new_web_run_id(prefix: str) -> str:
    return f"{new_run_id()}-{prefix}-{uuid.uuid4().hex[:6]}"


def _coerce(name: str, value: Any):
    if value in (None, ""):
        return None
    target_type = _FIELD_TYPES[name]
    if target_type is bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value


def _settings_to_form(settings: Settings) -> dict:
    return {field.name: getattr(settings, field.name) for field in fields(Settings)}


def _sanitize_preset(payload: dict) -> dict:
    cleaned = {}
    for name in _PRESET_FIELDS:
        coerced = _coerce(name, payload.get(name))
        if coerced is not None:
            cleaned[name] = coerced
    return cleaned


def _build_overrides(payload: dict, preset_config: dict | None = None) -> dict:
    merged = dict(preset_config or {})
    for field_name in _FIELD_TYPES:
        if field_name in payload:
            coerced = _coerce(field_name, payload.get(field_name))
            if coerced is not None:
                merged[field_name] = coerced
    return merged


class JobManager:
    def __init__(self, settings: Settings, store: RunStore):
        self.settings = settings
        self.store = store
        self._active: dict[str, dict] = {}
        self._lock = threading.Lock()

    def progress(self, run_id: str) -> dict | None:
        with self._lock:
            active = self._active.get(run_id)
        if active is not None:
            return active
        return self.store.get_progress(run_id)

    def submit_synth(self, payload: dict) -> str:
        segments = load_segments_payload(payload["segments"])
        preset = self._preset(payload)
        overrides = _build_overrides(payload, preset)
        settings = load_settings(overrides)
        run_id = _new_web_run_id(settings.provider)
        out_dir = Path(settings.output_dir) / run_id
        provider = get_provider(settings.provider, settings)
        resolved = resolved_config(settings, provider)
        input_payload = segments_to_payload(segments)
        self.store.create_run(
            run_id,
            "synth",
            "queued",
            provider=settings.provider,
            input_payload=input_payload,
            resolved_config=resolved,
            output_dir=str(out_dir),
            message="Queued synth run",
        )
        self._set_active(run_id, {"state": "queued", "message": "Queued synth run"})
        self.store.append_event(run_id, {"state": "queued", "message": "Queued synth run"})
        threading.Thread(
            target=self._run_synth_job,
            args=(
                run_id,
                settings,
                segments,
                out_dir,
                bool(payload.get("no_cache", False)),
                input_payload,
                resolved,
            ),
            daemon=True,
        ).start()
        return run_id

    def submit_compare(self, payload: dict) -> str:
        segments = load_segments_payload(payload["segments"])
        parent_id = _new_web_run_id("compare")
        out_dir = Path(self.settings.output_dir) / parent_id
        input_payload = segments_to_payload(segments)
        self.store.create_run(
            parent_id,
            "compare",
            "queued",
            providers=["gemini", "azure"],
            input_payload=input_payload,
            output_dir=str(out_dir),
            message="Queued compare run",
        )
        self._set_active(parent_id, {"state": "queued", "message": "Queued compare run"})
        self.store.append_event(parent_id, {"state": "queued", "message": "Queued compare run"})
        threading.Thread(
            target=self._run_compare_job,
            args=(
                parent_id,
                payload,
                segments,
                out_dir,
                bool(payload.get("no_cache", False)),
            ),
            daemon=True,
        ).start()
        return parent_id

    def _run_synth_job(
        self,
        run_id: str,
        settings: Settings,
        segments: list,
        out_dir: Path,
        no_cache: bool,
        input_payload: dict,
        resolved: dict,
    ) -> None:
        try:
            progress = self._progress_callback(run_id)
            report = run_synth(
                settings,
                settings.provider,
                segments,
                out_dir,
                False,
                no_cache,
                progress=progress,
            )
            self.store.finalize_synth_run(
                run_id,
                report,
                output_dir=str(out_dir),
                input_payload=input_payload,
                resolved_config=resolved,
            )
            self._set_active(run_id, {"state": "completed", "message": "Run completed"})
        except Exception as exc:
            self.store.fail_run(run_id, str(exc))
            self.store.append_event(run_id, {"state": "failed", "message": str(exc), "lastError": str(exc)})
            self._set_active(run_id, {"state": "failed", "message": str(exc), "lastError": str(exc)})

    def _run_compare_job(
        self,
        parent_id: str,
        payload: dict,
        segments: list,
        out_dir: Path,
        no_cache: bool,
    ) -> None:
        reports: dict[str, dict] = {}
        try:
            for provider_name in ("gemini", "azure"):
                self._record_progress(
                    parent_id,
                    {
                        "state": "starting_provider",
                        "provider": provider_name,
                        "message": f"Starting {provider_name} run",
                    },
                )
                child_payload = dict(payload)
                child_payload["provider"] = provider_name
                preset = self._preset(child_payload)
                overrides = _build_overrides(child_payload, preset)
                settings = load_settings(overrides)
                child_id = f"{parent_id}-{provider_name}"
                provider = get_provider(provider_name, settings)
                child_resolved = resolved_config(settings, provider)
                child_out_dir = out_dir / provider_name
                self.store.create_run(
                    child_id,
                    "synth",
                    "queued",
                    provider=provider_name,
                    parent_run_id=parent_id,
                    input_payload=segments_to_payload(segments),
                    resolved_config=child_resolved,
                    output_dir=str(child_out_dir),
                    message=f"Queued {provider_name} child run",
                )
                report = run_synth(
                    settings,
                    provider_name,
                    segments,
                    child_out_dir,
                    False,
                    no_cache,
                    progress=self._progress_callback(child_id, parent_id=parent_id),
                )
                reports[provider_name] = report
                self.store.finalize_synth_run(
                    child_id,
                    report,
                    output_dir=str(child_out_dir),
                    input_payload=segments_to_payload(segments),
                    resolved_config=child_resolved,
                )
            comparison = build_comparison(["gemini", "azure"], reports, segments)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
            self.store.finalize_compare_run(parent_id, comparison, output_dir=str(out_dir))
            self._set_active(parent_id, {"state": "completed", "message": "Compare completed"})
        except Exception as exc:
            self.store.fail_run(parent_id, str(exc))
            self.store.append_event(parent_id, {"state": "failed", "message": str(exc), "lastError": str(exc)})
            self._set_active(parent_id, {"state": "failed", "message": str(exc), "lastError": str(exc)})

    def _preset(self, payload: dict) -> dict:
        preset_name = payload.get("preset_name") or ""
        if not preset_name:
            return {}
        for preset in self.store.list_presets():
            if preset["name"] == preset_name:
                return preset["config"]
        return {}

    def _progress_callback(self, run_id: str, parent_id: str | None = None):
        def callback(update: dict):
            self._record_progress(run_id, update)
            if parent_id:
                parent_update = dict(update)
                parent_update["message"] = f"{update.get('provider', 'provider')}: {update.get('message', '')}"
                self._record_progress(parent_id, parent_update)

        return callback

    def _record_progress(self, run_id: str, update: dict) -> None:
        self._set_active(run_id, update)
        self.store.update_run_status(run_id, update.get("state", "running"), update)
        self.store.append_event(run_id, update)

    def _set_active(self, run_id: str, update: dict) -> None:
        with self._lock:
            self._active[run_id] = update


def create_app(settings: Settings | None = None, store: RunStore | None = None) -> FastAPI:
    settings = settings or load_settings()
    store = store or RunStore(settings.web_db_path)
    job_manager = JobManager(settings, store)
    app = FastAPI(title="tts-harness web")
    templates = Jinja2Templates(directory=str(Path(__file__).with_name("templates")))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        sample_path = Path("segments.example.json")
        sample_segments = sample_path.read_text() if sample_path.exists() else json.dumps({"segments": []}, indent=2)
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "settings": _settings_to_form(settings),
                "sample_segments": sample_segments,
                "presets": store.list_presets(),
                "history": store.list_runs()[:10],
                "gemini_voices": [voice.__dict__ for voice in get_provider("gemini", settings).list_voices()],
            },
        )

    @app.get("/history", response_class=HTMLResponse)
    async def history(request: Request):
        return templates.TemplateResponse(
            request,
            "history.html",
            {"runs": store.list_runs()},
        )

    @app.get("/runs/{run_id}", response_class=HTMLResponse)
    async def run_detail(request: Request, run_id: str):
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return templates.TemplateResponse(request, "run.html", {"run": run})

    @app.get("/api/runs/{run_id}/status")
    async def run_status(run_id: str):
        progress = job_manager.progress(run_id)
        if progress is None:
            raise HTTPException(status_code=404, detail="run not found")
        return JSONResponse(progress)

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str):
        run = store.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return JSONResponse(run["events"])

    @app.get("/api/presets")
    async def presets():
        return JSONResponse(store.list_presets())

    @app.post("/api/presets")
    async def save_preset(request: Request):
        payload = await request.json()
        name = (payload.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="preset name is required")
        preset = store.save_preset(name, _sanitize_preset(payload.get("config") or {}))
        return JSONResponse(preset)

    @app.post("/api/providers/{provider_name}/voices")
    async def provider_voices(provider_name: str, request: Request):
        payload = await request.json()
        overrides = _build_overrides(payload)
        overrides["provider"] = provider_name
        provider_settings = load_settings(overrides)
        provider = get_provider(provider_name, provider_settings)
        try:
            voices = [voice.__dict__ for voice in provider.list_voices()]
        except Exception as exc:
            return JSONResponse({"voices": [], "error": str(exc)}, status_code=200)
        return JSONResponse({"voices": voices})

    @app.post("/api/runs/synth")
    async def create_synth_run(request: Request):
        payload = await request.json()
        run_id = job_manager.submit_synth(payload)
        return JSONResponse({"runId": run_id, "redirect": f"/runs/{run_id}"})

    @app.post("/api/runs/compare")
    async def create_compare_run(request: Request):
        payload = await request.json()
        run_id = job_manager.submit_compare(payload)
        return JSONResponse({"runId": run_id, "redirect": f"/runs/{run_id}"})

    @app.post("/runs")
    async def create_run_from_form(request: Request):
        form = await request.form()
        payload = {key: value for key, value in form.items()}
        payload["no_cache"] = form.get("no_cache") == "on"
        payload["segments"] = json.loads(form["segments_json"])["segments"]
        if form.get("mode") == "compare":
            run_id = job_manager.submit_compare(payload)
        else:
            run_id = job_manager.submit_synth(payload)
        return RedirectResponse(url=f"/runs/{run_id}", status_code=303)

    @app.get("/media/{run_id}/{relative_path:path}")
    async def media(run_id: str, relative_path: str):
        run = store.get_run(run_id)
        if run is None or not run.get("outputDir"):
            raise HTTPException(status_code=404, detail="run not found")
        base = Path(run["outputDir"]).resolve()
        target = (base / relative_path).resolve()
        if not str(target).startswith(str(base)) or not target.exists():
            raise HTTPException(status_code=404, detail="artifact not found")
        return FileResponse(target)

    return app
