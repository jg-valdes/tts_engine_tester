"""Shared run orchestration for CLI and web surfaces."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .audio import assemble_timeline, measure_duration_ms, strip_riff_header, write_wav
from .cache import Cache, cache_key
from .config import Settings
from .fitting import classify_fit, word_budget, word_count
from .providers.azure import AzureProvider
from .providers.base import AttemptTrace, ProviderRateLimitError, SynthesisResult
from .providers.gemini import GeminiProvider
from .report import build_report, new_run_id

log = logging.getLogger("tts_harness")

ProgressCallback = Callable[[dict], None]


@dataclass
class Segment:
    id: str
    start_time: int
    end_time: int
    description: str


_TIME_WITH_MILLIS_RE = re.compile(r"^(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d)[\.,](?P<millis>\d{1,3})$")
_TIME_WITH_CENTIS_RE = re.compile(r"^(?P<hours>\d+):(?P<minutes>[0-5]\d):(?P<seconds>[0-5]\d):(?P<centis>\d{1,2})$")


def parse_time_value(value: int | str) -> int:
    if isinstance(value, int):
        return value

    raw = str(value).strip()
    if not raw:
        raise ValueError("time value cannot be blank")
    if raw.isdigit():
        return int(raw)

    millis_match = _TIME_WITH_MILLIS_RE.match(raw)
    if millis_match:
        millis = millis_match.group("millis").ljust(3, "0")
        return _time_parts_to_ms(
            int(millis_match.group("hours")),
            int(millis_match.group("minutes")),
            int(millis_match.group("seconds")),
            int(millis),
        )

    centis_match = _TIME_WITH_CENTIS_RE.match(raw)
    if centis_match:
        return _time_parts_to_ms(
            int(centis_match.group("hours")),
            int(centis_match.group("minutes")),
            int(centis_match.group("seconds")),
            int(centis_match.group("centis")) * 10,
        )

    raise ValueError(
        "unsupported time format; use integer milliseconds, HH:MM:SS:mmm, or HH:MM:SS:CC"
    )


def _time_parts_to_ms(hours: int, minutes: int, seconds: int, millis: int) -> int:
    return ((hours * 60 + minutes) * 60 + seconds) * 1000 + millis


def get_provider(name: str, settings: Settings):
    if name == "gemini":
        return GeminiProvider(settings)
    if name == "azure":
        return AzureProvider(settings)
    raise ValueError(f"unknown provider: {name!r} (expected 'gemini' or 'azure')")


def load_segments(input_path: str) -> list[Segment]:
    data = json.loads(Path(input_path).read_text())
    return load_segments_payload(data)


def load_segments_payload(data: dict | list[dict]) -> list[Segment]:
    if isinstance(data, list):
        raw_segments = data
    else:
        raw_segments = data["segments"]
    segments = []
    for i, raw in enumerate(raw_segments):
        start_time = parse_time_value(raw["startTime"])
        end_time = parse_time_value(raw["endTime"])
        segments.append(
            Segment(
                id=raw.get("id") or f"s{i + 1}",
                start_time=start_time,
                end_time=end_time,
                description=raw["description"],
            )
        )
    return segments


def segments_to_payload(segments: list[Segment]) -> dict:
    return {
        "segments": [
            {
                "id": seg.id,
                "startTime": seg.start_time,
                "endTime": seg.end_time,
                "description": seg.description,
            }
            for seg in segments
        ]
    }


def resolved_config(settings: Settings, provider) -> dict:
    common = {
        "fitMode": settings.fit_mode,
        "sampleRate": settings.sample_rate,
        "sampleWidth": settings.sample_width,
        "channels": settings.channels,
    }
    if provider.name == "gemini":
        common.update(
            {
                "model": settings.gemini_model,
                "voice": settings.gemini_voice,
                "stylePrompt": settings.gemini_style_prompt,
                "timingAttempts": settings.gemini_timing_attempts,
                "timingToleranceMs": settings.gemini_timing_tolerance_ms,
                "baseUrl": settings.gemini_base_url or "https://generativelanguage.googleapis.com",
                "payloadSample": provider.build_payload("<sample segment text>"),
            }
        )
    else:
        common.update(
            {
                "voice": settings.azure_voice,
                "region": settings.azure_speech_region,
                "style": settings.azure_style,
                "payloadSample": provider.build_payload("<sample segment text>", None),
            }
        )
    return common


def build_comparison(provider_names: list[str], reports: dict[str, dict], segments: list[Segment]) -> dict:
    comparison_segments = []
    by_provider_segments = {p: {s["id"]: s for s in reports[p]["segments"]} for p in provider_names}
    for seg in segments:
        entry: dict = {
            "id": seg.id,
            "text": seg.description,
            "startTime": seg.start_time,
            "endTime": seg.end_time,
        }
        for p in provider_names:
            row = by_provider_segments[p].get(seg.id, {})
            entry[p] = {
                "naturalMs": row.get("naturalMs"),
                "finalMs": row.get("finalMs"),
                "overflowRatio": row.get("overflowRatio"),
                "finalOverflowRatio": row.get("finalOverflowRatio"),
                "fit": row.get("fit"),
                "finalFit": row.get("finalFit"),
                "error": row.get("error"),
            }
        comparison_segments.append(entry)
    return {
        "runId": new_run_id(),
        "providers": provider_names,
        "totals": {p: reports[p]["totals"] for p in provider_names},
        "segments": comparison_segments,
    }


def _emit(progress: ProgressCallback | None, **payload) -> None:
    if progress:
        progress(payload)


def _wav_bytes(pcm: bytes, sample_rate: int, sample_width: int, channels: int) -> bytes:
    from .audio import pcm_to_wav_bytes

    return pcm_to_wav_bytes(pcm, sample_rate, sample_width, channels)


def _add_billed_units(first: dict, second: dict) -> dict:
    combined = dict(first)
    for key, value in second.items():
        combined[key] = combined.get(key, 0) + value
    return combined


def _attempt_rows(
    result: SynthesisResult,
    phase: str,
    selected_duration_ms: int,
) -> list[dict]:
    traces = result.attempts or [
        AttemptTrace(
            pcm=result.pcm,
            duration_ms=result.duration_ms,
            billed_units=result.billed_units,
            request_payload=result.request_payload,
            duration_constrained=result.duration_constrained,
        )
    ]
    rows = []
    for index, trace in enumerate(traces, start=1):
        rows.append(
            {
                "phase": phase,
                "ordinal": index,
                "targetMs": trace.target_ms,
                "actualMs": trace.duration_ms,
                "durationConstrained": trace.duration_constrained,
                "billedUnits": trace.billed_units,
                "requestPayload": trace.request_payload,
                "selected": trace.duration_ms == selected_duration_ms and trace.pcm == result.pcm,
                "_pcm": trace.pcm,
            }
        )
    return rows


def _write_attempt_audio(
    attempt_root: Path,
    segment_id: str,
    attempts: list[dict],
    settings: Settings,
) -> list[dict]:
    segment_dir = attempt_root / segment_id
    segment_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for attempt in attempts:
        file_name = f"{attempt['phase']}-{attempt['ordinal']}.wav"
        file_path = segment_dir / file_name
        write_wav(
            file_path,
            attempt.pop("_pcm"),
            settings.sample_rate,
            settings.sample_width,
            settings.channels,
        )
        written.append(
            {
                **attempt,
                "file": f"attempts/{segment_id}/{file_name}",
            }
        )
    return written


def _rate_limited_row(row: dict, provider_name: str, message: str) -> dict:
    return {
        **row,
        "provider": provider_name,
        "naturalMs": None,
        "finalMs": None,
        "overflowRatio": None,
        "fit": "RATE_LIMIT",
        "finalOverflowRatio": None,
        "finalFit": "RATE_LIMIT",
        "durationConstrained": False,
        "timingAdjusted": False,
        "wordBudget": None,
        "wordCount": word_count(row["text"]),
        "cached": False,
        "billedUnits": {},
        "file": None,
        "attempts": [],
        "error": message,
        "rateLimited": True,
    }


def run_synth(
    settings: Settings,
    provider_name: str,
    segments: list[Segment],
    out_dir: Path,
    dry_run: bool,
    no_cache: bool,
    progress: ProgressCallback | None = None,
) -> dict:
    provider = get_provider(provider_name, settings)
    total_segments = len(segments)

    if dry_run:
        config = resolved_config(settings, provider)
        return {"dryRun": True, "provider": provider_name, "config": config}

    _emit(
        progress,
        state="loading_config",
        provider=provider_name,
        totalSegments=total_segments,
        message=f"Preparing {provider_name} run",
    )

    cache = None if no_cache else Cache(settings.cache_dir)
    seg_dir = out_dir / "segments"
    attempt_root = out_dir / "attempts"
    seg_dir.mkdir(parents=True, exist_ok=True)

    report_segments = []
    placements: list[tuple[int, bytes]] = []
    last_end_ms = max((segment.end_time for segment in segments), default=0)

    for index, seg in enumerate(segments, start=1):
        target_ms = seg.end_time - seg.start_time
        row: dict = {
            "id": seg.id,
            "text": seg.description,
            "startTime": seg.start_time,
            "endTime": seg.end_time,
            "targetMs": target_ms,
        }
        try:
            _emit(
                progress,
                state="rendering_segment",
                provider=provider_name,
                currentSegmentId=seg.id,
                currentSegmentIndex=index,
                totalSegments=total_segments,
                message=f"Rendering segment {seg.id}",
            )

            payload = provider.build_payload(seg.description, None)
            key = cache_key(
                provider.name,
                getattr(settings, f"{provider.name}_model", ""),
                getattr(settings, f"{provider.name}_voice", ""),
                payload,
                seg.description,
                None,
                settings.sample_rate,
            )
            cached_wav = cache.get(key) if cache else None
            natural_attempts: list[dict] = []

            if cached_wav is not None:
                pcm = strip_riff_header(cached_wav)
                natural_ms = measure_duration_ms(
                    pcm, settings.sample_rate, settings.sample_width, settings.channels
                )
                cached = True
                billed_units = {}
                duration_constrained = False
                natural_attempts = [
                    {
                        "phase": "natural",
                        "ordinal": 1,
                        "targetMs": None,
                        "actualMs": natural_ms,
                        "durationConstrained": False,
                        "billedUnits": {},
                        "requestPayload": payload,
                        "selected": True,
                        "_pcm": pcm,
                    }
                ]
            else:
                _emit(
                    progress,
                    state="waiting_provider_response",
                    provider=provider_name,
                    currentSegmentId=seg.id,
                    currentSegmentIndex=index,
                    totalSegments=total_segments,
                    currentAttempt=1,
                    attemptType="natural",
                    message=f"Waiting for {provider_name} response for {seg.id}",
                )
                result = provider.synthesize(seg.description, target_ms=None)
                _emit(
                    progress,
                    state="decoding_audio",
                    provider=provider_name,
                    currentSegmentId=seg.id,
                    currentSegmentIndex=index,
                    totalSegments=total_segments,
                    currentAttempt=1,
                    attemptType="natural",
                    message=f"Decoding {provider_name} audio for {seg.id}",
                )
                pcm = result.pcm
                natural_ms = result.duration_ms
                billed_units = result.billed_units
                duration_constrained = result.duration_constrained
                cached = False
                natural_attempts = _attempt_rows(result, "natural", result.duration_ms)
                if cache:
                    cache.put(
                        key,
                        _wav_bytes(pcm, settings.sample_rate, settings.sample_width, settings.channels),
                    )

            final_pcm = pcm
            final_ms = natural_ms
            overflow_ratio, fit = classify_fit(natural_ms, target_ms, settings.max_compression_ratio)

            all_attempts = list(natural_attempts)
            timing_adjusted = False
            if settings.fit_mode == "constrain" and fit in ("TIGHT", "OVERFLOW"):
                _emit(
                    progress,
                    state="waiting_provider_response",
                    provider=provider_name,
                    currentSegmentId=seg.id,
                    currentSegmentIndex=index,
                    totalSegments=total_segments,
                    currentAttempt=1,
                    attemptType="targeted",
                    message=f"Requesting targeted render for {seg.id}",
                )
                targeted_result = provider.synthesize(seg.description, target_ms=target_ms)
                billed_units = _add_billed_units(billed_units, targeted_result.billed_units)
                targeted_attempts = _attempt_rows(
                    targeted_result, "targeted", targeted_result.duration_ms
                )
                all_attempts.extend(targeted_attempts)
                if abs(targeted_result.duration_ms - target_ms) < abs(natural_ms - target_ms):
                    final_pcm = targeted_result.pcm
                    final_ms = targeted_result.duration_ms
                    duration_constrained = targeted_result.duration_constrained
                    timing_adjusted = True
                    for attempt in all_attempts:
                        attempt["selected"] = (
                            attempt["phase"] == "targeted"
                            and attempt["actualMs"] == targeted_result.duration_ms
                        )
                else:
                    for attempt in all_attempts:
                        attempt["selected"] = (
                            attempt["phase"] == "natural"
                            and attempt["actualMs"] == natural_ms
                        )

            final_overflow_ratio, final_fit = classify_fit(
                final_ms, target_ms, settings.max_compression_ratio
            )

            _emit(
                progress,
                state="writing_segment_file",
                provider=provider_name,
                currentSegmentId=seg.id,
                currentSegmentIndex=index,
                totalSegments=total_segments,
                message=f"Writing audio files for {seg.id}",
            )
            file_path = seg_dir / f"{seg.id}.wav"
            write_wav(
                file_path,
                final_pcm,
                settings.sample_rate,
                settings.sample_width,
                settings.channels,
            )
            attempt_rows = _write_attempt_audio(attempt_root, seg.id, all_attempts, settings)

            placements.append((seg.start_time, final_pcm))

            row.update(
                {
                    "provider": provider_name,
                    "naturalMs": natural_ms,
                    "finalMs": final_ms,
                    "overflowRatio": round(overflow_ratio, 4),
                    "fit": fit,
                    "finalOverflowRatio": round(final_overflow_ratio, 4),
                    "finalFit": final_fit,
                    "durationConstrained": duration_constrained,
                    "timingAdjusted": timing_adjusted,
                    "wordBudget": word_budget(target_ms, settings.words_per_second),
                    "wordCount": word_count(seg.description),
                    "cached": cached,
                    "billedUnits": billed_units,
                    "file": f"segments/{seg.id}.wav",
                    "attempts": attempt_rows,
                }
            )
        except ProviderRateLimitError as exc:
            log.warning("segment %s rate limited by %s", seg.id, exc.provider)
            _emit(
                progress,
                state="rate_limited",
                provider=provider_name,
                currentSegmentId=seg.id,
                currentSegmentIndex=index,
                totalSegments=total_segments,
                message=str(exc),
                lastError=str(exc),
            )
            row.update(_rate_limited_row(row, provider_name, str(exc)))
        except Exception as exc:
            log.exception("segment %s failed", seg.id)
            row["error"] = str(exc)

        report_segments.append(row)

    _emit(
        progress,
        state="assembling_track",
        provider=provider_name,
        totalSegments=total_segments,
        message=f"Assembling track for {provider_name}",
    )
    track_pcm, collisions = assemble_timeline(
        placements, last_end_ms, settings.sample_rate, settings.sample_width, settings.channels
    )
    write_wav(
        out_dir / "track.wav",
        track_pcm,
        settings.sample_rate,
        settings.sample_width,
        settings.channels,
    )
    track_duration_ms = measure_duration_ms(
        track_pcm, settings.sample_rate, settings.sample_width, settings.channels
    )

    _emit(
        progress,
        state="writing_report",
        provider=provider_name,
        totalSegments=total_segments,
        message=f"Writing report for {provider_name}",
    )
    config = resolved_config(settings, provider)
    report = build_report(
        new_run_id(), provider_name, config, report_segments, collisions, track_duration_ms, settings
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    _emit(
        progress,
        state="completed",
        provider=provider_name,
        totalSegments=total_segments,
        message=f"Completed {provider_name} run",
    )
    return report
