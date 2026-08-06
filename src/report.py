"""JSON report generation. See docs/source.md Output."""

from __future__ import annotations

from datetime import datetime, timezone


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")


def estimate_cost_usd(settings, billed: dict) -> float:
    cost = 0.0
    if "characters" in billed:
        cost += billed["characters"] / 1_000_000 * settings.azure_usd_per_1m_chars
    if "input_tokens" in billed:
        cost += billed["input_tokens"] / 1_000_000 * settings.gemini_usd_per_1m_input_tokens
    if "output_audio_tokens" in billed:
        cost += billed["output_audio_tokens"] / 1_000_000 * settings.gemini_usd_per_1m_audio_tokens
    return round(cost, 6)


def build_totals(settings, segments: list[dict], collisions: list[dict], track_duration_ms: int) -> dict:
    billed: dict = {}
    for seg in segments:
        for key, value in (seg.get("billedUnits") or {}).items():
            billed[key] = billed.get(key, 0) + value

    return {
        "segments": len(segments),
        "natural": sum(1 for s in segments if s.get("fit") == "NATURAL"),
        "tight": sum(1 for s in segments if s.get("fit") == "TIGHT"),
        "overflow": sum(1 for s in segments if s.get("fit") == "OVERFLOW"),
        "failed": sum(1 for s in segments if s.get("error")),
        "collisions": len(collisions),
        "billed": billed,
        "estimatedCostUsd": estimate_cost_usd(settings, billed),
        "trackDurationMs": track_duration_ms,
        "cacheHits": sum(1 for s in segments if s.get("cached")),
    }


def build_report(run_id: str, provider: str, config: dict, segments: list[dict], collisions: list[dict], track_duration_ms: int, settings) -> dict:
    return {
        "runId": run_id,
        "provider": provider,
        "config": config,
        "totals": build_totals(settings, segments, collisions, track_duration_ms),
        "collisions": collisions,
        "segments": segments,
    }
