"""CLI entry point. See docs/source.md Commands."""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import typer

from .audio import assemble_timeline, measure_duration_ms, strip_riff_header, write_wav
from .cache import Cache, cache_key
from .config import Settings, load_settings
from .fitting import classify_fit, word_budget, word_count
from .providers.azure import AzureProvider
from .providers.gemini import GeminiProvider
from .report import build_report, new_run_id

app = typer.Typer(help="Evaluate TTS providers on timed segments.")

log = logging.getLogger("tts_harness")


@dataclass
class Segment:
    id: str
    start_time: int
    end_time: int
    description: str


def get_provider(name: str, settings: Settings):
    if name == "gemini":
        return GeminiProvider(settings)
    if name == "azure":
        return AzureProvider(settings)
    raise typer.BadParameter(f"unknown provider: {name!r} (expected 'gemini' or 'azure')")


def load_segments(input_path: str) -> list[Segment]:
    data = json.loads(Path(input_path).read_text())
    segments = []
    for i, raw in enumerate(data["segments"]):
        segments.append(
            Segment(
                id=raw.get("id") or f"s{i + 1}",
                start_time=raw["startTime"],
                end_time=raw["endTime"],
                description=raw["description"],
            )
        )
    return segments


def _cli_overrides(**kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}


def _resolved_config(settings: Settings, provider) -> dict:
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
                "baseUrl": settings.gemini_base_url or "https://generativelanguage.googleapis.com (direct)",
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


def _print_table(rows: list[dict]) -> None:
    header = f"{'id':<8}{'target_ms':>10}{'actual_ms':>10}{'ratio':>8}{'fit':>10}{'cached':>8}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for r in rows:
        typer.echo(
            f"{r['id']:<8}{r['targetMs']:>10}{r.get('finalMs', r.get('naturalMs', 0)):>10}"
            f"{r.get('overflowRatio', 0.0):>8.2f}{r.get('fit', 'ERROR'):>10}{str(r.get('cached', False)):>8}"
        )


def run_synth(
    settings: Settings,
    provider_name: str,
    segments: list[Segment],
    out_dir: Path,
    dry_run: bool,
    no_cache: bool,
) -> dict:
    provider = get_provider(provider_name, settings)

    if dry_run:
        config = _resolved_config(settings, provider)
        typer.echo(json.dumps({"provider": provider_name, "config": config}, indent=2))
        return {"dryRun": True, "provider": provider_name, "config": config}

    cache = None if no_cache else Cache(settings.cache_dir)
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)

    report_segments = []
    placements: list[tuple[int, bytes]] = []
    last_end_ms = 0

    for seg in segments:
        target_ms = seg.end_time - seg.start_time
        row: dict = {
            "id": seg.id,
            "text": seg.description,
            "startTime": seg.start_time,
            "endTime": seg.end_time,
            "targetMs": target_ms,
        }
        try:
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
            if cached_wav is not None:
                pcm = strip_riff_header(cached_wav)
                natural_ms = measure_duration_ms(
                    pcm, settings.sample_rate, settings.sample_width, settings.channels
                )
                cached = True
                billed_units = {}
                duration_constrained = False
            else:
                result = provider.synthesize(seg.description, target_ms=None)
                pcm = result.pcm
                natural_ms = result.duration_ms
                billed_units = result.billed_units
                duration_constrained = result.duration_constrained
                cached = False
                if cache:
                    cache.put(
                        key,
                        _wav_bytes(pcm, settings.sample_rate, settings.sample_width, settings.channels),
                    )

            final_ms = natural_ms
            overflow_ratio, fit = classify_fit(natural_ms, target_ms, settings.max_compression_ratio)

            if settings.fit_mode == "constrain" and fit == "TIGHT":
                constrained_result = provider.synthesize(seg.description, target_ms=target_ms)
                if constrained_result.duration_constrained:
                    pcm = constrained_result.pcm
                    final_ms = constrained_result.duration_ms
                    billed_units = constrained_result.billed_units
                    duration_constrained = True

            file_path = seg_dir / f"{seg.id}.wav"
            write_wav(file_path, pcm, settings.sample_rate, settings.sample_width, settings.channels)

            placements.append((seg.start_time, pcm))
            last_end_ms = max(last_end_ms, seg.end_time)

            row.update(
                {
                    "naturalMs": natural_ms,
                    "finalMs": final_ms,
                    "overflowRatio": round(overflow_ratio, 4),
                    "fit": fit,
                    "durationConstrained": duration_constrained,
                    "wordBudget": word_budget(target_ms, settings.words_per_second),
                    "wordCount": word_count(seg.description),
                    "cached": cached,
                    "billedUnits": billed_units,
                    "file": f"segments/{seg.id}.wav",
                }
            )
        except Exception as exc:  # one failed segment must not fail the run
            log.exception("segment %s failed", seg.id)
            row["error"] = str(exc)

        report_segments.append(row)

    track_pcm, collisions = assemble_timeline(
        placements, last_end_ms, settings.sample_rate, settings.sample_width, settings.channels
    )
    write_wav(out_dir / "track.wav", track_pcm, settings.sample_rate, settings.sample_width, settings.channels)
    track_duration_ms = measure_duration_ms(
        track_pcm, settings.sample_rate, settings.sample_width, settings.channels
    )

    config = _resolved_config(settings, provider)
    report = build_report(new_run_id(), provider_name, config, report_segments, collisions, track_duration_ms, settings)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    _print_table(report_segments)
    typer.echo(
        f"\n{report['totals']['natural']} natural, {report['totals']['tight']} tight, "
        f"{report['totals']['overflow']} overflow, {report['totals']['failed']} failed "
        f"({report['totals']['cacheHits']} cache hits)"
    )
    typer.echo(f"track: {out_dir / 'track.wav'}")

    return report


def _wav_bytes(pcm: bytes, sample_rate: int, sample_width: int, channels: int) -> bytes:
    from .audio import pcm_to_wav_bytes

    return pcm_to_wav_bytes(pcm, sample_rate, sample_width, channels)


@app.command()
def synth(
    input: Optional[str] = typer.Option(None, "--input", help="Path to a segments JSON file."),
    text: Optional[str] = typer.Option(None, "--text", help="Single ad-hoc sentence, no file needed."),
    duration: Optional[int] = typer.Option(None, "--duration", help="Target duration in ms for --text."),
    provider: Optional[str] = typer.Option(None, "--provider"),
    model: Optional[str] = typer.Option(None, "--model"),
    voice: Optional[str] = typer.Option(None, "--voice"),
    style_prompt: Optional[str] = typer.Option(None, "--style-prompt"),
    fit_mode: Optional[str] = typer.Option(None, "--fit-mode"),
    out: Optional[str] = typer.Option(None, "--out"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Render segments (from --input or a single --text) and produce a report."""
    if not input and not text:
        raise typer.BadParameter("provide either --input or --text")

    overrides = _cli_overrides(
        provider=provider,
        gemini_model=model if provider == "gemini" else None,
        azure_voice=voice if provider == "azure" else None,
        gemini_voice=voice if provider == "gemini" else None,
        gemini_style_prompt=style_prompt,
        fit_mode=fit_mode,
    )
    settings = load_settings(overrides)

    if input:
        segments = load_segments(input)
    else:
        end_time = duration or 3000
        segments = [Segment(id="adhoc", start_time=0, end_time=end_time, description=text)]

    run_id = new_run_id()
    out_dir = Path(out) if out else Path(settings.output_dir) / run_id
    run_synth(settings, settings.provider, segments, out_dir, dry_run, no_cache)


@app.command()
def variance(
    text: str = typer.Option(..., "--text"),
    repeat: int = typer.Option(5, "--repeat"),
    provider: Optional[str] = typer.Option(None, "--provider"),
) -> None:
    """Repeat a single render N times and report duration variance."""
    settings = load_settings(_cli_overrides(provider=provider))
    prov = get_provider(settings.provider, settings)

    durations = []
    for i in range(repeat):
        result = prov.synthesize(text, target_ms=None)
        durations.append(result.duration_ms)
        typer.echo(f"run {i + 1}/{repeat}: {result.duration_ms} ms")

    typer.echo(
        f"\nmin={min(durations)} max={max(durations)} "
        f"mean={statistics.mean(durations):.1f} "
        f"stddev={statistics.pstdev(durations):.1f}"
    )


@app.command()
def voices(
    provider: Optional[str] = typer.Option(None, "--provider"),
) -> None:
    """List voices for the configured provider."""
    settings = load_settings(_cli_overrides(provider=provider))
    prov = get_provider(settings.provider, settings)
    for v in prov.list_voices():
        typer.echo(f"{v.name}\t{v.locale or ''}")


@app.command()
def compare(
    input: str = typer.Option(..., "--input"),
    provider: list[str] = typer.Option(..., "--provider"),
    out: str = typer.Option(..., "--out"),
    no_cache: bool = typer.Option(False, "--no-cache"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Head-to-head: same segments, multiple providers, one comparison report."""
    segments = load_segments(input)
    out_dir = Path(out)
    reports: dict[str, dict] = {}

    for prov_name in provider:
        settings = load_settings(_cli_overrides(provider=prov_name))
        typer.echo(f"\n=== {prov_name} ===")
        reports[prov_name] = run_synth(settings, prov_name, segments, out_dir / prov_name, dry_run, no_cache)

    if dry_run:
        return

    comparison_segments = []
    by_provider_segments = {p: {s["id"]: s for s in reports[p]["segments"]} for p in provider}
    for seg in segments:
        entry: dict = {"id": seg.id, "text": seg.description, "startTime": seg.start_time, "endTime": seg.end_time}
        for p in provider:
            s = by_provider_segments[p].get(seg.id, {})
            entry[p] = {
                "naturalMs": s.get("naturalMs"),
                "overflowRatio": s.get("overflowRatio"),
                "fit": s.get("fit"),
                "error": s.get("error"),
            }
        comparison_segments.append(entry)

    comparison = {
        "runId": new_run_id(),
        "providers": provider,
        "totals": {p: reports[p]["totals"] for p in provider},
        "segments": comparison_segments,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    typer.echo(f"\ncomparison: {out_dir / 'comparison.json'}")


if __name__ == "__main__":
    app()
