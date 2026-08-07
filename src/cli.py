"""CLI entry point. See docs/source.md Commands."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Optional

import typer

from .config import load_settings
from .report import new_run_id
from .runner import (
    Segment,
    build_comparison,
    get_provider,
    load_segments,
    resolved_config,
    run_synth,
)
from .web import create_app

app = typer.Typer(help="Evaluate TTS providers on timed segments.")


def _cli_overrides(**kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if v is not None}


def _print_table(rows: list[dict]) -> None:
    header = f"{'id':<8}{'target_ms':>10}{'actual_ms':>10}{'ratio':>8}{'fit':>10}{'cached':>8}"
    typer.echo(header)
    typer.echo("-" * len(header))
    for row in rows:
        typer.echo(
            f"{row['id']:<8}{row['targetMs']:>10}{row.get('finalMs', row.get('naturalMs', 0)):>10}"
            f"{row.get('finalOverflowRatio', row.get('overflowRatio', 0.0)):>8.2f}"
            f"{row.get('finalFit', row.get('fit', 'ERROR')):>10}{str(row.get('cached', False)):>8}"
        )


def _print_report(report: dict, out_dir: Path) -> None:
    _print_table(report["segments"])
    typer.echo(
        f"\n{report['totals']['natural']} natural, {report['totals']['tight']} tight, "
        f"{report['totals']['overflow']} overflow, {report['totals']['failed']} failed "
        f"({report['totals']['cacheHits']} cache hits)"
    )
    typer.echo(f"track: {out_dir / 'track.wav'}")


@app.command()
def synth(
    input: Optional[str] = typer.Option(None, "--input", help="Path to a segments JSON file."),
    text: Optional[str] = typer.Option(None, "--text", help="Single ad-hoc sentence, no file needed."),
    duration: Optional[int] = typer.Option(None, "--duration", help="Target duration in ms for --text."),
    provider: Optional[str] = typer.Option(None, "--provider"),
    model: Optional[str] = typer.Option(None, "--model"),
    voice: Optional[str] = typer.Option(None, "--voice"),
    style_prompt: Optional[str] = typer.Option(None, "--style-prompt"),
    timing_attempts: Optional[int] = typer.Option(None, "--timing-attempts", min=1),
    timing_tolerance_ms: Optional[int] = typer.Option(None, "--timing-tolerance-ms", min=0),
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
        gemini_timing_attempts=timing_attempts,
        gemini_timing_tolerance_ms=timing_tolerance_ms,
        fit_mode=fit_mode,
    )
    settings = load_settings(overrides)

    if input:
        segments = load_segments(input)
    else:
        end_time = duration or 3000
        segments = [Segment(id="adhoc", start_time=0, end_time=end_time, description=text or "")]

    run_id = new_run_id()
    out_dir = Path(out) if out else Path(settings.output_dir) / run_id
    report = run_synth(settings, settings.provider, segments, out_dir, dry_run, no_cache)
    if dry_run:
        typer.echo(json.dumps(report, indent=2))
        return
    _print_report(report, out_dir)


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
    for voice in prov.list_voices():
        details = [value for value in (voice.locale, voice.description) if value]
        typer.echo(" -> ".join([voice.name, *details]))


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

    for provider_name in provider:
        settings = load_settings(_cli_overrides(provider=provider_name))
        typer.echo(f"\n=== {provider_name} ===")
        report = run_synth(settings, provider_name, segments, out_dir / provider_name, dry_run, no_cache)
        reports[provider_name] = report
        if dry_run:
            typer.echo(json.dumps(report, indent=2))
        else:
            _print_report(report, out_dir / provider_name)

    if dry_run:
        return

    comparison = build_comparison(provider, reports, segments)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    typer.echo(f"\ncomparison: {out_dir / 'comparison.json'}")


@app.command()
def web(
    host: Optional[str] = typer.Option(None, "--host"),
    port: Optional[int] = typer.Option(None, "--port"),
) -> None:
    """Run the local web UI for configuring and testing renders."""
    settings = load_settings(_cli_overrides(web_host=host, web_port=port))
    import uvicorn

    uvicorn.run(create_app(settings=settings), host=settings.web_host, port=settings.web_port)


if __name__ == "__main__":
    app()
