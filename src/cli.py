"""CLI entry point. See docs/source.md Commands."""

from __future__ import annotations

import typer

app = typer.Typer(help="Evaluate TTS providers on timed segments.")


@app.command()
def synth() -> None:
    """Render segments and produce a per-provider report. Not yet implemented."""
    raise NotImplementedError


@app.command()
def variance() -> None:
    """Repeat a single render N times and report duration variance. Not yet implemented."""
    raise NotImplementedError


@app.command()
def voices() -> None:
    """List voices for the configured provider. Not yet implemented."""
    raise NotImplementedError


@app.command()
def compare() -> None:
    """Head-to-head run across providers with an aligned comparison report. Not yet implemented."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
