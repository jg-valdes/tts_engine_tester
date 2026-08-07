# tts-harness

A standalone Python CLI for evaluating text-to-speech providers on **timed
segments** — text that must be spoken within a fixed time window. It
synthesizes each segment, places the audio on a timeline, and reports how
well each fit its window.

Two providers are supported: **Google Gemini TTS** and **Microsoft Azure AI
Speech**, both measured and reported through identical harness code so
differences reflect the providers, not the tooling.

## Requirements

- [uv](https://docs.astral.sh/uv/)

Python 3.11 is pinned in `.python-version`. On supported platforms, `uv`
automatically installs a compatible Python interpreter when it is not already
available, so a separate Python installation is not required.

## Quickstart

```bash
uv sync
cp .env.example .env   # fill in GEMINI_API_KEY and/or AZURE_SPEECH_KEY
uv run tts-harness synth --input segments.example.json
```

The default `measure` mode preserves each provider's natural duration. To ask
Gemini for best-effort timing that more closely fits every segment window, use:

```bash
uv run tts-harness synth --input segments.example.json \
  --provider gemini --fit-mode constrain \
  --timing-attempts 2 --timing-tolerance-ms 150
```

Gemini has prompt-based pace control rather than an exact duration parameter.
In constrain mode the harness tries targeted renders and keeps the one closest
to each window; `GEMINI_TIMING_ATTEMPTS` and `GEMINI_TIMING_TOLERANCE_MS`
control the quota/precision tradeoff.

See `uv run tts-harness --help` for all commands (`synth`, `variance`, `voices`, `compare`).

Google does not expose a Gemini TTS voice-list endpoint. The harness includes
the 30 voices documented by Google (last checked 2026-08-07), so they can be
listed without credentials or a network call:

```bash
uv run tts-harness voices --provider gemini
```

If uv reports `No interpreter found for executable name \`system\``, remove the
invalid interpreter override from the current shell before running uv:

```bash
unset UV_PYTHON
uv sync
```

The repository pin cannot override an explicitly exported `UV_PYTHON` value.

## Docs

- [`docs/source.md`](docs/source.md) — full design spec: provider
  interface, configuration, fitting rules, caching, output format,
  acceptance criteria.
- [`.agents/instructions.md`](.agents/instructions.md) — conventions for
  coding agents working in this repo (also available as `AGENTS.md` /
  `CLAUDE.md` symlinks).

## Non-goals

Video muxing, ffmpeg integration, a web UI, production hardening, and
providers beyond Gemini and Azure.
