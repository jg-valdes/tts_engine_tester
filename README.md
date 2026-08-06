# tts-harness

A standalone Python CLI for evaluating text-to-speech providers on **timed
segments** — text that must be spoken within a fixed time window. It
synthesizes each segment, places the audio on a timeline, and reports how
well each fit its window.

Two providers are supported: **Google Gemini TTS** and **Microsoft Azure AI
Speech**, both measured and reported through identical harness code so
differences reflect the providers, not the tooling.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Quickstart

```bash
uv sync
cp .env.example .env   # fill in GEMINI_API_KEY and/or AZURE_SPEECH_KEY
uv run tts-harness synth --input segments.example.json
```

See `uv run tts-harness --help` for all commands (`synth`, `variance`,
`voices`, `compare`).

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
