# Agent instructions — tts-harness

This repo is a standalone Python CLI for evaluating TTS providers (Gemini,
Azure) on timed segments. It is an evaluation tool, not a production
service.

## Dependencies: uv only

This repo uses **uv** for dependency management. Do not use bare `pip` or
`python -m venv`.

- `uv sync` — install dependencies into `.venv`.
- `uv run <cmd>` — run anything inside the project environment, e.g.
  `uv run tts-harness synth --input segments.example.json`.
- `uv add <package>` — add a new dependency (updates `pyproject.toml` and
  `uv.lock`).

## Spec of record

`docs/source.md` is the authoritative design/spec document for this
project — provider interface, configuration precedence, fitting rules,
caching, timeline assembly, output format, and acceptance criteria all
live there. Read it before implementing or changing provider, CLI, or
report behavior. This file only covers meta/contribution conventions.

## Local scratch space

`.agents/local/` is gitignored. Put personal/local plans, notes, or
scratch work there — it will never be committed.

## User-facing docs

See `README.md` for install and usage instructions aimed at humans.
