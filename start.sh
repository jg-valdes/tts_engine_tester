#!/usr/bin/env bash
set -euo pipefail

script_port="${1:-}"
resolved_port="${script_port:-${PORT:-${WEB_PORT:-8000}}}"

exec env -u UV_PYTHON uv run tts-harness web --port "${resolved_port}"
