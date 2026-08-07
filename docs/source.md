# Build a TTS Evaluation Harness (Python)

## Objective

Build a standalone Python CLI tool for evaluating text-to-speech providers on **timed segments** — text that must be spoken within a fixed time window.

The tool takes a list of segments (`startTime`, `endTime`, `description`), synthesizes each one, places them on a timeline, and reports how well each fit its window.

Two providers must be supported: **Google Gemini TTS** and **Microsoft Azure AI Speech**.

This is an evaluation tool, not a production service. Optimize for flexibility and observability. Anything hardcoded is something that cannot be tested.

## Why the design is shaped this way

The tool exists to answer three questions with data:

1. **Quality** — which provider sounds better on real text.
2. **Fit** — what percentage of segments fit their time window at a natural speaking rate.
3. **Consistency** — whether repeated renders of the same text produce the same duration.

That third question matters because the providers differ fundamentally. Azure is a conventional TTS engine: deterministic, SSML-driven, and it accepts a **target output duration** (`mstts:audioduration`). Gemini is LLM-native: higher ceiling on naturalness and steerable with plain-language direction, but non-deterministic and with **no duration control at all**.

Both providers must flow through identical measurement, assembly and reporting code, so that differences in the output reflect the providers rather than the harness.

## Step 0 — verify both APIs before writing the project

Both APIs may differ from any code sample in your training data. `gemini-3.1-flash-tts-preview` in particular is a preview model.

1. Make one minimal call to each provider.
2. Print the full response structure — part types, content type, encoding, sample rate.
3. Confirm the audio container for each. Expected: **Gemini returns raw headerless PCM** (base64 inline data, 24 kHz / 16-bit / mono); **Azure returns a RIFF/WAV byte stream** when `X-Microsoft-OutputFormat: riff-24khz-16bit-mono-pcm` is requested.
4. Report what you found, then build against reality.

That container difference is the main integration trap. Normalize both to raw PCM immediately after the call, so nothing downstream knows which provider produced the bytes.

## Project structure

```
tts-harness/
├── .env.example
├── pyproject.toml
├── README.md
├── segments.example.json
├── src/
│   ├── config.py            # env + CLI merge, precedence rules
│   ├── providers/
│   │   ├── base.py          # Provider protocol
│   │   ├── gemini.py
│   │   └── azure.py
│   ├── audio.py             # PCM/WAV helpers, timeline assembly
│   ├── fitting.py           # word budget, duration classification
│   ├── cache.py             # content-addressed cache
│   ├── report.py            # JSON report generation
│   └── cli.py               # entry point
└── tests/
```

Python 3.11+. Dependencies are managed with **uv** (`pyproject.toml` + `uv.lock`, no `requirements.txt`): `google-genai`, `httpx`, `python-dotenv`, `typer` (or `click`), `pydantic`, added via `uv add <package>`.

Use the stdlib `wave` module for WAV output. **Do not depend on ffmpeg or pydub** — the tool must run anywhere with a single `uv sync`.

Azure needs no SDK; it is a plain HTTPS POST via `httpx`.

## Provider interface

```python
class Provider(Protocol):
    name: str

    def synthesize(
        self,
        text: str,
        *,
        target_ms: int | None = None,   # None = natural rate
    ) -> SynthesisResult: ...

    def list_voices(self) -> list[Voice]: ...
```

```python
@dataclass
class SynthesisResult:
    pcm: bytes                  # raw, headerless, normalized
    duration_ms: int            # measured from pcm, never from the API
    duration_constrained: bool  # did the provider actually target a duration
    billed_units: dict          # {"characters": int} or {"input_tokens": int, "output_audio_tokens": int}
    request_payload: str        # SSML or prompt, for the report and --dry-run
```

`target_ms` is **advisory**. Azure provides hard duration targeting; Gemini
uses it for best-effort prompt-based pace correction and must return
`duration_constrained=False`. Calling code branches on provider capabilities,
never on the provider name — no `if provider == "azure"` checks outside the
provider modules.

## Configuration

Every setting resolves in this order: **CLI argument → `.env` → built-in default**. No setting may be reachable only by editing code.

`.env.example`:

```
# Which provider by default
PROVIDER=gemini                   # gemini | azure

# --- Gemini ---
GEMINI_API_KEY=
GEMINI_BASE_URL=                  # optional: custom/proxy base URL, blank = Google direct
GEMINI_MODEL=gemini-3.1-flash-tts-preview
GEMINI_VOICE=Charon
GEMINI_STYLE_PROMPT=Read as a neutral documentary narrator. American English. Even pacing, no emotional colour. Do not add, omit or reorder any words.
GEMINI_TIMING_ATTEMPTS=2
GEMINI_TIMING_TOLERANCE_MS=150

# --- Azure ---
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=eastus
AZURE_VOICE=en-US-AvaMultilingualNeural
AZURE_OUTPUT_FORMAT=riff-24khz-16bit-mono-pcm
AZURE_STYLE=                      # optional mstts:express-as style, blank = none

# --- Audio (must match across providers for comparison to be valid) ---
SAMPLE_RATE=24000
SAMPLE_WIDTH=2
CHANNELS=1

# --- Fitting ---
WORDS_PER_SECOND=2.5
MAX_COMPRESSION_RATIO=1.15
FIT_MODE=measure                  # measure | constrain

# --- Cost rates (correctable without a code change) ---
AZURE_USD_PER_1M_CHARS=16.0
GEMINI_USD_PER_1M_INPUT_TOKENS=1.0
GEMINI_USD_PER_1M_AUDIO_TOKENS=20.0

# --- Runtime ---
OUTPUT_DIR=./out
CACHE_DIR=./.cache
MAX_CONCURRENCY=2                 # Gemini preview rate limits are tight
LOG_LEVEL=INFO
```

`GEMINI_BASE_URL` supports routing through a proxy instead of calling Google directly. Support both paths and log which one was used.

## Input format

`segments.example.json`:

```json
{
  "segments": [
    { "id": "s1", "startTime": 1500, "endTime": 4900,
      "description": "A woman crosses the street holding a red umbrella." },
    { "id": "s2", "startTime": 7200, "endTime": 9000,
      "description": "The camera pans across an empty warehouse floor." }
  ]
}
```

`startTime` and `endTime` accept integer milliseconds, `HH:MM:SS:mmm`, or
`HH:MM:SS:CC` centiseconds. Normalize them to integer milliseconds at input
load time. `id` is optional — generate one if absent.

## Commands

```bash
# Full run, provider from .env
uv run tts-harness synth --input segments.json

# Explicit provider + overrides
uv run tts-harness synth --input segments.json --provider azure \
    --voice en-US-AndrewMultilingualNeural --fit-mode constrain --out ./out/azure-a

uv run tts-harness synth --input segments.json --provider gemini \
    --model gemini-3.1-flash-tts-preview \
    --style-prompt "Calm narrator, slightly slower than natural."

# Single ad-hoc sentence, no file needed
uv run tts-harness synth --text "A woman crosses the street." --duration 3400 --provider azure

# Consistency check: same text N times, report duration variance
uv run tts-harness variance --text "A woman crosses the street." --repeat 5 --provider gemini

# Voices for the configured provider
uv run tts-harness voices --provider azure

# Head-to-head: same segments, both providers, one comparison report
uv run tts-harness compare --input segments.json \
    --provider gemini --provider azure \
    --out ./out/compare-1

# Local browser UI for iterative testing
uv run tts-harness web
```

Add `--no-cache` to force fresh renders and `--dry-run` to print the fully resolved config and the exact payload (SSML or prompt) without making a call.

`compare` is the most important command. It must produce per-provider output directories plus a single top-level `comparison.json` aligning the same segment across providers side by side.

## Provider specifics

### Gemini

- Use the Interactions API: `client.interactions.create(...)` with
  `response_format={"type": "audio"}` and
  `generation_config={"speech_config": [{"voice": VOICE}]}`. Read audio from
  `interaction.output_audio.data` and usage from `interaction.usage`.
- **No SSML.** Send the style prompt and the segment text as **clearly separated parts**, never concatenated into one blob. Instructions bleeding into spoken output is the most common failure mode with LLM-native TTS, and prompt separation is the main defence against it.
- Response is raw headerless PCM — no stripping needed, but a WAV header must be added on write.
- Gemini has no hard duration parameter, so always return
  `duration_constrained=False`. When `target_ms` is provided in `constrain`
  mode, add explicit duration and words-per-second direction, retry up to
  `GEMINI_TIMING_ATTEMPTS`, and return the render closest to the target. Stop
  early within `GEMINI_TIMING_TOLERANCE_MS`. This is best-effort prompt control,
  not a duration guarantee.
- Billing: audio output at **25 tokens per second of audio**. Record input and output tokens separately.
- Retry on 429/5xx with exponential backoff. Preview rate limits are tight.
- Google provides no voice-list endpoint. `list_voices()` returns the static set
  of 30 prebuilt voice ids and style descriptions from Google's speech
  generation guide, last verified on 2026-08-07. Keep arbitrary configured
  voice ids pass-through so newly released voices can be evaluated before this
  snapshot is refreshed.

### Azure

| Item | Value |
|---|---|
| URL | `https://{region}.tts.speech.microsoft.com/cognitiveservices/v1` |
| Method | POST |
| `Ocp-Apim-Subscription-Key` | `AZURE_SPEECH_KEY` |
| `Content-Type` | `application/ssml+xml` |
| `X-Microsoft-OutputFormat` | `AZURE_OUTPUT_FORMAT` |
| `User-Agent` | required by the service — send an identifier |

Voices: `GET https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list`

SSML at natural rate:

```xml
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">
  <voice name="en-US-AvaMultilingualNeural">TEXT</voice>
</speak>
```

Duration-constrained (only when `target_ms` is set and `FIT_MODE=constrain`):

```xml
<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis"
       xmlns:mstts="http://www.w3.org/2001/mstts" xml:lang="en-US">
  <voice name="en-US-AvaMultilingualNeural">
    <mstts:audioduration value="3400ms"/>
    TEXT
  </voice>
</speak>
```

- Escape `&`, `<`, `>` in the text before insertion.
- `mstts:audioduration` is valid only within 0.5×–2× of natural duration. Outside that range it is clamped, not honoured — **always verify by measuring the returned audio**, never trust the requested value.
- Response is RIFF/WAV — strip the header to get raw PCM. **Parse the header rather than assuming 44 bytes.**
- Billing counts the **full SSML string**, including tags and whitespace. Set `billed_units["characters"] = len(ssml)` and keep the emitted SSML minimal.

## Fitting

Per segment, with `target_ms = endTime - startTime`:

```
word_budget    = floor(target_ms / 1000 * WORDS_PER_SECOND)
natural_ms     = measured duration of the natural-rate render
overflow_ratio = natural_ms / target_ms

fit:
  overflow_ratio <= 1.0                         -> "NATURAL"
  1.0 < overflow_ratio <= MAX_COMPRESSION_RATIO -> "TIGHT"
  overflow_ratio > MAX_COMPRESSION_RATIO        -> "OVERFLOW"
```

`FIT_MODE` controls what happens next:

- **`measure`** (default) — classify and stop. Both providers behave identically, which is what makes the head-to-head fair.
- **`constrain`** — re-render `TIGHT` and `OVERFLOW` segments with `target_ms`
  and record `final_ms` alongside `natural_ms`. Providers with hard duration
  control report `durationConstrained: true`; best-effort providers may supply
  a closer render with `timingAdjusted: true` while leaving
  `durationConstrained: false`.

Measure duration from the PCM byte count, never from any value the API reports:

```python
duration_ms = len(pcm_bytes) * 1000 // (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
```

## Timeline assembly

Do **not** concatenate clips with silence between them. Rounding accumulates across hundreds of segments and the track drifts out of sync.

Allocate one zero-filled buffer for the full timeline (zeros are silence in signed PCM), then place each segment at its absolute byte offset:

```python
byte_offset = start_ms * SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS // 1000
timeline[byte_offset:byte_offset + len(pcm)] = pcm
```

Timeline length is the last `endTime`, or the end of the longest overflowing segment, whichever is greater. If two segments overlap because one overflowed, place both and **flag the collision in the report** — do not silently truncate.

## Caching

Content-addressed: `sha256(provider + model + voice + style_or_ssml + text + target_ms + sample_rate)` → `CACHE_DIR/<hash>.wav`.

This matters more than usual here. Gemini is non-deterministic — the same input produces different audio on each call. Re-rendering during development both burns quota and silently changes results between runs. The cache is what makes repeated runs comparable.

## Output

```
out/<run-id>/
├── segments/s1.wav, s2.wav, ...
├── track.wav
├── report.json
└── run.log
```

`report.json`:

```json
{
  "runId": "2026-08-06T14-22-01",
  "provider": "azure",
  "config": { "model": "...", "voice": "...", "fitMode": "measure", "payloadSample": "..." },
  "totals": {
    "segments": 12,
    "natural": 9, "tight": 2, "overflow": 1, "failed": 0,
    "collisions": 0,
    "billed": { "characters": 1840 },
    "estimatedCostUsd": 0.0294,
    "trackDurationMs": 84000,
    "cacheHits": 4
  },
  "segments": [
    {
      "id": "s1",
      "text": "A woman crosses the street holding a red umbrella.",
      "startTime": 1500, "endTime": 4900,
      "targetMs": 3400, "naturalMs": 3180, "finalMs": 3180,
      "overflowRatio": 0.94,
      "fit": "NATURAL",
      "durationConstrained": false,
      "wordBudget": 8, "wordCount": 8,
      "cached": false,
      "file": "segments/s1.wav"
    }
  ]
}
```

`comparison.json` (from `compare`) aligns the same segment id across providers, with each provider's `naturalMs`, `overflowRatio` and `fit`, plus a per-provider totals block.

## Console output

Print a compact table as the run proceeds — id, target, actual, ratio, fit, cached. This will be watched live. End with the totals and the path to `track.wav`. For `compare`, print the providers as adjacent columns.

## Error handling

**One failed segment must not fail the run.** Record the error in the report and continue. A run that dies on segment 7 of 200 wastes everything before it.

## Acceptance criteria

1. `synth --input segments.example.json` works against **both** providers and produces per-segment WAVs, `track.wav`, and a valid `report.json`.
2. `track.wav` total duration matches the last `endTime` within 5 ms.
3. Every segment's audio starts at its `startTime` within 5 ms — verified by a test using synthetic PCM, not a live API call.
4. Provider, model, voice, and style prompt or SSML style are each overridable from the CLI, and the resolved values appear in `report.json`.
5. `compare --provider gemini --provider azure` produces both runs plus a `comparison.json` aligned by segment id.
6. `--fit-mode constrain` produces Azure segments with
   `durationConstrained: true` and both `naturalMs` and `finalMs` recorded. For
   Gemini it records the closest prompt-adjusted render as `finalMs`, sets
   `timingAdjusted: true`, and leaves `durationConstrained: false`.
7. `variance --repeat 5` reports min/max/mean/stddev of duration for the same input, on either provider.
8. A forced API failure on one segment leaves the run completing, with that segment marked failed.
9. `--dry-run` prints the resolved config and the exact payload without making a call.
10. Unit tests cover WAV header parsing and writing, RIFF-header stripping, duration measurement, byte-offset placement, word-budget calculation, and SSML escaping. Assembly and SSML tests must not require network access.

## Local web UI

Add a local-only browser UI on top of the same orchestration used by the CLI.
It should:

- render defaults from `.env`
- allow temporary per-run overrides in the UI
- persist non-secret presets and run history in SQLite
- mask secrets by default with a show/hide toggle
- let the user launch `synth` and `compare` jobs
- expose live progress states while runs are active, including waiting on a provider response
- play `track.wav`, per-segment WAVs, and saved attempt WAVs

This remains an evaluation tool, not a multi-user service.

## Non-goals

Video muxing, ffmpeg integration, production hardening, and providers beyond these two. Keep the surface small.

## Optional stretch

`--validate-stt` — transcribe each rendered segment back to text and report word error rate against the source. This detects a model reading style instructions aloud, inserting filler, or dropping words. Skip if it adds meaningful complexity and flag it as a follow-up.
