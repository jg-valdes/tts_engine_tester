"""PCM/WAV helpers, RIFF parsing, timeline assembly. See docs/source.md."""

from __future__ import annotations

import struct
import wave
from io import BytesIO
from pathlib import Path


def write_wav(path: str | Path, pcm: bytes, sample_rate: int, sample_width: int, channels: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm)


def pcm_to_wav_bytes(pcm: bytes, sample_rate: int, sample_width: int, channels: int) -> bytes:
    buf = BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sample_width)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def strip_riff_header(data: bytes) -> bytes:
    """Parse a RIFF/WAVE byte stream and return the raw PCM from its data chunk.

    Walks chunks rather than assuming a fixed 44-byte header, since chunk
    layout (extra fmt fields, LIST chunks, etc.) varies by encoder.
    """
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE byte stream")
    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        chunk_size = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
        chunk_start = pos + 8
        if chunk_id == b"data":
            return data[chunk_start : chunk_start + chunk_size]
        pos = chunk_start + chunk_size + (chunk_size % 2)
    raise ValueError("no data chunk found in RIFF/WAVE stream")


def measure_duration_ms(pcm: bytes, sample_rate: int, sample_width: int, channels: int) -> int:
    return len(pcm) * 1000 // (sample_rate * sample_width * channels)


def byte_offset_for_ms(ms: int, sample_rate: int, sample_width: int, channels: int) -> int:
    return ms * sample_rate * sample_width * channels // 1000


def assemble_timeline(
    placements: list[tuple[int, bytes]],
    min_length_ms: int,
    sample_rate: int,
    sample_width: int,
    channels: int,
) -> tuple[bytes, list[dict]]:
    """Place each (start_ms, pcm) clip into one zero-filled buffer.

    No inter-clip silence concatenation — every clip is placed at its
    absolute byte offset so rounding never accumulates across segments.
    Returns (timeline_bytes, collisions) where collisions record any
    byte ranges that two placed segments both wrote to.
    """
    min_length_bytes = byte_offset_for_ms(min_length_ms, sample_rate, sample_width, channels)
    computed: list[tuple[int, bytes]] = []
    max_end = min_length_bytes
    for start_ms, pcm in placements:
        offset = byte_offset_for_ms(start_ms, sample_rate, sample_width, channels)
        computed.append((offset, pcm))
        max_end = max(max_end, offset + len(pcm))

    timeline = bytearray(max_end)
    occupied: list[tuple[int, int]] = []
    collisions: list[dict] = []
    for offset, pcm in computed:
        end = offset + len(pcm)
        for occ_start, occ_end in occupied:
            if offset < occ_end and end > occ_start:
                collisions.append({"rangeStart": max(offset, occ_start), "rangeEnd": min(end, occ_end)})
        timeline[offset:end] = pcm
        occupied.append((offset, end))

    return bytes(timeline), collisions
