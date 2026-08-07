"""SQLite persistence for presets, runs, and progress events."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS presets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    config_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    parent_run_id TEXT,
                    provider TEXT,
                    providers_json TEXT,
                    status TEXT NOT NULL,
                    message TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    output_dir TEXT,
                    report_path TEXT,
                    comparison_path TEXT,
                    input_payload_json TEXT,
                    resolved_config_json TEXT,
                    totals_json TEXT,
                    progress_json TEXT,
                    error TEXT
                );

                CREATE TABLE IF NOT EXISTS run_segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    provider TEXT,
                    ordinal INTEGER NOT NULL,
                    data_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    segment_id TEXT NOT NULL,
                    provider TEXT,
                    attempt_key TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS run_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_json TEXT NOT NULL
                );
                """
            )

    def list_presets(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, name, config_json, created_at, updated_at FROM presets ORDER BY name"
            ).fetchall()
        return [self._preset_row(row) for row in rows]

    def save_preset(self, name: str, config: dict) -> dict:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO presets (name, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    config_json=excluded.config_json,
                    updated_at=excluded.updated_at
                """,
                (name, json.dumps(config), now, now),
            )
            row = conn.execute(
                "SELECT id, name, config_json, created_at, updated_at FROM presets WHERE name = ?",
                (name,),
            ).fetchone()
        return self._preset_row(row)

    def create_run(
        self,
        run_id: str,
        kind: str,
        status: str,
        *,
        provider: str | None = None,
        providers: list[str] | None = None,
        parent_run_id: str | None = None,
        input_payload: dict | None = None,
        resolved_config: dict | None = None,
        output_dir: str | None = None,
        message: str | None = None,
    ) -> None:
        now = _utcnow()
        progress_json = json.dumps({"state": status, "message": message}) if message else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    id, kind, parent_run_id, provider, providers_json, status, message,
                    created_at, updated_at, output_dir, input_payload_json,
                    resolved_config_json, progress_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    kind,
                    parent_run_id,
                    provider,
                    json.dumps(providers) if providers else None,
                    status,
                    message,
                    now,
                    now,
                    output_dir,
                    json.dumps(input_payload) if input_payload is not None else None,
                    json.dumps(resolved_config) if resolved_config is not None else None,
                    progress_json,
                ),
            )

    def update_run_status(self, run_id: str, status: str, progress: dict) -> None:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, message = ?, updated_at = ?, progress_json = ?, error = ?
                WHERE id = ?
                """,
                (
                    status,
                    progress.get("message"),
                    now,
                    json.dumps(progress),
                    progress.get("lastError"),
                    run_id,
                ),
            )

    def append_event(self, run_id: str, event: dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO run_events (run_id, created_at, event_json) VALUES (?, ?, ?)",
                (run_id, _utcnow(), json.dumps(event)),
            )

    def finalize_synth_run(
        self,
        run_id: str,
        report: dict,
        *,
        output_dir: str,
        input_payload: dict,
        resolved_config: dict,
    ) -> None:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute("DELETE FROM run_segments WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM run_attempts WHERE run_id = ?", (run_id,))
            for ordinal, segment in enumerate(report["segments"], start=1):
                conn.execute(
                    """
                    INSERT INTO run_segments (run_id, segment_id, provider, ordinal, data_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        segment["id"],
                        report["provider"],
                        ordinal,
                        json.dumps(segment),
                    ),
                )
                for attempt in segment.get("attempts", []):
                    conn.execute(
                        """
                        INSERT INTO run_attempts (run_id, segment_id, provider, attempt_key, data_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            segment["id"],
                            report["provider"],
                            f"{attempt['phase']}-{attempt['ordinal']}",
                            json.dumps(attempt),
                        ),
                    )
            conn.execute(
                """
                UPDATE runs
                SET status = ?, message = ?, updated_at = ?, output_dir = ?, report_path = ?,
                    input_payload_json = ?, resolved_config_json = ?, totals_json = ?, progress_json = ?, error = NULL
                WHERE id = ?
                """,
                (
                    "completed",
                    "Run completed",
                    now,
                    output_dir,
                    str(Path(output_dir) / "report.json"),
                    json.dumps(input_payload),
                    json.dumps(resolved_config),
                    json.dumps(report["totals"]),
                    json.dumps({"state": "completed", "message": "Run completed"}),
                    run_id,
                ),
            )

    def finalize_compare_run(
        self,
        run_id: str,
        comparison: dict,
        *,
        output_dir: str,
    ) -> None:
        now = _utcnow()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE runs
                SET status = ?, message = ?, updated_at = ?, output_dir = ?, comparison_path = ?,
                    totals_json = ?, progress_json = ?, error = NULL
                WHERE id = ?
                """,
                (
                    "completed",
                    "Compare completed",
                    now,
                    output_dir,
                    str(Path(output_dir) / "comparison.json"),
                    json.dumps(comparison["totals"]),
                    json.dumps({"state": "completed", "message": "Compare completed"}),
                    run_id,
                ),
            )

    def fail_run(self, run_id: str, message: str) -> None:
        self.update_run_status(
            run_id,
            "failed",
            {
                "state": "failed",
                "message": message,
                "lastError": message,
            },
        )

    def list_runs(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM runs
                WHERE parent_run_id IS NULL
                ORDER BY created_at DESC
                """
            ).fetchall()
        return [self._run_row(row) for row in rows]

    def get_run(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                return None
            run = self._run_row(row)
            run["segments"] = self._segments(conn, run_id)
            run["events"] = self._events(conn, run_id)
            run["children"] = [
                self._run_row(child)
                for child in conn.execute(
                    "SELECT * FROM runs WHERE parent_run_id = ? ORDER BY created_at",
                    (run_id,),
                ).fetchall()
            ]
        return run

    def get_progress(self, run_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute("SELECT progress_json, status FROM runs WHERE id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        progress = json.loads(row["progress_json"]) if row["progress_json"] else {"state": row["status"]}
        progress.setdefault("state", row["status"])
        return progress

    def _segments(self, conn: sqlite3.Connection, run_id: str) -> list[dict]:
        segment_rows = conn.execute(
            "SELECT segment_id, data_json FROM run_segments WHERE run_id = ? ORDER BY ordinal",
            (run_id,),
        ).fetchall()
        attempts_by_segment: dict[str, list[dict]] = {}
        for attempt_row in conn.execute(
            "SELECT segment_id, data_json FROM run_attempts WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall():
            attempts_by_segment.setdefault(attempt_row["segment_id"], []).append(
                json.loads(attempt_row["data_json"])
            )
        segments = []
        for row in segment_rows:
            segment = json.loads(row["data_json"])
            segment["attempts"] = attempts_by_segment.get(row["segment_id"], [])
            segments.append(segment)
        return segments

    def _events(self, conn: sqlite3.Connection, run_id: str) -> list[dict]:
        rows = conn.execute(
            "SELECT created_at, event_json FROM run_events WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
        return [{"createdAt": row["created_at"], **json.loads(row["event_json"])} for row in rows]

    def _preset_row(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "name": row["name"],
            "config": json.loads(row["config_json"]),
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
        }

    def _run_row(self, row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "kind": row["kind"],
            "parentRunId": row["parent_run_id"],
            "provider": row["provider"],
            "providers": json.loads(row["providers_json"]) if row["providers_json"] else [],
            "status": row["status"],
            "message": row["message"],
            "createdAt": row["created_at"],
            "updatedAt": row["updated_at"],
            "outputDir": row["output_dir"],
            "reportPath": row["report_path"],
            "comparisonPath": row["comparison_path"],
            "inputPayload": json.loads(row["input_payload_json"]) if row["input_payload_json"] else None,
            "resolvedConfig": (
                json.loads(row["resolved_config_json"]) if row["resolved_config_json"] else None
            ),
            "totals": json.loads(row["totals_json"]) if row["totals_json"] else None,
            "progress": json.loads(row["progress_json"]) if row["progress_json"] else None,
            "error": row["error"],
        }
