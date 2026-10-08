"""SQLite store (swap for Postgres in the hosted version; the schema is the same)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY, app TEXT, experiment TEXT, created TEXT, brain TEXT, n_per_cell INT,
  params TEXT, calibration_id INT, cost_usd REAL, seconds REAL, summary TEXT, report_path TEXT);
CREATE TABLE IF NOT EXISTS sessions (
  id TEXT, run_id TEXT, segment_id TEXT, variant_id TEXT, traits TEXT, outcome TEXT,
  abandon_step INT, abandon_concern TEXT, cost_usd REAL, PRIMARY KEY (run_id, id));
CREATE TABLE IF NOT EXISTS steps (
  run_id TEXT, session_id TEXT, idx INT, data TEXT, PRIMARY KEY (run_id, session_id, idx));
CREATE TABLE IF NOT EXISTS findings (
  run_id TEXT, variant_id TEXT, concern TEXT, share REAL, data TEXT);
CREATE TABLE IF NOT EXISTS calibrations (
  id INTEGER PRIMARY KEY AUTOINCREMENT, app TEXT, variant_id TEXT, created TEXT, source TEXT,
  observed TEXT, fitted TEXT, fit_error REAL, status TEXT);
CREATE TABLE IF NOT EXISTS backtests (
  id INTEGER PRIMARY KEY AUTOINCREMENT, app TEXT, change TEXT, created TEXT,
  predicted TEXT, actual TEXT, hit INT, direction_hit INT, data TEXT);
"""


class Store:
    def __init__(self, path: str | Path = "standin.db"):
        self.path = Path(path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def save_run(self, result, analysis, report_path: Optional[Path], calibration_id: Optional[int]) -> None:
        from dataclasses import asdict
        exp = result.exp
        summary = {"overall": analysis.overall, "rates": analysis.rates,
                   "comparisons": [asdict(c) for c in analysis.comparisons]}
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                result.run_id, exp.app, exp.name, datetime.now().isoformat(timespec="seconds"), result.brain,
                result.n_per_cell, result.params.model_dump_json(), calibration_id, result.cost_usd,
                result.seconds, json.dumps(summary), str(report_path) if report_path else None))
            for s in result.sessions:
                self.db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?)", (
                    s.id, s.run_id, s.segment_id, s.variant_id, json.dumps(s.traits), s.outcome,
                    s.abandon_step, s.abandon_concern, s.cost_usd))
                self.db.executemany("INSERT OR REPLACE INTO steps VALUES (?,?,?,?)",
                                    [(s.run_id, s.id, st.idx, st.model_dump_json()) for st in s.steps])
            for f in analysis.findings:
                self.db.execute("INSERT INTO findings VALUES (?,?,?,?,?)",
                                (result.run_id, f.variant_id, f.concern, f.share, json.dumps(asdict(f))))

    def save_calibration(self, app: str, variant_id: str, source: str, observed: dict, fitted: dict,
                         fit_error: float, status: str) -> int:
        with self.db:
            cur = self.db.execute(
                "INSERT INTO calibrations (app, variant_id, created, source, observed, fitted, fit_error, status) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (app, variant_id, datetime.now().isoformat(timespec="seconds"), source, json.dumps(observed),
                 json.dumps(fitted), fit_error, status))
            return int(cur.lastrowid)

    def latest_calibration(self, app: str) -> Optional[dict]:
        r = self.db.execute("SELECT * FROM calibrations WHERE app=? ORDER BY id DESC LIMIT 1", (app,)).fetchone()
        if not r:
            return None
        d = dict(r)
        d["observed"], d["fitted"] = json.loads(d["observed"]), json.loads(d["fitted"])
        return d

    def save_backtest(self, app: str, change: str, predicted: str, actual: str, hit: bool,
                      direction_hit: Optional[bool], data: dict) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO backtests (app, change, created, predicted, actual, hit, direction_hit, data) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (app, change, datetime.now().isoformat(timespec="seconds"), predicted, actual, int(hit),
                 None if direction_hit is None else int(direction_hit), json.dumps(data)))

    def backtest_summary(self, app: Optional[str] = None) -> dict:
        q = "SELECT hit, direction_hit FROM backtests" + (" WHERE app=?" if app else "")
        rows = self.db.execute(q, (app,) if app else ()).fetchall()
        n = len(rows)
        dir_rows = [r["direction_hit"] for r in rows if r["direction_hit"] is not None]
        return {"n": n, "hits": sum(r["hit"] for r in rows),
                "direction_n": len(dir_rows), "direction_hits": sum(dir_rows)}

    def runs(self) -> list[dict]:
        return [dict(r) for r in self.db.execute(
            "SELECT id, app, experiment, created, brain, n_per_cell, cost_usd, seconds, report_path FROM runs ORDER BY created DESC")]
