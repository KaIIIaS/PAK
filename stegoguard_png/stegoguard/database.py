"""Журнал проверок (SQLite) для аудита и расследования инцидентов."""
from __future__ import annotations

import json
import sqlite3
import threading
from typing import Dict, List, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    file TEXT NOT NULL,
    sha256 TEXT,
    verdict TEXT NOT NULL,
    risk REAL,
    payload REAL,
    findings TEXT,
    report TEXT
);
CREATE INDEX IF NOT EXISTS idx_scans_sha ON scans(sha256);
CREATE INDEX IF NOT EXISTS idx_scans_verdict ON scans(verdict);
"""


class History:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        with self._connect() as con:
            con.executescript(SCHEMA)

    def _connect(self):
        return sqlite3.connect(self.path, timeout=30)

    def add(self, result: Dict) -> int:
        st = result.get("statistics") or {}
        findings = [f["code"] for f in (result.get("structure") or {}).get("findings", []) if f["severity"] != "info"]
        with self._lock, self._connect() as con:
            cur = con.execute(
                "INSERT INTO scans(ts, file, sha256, verdict, risk, payload, findings, report) VALUES (?,?,?,?,?,?,?,?)",
                (result.get("timestamp"), result.get("file"), result.get("sha256"), result.get("verdict"),
                 result.get("risk_score"), st.get("effective_payload"), ",".join(findings),
                 json.dumps(result, ensure_ascii=False)))
            return int(cur.lastrowid)

    def recent(self, limit: int = 20, verdict: Optional[str] = None) -> List[Dict]:
        q = "SELECT id, ts, file, sha256, verdict, risk, payload, findings FROM scans"
        args: tuple = ()
        if verdict:
            q += " WHERE verdict = ?"
            args = (verdict,)
        q += " ORDER BY id DESC LIMIT ?"
        with self._connect() as con:
            rows = con.execute(q, args + (limit,)).fetchall()
        keys = ["id", "ts", "file", "sha256", "verdict", "risk", "payload", "findings"]
        return [dict(zip(keys, r)) for r in rows]

    def by_hash(self, sha256: str) -> List[Dict]:
        with self._connect() as con:
            rows = con.execute("SELECT id, ts, file, verdict, risk FROM scans WHERE sha256 = ? ORDER BY id DESC",
                               (sha256,)).fetchall()
        return [dict(zip(["id", "ts", "file", "verdict", "risk"], r)) for r in rows]

    def stats(self) -> Dict[str, int]:
        with self._connect() as con:
            rows = con.execute("SELECT verdict, COUNT(*) FROM scans GROUP BY verdict").fetchall()
        return {v: c for v, c in rows}
