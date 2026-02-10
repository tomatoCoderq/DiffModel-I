import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

DB_PATH = os.environ.get("TRAINING_IPC_DB", "training_ipc.sqlite")
ALLOWED_COMMANDS = {"start", "stop", "status"}


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_commands (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                command TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                processed_at TEXT
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_status (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                state TEXT NOT NULL,
                detail TEXT,
                epoch INTEGER,
                step INTEGER,
                loss REAL,
                updated_at TEXT NOT NULL,
                pid INTEGER
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                epoch INTEGER,
                step INTEGER,
                loss REAL
            );
            """
        )
        row = conn.execute("SELECT id FROM training_status WHERE id = 1").fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO training_status (id, state, detail, epoch, step, loss, updated_at, pid)
                VALUES (1, 'idle', 'initialized', NULL, NULL, NULL, ?, NULL)
                """,
                (_utc_now_iso(),),
            )


def enqueue_command(command, payload=None):
    init_db()
    if command not in ALLOWED_COMMANDS:
        raise ValueError(f"Unknown command: {command}")
    payload = payload or {}
    payload_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO training_commands (command, payload_json, created_at, processed_at)
            VALUES (?, ?, ?, NULL)
            """,
            (command, payload_json, _utc_now_iso()),
        )


def enqueue_start(payload=None):
    enqueue_command("start", payload=payload)


def enqueue_stop():
    enqueue_command("stop", payload={})


def enqueue_status():
    enqueue_command("status", payload={})


def fetch_next_command():
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT id, command, payload_json
            FROM training_commands
            WHERE processed_at IS NULL
            ORDER BY id ASC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        cmd_id, command, payload_json = row
        conn.execute(
            "UPDATE training_commands SET processed_at = ? WHERE id = ?",
            (_utc_now_iso(), cmd_id),
        )
    payload = json.loads(payload_json) if payload_json else {}
    return {"id": cmd_id, "command": command, "payload": payload}


def fetch_command(command):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT id, payload_json
            FROM training_commands
            WHERE processed_at IS NULL AND command = ?
            ORDER BY id ASC
            LIMIT 1
            """,
            (command,),
        ).fetchone()
        if row is None:
            return None
        cmd_id, payload_json = row
        conn.execute(
            "UPDATE training_commands SET processed_at = ? WHERE id = ?",
            (_utc_now_iso(), cmd_id),
        )
    payload = json.loads(payload_json) if payload_json else {}
    return {"id": cmd_id, "command": command, "payload": payload}


def fetch_stop_request():
    return fetch_command("stop") is not None


def log_metric(epoch=None, step=None, loss=None):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO training_metrics (created_at, epoch, step, loss)
            VALUES (?, ?, ?, ?)
            """,
            (_utc_now_iso(), epoch, step, loss),
        )


def get_metrics(limit=200):
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT created_at, epoch, step, loss
            FROM training_metrics
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    rows.reverse()
    return [
        {"created_at": row[0], "epoch": row[1], "step": row[2], "loss": row[3]}
        for row in rows
    ]


def reset_metrics():
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM training_metrics")


def _is_pid_running(pid):
    if pid is None:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def is_worker_running():
    status = get_status()
    if not status:
        return False
    return _is_pid_running(status.get("pid"))


def start_worker_if_needed():
    if is_worker_running():
        return False
    subprocess.Popen(
        [sys.executable, "training_worker.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return True


def update_status(**fields):
    init_db()
    allowed = {"state", "detail", "epoch", "step", "loss", "pid"}
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"Unknown status fields: {sorted(unknown)}")
    if not fields:
        return

    set_clauses = []
    values = []
    for key, value in fields.items():
        set_clauses.append(f"{key} = ?")
        values.append(value)
    set_clauses.append("updated_at = ?")
    values.append(_utc_now_iso())

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            f"UPDATE training_status SET {', '.join(set_clauses)} WHERE id = 1",
            values,
        )


def get_status():
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT state, detail, epoch, step, loss, updated_at, pid
            FROM training_status
            WHERE id = 1
            """
        ).fetchone()
    if row is None:
        return None
    return {
        "state": row[0],
        "detail": row[1],
        "epoch": row[2],
        "step": row[3],
        "loss": row[4],
        "updated_at": row[5],
        "pid": row[6],
    }
