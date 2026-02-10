import os
import time
import traceback

from train import train as train_fn
from training_ipc import fetch_next_command, init_db, reset_metrics, update_status

POLL_INTERVAL_SEC = 1.0


def _set_idle(detail=None):
    update_status(state="idle", detail=detail, pid=os.getpid())


def _set_running(detail=None):
    update_status(state="running", detail=detail, pid=os.getpid())


def _set_error(detail=None):
    update_status(state="error", detail=detail, pid=os.getpid())


def main():
    init_db()
    _set_idle(detail="worker started")

    while True:
        cmd = fetch_next_command()
        if cmd is None:
            time.sleep(POLL_INTERVAL_SEC)
            continue

        command = cmd.get("command")
        payload = cmd.get("payload") or {}

        if command == "start":
            reset_metrics()
            _set_running(detail="training started")
            try:
                result = train_fn(payload)
                if isinstance(result, dict) and result.get("stopped"):
                    update_status(state="stopped", detail="training stopped by request")
                else:
                    _set_idle(detail="training finished")
            except Exception:
                _set_error(detail=traceback.format_exc())
        elif command == "stop":
            # Мы фиксируем запрос на остановку, пока полноценная обработка еще не реализована
            update_status(state="stop_requested", detail="stop requested (not implemented)")
        elif command == "status":
            update_status(detail="status requested")
        else:
            update_status(state="error", detail=f"unknown command: {command}")


if __name__ == "__main__":
    main()
