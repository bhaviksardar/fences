"""
init(export="file:///path/to/dir"): every event that leaves the process (after capture="metadata"
and the redact hook) is also written, as one JSON line, to {dir}/{agent}/{YYYY-MM-DD}.jsonl (the UTC
date), in local and cloud mode. Written from a background thread, so the agent never waits on the
disk; a failed write warns once and never reaches the agent.
"""
import json
import logging
import os
import queue
import re
import threading
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse
from urllib.request import url2pathname

log = logging.getLogger("agentfences")

MAX_QUEUED = 10_000  # a stuck disk can't make a runaway agent eat memory; extra lines are dropped

_dir: Optional[str] = None
_queue: "queue.Queue" = queue.Queue(MAX_QUEUED)
_worker: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
_failed = False


def configure(url: Optional[str]):
    global _dir, _failed
    _dir, _failed = None, False
    if url is None:
        return
    parsed = urlparse(url) if isinstance(url, str) else None
    if parsed is None or parsed.scheme != "file" or parsed.netloc not in ("", "localhost") or not parsed.path:
        raise ValueError(f"export must be a file:// URL of a directory, e.g. file:///var/log/fences, not {url!r}")
    _dir = url2pathname(parsed.path)


def active() -> bool:
    return _dir is not None


def write(agent_name: str, run_id: str, event: dict):
    """Queue one event, as sent, for the run's file. Never raises."""
    if _dir is None:
        return
    _start()
    try:
        _queue.put_nowait((_dir, agent_name, {"type": event.get("type"), "run_id": run_id, "ts": time.time(), **event}))
    except queue.Full:
        log.debug("agentfences: export queue full, dropped a %s", event.get("type"))


def wait(deadline: float):
    """Block until everything queued is written, or time.time() passes deadline."""
    while _queue.unfinished_tasks and time.time() < deadline:
        time.sleep(0.02)


def _file(base: str, agent_name: str, ts: float) -> str:
    agent = re.sub(r"[^A-Za-z0-9_.-]", "_", agent_name)  # a name can't climb out of the directory
    if agent in ("", ".", ".."):
        agent = "_"
    return os.path.join(base, agent, datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d") + ".jsonl")


def _work():
    global _failed
    while True:
        items = [_queue.get()]
        while True:  # take everything queued so far: one open and write per file
            try:
                items.append(_queue.get_nowait())
            except queue.Empty:
                break
        try:
            files: dict = {}
            for base, agent_name, line in items:
                files.setdefault(_file(base, agent_name, line["ts"]), []).append(json.dumps(line, default=str))
            for path, lines in files.items():
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write("\n".join(lines) + "\n")
        except Exception as e:  # a full disk or a read-only directory must never break the agent
            if not _failed:
                _failed = True
                log.warning("agentfences: couldn't write the trace export (%s: %s); events are still sent as usual",
                            type(e).__name__, e)
        finally:
            for _ in items:
                _queue.task_done()


def _start():
    global _worker
    with _worker_lock:
        if _worker is None:
            _worker = threading.Thread(target=_work, name="agentfences-export", daemon=True)
            _worker.start()
