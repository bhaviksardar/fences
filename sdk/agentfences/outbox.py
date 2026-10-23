"""
Decisions and spans go to the server from one background thread, in order, so
log_decision() and recorded calls never block the agent on a network round trip.
Consecutive spans of a run go in one request.
"""
import logging
import queue
import threading
import time
from typing import Optional

log = logging.getLogger("agentfences")

SPAN_BATCH = 50
MAX_QUEUED = 10_000  # a slow server can't make a runaway agent eat memory; extra items are dropped

# Items: ("decision", client, payload) or ("span", client, run_id, span)
_queue: "queue.Queue" = queue.Queue(MAX_QUEUED)
_worker: Optional[threading.Thread] = None
_worker_lock = threading.Lock()
spans_supported = True  # off once a server says it has no spans endpoint


def put_decision(client, payload: dict):
    _put(("decision", client, payload))


def put_span(client, run_id: str, span: dict):
    _put(("span", client, run_id, span))


def _put(item):
    _start()
    try:
        _queue.put_nowait(item)
    except queue.Full:
        log.debug("agentfences: outbox full, dropped a %s", item[0])


def wait(deadline: float):
    """Block until everything queued has been sent, or time.time() passes deadline."""
    while _queue.unfinished_tasks and time.time() < deadline:
        time.sleep(0.02)


def _send(item, more: list):
    global spans_supported
    if item[0] == "decision":
        item[1].log_decision(**item[2])
        return
    _, client, run_id, span = item
    batch = [span]
    while more and more[0][0] == "span" and more[0][2] == run_id and len(batch) < SPAN_BATCH:
        batch.append(more.pop(0)[3])
    if spans_supported and client.log_spans(run_id, batch).get("unsupported"):
        spans_supported = False
        log.warning("agentfences: this Fences server doesn't accept tool and model call events yet; "
                    "they're kept on the run locally but not sent")


def _work():
    pending: list = []
    while True:
        if not pending:
            pending.append(_queue.get())
        while True:  # take everything queued so far, so a burst of spans goes out together
            try:
                pending.append(_queue.get_nowait())
            except queue.Empty:
                break
        item = pending.pop(0)
        before = len(pending)
        try:
            _send(item, pending)
        except Exception:  # delivery must never break the agent
            pass
        finally:
            for _ in range(1 + before - len(pending)):  # this item plus any batched with it
                _queue.task_done()


def _start():
    global _worker
    with _worker_lock:  # one worker, or items could go out of order
        if _worker is None:
            _worker = threading.Thread(target=_work, name="agentfences-outbox", daemon=True)
            _worker.start()
