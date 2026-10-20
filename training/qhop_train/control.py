"""Atomic editor/trainer control files and the human stage-review gate."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid

from .atomic_io import cleanup_temporary, replace_with_retry


REVIEW_VERSION = 1


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent,
                                         prefix=".qhop-", suffix=".json", delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        replace_with_retry(temporary, path)
    finally:
        cleanup_temporary(temporary)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def review_state(training):
    value = training.setdefault("review", {"version": REVIEW_VERSION,
                                            "status": "training"})
    value.setdefault("version", REVIEW_VERSION)
    value.setdefault("status", "training")
    return value


def may_train(training):
    return review_state(training)["status"] == "training"


def freeze_candidate(training, checkpoint_path, report):
    review = review_state(training)
    if review["status"] != "training":
        raise ValueError("Stage is already awaiting review")
    if training.get("passes", 0) < 3:
        raise ValueError("Three consecutive passing evaluations are required")
    review.update(status="pending", candidate=str(Path(checkpoint_path).resolve()),
                  candidate_sha256=digest(checkpoint_path), report=report,
                  preview_episodes=0, preview_seeds=[])
    return review


def record_preview(training, candidate_path, seed, episodes):
    review = review_state(training)
    if review["status"] != "pending" or digest(candidate_path) != review.get("candidate_sha256"):
        raise ValueError("Preview checkpoint does not match the pending candidate")
    if episodes < 1:
        raise ValueError("At least one complete preview episode is required")
    review["preview_episodes"] = review.get("preview_episodes", 0) + episodes
    review.setdefault("preview_seeds", []).append(seed)
    return review


def approve(training, candidate_path, when=None):
    review = review_state(training)
    if training.get("passes", 0) < 3 or review["status"] != "pending" or review.get("preview_episodes", 0) < 1:
        raise ValueError("Automatic evaluation and a complete preview are both required")
    if digest(candidate_path) != review.get("candidate_sha256"):
        raise ValueError("Candidate checkpoint hash mismatch")
    review["status"] = "approved"
    review["approved_at"] = when or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return review


def continue_stage(training):
    review = review_state(training)
    review.clear()
    review.update(version=REVIEW_VERSION, status="training")
    training["passes"] = 0


class StopRequested(Exception):
    pass


class Session:
    def __init__(self, directory, state=None):
        self.directory = Path(directory)
        self.id = uuid.uuid4().hex
        self.state = state
        self.status_path = self.directory / "session-status.json"
        self.request_path = self.directory / "session-request.json"
        self.last_request = 0
        self.publish("starting")

    def publish(self, status, **extra):
        state = self.state or {}
        value = {"version": 1, "session_id": self.id, "pid": os.getpid(),
                 "status": status, "stage": state.get("stage", 1),
                 "global_steps": state.get("global_steps", 0),
                 "phase_steps": state.get("phase_steps", 0),
                 "updates": state.get("global_updates", 0),
                 "passes": state.get("passes", 0),
                 "task_index":state.get("stage1_task_index",0) if state.get("stage")==1
                     else state.get("stage2_task_index",0) if state.get("stage")==2 else 0,
                 "recent100":state.get("recent_episodes_summary",{}),
                 "learning_rate": extra.pop("learning_rate", None),
                 "checkpoint": str(self.directory / "latest.pt"),
                 "review": state.get("review", {}),
                 "approvals": state.get("approvals", []),
                 "last_evaluation": state.get("history", [])[-1] if state.get("history") else None,
                 "horse_steps_per_second": state.get("horse_steps_per_second", 0),
                 "updated_at": time.time(), **extra}
        atomic_json(self.status_path, value)

    def check_stop(self):
        request = read_json(self.request_path, {})
        if request.get("session_id") != self.id:
            return
        sequence = int(request.get("sequence", 0))
        if sequence > self.last_request and request.get("command") == "stop":
            self.last_request = sequence
            raise StopRequested()


def request_stop(directory, session_id):
    path = Path(directory) / "session-request.json"
    prior = read_json(path, {})
    atomic_json(path, {"session_id": session_id, "sequence":
                int(prior.get("sequence", 0)) + 1, "command": "stop"})
