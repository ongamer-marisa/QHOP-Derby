"""Offline, atomic human-review operations on a stopped experiment."""
from pathlib import Path
import json

from . import checkpoint
from .control import approve, atomic_json, continue_stage, digest, read_json, record_preview, review_state
from .curriculum import PHASES
from .curriculum import REWARD_HASH, COURSE_HASH
from .ppo import PPO


def load_run(directory):
    directory = Path(directory)
    source = directory / "latest.pt"
    ppo = PPO("cpu")
    payload = checkpoint.load(source, ppo)
    state = payload["training"]
    review_state(state)
    return ppo, state, payload["metadata"]


def _sync_status(directory, state, status):
    path = Path(directory) / "session-status.json"
    value = read_json(path, {})
    value.update(status=status, stage=state["stage"],
                 global_steps=state.get("global_steps", 0),
                 phase_steps=state.get("phase_steps", 0),
                 updates=state.get("global_updates", 0),
                 passes=state.get("passes", 0), review=state["review"],
                 task_index=state.get("stage1_task_index",0) if state.get("stage")==1
                     else state.get("stage2_task_index",0) if state.get("stage")==2 else 0,
                 recent100=state.get("recent_episodes_summary",{}),
                 approvals=state.get("approvals", []))
    atomic_json(path, value)


def checkpoint_info(path):
    ppo = PPO("cpu")
    payload = checkpoint.load(path, ppo)
    state = payload["training"]
    return {"checkpoint": str(Path(path).resolve()),
            "sha256": digest(path), "stage": state.get("stage", 1),
            "batch": payload.get("metadata", {}).get("batch"),
            "workers": payload.get("metadata", {}).get("workers"),
            "seed": payload.get("metadata", {}).get("seed"),
            "updates": state.get("global_updates", 0),
            "global_steps": state.get("global_steps", 0),
            "learning_rate": ppo.schedule.rate(),
            "task_index": state.get("stage1_task_index",0) if state.get("stage")==1
                else state.get("stage2_task_index",0) if state.get("stage")==2 else 0,
            "reward_hash": payload.get("metadata",{}).get("reward_hash"),
            "course_hash": payload.get("metadata",{}).get("course_hash"),
            "review": review_state(state),
            "unity_build": payload.get("metadata", {}).get("unity_build"),
            "physics_fingerprint": payload.get("metadata", {}).get("physics_fingerprint")}


def save_preview_evidence(directory, candidate, seed, episodes, results):
    directory = Path(directory)
    ppo, state, metadata = load_run(directory)
    review = record_preview(state, candidate, seed, episodes)
    record = {"checkpoint": str(Path(candidate).resolve()),
              "sha256": review["candidate_sha256"], "stage": state["stage"],
              "seed": seed, "episodes": episodes, "results": results}
    with (directory / "preview-evidence.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    checkpoint.save(directory / "latest.pt", ppo, state, metadata)
    _sync_status(directory, state, "pending_review")
    return record


def approve_stage(directory):
    directory = Path(directory)
    ppo, state, metadata = load_run(directory)
    review = review_state(state)
    candidate = Path(review.get("candidate", ""))
    if not candidate.is_file():
        raise ValueError("Pending candidate checkpoint is missing")
    approve(state, candidate)
    selected = checkpoint.load(candidate, ppo)
    if selected["metadata"].get("unity_build") != metadata.get("unity_build") or \
            selected["metadata"].get("batch") != metadata.get("batch") or \
            selected["metadata"].get("reward_hash") != REWARD_HASH or \
            selected["metadata"].get("course_hash") != COURSE_HASH:
        raise ValueError("Candidate checkpoint metadata differs from the current experiment")
    stage = state["stage"]
    state.setdefault("approvals", []).append({"stage": stage, **dict(review)})
    checkpoint.save(directory / f"stage{stage}-approved.pt", ppo, state, metadata)
    if stage == 5:
        state["opponents"] = [str(directory / "stage5-approved.pt")]
    if stage == 6:
        state["stage6_baseline"] = review["report"]
    if stage == 7:
        checkpoint.save(directory / "final.pt", ppo, state, metadata)
        checkpoint.save(directory / "latest.pt", ppo, state, metadata)
        _sync_status(directory, state, "completed")
        return state
    state["stage"] = stage + 1
    state["phase_steps"] = 0
    state["phase_budget_extension"] = 0
    state["passes"] = 0
    if state["stage"] == 7:
        state["shaping_index"] = 0
        state["shaping_level_start"] = 0
    phase = PHASES[state["stage"] - 1]
    if stage==1:
        from .ppo import PPO
        fresh=PPO(ppo.device,phase.peak_lr,phase.updates(metadata["batch"]))
        fresh.actor.load_state_dict(ppo.actor.state_dict())
        fresh.normalizer.load_state_dict(ppo.normalizer.state_dict())
        ppo=fresh
        state["gait_warmup"]=True
    else:
        ppo.reset_phase(phase.peak_lr, phase.updates(metadata["batch"]))
    state["review"] = {"version": 1, "status": "training",
                       "previous_approval": dict(review)}
    checkpoint.save(directory / "latest.pt", ppo, state, metadata)
    _sync_status(directory, state, "idle")
    return state


def reject_and_continue(directory, extra_steps=0):
    if extra_steps < 0:
        raise ValueError("Additional stage steps cannot be negative")
    directory = Path(directory)
    ppo, state, metadata = load_run(directory)
    if review_state(state)["status"] != "pending":
        raise ValueError("There is no pending candidate to continue past")
    continue_stage(state)
    state["phase_budget_extension"] = state.get("phase_budget_extension", 0) + extra_steps
    checkpoint.save(directory / "latest.pt", ppo, state, metadata)
    _sync_status(directory, state, "idle")
    return state
