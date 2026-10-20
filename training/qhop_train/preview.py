"""Visible, real-time, deterministic CPU checkpoint inspection."""
from pathlib import Path
import logging
import os
import time
import uuid

import numpy as np
import torch

from . import checkpoint
from .build_contract import verify as verify_build
from .control import atomic_json, digest, read_json, review_state
from .episode_channel import EpisodeChannel
from .curriculum import difficulty, PHASES, SHAPING_LEVELS, REWARD_HASH, COURSE_HASH, task
from .opponents import FrozenPolicies
from .ppo import PPO
from .review import load_run, save_preview_evidence
from .schema import ACTION, FRAME, HISTORY, PHYSICS_FINGERPRINT
from .unity_env import _snapshot


def preview(executable, source, directory, seed=92500, max_episodes=None,
            max_seconds=None, task_override=None):
    from mlagents_envs.environment import ActionTuple, UnityEnvironment
    from mlagents_envs.exception import UnityCommunicationException

    executable, source, directory = Path(executable), Path(source), Path(directory)
    build_hash = verify_build(executable)
    ppo = PPO("cpu")
    payload = checkpoint.load(source, ppo, weights_only=True)
    metadata, state = payload["metadata"], payload["training"]
    if metadata.get("unity_build") != build_hash or metadata.get("physics_fingerprint") != PHYSICS_FINGERPRINT or \
            metadata.get("reward_hash") != REWARD_HASH or metadata.get("course_hash") != COURSE_HASH:
        raise ValueError("Preview checkpoint is incompatible with the current Unity build")
    stage = state["stage"]
    task_index=state.get("stage1_task_index",0) if stage==1 else \
        state.get("stage2_task_index",0) if stage==2 else 0
    if task_override is not None:
        if stage!=2 or task_override not in range(3):
            raise ValueError("Preview task override is only valid for Stage 2 tasks 0-2")
        task_index=task_override
    task_name,distance,time_limit=task(stage,task_index)
    latest = directory / "latest.pt"
    review = review_state(load_run(directory)[1]) if latest.is_file() else review_state(state)
    candidate = review.get("status") == "pending" and \
        digest(source) == review.get("candidate_sha256") and task_override is None
    fraction = state.get("phase_steps", 0) / PHASES[stage - 1].budget
    level = 1.0 if candidate and stage in (4, 5, 6, 7) else difficulty(stage, fraction)
    if stage==2:
        level=0.0 if task_index==0 and state.get("gait_warmup",False) else 1.0
    shape = 0.0 if candidate and stage == 7 else \
        SHAPING_LEVELS[state.get("shaping_index", 0)] if stage == 7 else 1.0
    session_id = uuid.uuid4().hex
    status_path = directory / "preview-status.json"
    control_path = directory / "preview-control.json"
    last_control_sequence = 0
    episodes = 0
    results = []
    def publish(status, **extra):
        atomic_json(status_path, {"version": 1, "session_id": session_id,
            "pid": os.getpid(), "status": status, "stage": stage, "seed": seed,
            "task_index":task_index,"task_name":task_name,
            "task_distance":distance,"task_time_limit":time_limit,
            "checkpoint": str(source.resolve()), "checkpoint_sha256": digest(source),
            "episodes": episodes, "updated_at": time.time(), **extra})
    publish("launching")
    log_dir = directory / "preview-logs"
    log_dir.mkdir(exist_ok=True)
    args = [f"--qhop-stage={stage}", f"--qhop-difficulty={level}",
            f"--qhop-shaping={shape}", f"--qhop-seed={seed}",
            f"--qhop-task-index={task_index}",f"--qhop-distance={distance}",
            f"--qhop-time-limit={time_limit}",
            "--qhop-areas=1", "--qhop-preview=1",
            f"--qhop-preview-update={state.get('global_updates', 0)}",
            f"--qhop-preview-control={control_path.resolve()}",
            f"--qhop-preview-session={session_id}",
            "-logFile", str(log_dir / f"preview-{session_id}.log")]
    if os.environ.get("QHOP_PREVIEW_SCREENSHOT"):
        args.extend(["--qhop-lidar-debug=1","--qhop-screenshot="+os.environ["QHOP_PREVIEW_SCREENSHOT"]])
    opponent_paths = state.get("opponents", [])
    opponents = FrozenPolicies(opponent_paths, ppo, seed)
    env = None
    channel=EpisodeChannel()
    try:
        env = UnityEnvironment(file_name=str(executable), worker_id=900, seed=seed,
                               no_graphics=False, additional_args=args,
                               side_channels=[channel], timeout_wait=120)
        env.reset()
        publish("running")
        started=time.monotonic()
        while (max_episodes is None or episodes < max_episodes) and \
                (max_seconds is None or time.monotonic()-started < max_seconds):
            command = read_json(control_path, {})
            if command.get("session_id") == session_id and \
                    int(command.get("sequence", 0)) > last_control_sequence:
                last_control_sequence = int(command["sequence"])
                seed = int(command["seed"])
                opponents = FrozenPolicies(opponent_paths, ppo, seed)
                ppo.vision_states.clear()
                publish("running")
            state_now = _snapshot(env)
            for role in ("learner", "frozen"):
                group = state_now[role]
                if not len(group["decision_id"]):
                    continue
                if role == "learner":
                    raw = ppo.normalizer.normalize(group["decision_obs"])
                    with torch.no_grad():
                        memory=torch.cat([ppo.vision_states.consume((0,int(agent)),group["decision_obs"][row])[0] for row,agent in enumerate(group["decision_id"])])
                        actions = ppo.actor(torch.as_tensor(raw),memory).cpu().numpy()
                        channel.vision_debug([ppo.vision_states.entry((0,int(agent)))['updates'] for agent in group['decision_id']])
                else:
                    actions = opponents.actions(0, group["decision_id"], group["decision_obs"])
                env.set_actions(state_now["names"][role],
                                ActionTuple(continuous=np.asarray(actions, np.float32).reshape(-1, ACTION)))
            env.step()
            diagnostic=channel.drain()
            following = _snapshot(env)
            opponents.terminal(0, following["frozen"]["terminal_id"])
            for agent in following["learner"]["terminal_id"]:ppo.vision_states.terminal((0,int(agent)))
            count = len(following["learner"]["terminal_id"])
            if count:
                episodes += count
                for observation, reward in zip(following["learner"]["terminal_obs"],
                                               following["learner"]["terminal_reward"]):
                    frame=observation[(HISTORY-1)*FRAME:HISTORY*FRAME]
                    results.append({"finished": bool(frame[210]>.5),
                                    "dnf": bool(frame[211]>.5),
                                    "progress_m": float(frame[198]*100),
                                    "falls": int(round(frame[212]*10)),
                                    "elapsed_s": float(time_limit-frame[203]*90),
                                    "terminal_reward": float(reward),
                                    "gait": diagnostic[-1] if diagnostic else {}})
                if candidate and episodes == count:
                    save_preview_evidence(directory, source, seed, count, results)
                publish("running", last_result=results[-1])
    except (EOFError, ConnectionError, BrokenPipeError, UnityCommunicationException):
        # The visible Player's Exit button closes the ML-Agents connection.
        pass
    except BaseException as error:
        try:
            publish("failed", error=repr(error))
        except Exception:
            logging.exception("Could not publish failed preview status; original error: %r", error)
        raise
    finally:
        if env is not None:
            try:
                env.close()
            except UnityCommunicationException:
                pass
    publish("closed")
    return {"episodes": episodes, "candidate": candidate, "results": results}
