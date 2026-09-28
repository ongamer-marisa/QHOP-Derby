"""Synchronous Unity sampling, evaluation and six-stage training orchestration."""
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
import json
import math
import platform
import random
import statistics
import time

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import checkpoint
from .control import Session, StopRequested, freeze_candidate, may_train, review_state
from .build_contract import verify as verify_build
from .curriculum import (PHASES, SHAPING_LEVELS, REWARD_HASH, COURSE_HASH,
                         STAGE1_TASKS, difficulty, passes, task,
                         evaluation_interval, stage1_result,
                         stage6_can_advance, stage6_final_ready)
from .ppo import PPO, Rollout
from .opponents import FrozenPolicies
from .schema import ACTION, FRAME, HISTORY, OBSERVATION, PHYSICS_FINGERPRINT, VERSION
from .unity_env import UnityWorkers


def episode_summary(episodes):
    if not episodes:
        return {"episodes":0}
    number=len(episodes)
    mean=lambda field:sum(float(item.get(field,0)) for item in episodes)/number
    reasons={}
    for item in episodes:
        if item.get("dnf"):
            reason=item.get("reason") or "unknown"
            reasons[reason]=reasons.get(reason,0)+1
    return {"episodes":number,"completion":sum(bool(item.get("finished")) for item in episodes)/number,
            "dnf_reasons":reasons,"dnf_total":sum(reasons.values()),
            "progress_m":mean("progressMeters"),
            "reward_progress":mean("weightedProgress"),
            "reward_posture":mean("weightedPosture"),
            "reward_time":mean("weightedTime"),
            "reward_fall":mean("weightedFall"),
            "reward_finish":mean("weightedFinish"),
            "reward_dnf":mean("weightedDnf"),"reward_total":mean("total"),
            "falls":mean("falls"),"stationary_fraction":mean("stationaryFraction"),
            "filtered_posture_m":mean("filteredPosture"),
            "filtered_ground_m":mean("filteredGround"),
            "filtered_inactive_m":mean("filteredInactive"),
            "filtered_boundary_m":mean("filteredBoundary")}


def active_mask(raw):
    frame = raw[(HISTORY - 1) * FRAME:HISTORY * FRAME]
    return float(frame[206] == 0 and frame[207] == 0 and
                 frame[210] == 0 and frame[211] == 0)


def collect(ppo, workers, opponents, target=16384, check_stop=None):
    records = []
    raw_history = []
    streams = {}
    episodes = 0
    episode_logs = []
    while len(records) < target:
        if check_stop is not None:
            check_stop()
        submitted = []
        pending = []
        for worker_index, all_state in enumerate(workers.current):
            state = all_state["learner"]
            raw = state["decision_obs"]
            normalized = ppo.normalizer.normalize(raw)
            with torch.no_grad():
                tensor = torch.as_tensor(normalized, device=ppo.device)
                action, logp = ppo.actor.sample(tensor)
                value = ppo.critic(tensor)
            actions = action.cpu().numpy().astype(np.float32)
            frozen = all_state["frozen"]
            submitted.append({"learner": actions,
                              "frozen": opponents.actions(worker_index,
                                  frozen["decision_id"], frozen["decision_obs"])})
            for row, agent_id in enumerate(state["decision_id"]):
                key = (worker_index, int(agent_id))
                stream = streams.setdefault(key, len(streams))
                pending.append((key, stream, raw[row], normalized[row], actions[row],
                                float(logp[row].item()), float(value[row].item())))
        next_states = workers.step(submitted)
        lookups = []
        for worker_index, all_state in enumerate(next_states):
            for event in all_state.get("episode_logs", []):
                episode_logs.append({"worker":worker_index,**event})
            opponents.terminal(worker_index, all_state["frozen"]["terminal_id"])
            state = all_state["learner"]
            episodes += len(state["terminal_id"])
            decisions = {int(agent_id): (state["decision_obs"][i],
                         float(state["decision_reward"][i]), False)
                         for i, agent_id in enumerate(state["decision_id"])}
            terminals = {int(agent_id): (state["terminal_obs"][i],
                         float(state["terminal_reward"][i]), True)
                         for i, agent_id in enumerate(state["terminal_id"])}
            decisions.update(terminals)
            lookups.append(decisions)
        for key, stream, raw, normalized, action, logp, value in pending:
            next_observation, reward, done = lookups[key[0]][key[1]]
            with torch.no_grad():
                next_tensor = torch.as_tensor(
                    ppo.normalizer.normalize(next_observation)[None, :], device=ppo.device)
                next_value = 0.0 if done else float(ppo.critic(next_tensor).item())
            records.append((normalized, action, logp, value, next_value, reward,
                            done, active_mask(raw), stream))
            raw_history.append(raw)
    arrays = list(zip(*records))
    rollout = Rollout(
        observation=np.stack(arrays[0])[:, None, :],
        action=np.stack(arrays[1])[:, None, :],
        logp=np.asarray(arrays[2], dtype=np.float32)[:, None],
        value=np.asarray(arrays[3], dtype=np.float32)[:, None],
        next_value=np.asarray(arrays[4], dtype=np.float32)[:, None],
        reward=np.asarray(arrays[5], dtype=np.float32)[:, None],
        done=np.asarray(arrays[6], dtype=bool)[:, None],
        actor_mask=np.asarray(arrays[7], dtype=np.float32)[:, None],
        stream_id=np.asarray(arrays[8], dtype=np.int64))
    return rollout, np.stack(raw_history), episodes, episode_logs


def evaluate(ppo, executable, stage, seed=72500, races=100, worker_count=1,
             difficulty_level=1.0, shaping_level=0.0, opponent_paths=(),
             check_stop=None, log_dir=None, task_index=3):
    phase = PHASES[stage - 1]
    target = races * (4 if stage >= 5 else 1)
    episodes = []
    opponents = FrozenPolicies(opponent_paths, ppo, seed)
    task_name, distance, time_limit = task(stage, task_index)
    with UnityWorkers(executable, worker_count, stage, difficulty_level, shaping_level,
                      seed, phase.areas, worker_id_base=500, log_dir=log_dir,
                      task_index=task_index) as workers:
        while len(episodes) < target:
            if check_stop is not None:
                check_stop()
            submitted = []
            for worker_index, all_state in enumerate(workers.current):
                state = all_state["learner"]
                raw = ppo.normalizer.normalize(state["decision_obs"])
                with torch.no_grad():
                    actions = ppo.actor(torch.as_tensor(raw, device=ppo.device)).cpu().numpy()
                frozen = all_state["frozen"]
                submitted.append({"learner": actions.astype(np.float32).reshape(-1, ACTION),
                    "frozen": opponents.actions(worker_index, frozen["decision_id"],
                                                frozen["decision_obs"])})
            states = workers.step(submitted)
            for worker_index, all_state in enumerate(states):
                episodes.extend(all_state["learner"]["terminal_obs"])
                opponents.terminal(worker_index, all_state["frozen"]["terminal_id"])
    completed, times, speeds, falls = [], [], [], []
    buckets = defaultdict(list)
    disturbance_total = 0.0
    recovery_total = 0.0
    for obs in episodes[:target]:
        frame = obs[(HISTORY - 1) * FRAME:HISTORY * FRAME]
        finished = bool(frame[210] > .5)
        fall_count = frame[212] * 10.0
        elapsed = time_limit - frame[203] * 90
        elapsed = max(.01, elapsed)
        completed.append(finished)
        falls.append(fall_count)
        if finished:
            times.append(elapsed)
            speeds.append(distance / elapsed)
        mass = frame[196] * 750
        bucket = sum(mass >= threshold for threshold in (400, 450, 500))
        buckets[bucket].append(finished)
        disturbance_total += frame[213] * 10
        recovery_total += frame[214] * 10
    report = {
        "stage": stage, "episodes": len(completed), "completion": float(np.mean(completed)),
        "fall_free": float(np.mean([a and b < .5 for a, b in zip(completed, falls)])),
        "median_time": float(statistics.median(times)) if times else None,
        "median_speed": float(statistics.median(speeds)) if speeds else 0.0,
        "falls_per_100m": float(np.sum(falls) / max(1, sum(completed))),
        "worst_morphology_completion": min((float(np.mean(v)) for v in buckets.values()), default=0),
        "disturbance_recovery": recovery_total / max(1, disturbance_total),
        "collision_rule_errors": 0,
        "collision_rule_validation": "HorseRaceSession.ValidateResults; Unity failure aborts evaluation",
        "shaping": shaping_level,
        "task_index": task_index,"task_name": task_name,
        "task_distance": distance,"task_time_limit": time_limit,
    }
    return report


def _build_digest(executable):
    return verify_build(executable)


def train(executable, output, *, seed=20260927, worker_count=4,
          resume=None, finetune=None, initial_stage=1, batch_size=16384,
          extra_steps=0, max_updates=None):
    if resume and finetune:
        raise ValueError("Choose resume or finetune")
    if initial_stage>=5 and not (resume or finetune):
        raise ValueError("Stages 5-6 require a passed earlier checkpoint or finetune source")
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive")
    run_dir = Path(output)
    run_dir.mkdir(parents=True, exist_ok=True)
    build_hash=_build_digest(executable)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    phase = PHASES[initial_stage - 1]
    ppo = PPO("cuda" if torch.cuda.is_available() else "cpu", phase.peak_lr,
              phase.updates(batch_size))
    state = {"stage": initial_stage, "global_steps": 0, "phase_steps": 0,
             "global_updates": 0, "passes": 0, "episode_seed": seed,
             "stage1_task_index":0,"recent_episodes":[],
             "recent_episodes_summary":{"episodes":0},
             "stage5_baseline": None, "history": [], "opponents": [],
             "phase_budget_extension":0, "shape_best":{},
             "shaping_index":0,"shaping_level_start":0,"worker_restarts":0,
             "episodes":0,
             "best":{},"review":{"version":1,"status":"training"},
             "approvals":[]}
    if resume:
        payload = checkpoint.load(resume, ppo)
        state = payload["training"]
        state.setdefault("shape_best",{})
        state.setdefault("shaping_index",0)
        state.setdefault("shaping_level_start",0)
        state.setdefault("worker_restarts",0)
        state.setdefault("episodes",0)
        state.setdefault("best",{})
        state.setdefault("approvals",[])
        state.setdefault("stage1_task_index",0)
        state.setdefault("recent_episodes",[])
        state.setdefault("recent_episodes_summary",{"episodes":0})
        if payload.get("metadata",{}).get("unity_build") != build_hash:
            raise ValueError("Unity build changed since checkpoint; use finetune for a new experiment")
        if payload.get("metadata",{}).get("physics_fingerprint") != PHYSICS_FINGERPRINT:
            raise ValueError("Physics contract changed since checkpoint")
        if payload.get("metadata",{}).get("reward_hash") != REWARD_HASH or \
                payload.get("metadata",{}).get("course_hash") != COURSE_HASH:
            raise ValueError("Reward or curriculum contract changed; start a new experiment")
        if payload.get("metadata",{}).get("batch") != batch_size:
            raise ValueError("Resume must retain the checkpoint batch size")
        if payload.get("metadata",{}).get("workers") != worker_count:
            raise ValueError("Resume must retain the checkpoint worker count")
        if payload.get("metadata",{}).get("seed") != seed:
            raise ValueError("Resume must retain the checkpoint experiment seed")
        review_state(state)
        if not may_train(state):
            raise ValueError("Stage awaits human review or is complete; use preview/approve or continue-stage")
        state["phase_budget_extension"] += extra_steps
    if finetune:
        source=checkpoint.load(finetune, ppo, weights_only=True)
        if source.get("metadata",{}).get("physics_fingerprint")!=PHYSICS_FINGERPRINT:
            raise ValueError("Finetune source physics contract differs from current Unity build")
        if source.get("metadata",{}).get("reward_hash")!=REWARD_HASH or \
                source.get("metadata",{}).get("course_hash")!=COURSE_HASH:
            raise ValueError("Finetune source reward or curriculum contract differs")
        if initial_stage==6:
            state["stage5_baseline"]=source["training"].get("stage5_baseline")
            if state["stage5_baseline"] is None:
                raise ValueError("Stage 6 finetune requires a Stage 5 baseline")
    session=Session(run_dir,state)
    writer = SummaryWriter(run_dir / "tensorboard")
    metadata={"executable":str(executable),"unity_build":build_hash,
              "batch":batch_size,"workers":worker_count,"seed":seed,
              "phases":[vars(phase) for phase in PHASES],
              "python":platform.python_version(),"torch":torch.__version__,
              "cuda":torch.version.cuda,"numpy":np.__version__,
              "mlagents_envs":version("mlagents-envs"),
              "onnx":version("onnx"),"tensorboard":version("tensorboard"),
              "schema_version":VERSION,"action_size":ACTION,
              "observation_size":OBSERVATION,
              "physics_fingerprint":PHYSICS_FINGERPRINT,
              "reward_hash":REWARD_HASH,"course_hash":COURSE_HASH,
              "model_architecture":"grouped-mlp-v1"}
    if finetune and initial_stage>=5:
        opponent_path=run_dir / "finetune-frozen-opponent.pt"
        checkpoint.save(opponent_path,ppo,state,metadata)
        state["opponents"]=[str(opponent_path)]
    def save_latest():
        checkpoint.save(run_dir / "latest.pt",ppo,state,metadata)
        session.publish("saved", learning_rate=ppo.schedule.rate())
    def save_periodic():
        path=run_dir / f"step-{state['global_updates']:08d}.pt"
        checkpoint.save(path,ppo,state,metadata)
        save_latest()
        older=sorted(run_dir.glob("step-*.pt"),key=lambda item:item.stat().st_mtime)
        for item in older[:-5]:item.unlink()
    # latest.pt is always the last fully completed update. A resumed run keeps
    # its original checkpoint until another complete update has been committed.
    if not resume or Path(resume).resolve() != (run_dir / "latest.pt").resolve():
        save_latest()
    stopped=False
    try:
        while state["stage"] <= 6:
            session.check_stop()
            if max_updates is not None and state["global_updates"]>=max_updates:
                break
            phase = PHASES[state["stage"] - 1]
            task_index=state["stage1_task_index"] if phase.stage==1 else 0
            fraction = state["phase_steps"] / phase.budget
            diff = difficulty(phase.stage, fraction)
            shape = SHAPING_LEVELS[state["shaping_index"]] if phase.stage==6 else 1.0
            unity_seed = state["episode_seed"]
            opponents=FrozenPolicies(state["opponents"], ppo, unity_seed)
            worker_failed=False
            session.publish("launching_workers",learning_rate=ppo.schedule.rate(),
                            difficulty=diff,shaping=shape)
            with UnityWorkers(executable, worker_count, phase.stage, diff, shape,
                              unity_seed, phase.areas,
                              log_dir=run_dir / "worker-logs",task_index=task_index) as workers:
                # Environment parameters change only when a curriculum bin changes.
                bin_value = (diff, shape)
                while state["phase_steps"] < phase.budget+state["phase_budget_extension"]:
                    session.check_stop()
                    session.publish("sampling",learning_rate=ppo.schedule.rate(),
                                    difficulty=diff,shaping=shape)
                    collection_start=time.perf_counter()
                    try:
                        rollout, raw, episodes, episode_logs = collect(ppo, workers, opponents,
                                                         batch_size, session.check_stop)
                    except (TimeoutError,ConnectionError,EOFError,BrokenPipeError,
                            RuntimeError,OSError) as error:
                        state["worker_restarts"]+=1
                        state["last_worker_error"]=repr(error)
                        writer.add_scalar("sampling/worker_restarts",
                                          state["worker_restarts"],state["global_updates"])
                        if state["worker_restarts"]>3:raise
                        worker_failed=True
                        break
                    sampling_seconds=max(1e-6,time.perf_counter()-collection_start)
                    session.publish("updating",learning_rate=ppo.schedule.rate())
                    metrics = ppo.update(rollout, phase.entropy)
                    update_seconds=max(1e-6,time.perf_counter()-collection_start-
                                       sampling_seconds)
                    ppo.normalizer.update(raw)
                    samples = len(raw)
                    state["phase_steps"] += samples
                    state["global_steps"] += samples
                    state["global_updates"] += 1
                    state["episodes"] += episodes
                    state["horse_steps_per_second"] = samples/sampling_seconds
                    state["episode_seed"] += worker_count * 100000
                    seen={(item["source_seed"],item["episodeId"])
                          for item in state["recent_episodes"]}
                    unique=[]
                    for item in episode_logs:
                        item["source_seed"]=unity_seed+item["worker"]*100000
                        key=(item["source_seed"],item["episodeId"])
                        if key in seen:continue
                        seen.add(key)
                        unique.append(item)
                    state["recent_episodes"]=(state["recent_episodes"]+unique)[-100:]
                    episode_stats=episode_summary(state["recent_episodes"])
                    state["recent_episodes_summary"]=episode_stats
                    # Commit before evaluation, which may be interrupted. Never
                    # serialize a partial PPO optimizer update in finally.
                    save_latest()
                    stop_after_update = max_updates is not None and \
                        state["global_updates"]>=max_updates
                    for name, value in metrics.items():
                        writer.add_scalar("ppo/" + name, value, state["global_updates"])
                    writer.add_scalar("reward/mean", float(rollout.reward.mean()), state["global_updates"])
                    writer.add_scalar("sampling/horse_steps_per_second",
                                      samples/sampling_seconds,state["global_updates"])
                    for name,value in episode_stats.items():
                        if isinstance(value,(int,float)):
                            writer.add_scalar("episode_recent100/"+name,value,state["global_updates"])
                    writer.add_scalar("sampling/actor_active_fraction",
                                      float(rollout.actor_mask.mean()),state["global_updates"])
                    writer.add_scalar("sampling/action_std",
                                      float(rollout.action.std()),state["global_updates"])
                    writer.add_scalar("curriculum/stage", phase.stage, state["global_updates"])
                    record={"update":state["global_updates"],"stage":phase.stage,
                            "global_steps":state["global_steps"],
                            "phase_steps":state["phase_steps"],"batch":samples,
                            "episodes_total":state["episodes"],"difficulty":diff,
                            "shaping":shape,"reward_mean":float(rollout.reward.mean()),
                            "sampling_seconds":sampling_seconds,
                            "update_seconds":update_seconds,
                            "horse_steps_per_second":samples/sampling_seconds,
                            "worker_restarts":state["worker_restarts"],
                            "task_index":task_index,"task_name":task(phase.stage,task_index)[0],
                            "task_distance":task(phase.stage,task_index)[1],
                            "task_time_limit":task(phase.stage,task_index)[2],
                            "actor_active_fraction":float(rollout.actor_mask.mean()),
                            "action_std":float(rollout.action.std()),
                            "recent100":episode_stats,**metrics}
                    with (run_dir / "training_metrics.jsonl").open("a",encoding="utf-8") as log:
                        log.write(json.dumps(record,ensure_ascii=False)+"\n")
                    if unique:
                        with (run_dir / "episode_metrics.jsonl").open("a",encoding="utf-8") as log:
                            for item in unique:
                                log.write(json.dumps(item,ensure_ascii=False)+"\n")
                    if state["global_updates"] % 50 == 0:
                        save_periodic()
                    if state["global_updates"] % evaluation_interval(phase.stage) == 0:
                        session.check_stop()
                        session.publish("evaluating",learning_rate=ppo.schedule.rate())
                        report = evaluate(ppo, executable, phase.stage, seed + 500000,
                                          difficulty_level=1.0 if phase.stage in (3, 4) else diff,
                                          shaping_level=shape,opponent_paths=state["opponents"],
                                          check_stop=session.check_stop,
                                          log_dir=run_dir / "evaluation-logs",
                                          task_index=task_index)
                        state["history"].append(report)
                        (run_dir / "evaluations.jsonl").open("a", encoding="utf-8").write(
                            json.dumps(report, ensure_ascii=False) + "\n")
                        for name in ("completion","fall_free","median_time","falls_per_100m",
                                     "disturbance_recovery"):
                            if report[name] is not None:
                                writer.add_scalar("evaluation/"+name,report[name],state["global_updates"])
                        if phase.stage==1:
                            next_task,consecutive,ready=stage1_result(task_index,report,state["passes"])
                            state["passes"]=consecutive
                            if next_task!=task_index:
                                state["stage1_task_index"]=next_task
                                state["passes"]=0
                                state["recent_episodes"]=[]
                                state["recent_episodes_summary"]={"episodes":0}
                                save_latest()
                                break
                        if phase.stage==6:
                            key=str(shape)
                            previous=state["shape_best"].get(key)
                            if previous is None or (report["completion"],
                                    -(report["median_time"] or math.inf)) > (
                                    previous["completion"],-(previous["median_time"] or math.inf)):
                                best_path=run_dir / ("stage6-shaping-"+key+".pt")
                                state["shape_best"][key]={"completion":report["completion"],
                                    "median_time":report["median_time"],"path":str(best_path)}
                                checkpoint.save(best_path,ppo,state,metadata)
                            index=state["shaping_index"]
                            preceding=SHAPING_LEVELS[index-1] if index>0 else None
                            baseline=(state["shape_best"].get(str(preceding)) if index>0
                                      else state["stage5_baseline"])
                            baseline_completion=baseline["completion"] if baseline else 0.0
                            if index>0 and baseline and \
                                    report["completion"]<baseline_completion-.05:
                                saved=checkpoint.load(baseline["path"],ppo,weights_only=True)
                                ppo.optimizer.load_state_dict(saved["optimizer"])
                                state["shaping_index"]=index-1
                                state["shaping_level_start"]=state["phase_steps"]
                                state["phase_budget_extension"]+=phase.budget//10
                                state["passes"]=0
                                save_latest()
                                break
                            if stage6_can_advance(index,state["phase_steps"],
                                    state["shaping_level_start"],phase.budget,
                                    report["completion"],baseline_completion):
                                state["shaping_index"]=index+1
                                state["shaping_level_start"]=state["phase_steps"]
                                if index+1==len(SHAPING_LEVELS)-1:
                                    final_end=state["phase_steps"]+math.ceil(.2*phase.budget)
                                    state["phase_budget_extension"]=max(
                                        state["phase_budget_extension"],final_end-phase.budget)
                                save_latest()
                                break
                        eligible=(phase.stage!=1 or task_index==len(STAGE1_TASKS)-1) and \
                            (phase.stage not in (3,4,5) or diff>=1.0) and \
                            (phase.stage!=6 or stage6_final_ready(
                                state["shaping_index"],state["phase_steps"],
                                state["shaping_level_start"],phase.budget))
                        best_key=str(phase.stage)
                        score=(report["completion"],-(report["median_time"] or math.inf),
                               -report["falls_per_100m"])
                        prior=state["best"].get(best_key)
                        if eligible and (prior is None or score>tuple(prior["score"])):
                            best_path=run_dir / f"stage{phase.stage}-best.pt"
                            state["best"][best_key]={"score":score,"report":report,
                                "path":str(best_path)}
                            checkpoint.save(best_path,ppo,state,metadata)
                        if phase.stage!=1:
                            state["passes"] = state["passes"] + 1 if eligible and passes(
                                phase.stage, report, state["stage5_baseline"]) else 0
                        if state["passes"] >= 3:
                            candidate=run_dir / f"stage{phase.stage}-candidate.pt"
                            checkpoint.save(candidate,ppo,state,metadata)
                            freeze_candidate(state,candidate,report)
                            save_latest()
                            session.publish("pending_review",learning_rate=ppo.schedule.rate())
                            break
                        if phase.stage==5 and report["completion"]>=.85:
                            opponent_path=run_dir / f"stage5-opponent-{state['global_updates']}.pt"
                            checkpoint.save(opponent_path,ppo,state,
                                metadata)
                            state["opponents"].append(str(opponent_path))
                            state["opponents"]=state["opponents"][:1]+state["opponents"][-8:]
                            opponents.paths=list(state["opponents"])
                        save_latest()
                    new_bin = (difficulty(phase.stage, state["phase_steps"] / phase.budget),
                               SHAPING_LEVELS[state["shaping_index"]] if phase.stage==6 else 1.0)
                    if new_bin != bin_value:
                        break
                    if stop_after_update:
                        break
            if max_updates is not None and state["global_updates"]>=max_updates:
                break
            if worker_failed:
                save_latest()
                continue
            if state["passes"] >= 3:
                break
            elif state["phase_steps"] >= phase.budget+state["phase_budget_extension"]:
                checkpoint.save(run_dir / f"stage{phase.stage}-needs-work.pt", ppo, state,
                                metadata)
                break
    except StopRequested:
        stopped=True
        session.publish("stopping",learning_rate=ppo.schedule.rate(),
                        message="Uncommitted rollout discarded; latest.pt is the last complete update")
    except BaseException as error:
        session.publish("failed",error=repr(error),learning_rate=ppo.schedule.rate())
        raise
    finally:
        writer.close()
    session.publish("stopped" if stopped else
                    "pending_review" if review_state(state)["status"]=="pending" else
                    "completed" if review_state(state)["status"]=="approved" else "idle",
                    learning_rate=ppo.schedule.rate())
    return state
