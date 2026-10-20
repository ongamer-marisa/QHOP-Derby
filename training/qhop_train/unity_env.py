"""One spawned Python process per headless Unity simulator."""
import multiprocessing as mp
import os
import math
from pathlib import Path
import time

import numpy as np

from .schema import ACTION, OBSERVATION


def _one(env, behavior):
    if behavior is None:
        return {
            "decision_id": np.empty(0,dtype=np.int32),
            "decision_obs": np.empty((0,OBSERVATION),dtype=np.float32),
            "decision_reward": np.empty(0,dtype=np.float32),
            "terminal_id": np.empty(0,dtype=np.int32),
            "terminal_obs": np.empty((0,OBSERVATION),dtype=np.float32),
            "terminal_reward": np.empty(0,dtype=np.float32),
            "interrupted": np.empty(0,dtype=bool),
        }
    decision, terminal = env.get_steps(behavior)
    if terminal.interrupted.any():
        raise RuntimeError("Unexpected Unity MaxStep interruption; discard this rollout")
    observations = decision.obs[0].astype(np.float32).copy()
    final_observations = terminal.obs[0].astype(np.float32).copy()
    if observations.shape[1:] != (OBSERVATION,) or final_observations.shape[1:] != (OBSERVATION,):
        raise ValueError("Unity observation schema mismatch")
    if not np.isfinite(observations).all() or not np.isfinite(final_observations).all() or not np.isfinite(decision.reward).all() or not np.isfinite(terminal.reward).all():
        raise ValueError("Non-finite Unity observation or reward")
    return {
        "decision_id": decision.agent_id.copy(),
        "decision_obs": observations,
        "decision_reward": decision.reward.copy(),
        "terminal_id": terminal.agent_id.copy(),
        "terminal_obs": final_observations,
        "terminal_reward": terminal.reward.copy(),
        "interrupted": terminal.interrupted.copy(),
    }


def _snapshot(env):
    names = list(env.behavior_specs)
    learner = next((name for name in names if name.startswith("QHopCurrentV2")), None)
    frozen = next((name for name in names if name.startswith("QHopFrozenV2")), None)
    if learner is None:
        raise ValueError(f"QHopCurrentV2 behavior missing: {names}")
    for name in (learner, frozen):
        if name is not None and env.behavior_specs[name].action_spec.continuous_size != ACTION:
            raise ValueError("Unity action schema mismatch")
    return {"learner": _one(env, learner), "frozen": _one(env, frozen),
            "names": {"learner": learner, "frozen": frozen}}


def _worker(pipe, executable, worker_id, stage, difficulty, shaping, seed, areas,
            log_dir=None, task_index=0):
    from mlagents_envs.environment import ActionTuple, UnityEnvironment
    from .curriculum import task
    from .episode_channel import EpisodeChannel

    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
    engine=EngineConfigurationChannel()
    scale=float(os.environ.get("QHOP_TRAIN_TIME_SCALE","1"))
    if not math.isfinite(scale) or not 1<=scale<=20:
        raise ValueError("QHOP_TRAIN_TIME_SCALE must be within [1,20]")
    engine.set_configuration_parameters(time_scale=scale,target_frame_rate=-1)
    channel = EpisodeChannel()
    _, distance, time_limit = task(stage, task_index)
    log_dir=Path(log_dir) if log_dir is not None else Path(executable).resolve().parent / "worker-logs"
    log_dir.mkdir(parents=True,exist_ok=True)
    args = [f"--qhop-stage={stage}", f"--qhop-difficulty={difficulty}",
            f"--qhop-shaping={shaping}", f"--qhop-seed={seed}", f"--qhop-areas={areas}",
            f"--qhop-task-index={task_index}",f"--qhop-distance={distance}",
            f"--qhop-time-limit={time_limit}",
            "-logFile",str(log_dir / f"worker-{worker_id}.log")]
    if os.environ.get("QHOP_PHYSICS_TRACE")=="1":args.append("--qhop-physics-trace")
    if os.environ.get("QHOP_VALIDATE_MORPH")=="1":args.append("--qhop-validation-morph")
    env = None
    try:
        env = UnityEnvironment(file_name=str(executable), worker_id=worker_id,
                               seed=seed, no_graphics=True, additional_args=args,
                               side_channels=[channel,engine], timeout_wait=120)
        env.reset()
        initial=_snapshot(env)
        initial["episode_logs"]=channel.drain()
        pipe.send(("ok", initial))
        while True:
            command, value = pipe.recv()
            if command == "close":
                break
            if command != "step":
                raise ValueError(command)
            state = _snapshot(env)
            for role in ("learner", "frozen"):
                actions = np.asarray(value[role], dtype=np.float32).reshape(-1, ACTION)
                decision = state[role]
                if len(decision["decision_id"]) != len(actions):
                    raise ValueError(f"{role} action count mismatch")
                if len(actions):
                    env.set_actions(state["names"][role], ActionTuple(continuous=actions))
            # UnityEnvironment.step() advances until an Academy event, which
            # can be an early terminal rather than the next 20 Hz decision.
            # Wait until every submitted agent has either terminated or made
            # its next decision; otherwise PPO would lose part of the batch.
            terminals = {role: [] for role in ("learner", "frozen")}
            episode_logs=[]
            expected = {role: set(map(int, state[role]["decision_id"]))
                        for role in ("learner", "frozen")}
            for _ in range(10):
                env.step()
                episode_logs.extend(channel.drain())
                following = _snapshot(env)
                for role in terminals:
                    if len(following[role]["terminal_id"]):
                        terminals[role].append(following[role])
                    expected[role].difference_update(map(int, following[role]["terminal_id"]))
                    expected[role].difference_update(map(int, following[role]["decision_id"]))
                if all(not remaining for remaining in expected.values()):
                    break
                if any(len(following[role]["decision_id"]) for role in terminals):
                    raise RuntimeError("Unity areas requested asynchronous decisions")
            else:
                raise RuntimeError("Unity did not resolve every submitted action")
            for role, events in terminals.items():
                for field in ("terminal_id", "terminal_obs", "terminal_reward", "interrupted"):
                    if events:
                        following[role][field] = np.concatenate([event[field] for event in events])
            if not any(len(following[role][field]) for role in terminals
                       for field in ("decision_id", "terminal_id")):
                raise RuntimeError("No Unity decision or terminal at the expected boundary")
            following["episode_logs"]=episode_logs
            pipe.send(("ok", following))
    except BaseException as error:
        try:
            pipe.send(("error", repr(error)))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if env is not None:
            env.close()
        pipe.close()


class UnityWorkers:
    def __init__(self, executable, count, stage, difficulty, shaping, seed, areas,
                 worker_id_base=100, log_dir=None, task_index=0):
        context = mp.get_context("spawn")
        self.pipes = []
        self.processes = []
        self.current = []
        for index in range(count):
            parent, child = context.Pipe()
            process = context.Process(target=_worker, args=(child, executable, worker_id_base + index,
                stage, difficulty, shaping, seed + index * 100000, areas, log_dir,
                task_index), daemon=True)
            process.start()
            child.close()
            self.pipes.append(parent)
            self.processes.append(process)
        try:
            self.current = [self._receive(index) for index in range(count)]
        except BaseException:
            self.close()
            raise

    def _receive(self, index, timeout=180):
        if not self.pipes[index].poll(timeout):
            raise TimeoutError(f"Unity worker {index} did not respond")
        status, payload = self.pipes[index].recv()
        if status != "ok":
            raise RuntimeError(f"Unity worker {index}: {payload}")
        return payload

    def step(self, actions):
        for pipe, action in zip(self.pipes, actions):
            pipe.send(("step", action))
        self.current = [self._receive(index) for index in range(len(self.pipes))]
        return self.current

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
        for pipe in self.pipes:
            pipe.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
