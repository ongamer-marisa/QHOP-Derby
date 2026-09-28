"""qhop-train: train, resume, evaluate, benchmark and export."""
import argparse
from importlib.util import find_spec
from pathlib import Path
import json
import time
import sys

import numpy as np

from .build_contract import verify as verify_build
from .schema import ACTION, FRAME, OBSERVATION, PHYSICS_FINGERPRINT, VERSION
from .curriculum import REWARD_HASH, COURSE_HASH


def default_executable():
    return Path(__file__).resolve().parents[2] / "Builds" / "Training" / "QHopTraining.exe"


def doctor(executable):
    packages = {name: find_spec(name) is not None for name in
                ("numpy", "torch", "onnx", "mlagents_envs", "tensorboard", "psutil")}
    report = {"python_packages": packages, "unity_build_exists": executable.is_file(),
              "unity_build": str(executable), "schema_version": VERSION,
              "observation": OBSERVATION, "action": ACTION}
    report["reward_hash"] = REWARD_HASH
    report["course_hash"] = COURSE_HASH
    source=Path(__file__).resolve().parents[2] / "Assets/Scripts/HorseTrainingObservation.cs"
    contract=source.read_text(encoding="utf-8") if source.is_file() else ""
    report["unity_schema_matches"]=(
        f"public const int Version={VERSION};" in contract and
        f'PhysicsFingerprint="{PHYSICS_FINGERPRINT}"' in contract and
        f"public const int ProprioFrame={FRAME};" in contract)
    try:
        report["unity_build_sha256"]=verify_build(executable)
        report["unity_build_contract_matches"]=True
    except (OSError,ValueError,KeyError) as error:
        report["unity_build_contract_matches"]=False
        report["unity_build_error"]=str(error)
    if packages["torch"]:
        import torch
        report["torch_version"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if all(packages.values()) and executable.is_file() and \
        report["unity_schema_matches"] and report["unity_build_contract_matches"] and \
        report.get("cuda_available",False) else 2


def benchmark(executable, stage, seconds, seed):
    verify_build(executable)
    from .unity_env import UnityWorkers
    import psutil

    results = []
    total_areas = 32 if stage < 5 else 8
    for count in (1, 2, 4):
        arenas = total_areas // count
        with UnityWorkers(executable, count, stage, 1.0, 0.0, seed, arenas,
                          worker_id_base=700) as workers:
            total = 0
            peak_memory = 0
            started = time.perf_counter()
            next_memory_sample=started
            while time.perf_counter() - started < seconds:
                actions = [{role: np.zeros((len(s[role]["decision_id"]), ACTION), dtype=np.float32)
                            for role in ("learner","frozen")}
                           for s in workers.current]
                states = workers.step(actions)
                total += sum(len(s["learner"]["decision_id"]) +
                             len(s["learner"]["terminal_id"]) for s in states)
                if time.perf_counter()>=next_memory_sample:
                    memory=0
                    for worker in workers.processes:
                        try:
                            parent=psutil.Process(worker.pid)
                            memory+=sum(process.memory_info().rss for process in
                                        (parent,*parent.children(recursive=True)))
                        except (psutil.NoSuchProcess,psutil.AccessDenied):
                            pass
                    peak_memory=max(peak_memory,memory)
                    next_memory_sample=time.perf_counter()+1.0
            elapsed = time.perf_counter() - started
            results.append({"workers": count, "areas_per_worker": arenas,
                            "horse_steps_per_second": total / elapsed,
                            "peak_worker_memory_mb": round(peak_memory / 1024**2, 1)})
    print(json.dumps(results, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qhop-train")
    parser.add_argument("--executable", type=Path, default=default_executable())
    parser.add_argument("--log-file", type=Path,
                        help="Write this process's stdout/stderr to a persistent log")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")
    bench = sub.add_parser("benchmark")
    bench.add_argument("--stage", type=int, choices=range(1, 7), default=1)
    bench.add_argument("--seconds", type=float, default=20)
    bench.add_argument("--seed", type=int, default=20260927)
    for name in ("train", "resume", "finetune"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=name != "train")
        command.add_argument("--stage", type=int, choices=range(1, 7), default=1)
        command.add_argument("--workers", type=int, default=4)
        command.add_argument("--batch", type=int, default=16384)
        command.add_argument("--max-updates", type=int, default=None,
                             help="Stop cleanly after this many total PPO updates")
        command.add_argument("--seed", type=int, default=20260927)
        command.add_argument("--extra-steps", type=int, default=0,
                             help="Extend an unfinished stage when resuming; LR remains at its floor")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    evaluate_parser.add_argument("--stage", type=int, choices=range(1, 7), required=True)
    evaluate_parser.add_argument("--races", type=int, default=100)
    evaluate_parser.add_argument("--seed", type=int, default=72500)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--destination", type=Path, default=
        Path(__file__).resolve().parents[2] / "Assets" / "Resources" / "QHopPolicy.onnx")
    info_parser = sub.add_parser("checkpoint-info")
    info_parser.add_argument("--checkpoint", type=Path, required=True)
    preview_parser = sub.add_parser("preview")
    preview_parser.add_argument("--checkpoint", type=Path, required=True)
    preview_parser.add_argument("--output", type=Path, required=True)
    preview_parser.add_argument("--seed", type=int, default=92500)
    preview_parser.add_argument("--max-episodes", type=int)
    preview_parser.add_argument("--max-seconds", type=float,
                                help="Short environment-launch smoke test only")
    for name in ("approve-stage", "continue-stage"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        if name == "continue-stage":
            command.add_argument("--extra-steps", type=int, default=0)
    stop_parser = sub.add_parser("stop")
    stop_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log = args.log_file.open("a", encoding="utf-8", buffering=1)
        sys.stdout = log
        sys.stderr = log
    if args.command == "doctor":
        return doctor(args.executable)
    if args.command == "benchmark":
        benchmark(args.executable, args.stage, args.seconds, args.seed)
        return 0
    if args.command in ("train", "resume", "finetune"):
        if args.command == "resume" and args.checkpoint.resolve().parent == args.output.resolve() \
                and args.checkpoint.name != "latest.pt":
            raise ValueError("Resume from an older checkpoint into a new branch output directory")
        from .runner import train
        train(args.executable, args.output, seed=args.seed, worker_count=args.workers,
              resume=args.checkpoint if args.command == "resume" else None,
              finetune=args.checkpoint if args.command == "finetune" else None,
              initial_stage=args.stage, batch_size=args.batch,
              extra_steps=args.extra_steps,max_updates=args.max_updates)
        return 0
    if args.command == "checkpoint-info":
        from .review import checkpoint_info
        print(json.dumps(checkpoint_info(args.checkpoint), indent=2, ensure_ascii=False))
        return 0
    if args.command == "preview":
        from .preview import preview
        print(json.dumps(preview(args.executable, args.checkpoint, args.output,
                                 seed=args.seed, max_episodes=args.max_episodes,
                                 max_seconds=args.max_seconds),
                         indent=2, ensure_ascii=False))
        return 0
    if args.command == "approve-stage":
        from .review import approve_stage
        state = approve_stage(args.output)
        print(json.dumps({"stage": state["stage"], "review": state["review"]},
                         ensure_ascii=False))
        return 0
    if args.command == "continue-stage":
        from .review import reject_and_continue
        state = reject_and_continue(args.output, args.extra_steps)
        print(json.dumps({"stage": state["stage"], "review": state["review"]},
                         ensure_ascii=False))
        return 0
    if args.command == "stop":
        from .control import read_json, request_stop
        status = read_json(args.output / "session-status.json")
        if not status or not status.get("session_id"):
            raise ValueError("No trainer session is recorded in this experiment")
        request_stop(args.output, status["session_id"])
        print(json.dumps({"requested": "stop", "session_id": status["session_id"]}))
        return 0
    if args.command == "evaluate":
        build_hash=verify_build(args.executable)
        from .ppo import PPO
        from .checkpoint import load
        from .runner import evaluate
        import torch
        ppo = PPO("cuda" if torch.cuda.is_available() else "cpu")
        payload=load(args.checkpoint, ppo, weights_only=True)
        source=payload.get("metadata",{})
        if source.get("unity_build")!=build_hash or \
                source.get("physics_fingerprint")!=PHYSICS_FINGERPRINT or \
                source.get("reward_hash")!=REWARD_HASH or \
                source.get("course_hash")!=COURSE_HASH:
            raise ValueError("Checkpoint Unity, reward, or curriculum contract differs")
        task_index=payload["training"].get("stage1_task_index",3) if args.stage==1 else 0
        report = evaluate(ppo, args.executable, args.stage, seed=args.seed,
                          races=args.races, difficulty_level=1.0, shaping_level=0.0,
                          opponent_paths=payload["training"].get("opponents", []),
                          task_index=task_index)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0
    if args.command == "export":
        from .export import export
        print(json.dumps(export(args.checkpoint, args.destination), indent=2))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
