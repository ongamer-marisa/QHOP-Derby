"""Read-only deterministic replay of a checkpoint's hoof observations.

Landing times are reconstructed from the shared 100 Hz tracker's age fields;
the CSV is sampled at policy decisions (20 Hz), so it is not a full physics log.
"""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from qhop_train import checkpoint
from qhop_train.ppo import PPO
from qhop_train.schema import SLICES,FRAME,HISTORY
from qhop_train.unity_env import UnityWorkers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["QHOP_TRAIN_TIME_SCALE"] = "20"
    torch.set_num_threads(2)
    ppo = PPO("cpu")
    checkpoint.load(args.checkpoint, ppo, weights_only=True)
    executable = Path(__file__).resolve().parents[1] / "Builds/Training/QHopTraining.exe"
    columns = ["time", "forward_speed", "quality", "cycle_age", "period"]
    for foot in ("FL", "FR", "HL", "HR"):
        columns.extend(foot + "_" + name for name in
                       ("contact", "landing_age", "liftoff_age", "swing_height", "relative_z"))
    rows, episodes = [], []
    with UnityWorkers(executable, 1, 2, 0., 1., 20260927, 1,
                      worker_id_base=980, log_dir=args.output) as env:
        for step in range(400):
            obs = env.current[0]["learner"]["decision_obs"]
            if len(obs):
                raw = obs[0]
                gait = raw[SLICES.gait]
                frame=raw[(HISTORY-1)*FRAME:HISTORY*FRAME]
                row = [step * .05, float(frame[2] * 15), float(gait[25]),
                       float(gait[26] * 2.5), float(gait[27] * 2.5)]
                for i in range(4):
                    row.extend(float(x) for x in (frame[174 + i * 4], gait[8+i] * 2.5,
                               gait[12+i] * 2.5, gait[16+i], gait[20+i]))
                rows.append(row)
            with torch.no_grad():
                memories=[ppo.vision_states.consume((0,int(agent)),obs[i])[0] for i,agent in enumerate(env.current[0]["learner"]["decision_id"])]
                actions = ppo.actor(torch.as_tensor(ppo.normalizer.normalize(obs)),torch.cat(memories)).numpy() if memories else np.empty((0,28),np.float32)
            state = env.step([{"learner": actions, "frozen": np.empty((0, 28), np.float32)}])[0]
            episodes.extend(state["episode_logs"])
            for agent in state["learner"]["terminal_id"]:ppo.vision_states.terminal((0,int(agent)))
    with (args.output / "hoof-observations.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)
    (args.output / "episodes.json").write_text(json.dumps(episodes, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(rows), "episodes": len(episodes), "output": str(args.output)}))


if __name__ == "__main__":
    main()
