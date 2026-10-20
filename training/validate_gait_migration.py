"""Read-only identity audit of both actual r12 sources before any PPO update."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from qhop_train.migration import adapt
from qhop_train.ppo import PPO
from qhop_train.schema import OBSERVATION


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    sources=[root/"Runs/editor-20261006-163124-branch-20261006-163650/stage1-approved.pt",
             root/"Runs/editor-20261008-093459-branch-20261008-093509/latest.pt"]
    torch.set_num_threads(2)
    raw=np.random.default_rng(23).normal(size=(64,OBSERVATION)).astype(np.float32)
    results=[]
    for source in sources:
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        old=torch.load(source,map_location="cpu",weights_only=False)
        original,destination=PPO("cpu"),PPO("cpu")
        original.actor.load_state_dict(old["actor"])
        original.normalizer.load_state_dict(old["normalizer"])
        critic={k:v.clone() for k,v in destination.critic.state_dict().items()}
        provenance=adapt(source,destination)
        with torch.no_grad():
            a=original.actor(torch.from_numpy(original.normalizer.normalize(raw)))
            b=destination.actor(torch.from_numpy(destination.normalizer.normalize(raw)))
        error=float((a-b).abs().max())
        assert error<=1e-6, error
        for key in ("mean","var"):
            np.testing.assert_array_equal(getattr(original.normalizer,key),getattr(destination.normalizer,key))
        assert original.normalizer.count==destination.normalizer.count
        assert all(torch.equal(v,destination.critic.state_dict()[k]) for k,v in critic.items())
        assert not destination.optimizer.state and destination.schedule.update==0
        assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
        results.append({**provenance,"actor_max_error":error,"normalization_identical":True,
            "critic_fresh":True,"adam_empty":True,"schedule_update":0,"source_unchanged":True})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(results,indent=2,ensure_ascii=False))


if __name__=="__main__":main()
