"""Atomic checkpoint save and exact optimizer/scheduler restoration."""
from pathlib import Path
import os
import random
import tempfile

import numpy as np
import torch

from .schema import VERSION


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(value):
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    # torch.load(map_location="cuda") also moves saved CPU RNG bytes to CUDA;
    # the CPU generator only accepts a CPU ByteTensor.
    torch.set_rng_state(value["torch"].cpu())
    if value["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in value["cuda"]])


def save(path, ppo, training_state, metadata):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": VERSION,
        "actor": ppo.actor.state_dict(), "critic": ppo.critic.state_dict(),
        "optimizer": ppo.optimizer.state_dict(), "schedule": ppo.schedule.state_dict(),
        "normalizer": ppo.normalizer.state_dict(), "rng": capture_rng(),
        "training": training_state, "metadata": metadata,
    }
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".qhop-", suffix=".tmp",
                                     delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load(path, ppo, *, weights_only=False):
    payload = torch.load(path, map_location=ppo.device, weights_only=False)
    if payload["schema_version"] != VERSION:
        raise ValueError("Observation schema version mismatch")
    ppo.actor.load_state_dict(payload["actor"])
    ppo.critic.load_state_dict(payload["critic"])
    ppo.normalizer.load_state_dict(payload["normalizer"])
    if not weights_only:
        ppo.optimizer.load_state_dict(payload["optimizer"])
        ppo.schedule.load_state_dict(payload["schedule"])
        restore_rng(payload["rng"])
    return payload
