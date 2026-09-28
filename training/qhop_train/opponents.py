"""Frozen opponent policies; a policy is chosen once per opponent episode."""
from collections import defaultdict
import random

import numpy as np
import torch

from .checkpoint import load
from .ppo import PPO
from .schema import ACTION, PHYSICS_FINGERPRINT


class FrozenPolicies:
    def __init__(self, paths, fallback_ppo, seed=0):
        self.paths = list(paths)
        self.fallback = fallback_ppo
        self.cache = {}
        self.assignment = {}
        self.random = random.Random(seed)

    def _policy(self, path):
        if path is None:
            return self.fallback
        if path not in self.cache:
            policy = PPO("cpu")
            payload=load(path, policy, weights_only=True)
            if payload.get("metadata",{}).get("physics_fingerprint")!=PHYSICS_FINGERPRINT:
                raise ValueError(f"Frozen opponent uses an older physics contract: {path}")
            policy.actor.eval()
            self.cache[path] = policy
        return self.cache[path]

    def actions(self, worker, ids, raw):
        actions = np.empty((len(ids), ACTION), dtype=np.float32)
        groups = defaultdict(list)
        for index, agent_id in enumerate(ids):
            key = (worker, int(agent_id))
            if key not in self.assignment:
                self.assignment[key] = self.random.choice(self.paths) if self.paths else None
            groups[self.assignment[key]].append(index)
        for path, indices in groups.items():
            policy = self._policy(path)
            normalized = policy.normalizer.normalize(raw[indices])
            with torch.no_grad():
                tensor = torch.as_tensor(normalized, device=policy.device)
                actions[indices] = policy.actor(tensor).cpu().numpy()
        return actions

    def terminal(self, worker, ids):
        for agent_id in ids:
            self.assignment.pop((worker, int(agent_id)), None)
