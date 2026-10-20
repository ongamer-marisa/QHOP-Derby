"""Embed input normalization in a deterministic ONNX actor."""
from pathlib import Path
import json
import hashlib

import numpy as np
import torch
from torch import nn

from . import checkpoint
from .ppo import PPO
from .schema import ACTION, FRAME, HISTORY, OBSERVATION, PHYSICS_FINGERPRINT, VERSION, normalization_mask, VISION_FEATURES, VISION_HIDDEN, LIDAR_CONTRACT, MODEL_ARCHITECTURE, RIG_CONTRACT


class DeploymentActor(nn.Module):
    def __init__(self, actor, mean, variance):
        super().__init__()
        self.actor = actor
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.as_tensor(np.sqrt(variance + 1e-8), dtype=torch.float32))
        self.register_buffer("mask", torch.as_tensor(normalization_mask(), dtype=torch.float32))

    def forward(self, raw, hidden):
        normalized = ((raw - self.mean) / self.scale).clamp(-10, 10)
        prepared = normalized * self.mask + raw * (1 - self.mask)
        return self.actor(prepared, hidden)


def export(checkpoint_path, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ppo = PPO("cpu")
    payload = checkpoint.load(checkpoint_path, ppo, weights_only=True)
    if payload.get("metadata",{}).get("physics_fingerprint")!=PHYSICS_FINGERPRINT:
        raise ValueError("Checkpoint physics contract differs from the current Unity model")
    deploy = DeploymentActor(ppo.actor.eval(), ppo.normalizer.mean,
                             ppo.normalizer.var).eval()
    dummy = torch.zeros((1, OBSERVATION), dtype=torch.float32)
    hidden = torch.zeros((1, VISION_HIDDEN), dtype=torch.float32)
    torch.onnx.export(deploy, (dummy,hidden), destination, input_names=["observation","hidden"],
                      output_names=["action"], opset_version=15,
                      dynamic_axes={"observation": {0: "horses"}, "hidden": {0:"horses"}, "action": {0: "horses"}})
    memory_path=destination.with_name(destination.stem+"Vision.onnx")
    torch.onnx.export(ppo.vision.eval(),(torch.zeros(1,VISION_FEATURES),hidden),memory_path,
        input_names=["scan","hidden"],output_names=["next_hidden"],opset_version=15,
        dynamic_axes={"scan":{0:"horses"},"hidden":{0:"horses"},"next_hidden":{0:"horses"}})
    metadata = {"schema_version": VERSION, "observation_size": OBSERVATION,
                "action_size": ACTION, "frame_size": FRAME, "history": HISTORY,
                "stage": payload["training"]["stage"], "model_name": destination.stem,
                "normalization_embedded": True,
                "physics_fingerprint": PHYSICS_FINGERPRINT,
                "unity_build": payload.get("metadata", {}).get("unity_build", "unverified"),
                "vision_features":VISION_FEATURES,"vision_hidden":VISION_HIDDEN,"lidar_contract":LIDAR_CONTRACT,
                "rig_contract":RIG_CONTRACT,
                "model_architecture":MODEL_ARCHITECTURE,"vision_model":memory_path.name,
                "action_sha256":hashlib.sha256(destination.read_bytes()).hexdigest(),
                "vision_sha256":hashlib.sha256(memory_path.read_bytes()).hexdigest()}
    destination.with_name(destination.stem + "Metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
