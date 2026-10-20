"""Shared observation contract. Keep these constants aligned with HorseTrainingObservation.cs."""
from dataclasses import dataclass

VERSION = 4
# Bump together with HorseTrainingObservation.cs after physics/control changes.
PHYSICS_FINGERPRINT = "qh-100hz-5step-19rb-28a-lidar-gru-terminal-r14"
FRAME = 243
HISTORY = 4
TERRAIN = 70
OPPONENT = 12
OPPONENTS = 3
ACTION = 28
LEGACY_OBSERVATION = FRAME * HISTORY + TERRAIN + OPPONENT * OPPONENTS
GAIT = 28
BASE_OBSERVATION = LEGACY_OBSERVATION + GAIT
LIDAR_RAYS = 57
VISION_FEATURES = 177
VISION_HIDDEN = 128
LIDAR_SIZE = 179
OBSERVATION = BASE_OBSERVATION + LIDAR_SIZE
LIDAR_CONTRACT = "lidar-v1:57;front=-70:10:70x3@25m;side=90,-90,135,-135x3@10m;heights=.1,.6,1.1;5hz;upright;self-filter;motion;dt;age;sequence"
MODEL_ARCHITECTURE = "grouped-mlp-shared-lidar-gru-r14"
RIG_CONTRACT = "rig-r14:19rb;18j;28a;mass-preserved;canonical-r13-limits;spring=240000;damper=3000;cap=40000;passive=5000,350;initial-power=.85;solver=64,16;spawn=fourhoof-plane;torso-com-forward=.2*bodyLength"


@dataclass(frozen=True)
class Slices:
    proprio: slice = slice(0, FRAME * HISTORY)
    terrain: slice = slice(FRAME * HISTORY, FRAME * HISTORY + TERRAIN)
    opponents: slice = slice(FRAME * HISTORY + TERRAIN, LEGACY_OBSERVATION)
    gait: slice = slice(LEGACY_OBSERVATION, BASE_OBSERVATION)
    lidar: slice = slice(BASE_OBSERVATION, BASE_OBSERVATION + VISION_FEATURES)
    age: int = BASE_OBSERVATION + VISION_FEATURES
    scan_id: int = OBSERVATION - 1


SLICES = Slices()


def normalization_mask():
    import numpy as np

    mask = np.ones(OBSERVATION, dtype=np.float32)
    # Each proprio frame: four contacts, 3 condition flags and ground contact.
    for frame in range(HISTORY):
        base = frame * FRAME
        for i in range(4):
            mask[base + 174 + i * 4] = 0
        mask[base + 205:base + 209] = 0
        mask[base + 210:base + 212] = 0
    mask[FRAME * HISTORY + 1:FRAME * HISTORY + TERRAIN:2] = 0
    for i in range(OPPONENTS):
        base = FRAME * HISTORY + TERRAIN + i * OPPONENT
        mask[base + 9:base + 12] = 0
    mask[LEGACY_OBSERVATION:] = 0
    return mask
