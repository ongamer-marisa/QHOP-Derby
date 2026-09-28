"""Shared observation contract. Keep these constants aligned with HorseTrainingObservation.cs."""
from dataclasses import dataclass

VERSION = 2
# Bump together with HorseTrainingObservation.cs after physics/control changes.
PHYSICS_FINGERPRINT = "qh-100hz-5step-19rb-28a-r10"
FRAME = 243
HISTORY = 4
TERRAIN = 70
OPPONENT = 12
OPPONENTS = 3
ACTION = 28
OBSERVATION = FRAME * HISTORY + TERRAIN + OPPONENT * OPPONENTS


@dataclass(frozen=True)
class Slices:
    proprio: slice = slice(0, FRAME * HISTORY)
    terrain: slice = slice(FRAME * HISTORY, FRAME * HISTORY + TERRAIN)
    opponents: slice = slice(FRAME * HISTORY + TERRAIN, OBSERVATION)


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
    return mask
