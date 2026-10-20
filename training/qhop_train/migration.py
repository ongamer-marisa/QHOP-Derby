"""Explicit old-contract migration. Never used by ordinary resume."""
from pathlib import Path
import hashlib
import numpy as np
import torch
from .schema import ACTION, LEGACY_OBSERVATION, OBSERVATION

R11_REWARD = "gait-v1:stage2:p=.2+.8q;process=.3;cycle=.5;bad=-1;high=-.5;fold=-.5;slip=-.2;posture=-.2;fall=-2;finish=2q;dnf=-5;later=.25"
R11_COURSE = "seven-stage-v1:stage2=30m20s-trot,50m20s-canter,80m20s-gallop;eval25;gait-gate=.85;coverage=.9"
R12_REWARD = "gait-v2:stage1=.1;coord=1.2-independent;speed=.6;process=.3;cycle=.5period;order=-.5;sync=-.5;over=-.3;high=-.5;fold=-.5;slip=-.2;posture=-.2;fall=-2;finish=2q;dnf=-5;early=-2.82remaining;later=.25"
R12_COURSE = "seven-stage-v2:stage1=walk-trot;stage2=30,50,80m-fixed20s;warmup=1.5-2;eval25;gait-gate=.85;coverage=.9;speed-band=.8"

R10_REWARD = "stage1-v2:p=2;posture=-0.10(1-u);time=-0.05;fall=-0.25;finish=2;dnf=-1;early-dnf-remaining-time=-0.05"
R10_COURSE = "stage1-v2:2m-10s;5m-15s;10m-20s;20m-20s;eval-25;early-pass=0.80,0.60;final-pass=0.90,0.80x3"

def adapt(source_path, ppo, *, legacy_stage1=False):
    raise ValueError("r14 requires training from scratch; old policy migration is disabled")
