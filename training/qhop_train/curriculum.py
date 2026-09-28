"""Frozen, auditable six-stage curriculum settings."""
from dataclasses import dataclass
import hashlib
import math


@dataclass(frozen=True)
class Phase:
    stage: int
    name: str
    budget: int
    peak_lr: float
    entropy: float
    opponents: int
    areas: int

    def updates(self, batch=16384):
        return math.ceil(self.budget / batch)


PHASES = (
    Phase(1, "forward", 3_000_000, 3e-4, .01, 1, 8),
    Phase(2, "efficient", 8_000_000, 3e-4, .005, 1, 8),
    Phase(3, "morphology", 12_000_000, 3e-4, .005, 1, 8),
    Phase(4, "robustness", 15_000_000, 3e-4, .005, 1, 8),
    Phase(5, "race", 15_000_000, 1.5e-4, .003, 4, 2),
    Phase(6, "final", 10_000_000, 1e-4, .001, 4, 2),
)

SHAPING_LEVELS = (1.0, .75, .5, .25, 0.0)
STAGE1_TASKS = (("初步迈步", 2.0, 10.0), ("连续移动", 5.0, 15.0),
                ("延长距离", 10.0, 20.0), ("阶段验收", 20.0, 20.0))
REWARD_CONTRACT = "stage1-v2:p=2;posture=-0.10(1-u);time=-0.05;fall=-0.25;finish=2;dnf=-1;early-dnf-remaining-time=-0.05"
COURSE_CONTRACT = "stage1-v2:2m-10s;5m-15s;10m-20s;20m-20s;eval-25;early-pass=0.80,0.60;final-pass=0.90,0.80x3"
REWARD_HASH = hashlib.sha256(REWARD_CONTRACT.encode("utf-8")).hexdigest()
COURSE_HASH = hashlib.sha256(COURSE_CONTRACT.encode("utf-8")).hexdigest()


def task(stage, index=0):
    if stage == 1:
        return STAGE1_TASKS[index]
    return (PHASES[stage - 1].name, 100.0,
            45.0 if stage in (2, 3) else 60.0 if stage == 4 else 90.0)


def evaluation_interval(stage):
    return 25 if stage == 1 else 100


def stage1_result(index, report, consecutive):
    if index < len(STAGE1_TASKS) - 1:
        return (index + 1, 0, False) if report["completion"] >= .8 and \
            report["fall_free"] >= .6 else (index, 0, False)
    consecutive = consecutive + 1 if passes(1, report) else 0
    return index, consecutive, consecutive >= 3


def stage6_can_advance(index, phase_steps, level_start, budget,
                       completion, baseline_completion):
    """Advance only after at least 10% of budget and no >5 point regression."""
    return (index < len(SHAPING_LEVELS) - 1 and
            phase_steps - level_start >= budget * .1 and
            completion >= baseline_completion - .05)


def stage6_final_ready(index, phase_steps, level_start, budget):
    return (index == len(SHAPING_LEVELS) - 1 and
            phase_steps - level_start >= budget * .2)


def difficulty(stage, fraction):
    if stage == 3:
        return (.25, .5, .75, 1.0)[min(3, int(max(0, fraction) * 4))]
    if stage == 4:
        return (.2, .4, .6, .8, 1.0)[min(4, int(max(0, fraction) * 5))]
    if stage == 5:
        return 0.0 if fraction < .2 else 1.0
    if stage == 6:
        return 1.0
    return 0.0


def shaping(stage, fraction):
    if stage != 6:
        return 1.0
    return SHAPING_LEVELS[min(4, int(max(0, fraction) * 5))]


def passes(stage, report, stage5=None):
    completion = report.get("completion", 0)
    if stage == 1:
        return completion >= .9 and report.get("fall_free", 0) >= .8
    if stage == 2:
        return completion >= .95 and report.get("median_speed", 0) >= 3 and report.get("falls_per_100m", 1e9) <= .1
    if stage == 3:
        return completion >= .9 and report.get("worst_morphology_completion", 0) >= .8
    if stage == 4:
        return completion >= .85 and report.get("disturbance_recovery", 0) >= .8
    if stage == 5:
        return completion >= .9 and report.get("collision_rule_errors", 1) == 0
    if stage == 6 and stage5:
        return (report.get("shaping", 1) == 0 and
                completion >= stage5["completion"] - .02 and
                report.get("median_time") is not None and
                stage5.get("median_time") is not None and
                report["median_time"] <= stage5["median_time"] * 1.05)
    return False
