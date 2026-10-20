"""Frozen, auditable seven-stage curriculum settings."""
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
    Phase(2, "natural_gait", 8_000_000, 1.5e-4, .01, 1, 8),
    Phase(3, "efficient", 8_000_000, 3e-4, .005, 1, 8),
    Phase(4, "morphology", 12_000_000, 3e-4, .005, 1, 8),
    Phase(5, "robustness", 15_000_000, 3e-4, .005, 1, 8),
    Phase(6, "race", 15_000_000, 1.5e-4, .003, 4, 2),
    Phase(7, "final", 10_000_000, 1e-4, .001, 4, 2),
)

SHAPING_LEVELS = (1.0, .75, .5, .25, 0.0)
STAGE1_TASKS = (("初步迈步", 2.0, 10.0), ("连续移动", 5.0, 15.0),
                ("延长距离", 10.0, 20.0), ("阶段验收", 20.0, 20.0))
STAGE2_TASKS = (("对角小跑", 30.0, 20.0), ("三拍跑", 50.0, 20.0),
                ("四拍疾驰", 80.0, 20.0))
REWARD_CONTRACT = "gait-terminal-v4:stage1=.1;participation=.1;coord=.9(.75D+.25E)P;speed=.6-linear(.1+.9maxCQ);process=.2-token;cycle=.75period;order=-.75period;sync=-1.5period;missing=-.5period;drag=-.3period;over=-.5;high=-.6;fold=-.6-descriptor;slip=-.2;posture=-.2;fall=-2;finish=2q;dnf=-5;early=-5.27remaining;later=.25;fall-terminal;fall-no-remaining-penalty"
COURSE_CONTRACT = "seven-stage-v4:r14-canonical-limits-lidar-gru-fall-terminal;stage1=walk-trot;stage2=30,50,80m-fixed20s;warmup=1.5-2;eval25;gait-gate=.85;coverage=.9;speed-band=.8"
REWARD_HASH = hashlib.sha256(REWARD_CONTRACT.encode("utf-8")).hexdigest()
COURSE_HASH = hashlib.sha256(COURSE_CONTRACT.encode("utf-8")).hexdigest()


def task(stage, index=0):
    if stage == 1:
        return STAGE1_TASKS[index]
    if stage == 2:
        return STAGE2_TASKS[index]
    return (PHASES[stage - 1].name, 100.0,
            45.0 if stage in (3, 4) else 60.0 if stage == 5 else 90.0)


def evaluation_interval(stage):
    return 25 if stage in (1, 2) else 100


def stage2_result(index, report, consecutive):
    if report.get("episodes", 0) < 100 or report.get("gait_coverage", 0) < .9:
        return index, 0, False
    required = (report.get("completion", 0) >= .9 and
                report.get("fall_free", 0) >= (.9 if index == 2 else .85) and
                report.get("median_speed", 0) >= (2, 4, 6)[index] and
                report.get("speed_band_fraction", 0) >= .8 and
                report.get("correct_cycle_ratio", 0) >= (.85 if index == 2 else .8))
    if index == 2:
        required = required and report.get("synchrony_cycle_ratio", 1) <= .05 and \
            report.get("high_hoof_fraction", 1) <= .05
    consecutive = consecutive + 1 if required else 0
    needed = 3 if index == 2 else 2
    if consecutive >= needed and index < 2:
        return index + 1, 0, False
    return index, consecutive, consecutive >= needed


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
    if stage == 4:
        return (.25, .5, .75, 1.0)[min(3, int(max(0, fraction) * 4))]
    if stage == 5:
        return (.2, .4, .6, .8, 1.0)[min(4, int(max(0, fraction) * 5))]
    if stage == 6:
        return 0.0 if fraction < .2 else 1.0
    if stage == 7:
        return 1.0
    return 0.0


def shaping(stage, fraction):
    if stage != 7:
        return 1.0
    return SHAPING_LEVELS[min(4, int(max(0, fraction) * 5))]


def passes(stage, report, stage5=None):
    completion = report.get("completion", 0)
    if stage == 1:
        return completion >= .9 and report.get("fall_free", 0) >= .8
    if stage == 2:
        return stage2_result(2, report, 0)[1] == 1
    if not (report.get("gait_coverage", 0) >= .9 and
            report.get("correct_cycle_ratio", 0) >= .8 and
            report.get("synchrony_cycle_ratio", 1) <= .1 and
            report.get("high_hoof_fraction", 1) <= .1):
        return False
    if stage == 3:
        return completion >= .95 and report.get("median_speed", 0) >= 3 and report.get("falls_per_100m", 1e9) <= .1
    if stage == 4:
        return completion >= .9 and report.get("worst_morphology_completion", 0) >= .8
    if stage == 5:
        return completion >= .85 and report.get("disturbance_recovery", 0) >= .8
    if stage == 6:
        return completion >= .9 and report.get("collision_rule_errors", 1) == 0
    if stage == 7 and stage5:
        return (report.get("shaping", 1) == 0 and
                completion >= stage5["completion"] - .02 and
                report.get("median_time") is not None and
                stage5.get("median_time") is not None and
                report["median_time"] <= stage5["median_time"] * 1.05)
    return False


def gait_pilot_passes(report):
    """A diagnostic launch gate, never a replacement for strict stage acceptance."""
    return (report.get("episodes",0)>=100 and report.get("diagnostic_episodes",0)>=100 and
            report.get("correct_cycle_ratio",0)>=.2 and report.get("fall_free",0)>=.85 and
            report.get("four_leg_participation",0)>=.9 and
            report.get("synchrony_cycle_ratio",1)<=.2 and report.get("speed_band_fraction",0)>=.8)
