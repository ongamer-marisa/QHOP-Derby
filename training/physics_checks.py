"""Real Unity Player action-channel and repeated-reset checks."""
import argparse
import multiprocessing
from pathlib import Path
import time

import numpy as np

from qhop_train.schema import ACTION, FRAME, HISTORY, OBSERVATION
from qhop_train.unity_env import UnityWorkers


def standing(count):
    actions=np.zeros((count,ACTION),dtype=np.float32)
    actions[:,16:]=.7
    return actions


def validate(state):
    for role in ("learner","frozen"):
        for field in ("decision_obs","terminal_obs"):
            observations=state[role][field]
            if observations.shape[1:]!=(OBSERVATION,) or not np.isfinite(observations).all():
                raise AssertionError(f"Invalid {role} {field}: {observations.shape}")
        for field in ("decision_reward","terminal_reward"):
            if not np.isfinite(state[role][field]).all():
                raise AssertionError(f"Non-finite {role} {field}")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--executable",type=Path,default=Path(__file__).resolve().parents[1]/
                        "Builds/Training/QHopTraining.exe")
    parser.add_argument("--areas",type=int,default=8)
    parser.add_argument("--episodes",type=int,default=100)
    parser.add_argument("--diagnose",action="store_true")
    args=parser.parse_args()
    started=time.perf_counter()
    terminals=0
    falls=0
    decision_in_episode=0
    trace=[]
    with UnityWorkers(args.executable,1,1,0.0,1.0,20261003,args.areas,
                      worker_id_base=960) as workers:
        for channel in range(ACTION):
            current=workers.current[0]
            actions=standing(len(current["learner"]["decision_id"]))
            expected=.1 if channel<16 else .6
            actions[:,channel]=expected
            state=workers.step([{"learner":actions,"frozen":standing(0)}])[0]
            validate(state)
            if len(state["learner"]["terminal_id"]):
                raise AssertionError("Early terminal during action-channel probe")
            frame=state["learner"]["decision_obs"][:,(HISTORY-1)*FRAME:HISTORY*FRAME]
            if not np.allclose(frame[:,215+channel],expected,atol=1e-6):
                raise AssertionError(f"Action channel {channel} was not applied")
        while terminals < args.episodes:
            current=workers.current[0]
            if args.diagnose and len(current["learner"]["decision_obs"]) and \
                    decision_in_episode in (0,1,2,5,10,20,40,80,160):
                sample=current["learner"]["decision_obs"][0,
                    (HISTORY-1)*FRAME:HISTORY*FRAME]
                trace.append((decision_in_episode,round(float(sample[202]),3),
                              round(float(sample[2]*15),2),
                              [int(sample[174+i*4]) for i in range(4)]))
            decision_in_episode+=1
            state=workers.step([{"learner":standing(len(current["learner"]["decision_id"])),
                                 "frozen":standing(0)}])[0]
            validate(state)
            final=state["learner"]["terminal_obs"]
            terminals+=len(final)
            if len(final):
                frame=final[:,(HISTORY-1)*FRAME:HISTORY*FRAME]
                falls+=int(np.rint(frame[:,212]*10).sum())
                if np.any(frame[:,210]>.5):
                    raise AssertionError("Standing controller unexpectedly completed 20 m")
                fresh=state["learner"]["decision_obs"]
                first=fresh[0,(HISTORY-1)*FRAME:HISTORY*FRAME] if len(fresh) else None
                print({"reset_episodes":terminals,"falls":falls,
                       "new_height":round(float(first[202]),3) if first is not None else None,
                       "new_vertical_speed":round(float(first[1]*15),3) if first is not None else None,
                       "new_contacts":[int(first[174+i*4]) for i in range(4)]
                           if first is not None else None,
                       "trajectory":trace if args.diagnose else None},flush=True)
                trace=[]
                decision_in_episode=0
                if falls and not args.diagnose:
                    raise AssertionError(f"Standing reset stability failed: {falls} falls")
    print({"checked_action_channels":ACTION,"completed_reset_episodes":terminals,
           "falls":falls,"wall_seconds":round(time.perf_counter()-started,1)})


if __name__=="__main__":
    multiprocessing.freeze_support()
    main()
