"""Short real-Unity communication check before starting a long PPO run."""
import argparse
import multiprocessing
from pathlib import Path

import numpy as np

from qhop_train.schema import ACTION, FRAME, HISTORY, OBSERVATION
from qhop_train.unity_env import UnityWorkers


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--executable",type=Path,default=
        Path(__file__).resolve().parents[1]/"Builds/Training/QHopTraining.exe")
    parser.add_argument("--decisions",type=int,default=40)
    parser.add_argument("--stage",type=int,choices=range(1,8),default=1)
    parser.add_argument("--task-index",type=int,choices=range(3),default=0,
                        help="Stage 2: 0 trot, 1 canter, 2 gallop")
    parser.add_argument("--areas",type=int,default=1)
    parser.add_argument("--difficulty",type=float,default=0.0)
    parser.add_argument("--shaping",type=float,default=1.0)
    parser.add_argument("--diagnose",action="store_true")
    parser.add_argument("--front-angle",type=float,default=0.0)
    parser.add_argument("--rear-angle",type=float,default=0.0)
    parser.add_argument("--drive",type=float,default=0.7)
    parser.add_argument("--expect-terminals",type=int,default=0,
                        help="Minimum combined learner and frozen terminals")
    args=parser.parse_args()
    if not args.executable.is_file():
        raise FileNotFoundError(args.executable)
    transitions=0
    terminals=0
    frozen_transitions=0
    frozen_terminals=0
    minimum_height=10.0
    final_height=0.0
    maximum_pitch=0.0
    fall_events=[]
    last_fallen=False
    first_recovery=None
    with UnityWorkers(args.executable,1,args.stage,args.difficulty,args.shaping,20260928,args.areas,
                      worker_id_base=900,task_index=args.task_index) as workers:
        for index in range(args.decisions):
            state=workers.current[0]["learner"]
            if len(state["decision_obs"]):
                frame=state["decision_obs"][0,(HISTORY-1)*FRAME:HISTORY*FRAME]
                final_height=float(frame[202])
                minimum_height=min(minimum_height,final_height)
                maximum_pitch=max(maximum_pitch,abs(float(frame[8])))
                fallen=frame[206] > .5
                if fallen and not last_fallen:
                    fall_events.append(index)
                if last_fallen and not fallen and first_recovery is None:
                    first_recovery=index
                last_fallen=fallen
            if args.diagnose and index in (0,1,5,10,20,40,80,160,320) and \
                    len(state["decision_obs"]):
                frame=state["decision_obs"][0,(HISTORY-1)*FRAME:HISTORY*FRAME]
                print({"decision":index,"height_ratio":round(float(frame[202]),3),
                       "gravity_local":[round(float(value),3) for value in frame[6:9]],
                       "velocity_local":[round(float(value*15),2) for value in frame[:3]],
                       "contacts":[int(frame[174+i*4]) for i in range(4)],
                       "unstable":int(frame[205]),"fallen":int(frame[206]),
                       "recovering":int(frame[207]),"torso_grounded":int(frame[208]),
                       "falls":round(float(frame[212]*10),1)},flush=True)
            count=len(state["decision_id"])
            actions=np.zeros((count,ACTION),dtype=np.float32)
            actions[:,[0,4]]=args.front_angle
            actions[:,[8,12]]=args.rear_angle
            actions[:,16:]=args.drive
            frozen=np.zeros((len(workers.current[0]["frozen"]["decision_id"]),ACTION),
                            dtype=np.float32)
            frozen[:,16:]=args.drive
            result=workers.step([{"learner":actions,"frozen":frozen}])[0]
            for role in ("learner","frozen"):
                following=result[role]
                for kind in ("decision_obs","terminal_obs"):
                    observation=following[kind]
                    if observation.shape[1:]!=(OBSERVATION,) or not np.isfinite(observation).all():
                        raise AssertionError(f"Invalid Unity {role} {kind}: {observation.shape}")
                for kind in ("decision_reward","terminal_reward"):
                    if not np.isfinite(following[kind]).all():
                        raise AssertionError(f"Non-finite Unity {role} {kind}")
            transitions+=count
            terminals+=len(result["learner"]["terminal_id"])
            frozen_transitions+=len(frozen)
            frozen_terminals+=len(result["frozen"]["terminal_id"])
    if transitions==0:
        raise AssertionError("Unity provided no trainable decisions")
    if terminals+frozen_terminals<args.expect_terminals:
        raise AssertionError(f"Only {terminals+frozen_terminals} terminals, expected "
                             f"at least {args.expect_terminals}")
    print({"transitions":transitions,"terminals":terminals,
           "frozen_transitions":frozen_transitions,"frozen_terminals":frozen_terminals,
           "observation":OBSERVATION,"action":ACTION,
           "stage":args.stage,"task_index":args.task_index,"areas":args.areas,
           "difficulty":args.difficulty,"shaping":args.shaping,
           "front_angle":args.front_angle,"rear_angle":args.rear_angle,
           "drive":args.drive,"minimum_height":round(minimum_height,3),
           "final_height":round(final_height,3),
           "maximum_pitch":round(maximum_pitch,3),
           "fall_decisions":fall_events[:5],"first_recovery":first_recovery})


if __name__=="__main__":
    multiprocessing.freeze_support()
    main()
