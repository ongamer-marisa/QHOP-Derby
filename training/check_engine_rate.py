"""Verify wall-clock acceleration leaves the scripted 100 Hz physics unchanged."""
from pathlib import Path
import argparse
import time
import numpy as np
from mlagents_envs.environment import UnityEnvironment,ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path,default=Path('Builds/Training/QHopTraining.exe'))
    parser.add_argument('--decisions',type=int,default=100);args=parser.parse_args()
    traces=[]
    root=Path(__file__).resolve().parents[1]
    for rate in (1.,20.):
        channel=EngineConfigurationChannel();channel.set_configuration_parameters(time_scale=rate,target_frame_rate=-1)
        start=time.perf_counter()
        env=UnityEnvironment(str(args.executable),worker_id=980,seed=55111,no_graphics=True,
            side_channels=[channel],additional_args=["--qhop-stage=1","--qhop-task-index=0","--qhop-areas=1","-logFile",str(root/f"Builds/BuildLogs/engine-rate-{rate}.log")])
        try:
            env.reset();behavior=next(iter(env.behavior_specs));trace=[]
            for k in range(args.decisions):
                decision,terminal=env.get_steps(behavior)
                action=np.zeros((len(decision),28),dtype=np.float32);action[:,16:]=.7
                env.set_actions(behavior,ActionTuple(continuous=action));env.step()
                decision,terminal=env.get_steps(behavior);trace.append(decision.obs[0].copy())
            traces.append(np.concatenate(trace))
        finally:env.close()
        print({"time_scale":rate,"seconds":round(time.perf_counter()-start,2)},flush=True)
    difference=float(np.abs(traces[0]-traces[1]).max());print({"maximum_observation_difference":difference},flush=True)
    if difference>1e-5:raise AssertionError("Wall-clock engine rate changed scripted physics")
