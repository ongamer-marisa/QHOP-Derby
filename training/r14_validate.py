"""Bounded real Player standing acceptance; no policy training."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
from qhop_train.unity_env import UnityWorkers
from qhop_train.schema import FRAME,HISTORY,OBSERVATION


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--executable',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=100);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--morph',action='store_true')
    args=parser.parse_args()
    if args.morph:os.environ['QHOP_VALIDATE_MORPH']='1'
    count=0;falls=0;minheight=1.;maxtilt=0.;maxdrift=0.;traces=[];cases=set();failed_cases=set()
    with UnityWorkers(args.executable,1,1,0,1,20261013,8,worker_id_base=970,task_index=3,
                      log_dir=args.output.parent/'standing-player') as workers:
        while len(cases)<81 if args.morph else count<args.episodes:
            state=workers.current[0]['learner'];x=state['decision_obs'];f=x[:,(HISTORY-1)*FRAME:HISTORY*FRAME]
            if x.shape[1:]!=(OBSERVATION,) or not np.isfinite(x).all():raise AssertionError('Non-finite or malformed observation')
            if len(f):
                age=20-f[:,203]*90;stable=age>=.5
                if stable.any():
                    z=f[stable];minheight=min(minheight,float(z[:,202].min()))
                    maxtilt=max(maxtilt,float(np.degrees(np.arccos(np.clip(-z[:,7],-1,1))).max()))
                    maxdrift=max(maxdrift,float(np.sqrt((z[:,198]*100)**2+(z[:,199]*6)**2).max()))
                if len(traces)<600:
                    traces.append({'age':float(age[0]),'height':float(f[0,202]),'tilt':float(np.degrees(np.arccos(np.clip(-f[0,7],-1,1)))),
                        'position':[float(f[0,199]*6),float(f[0,198]*100)],'velocity':f[0,:6].tolist(),
                        'contacts':f[0,174:190:4].tolist(),'condition':f[0,205:209].tolist()})
            action=np.zeros((len(x),28),np.float32);action[:,16:]=.7
            following=workers.step([{'learner':action,'frozen':np.empty((0,28),np.float32)}])[0]
            for name in ('decision_obs','terminal_obs','decision_reward','terminal_reward'):
                if not np.isfinite(following['learner'][name]).all():raise AssertionError('Non-finite '+name)
            for event in following.get('episode_logs',[]):
                count+=1;falls+=event['falls']
                if args.morph:
                    cases.add(event['morphValidationCase'])
                    if event['falls']:failed_cases.add(event['morphValidationCase'])
                if event['falls']:print('FALL',json.dumps(event),flush=True)
            if (args.morph and len(cases)>=81) or (not args.morph and count>=args.episodes):break
            if count>1000:raise AssertionError('Morph coverage did not complete')
    result={'episodes':count,'falls':falls,'min_height':minheight,'max_tilt':maxtilt,'max_drift':maxdrift,
            'morph_cases':sorted(cases),'failed_morph_cases':sorted(failed_cases),
            'pass':falls==0 and (args.morph or minheight>=.8 and maxtilt<=20 and maxdrift<=.5),'trace':traces}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='trace'}),flush=True)
    if not result['pass']:raise AssertionError('Standing acceptance failed; inspect '+str(args.output))


if __name__=='__main__':main()
