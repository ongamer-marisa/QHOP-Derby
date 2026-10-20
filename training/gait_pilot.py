"""Bounded A/B gait repair trial; source checkpoints are read only."""
import argparse
import os
import json
from pathlib import Path
import subprocess
import sys
from qhop_train.control import atomic_json
from qhop_train.curriculum import gait_pilot_passes


def run_branch(source, output, seed, executable,device):
    existing=(output/"latest.pt").exists()
    command="resume" if existing else "adapt-gait"
    args=[sys.executable,"-m","qhop_train.cli","--executable",str(executable),
        "--log-file",str(output/"console.log"),command,"--checkpoint",str(output/"latest.pt" if existing else source),
        "--output",str(output),"--workers","4","--batch","16384","--seed",str(seed),"--max-updates","30","--device",device]
    if not existing:args.extend(["--eval-interval","10"])
    subprocess.run(args,check=True,env={**os.environ,"QHOP_TRAIN_TIME_SCALE":"20"})
    import torch
    state=torch.load(output/"latest.pt",map_location="cpu",weights_only=False)["training"]
    if state["global_updates"]<30:
        raise RuntimeError("Trial interrupted before its bounded endpoint; rerun with --resume")
    reports=[json.loads(line) for line in (output/"evaluations.jsonl").read_text(encoding="utf-8").splitlines()]
    last=reports[-1] if reports else {}
    return {"source":str(source),"output":str(output),"seed":seed,"report":last,
            "passed":gait_pilot_passes(last)}


def main():
    parser=argparse.ArgumentParser()
    root=Path(__file__).resolve().parents[1]
    parser.add_argument("--source-directory",type=Path,help="Legacy convenience: both sources in one directory")
    parser.add_argument("--stage1-source",type=Path,default=root/"Runs/editor-20261006-163124-branch-20261006-163650/stage1-approved.pt")
    parser.add_argument("--stage2-source",type=Path,default=root/"Runs/editor-20261008-093459-branch-20261008-093509/latest.pt")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--executable",type=Path,default=root/"Builds/Training/QHopTraining.exe")
    parser.add_argument("--device",choices=("auto","cpu","cuda"),default="auto")
    parser.add_argument("--resume",action="store_true",help="Continue only these existing bounded pilot branches")
    parser.add_argument("--seed",type=int,default=20260927)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=args.resume)
    branches=[]
    sources=(("from-stage1",args.stage1_source),("from-stage2",args.stage2_source))
    if args.source_directory:sources=(("from-stage1",args.source_directory/"stage1-approved.pt"),("from-stage2",args.source_directory/"latest.pt"))
    for name,source in sources:
        print("Running",name,flush=True)
        branch=run_branch(source,args.output/name,args.seed,args.executable,args.device)
        branches.append(branch);atomic_json(args.output/"pilot-results.json",{"branches":branches,"status":"testing"})
    qualified=[x for x in branches if x["passed"]]
    if not qualified:
        atomic_json(args.output/"pilot-results.json",{"branches":branches,"status":"stopped-needs-diagnosis",
            "reason":"Neither source passed all pilot gates; no further training budget spent."})
        return
    winner=max(qualified,key=lambda x:(x["report"]["correct_cycle_ratio"],x["report"]["fall_free"],-x["report"]["synchrony_cycle_ratio"]))
    replications=[]
    for offset in (1,2):
        replicate=run_branch(Path(winner["source"]),args.output/("replicate-"+str(offset)),args.seed+offset,args.executable,args.device)
        replications.append(replicate)
    passed=all(x["passed"] for x in replications)
    atomic_json(args.output/"pilot-results.json",{"branches":branches,"winner":winner,"replications":replications,
        "status":"qualified-for-full-training" if passed else "stopped-seed-instability"})
    if passed:
        print("All pilot gates passed; continuing the original winning branch within its 8M stage budget",flush=True)
        subprocess.run([sys.executable,"-m","qhop_train.cli","--executable",str(args.executable),
            "--log-file",str(Path(winner["output"])/"console.log"),"resume","--checkpoint",
            str(Path(winner["output"])/"latest.pt"),"--output",winner["output"],"--workers","4",
            "--batch","16384","--seed",str(winner["seed"]),"--device",args.device],check=True,env={**os.environ,"QHOP_TRAIN_TIME_SCALE":"20"})

if __name__=="__main__":
    main()
