using System;
using System.Reflection;
using QHop;
using UnityEngine;

// Run with Unity -batchmode -nographics -executeMethod QHopRewardProbe.Run.
public static class QHopRewardProbe
{
    static void Check(bool value,string message)
    {
        if(!value)throw new InvalidOperationException("QHOP_REWARD_PROBE_FAIL: "+message);
    }

    public static void Run()
    {
        var root=new GameObject("RewardProbe");
        try
        {
            var horse=root.AddComponent<HorseActor>();
            var body=root.AddComponent<Rigidbody>();
            body.useGravity=false;
            body.position=Vector3.up;
            typeof(HorseActor).GetField("<Torso>k__BackingField",
                BindingFlags.Instance|BindingFlags.NonPublic).SetValue(horse,body);
            typeof(HorseActor).GetField("restTorsoHeight",
                BindingFlags.Instance|BindingFlags.NonPublic).SetValue(horse,1f);

            var state=new HorseRaceResult();
            var once=new HorseTrainingReward(2);
            once.Step(horse,state,1f,1);
            var hundred=new HorseTrainingReward(2);
            for(int i=0;i<100;i++)hundred.Step(horse,state,.01f,1);
            Check(Mathf.Abs(once.Total-hundred.Total)<1e-5f,
                "time-integrated reward changed with logging frequency");

            var forward=new HorseTrainingReward(2);
            state.progressDelta=1f;
            forward.Step(horse,state,.05f,1);
            Check(forward.Total>once.Total*.05f,
                "legitimate forward progress did not improve reward");

            state.progressDelta=0f;
            var still=new HorseTrainingReward(1,1f,10f);
            for(int i=0;i<1000;i++)still.Step(horse,state,.01f,1,i*.01f);
            Check(Mathf.Abs(still.Total+.5f)<.001f,
                "Stage 1 standing still earned a positive reward");
            var moving=new HorseTrainingReward(1,1f,10f);
            state.progressDelta=2f;
            moving.Step(horse,state,.01f,1,.01f);
            Check(moving.Total>3.9f,"Stage 1 legal forward step was underrewarded");
            state.progressDelta=0f;state.finished=true;
            moving.Step(horse,state,.01f,1,1f);
            Check(moving.Total>still.Total+5f,
                "Stage 1 short legal finish did not beat standing timeout");
            state.finished=false;
            var early=new HorseTrainingReward(1,1f,10f);
            state.dnf=true;state.reason="boundary";
            early.Step(horse,state,.01f,1,.01f);
            Check(Mathf.Abs(early.WeightedTime+.5f)<.001f &&
                Mathf.Abs(early.WeightedDnf+1f)<.001f,
                "Stage 1 early DNF saved the remaining time cost");
            state.dnf=false;state.reason=null;

            var final=new HorseTrainingReward(6,0f);
            state.progressDelta=1f;
            state.falls=1;
            float first=final.Step(horse,state,.05f,4);
            Check(Mathf.Abs(first+.005f)<1e-5f && final.WeightedProgress==0f &&
                final.WeightedFall==0f,"zero shaping retained auxiliary reward");
            state.finished=true;state.rank=1;state.progressDelta=0f;
            float finish=final.Step(horse,state,.05f,4);
            float duplicate=final.Step(horse,state,.05f,4);
            Check(Mathf.Abs(finish-10.995f)<1e-4f &&
                Mathf.Abs(duplicate+.005f)<1e-5f,
                "finish bonus was missing or repeated");

            var dnf=new HorseTrainingReward(6,0f);
            var failed=new HorseRaceResult { dnf=true };
            float dnfFirst=dnf.Step(horse,failed,.05f,4);
            float dnfRepeat=dnf.Step(horse,failed,.05f,4);
            Check(Mathf.Abs(dnfFirst+10.005f)<1e-4f &&
                Mathf.Abs(dnfRepeat+.005f)<1e-5f,
                "DNF cost was missing or repeated");

            Debug.Log("QHOP_REWARD_PROBE_PASS: time integration, progress, zero shaping, one-shot finish/DNF");
        }
        finally{UnityEngine.Object.DestroyImmediate(root);}
    }
}
