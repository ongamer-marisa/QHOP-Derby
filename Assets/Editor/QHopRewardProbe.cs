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

    static void Feed(HorseGaitTracker gait,bool correct,float speed)
    {
        bool[] contact={true,true,true,true};
        var heights=new float[4];var positions=new float[4];
        for(int k=0;k<3;k++)gait.StepContactSample(contact,heights,1f,speed,true,.01f,0,0f,null,positions);
        int[][] groups=correct?new[]{new[]{0,3},new[]{1,2}}:
            new[]{new[]{2,3},new[]{0,1}};
        for(int cycle=0;cycle<4;cycle++)foreach(var group in groups)
        {
            foreach(int foot in group){contact[foot]=false;heights[foot]=.1f;}
            for(int k=0;k<8;k++){foreach(int foot in group)positions[foot]=k*.1f;gait.StepContactSample(contact,heights,1f,speed,true,.01f,0,0f,null,positions);}
            foreach(int foot in group){contact[foot]=true;heights[foot]=0f;}
            for(int k=0;k<32;k++)gait.StepContactSample(contact,heights,1f,speed,true,.01f,0,0f,null,positions);
        }
    }

    // Every sample is followed by reward settlement, as in the 100 Hz runtime.
    // Both sequences cover the same time and distance; only the hoof pairs differ.
    static HorseTrainingReward Window(HorseActor horse, bool correct, float speed, float period=.8f, int stage=2)
    {
        var reward=new HorseTrainingReward(stage);
        var state=new HorseRaceResult {progressDelta=speed*.01f};
        var contact=new bool[4];var heights=new float[4];var positions=new float[4];
        horse.Torso.linearVelocity=Vector3.forward*speed;
        for(int tick=0;tick<2000;tick++)
        {
            float time=tick*.01f;
            for(int foot=0;foot<4;foot++)
            {
                int group=correct?(foot==0 || foot==3?0:1):(foot<2?0:1);
                float phase=Mathf.Repeat(time-group*period*.5f,period);
                bool swing=phase>=period-.12f;
                contact[foot]=!swing;heights[foot]=swing?.1f:0f;
                positions[foot]=swing?(phase-(period-.12f))*2f:0f;
            }
            reward.Gait.StepContactSample(contact,heights,1f,speed,true,.01f,0,0f,null,positions);
            if(tick==1999)state.finished=true;
            reward.Step(horse,state,.01f,1,(tick+1)*.01f);
        }
        return reward;
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
            body.linearVelocity=Vector3.forward*3f;
            forward.Step(horse,state,.05f,1);
            Check(forward.Total>once.Total*.05f,
                "legitimate forward progress did not improve reward");
            var correctGait=new HorseTrainingReward(2);
            var wrongGait=new HorseTrainingReward(2);
            body.linearVelocity=new Vector3(0f,0f,3f);
            Feed(correctGait.Gait,true,3f);
            Feed(wrongGait.Gait,false,3f);
            state.progressDelta=.1f;
            correctGait.Step(horse,state,.05f,1);
            wrongGait.Step(horse,state,.05f,1);
            Check(correctGait.Total>wrongGait.Total+.02f,
                "correct gait did not outscore synchronized hoof slaps");
            var stationaryGait=new HorseTrainingReward(2);
            Feed(stationaryGait.Gait,true,0f);
            state.progressDelta=0f;
            stationaryGait.Step(horse,state,.05f,1);
            Check(stationaryGait.WeightedGait==0f &&
                  stationaryGait.WeightedProgress==0f,
                  "stationary footfalls earned positive gait reward");

            body.linearVelocity=Vector3.zero;
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

            var final=new HorseTrainingReward(7,0f);
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

            var dnf=new HorseTrainingReward(7,0f);
            var failed=new HorseRaceResult { dnf=true };
            float dnfFirst=dnf.Step(horse,failed,.05f,4);
            float dnfRepeat=dnf.Step(horse,failed,.05f,4);
            Check(Mathf.Abs(dnfFirst+10.005f)<1e-4f &&
                Mathf.Abs(dnfRepeat+.005f)<1e-5f,
                "DNF cost was missing or repeated");

            // Stage 1 gives exactly 10% of the stage-2 gait contribution.
            var low=new HorseTrainingReward(1);Feed(low.Gait,true,3f);
            body.linearVelocity=Vector3.forward*3f;state=new HorseRaceResult{progressDelta=.03f};
            low.Step(horse,state,.01f,1);
            var full=new HorseTrainingReward(2);Feed(full.Gait,true,3f);full.Step(horse,state,.01f,1);
            Check(low.WeightedGait>0f && Mathf.Abs(low.WeightedGait-.1f*full.WeightedGait)<1e-6f,"Stage 1 low gait weight");
            var failure=new HorseTrainingReward(2);
            failure.Step(horse,new HorseRaceResult{dnf=true,reason="boundary"},.01f,1,1f);
            Check(failure.Total < -58f,"early DNF avoided remaining penalty budget");
            var legalWindow=Window(horse,true,3f);
            var syncWindow=Window(horse,false,3f);
            var frequentSync=Window(horse,false,3f,.4f);
            var stationaryWindow=Window(horse,true,0f);
            Check(legalWindow.Gait.CorrectRatio>.9f,"legal full-window sequence was rejected");
            Check(syncWindow.Gait.CorrectRatio==0f && syncWindow.Gait.SynchronyRatio==1f,
                "synchronized wrong cycles escaped separate classification");
            Check(legalWindow.Total>syncWindow.Total,"equal-progress correct gait did not win");
            Check(syncWindow.Total<=-10f,"20 seconds of synchronized wrong gait earned "+syncWindow.Total);
            Check(stationaryWindow.WeightedGait==0f && stationaryWindow.WeightedProgress==0f,
                "stationary full window earned positive reward");
            Check(syncWindow.WeightedProcess==0f && frequentSync.WeightedProcess==0f &&
                frequentSync.WeightedCoordination==0f && frequentSync.WeightedCycle==0f &&
                frequentSync.WeightedGait<=2.001f,
                "wrong contact frequency farmed process/coordination/cycle or exceeded participation's time budget");
            Check(legalWindow.WeightedProcess<=4.001f && legalWindow.WeightedCycle<=15.001f,
                "process or cycle payout exceeded its time budget");
            Debug.Log("QHOP_REWARD_WINDOW correct="+legalWindow.Total+" wrong="+syncWindow.Total+
                " faster_wrong="+frequentSync.Total+" process="+legalWindow.WeightedProcess+
                " cycle="+legalWindow.WeightedCycle);
            Debug.Log("QHOP_REWARD_PROBE_PASS: time integration, progress, zero shaping, one-shot finish/DNF");
        }
        finally{UnityEngine.Object.DestroyImmediate(root);}
    }
}
