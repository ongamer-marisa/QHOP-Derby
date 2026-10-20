using System;
using QHop;
using UnityEditor;
using UnityEngine;

public static class QHopGaitProbe
{
    static void Check(bool value,string message)
    { if(!value)throw new InvalidOperationException("QHOP_GAIT_PROBE_FAIL: "+message); }

    static void Tick(HorseGaitTracker tracker,bool[] feet,float[] heights,float speed,int task,int frames)
    {
        var positions=new float[4];
        for(int i=0;i<frames;i++)
        {for(int foot=0;foot<4;foot++)positions[foot]=feet[foot]?0f:i*.1f;
         tracker.StepContactSample(feet,heights,1f,speed,true,.01f,task,0f,null,positions);}
    }

    static HorseGaitTracker Pattern(int task,int[][] groups,float[] gaps,float speed=4f,
        float clearance=.1f)
    {
        var tracker=new HorseGaitTracker();
        bool[] feet={true,true,true,true};
        var heights=new float[4];
        Tick(tracker,feet,heights,speed,task,3);
        for(int cycle=0;cycle<4;cycle++)
        for(int group=0;group<groups.Length;group++)
        {
            foreach(int foot in groups[group]){feet[foot]=false;heights[foot]=clearance;}
            Tick(tracker,feet,heights,speed,task,4);
            foreach(int foot in groups[group]){feet[foot]=true;heights[foot]=0f;}
            Tick(tracker,feet,heights,speed,task,2);
            Tick(tracker,feet,heights,speed,task,
                Mathf.RoundToInt(gaps[group]*100f));
        }
        return tracker;
    }

    static void CheckRewardBudget(float gap)
    {
        var tracker=new HorseGaitTracker();bool[] feet={true,true,true,true};var h=new float[4];var z=new float[4];
        float elapsed=0f,cycleReward=0f,processReward=0f;
        Action tick=()=>{tracker.StepContactSample(feet,h,1f,3f,true,.01f,0,0f,null,z);
            elapsed+=.01f;cycleReward+=tracker.Cycle*.01f;processReward+=tracker.Process*.01f;};
        for(int k=0;k<4;k++)tick();
        for(int c=0;c<20;c++)foreach(var pair in new[]{new[]{0,3},new[]{1,2}})
        {
            foreach(int f in pair){feet[f]=false;h[f]=.1f;}
            for(int k=0;k<8;k++){foreach(int f in pair)z[f]=k*.1f;tick();}
            foreach(int f in pair){feet[f]=true;h[f]=0f;}
            for(int k=0;k<Mathf.RoundToInt(gap*100);k++)tick();
        }
        Check(cycleReward<=.75f*elapsed+.001f,"touchdown frequency exceeded cycle reward budget");
        Check(processReward<=.20f*elapsed+.001f,"prefix frequency exceeded process reward budget");
    }

    [MenuItem("QHOP/Probes/Gait")]
    public static void Run()
    {
        var positionProbe=new HorseGaitTracker();
        positionProbe.StepContactSample(new[]{true,true,true,true},new float[4],1f,0f,true,.01f,-2,0f,null,new[]{-.2f,.2f,-.1f,.1f});
        var features=positionProbe.Observation(-2,0f);
        Check(features.Length==28 && features[20]<0f && features[21]>0f,"signed hoof-relative position observations");
        CheckRewardBudget(.12f);CheckRewardBudget(.32f);CheckRewardBudget(.6f);
        var walk=Pattern(-2,new[]{new[]{2},new[]{0},new[]{3},new[]{1}},new[]{.15f,.15f,.15f,.15f},1.5f);
        Check(walk.CorrectRatio>.75f,"natural four-beat walk");
        for(int shift=0;shift<4;shift++)
        {
            foreach(var pattern in new[]{new[]{3,2,1,0},new[]{2,3,0,1}})
            {
                var rotated=new int[4][];
                for(int k=0;k<4;k++)rotated[k]=new[]{pattern[(k+shift)%4]};
                Check(Pattern(2,rotated,new[]{.1f,.1f,.1f,.1f},8f).CorrectRatio>.75f,"cyclic gallop "+shift);
            }
            var walking=new[]{2,0,3,1};var walkGroups=new int[4][];
            for(int k=0;k<4;k++)walkGroups[k]=new[]{walking[(k+shift)%4]};
            Check(Pattern(-2,walkGroups,new[]{.15f,.15f,.15f,.15f},1.5f).CorrectRatio>.75f,"cyclic walk "+shift);
        }
        // Within a diagonal pair either hoof may land first, with a 20 ms gap.
        foreach(var order in new[]{new[]{0,3,1,2},new[]{0,3,2,1},new[]{3,0,1,2},new[]{3,0,2,1},new[]{1,2,0,3},new[]{1,2,3,0},new[]{2,1,0,3},new[]{2,1,3,0}})
        {
            var probe=new HorseGaitTracker();bool[] feet={true,true,true,true};var h=new float[4];var z=new float[4];
            for(int f=0;f<4;f++)probe.StepContactSample(feet,h,1f,3f,true,.01f,0,0f,null,z);
            for(int c=0;c<5;c++)for(int pair=0;pair<2;pair++)
            {
                int a=order[pair*2],b=order[pair*2+1];feet[a]=feet[b]=false;
                for(int t=0;t<8;t++){h[a]=h[b]=.1f;z[a]=z[b]=t*.1f;probe.StepContactSample(feet,h,1f,3f,true,.01f,0,0f,null,z);}
                feet[a]=true;for(int t=0;t<2;t++)probe.StepContactSample(feet,h,1f,3f,true,.01f,0,0f,null,z);
                feet[b]=true;for(int t=0;t<25;t++)probe.StepContactSample(feet,h,1f,3f,true,.01f,0,0f,null,z);
            }
            Check(probe.CorrectRatio>.75f,"legal diagonal pair permutation "+string.Join(",",order));
        }
        for(int shift=0;shift<3;shift++)foreach(var lead in new[]{new[]{new[]{3},new[]{2,1},new[]{0}},new[]{new[]{2},new[]{3,0},new[]{1}}})
        {
            var groups=new int[3][];for(int k=0;k<3;k++)groups[k]=lead[(k+shift)%3];
            Check(Pattern(1,groups,new[]{.15f,.15f,.15f}).CorrectRatio>.75f,"cyclic canter "+shift);
        }
        var trot=Pattern(0,new[]{new[]{0,3},new[]{1,2}},new[]{.30f,.30f});
        Check(trot.Cycles>=3 && trot.CorrectRatio>.75f,"diagonal trot");
        var leftCanter=Pattern(1,new[]{new[]{3},new[]{2,1},new[]{0}},
            new[]{.15f,.15f,.15f});
        Check(leftCanter.Cycles>=3 && leftCanter.CorrectRatio>.75f,"left canter");
        var rightCanter=Pattern(1,new[]{new[]{2},new[]{3,0},new[]{1}},
            new[]{.15f,.15f,.15f});
        Check(rightCanter.CorrectRatio>.75f,"right canter");
        var leftGallop=Pattern(2,new[]{new[]{3},new[]{2},new[]{1},new[]{0}},
            new[]{.10f,.10f,.10f,.10f},8f);
        Check(leftGallop.CorrectRatio>.75f,"left gallop");
        var rightGallop=Pattern(2,new[]{new[]{2},new[]{3},new[]{0},new[]{1}},
            new[]{.10f,.10f,.10f,.10f},8f);
        Check(rightGallop.CorrectRatio>.75f,"right gallop");
        var hindSlap=Pattern(2,new[]{new[]{2,3},new[]{0,1}},new[]{.30f,.30f},8f);
        Check(hindSlap.CorrectRatio==0f && hindSlap.BadCycles>0,"synchronized hind slap");
        Check(hindSlap.SynchronyCycles>0,"synchrony must be reported separately");
        var stale=Pattern(0,new[]{new[]{0,3},new[]{1,2}},new[]{.30f,.30f});
        Tick(stale,new[]{true,true,true,true},new float[4],4f,0,300);
        Check(stale.Quality==0f && stale.Cycle==0f && stale.Coordination==0f,"stale cycle reward");
        var high=Pattern(0,new[]{new[]{0,3},new[]{1,2}},
            new[]{.30f,.30f},4f,.8f);
        Check(high.HighFraction>.01f,"excessive hoof height");
        var still=new HorseGaitTracker();
        Tick(still,new[]{true,true,true,true},new float[4],0f,0,500);
        Check(still.Cycles==0 && still.Quality==0f && still.Process==0f,
            "standing cannot earn gait reward");
        var drag=Pattern(0,new[]{new[]{0,3},new[]{1,2}},
            new[]{.30f,.30f},4f,.01f);
        Check(drag.CorrectCycles==0,"dragged hooves cannot form a valid cycle");
        var chatter=new HorseGaitTracker();
        bool[] contacts={true,true,true,true};
        Tick(chatter,contacts,new float[4],4f,0,3);
        for(int i=0;i<100;i++)
        {
            contacts[0]=false;
            Tick(chatter,contacts,new[]{.1f,0f,0f,0f},4f,0,1);
            contacts[0]=true;
            Tick(chatter,contacts,new float[4],4f,0,1);
        }
        Check(chatter.Cycles==0,"single-frame contact chatter was counted");
        trot.StepContactSample(new[]{true,true,true,true},new float[4],
            1f,0f,false,.01f,0);
        Check(trot.Quality==0f && trot.Process==0f && trot.Cycle==0f,
            "fall did not clear active cycle quality");
        Debug.Log("QHOP_GAIT_PROBE_PASS");
    }
}
