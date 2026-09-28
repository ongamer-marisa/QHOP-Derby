using System;
using System.Collections.Generic;
using System.Reflection;
using QHop;
using UnityEngine;

// Run with Unity -batchmode -nographics -executeMethod QHopRaceProbe.Run.
// Uses simple rigidbodies so finish rules can be checked without a trained gait.
public static class QHopRaceProbe
{
    static readonly FieldInfo TorsoField=typeof(HorseActor).GetField(
        "<Torso>k__BackingField",BindingFlags.Instance|BindingFlags.NonPublic);
    static readonly FieldInfo ConditionField=typeof(HorseActor).GetField(
        "<Condition>k__BackingField",BindingFlags.Instance|BindingFlags.NonPublic);

    static void Check(bool value,string message)
    {
        if(!value)throw new InvalidOperationException("QHOP_RACE_PROBE_FAIL: "+message);
    }

    static HorseActor Horse(string name,float z)
    {
        var root=new GameObject(name);
        var horse=root.AddComponent<HorseActor>();
        var torso=new GameObject("Torso");
        torso.transform.SetParent(root.transform);
        var body=torso.AddComponent<Rigidbody>();
        body.useGravity=false;
        torso.AddComponent<BoxCollider>();
        horse.bodies.Add(body);
        TorsoField.SetValue(horse,body);
        Move(horse,z);
        return horse;
    }

    static void Move(HorseActor horse,float z)
    {
        horse.Torso.position=new Vector3(0f,1f,z);
        horse.Torso.linearVelocity=Vector3.forward*2f;
        horse.Torso.WakeUp();
    }

    static void Destroy(IReadOnlyList<HorseActor> horses)
    {
        foreach(var horse in horses)UnityEngine.Object.DestroyImmediate(horse.gameObject);
    }

    public static void Run()
    {
        Check(TorsoField!=null && ConditionField!=null,"HorseActor fields unavailable");
        var horses=new List<HorseActor>{Horse("First",9.9f),Horse("Second",9.9f)};
        try
        {
            var race=new HorseRaceSession(horses,10f,5f);
            Move(horses[0],10.1f);
            race.Step(.01f);
            Check(race.Results[0].finished && race.Results[0].rank==1,
                "first horse did not finish");
            Check(Mathf.Abs(race.Results[0].finishTime-.005f)<1e-4f,
                "finish time was not interpolated");
            Check(!race.Complete && !race.Results[1].dnf,"first finish stopped the race");
            Check(!horses[0].Torso.GetComponent<Collider>().enabled,
                "finished horse still has collision");
            Move(horses[1],10.1f);
            race.Step(.01f);
            Check(race.Complete && race.Results[1].rank==2,
                "second horse was not ranked independently");
        }
        finally{Destroy(horses);}

        horses=new List<HorseActor>{Horse("NoCatchup",0f)};
        try
        {
            var race=new HorseRaceSession(horses,20f,5f);
            horses[0].Torso.position=new Vector3(7f,1f,4f);
            race.Step(.01f);
            Check(race.Results[0].progressDelta==0f &&
                race.Results[0].filteredBoundary>3.9f,
                "off-track displacement earned progress or was not counted");
            Move(horses[0],4.1f);
            race.Step(.01f);
            Check(race.Results[0].progressDelta<=.101f,
                "returning on track paid a backlog of invalid displacement");
            Move(horses[0],3f);
            race.Step(.01f);
            Move(horses[0],4.1f);
            race.Step(.01f);
            Check(race.Results[0].progressDelta==0f,
                "backtracking and replaying progress paid twice");
        }
        finally{Destroy(horses);}

        horses=new List<HorseActor>{Horse("TieA",9.9f),Horse("TieB",9.8f)};
        try
        {
            var race=new HorseRaceSession(horses,10f,5f);
            Move(horses[0],10.1f);
            Move(horses[1],10.2f);
            race.Step(.01f);
            Check(race.Complete && race.Results[0].rank==1 && race.Results[1].rank==2,
                "same-step tie was not deterministically ranked");
            Check(Mathf.Abs(race.Results[0].finishTime-race.Results[1].finishTime)<1e-4f,
                "same-step finish interpolation differs");
        }
        finally{Destroy(horses);}

        horses=new List<HorseActor>{Horse("Penalty",0f)};
        try
        {
            var race=new HorseRaceSession(horses,10f,5f);
            ConditionField.SetValue(horses[0],HorseCondition.Fallen);
            Move(horses[0],5f);
            race.Step(.01f);
            Check(race.Results[0].progressDelta==0f && !race.Results[0].finished,
                "fallen displacement earned progress");
            ConditionField.SetValue(horses[0],HorseCondition.Standing);
            Move(horses[0],0f);
            race.Step(.01f);
            Move(horses[0],5f);
            race.Step(.01f);
            Check(race.Results[0].progressDelta==0f,
                "repeated progress after recovery earned a second reward");
            Move(horses[0],10.1f);
            ConditionField.SetValue(horses[0],HorseCondition.Fallen);
            race.Step(.01f);
            Check(!race.Results[0].finished,"fallen horse crossed finish");
            ConditionField.SetValue(horses[0],HorseCondition.Standing);
            Move(horses[0],9.9f);
            race.Step(.01f);
            Move(horses[0],10.1f);
            race.Step(.01f);
            Check(race.Results[0].finished,"recovered horse could not finish legally");
        }
        finally{Destroy(horses);}

        horses=new List<HorseActor>{Horse("Timeout",0f)};
        try
        {
            var race=new HorseRaceSession(horses,10f,.02f);
            race.Step(.01f);
            Check(!race.Complete,"race ended before timeout");
            race.Step(.01f);
            Check(race.Complete && race.Results[0].dnf &&
                race.Results[0].reason=="timeout" && race.Results[0].rank==0,
                "timeout was not DNF without rank");
        }
        finally{Destroy(horses);}

        horses=new List<HorseActor>{Horse("Boundary",0f),Horse("Ground",0f)};
        try
        {
            var race=new HorseRaceSession(horses,10f,5f);
            horses[0].Torso.position=new Vector3(9f,1f,0f);
            horses[1].Torso.position=new Vector3(0f,-.6f,0f);
            race.Step(.01f);
            Check(race.Complete && race.Results[0].dnf &&
                race.Results[0].reason=="boundary" && race.Results[0].rank==0,
                "boundary violation was not DNF");
            Check(race.Results[1].dnf && race.Results[1].reason=="ground" &&
                race.Results[1].rank==0,"lost ground was not DNF");
        }
        finally{Destroy(horses);}

        Debug.Log("QHOP_RACE_PROBE_PASS: finish, interpolation, tie, recovery, timeout, boundary, ground");
    }
}
