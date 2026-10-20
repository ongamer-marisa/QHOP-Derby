using System;
using System.Collections.Generic;
using UnityEngine;

namespace QHop
{
    public sealed class HorseRaceResult
    {
        public bool distanceReached;
        public bool finished;
        public bool dnf;
        public float finishTime;
        public int rank;
        public string reason;
        public float highWater;
        public float progressDelta;
        public float filteredPosture,filteredGround,filteredInactive,filteredBoundary;
        public float activeTime,stationaryTime;
        public int falls;
        internal float previousZ;
        internal HorseCondition previousCondition;
    }

    // Runs after each physics step and is shared by presentation and training.
    public sealed class HorseRaceSession
    {
        readonly IReadOnlyList<HorseActor> horses;
        readonly HorseRaceResult[] results;
        readonly float distance, timeout;
        readonly bool fixedWindow;
        readonly bool fallTerminatesEpisode;
        int nextRank=1;
        public float Elapsed { get; private set; }
        public IReadOnlyList<HorseRaceResult> Results => results;
        public bool Complete { get; private set; }

        public HorseRaceSession(IReadOnlyList<HorseActor> horses, float distance, float timeout,bool fixedWindow=false,bool fallTerminatesEpisode=false)
        {
            this.horses=horses;
            this.distance=distance;
            this.timeout=timeout;this.fixedWindow=fixedWindow;
            this.fallTerminatesEpisode=fallTerminatesEpisode;
            results=new HorseRaceResult[horses.Count];
            for(int i=0;i<horses.Count;i++)
                results[i]=new HorseRaceResult { previousZ=horses[i].Torso.position.z,
                    previousCondition=horses[i].Condition,
                    highWater=horses[i].Torso.position.z };
        }

        public void Step(float dt)
        {
            if(Complete)return;
            float before=Elapsed;
            Elapsed+=dt;
            int remaining=0;
            bool settlementChanged=false;
            var newlyFinished=new List<int>();
            for(int i=0;i<horses.Count;i++)
            {
                var horse=horses[i];var result=results[i];
                result.progressDelta=0f;
                if(result.finished || result.dnf)continue;
                var p=horse.Torso.position;
                bool valid=Finite(p.x)&&Finite(p.y)&&Finite(p.z)&&
                    Finite(horse.Torso.linearVelocity.x)&&Finite(horse.Torso.linearVelocity.y)&&
                    Finite(horse.Torso.linearVelocity.z);
                if(!valid)throw new InvalidOperationException("Non-finite horse physics state: "+i);
                if(result.previousCondition!=HorseCondition.Fallen && horse.Condition==HorseCondition.Fallen)
                    result.falls++;
                if(fallTerminatesEpisode && horse.Condition==HorseCondition.Fallen)
                {
                    result.dnf=true;result.reason="fall";
                    result.previousCondition=horse.Condition;
                    horse.RetireFromRace();settlementChanged=true;
                    continue;
                }
                bool active=horse.IsPolicyActive && !horse.Torso.IsSleeping();
                bool upright=Vector3.Dot(horse.Torso.transform.up,Vector3.up)>.55f &&
                    p.y>horse.RestTorsoHeight*.65f;
                if(active)
                {
                    result.activeTime+=dt;
                    Vector3 velocity=horse.Torso.linearVelocity;
                    if(new Vector2(velocity.x,velocity.z).magnitude<.1f)
                        result.stationaryTime+=dt;
                }
                if(p.z>result.highWater)
                {
                    float forward=Mathf.Min(p.z-result.highWater,
                        Mathf.Max(0f,p.z-result.previousZ));
                    if(forward>0f)
                    {
                        if(Mathf.Abs(p.x)>6f)result.filteredBoundary+=forward;
                        else if(!active)result.filteredInactive+=forward;
                        else if(horse.TorsoGrounded)result.filteredGround+=forward;
                        else if(!upright)result.filteredPosture+=forward;
                        else result.progressDelta=forward;
                    }
                    result.highWater=p.z;
                }
                if(!result.distanceReached && active && upright && !horse.TorsoGrounded && Mathf.Abs(p.x)<=6f &&
                    result.previousZ<distance && p.z>=distance &&
                    horse.Torso.linearVelocity.z>0f)
                {
                    float fraction=Mathf.Clamp01((distance-result.previousZ)/Mathf.Max(1e-6f,p.z-result.previousZ));
                    result.distanceReached=true;result.finishTime=before+fraction*dt;
                    if(!fixedWindow)
                    {result.finished=true;newlyFinished.Add(i);settlementChanged=true;horse.RetireFromRace();}
                }
                if(!result.finished && (Mathf.Abs(p.x)>8f || p.y<-.5f || Elapsed>=timeout))
                {
                    if(fixedWindow && Elapsed>=timeout && Mathf.Abs(p.x)<=8f && p.y>=-.5f && result.distanceReached)
                    {result.finished=true;newlyFinished.Add(i);result.finishTime=Elapsed;}
                    else result.dnf=true;
                    settlementChanged=true;
                    result.reason=result.dnf?(Mathf.Abs(p.x)>8f?"boundary":p.y<-.5f?"ground":"timeout"):null;
                    horse.RetireFromRace();
                }
                else if(!result.finished)remaining++;
                result.previousZ=p.z;
                result.previousCondition=horse.Condition;
            }
            newlyFinished.Sort((a,b)=>
            {
                int order=results[a].finishTime.CompareTo(results[b].finishTime);
                return order!=0?order:a.CompareTo(b);
            });
            foreach(int index in newlyFinished)results[index].rank=nextRank++;
            if(settlementChanged)ValidateResults();
            Complete=remaining==0;
        }

        void ValidateResults()
        {
            var ranks=new HashSet<int>();
            foreach(var result in results)
            {
                if(result.finished && result.dnf)
                    throw new InvalidOperationException("Horse cannot both finish and DNF");
                if(result.finished && (result.rank<=0 || !ranks.Add(result.rank)))
                    throw new InvalidOperationException("Duplicate or missing race rank");
                if(!result.finished && result.rank!=0)
                    throw new InvalidOperationException("Unfinished horse has a rank");
            }
            for(int i=0;i<horses.Count;i++)if(results[i].finished || results[i].dnf)
                foreach(var collider in horses[i].GetComponentsInChildren<Collider>())
                    if(collider.enabled && collider.attachedRigidbody!=null)
                        throw new InvalidOperationException("Retired horse still blocks the track");
        }

        static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
    }
}
