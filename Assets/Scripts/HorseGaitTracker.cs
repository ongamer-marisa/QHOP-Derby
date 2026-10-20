using System;
using System.Collections.Generic;
using UnityEngine;

namespace QHop
{
    // FL, FR, HL, HR. Event-driven: no imposed clock or animation phase.
    public sealed class HorseGaitTracker
    {
        struct Landing { public int foot; public float time; public bool valid; }
        readonly List<Landing> events=new List<Landing>(5);
        readonly bool[] contact=new bool[4];
        readonly int[] pending=new int[4];
        readonly float[] changed=new float[4],down=new float[4],up=new float[4];
        readonly float[] height=new float[4],peak=new float[4],startZ=new float[4],travel=new float[4];
        readonly float[] validAt=new float[4],relativeZ=new float[4];
        readonly Queue<float> recent=new Queue<float>();
        readonly Queue<string>[] timelines={new Queue<string>(),new Queue<string>(),new Queue<string>(),new Queue<string>()};
        sealed class Candidate
        {
            public int[] feet,groups;public int shift;public string gait;
        }
        readonly List<Candidate> candidates=new List<Candidate>();
        bool initialized,cycleSync,cycleIllegal,advancing;
        float settlement,credit,lastDt=.01f;
        public bool Advancing => advancing;
        public float Participation {get;private set;}
        public float Missing {get;private set;}
        public float Drag {get;private set;}
        float clock,lastEvent=-10f,lastCycle=-10f;
        int target=-1;
        public int Cycles {get;private set;}
        public int CorrectCycles {get;private set;}
        public int BadCycles {get;private set;}
        public int SynchronyCycles {get;private set;}
        public int DragCycles {get;private set;}
        public int MissingCycles {get;private set;}
        public int ActiveTicks {get;private set;}
        public int HighTicks {get;private set;}
        public int ParticipationTicks {get;private set;}
        public int SpeedTicks {get;private set;}
        public int StableTicks {get;private set;}
        public float Process {get;private set;}
        public float Quality {get;private set;}
        public float Coordination {get;private set;}
        public float Bad {get;private set;}
        public float Synchrony {get;private set;}
        public float High {get;private set;}
        public float Fold {get;private set;}
        // Cycle is a period-weighted payout rate on the closing physics frame only.
        public float Cycle {get;private set;}
        public float Period {get;private set;}=.8f;
        public float SpeedMin {get;set;}=2f;
        public float SpeedMax {get;set;}=4f;
        public string Gait {get;private set;}="none";
        public string Lead {get;private set;}="none";
        public string Contacts => string.Concat(Array.ConvertAll(contact,x=>x?"1":"0"));
        public float CorrectRatio => Cycles>0?(float)CorrectCycles/Cycles:0f;
        public float BadRatio => Cycles>0?(float)BadCycles/Cycles:1f;
        public float SynchronyRatio => Cycles>0?(float)SynchronyCycles/Cycles:0f;
        public float HighFraction => ActiveTicks>0?(float)HighTicks/ActiveTicks:0f;
        public float ParticipationFraction => ActiveTicks>0?(float)ParticipationTicks/ActiveTicks:0f;
        public float SpeedFraction => StableTicks>0?(float)SpeedTicks/StableTicks:0f;
        float MaxPeriod => target<0?2.5f:1.5f;
        float Tolerance => Mathf.Clamp(Period*.08f,.02f,.06f);
        public string Timeline(int foot) => string.Join(" ",timelines[foot].ToArray());

        void ClearActive()
        {
            initialized=false;events.Clear();candidates.Clear();recent.Clear();cycleSync=cycleIllegal=false;
            Quality=Coordination=Participation=Process=Cycle=Bad=Synchrony=Missing=Drag=High=Fold=credit=0f;settlement=clock;
            Gait=Lead="none";lastCycle=lastEvent=-10f;Period=.8f;
            Array.Clear(pending,0,4);
            for(int i=0;i<4;i++){height[i]=peak[i]=travel[i]=relativeZ[i]=0f;validAt[i]=-10f;down[i]=up[i]=clock;}
        }
        public void Reset()
        {
            clock=0f;ClearActive();
            Cycles=CorrectCycles=BadCycles=SynchronyCycles=DragCycles=MissingCycles=0;
            ActiveTicks=HighTicks=ParticipationTicks=SpeedTicks=StableTicks=0;
            foreach(var line in timelines)line.Clear();
        }

        bool Match(int[] feet,int[] groups,float duration)
        {
            if(events.Count!=4)return false;
            for(int shift=0;shift<4;shift++)
            {
                bool ok=true;
                for(int i=0;i<4;i++)
                {
                    int j=(shift+i)%4,k=(shift+i+1)%4;
                    if(events[i].foot!=feet[j]){ok=false;break;}
                    float gap=i==3?duration-(events[3].time-events[0].time):events[i+1].time-events[i].time;
                    if(groups[j]==groups[k]?gap>Tolerance:gap<=Tolerance){ok=false;break;}
                }
                if(ok)return true;
            }
            return false;
        }
        bool Classify(float duration)
        {
            bool walk=duration>=.6f && duration<=2.5f && Match(new[]{2,0,3,1},new[]{0,1,2,3},duration);
            bool running=duration>=.35f && duration<=1.5f;
            bool trot=false;
            foreach(int[] pattern in new[]{new[]{0,3,1,2},new[]{3,0,1,2},new[]{0,3,2,1},new[]{3,0,2,1}})
                trot|=running && Match(pattern,new[]{0,0,1,1},duration);
            bool lc=running && (Match(new[]{3,2,1,0},new[]{0,1,1,2},duration)||Match(new[]{3,1,2,0},new[]{0,1,1,2},duration));
            bool rc=running && (Match(new[]{2,3,0,1},new[]{0,1,1,2},duration)||Match(new[]{2,0,3,1},new[]{0,1,1,2},duration));
            bool lg=running && Match(new[]{3,2,1,0},new[]{0,1,2,3},duration);
            bool rg=running && Match(new[]{2,3,0,1},new[]{0,1,2,3},duration);
            Gait=trot?"trot":walk?"walk":lc||rc?"canter":lg||rg?"gallop":"invalid";
            Lead=lc||lg?"left":rc||rg?"right":"none";
            bool valid=target==0?trot:target==1?lc||rc:target==2?lg||rg:
                target==-2?walk||trot:walk||trot||lc||rc||lg||rg;
            foreach(var landing in events)valid&=landing.valid;
            return valid;
        }
        void AddCandidates(int[] feet,int[] groups,string gait,int first)
        {
            for(int shift=0;shift<4;shift++)if(feet[shift]==first)
                candidates.Add(new Candidate{feet=feet,groups=groups,gait=gait,shift=shift});
        }
        void Begin(Landing first)
        {
            events.Clear();candidates.Clear();cycleIllegal=false;cycleSync=false;
            settlement=Mathf.Max(settlement,first.time);
            if(target<0 || target==0)foreach(var feet in new[]{new[]{0,3,1,2},new[]{3,0,1,2},new[]{0,3,2,1},new[]{3,0,2,1}})
                AddCandidates(feet,new[]{0,0,1,1},"trot",first.foot);
            if(target<0)AddCandidates(new[]{2,0,3,1},new[]{0,1,2,3},"walk",first.foot);
            if(target==-1 || target==1)
            {
                AddCandidates(new[]{3,2,1,0},new[]{0,1,1,2},"canter",first.foot);
                AddCandidates(new[]{3,1,2,0},new[]{0,1,1,2},"canter",first.foot);
                AddCandidates(new[]{2,3,0,1},new[]{0,1,1,2},"canter",first.foot);
                AddCandidates(new[]{2,0,3,1},new[]{0,1,1,2},"canter",first.foot);
            }
            if(target==-1 || target==2)
            {
                AddCandidates(new[]{3,2,1,0},new[]{0,1,2,3},"gallop",first.foot);
                AddCandidates(new[]{2,3,0,1},new[]{0,1,2,3},"gallop",first.foot);
            }
        }
        void ValidateCandidates(bool waiting)
        {
            if(events.Count==0)return;
            for(int n=candidates.Count-1;n>=0;n--)
            {
                var c=candidates[n];bool ok=true;
                for(int i=0;i<events.Count;i++)
                {
                    int j=(c.shift+i)%4;
                    if(!events[i].valid || events[i].foot!=c.feet[j]){ok=false;break;}
                    if(i>0)
                    {
                        float gap=events[i].time-events[i-1].time;
                        bool pair=c.groups[j]==c.groups[(j+3)%4];
                        if(pair?gap>Tolerance:gap<=Tolerance){ok=false;break;}
                    }
                }
                if(ok && waiting)
                {
                    int last=(c.shift+events.Count-1)%4,next=(last+1)%4;
                    if(c.groups[last]==c.groups[next] && clock-events[events.Count-1].time>Tolerance+.011f)ok=false;
                }
                if(!ok)candidates.RemoveAt(n);
            }
            if(candidates.Count==0)cycleIllegal=true;
        }
        float Prefix()=>!cycleIllegal && candidates.Count>0?events.Count/4f:0f;
        float MatchSupport()
        {
            float best=0f;
            foreach(var c in candidates)
            {
                if(c.gait=="trot")
                {best=Mathf.Max(best,contact[0]==contact[3] && contact[1]==contact[2] && contact[0]!=contact[1]?1f:0f);continue;}
                int groups=0,complete=0;
                for(int g=0;g<4;g++)
                {
                    int expected=0,seen=0;
                    for(int k=0;k<4;k++)if(c.groups[k]==g)expected++;
                    if(expected==0)continue;groups++;
                    for(int k=0;k<events.Count;k++)if(c.groups[(c.shift+k)%4]==g)seen++;
                    if(seen==expected)complete++;
                }
                best=Mathf.Max(best,(float)complete/groups);
            }
            return best;
        }
        void Settle(float end,bool correct,bool bad,bool sync,bool missing,bool drag)
        {
            float duration=Mathf.Max(0f,end-settlement);settlement=Mathf.Max(settlement,end);
            if(correct)Cycle+=.75f*duration/lastDt;
            if(bad)Bad+=duration/lastDt;
            if(sync)Synchrony+=duration/lastDt;
            if(missing)Missing+=duration/lastDt;
            if(drag)Drag+=duration/lastDt;
        }
        void Close(float duration,float dt,bool missing=false)
        {
            bool template=Classify(duration);
            bool valid=!missing && !cycleIllegal && template && advancing;
            if(!valid){Gait="invalid";Lead="none";}
            bool drag=events.Exists(x=>!x.valid);
            Cycles++;if(valid)CorrectCycles++;else BadCycles++;
            if(cycleSync)SynchronyCycles++;if(missing)MissingCycles++;if(drag)DragCycles++;
            recent.Enqueue(valid?1f:0f);if(recent.Count>3)recent.Dequeue();
            float sum=0f;foreach(float x in recent)sum+=x;Quality=sum/3f;lastCycle=clock;
            Settle(events[0].time+duration,valid,!valid,cycleSync,missing,drag);
            if(valid)Period=duration;
        }
        // Called after the final physics tick, before terminal reward. It does not
        // award an incomplete cycle or count it as a fully evaluated cycle.
        public void FinishPending(bool inactive=false)
        {
            if(events.Count==0)return;
            bool drag=events.Exists(x=>!x.valid);
            Settle(clock,false,cycleIllegal||cycleSync||drag||inactive,cycleSync,inactive,drag);
            events.Clear();candidates.Clear();cycleSync=cycleIllegal=false;credit=0f;
        }
        public void Step(HorseActor horse,float dt,int task)
        {
            var clearances=new float[4];var lengths=new float[4];var positions=new float[4];
            for(int i=0;i<4;i++)
            {
                lengths[i]=horse.LegLength(i);clearances[i]=horse.HoofContact[i]?0f:horse.HoofClearance(i);
                positions[i]=horse.Torso.transform.InverseTransformPoint(horse.Hooves[i].position).z;
            }
            StepContactSample(horse.HoofContact,clearances,1f,horse.Torso.linearVelocity.z,
                horse.IsPolicyActive,dt,task,Mathf.Max(0f,horse.Torso.position.y-horse.RestTorsoHeight),lengths,positions);
            foreach(var descriptor in horse.LegJoints)Fold=Mathf.Max(Fold,descriptor.SoftError);
        }
        public void StepContactSample(bool[] raw,float[] clearances,float legLength,float speed,
            bool active,float dt,int task,float bodyRise=0f,float[] lengths=null,float[] positions=null)
        {
            target=task;clock+=dt;lastDt=dt;
            Process=Cycle=Bad=Synchrony=Missing=Drag=High=Fold=0f;
            if(!active)
            {
                FinishPending(true);float bad=Bad,sync=Synchrony,missing=Missing,drag=Drag;
                ClearActive();Bad=bad;Synchrony=sync;Missing=missing;Drag=drag;return;
            }
            bool moving=speed>.2f;advancing=moving;
            if(moving)ActiveTicks++;
            if(clock>2f){StableTicks++;if(speed>=SpeedMin && speed<=SpeedMax)SpeedTicks++;}
            if(events.Count>0 && clock-events[0].time>MaxPeriod)
            {Close(clock-events[0].time,dt,true);events.Clear();candidates.Clear();cycleSync=cycleIllegal=false;}
            if(clock-lastCycle>Mathf.Min(MaxPeriod,Period*1.5f)){Quality=0f;recent.Clear();}
            bool newValidLanding=false;
            for(int i=0;i<4;i++)
            {
                float length=Mathf.Max(.1f,lengths==null?legLength:lengths[i]);
                float z=positions==null?0f:positions[i];relativeZ[i]=z/length;
                if(!initialized){contact[i]=raw[i];changed[i]=down[i]=up[i]=clock;startZ[i]=z;validAt[i]=-10f;continue;}
                if(!raw[i])
                {
                    float net=Mathf.Max(0f,clearances[i]-bodyRise);
                    height[i]=Mathf.Max(height[i],net/length);
                    travel[i]=Mathf.Max(travel[i],(z-startZ[i])/length);
                    float threshold=task==1?.35f:task==2 || task==-1 && Gait=="gallop"?.4f:.25f;
                    High=Mathf.Max(High,Mathf.Clamp01((net/length-threshold)/.20f));
                }
                pending[i]=raw[i]==contact[i]?0:pending[i]+1;
                if(pending[i]<2)continue;
                float when=clock-dt;
                contact[i]=raw[i];pending[i]=0;changed[i]=when;
                if(!raw[i]){up[i]=when;startZ[i]=z;height[i]=travel[i]=0f;continue;}
                bool valid=when-up[i]>=.029f && up[i]-down[i]>=.029f && height[i]>=.02f && travel[i]>=.05f;
                down[i]=when;peak[i]=height[i];
                if(valid)validAt[i]=clock;
                timelines[i].Enqueue(when.ToString("F2")+(valid?"+":"!"));
                if(timelines[i].Count>8)timelines[i].Dequeue();
                var landing=new Landing{foot=i,time=when,valid=valid};
                if(events.Count==4 && i==events[0].foot)
                {
                    var last=events[3];
                    if(when-last.time<=Tolerance && (i/2==last.foot/2 || (task==0 || task==-2) && i%2==last.foot%2))cycleSync=true;
                    Close(when-events[0].time,dt);Begin(landing);}
                else if(events.Exists(x=>x.foot==i))
                {Close(when-events[0].time,dt,true);Begin(landing);}
                else if(events.Count==0)Begin(landing);
                foreach(var e in events)if(when-e.time<=Tolerance &&
                    (i/2==e.foot/2 || (task==0 || task==-2) && i%2==e.foot%2))cycleSync=true;
                events.Add(landing);lastEvent=clock;
                ValidateCandidates(false);
                newValidLanding|=valid;
            }
            initialized=true;
            int participating=0;float swings=0f;
            for(int i=0;i<4;i++)
            {
                if(clock-validAt[i]<=MaxPeriod)participating++;
                if(!contact[i])swings+=Mathf.Min(Mathf.Clamp01(height[i]/.02f),Mathf.Clamp01(travel[i]/.05f));
            }
            ValidateCandidates(true);
            // Settle only after the whole frame's event group has been checked.
            // Paying the first hoof before its simultaneous partner is processed
            // lets an illegal pair farm a fresh prefix on every cycle.
            bool groupComplete=false;
            foreach(var c in candidates)
            {
                int group=c.groups[(c.shift+events.Count-1)%4],expected=0,seen=0;
                for(int k=0;k<4;k++)if(c.groups[k]==group)expected++;
                for(int k=0;k<events.Count;k++)if(c.groups[(c.shift+k)%4]==group)seen++;
                groupComplete|=seen==expected;
            }
            float prefix=Prefix();
            if(newValidLanding && moving && groupComplete && prefix>0f)
            {Process+=prefix*credit/dt;credit=0f;}
            Participation=moving && clock-lastEvent<MaxPeriod?participating/4f:0f;
            Coordination=moving && clock-lastEvent<MaxPeriod?
                Mathf.Clamp01((.75f*MatchSupport()+.25f*Participation)*Prefix()):0f;
            if(moving && participating==4)ParticipationTicks++;
            if(moving && High>0f)HighTicks++;
            if(moving)credit=Mathf.Min(.20f*MaxPeriod,credit+.20f*dt);
            else {Process=Cycle=Coordination=Participation=credit=0f;}
        }

        public float[] Observation(int task,float targetSpeed)
        {
            var data=new float[28];if(task>=0 && task<3)data[task]=1f;data[3]=targetSpeed/15f;
            if(!cycleIllegal)foreach(var c in candidates)data[4+c.feet[(c.shift+events.Count)%4]]=1f;
            if(events.Count==0)for(int foot=0;foot<4;foot++)data[4+foot]=1f;
            for(int i=0;i<4;i++)
            {
                data[8+i]=Mathf.Clamp01((clock-down[i])/2.5f);
                data[12+i]=Mathf.Clamp01((clock-up[i])/2.5f);
                data[16+i]=Mathf.Clamp01(contact[i]?peak[i]:height[i]);
                data[20+i]=Mathf.Clamp(relativeZ[i],-1f,1f);
            }
            data[24]=Lead=="left"?-1f:Lead=="right"?1f:0f;data[25]=Quality;
            data[26]=Mathf.Clamp01((clock-lastCycle)/2.5f);data[27]=Period/2.5f;
            return data;
        }
    }
}
