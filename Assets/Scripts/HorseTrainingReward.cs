using UnityEngine;

namespace QHop
{
    public sealed class HorseTrainingReward
    {
        public const string Contract="gait-terminal-v4:stage1=.1;participation=.1;coord=.9(.75D+.25E)P;speed=.6-linear(.1+.9maxCQ);process=.2-token;cycle=.75period;order=-.75period;sync=-1.5period;missing=-.5period;drag=-.3period;over=-.5;high=-.6;fold=-.6-descriptor;slip=-.2;posture=-.2;fall=-2;finish=2q;dnf=-5;early=-5.27remaining;later=.25;fall-terminal;fall-no-remaining-penalty";
        readonly int stage;
        readonly float shaping;
        readonly float timeLimit;
        readonly int taskIndex;
        public readonly HorseGaitTracker Gait;
        float[] previousAction=new float[HorseAction.Size];
        bool previousFinished,previousDnf;
        int previousFalls;
        public float Progress { get; private set; }
        public float Posture { get; private set; }
        public float Smoothness { get; private set; }
        public float Effort { get; private set; }
        public float Slip { get; private set; }
        public float Total { get; private set; }
        public float WeightedProgress { get; private set; }
        public float WeightedPosture { get; private set; }
        public float WeightedSmoothness { get; private set; }
        public float WeightedEffort { get; private set; }
        public float WeightedSlip { get; private set; }
        public float WeightedTime { get; private set; }
        public float WeightedFall { get; private set; }
        public float WeightedFinish { get; private set; }
        public float WeightedDnf { get; private set; }
        public float WeightedGait { get; private set; }
        public float WeightedBadGait { get; private set; }
        public float WeightedHighHoof { get; private set; }
        public float WeightedFold { get; private set; }
        public float WeightedParticipation {get;private set;}
        public float WeightedCoordination {get;private set;}
        public float WeightedProcess {get;private set;}
        public float WeightedCycle {get;private set;}
        public float WeightedOrder {get;private set;}
        public float WeightedSynchrony {get;private set;}
        public float WeightedMissing {get;private set;}
        public float WeightedDrag {get;private set;}
        public float WeightedOverspeed { get; private set; }

        public HorseTrainingReward(int stage,float shaping=1f,float timeLimit=20f,int taskIndex=0,HorseGaitTracker gait=null)
        {this.stage=stage;this.shaping=shaping;this.timeLimit=timeLimit;this.taskIndex=taskIndex;
            Gait=gait??new HorseGaitTracker();}

        public float Step(HorseActor horse,HorseRaceResult race,float dt,int competitors,
            float elapsed=0f)
        {
            float p=race.progressDelta/Mathf.Max(.1f,horse.RestTorsoHeight);
            float upright=Mathf.Clamp01(Vector3.Dot(horse.Torso.transform.up,Vector3.up));
            float height=Mathf.Clamp01(horse.Torso.position.y/Mathf.Max(.1f,horse.RestTorsoHeight));
            float u=upright*height;
            Vector3 v=horse.Torso.linearVelocity;
            float b=Mathf.Clamp01((1f-u)+Mathf.Abs(v.x)/10f+
                Mathf.Abs(horse.Torso.angularVelocity.y)/10f);
            float s=0f,e=0f;
            for(int i=0;i<HorseAction.Size;i++)
            {
                float a=horse.LastActionValues[i];
                s+=(a-previousAction[i])*(a-previousAction[i]);
                previousAction[i]=a;
                if(i>=HorseAction.AngleCount)e+=Mathf.Pow((a+1f)*.5f,2f);
            }
            s=Mathf.Clamp01(s/HorseAction.Size/4f);
            e=Mathf.Clamp01(e/HorseAction.StrengthCount);
            float l=0f;
            for(int i=0;i<horse.HoofContact.Length;i++)if(horse.HoofContact[i])
            {
                Vector3 hoofVelocity=horse.Hooves[i].linearVelocity;
                l+=Mathf.Clamp01(new Vector2(hoofVelocity.x,hoofVelocity.z).magnitude/5f);
            }
            l=Mathf.Clamp01(l/4f);
            int newFalls=race.falls-previousFalls;
            previousFalls=race.falls;
            bool finish=race.finished&&!previousFinished;
            bool dnf=race.dnf&&!previousDnf;
            previousFinished=race.finished;previousDnf=race.dnf;
            if(finish || dnf)Gait.FinishPending();
            bool advancing=race.progressDelta>0f && (v.z>.2f || (finish && Gait.Advancing));
            float rank=competitors>1?(competitors-race.rank)/(float)(competitors-1):0f;
            float progressWeight=1f,posture=0f,smoothness=0f,effort=0f,slip=0f;
            float time=-.05f*dt,fall=-2f*newFalls,finishBonus=0f;
            if(stage==1)
            {
                progressWeight=2f;
                posture=-.10f*(1f-u)*dt;
                fall=-.25f*newFalls;
                finishBonus=finish?2f:0f;
                if(dnf && race.reason!="timeout" && race.reason!="fall")
                    time-=.05f*Mathf.Max(0f,timeLimit-elapsed);
            }
            else if(stage==2)
            {
                float speed=Mathf.Max(0f,v.z);
                float width=taskIndex==0?1f:taskIndex==1?1.5f:2f;
                float distance=Mathf.Max(Gait.SpeedMin-speed,speed-Gait.SpeedMax);
                p=advancing?.6f*Mathf.Clamp01(1f-Mathf.Max(0f,distance)/width)*dt:0f;
                progressWeight=.1f+.9f*Mathf.Max(Gait.Coordination,Gait.Quality);
                posture=-.2f*b*dt;smoothness=-.05f*s*dt;
                effort=-.02f*e*dt;slip=-.2f*l*dt;
                finishBonus=finish?2f*Gait.CorrectRatio:0f;
                if(dnf && race.reason!="timeout" && race.reason!="fall")
                    time-=5.27f*Mathf.Max(0f,timeLimit-elapsed);
            }
            else if(stage==3||stage==4)
            {posture=-.20f*b*dt;smoothness=-.05f*s*dt;effort=-.02f*e*dt;
                slip=-.05f*l*dt;finishBonus=finish?5f:0f;}
            else if(stage==5)
            {posture=-.15f*b*dt;smoothness=-.05f*s*dt;effort=-.02f*e*dt;
                slip=-.05f*l*dt;finishBonus=finish?8f:0f;}
            else if(stage==6)
            {progressWeight=.5f;posture=-.10f*b*dt;smoothness=-.025f*s*dt;
                effort=-.01f*e*dt;slip=-.025f*l*dt;time=-.10f*dt;
                fall=-newFalls;finishBonus=finish?10f+rank:0f;}
            else
            {progressWeight=.5f*shaping;posture=-shaping*.10f*b*dt;
                smoothness=-shaping*.025f*s*dt;effort=-shaping*.01f*e*dt;
                slip=-shaping*.025f*l*dt;time=-.10f*dt;
                fall=-shaping*newFalls;finishBonus=finish?10f+rank:0f;}
            float progress=progressWeight*p;
            float dnfCost=dnf?(stage==1?-1f:stage==7?-10f:-5f):0f;
            float gaitWeight=stage==1?.1f:stage==2?1f:.25f;
            float participation=advancing?gaitWeight*.1f*Gait.Participation*dt:0f;
            float coordination=advancing?gaitWeight*.9f*Gait.Coordination*dt:0f;
            float process=advancing?gaitWeight*Gait.Process*dt:0f;
            float cycle=advancing?gaitWeight*Gait.Cycle*dt:0f;
            float gaitPositive=participation+coordination+process+cycle;
            float order=-gaitWeight*.75f*Gait.Bad*dt;
            float sync=-gaitWeight*1.5f*Gait.Synchrony*dt;
            float missing=-gaitWeight*.5f*Gait.Missing*dt;
            float drag=-gaitWeight*.3f*Gait.Drag*dt;
            float gaitBad=order+sync+missing+drag;
            float overspeed=stage==2?-.5f*Mathf.Pow(Mathf.Clamp01((v.z-Gait.SpeedMax)/Mathf.Max(.5f,Gait.SpeedMax-Gait.SpeedMin)),2f)*dt:0f;
            float hoofHigh=-gaitWeight*.6f*Gait.High*Gait.High*dt;
            float fold=-gaitWeight*.6f*Gait.Fold*Gait.Fold*dt;
            WeightedParticipation+=participation;WeightedCoordination+=coordination;
            WeightedProcess+=process;WeightedCycle+=cycle;WeightedOrder+=order;
            WeightedSynchrony+=sync;WeightedMissing+=missing;WeightedDrag+=drag;
            float value=progress+posture+smoothness+effort+slip+time+fall+
                finishBonus+dnfCost+gaitPositive+gaitBad+hoofHigh+fold+overspeed;
            Progress+=race.progressDelta/Mathf.Max(.1f,horse.RestTorsoHeight);Posture+=b*dt;Smoothness+=s*dt;Effort+=e*dt;Slip+=l*dt;
            WeightedProgress+=progress;WeightedPosture+=posture;
            WeightedSmoothness+=smoothness;WeightedEffort+=effort;
            WeightedSlip+=slip;WeightedTime+=time;WeightedFall+=fall;
            WeightedFinish+=finishBonus;WeightedDnf+=dnfCost;
            WeightedGait+=gaitPositive;WeightedBadGait+=gaitBad;
            WeightedHighHoof+=hoofHigh;WeightedFold+=fold;WeightedOverspeed+=overspeed;
            Total+=value;
            return value;
        }
    }
}
