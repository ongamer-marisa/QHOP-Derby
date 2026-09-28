using UnityEngine;

namespace QHop
{
    public sealed class HorseTrainingReward
    {
        public const string Contract="stage1-v2:p=2;posture=-0.10(1-u);time=-0.05;fall=-0.25;finish=2;dnf=-1;early-dnf-remaining-time=-0.05";
        readonly int stage;
        readonly float shaping;
        readonly float timeLimit;
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

        public HorseTrainingReward(int stage,float shaping=1f,float timeLimit=20f)
        {this.stage=stage;this.shaping=shaping;this.timeLimit=timeLimit;}

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
                Vector3 hoofVelocity=horse.bodies[6+i*4].linearVelocity;
                l+=Mathf.Clamp01(new Vector2(hoofVelocity.x,hoofVelocity.z).magnitude/5f);
            }
            l=Mathf.Clamp01(l/4f);
            int newFalls=race.falls-previousFalls;
            previousFalls=race.falls;
            bool finish=race.finished&&!previousFinished;
            bool dnf=race.dnf&&!previousDnf;
            previousFinished=race.finished;previousDnf=race.dnf;
            float rank=competitors>1?(competitors-race.rank)/(float)(competitors-1):0f;
            float progressWeight=1f,posture=0f,smoothness=0f,effort=0f,slip=0f;
            float time=-.05f*dt,fall=-2f*newFalls,finishBonus=0f;
            if(stage==1)
            {
                progressWeight=2f;
                posture=-.10f*(1f-u)*dt;
                fall=-.25f*newFalls;
                finishBonus=finish?2f:0f;
                if(dnf && race.reason!="timeout")
                    time-=.05f*Mathf.Max(0f,timeLimit-elapsed);
            }
            else if(stage==2||stage==3)
            {posture=-.20f*b*dt;smoothness=-.05f*s*dt;effort=-.02f*e*dt;
                slip=-.05f*l*dt;finishBonus=finish?5f:0f;}
            else if(stage==4)
            {posture=-.15f*b*dt;smoothness=-.05f*s*dt;effort=-.02f*e*dt;
                slip=-.05f*l*dt;finishBonus=finish?8f:0f;}
            else if(stage==5)
            {progressWeight=.5f;posture=-.10f*b*dt;smoothness=-.025f*s*dt;
                effort=-.01f*e*dt;slip=-.025f*l*dt;time=-.10f*dt;
                fall=-newFalls;finishBonus=finish?10f+rank:0f;}
            else
            {progressWeight=.5f*shaping;posture=-shaping*.10f*b*dt;
                smoothness=-shaping*.025f*s*dt;effort=-shaping*.01f*e*dt;
                slip=-shaping*.025f*l*dt;time=-.10f*dt;
                fall=-shaping*newFalls;finishBonus=finish?10f+rank:0f;}
            float progress=progressWeight*p;
            float dnfCost=dnf?(stage==1?-1f:stage==6?-10f:-5f):0f;
            float value=progress+posture+smoothness+effort+slip+time+fall+finishBonus+dnfCost;
            Progress+=p;Posture+=b*dt;Smoothness+=s*dt;Effort+=e*dt;Slip+=l*dt;
            WeightedProgress+=progress;WeightedPosture+=posture;
            WeightedSmoothness+=smoothness;WeightedEffort+=effort;
            WeightedSlip+=slip;WeightedTime+=time;WeightedFall+=fall;
            WeightedFinish+=finishBonus;WeightedDnf+=dnfCost;
            Total+=value;
            return value;
        }
    }
}
