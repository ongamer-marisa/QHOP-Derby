using System;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace QHop
{
    // Tick based, world-upright scanner. Cache the actual scan geometry for visualization.
    public sealed class HorseLidar
    {
        public const int Rays=57, Features=177, Size=179, IntervalTicks=20;
        public const string Contract="lidar-v1:57;front=-70:10:70x3@25m;side=90,-90,135,-135x3@10m;heights=.1,.6,1.1;5hz;upright;self-filter;motion;dt;age;sequence";
        public readonly float[] Data=new float[Size];
        public readonly Vector3[] Origins=new Vector3[Rays],Ends=new Vector3[Rays];
        public readonly bool[] Hits=new bool[Rays];
        RaycastHit[] buffer=new RaycastHit[32];
        Vector3 previousPosition;
        Quaternion previousHeading=Quaternion.identity;
        int tick,lastTick,sequence;
        bool initialized;
        GameObject drawing;
        LineRenderer[] lines;
        Material drawingMaterial;
        public int Sequence=>sequence;
        public float Age=>Data[177];
        public void Reset(HorseActor horse)
        {tick=lastTick=sequence=0;initialized=false;Scan(horse);}
        public void Step(HorseActor horse)
        {
            tick++;if(tick-lastTick>=IntervalTicks)Scan(horse);
            Data[177]=(tick-lastTick)*.01f;
        }
        public void Scan(HorseActor horse)
        {
            var position=horse.Torso.position;
            Vector3 forward=Vector3.ProjectOnPlane(horse.Torso.transform.forward,Vector3.up).normalized;
            if(forward.sqrMagnitude<.01f)forward=previousHeading*Vector3.forward;
            var heading=Quaternion.LookRotation(forward,Vector3.up);
            float clearance=horse.RestTorsoClearance;
            float[] levels={.1f,.6f,1.1f};int ray=0;
            for(int direction=0;direction<19;direction++)
            {
                float angle=direction<15?-70f+direction*10f:new[]{90f,-90f,135f,-135f}[direction-15];
                float range=direction<15?25f:10f;
                Vector3 vector=heading*Quaternion.AngleAxis(angle,Vector3.up)*Vector3.forward;
                for(int level=0;level<3;level++,ray++)
                {
                    Vector3 origin=position+Vector3.up*((levels[level]-1f)*clearance);
                    int count;
                    do
                    {
                        count=horse.gameObject.scene.GetPhysicsScene().Raycast(origin,vector,buffer,range,
                            Physics.DefaultRaycastLayers,QueryTriggerInteraction.Ignore);
                        if(count<buffer.Length)break;
                        if(buffer.Length>=65536)throw new InvalidOperationException("LiDAR hit buffer saturated");
                        buffer=new RaycastHit[buffer.Length*2];
                    }while(true);
                    bool hit=false;float distance=range;
                    for(int i=0;i<count;i++)
                    {
                        if(buffer[i].collider.GetComponentInParent<HorseActor>()==horse)continue;
                        if(buffer[i].distance>distance)continue;
                        distance=buffer[i].distance;hit=true;
                    }
                    Data[ray*3]=distance;Data[ray*3+1]=range;Data[ray*3+2]=hit?1f:0f;
                    Origins[ray]=origin;Ends[ray]=origin+vector*distance;Hits[ray]=hit;
                }
            }
            Vector3 motion=initialized?Quaternion.Inverse(previousHeading)*(position-previousPosition):Vector3.zero;
            float yaw=initialized?Mathf.DeltaAngle(previousHeading.eulerAngles.y,heading.eulerAngles.y)*Mathf.Deg2Rad:0f;
            Data[171]=motion.x;Data[172]=motion.y;Data[173]=motion.z;
            Data[174]=Mathf.Sin(yaw);Data[175]=Mathf.Cos(yaw);
            Data[176]=initialized?(tick-lastTick)*.01f:0f;
            Data[177]=0f;Data[178]=++sequence;
            previousPosition=position;previousHeading=heading;lastTick=tick;initialized=true;
        }
        public void Draw(HorseActor horse,bool visible)
        {
            if(drawing==null && visible)
            {
                drawing=new GameObject("LidarDebug");drawing.transform.SetParent(horse.transform,false);
                var template=Resources.Load<Material>("QHopLidarDebug");
                if(template==null)throw new InvalidOperationException("LiDAR debug material missing; rebuild training Player");
                drawingMaterial=new Material(template);lines=new LineRenderer[Rays];
                for(int i=0;i<Rays;i++)
                {
                    var go=new GameObject("Ray_"+i);go.transform.SetParent(drawing.transform,false);
                    lines[i]=go.AddComponent<LineRenderer>();lines[i].sharedMaterial=drawingMaterial;
                    lines[i].useWorldSpace=true;lines[i].widthMultiplier=.012f;lines[i].positionCount=3;
                }
            }
            if(drawing==null)return;drawing.SetActive(visible);
            if(!visible)return;
            for(int i=0;i<Rays;i++)
            {
                lines[i].startColor=lines[i].endColor=Hits[i]?Color.red:Color.cyan;
                lines[i].SetPosition(0,Origins[i]);lines[i].SetPosition(1,Ends[i]);
                lines[i].SetPosition(2,Ends[i]+(Hits[i]?Vector3.up*.06f:Vector3.zero));
            }
        }
        public void Dispose(){if(drawingMaterial!=null)UnityEngine.Object.Destroy(drawingMaterial);}
    }
}
