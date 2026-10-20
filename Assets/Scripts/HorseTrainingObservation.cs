using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace QHop
{
    public static class HorseTrainingObservation
    {
        public const int Version=4;
        // Bump together with training/schema.py after changing control/physics semantics.
        public const string PhysicsFingerprint="qh-100hz-5step-19rb-28a-lidar-gru-terminal-r14";
        public const int ProprioFrame=243;
        public const int History=4;
        public const int Terrain=70;
        public const int Opponent=12;
        public const int Opponents=3;
        public const int GaitState=28;
        public const int BaseSize=ProprioFrame*History+Terrain+Opponent*Opponents+GaitState;
        public const int Size=BaseSize+HorseLidar.Size;

        public static float[] Frame(HorseActor horse, HorseRaceResult race, float remainingTime,
            float courseLength,int disturbanceCount=0,int recoveredCount=0)
        {
            var data=new List<float>(ProprioFrame);
            var torso=horse.Torso;
            Add(data,torso.transform.InverseTransformDirection(torso.linearVelocity)/15f);
            Add(data,torso.transform.InverseTransformDirection(torso.angularVelocity)/15f);
            Add(data,torso.transform.InverseTransformDirection(Vector3.down));
            Add(data,torso.transform.InverseTransformDirection(Vector3.forward));
            foreach(var joint in horse.allJoints)
            {
                Quaternion relative=Quaternion.Inverse(joint.connectedBody.rotation)*joint.transform.rotation;
                Add(data,relative*Vector3.right);
                Add(data,relative*Vector3.up);
                Add(data,torso.transform.InverseTransformDirection(
                    joint.GetComponent<Rigidbody>().angularVelocity-joint.connectedBody.angularVelocity)/15f);
            }
            // Hoof order follows the four leg chains in HorseRigDefinition.
            for(int i=0;i<4;i++)
            {
                data.Add(horse.HoofContact[i]?1f:0f);
                var hoof=horse.bodies[3+i*4+3];
                Add(data,torso.transform.InverseTransformDirection(hoof.linearVelocity)/15f);
            }
            var c=horse.Config;
            data.Add(c.bulk);data.Add(c.frontLength);data.Add(c.backLength);data.Add(c.bodyLength);
            data.Add(c.frontStrength);data.Add(c.backStrength);data.Add(c.massKg/750f);data.Add(c.hoofSmoothness);
            data.Add(torso.position.z/100f);data.Add(torso.position.x/6f);
            data.Add((torso.position.x+6f)/12f);data.Add((6f-torso.position.x)/12f);
            data.Add(torso.position.y/Mathf.Max(.1f,horse.RestTorsoHeight));
            data.Add(remainingTime/90f);data.Add(horse.FallWaitRemaining/3f);
            data.Add(horse.Condition==HorseCondition.Unstable?1f:0f);
            data.Add(horse.Condition==HorseCondition.Fallen?1f:0f);
            data.Add(horse.Condition==HorseCondition.Recovering?1f:0f);
            data.Add(horse.TorsoGrounded?1f:0f);
            data.Add(horse.ConditionTimer/3f);
            data.Add(race.finished?1f:0f);
            data.Add(race.dnf?1f:0f);
            data.Add(race.falls/10f);
            data.Add(disturbanceCount/10f);
            data.Add(recoveredCount/10f);
            foreach(float action in horse.LastActionValues)data.Add(action);
            if(data.Count!=ProprioFrame)throw new InvalidOperationException("Proprio observation mismatch: "+data.Count);
            return data.ToArray();
        }

        public static float[] TerrainGrid(HorseActor horse)
        {
            var data=new float[Terrain];
            var hits=new RaycastHit[32];
            var center=horse.Torso.position;
            Vector3 forward=Vector3.ProjectOnPlane(horse.Torso.transform.forward,Vector3.up).normalized;
            if(forward.sqrMagnitude<.1f)forward=Vector3.forward;
            Vector3 right=Vector3.Cross(Vector3.up,forward);
            int index=0;
            for(int row=0;row<7;row++)for(int col=0;col<5;col++)
            {
                Vector3 point=center+forward*new[]{0f,.5f,1f,2f,3f,4f,6f}[row]+right*((col-2)*2f);
                int count=horse.gameObject.scene.GetPhysicsScene().Raycast(
                    point+Vector3.up*3f,Vector3.down,hits,8f,
                    Physics.DefaultRaycastLayers,QueryTriggerInteraction.Ignore);
                bool valid=false;float height=0f;float closest=float.MaxValue;
                for(int h=0;h<count;h++)
                {
                    if(hits[h].collider.GetComponentInParent<HorseGroundSurface>()==null)continue;
                    if(hits[h].distance>=closest)continue;
                    closest=hits[h].distance;
                    height=hits[h].point.y;
                    valid=true;
                }
                data[index++]=valid?Mathf.Clamp((height-center.y)/3f,-1f,1f):0f;
                data[index++]=valid?1f:0f;
            }
            return data;
        }

        public static float[] OpponentData(HorseActor horse,IReadOnlyList<HorseActor> horses,
            IReadOnlyList<HorseRaceResult> results)
        {
            var data=new float[Opponent*Opponents];int slot=0;
            foreach(var other in horses)
            {
                if(other==horse || slot>=Opponents)continue;
                int start=slot++*Opponent;
                if(results!=null && (results[other.Id].finished || results[other.Id].dnf))continue;
                Vector3 delta=other.Torso.position-horse.Torso.position;
                Vector3 velocity=other.Torso.linearVelocity-horse.Torso.linearVelocity;
                Vector3 heading=other.Torso.transform.forward;
                var t=horse.Torso.transform;
                Write(data,start,t.InverseTransformDirection(delta)/20f);
                Write(data,start+3,t.InverseTransformDirection(velocity)/15f);
                Write(data,start+6,t.InverseTransformDirection(heading));
                data[start+9]=other.Condition==HorseCondition.Fallen?1f:0f;
                data[start+10]=other.Condition==HorseCondition.Recovering?1f:0f;
                data[start+11]=1f;
            }
            return data;
        }

        static void Add(List<float> values,Vector3 value)
        {values.Add(value.x);values.Add(value.y);values.Add(value.z);}
        static void Write(float[] values,int index,Vector3 value)
        {values[index]=value.x;values[index+1]=value.y;values[index+2]=value.z;}
    }

    public sealed class HorseObservationHistory
    {
        readonly Queue<float[]> frames=new Queue<float[]>();
        public void Reset(float[] frame)
        {
            frames.Clear();
            for(int i=0;i<HorseTrainingObservation.History;i++)frames.Enqueue((float[])frame.Clone());
        }
        public float[] Build(float[] frame,float[] terrain,float[] opponents,float[] gait=null,float[] lidar=null)
        {
            if(frames.Count==0)Reset(frame);
            else {frames.Dequeue();frames.Enqueue((float[])frame.Clone());}
            var result=new float[HorseTrainingObservation.Size];int offset=0;
            foreach(var old in frames){Array.Copy(old,0,result,offset,old.Length);offset+=old.Length;}
            Array.Copy(terrain,0,result,offset,terrain.Length);offset+=terrain.Length;
            Array.Copy(opponents,0,result,offset,opponents.Length);offset+=opponents.Length;
            if(gait!=null)Array.Copy(gait,0,result,offset,HorseTrainingObservation.GaitState);
            if(lidar!=null)Array.Copy(lidar,0,result,HorseTrainingObservation.BaseSize,HorseLidar.Size);
            return result;
        }
    }
}
