using System.Collections.Generic;
using UnityEngine;

namespace QHop
{
    public sealed class ArenaEffects : MonoBehaviour
    {
        readonly Queue<ParticleSystem> pool = new Queue<ParticleSystem>();
        readonly List<ParticleSystem> all = new List<ParticleSystem>();
        readonly List<Transform> impactStars = new List<Transform>();
        readonly List<float> starExpiry = new List<float>();
        Material starMaterial;
        Material dustMaterial;
        AudioSource source;
        AudioClip[] clips;
        void Awake()
        {
            source=gameObject.AddComponent<AudioSource>();
            source.spatialBlend=.35f;
            clips=new[] { BuildClip("bump",110f),BuildClip("fall",72f),BuildClip("slip",180f) };
            var starShader=Shader.Find("Universal Render Pipeline/Unlit");
            var dustShader=Shader.Find("Universal Render Pipeline/Particles/Unlit");
            if(starShader==null){enabled=false;return;}
            starMaterial=new Material(starShader);
            starMaterial.color=new Color(1f,.87f,.28f);
            dustMaterial=new Material(dustShader!=null?dustShader:starShader);
            dustMaterial.color=new Color(.94f,.75f,.51f,.8f);
            for(int i=0;i<24;i++)
            {
                var go=new GameObject("PooledDust_"+i);
                go.SetActive(false);
                go.transform.SetParent(transform);
                var ps=go.AddComponent<ParticleSystem>();
                ps.Stop(true,ParticleSystemStopBehavior.StopEmittingAndClear);
                var main=ps.main;main.playOnAwake=false;main.duration=.8f;
                main.startLifetime=.55f;main.startSpeed=1.8f;main.startSize=.19f;
                main.startColor=new Color(1f,.83f,.58f,.75f);
                main.simulationSpace=ParticleSystemSimulationSpace.World;
                var emission=ps.emission;emission.rateOverTime=0f;
                var shape=ps.shape;shape.shapeType=ParticleSystemShapeType.Sphere;shape.radius=.2f;
                var renderer=ps.GetComponent<ParticleSystemRenderer>();
                renderer.renderMode=ParticleSystemRenderMode.Billboard;
                renderer.sharedMaterial=dustMaterial;
                pool.Enqueue(ps);all.Add(ps);
            }
            Mesh star=MakeStar();
            for(int i=0;i<16;i++)
            {
                var go=new GameObject("Pooled Comic Star",typeof(MeshFilter),typeof(MeshRenderer));
                go.transform.SetParent(transform);go.GetComponent<MeshFilter>().sharedMesh=star;
                go.GetComponent<MeshRenderer>().sharedMaterial=starMaterial;
                go.SetActive(false);impactStars.Add(go.transform);starExpiry.Add(0f);
            }
        }
        static Mesh MakeStar()
        {
            var mesh=new Mesh { name="Five Point Impact Star" };
            var vertices=new Vector3[11];var triangles=new int[30];
            vertices[0]=Vector3.zero;
            for(int i=0;i<10;i++)
            {
                float a=i*Mathf.PI/5f;
                float r=i%2==0?.5f:.21f;
                vertices[i+1]=new Vector3(Mathf.Cos(a)*r,Mathf.Sin(a)*r,0f);
                int t=i*3;triangles[t]=0;triangles[t+1]=i+1;triangles[t+2]=(i+1)%10+1;
            }
            mesh.vertices=vertices;mesh.triangles=triangles;mesh.RecalculateNormals();return mesh;
        }
        AudioClip BuildClip(string name,float pitch)
        {
            const int rate=22050;
            var samples=new float[rate/4];
            var random=new System.Random(name.GetHashCode());
            for(int i=0;i<samples.Length;i++)
            {
                float t=(float)i/rate;
                float envelope=Mathf.Exp(-18*t);
                samples[i]=(Mathf.Sin(2*Mathf.PI*(pitch-65*t)*t)*.3f+
                    ((float)random.NextDouble()*2f-1f)*.13f)*envelope;
            }
            var clip=AudioClip.Create(name,samples.Length,1,rate,false);
            clip.SetData(samples,0);return clip;
        }
        public void Play(HorseEvent value)
        {
            if(starMaterial==null)return;
            if(value.type!=HorseEventType.Fall && value.type!=HorseEventType.Collision && value.type!=HorseEventType.Slip) return;
            ParticleSystem ps=pool.Count>0?pool.Dequeue():all[0];
            ps.gameObject.SetActive(true);
            ps.transform.position=value.point;
            var main=ps.main;
            main.startSize=value.type==HorseEventType.Fall?.3f:.15f;
            ps.Emit(value.type==HorseEventType.Fall?32:value.type==HorseEventType.Slip?5:12);
            if(value.type!=HorseEventType.Slip)
            {
                int count=value.type==HorseEventType.Fall?4:2;
                for(int n=0;n<count;n++)
                {
                    for(int i=0;i<impactStars.Count;i++)
                    {
                        if(impactStars[i].gameObject.activeSelf)continue;
                        var star=impactStars[i];star.gameObject.SetActive(true);
                        float angle=n*Mathf.PI*2f/count;
                        star.position=value.point+new Vector3(Mathf.Cos(angle)*.45f,.5f+Mathf.Sin(angle)*.25f,Mathf.Sin(angle)*.45f);
                        star.localScale=Vector3.one*(value.type==HorseEventType.Fall?.5f:.3f);
                        starExpiry[i]=Time.unscaledTime+.65f;
                        break;
                    }
                }
            }
            if(source!=null)source.PlayOneShot(clips[value.type==HorseEventType.Fall?1:value.type==HorseEventType.Slip?2:0],Mathf.Clamp01(value.intensity*.3f));
        }
        void Update()
        {
            foreach(var ps in all)
                if(ps.gameObject.activeSelf && !ps.IsAlive(true))
                { ps.gameObject.SetActive(false); if(!pool.Contains(ps))pool.Enqueue(ps); }
            for(int i=0;i<impactStars.Count;i++)
            {
                var star=impactStars[i];if(!star.gameObject.activeSelf)continue;
                if(Time.unscaledTime>=starExpiry[i]){star.gameObject.SetActive(false);continue;}
                if(Camera.main!=null)star.rotation=Camera.main.transform.rotation;
                star.Rotate(0f,0f,Time.unscaledDeltaTime*180f,Space.Self);
                star.localScale*=Mathf.Max(.01f,1f-Time.unscaledDeltaTime*.9f);
            }
        }
        void OnDestroy()
        {
            if(starMaterial!=null)Destroy(starMaterial);
            if(dustMaterial!=null)Destroy(dustMaterial);
            if(clips!=null)foreach(var clip in clips)if(clip!=null)Destroy(clip);
        }
    }
}
