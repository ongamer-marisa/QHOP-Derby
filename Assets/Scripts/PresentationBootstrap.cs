using UnityEngine;
using UnityEngine.SceneManagement;

namespace QHop
{
    /// <summary>Allows the checked-in scene shells to play before Editor scene generation.</summary>
    public static class PresentationBootstrap
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Start()
        {
            string name=SceneManager.GetActiveScene().name;
            if(name!="DerbyStraight" && name!="AttributeLab")return;
            if(Object.FindFirstObjectByType<ArenaController>()!=null)return;
            var prefab=Resources.Load<GameObject>("HorseGame");
            if(prefab==null){Debug.LogError("HorseGame.fbx was not imported.");return;}
            bool experiment=name=="AttributeLab";
            var shader=Shader.Find("QHOP/SoftToon");
            if(shader==null){Debug.LogError("SoftToon shader could not be found.");return;}
            Material horse=new Material(shader), track=new Material(shader), rail=new Material(shader);
            horse.SetColor("_BaseColor",new Color(.72f,.49f,.29f));
            track.SetColor("_BaseColor",new Color(.68f,.72f,.52f));
            rail.SetColor("_BaseColor",new Color(.87f,.81f,.64f));
            RenderSettings.ambientLight=new Color(.65f,.72f,.77f);
            RenderSettings.fog=true;RenderSettings.fogColor=new Color(.74f,.85f,.84f);
            RenderSettings.fogDensity=.005f;
            var camera=Camera.main;
            if(camera==null)
            {
                var cam=new GameObject("Main Camera",typeof(Camera),typeof(AudioListener));cam.tag="MainCamera";
                camera=cam.GetComponent<Camera>();
            }
            camera.transform.position=new Vector3(12,8,-12);
            camera.transform.LookAt(new Vector3(0,1,3));
            camera.clearFlags=CameraClearFlags.SolidColor;
            camera.backgroundColor=new Color(.68f,.84f,.9f);
            camera.fieldOfView=48f;
            var sun=Object.FindFirstObjectByType<Light>();
            if(sun==null) sun=new GameObject("Warm Light").AddComponent<Light>();
            sun.type=LightType.Directional;sun.intensity=1.1f;sun.color=new Color(1f,.89f,.74f);
            sun.transform.rotation=Quaternion.Euler(45f,-28f,0f);sun.shadows=LightShadows.Soft;
            float length=experiment?32f:120f;
            float center=experiment?0f:50f;
            var ground=Cube("Track Ground",new Vector3(0,-.12f,center),new Vector3(12,.2f,length),track);
            ground.AddComponent<HorseGroundSurface>();
            ground.GetComponent<Collider>().material=new PhysicsMaterial("Track Grip")
            { staticFriction=1f,dynamicFriction=1f,frictionCombine=PhysicsMaterialCombine.Multiply };
            for(int i=0;i<5;i++)
            {
                var stripe=Cube("Lane Stripe",new Vector3(-6+i*3,.002f,center),new Vector3(.045f,.005f,length),rail);
                Object.Destroy(stripe.GetComponent<Collider>());
            }
            for(int side=-1;side<=1;side+=2)
            {
                Cube("Outside Rail",new Vector3(side*6.8f,.55f,center),new Vector3(.12f,.12f,length),rail);
                for(int i=0;i<(experiment?5:16);i++)
                    Cube("Rail Post",new Vector3(side*6.8f,.3f,center-length*.5f+i*8),new Vector3(.14f,.6f,.14f),rail);
            }
            if(!experiment)
            {
                var finish=Cube("Finish Stripe",new Vector3(0,.005f,100),new Vector3(12,.015f,.16f),rail);
                Object.Destroy(finish.GetComponent<Collider>());
            }
            for(int i=0;i<20;i++)
            {
                float z=center-length*.5f+i*length/20f;
                var hill=GameObject.CreatePrimitive(PrimitiveType.Cube);
                hill.name="Low Poly Hill";
                hill.transform.position=new Vector3(i%2==0?-12f:12f,-1.5f,z);
                hill.transform.localScale=new Vector3(8f,3.2f,9f);
                hill.transform.rotation=Quaternion.Euler(0f,i*37f,9f);
                hill.GetComponent<Renderer>().sharedMaterial=track;
            }
            var arena=new GameObject("QHOP Arena");
            arena.SetActive(false);
            arena.AddComponent<HorseTelemetryRecorder>();
            arena.AddComponent<ArenaResetCoordinator>();
            var component=arena.AddComponent<ArenaController>();
            component.horsePrefab=prefab;
            component.rig=ScriptableObject.CreateInstance<HorseRigDefinition>();
            component.horseMaterial=horse;
            component.experimentScene=experiment;
            component.viewCamera=camera;
            arena.GetComponent<ArenaResetCoordinator>().arena=component;
            arena.AddComponent<ArenaEffects>();
            arena.SetActive(true);
        }
        static GameObject Cube(string name,Vector3 position,Vector3 scale,Material material)
        {
            var go=GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name=name;go.transform.position=position;go.transform.localScale=scale;
            go.GetComponent<Renderer>().sharedMaterial=material;
            return go;
        }
    }
}
