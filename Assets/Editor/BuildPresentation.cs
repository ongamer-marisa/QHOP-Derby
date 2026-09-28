using System.IO;
using QHop;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class BuildPresentation
{
    const string AssetsRoot="Assets/QHopGenerated";
    [MenuItem("QHOP/Generate presentation scenes")]
    public static void Generate()
    {
        Directory.CreateDirectory(AssetsRoot);
        Directory.CreateDirectory("Assets/Scenes");
        var prefab=AssetDatabase.LoadAssetAtPath<GameObject>("Assets/Resources/HorseGame.fbx");
        if(prefab==null)throw new System.Exception("HorseGame.fbx could not be imported");
        var rig=AssetDatabase.LoadAssetAtPath<HorseRigDefinition>(AssetsRoot+"/HorseRig.asset");
        if(rig==null)
        {
            rig=ScriptableObject.CreateInstance<HorseRigDefinition>();
            AssetDatabase.CreateAsset(rig,AssetsRoot+"/HorseRig.asset");
        }
        // Training creates this definition from code. Keep a previously
        // generated presentation asset on the exact same physical baseline.
        var validated=ScriptableObject.CreateInstance<HorseRigDefinition>();
        rig.referenceShoulderHeight=validated.referenceShoulderHeight;
        rig.rootDrive=validated.rootDrive;
        rig.jointSpring=validated.jointSpring;
        rig.jointDamper=validated.jointDamper;
        rig.passiveSpring=validated.passiveSpring;
        rig.passiveDamper=validated.passiveDamper;
        rig.swingLimit=validated.swingLimit;
        rig.sideLimit=validated.sideLimit;
        rig.hingeLimit=validated.hingeLimit;
        rig.hoofLimit=validated.hoofLimit;
        EditorUtility.SetDirty(rig);
        Object.DestroyImmediate(validated);
        var shader=Shader.Find("QHOP/SoftToon");
        if(shader==null)throw new System.Exception("SoftToon shader is missing");
        var horseMaterial=MaterialAt(AssetsRoot+"/HorseToon.mat",shader,new Color(.72f,.49f,.29f));
        var trackMaterial=MaterialAt(AssetsRoot+"/Track.mat",shader,new Color(.68f,.72f,.52f));
        var railMaterial=MaterialAt(AssetsRoot+"/Rail.mat",shader,new Color(.87f,.81f,.64f));
        MakeScene("AttributeLab",true,prefab,rig,horseMaterial,trackMaterial,railMaterial);
        MakeScene("DerbyStraight",false,prefab,rig,horseMaterial,trackMaterial,railMaterial);
        EditorBuildSettings.scenes=new[]
        {
            new EditorBuildSettingsScene("Assets/Scenes/DerbyStraight.unity",true),
            new EditorBuildSettingsScene("Assets/Scenes/AttributeLab.unity",true)
        };
        EditorSceneManager.OpenScene("Assets/Scenes/DerbyStraight.unity");
        AssetDatabase.SaveAssets();
        Debug.Log("QHOP_PRESENTATION_SCENES_READY");
    }

    static Material MaterialAt(string path,Shader shader,Color color)
    {
        var material=AssetDatabase.LoadAssetAtPath<Material>(path);
        if(material==null)
        {
            material=new Material(shader);AssetDatabase.CreateAsset(material,path);
        }
        material.shader=shader;
        material.SetColor("_BaseColor",color);
        return material;
    }

    static void MakeScene(string name,bool experiment,GameObject prefab,HorseRigDefinition rig,
        Material horse,Material track,Material rail)
    {
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
        var sun=new GameObject("Warm Main Light").AddComponent<Light>();
        sun.type=LightType.Directional;sun.intensity=1.1f;sun.color=new Color(1f,.89f,.74f);
        sun.transform.rotation=Quaternion.Euler(45f,-28f,0f);sun.shadows=LightShadows.Soft;
        var fill=new GameObject("Cool Fill Light").AddComponent<Light>();
        fill.type=LightType.Directional;fill.intensity=.35f;fill.color=new Color(.58f,.74f,1f);
        fill.transform.rotation=Quaternion.Euler(30f,145f,0f);
        RenderSettings.ambientLight=new Color(.65f,.72f,.77f);
        RenderSettings.fog=true;RenderSettings.fogColor=new Color(.74f,.85f,.84f);
        RenderSettings.fogDensity=.005f;
        var cam=new GameObject("Main Camera",typeof(Camera),typeof(AudioListener));
        cam.tag="MainCamera";cam.transform.position=new Vector3(12,8,-12);
        cam.transform.LookAt(new Vector3(0,1,3));
        var camera=cam.GetComponent<Camera>();camera.clearFlags=CameraClearFlags.SolidColor;
        camera.backgroundColor=new Color(.68f,.84f,.9f);camera.fieldOfView=48f;
        camera.farClipPlane=300f;

        float trackLength=experiment?32f:120f;
        float centerZ=experiment?0f:50f;
        var ground=Primitive("Track Ground",PrimitiveType.Cube,new Vector3(0,-.12f,centerZ),
            new Vector3(12f,.2f,trackLength),track);
        ground.AddComponent<HorseGroundSurface>();
        var material=new PhysicsMaterial("Track Grip") { staticFriction=1f,dynamicFriction=1f,
            frictionCombine=PhysicsMaterialCombine.Multiply,bounciness=0f };
        ground.GetComponent<Collider>().material=material;
        for(int i=0;i<5;i++)
        {
            float x=-6f+i*3f;
            var line=Primitive("Lane Stripe",PrimitiveType.Cube,new Vector3(x,.002f,centerZ),
                new Vector3(.045f,.005f,trackLength),rail);
            Object.DestroyImmediate(line.GetComponent<Collider>());
        }
        for(int side=-1;side<=1;side+=2)
        {
            Primitive("Outside Rail",PrimitiveType.Cube,new Vector3(side*6.8f,.55f,centerZ),
                new Vector3(.12f,.12f,trackLength),rail);
            for(int k=0;k<=(experiment?4:15);k++)
            {
                float z=centerZ-trackLength*.5f+k*8f;
                Primitive("Rail Post",PrimitiveType.Cube,new Vector3(side*6.8f,.3f,z),
                    new Vector3(.14f,.6f,.14f),rail);
            }
        }
        if(!experiment)
        {
            var finish=Primitive("Finish Stripe",PrimitiveType.Cube,new Vector3(0,.005f,100f),
                new Vector3(12f,.015f,.16f),rail);
            Object.DestroyImmediate(finish.GetComponent<Collider>());
        }
        for(int i=0;i<20;i++)
        {
            float z=centerZ-trackLength*.5f+i*(trackLength/20f);
            float x=i%2==0?-12f:12f;
            var hill=Primitive("Low Poly Hill",PrimitiveType.Cube,new Vector3(x,-1.5f,z),
                new Vector3(8f,3.2f,9f),track);
            hill.transform.rotation=Quaternion.Euler(0f,i*37f,9f);
        }
        var arena=new GameObject("QHOP Arena");
        arena.AddComponent<HorseTelemetryRecorder>();
        var reset=arena.AddComponent<ArenaResetCoordinator>();
        var controller=arena.AddComponent<ArenaController>();
        reset.arena=controller;
        controller.horsePrefab=prefab;controller.rig=rig;controller.horseMaterial=horse;
        controller.experimentScene=experiment;controller.viewCamera=camera;
        arena.AddComponent<ArenaEffects>();
        EditorSceneManager.SaveScene(SceneManager.GetActiveScene(),"Assets/Scenes/"+name+".unity");
    }

    static GameObject Primitive(string name,PrimitiveType type,Vector3 pos,Vector3 scale,Material material)
    {
        var go=GameObject.CreatePrimitive(type);go.name=name;
        go.transform.position=pos;go.transform.localScale=scale;
        go.GetComponent<Renderer>().sharedMaterial=material;
        return go;
    }

    [MenuItem("QHOP/Build Windows")]
    public static void BuildWindows()
    {
        Generate();
        Directory.CreateDirectory("Builds/Windows");
        var options=new BuildPlayerOptions
        {
            scenes=new[]{"Assets/Scenes/DerbyStraight.unity","Assets/Scenes/AttributeLab.unity"},
            locationPathName="Builds/Windows/QHOP Derby.exe",
            target=BuildTarget.StandaloneWindows64,
            options=BuildOptions.None
        };
        var result=BuildPipeline.BuildPlayer(options);
        if(result.summary.result!=UnityEditor.Build.Reporting.BuildResult.Succeeded)
            throw new System.Exception("Windows build failed: "+result.summary.result);
        Debug.Log("QHOP_WINDOWS_BUILD_READY");
    }
}
