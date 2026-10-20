using System;
using System.Collections.Generic;
using System.IO;
using QHop;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.InferenceEngine;

public static class QHopR14Probe
{
    public static void ValidateAndBuild()
    {
        Contracts();QHopJointProbe.RunDynamic();QHopInferenceProbe.Run();HostMemory();Standing();BuildTraining.Build();
        Debug.Log("R14_VALIDATE_AND_BUILD_PASS");
    }
    public static void HostMemory()
    {
        var scene=EditorSceneManager.NewPreviewScene();var root=new GameObject("HostMemoryProbe");SceneManager.MoveGameObjectToScene(root,scene);
        var h=root.AddComponent<HorseActor>();var mat=new Material(Shader.Find("Standard"));
        h.Initialize(0,new HorseConfig(),Resources.Load<GameObject>("HorseGame"),AssetDatabase.LoadAssetAtPath<HorseRigDefinition>("Assets/QHopGenerated/HorseRig.asset"),mat,true,true);
        var race=new HorseRaceSession(new[]{h},100,90);var runner=root.AddComponent<HorsePolicyRunner>();
        Check(runner.Load(new[]{h},race,AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/Editor/TestAssets/QHopSmoke.onnx"),
            AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/Editor/TestAssets/QHopSmokeMetadata.json"),
            AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/Editor/TestAssets/QHopSmokeVision.onnx")),"host models failed to load");
        var flags=System.Reflection.BindingFlags.Instance|System.Reflection.BindingFlags.NonPublic;
        var clock=typeof(HorsePolicyRunner).GetField("lastTime",flags);
        var updates=(List<int>)typeof(HorsePolicyRunner).GetField("visionUpdates",flags).GetValue(runner);
        for(int decision=0;decision<12;decision++)
        {
            clock.SetValue(runner,float.NegativeInfinity);
            Check(runner.Controller(0).Decide(null).IsValid(),"host invalid action");
            Check(updates[0]==1+decision/4,"host repeated frozen GRU step");
            for(int tick=0;tick<5;tick++)h.Lidar.Step(h);
        }
        typeof(HorseActor).GetProperty("Condition").SetValue(h,HorseCondition.Recovering);
        clock.SetValue(runner,float.NegativeInfinity);runner.Controller(0).Decide(null);
        typeof(HorseActor).GetProperty("Condition").SetValue(h,HorseCondition.Standing);h.Lidar.Reset(h);
        clock.SetValue(runner,float.NegativeInfinity);runner.Controller(0).Decide(null);
        Check(updates[0]==1,"host recovery did not reset memory");
        UnityEngine.Object.DestroyImmediate(root);UnityEngine.Object.DestroyImmediate(mat);EditorSceneManager.ClosePreviewScene(scene);
        Debug.Log("R14_HOST_MEMORY_PROBE_PASS");
    }
    static void Check(bool ok,string message){if(!ok)throw new Exception("R14_PROBE_FAIL: "+message);}
    static float Setting(string key,float fallback)
    {
        foreach(string arg in Environment.GetCommandLineArgs())
            if(arg.StartsWith("--"+key+"=") && float.TryParse(arg.Substring(key.Length+3),out float v))return v;
        return fallback;
    }
    public static void Contracts()
    {
        Physics.simulationMode=SimulationMode.Script;Time.fixedDeltaTime=.01f;
        var scene=EditorSceneManager.NewPreviewScene();
        var prefab=Resources.Load<GameObject>("HorseGame");
        var rig=AssetDatabase.LoadAssetAtPath<HorseRigDefinition>("Assets/QHopGenerated/HorseRig.asset");
        var mat=new Material(Shader.Find("Standard"));var root=new GameObject("R14Contracts");
        SceneManager.MoveGameObjectToScene(root,scene);var h=root.AddComponent<HorseActor>();h.Initialize(0,new HorseConfig(),prefab,rig,mat,false,true);
        Check(h.Lidar.Sequence==1 && h.Lidar.Data.Length==179,"scan initialization");
        for(int i=0;i<57;i++)
        {
            Check(h.Lidar.Data[i*3]==(i<45?25:10) && h.Lidar.Data[i*3+2]==0,"self filtering / miss range");
            int direction=i/3;float angle=direction<15?-70+direction*10:new[]{90f,-90f,135f,-135f}[direction-15];
            var expected=Quaternion.AngleAxis(angle,Vector3.up)*Vector3.forward;
            Check(Vector3.Distance((h.Lidar.Ends[i]-h.Lidar.Origins[i]).normalized,expected)<1e-5,"ray direction "+i);
            float level=new[]{.1f,.6f,1.1f}[i%3];
            Check(Mathf.Abs(h.Lidar.Origins[i].y-h.Torso.position.y-(level-1)*h.RestTorsoClearance)<1e-5,"ray height "+i);
        }
        Vector3 center=h.Lidar.Origins[22];
        var trigger=GameObject.CreatePrimitive(PrimitiveType.Cube);SceneManager.MoveGameObjectToScene(trigger,scene);
        trigger.transform.position=center+Vector3.forward*2;trigger.GetComponent<Collider>().isTrigger=true;
        var wall=GameObject.CreatePrimitive(PrimitiveType.Cube);SceneManager.MoveGameObjectToScene(wall,scene);wall.transform.position=center+Vector3.forward*5;
        Physics.SyncTransforms();h.Lidar.Scan(h);
        Check(Mathf.Abs(h.Lidar.Data[22*3]-4.5f)<.01f && h.Lidar.Data[22*3+2]==1,"closest wall excluding trigger");
        for(int i=0;i<70;i++)
        {
            var crowded=GameObject.CreatePrimitive(PrimitiveType.Cube);SceneManager.MoveGameObjectToScene(crowded,scene);
            crowded.transform.position=center+Vector3.forward*(6+i*.01f);crowded.GetComponent<MeshRenderer>().enabled=false;
        }
        Physics.SyncTransforms();h.Lidar.Scan(h);
        Check(Mathf.Abs(h.Lidar.Data[22*3]-4.5f)<.01f,"truncated hit buffer lost nearest wall");
        int sequence=h.Lidar.Sequence;
        for(int i=0;i<19;i++)h.Lidar.Step(h);Check(h.Lidar.Sequence==sequence,"scan repeated early");
        h.Lidar.Step(h);Check(h.Lidar.Sequence==sequence+1 && Mathf.Abs(h.Lidar.Data[176]-.2f)<1e-6,"5 Hz cadence");
        h.Torso.rotation=Quaternion.Euler(25,0,30);h.Lidar.Scan(h);
        Check(Mathf.Abs(h.Lidar.Ends[22].y-h.Lidar.Origins[22].y)<1e-5,"scanner tilted with body");
        // A same-frame fallen finish must become exactly one fall DNF.
        var race=new HorseRaceSession(new[]{h},.01f,20,false,true);
        typeof(HorseActor).GetProperty("Condition").SetValue(h,HorseCondition.Fallen);h.Torso.position+=Vector3.forward*10;
        race.Step(.01f);Check(race.Complete && race.Results[0].dnf && race.Results[0].reason=="fall" && !race.Results[0].finished,"fall finish priority");
        Check(race.Results[0].falls==1,"fall count");
        var reward=new HorseTrainingReward(2,1,20);float value=reward.Step(h,race.Results[0],.01f,1,.01f);
        Check(reward.WeightedFall==-2 && reward.WeightedDnf==-5 && Mathf.Abs(reward.WeightedTime+.0005f)<1e-6,"fall remaining penalty");
        race.Step(.01f);Check(race.Results[0].falls==1,"duplicate terminal");
        var root2=new GameObject("Opponent");SceneManager.MoveGameObjectToScene(root2,scene);
        var second=root2.AddComponent<HorseActor>();second.Initialize(1,new HorseConfig(),prefab,rig,mat,false,true);
        var root3=new GameObject("FallingOpponent");SceneManager.MoveGameObjectToScene(root3,scene);
        var third=root3.AddComponent<HorseActor>();third.Initialize(2,new HorseConfig(),prefab,rig,mat,false,true);
        var multi=new HorseRaceSession(new[]{second,third},2,20,false,true);
        typeof(HorseActor).GetProperty("Condition").SetValue(third,HorseCondition.Fallen);multi.Step(.01f);
        Check(!multi.Complete && multi.Results[1].dnf && !multi.Results[0].dnf,"one fall ended other horse");
        second.Torso.position+=Vector3.forward*3;second.Torso.linearVelocity=Vector3.forward;multi.Step(.01f);
        Check(multi.Complete && multi.Results[0].finished && multi.Results[0].rank==1,"remaining horse cannot finish");
        UnityEngine.Object.DestroyImmediate(root2);UnityEngine.Object.DestroyImmediate(root3);
        var floor=GameObject.CreatePrimitive(PrimitiveType.Cube);SceneManager.MoveGameObjectToScene(floor,scene);
        floor.transform.position=new Vector3(0,-.12f,0);floor.transform.localScale=new Vector3(30,.2f,30);floor.AddComponent<HorseGroundSurface>();
        var recoveryRoot=new GameObject("GameRecovery");SceneManager.MoveGameObjectToScene(recoveryRoot,scene);
        var recovering=recoveryRoot.AddComponent<HorseActor>();recovering.Initialize(0,new HorseConfig(),prefab,rig,mat,true,true);
        typeof(HorseActor).GetProperty("Condition").SetValue(recovering,HorseCondition.Fallen);
        typeof(HorseActor).GetField("conditionTimer",System.Reflection.BindingFlags.Instance|System.Reflection.BindingFlags.NonPublic).SetValue(recovering,3f);
        Physics.SyncTransforms();recovering.FixedStep(new[]{recovering},true);
        Check(recovering.Condition==HorseCondition.Recovering && recovering.Lidar.Sequence==1,"game recovery disabled or stale scan");
        recovering.FixedStep(new[]{recovering},true);Check(recovering.Condition==HorseCondition.Standing,"recovery did not restore dynamics");
        UnityEngine.Object.DestroyImmediate(recoveryRoot);UnityEngine.Object.DestroyImmediate(floor);
        UnityEngine.Object.DestroyImmediate(root);UnityEngine.Object.DestroyImmediate(wall);UnityEngine.Object.DestroyImmediate(trigger);
        EditorSceneManager.ClosePreviewScene(scene);UnityEngine.Object.DestroyImmediate(mat);
        Debug.Log("R14_CONTRACT_PROBE_PASS");
    }
    public static void Standing()
    {
        Physics.simulationMode=SimulationMode.Script;Time.fixedDeltaTime=.01f;
        var scene=EditorSceneManager.NewPreviewScene();
        var physics=scene.GetPhysicsScene();
        var ground=GameObject.CreatePrimitive(PrimitiveType.Cube);SceneManager.MoveGameObjectToScene(ground,scene);
        ground.transform.position=new Vector3(0,-.12f,0);ground.transform.localScale=new Vector3(20,.2f,20);
        ground.AddComponent<HorseGroundSurface>();
        ground.GetComponent<Collider>().sharedMaterial=new PhysicsMaterial(){staticFriction=1,dynamicFriction=1,frictionCombine=PhysicsMaterialCombine.Multiply};
        var prefab=Resources.Load<GameObject>("HorseGame");
        var rig=AssetDatabase.LoadAssetAtPath<HorseRigDefinition>("Assets/QHopGenerated/HorseRig.asset");
        rig=UnityEngine.Object.Instantiate(rig);
        rig.jointSpring=Setting("spring",rig.jointSpring);rig.jointDamper=Setting("damper",rig.jointDamper);
        var mat=new Material(Shader.Find("Standard"));int cases=(int)Setting("cases",1),failed=0;
        for(int index=0;index<cases;index++)
        {
            int n=index+(int)Setting("start-case",0);
            var root=new GameObject("Standing_"+n);SceneManager.MoveGameObjectToScene(root,scene);
            var config=new HorseConfig();
            if(n>=100)
            {
                int k=n-100;float[] sizes={.85f,1f,1.15f};
                config.bulk=sizes[k%3];k/=3;config.frontLength=sizes[k%3];k/=3;
                config.backLength=sizes[k%3];k/=3;config.bodyLength=sizes[k%3];
                config.frontStrength=Mathf.Lerp(.55f,1.45f,(config.frontLength-.85f)/.3f);
                config.backStrength=Mathf.Lerp(.55f,1.45f,(config.backLength-.85f)/.3f);
                config.massKg=Mathf.Clamp(450f*config.bulk*config.bulk*config.bodyLength,250f,750f);
            }
            var h=root.AddComponent<HorseActor>();h.Initialize(n,config,prefab,rig,mat,false,true);h.ExternalActionMode=true;
            var a=new HorseAction();for(int j=16;j<28;j++)a.values[j]=.7f;h.ApplyAction(a);
            var initial=h.Torso.position;float minHeight=1,maxTilt=0,drift=0;bool bad=false;
            int ticks=(int)(Setting("seconds",20)*100);
            for(int t=0;t<ticks;t++)
            {
                if(t%5==0)h.SetExternalAction(a); // Same 20 Hz standing controller as the Player.
                physics.Simulate(.01f);h.FixedStep(new[]{h},true);
                float height=h.Torso.position.y/h.RestTorsoHeight,tilt=Vector3.Angle(h.Torso.transform.up,Vector3.up);
                if(t>=50){minHeight=Mathf.Min(minHeight,height);maxTilt=Mathf.Max(maxTilt,tilt);}
                drift=Vector3.ProjectOnPlane(h.Torso.position-initial,Vector3.up).magnitude;
                if(n==0 && (t%100==0 || h.Condition==HorseCondition.Fallen))
                {
                    var angles=new List<string>();foreach(var d in h.LegJoints)angles.Add(d.Measured.ToString("F2"));
                    Debug.Log("R14_TRACE tick="+t+" height="+height+" tilt="+tilt+" drift="+drift+
                        " contacts="+h.Gait.Contacts+" angles="+string.Join(",",angles)+" velocity="+h.Torso.linearVelocity);
                }
                foreach(var body in h.bodies)if(!float.IsFinite(body.position.sqrMagnitude))bad=true;
                if(h.Condition==HorseCondition.Fallen){bad=true;break;}
            }
            bad|=n<100 && (minHeight<.8f || maxTilt>20 || drift>.5f);
            if(bad)failed++;
            Debug.Log("R14_STANDING case="+n+" failed="+bad+" minHeight="+minHeight+" maxTilt="+maxTilt+" drift="+drift);
            UnityEngine.Object.DestroyImmediate(root);
        }
        Debug.Log("R14_STANDING_SUMMARY cases="+cases+" failures="+failed+" spring="+rig.jointSpring+" damper="+rig.jointDamper);
        Check(failed==0,"standing failures="+failed);
    }
}
