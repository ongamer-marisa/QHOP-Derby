using System;
using QHop;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEditor.SceneManagement;
using UnityEditor;

// Run with Unity -batchmode -nographics -executeMethod QHopMorphProbe.Run.
public static class QHopMorphProbe
{
    static void Check(bool value,string message)
    {
        if(!value)throw new InvalidOperationException("QHOP_MORPH_PROBE_FAIL: "+message);
    }

    public static void Run()
    {
        var prefab=Resources.Load<GameObject>("HorseGame");
        Check(prefab!=null,"HorseGame prefab missing");
        var rig=AssetDatabase.LoadAssetAtPath<HorseRigDefinition>("Assets/QHopGenerated/HorseRig.asset");
        Check(rig!=null,"loaded rig missing");
        var shader=Shader.Find("Standard");
        Check(shader!=null,"Standard shader missing");
        var material=new Material(shader);
        var oldScene=SceneManager.GetActiveScene();
        var mode=Physics.simulationMode;float oldDt=Time.fixedDeltaTime;
        var scene=EditorSceneManager.NewPreviewScene();
        Physics.simulationMode=SimulationMode.Script;Time.fixedDeltaTime=.01f;
        try
        {
            var physics=scene.GetPhysicsScene();
            var ground=GameObject.CreatePrimitive(PrimitiveType.Cube);
            SceneManager.MoveGameObjectToScene(ground,scene);ground.transform.position=new Vector3(0,-.12f,0);
            ground.transform.localScale=new Vector3(20f,.2f,20f);ground.AddComponent<HorseGroundSurface>();
            const float tolerance=.001f;
            float[] scales={.85f,1f,1.15f};
            int checkedMorphologies=0;
            foreach(float bulk in scales)
            foreach(float front in scales)
            foreach(float back in scales)
            foreach(float length in scales)
            {
                var config=new HorseConfig
                {
                    bulk=bulk,frontLength=front,backLength=back,bodyLength=length,
                    frontStrength=Mathf.Lerp(.55f,1.45f,(front-.85f)/.3f),
                    backStrength=Mathf.Lerp(.55f,1.45f,(back-.85f)/.3f),
                    massKg=Mathf.Clamp(450f*bulk*bulk*length,250f,750f)
                };
                var root=new GameObject("MorphProbe_"+checkedMorphologies);
                SceneManager.MoveGameObjectToScene(root,scene);
                try
                {
                    var horse=root.AddComponent<HorseActor>();
                    horse.Initialize(checkedMorphologies,config,prefab,rig,material,true,true);
                    Check(horse.bodies.Count==19,"rigidbody count "+checkedMorphologies);
                    Check(horse.allJoints.Count==18,"joint count "+checkedMorphologies);
                    float totalMass=0f;
                    foreach(var body in horse.bodies)
                    {
                        Check(body.mass>0f && !float.IsNaN(body.mass),
                              "invalid mass "+checkedMorphologies);
                        totalMass+=body.mass;
                    }
                    Check(Mathf.Abs(totalMass-config.massKg)<config.massKg*tolerance,
                          "total mass "+checkedMorphologies);
                    float lowest=float.PositiveInfinity;
                    foreach(var chain in rig.Legs)
                    {
                        Rigidbody hoof=null;
                        foreach(var body in horse.bodies)if(body.name==chain[3])hoof=body;
                        Check(hoof!=null,"hoof missing "+checkedMorphologies);
                        lowest=Mathf.Min(lowest,hoof.GetComponent<Collider>().bounds.min.y);
                    }
                    Check(Mathf.Abs(lowest+.02f)<tolerance,
                          "hoof does not meet ground "+checkedMorphologies+": "+lowest);
                    horse.ExternalActionMode=true;
                    var action=new HorseAction();for(int channel=16;channel<28;channel++)action.values[channel]=.7f;
                    horse.ApplyAction(action);Physics.SyncTransforms();
                    for(int tick=0;tick<100;tick++)
                    {
                        physics.Simulate(.01f);horse.FixedStep(new[]{horse},true);
                        foreach(var body in horse.bodies)
                            Check(float.IsFinite(body.position.sqrMagnitude),"nonfinite loaded morphology "+checkedMorphologies);
                    }
                    Check(horse.Condition!=HorseCondition.Fallen,"loaded morphology fell "+checkedMorphologies);
                    checkedMorphologies++;
                }
                finally{UnityEngine.Object.DestroyImmediate(root);}
            }
            UnityEngine.Object.DestroyImmediate(ground);
            for(int seed=0;seed<1000;seed++)
            {
                var config=HorseGenetics.Generate(seed);
                Check(config.bulk>=.85f && config.bulk<=1.15f &&
                      config.frontLength>=.85f && config.frontLength<=1.15f &&
                      config.backLength>=.85f && config.backLength<=1.15f &&
                      config.bodyLength>=.85f && config.bodyLength<=1.15f &&
                      config.massKg>=250f && config.massKg<=750f &&
                      Mathf.Abs(config.frontLength-config.backLength)<=.2f+
                      tolerance,"generator range "+seed);
            }
            Debug.Log("QHOP_MORPH_PROBE_PASS: 81 loaded rigs, 1000 genetic samples, hoof y=-0.02");
        }
        finally
        {
            EditorSceneManager.ClosePreviewScene(scene);
            Physics.simulationMode=mode;Time.fixedDeltaTime=oldDt;
            UnityEngine.Object.DestroyImmediate(material);
        }
    }
}
