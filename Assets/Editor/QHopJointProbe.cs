using System;
using System.Collections.Generic;
using QHop;
using UnityEditor;
using UnityEngine;

public static class QHopJointProbe
{
    public static void ValidateAndBuild(){RunAll();BuildTraining.Build();}
    public static void RunAll()
    {
        Run();RunDynamic();QHopMorphProbe.Run();QHopGaitProbe.Run();QHopRewardProbe.Run();
        Debug.Log("QHOP_ALL_PROBES_PASS");
    }
    static void Check(bool value,string message)
    {if(!value)throw new InvalidOperationException("QHOP_JOINT_PROBE_FAIL: "+message);}
    static HorseAction Baseline()
    {var a=new HorseAction();for(int i=16;i<28;i++)a.values[i]=.7f;return a;}
    [MenuItem("QHOP/Probes/Joint geometry")]
    public static void Run()
    {
        var prefab=Resources.Load<GameObject>("HorseGame");
        var rig=Resources.Load<HorseRigDefinition>("HorseRig") ?? AssetDatabase.LoadAssetAtPath<HorseRigDefinition>("Assets/QHopGenerated/HorseRig.asset");
        var root=new GameObject("JointActionProbe");var material=new Material(Shader.Find("Standard"));
        try
        {
            var horse=root.AddComponent<HorseActor>();horse.Initialize(0,new HorseConfig(),prefab,rig,material,false,false);
            Check(horse.bodies.Count==19 && horse.allJoints.Count==18 && horse.LegJoints.Count==16,"body/joint count");
            var baseline=Baseline();horse.ApplyAction(baseline);
            var targets=new Quaternion[18];var forces=new float[18];
            for(int i=0;i<18;i++){targets[i]=horse.allJoints[i].targetRotation;forces[i]=horse.allJoints[i].slerpDrive.maximumForce;}
            for(int channel=0;channel<28;channel++)
            {
                var action=Baseline();action.values[channel]=channel<16?.5f:-.5f;horse.ApplyAction(action);
                int changedTargets=0,changedForces=0;
                for(int i=0;i<18;i++)
                {if(Quaternion.Angle(targets[i],horse.allJoints[i].targetRotation)>1f)changedTargets++;
                 if(Mathf.Abs(forces[i]-horse.allJoints[i].slerpDrive.maximumForce)>1f)changedForces++;}
                Check(channel<16?changedTargets==1 && changedForces==0:changedTargets==0 && changedForces==1,"channel "+channel);
            }
            for(int i=0;i<16;i++)
            {
                var d=horse.LegJoints[i];float a=-d.BodySign*d.Min,b=-d.BodySign*d.Max;
                Check(Mathf.Abs(d.Joint.lowAngularXLimit.limit-Mathf.Min(a,b))<.01f &&
                    Mathf.Abs(d.Joint.highAngularXLimit.limit-Mathf.Max(a,b))<.01f,"coordinate conversion "+i);
                Check(Mathf.Abs(d.Joint.angularYLimit.limit-d.Twist)<.01f && Mathf.Abs(d.Joint.angularZLimit.limit-d.Side)<.01f,"side/twist "+i);
                Check(Mathf.Abs(d.Measured)<.01f && d.ActionAngle(0f)==0f,"rest angle "+i);
                var mirror=rig.Describe(i/4%2==0?i/4+1:i/4-1,i%4);
                Check(d.Min==mirror.Min && d.Max==mirror.Max && d.BodySign==mirror.BodySign,"mirror "+i);
            }
            // Check that visible bones bind to independent physical bodies, not a single thigh.
            var followers=typeof(HorseActor).GetField("follow",System.Reflection.BindingFlags.Instance|System.Reflection.BindingFlags.NonPublic).GetValue(horse) as System.Collections.IList;
            Check(followers!=null && followers.Count>16,"visible bone bindings missing");
            Debug.Log("QHOP_JOINT_PROBE_PASS: loaded r13 rig, 28 channels, canonical limits and visible bindings");
        }
        finally{UnityEngine.Object.DestroyImmediate(root);UnityEngine.Object.DestroyImmediate(material);}
    }
    public static void RunDynamic()
    {
        var mode=Physics.simulationMode;Physics.simulationMode=SimulationMode.Script;
        var prefab=Resources.Load<GameObject>("HorseGame");
        var rig=AssetDatabase.LoadAssetAtPath<HorseRigDefinition>("Assets/QHopGenerated/HorseRig.asset");
        var material=new Material(Shader.Find("Standard"));int probes=0;
        try
        {
            for(int leg=0;leg<4;leg++)for(int part=0;part<3;part++)
            foreach(float input in new[]{-.1f,.1f,-.5f,.5f,-1f,1f,-1.2f,1.2f})
            {
                var root=new GameObject("DynamicJointProbe");
                try
                {
                    var horse=root.AddComponent<HorseActor>();horse.Initialize(0,new HorseConfig(),prefab,rig,material,false,Mathf.Abs(input)!=.5f);
                    foreach(var body in horse.bodies){body.isKinematic=true;body.useGravity=false;foreach(var c in body.GetComponents<Collider>())c.enabled=false;}
                    var d=horse.LegJoints[leg*4+part];var joint=d.Joint;
                    horse.ApplyAction(Baseline());float target=d.ActionAngle(input);d.Drive(target);
                    foreach(var other in horse.allJoints)if(other!=joint)
                    {other.xMotion=other.yMotion=other.zMotion=ConfigurableJointMotion.Free;
                     other.angularXMotion=other.angularYMotion=other.angularZMotion=ConfigurableJointMotion.Free;other.slerpDrive=new JointDrive();}
                    joint.GetComponent<Rigidbody>().isKinematic=false;Physics.SyncTransforms();
                    int exceeded=0;var errors=new List<float>();
                    float expected=Mathf.Clamp(target,d.Min,d.Max);
                    for(int tick=0;tick<200;tick++)
                    {
                        Physics.Simulate(.01f);float measured=d.Measured;
                        if(measured<d.Min-3f || measured>d.Max+3f)exceeded++;
                        if(tick>=150)errors.Add(Mathf.Abs(measured-expected));
                    }
                    errors.Sort();Check(exceeded<=2,"hard limit overshoot leg="+leg+" part="+part+" input="+input);
                    Check(errors[25]<=3f && errors[47]<=5f,"target unreachable leg="+leg+" part="+part+" input="+input+" expected="+expected+" measured="+d.Measured);
                    if(Mathf.Abs(input)==.5f)
                    {
                        var flags=System.Reflection.BindingFlags.Instance|System.Reflection.BindingFlags.NonPublic;
                        typeof(HorseActor).GetMethod("LateUpdate",flags).Invoke(horse,null);
                        var followers=(System.Collections.IList)typeof(HorseActor).GetField("follow",flags).GetValue(horse);
                        bool found=false;
                        foreach(var item in followers)
                        {
                            var type=item.GetType();var bone=(Transform)type.GetField("bone").GetValue(item);
                            var body=(Rigidbody)type.GetField("body").GetValue(item);
                            if(bone.name!=rig.Legs[leg][part])continue;
                            Check(body==joint.GetComponent<Rigidbody>(),"visible segment bound to wrong body");
                            var offset=(Quaternion)type.GetField("rotationOffset").GetValue(item);
                            Check(Quaternion.Angle(bone.rotation,body.rotation*offset)<.01f,"visible segment did not follow physical joint");
                            found=true;
                        }
                        Check(found,"missing visible middle/lower segment");
                    }
                    Debug.Log("QHOP_DYNAMIC_JOINT leg="+leg+" part="+part+" input="+input+" expected="+expected+" measured="+d.Measured+" median_error="+errors[25]);
                    probes++;
                }
                finally{UnityEngine.Object.DestroyImmediate(root);}
            }
            Debug.Log("QHOP_DYNAMIC_JOINT_PROBE_PASS: "+probes+" target-reach and over-limit probes");
        }
        finally{Physics.simulationMode=mode;UnityEngine.Object.DestroyImmediate(material);}
    }
}
