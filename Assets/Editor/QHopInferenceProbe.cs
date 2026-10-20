using System;
using System.IO;
using QHop;
using Unity.InferenceEngine;
using UnityEditor;
using UnityEngine;

public static class QHopInferenceProbe
{
    [Serializable]
    sealed class Fixture
    {
        public float[] observation;
        public float[] action;
        public float[] hidden;
    }

    // Run after exporting a temporary model to Assets/Editor/TestAssets/QHopSmoke.onnx.
    // The saved fixture is compared against ONNX Runtime by the Python verifier.
    public static void Run()
    {
        var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>(
            "Assets/Editor/TestAssets/QHopSmoke.onnx");
        var visionAsset=AssetDatabase.LoadAssetAtPath<ModelAsset>("Assets/Editor/TestAssets/QHopSmokeVision.onnx");
        if(asset==null)throw new InvalidOperationException("ONNX model import failed");
        float[] observations=new float[2*HorseTrainingObservation.Size];
        for(int i=0;i<observations.Length;i++)
            observations[i]=Mathf.Sin(i*.0137f)*.5f;
        using(var worker=new Worker(ModelLoader.Load(asset),BackendType.CPU))
        using(var vision=new Worker(ModelLoader.Load(visionAsset),BackendType.CPU))
        {
            var hidden=new float[256];var records=new System.Collections.Generic.List<Fixture>();
            for(int step=0;step<20;step++)
            {
                if(step==12)Array.Clear(hidden,0,hidden.Length); // recovery / episode reset
                for(int i=0;i<observations.Length;i++)observations[i]=Mathf.Sin(i*.0137f+step*.17f)*.5f;
                if(step%4==0)
                {
                    var scan=new float[2*HorseLidar.Features];
                    for(int horse=0;horse<2;horse++)Array.Copy(observations,horse*HorseTrainingObservation.Size+HorseTrainingObservation.BaseSize,scan,horse*HorseLidar.Features,HorseLidar.Features);
                    using(var x=new Tensor<float>(new TensorShape(2,HorseLidar.Features),scan))
                    using(var h=new Tensor<float>(new TensorShape(2,128),hidden))
                    {vision.SetInput("scan",x);vision.SetInput("hidden",h);vision.Schedule();hidden=(vision.PeekOutput() as Tensor<float>).DownloadToArray();}
                }
                using(var input=new Tensor<float>(new TensorShape(2,HorseTrainingObservation.Size),observations))
                using(var memory=new Tensor<float>(new TensorShape(2,128),hidden))
                {
                    worker.SetInput("observation",input);worker.SetInput("hidden",memory);worker.Schedule();
                    var actions=(worker.PeekOutput() as Tensor<float>).DownloadToArray();
                    if(actions.Length!=2*HorseAction.Size)throw new InvalidOperationException("ONNX action dimension mismatch");
                    records.Add(new Fixture{observation=(float[])observations.Clone(),action=actions,hidden=(float[])hidden.Clone()});
                }
            }
            Directory.CreateDirectory("Builds/BuildLogs");
            File.WriteAllText("Builds/BuildLogs/qhop-inference-probe.json",JsonUtility.ToJson(new Sequence{steps=records.ToArray()}));
            Debug.Log("QHOP_INFERENCE_PROBE_OK");
        }
    }
    [Serializable]sealed class Sequence {public Fixture[] steps;}
}
