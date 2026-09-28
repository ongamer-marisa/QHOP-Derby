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
    }

    // Run after exporting a temporary model to Assets/Editor/TestAssets/QHopSmoke.onnx.
    // The saved fixture is compared against ONNX Runtime by the Python verifier.
    public static void Run()
    {
        var asset=AssetDatabase.LoadAssetAtPath<ModelAsset>(
            "Assets/Editor/TestAssets/QHopSmoke.onnx");
        if(asset==null)throw new InvalidOperationException("ONNX model import failed");
        float[] observations=new float[2*HorseTrainingObservation.Size];
        for(int i=0;i<observations.Length;i++)
            observations[i]=Mathf.Sin(i*.0137f)*.5f;
        using(var worker=new Worker(ModelLoader.Load(asset),BackendType.CPU))
        using(var input=new Tensor<float>(
            new TensorShape(2,HorseTrainingObservation.Size),observations))
        {
            worker.Schedule(input);
            var output=worker.PeekOutput() as Tensor<float>;
            if(output==null)throw new InvalidOperationException("ONNX output is not float");
            var actions=output.DownloadToArray();
            if(actions.Length!=2*HorseAction.Size)
                throw new InvalidOperationException("ONNX action dimension mismatch");
            Directory.CreateDirectory("Temp");
            File.WriteAllText("Temp/qhop-inference-probe.json",JsonUtility.ToJson(
                new Fixture{observation=observations,action=actions}));
            Debug.Log("QHOP_INFERENCE_PROBE_OK");
        }
    }
}
