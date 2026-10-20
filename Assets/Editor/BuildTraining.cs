using System;
using System.IO;
using System.Security.Cryptography;
using QHop;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

public static class BuildTraining
{
    [Serializable]
    sealed class TrainingContract
    {
        public int schema_version,observation_size,action_size;
        public string physics_fingerprint,reward_hash,course_hash,
            executable_sha256,assembly_sha256,resources_sha256,scene_sha256;
        public string lidar_contract,rig_contract,model_architecture;
    }

    const string ScenePath="Assets/Scenes/QHopTraining.unity";

    [MenuItem("QHOP/Generate training scene")]
    public static void Generate()
    {
        EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Single);
        EditorSceneManager.SaveScene(SceneManager.GetActiveScene(),ScenePath);
        AssetDatabase.Refresh();
        Debug.Log("QHOP_TRAINING_SCENE_READY");
    }

    [MenuItem("QHOP/Build training environment")]
    public static void Build()
    {
        const string debugMaterial="Assets/Resources/QHopLidarDebug.mat";
        if(AssetDatabase.LoadAssetAtPath<Material>(debugMaterial)==null)
            AssetDatabase.CreateAsset(new Material(Shader.Find("Sprites/Default")),debugMaterial);
        Generate();
        System.IO.Directory.CreateDirectory("Builds/Training");
        var report=BuildPipeline.BuildPlayer(new BuildPlayerOptions
        {
            scenes=new[]{ScenePath},
            locationPathName="Builds/Training/QHopTraining.exe",
            target=BuildTarget.StandaloneWindows64,
            options=BuildOptions.None
        });
        if(report.summary.result!=BuildResult.Succeeded)
            throw new System.InvalidOperationException("Training build failed: "+report.summary.result);
        const string player="Builds/Training/QHopTraining.exe";
        string Digest(string path)
        {
            using(var sha=SHA256.Create())
            using(var stream=File.OpenRead(path))
                return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-","").ToLowerInvariant();
        }
        string TextDigest(string value)
        {
            using(var sha=SHA256.Create())
                return BitConverter.ToString(sha.ComputeHash(
                    System.Text.Encoding.UTF8.GetBytes(value))).Replace("-","").ToLowerInvariant();
        }
        File.WriteAllText("Builds/Training/QHopTraining.contract.json",JsonUtility.ToJson(
            new TrainingContract
            {
                schema_version=HorseTrainingObservation.Version,
                observation_size=HorseTrainingObservation.Size,
                action_size=HorseAction.Size,
                lidar_contract=HorseLidar.Contract,
                rig_contract=HorseRigDefinition.Contract,
                model_architecture="grouped-mlp-shared-lidar-gru-r14",
                physics_fingerprint=HorseTrainingObservation.PhysicsFingerprint,
                reward_hash=TextDigest(HorseTrainingReward.Contract),
                course_hash=TextDigest(HorseTrainingArea.CourseContract),
                executable_sha256=Digest(player),
                assembly_sha256=Digest("Builds/Training/QHopTraining_Data/Managed/Assembly-CSharp.dll"),
                resources_sha256=Digest("Builds/Training/QHopTraining_Data/resources.assets"),
                scene_sha256=Digest("Builds/Training/QHopTraining_Data/level0")
            },true));
    }
}
