using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

namespace QHop
{
    [Serializable]
    public sealed class HorsePolicyMetadata
    {
        public int schema_version;
        public int observation_size;
        public int action_size;
        public int frame_size;
        public int history;
        public string model_name;
        public bool normalization_embedded;
        public string physics_fingerprint;
        public int vision_features,vision_hidden;
        public string vision_model,lidar_contract,model_architecture,rig_contract;
    }

    public sealed class HorsePolicyRunner : MonoBehaviour
    {
        Worker worker;
        Worker visionWorker;
        readonly List<float[]> visualStates=new List<float[]>();
        readonly List<int> scanIds=new List<int>();
        readonly List<int> visionUpdates=new List<int>();
        readonly List<HorseActor> horses=new List<HorseActor>();
        readonly List<HorseObservationHistory> histories=new List<HorseObservationHistory>();
        readonly List<HorseAction> actions=new List<HorseAction>();
        readonly List<int> lastFalls=new List<int>();
        readonly List<HorseCondition> lastConditions=new List<HorseCondition>();
        HorseRaceSession race;
        float lastTime=float.NegativeInfinity;
        public string ModelName { get; private set; }
        public bool Ready => worker!=null;

        public bool Load(IReadOnlyList<HorseActor> actors,HorseRaceSession session,
            ModelAsset policy=null,TextAsset policyMetadata=null,ModelAsset memory=null)
        {
            DisposeWorker();
            visualStates.Clear();scanIds.Clear();visionUpdates.Clear();
            horses.Clear();histories.Clear();actions.Clear();lastFalls.Clear();lastConditions.Clear();
            var asset=policy??Resources.Load<ModelAsset>("QHopPolicy");
            var json=policyMetadata??Resources.Load<TextAsset>("QHopPolicyMetadata");
            if(asset==null || json==null){Debug.LogWarning("No exported QHOP policy found.");return false;}
            var metadata=JsonUtility.FromJson<HorsePolicyMetadata>(json.text);
            if(metadata==null || metadata.schema_version!=HorseTrainingObservation.Version ||
                metadata.observation_size!=HorseTrainingObservation.Size ||
                metadata.action_size!=HorseAction.Size ||
                metadata.frame_size!=HorseTrainingObservation.ProprioFrame ||
                metadata.history!=HorseTrainingObservation.History || !metadata.normalization_embedded)
            {Debug.LogError("QHOP policy schema mismatch; using standing controller.");return false;}
            if(metadata.physics_fingerprint!=HorseTrainingObservation.PhysicsFingerprint)
            {Debug.LogError("QHOP policy schema mismatch; using standing controller.");return false;}
            if(metadata.vision_features!=HorseLidar.Features || metadata.vision_hidden!=128 ||
                metadata.lidar_contract!=HorseLidar.Contract || metadata.rig_contract!=HorseRigDefinition.Contract ||
                metadata.model_architecture!="grouped-mlp-shared-lidar-gru-r14")
            {Debug.LogError("QHOP vision contract mismatch");return false;}
            var visionAsset=memory??Resources.Load<ModelAsset>(System.IO.Path.GetFileNameWithoutExtension(metadata.vision_model));
            if(visionAsset==null){Debug.LogError("QHOP vision model missing");return false;}
            try{worker=new Worker(ModelLoader.Load(asset),BackendType.CPU);}
            catch(Exception exception){Debug.LogError("QHOP model load failed: "+exception);return false;}
            try{visionWorker=new Worker(ModelLoader.Load(visionAsset),BackendType.CPU);}
            catch(Exception exception){DisposeWorker();Debug.LogError("QHOP vision load failed: "+exception);return false;}
            race=session;ModelName=metadata.model_name;
            foreach(var actor in actors)
            {
                horses.Add(actor);histories.Add(new HorseObservationHistory());
                actions.Add(new HorseAction());lastFalls.Add(0);
                lastConditions.Add(actor.Condition);
                visualStates.Add(new float[128]);scanIds.Add(-1);visionUpdates.Add(0);
            }
            lastTime=float.NegativeInfinity;
            return true;
        }

        public IHorseController Controller(int index) => new PolicyController(this,index);

        HorseAction Decide(int index)
        {
            if(!Ready)return new StandingPoseController().Decide(null);
            if(lastTime!=Time.fixedTime)
            {
                lastTime=Time.fixedTime;
                float[] input=new float[horses.Count*HorseTrainingObservation.Size];
                try
                {
                for(int i=0;i<horses.Count;i++)
                {
                    var horse=horses[i];var result=race.Results[i];
                    var frame=HorseTrainingObservation.Frame(horse,result,
                        Mathf.Max(0f,90f-race.Elapsed),100f);
                    if(result.falls!=lastFalls[i] ||
                        lastConditions[i]==HorseCondition.Recovering &&
                        horse.Condition==HorseCondition.Standing)
                    {
                        histories[i].Reset(frame);
                        Array.Clear(visualStates[i],0,128);scanIds[i]=-1;visionUpdates[i]=0;
                    }
                    lastFalls[i]=result.falls;
                    lastConditions[i]=horse.Condition;
                    var raw=histories[i].Build(frame,
                        HorseTrainingObservation.TerrainGrid(horse),
                        HorseTrainingObservation.OpponentData(horse,horses,race.Results),horse.Gait.Observation(-1,0f),horse.Lidar.Data);
                    Array.Copy(raw,0,input,i*HorseTrainingObservation.Size,raw.Length);
                    if(horse.IsPolicyActive && !result.finished && !result.dnf && scanIds[i]!=horse.Lidar.Sequence)
                    {
                        var scan=new float[HorseLidar.Features];Array.Copy(horse.Lidar.Data,scan,scan.Length);
                        using(var x=new Tensor<float>(new TensorShape(1,HorseLidar.Features),scan))
                        using(var h=new Tensor<float>(new TensorShape(1,128),visualStates[i]))
                        {
                            visionWorker.SetInput("scan",x);visionWorker.SetInput("hidden",h);visionWorker.Schedule();
                            visualStates[i]=(visionWorker.PeekOutput("next_hidden") as Tensor<float>).DownloadToArray();
                        }
                        if(visualStates[i].Length!=128)throw new InvalidOperationException("Vision hidden size mismatch");
                        foreach(float value in visualStates[i])if(!float.IsFinite(value))throw new InvalidOperationException("Non-finite vision state");
                        scanIds[i]=horse.Lidar.Sequence;visionUpdates[i]++;
                    }
                }
                    float[] hidden=new float[horses.Count*128];
                    for(int i=0;i<horses.Count;i++)Array.Copy(visualStates[i],0,hidden,i*128,128);
                    using(var tensor=new Tensor<float>(
                        new TensorShape(horses.Count,HorseTrainingObservation.Size),input))
                    using(var memory=new Tensor<float>(new TensorShape(horses.Count,128),hidden))
                    {
                        worker.SetInput("observation",tensor);worker.SetInput("hidden",memory);worker.Schedule();
                        var result=worker.PeekOutput() as Tensor<float>;
                        if(result==null)throw new InvalidOperationException("Policy output is not float tensor");
                        float[] output=result.DownloadToArray();
                        if(output.Length!=horses.Count*HorseAction.Size)
                            throw new InvalidOperationException("Policy output action size mismatch");
                        for(int i=0;i<horses.Count;i++)
                            for(int k=0;k<HorseAction.Size;k++)
                                actions[i].values[k]=Mathf.Clamp(output[i*HorseAction.Size+k],-1f,1f);
                    }
                    foreach(var action in actions)if(!action.IsValid())
                        throw new InvalidOperationException("Policy emitted non-finite action");
                }
                catch(Exception exception)
                {
                    Debug.LogError("QHOP inference failed, falling back: "+exception);
                    DisposeWorker();
                    for(int i=0;i<actions.Count;i++)actions[i]=new StandingPoseController().Decide(null);
                }
            }
            return actions[index];
        }

        sealed class PolicyController : IHorseController
        {
            readonly HorsePolicyRunner owner;
            readonly int index;
            public PolicyController(HorsePolicyRunner owner,int index){this.owner=owner;this.index=index;}
            public string DisplayName => owner.Ready?"PPO: "+owner.ModelName:"站立演示控制器";
            public HorseAction Decide(HorseObservation _) => owner.Decide(index);
        }

        void DisposeWorker(){if(worker!=null){worker.Dispose();worker=null;}if(visionWorker!=null){visionWorker.Dispose();visionWorker=null;}}
        void OnDestroy(){DisposeWorker();}
    }
}
