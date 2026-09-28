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
    }

    public sealed class HorsePolicyRunner : MonoBehaviour
    {
        Worker worker;
        readonly List<HorseActor> horses=new List<HorseActor>();
        readonly List<HorseObservationHistory> histories=new List<HorseObservationHistory>();
        readonly List<HorseAction> actions=new List<HorseAction>();
        readonly List<int> lastFalls=new List<int>();
        readonly List<HorseCondition> lastConditions=new List<HorseCondition>();
        HorseRaceSession race;
        float lastTime=float.NegativeInfinity;
        public string ModelName { get; private set; }
        public bool Ready => worker!=null;

        public bool Load(IReadOnlyList<HorseActor> actors,HorseRaceSession session)
        {
            DisposeWorker();
            horses.Clear();histories.Clear();actions.Clear();lastFalls.Clear();lastConditions.Clear();
            var asset=Resources.Load<ModelAsset>("QHopPolicy");
            var json=Resources.Load<TextAsset>("QHopPolicyMetadata");
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
            try{worker=new Worker(ModelLoader.Load(asset),BackendType.CPU);}
            catch(Exception exception){Debug.LogError("QHOP model load failed: "+exception);return false;}
            race=session;ModelName=metadata.model_name;
            foreach(var actor in actors)
            {
                horses.Add(actor);histories.Add(new HorseObservationHistory());
                actions.Add(new HorseAction());lastFalls.Add(0);
                lastConditions.Add(actor.Condition);
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
                for(int i=0;i<horses.Count;i++)
                {
                    var horse=horses[i];var result=race.Results[i];
                    var frame=HorseTrainingObservation.Frame(horse,result,
                        Mathf.Max(0f,90f-race.Elapsed),100f);
                    if(result.falls!=lastFalls[i] ||
                        lastConditions[i]==HorseCondition.Recovering &&
                        horse.Condition==HorseCondition.Standing)
                        histories[i].Reset(frame);
                    lastFalls[i]=result.falls;
                    lastConditions[i]=horse.Condition;
                    var raw=histories[i].Build(frame,
                        HorseTrainingObservation.TerrainGrid(horse),
                        HorseTrainingObservation.OpponentData(horse,horses,race.Results));
                    Array.Copy(raw,0,input,i*HorseTrainingObservation.Size,raw.Length);
                }
                try
                {
                    using(var tensor=new Tensor<float>(
                        new TensorShape(horses.Count,HorseTrainingObservation.Size),input))
                    {
                        worker.Schedule(tensor);
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

        void DisposeWorker(){if(worker!=null){worker.Dispose();worker=null;}}
        void OnDestroy(){DisposeWorker();}
    }
}
