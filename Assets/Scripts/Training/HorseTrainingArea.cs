using System;
using System.Collections.Generic;
using System.IO;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.SideChannels;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace QHop
{
    [Serializable]
    public sealed class HorseTrainingEpisodeLog
    {
        public int version=1,taskIndex;
        public string episodeId,taskName;
        public float taskDistance,taskTimeLimit,stationaryFraction;
        public float progressMeters;
        public float filteredPosture,filteredGround,filteredInactive,filteredBoundary;
        public int stage,area,horse,falls,rank;
        public bool finished,dnf;
        public string reason;
        public float elapsed,progress,posture,smoothness,effort,slip,total;
        public float weightedProgress,weightedPosture,weightedSmoothness,
            weightedEffort,weightedSlip,weightedTime,weightedFall,
            weightedFinish,weightedDnf;
    }

    public sealed class HorseTrainingArea
    {
        public const string CourseContract="stage1-v2:2m-10s;5m-15s;10m-20s;20m-20s;eval-25;early-pass=0.80,0.60;final-pass=0.90,0.80x3";
        public static readonly float[] Stage1Distances={2f,5f,10f,20f};
        public static readonly float[] Stage1Times={10f,15f,20f,20f};
        public static readonly string[] Stage1Names={"初步迈步","连续移动","延长距离","阶段验收"};
        readonly Scene scene;
        readonly PhysicsScene physicsScene;
        readonly GameObject root;
        readonly GameObject horsePrefab;
        readonly HorseRigDefinition rig;
        readonly Material horseMaterial;
        System.Random random;
        readonly int stage;
        readonly int taskIndex;
        readonly int areaIndex;
        readonly bool mixedArea;
        readonly int learnerHorseId;
        readonly float difficulty,shaping;
        readonly bool preview;
        float episodeDifficulty;
        readonly List<GameObject> episodeObjects=new List<GameObject>();
        readonly List<PhysicsMaterial> episodeMaterials=new List<PhysicsMaterial>();
        readonly List<HorseTrainingAgent> agents=new List<HorseTrainingAgent>();
        readonly List<HorseTrainingReward> rewards=new List<HorseTrainingReward>();
        readonly List<bool> ended=new List<bool>();
        readonly List<HorseActor> horses=new List<HorseActor>();
        readonly List<int> disturbances=new List<int>();
        readonly List<int> recoveries=new List<int>();
        readonly List<float> disturbanceTime=new List<float>();
        readonly List<float> disturbanceSpeed=new List<float>();
        readonly List<int> disturbanceFalls=new List<int>();
        int tick;
        int episodeNumber;
        float nextDisturbance;
        public HorseRaceSession Race { get; private set; }
        public IReadOnlyList<HorseActor> Horses => horses;
        public IReadOnlyList<int> Disturbances => disturbances;
        public IReadOnlyList<int> Recoveries => recoveries;
        public float Distance => stage==1?Stage1Distances[taskIndex]:100f;
        public float TimeLimit => stage==1?Stage1Times[taskIndex]:stage==2||stage==3?45f:stage==4?60f:90f;
        public string TaskName => stage==1?Stage1Names[taskIndex]:"";
        public float TimeRemaining => Mathf.Max(0f,TimeLimit-(Race==null?0f:Race.Elapsed));
        public int LearnerHorseId => learnerHorseId;
        public string LastOutcome { get; private set; }

        public HorseTrainingArea(int index,int stage,float difficulty,float shaping,int seed,
            GameObject prefab,HorseRigDefinition definition,Material material,bool preview=false)
            :this(index,stage,difficulty,shaping,seed,prefab,definition,material,preview,0,null)
        {}

        public HorseTrainingArea(int index,int stage,float difficulty,float shaping,int seed,
            GameObject prefab,HorseRigDefinition definition,Material material,bool preview,
            int taskIndex,QHopEpisodeSideChannel episodeChannel)
        {
            this.stage=stage;this.difficulty=difficulty;this.shaping=shaping;
            this.taskIndex=taskIndex;
            this.episodeChannel=episodeChannel;
            this.preview=preview;
            areaIndex=index;
            mixedArea=stage>=5 && (difficulty<.5f || index%2==0);
            learnerHorseId=index%(stage<5?1:4);
            horsePrefab=prefab;rig=definition;horseMaterial=material;
            random=new System.Random(seed+index*100003);
            scene=SceneManager.CreateScene("QHopTrainingArea_"+index,
                new CreateSceneParameters(LocalPhysicsMode.Physics3D));
            physicsScene=scene.GetPhysicsScene();
            root=new GameObject("TrainingAreaRoot");
            SceneManager.MoveGameObjectToScene(root,scene);
            int count=stage<5?1:4;
            for(int i=0;i<count;i++)
            {
                var go=new GameObject("TrainingAgent_"+i);
                SceneManager.MoveGameObjectToScene(go,scene);
                go.transform.SetParent(root.transform);
                var behavior=go.AddComponent<BehaviorParameters>();
                behavior.BehaviorName=mixedArea && i!=learnerHorseId?
                    "QHopFrozenV2":"QHopCurrentV2";
                behavior.BehaviorType=BehaviorType.Default;
                behavior.BrainParameters.VectorObservationSize=HorseTrainingObservation.Size;
                behavior.BrainParameters.NumStackedVectorObservations=1;
                behavior.BrainParameters.ActionSpec=ActionSpec.MakeContinuous(HorseAction.Size);
                agents.Add(go.AddComponent<HorseTrainingAgent>());
            }
            Reset();
        }
        readonly QHopEpisodeSideChannel episodeChannel;

        void Add(GameObject go)
        {
            SceneManager.MoveGameObjectToScene(go,scene);
            go.transform.SetParent(root.transform);
            episodeObjects.Add(go);
        }

        GameObject Box(string name,Vector3 position,Vector3 scale, bool ground)
        {
            var go=GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name=name;Add(go);go.transform.position=position;go.transform.localScale=scale;
            if(ground)go.AddComponent<HorseGroundSurface>();
            return go;
        }

        void BuildTrack()
        {
            if(stage<4)
            {
                var flat=Box("TrainingGround",new Vector3(0,-.12f,50f),
                    new Vector3(12f,.2f,110f),true);
                var material=new PhysicsMaterial("TrainingGrip")
                {staticFriction=1f,dynamicFriction=1f,
                    frictionCombine=PhysicsMaterialCombine.Multiply};
                flat.GetComponent<Collider>().sharedMaterial=material;
                episodeMaterials.Add(material);
            }
            else
            {
            float rough=stage>=4?Mathf.Clamp01(episodeDifficulty*2f):0f;
            float obstacleLevel=stage>=4?Mathf.Clamp01((episodeDifficulty-.2f)*2.5f):0f;
            float gripLevel=stage>=4?Mathf.Clamp01((episodeDifficulty-.4f)*2.5f):0f;
            float boundaryHeight=0f;
            for(int section=-1;section<=20;section++)
            {
                float z=2.5f+section*5f;
                float nextHeight=Mathf.Clamp(boundaryHeight+
                    (float)(random.NextDouble()*2-1)*.08f*rough,-.12f*rough,.12f*rough);
                float slope=Mathf.Atan2(nextHeight-boundaryHeight,5f)*Mathf.Rad2Deg;
                var tile=Box("TrainingGround",new Vector3(0,-.12f+
                    (boundaryHeight+nextHeight)*.5f,z),new Vector3(12f,.2f,5.05f),true);
                tile.transform.rotation=Quaternion.Euler(-slope,0,0);
                boundaryHeight=nextHeight;
                float friction=Mathf.Lerp(1f,.4f+(float)random.NextDouble()*.8f,gripLevel);
                var pm=new PhysicsMaterial("TrainingGrip")
                {staticFriction=friction,dynamicFriction=friction,frictionCombine=PhysicsMaterialCombine.Multiply};
                tile.GetComponent<Collider>().sharedMaterial=pm;
                episodeMaterials.Add(pm);
                if(stage>=4 && section>2 && section<18 && random.NextDouble()<.2*obstacleLevel)
                {
                    float x=(float)(random.NextDouble()*8-4);
                    float obstacleHeight=.05f+.25f*(float)random.NextDouble()*obstacleLevel;
                    var obstacle=Box("TrainingObstacle",new Vector3(x,obstacleHeight*.5f,z),
                        new Vector3(.8f,obstacleHeight,.55f),true);
                    obstacle.GetComponent<HorseGroundSurface>().permitsRecovery=false;
                }
            }
            }
            Box("LeftBoundary",new Vector3(-6.8f,.5f,47.5f),new Vector3(.15f,1f,105f),false);
            Box("RightBoundary",new Vector3(6.8f,.5f,47.5f),new Vector3(.15f,1f,105f),false);
        }

        HorseConfig Configuration(int id)
        {
            var standard=new HorseConfig();
            if(stage<3 || random.NextDouble()<.2)return standard;
            var target=HorseGenetics.Generate(random.Next());
            float alpha=stage==3?difficulty:1f;
            standard.bulk=Mathf.Lerp(1f,target.bulk,alpha);
            standard.frontLength=Mathf.Lerp(1f,target.frontLength,alpha);
            standard.backLength=Mathf.Lerp(1f,target.backLength,alpha);
            standard.bodyLength=Mathf.Lerp(1f,target.bodyLength,alpha);
            standard.frontStrength=Mathf.Lerp(1f,target.frontStrength,alpha);
            standard.backStrength=Mathf.Lerp(1f,target.backStrength,alpha);
            standard.massKg=Mathf.Lerp(450f,target.massKg,alpha);
            standard.hoofSmoothness=stage>=4?target.hoofSmoothness:.3f;
            return standard;
        }

        public void Reset()
        {
            episodeNumber++;
            foreach(var go in episodeObjects)if(go!=null){go.SetActive(false);UnityEngine.Object.Destroy(go);}
            foreach(var material in episodeMaterials)if(material!=null)UnityEngine.Object.Destroy(material);
            episodeMaterials.Clear();
            episodeObjects.Clear();horses.Clear();rewards.Clear();ended.Clear();
            disturbances.Clear();recoveries.Clear();disturbanceTime.Clear();
            disturbanceSpeed.Clear();disturbanceFalls.Clear();
            episodeDifficulty=stage==4 && random.NextDouble()<.2?0f:difficulty;
            BuildTrack();
            int count=agents.Count;
            float[] lanes={-3.6f,-1.2f,1.2f,3.6f};
            for(int i=lanes.Length-1;i>0;i--)
            {int j=random.Next(i+1);float temporary=lanes[i];lanes[i]=lanes[j];lanes[j]=temporary;}
            for(int i=0;i<count;i++)
            {
                var go=new GameObject("TrainingHorse_"+i);Add(go);
                go.transform.position=new Vector3(count==1?0f:lanes[i],0f,0f);
                var horse=go.AddComponent<HorseActor>();
                horse.Initialize(i,Configuration(i),horsePrefab,rig,horseMaterial,true,!preview);
                horse.ExternalActionMode=true;
                horses.Add(horse);
                rewards.Add(new HorseTrainingReward(stage,shaping,TimeLimit));
                ended.Add(false);
                disturbances.Add(0);recoveries.Add(0);
                disturbanceTime.Add(-1f);disturbanceSpeed.Add(0f);disturbanceFalls.Add(0);
            }
            Physics.SyncTransforms();
            Race=new HorseRaceSession(horses,Distance,TimeLimit);
            for(int i=0;i<count;i++)
            {
                agents[i].Bind(this,horses[i]);
            }
            nextDisturbance=3f;tick=0;
            foreach(var agent in agents)agent.RequestDecision();
        }
        public void Restart(int seed)
        {
            random=new System.Random(seed+areaIndex*100003);
            LastOutcome=null;
            Reset();
        }

        public void Step()
        {
            if(stage>=4 && episodeDifficulty>=.8f && Race.Elapsed>=nextDisturbance)
            {
                nextDisturbance=Race.Elapsed+3f+(float)random.NextDouble()*5f;
                foreach(var horse in horses)if(horse.IsPolicyActive)
                {
                    int id=horse.Id;
                    disturbances[id]++;
                    disturbanceTime[id]=Race.Elapsed;
                    disturbanceSpeed[id]=Mathf.Max(0f,horse.Torso.linearVelocity.z);
                    disturbanceFalls[id]=Race.Results[id].falls;
                    var angle=(float)random.NextDouble()*Mathf.PI*2f;
                    horse.Push(new Vector3(Mathf.Cos(angle),0,Mathf.Sin(angle)),
                        Mathf.Lerp(.2f,1.5f,episodeDifficulty));
                }
            }
            physicsScene.Simulate(.01f);
            foreach(var horse in horses)horse.FixedStep(horses,true);
            Race.Step(.01f);
            for(int i=0;i<horses.Count;i++)if(disturbanceTime[i]>=0f &&
                Race.Elapsed-disturbanceTime[i]>=2f)
            {
                if(disturbanceSpeed[i]>=.5f &&
                    Race.Results[i].falls==disturbanceFalls[i] &&
                    horses[i].Torso.linearVelocity.z>=.8f*disturbanceSpeed[i]) recoveries[i]++;
                disturbanceTime[i]=-1f;
            }
            for(int i=0;i<agents.Count;i++)
            {
                if(ended[i])continue;
                agents[i].AddReward(rewards[i].Step(horses[i],Race.Results[i],.01f,
                    agents.Count,Race.Elapsed));
                if(Race.Results[i].finished || Race.Results[i].dnf)
                {
                    var result=Race.Results[i];var reward=rewards[i];
                    var episode=new HorseTrainingEpisodeLog
                    {
                        episodeId=areaIndex+"-"+episodeNumber+"-"+i,
                        taskIndex=taskIndex,taskName=TaskName,taskDistance=Distance,
                        taskTimeLimit=TimeLimit,
                        progressMeters=reward.Progress*horses[i].RestTorsoHeight,
                        stationaryFraction=result.activeTime>0f?
                            result.stationaryTime/result.activeTime:0f,
                        filteredPosture=result.filteredPosture,
                        filteredGround=result.filteredGround,
                        filteredInactive=result.filteredInactive,
                        filteredBoundary=result.filteredBoundary,
                        stage=stage,area=areaIndex,horse=i,falls=result.falls,rank=result.rank,
                        finished=result.finished,dnf=result.dnf,reason=result.reason,
                        elapsed=result.finished?result.finishTime:Race.Elapsed,
                        progress=reward.Progress,posture=reward.Posture,
                        smoothness=reward.Smoothness,effort=reward.Effort,
                        slip=reward.Slip,total=reward.Total,
                        weightedProgress=reward.WeightedProgress,
                        weightedPosture=reward.WeightedPosture,
                        weightedSmoothness=reward.WeightedSmoothness,
                        weightedEffort=reward.WeightedEffort,
                        weightedSlip=reward.WeightedSlip,
                        weightedTime=reward.WeightedTime,
                        weightedFall=reward.WeightedFall,
                        weightedFinish=reward.WeightedFinish,
                        weightedDnf=reward.WeightedDnf
                    };
                    string json=JsonUtility.ToJson(episode);
                    Debug.Log("QHOP_EPISODE "+json);
                    if(episodeChannel!=null)episodeChannel.Send(json);
                    ended[i]=true;agents[i].EndEpisode();
                }
            }
            tick++;
            // Keep every area's next decision on the same five-tick boundary.
            // An early finish still produces a terminal event this tick, while
            // the next episode begins at the common decision boundary.
            if(Race.Complete)
            {
                if(tick%5==0)
                {
                    var result=Race.Results[learnerHorseId];
                    LastOutcome=result.finished?"完赛 "+result.finishTime.ToString("F2")+" s":
                        "DNF: "+result.reason;
                    Reset();
                }
                return;
            }
            if(tick%5==0)for(int i=0;i<agents.Count;i++)if(!ended[i])agents[i].RequestDecision();
        }
    }

    public sealed class HorseTrainingBootstrap : MonoBehaviour
    {
        readonly List<HorseTrainingArea> areas=new List<HorseTrainingArea>();
        HorseRigDefinition rig;
        Material horseMaterial;
        bool preview;
        int stage,seed,observed,previewUpdate;
        float difficulty,shaping;
        Camera previewCamera;
        Font previewFont;
        StageGoals goals;
        string previewControlPath,previewSession;
        QHopEpisodeSideChannel episodeChannel;
        int taskIndex;
        long previewSequence;
        [Serializable] sealed class StageGoals { public StageGoal[] stages; public Stage1Task[] stage1Tasks; }
        [Serializable] sealed class StageGoal { public string name,goal,metric; }
        [Serializable] sealed class Stage1Task { public string name; public float distance,timeLimit; }
        static float Setting(string name,float fallback)
        {
            string prefix="--qhop-"+name+"=";
            foreach(string arg in Environment.GetCommandLineArgs())
                if(arg.StartsWith(prefix,StringComparison.Ordinal) &&
                    float.TryParse(arg.Substring(prefix.Length),System.Globalization.NumberStyles.Float,
                        System.Globalization.CultureInfo.InvariantCulture,out float value))return value;
            return Academy.Instance.EnvironmentParameters.GetWithDefault(name,fallback);
        }
        static string TextSetting(string name)
        {
            string prefix="--qhop-"+name+"=";
            foreach(string arg in Environment.GetCommandLineArgs())
                if(arg.StartsWith(prefix,StringComparison.Ordinal))return arg.Substring(prefix.Length);
            return null;
        }
        void PreviewRestart(bool newSeed)
        {
            if(newSeed)seed++;
            areas[0].Restart(seed);
            if(string.IsNullOrEmpty(previewControlPath))return;
            var text="{\"session_id\":\""+previewSession+"\",\"sequence\":"+
                (++previewSequence)+",\"seed\":"+seed+"}";
            string temporary=previewControlPath+".tmp";
            File.WriteAllText(temporary,text);
            if(File.Exists(previewControlPath))File.Replace(temporary,previewControlPath,null);
            else File.Move(temporary,previewControlPath);
        }
        void Awake()
        {
            if(SceneManager.GetActiveScene().name!="QHopTraining")return;
            Physics.simulationMode=SimulationMode.Script;
            Time.fixedDeltaTime=.01f;
            QualitySettings.vSyncCount=0;
            preview=Setting("preview",0)>0.5f;
            Application.targetFrameRate=preview?60:-1;
            stage=Mathf.Clamp(Mathf.RoundToInt(Setting("stage",1)),1,6);
            taskIndex=stage==1?Mathf.RoundToInt(Setting("task-index",0)):0;
            if(stage==1 && (taskIndex<0 || taskIndex>=HorseTrainingArea.Stage1Distances.Length))
                throw new InvalidOperationException("Invalid Stage 1 task index");
            float expectedDistance=stage==1?HorseTrainingArea.Stage1Distances[taskIndex]:100f;
            float expectedTime=stage==1?HorseTrainingArea.Stage1Times[taskIndex]:
                stage==2||stage==3?45f:stage==4?60f:90f;
            if(Mathf.Abs(Setting("distance",expectedDistance)-expectedDistance)>.001f ||
                Mathf.Abs(Setting("time-limit",expectedTime)-expectedTime)>.001f)
                throw new InvalidOperationException("Unity task distance/time contract mismatch");
            difficulty=Mathf.Clamp01(Setting("difficulty",1));
            shaping=Mathf.Clamp01(Setting("shaping",1));
            seed=Mathf.RoundToInt(Setting("seed",20260927));
            previewUpdate=Mathf.RoundToInt(Setting("preview-update",0));
            int count=preview?1:Mathf.Clamp(Mathf.RoundToInt(Setting("areas",stage<5?8:2)),1,32);
            var prefab=Resources.Load<GameObject>("HorseGame");
            if(prefab==null)throw new InvalidOperationException("HorseGame.fbx is unavailable");
            rig=ScriptableObject.CreateInstance<HorseRigDefinition>();
            Shader shader=Shader.Find("Universal Render Pipeline/Lit");
            if(shader==null)shader=Shader.Find("Standard");
            horseMaterial=new Material(shader);
            episodeChannel=new QHopEpisodeSideChannel();
            SideChannelManager.RegisterSideChannel(episodeChannel);
            for(int i=0;i<count;i++)areas.Add(new HorseTrainingArea(i,stage,difficulty,
                shaping,seed,prefab,rig,horseMaterial,preview,taskIndex,episodeChannel));
            if(preview)
            {
                previewControlPath=TextSetting("preview-control");
                previewSession=TextSetting("preview-session");
                var text=Resources.Load<TextAsset>("QHopStageGoals");
                if(text!=null)goals=JsonUtility.FromJson<StageGoals>(text.text);
                previewFont=Resources.Load<Font>("Fonts/QHOP-CJK");
                var cameraObject=new GameObject("Preview Camera");
                previewCamera=cameraObject.AddComponent<Camera>();
                previewCamera.clearFlags=CameraClearFlags.Skybox;
                previewCamera.fieldOfView=65f;
            }
        }
        void FixedUpdate(){foreach(var area in areas)area.Step();}
        void LateUpdate()
        {
            if(!preview || previewCamera==null || areas.Count==0)return;
            var horses=areas[0].Horses;
            if(horses.Count==0)return;
            observed=Mathf.Clamp(observed,0,horses.Count-1);
            var target=horses[observed].Torso.position;
            previewCamera.transform.position=Vector3.Lerp(previewCamera.transform.position,
                target+new Vector3(9f,5f,-12f),.12f);
            previewCamera.transform.LookAt(target+Vector3.up);
        }
        void OnGUI()
        {
            if(!preview || areas.Count==0)return;
            if(previewFont!=null)GUI.skin.font=previewFont;
            var area=areas[0];
            var goal=goals!=null && goals.stages!=null && goals.stages.Length>=stage?
                goals.stages[stage-1]:null;
            var horse=area.Horses[Mathf.Clamp(observed,0,area.Horses.Count-1)];
            var result=area.Race.Results[horse.Id];
            GUILayout.BeginArea(new Rect(12,12,710,260),GUI.skin.box);
            GUILayout.Label(goal==null?"Stage "+stage:goal.name+" | "+goal.goal);
            var shownTask=stage==1 && goals!=null && goals.stage1Tasks!=null &&
                goals.stage1Tasks.Length>taskIndex?goals.stage1Tasks[taskIndex].name:area.TaskName;
            if(stage==1)GUILayout.Label("当前子任务："+shownTask+"；"+
                area.Distance.ToString("F0")+" 米 / "+area.TimeLimit.ToString("F0")+" 秒；最终目标：20 米 / 20 秒");
            if(goal!=null)GUILayout.Label("自动目标："+goal.metric);
            GUILayout.Label("Checkpoint update "+previewUpdate+" | Seed "+seed+
                " | 观察马 "+horse.Id+" | 策略马 "+area.LearnerHorseId+
                " | 难度 "+difficulty.ToString("F2")+" | shaping "+shaping.ToString("F2"));
            GUILayout.Label("进度 "+result.highWater.ToString("F1")+" / "+area.Distance+
                " m | 速度 "+horse.Torso.linearVelocity.z.ToString("F1")+" m/s | 时间 "+
                area.Race.Elapsed.ToString("F1")+" s | 罚停 "+result.falls);
            GUILayout.Label(result.finished?"完赛，名次 "+result.rank:result.dnf?"DNF: "+result.reason:
                horse.FallWaitRemaining>0?"罚停中":"进行中");
            if(!string.IsNullOrEmpty(area.LastOutcome))GUILayout.Label("上回合："+area.LastOutcome);
            if(GUILayout.Button("切换观察马"))observed=(observed+1)%area.Horses.Count;
            if(GUILayout.Button("重试同一 seed"))PreviewRestart(false);
            if(GUILayout.Button("换 seed"))PreviewRestart(true);
            if(GUILayout.Button("退出验证"))Application.Quit();
            GUILayout.EndArea();
        }
        void OnDestroy()
        {
            if(episodeChannel!=null && Academy.IsInitialized)
                SideChannelManager.UnregisterSideChannel(episodeChannel);
            Physics.simulationMode=SimulationMode.FixedUpdate;
            if(rig!=null)Destroy(rig);
            if(horseMaterial!=null)Destroy(horseMaterial);
        }
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void StartTraining()
        {
            if(SceneManager.GetActiveScene().name=="QHopTraining")
                new GameObject("QHop Training Bootstrap").AddComponent<HorseTrainingBootstrap>();
        }
    }
}
