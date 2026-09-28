using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using TMPro;
using UnityEngine.TextCore.LowLevel;

namespace QHop
{
    public sealed class ArenaController : MonoBehaviour
    {
        public GameObject horsePrefab;
        public HorseRigDefinition rig;
        public Material horseMaterial;
        public bool experimentScene;
        public bool allowRecovery = true;
        public Camera viewCamera;
        public readonly List<HorseActor> horses = new List<HorseActor>();

        HorseConfig[] configs = new HorseConfig[4];
        int selected;
        int seed = 20260927;
        TMP_InputField seedInput;
        bool running;
        bool pendingReset;
        Vector3 pushDirection=Vector3.right;
        float pushSpeed=2f;
        int cameraMode;
        bool showDebug;
        Vector3 freeCameraPivot;
        bool freeCameraInitialized;
        Canvas canvas;
        TextMeshProUGUI status;
        TextMeshProUGUI debugText;
        TextMeshProUGUI[] cards = new TextMeshProUGUI[4];
        readonly List<Slider> sliders = new List<Slider>();
        readonly List<TextMeshProUGUI> sliderValues = new List<TextMeshProUGUI>();
        readonly string[] attributeLabels = { "体型", "前腿长度", "后腿长度", "身长", "前腿力量", "后腿力量", "体重 kg", "蹄子光滑度" };
        readonly float[] minima = { .85f,.85f,.85f,.85f,.55f,.55f,250f,0f };
        readonly float[] maxima = { 1.15f,1.15f,1.15f,1.15f,1.45f,1.45f,750f,1f };
        bool modifyingUI;
        readonly Queue<HorseEvent> events = new Queue<HorseEvent>();
        HorseTelemetryRecorder telemetry;
        HorseRaceSession race;
        HorsePolicyRunner policyRunner;
        bool policyMode;
        static TMP_FontAsset chineseFont;

        void Awake()
        {
            Time.fixedDeltaTime = .01f;
            telemetry=GetComponent<HorseTelemetryRecorder>();
            for (int i=0;i<4;i++) configs[i] = HorseGenetics.Generate(seed+i*101);
            BuildInterface();
            ResetArena();
        }

        void FixedUpdate()
        {
            if(pendingReset){pendingReset=false;ResetArena();return;}
            foreach (var horse in horses) horse.FixedStep(horses, running);
            if(running && telemetry!=null)telemetry.Sample(horses);
            if (!running) return;
            if(race==null)race=new HorseRaceSession(horses,100f,90f);
            race.Step(Time.fixedDeltaTime);
            for(int i=0;i<horses.Count;i++)
            {
                var result=race.Results[i];
                if(result.finished && result.rank>0 && Mathf.Abs(result.finishTime-race.Elapsed)<Time.fixedDeltaTime)
                    ShowStatus($"{i+1} 号马完赛，第 {result.rank} 名，{result.finishTime:F2} 秒");
            }
            if(race.Complete){running=false;ShowStatus("比赛结束，可按 R 重赛");}
        }

        void Update()
        {
            for(int i=0;i<4;i++) if(Input.GetKeyDown(KeyCode.Alpha1+i)) Select(i);
            if(Input.GetKeyDown(KeyCode.Space)) ToggleRunning();
            if(Input.GetKeyDown(KeyCode.R)) QueueReset();
            if(Input.GetKeyDown(KeyCode.M)) TogglePolicy();
            if(Input.GetKeyDown(KeyCode.C)) {cameraMode=(cameraMode+1)%3;freeCameraInitialized=false;}
            while(events.Count>0)
            {
                HorseEvent e=events.Dequeue();
                if(telemetry!=null)telemetry.Record(e);
                if(e.type==HorseEventType.Fall) ShowStatus((e.horseId+1)+" 号马摔倒了！");
                if(e.type==HorseEventType.Recovery) ShowStatus((e.horseId+1)+" 号马重新站起");
                var effects = GetComponent<ArenaEffects>();
                if(effects!=null) effects.Play(e);
            }
            for(int i=0;i<4;i++) if(cards[i]!=null && i<horses.Count)
            {
                var result=race.Results[i];
                string outcome=result.finished?$"第 {result.rank} 名 · {result.finishTime:F2}s":
                    result.dnf?"DNF · "+result.reason:"进行中";
                cards[i].text=$"{i+1} 号马 · {horses[i].Condition} · 罚停 {result.falls} 次\n{outcome}";
            }
            if(debugText!=null && showDebug && selected<horses.Count)
            {
                var h=horses[selected];
                debugText.text=$"控制器：{h.ControllerName}\n刚体：{h.bodies.Count}  关节：{h.allJoints.Count}\n"+
                    $"重心：{h.Torso.worldCenterOfMass:F2}\n速度：{h.Torso.linearVelocity.magnitude:F2} m/s\n"+
                    $"动作：{HorseAction.Size}  观测：{HorseObservation.Size}";
            }
        }

        void OnDrawGizmos()
        {
            if(!Application.isPlaying || !showDebug)return;
            for(int i=0;i<horses.Count;i++)
            {
                var horse=horses[i];if(horse==null)continue;
                Gizmos.color=new[]{Color.red,Color.yellow,Color.cyan,Color.magenta}[i];
                foreach(var c in horse.GetComponentsInChildren<Collider>())
                    Gizmos.DrawWireCube(c.bounds.center,c.bounds.size);
                Gizmos.DrawSphere(horse.Torso.worldCenterOfMass,.08f);
                Gizmos.color=Color.green;
                foreach(var joint in horse.allJoints)
                    Gizmos.DrawLine(joint.transform.TransformPoint(joint.anchor),
                        joint.connectedBody.transform.TransformPoint(joint.connectedAnchor));
            }
        }

        void LateUpdate()
        {
            if(viewCamera==null || horses.Count!=4) return;
            Vector3 target;
            float width;
            if(cameraMode==1)
            {
                target=horses[selected].Torso.position;
                width=8f;
            }
            else
            {
                Vector3 mid=Vector3.zero;
                float lowZ=float.MaxValue, highZ=float.MinValue;
                foreach(var horse in horses) { mid+=horse.Torso.position; lowZ=Mathf.Min(lowZ,horse.Torso.position.z); highZ=Mathf.Max(highZ,horse.Torso.position.z); }
                target=mid/4f;
                width=Mathf.Clamp(12f+(highZ-lowZ),12f,40f);
            }
            if(cameraMode==2)
            {
                if(!freeCameraInitialized){freeCameraPivot=target;freeCameraInitialized=true;}
                float h=Input.GetAxisRaw("Horizontal"), v=Input.GetAxisRaw("Vertical");
                freeCameraPivot+=new Vector3(h,0f,v)*8f*Time.unscaledDeltaTime;
                target=freeCameraPivot;
            }
            Vector3 desired=target+new Vector3(11f,Mathf.Max(6f,width*.45f),-Mathf.Max(8f,width*.7f));
            viewCamera.transform.position=Vector3.Lerp(viewCamera.transform.position,desired,Time.unscaledDeltaTime*2.5f);
            Quaternion angle=Quaternion.LookRotation(target+Vector3.up*1f-viewCamera.transform.position);
            viewCamera.transform.rotation=Quaternion.Slerp(viewCamera.transform.rotation,angle,Time.unscaledDeltaTime*3f);
        }

        void ResetArena()
        {
            running=false;
            Time.timeScale=0f;
            foreach(var horse in horses) if(horse!=null) {horse.gameObject.SetActive(false);Destroy(horse.gameObject);}
            horses.Clear();
            race=null;
            float[] lane={-3.6f,-1.2f,1.2f,3.6f};
            for(int i=0;i<4;i++)
            {
                var go=new GameObject("Horse_"+(i+1));
                go.transform.SetParent(transform);
                go.transform.position=new Vector3(lane[i],0f,experimentScene?0f:1f);
                var actor=go.AddComponent<HorseActor>();
                actor.Initialize(i,configs[i],horsePrefab,rig,horseMaterial,allowRecovery);
                actor.EventRaised+=e=>events.Enqueue(e);
                horses.Add(actor);
            }
            Physics.SyncTransforms();
            race=new HorseRaceSession(horses,100f,90f);
            if(policyMode && !ApplyPolicy())policyMode=false;
            Select(selected);
            ShowStatus(policyMode?"PPO 模型："+policyRunner.ModelName:
                "站立演示控制器 · 空格开始 / R 重置 / C 切镜头");
        }
        public void QueueReset()
        {
            if(running)pendingReset=true;
            else ResetArena();
        }

        void ToggleRunning()
        {
            if(race!=null && race.Complete)return;
            running=!running;
            Time.timeScale=running?1f:0f;
            ShowStatus(running?"物理演示运行中":"暂停");
        }
        void TogglePolicy()
        {
            policyMode=!policyMode;
            if(policyMode && !ApplyPolicy())policyMode=false;
            if(!policyMode)foreach(var horse in horses)horse.SetController(new StandingPoseController());
            ShowStatus(policyMode?"PPO 模型："+policyRunner.ModelName:"站立演示控制器");
        }
        bool ApplyPolicy()
        {
            if(policyRunner==null)policyRunner=GetComponent<HorsePolicyRunner>() ??
                gameObject.AddComponent<HorsePolicyRunner>();
            if(!policyRunner.Load(horses,race))return false;
            for(int i=0;i<horses.Count;i++)horses[i].SetController(policyRunner.Controller(i));
            return true;
        }
        void OnDestroy() { Time.timeScale=1f; }

        void ShowStatus(string message) { if(status!=null) status.text=message; }
        void Select(int index)
        {
            selected=Mathf.Clamp(index,0,3);
            modifyingUI=true;
            for(int i=0;i<sliders.Count;i++) sliders[i].value=GetAttribute(configs[selected],i);
            modifyingUI=false;
            RefreshValues();
        }

        static float GetAttribute(HorseConfig c,int i)
        {
            switch(i)
            {
                case 0:return c.bulk; case 1:return c.frontLength; case 2:return c.backLength;
                case 3:return c.bodyLength; case 4:return c.frontStrength; case 5:return c.backStrength;
                case 6:return c.massKg; default:return c.hoofSmoothness;
            }
        }
        static void SetAttribute(HorseConfig c,int i,float v)
        {
            switch(i)
            {
                case 0:c.bulk=v;break; case 1:c.frontLength=v;break; case 2:c.backLength=v;break;
                case 3:c.bodyLength=v;break; case 4:c.frontStrength=v;break; case 5:c.backStrength=v;break;
                case 6:c.massKg=v;break; default:c.hoofSmoothness=v;break;
            }
            c.independentOverrides=true; c.Clamp();
        }
        void RefreshValues()
        {
            for(int i=0;i<sliders.Count;i++)
                sliderValues[i].text=GetAttribute(configs[selected],i).ToString(i==6?"F0":"F2");
        }

        void BuildInterface()
        {
            var root=new GameObject("Presentation UI",typeof(RectTransform),typeof(Canvas),typeof(CanvasScaler),typeof(GraphicRaycaster));
            root.SetActive(false);
            canvas=root.GetComponent<Canvas>(); canvas.renderMode=RenderMode.ScreenSpaceOverlay;
            var scaler=root.GetComponent<CanvasScaler>(); scaler.uiScaleMode=CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution=new Vector2(1920,1080);
            var eventSystem=new GameObject("EventSystem",typeof(UnityEngine.EventSystems.EventSystem),typeof(UnityEngine.EventSystems.StandaloneInputModule));
            eventSystem.transform.SetParent(root.transform);
            Panel(root.transform,"Top",new Vector2(0,-85),new Vector2(1880,110),new Vector2(.5f,1),new Color(.05f,.13f,.17f,.7f));
            status=Label(root.transform,"Status",new Vector2(0,-48),new Vector2(1200,50),new Vector2(.5f,1),32);
            Button(root.transform,"开始 / 暂停",new Vector2(-390,-49),new Vector2(170,48),ToggleRunning);
            Button(root.transform,"重置",new Vector2(390,-49),new Vector2(135,48),QueueReset);
            Button(root.transform,"模型 / 站立",new Vector2(780,-49),new Vector2(170,48),TogglePolicy);
            Button(root.transform,"调试",new Vector2(570,-49),new Vector2(135,48),()=>{showDebug=!showDebug;debugText.gameObject.SetActive(showDebug);});
            debugText=Label(root.transform,"Debug",new Vector2(-15,-197),new Vector2(680,190),new Vector2(1,1),20);
            debugText.alignment=TextAlignmentOptions.TopLeft;
            debugText.gameObject.SetActive(false);
            var panel=Panel(root.transform,"ExperimentPanel",new Vector2(210,0),new Vector2(400,790),new Vector2(0,.5f),new Color(.04f,.15f,.18f,.83f));
            Label(panel.transform,"Title",new Vector2(0,352),new Vector2(360,46),new Vector2(.5f,.5f),29).text="QHOP 赛马实验场";
            for(int i=0;i<4;i++)
            {
                int choice=i;
                Button(panel.transform,(i+1).ToString(),new Vector2(-135+i*90,295),new Vector2(78,45),()=>Select(choice));
            }
            for(int i=0;i<8;i++)
            {
                int attribute=i;
                int y=225-i*53;
                Label(panel.transform,"Attr"+i,new Vector2(-112,y+16),new Vector2(155,30),new Vector2(.5f,.5f),20).text=attributeLabels[i];
                var slider=CreateSlider(panel.transform,new Vector2(28,y-9),minima[i],maxima[i]);
                sliders.Add(slider);
                sliderValues.Add(Label(panel.transform,"Value"+i,new Vector2(151,y-9),new Vector2(70,30),new Vector2(.5f,.5f),18));
                slider.onValueChanged.AddListener(v=> {if(modifyingUI)return;SetAttribute(configs[selected],attribute,v);RefreshValues();ShowStatus("配置已修改；点击 应用并重置");});
            }
            Button(panel.transform,"应用并重置",new Vector2(-93,-254),new Vector2(174,42),QueueReset);
            Button(panel.transform,"单马随机",new Vector2(103,-254),new Vector2(174,42),()=>{if(!ReadSeed())return;configs[selected]=HorseGenetics.Generate(seed+selected*101);Select(selected);QueueReset();});
            Button(panel.transform,"全部随机",new Vector2(-93,-306),new Vector2(174,42),()=>{if(!ReadSeed())return;for(int i=0;i<4;i++)configs[i]=HorseGenetics.Generate(seed+i*101);Select(selected);QueueReset();});
            Button(panel.transform,"保存配置",new Vector2(103,-306),new Vector2(174,42),Save);
            Button(panel.transform,"读取配置",new Vector2(-93,-356),new Vector2(174,42),Load);
            Button(panel.transform,"推一下马",new Vector2(103,-356),new Vector2(174,42),()=>horses[selected].Push(pushDirection,pushSpeed));
            var pushPanel=Panel(root.transform,"Push Controls",new Vector2(-215,185),new Vector2(395,250),new Vector2(1,0),new Color(.04f,.15f,.18f,.83f));
            Label(pushPanel.transform,"PushTitle",new Vector2(0,95),new Vector2(360,36),new Vector2(.5f,.5f),23).text="推力测试 · 已选马";
            string[] dirNames={"左","右","前","后"};
            Vector3[] directions={Vector3.left,Vector3.right,Vector3.forward,Vector3.back};
            for(int i=0;i<4;i++) {int k=i;Button(pushPanel.transform,dirNames[i],new Vector2(-132+i*88,38),new Vector2(78,44),()=>pushDirection=directions[k]);}
            for(int i=0;i<3;i++) {int k=i;Button(pushPanel.transform,new[]{"轻","中","重"}[i],new Vector2(-90+i*90,-18),new Vector2(78,44),()=>pushSpeed=new[]{1f,2f,4f}[k]);}
            Button(pushPanel.transform,"施加冲量",new Vector2(0,-77),new Vector2(220,42),()=>horses[selected].Push(pushDirection,pushSpeed));
            Button(pushPanel.transform,"导出记录",new Vector2(0,-122),new Vector2(220,36),()=>{if(telemetry!=null)ShowStatus("已导出："+telemetry.ExportCsv());});
            seedInput=CreateSeedInput(root.transform,new Vector2(0,-119));
            seedInput.text=seed.ToString();
            for(int i=0;i<4;i++)
                cards[i]=Label(root.transform,"HorseCard"+i,new Vector2(-540+i*350,70),new Vector2(330,65),new Vector2(.5f,0),19);
            Select(0);
            root.SetActive(true);
        }

        [Serializable] class SaveData { public int schemaVersion=1; public HorseConfig[] horses; }
        bool ReadSeed()
        {
            if(seedInput!=null && int.TryParse(seedInput.text,out int entered)){seed=entered;return true;}
            ShowStatus("种子请输入整数");return false;
        }
        string SavePath=>Path.Combine(Application.persistentDataPath,"horses.json");
        void Save()
        {
            File.WriteAllText(SavePath,JsonUtility.ToJson(new SaveData { horses=configs },true));
            ShowStatus("已保存到 "+SavePath);
        }
        void Load()
        {
            if(!File.Exists(SavePath)){ShowStatus("未找到保存的配置");return;}
            var value=JsonUtility.FromJson<SaveData>(File.ReadAllText(SavePath));
            if(value==null || value.schemaVersion!=1 || value.horses==null || value.horses.Length!=4)
            {ShowStatus("配置版本或数量错误");return;}
            foreach(var entry in value.horses)
                if(entry==null || entry.schemaVersion!=1){ShowStatus("配置内容损坏");return;}
            configs=value.horses;
            foreach(var config in configs)config.Clamp();
            seed=configs[0].seed;
            seedInput.text=seed.ToString();
            Select(selected);QueueReset();
        }

        static RectTransform Rect(GameObject go,Transform parent,Vector2 position,Vector2 size,Vector2 anchor)
        {
            go.transform.SetParent(parent,false);
            var rt=go.GetComponent<RectTransform>(); rt.anchorMin=rt.anchorMax=anchor;
            rt.pivot=new Vector2(.5f,.5f); rt.anchoredPosition=position; rt.sizeDelta=size; return rt;
        }
        static GameObject Panel(Transform parent,string name,Vector2 pos,Vector2 size,Vector2 anchor,Color color)
        {
            var go=new GameObject(name,typeof(RectTransform),typeof(Image));
            Rect(go,parent,pos,size,anchor);go.GetComponent<Image>().color=color;return go;
        }
        static TextMeshProUGUI Label(Transform parent,string name,Vector2 pos,Vector2 size,Vector2 anchor,int fontSize)
        {
            var go=new GameObject(name,typeof(RectTransform),typeof(TextMeshProUGUI));
            Rect(go,parent,pos,size,anchor);
            var text=go.GetComponent<TextMeshProUGUI>();
            text.fontSize=fontSize;text.color=Color.white;text.alignment=TextAlignmentOptions.Center;
            if(chineseFont==null)
            {
                var sourceFont=Resources.Load<Font>("Fonts/QHOP-CJK");
                if(sourceFont==null)throw new InvalidOperationException("Bundled Chinese UI font is missing.");
                chineseFont=TMP_FontAsset.CreateFontAsset(sourceFont,50,5,
                    GlyphRenderMode.SDFAA,1024,1024,AtlasPopulationMode.Dynamic,true);
                if(chineseFont==null)throw new InvalidOperationException("TextMesh Pro could not create the Chinese UI font.");
                chineseFont.TryAddCharacters("QHOP赛马实验场开始暂停前后腿长度力量体重蹄子光滑度随机应用重置推力测试",out string missing);
                if(!string.IsNullOrEmpty(missing))Debug.LogWarning("UI font is missing characters: "+missing);
            }
            text.font=chineseFont;
            return text;
        }
        static GameObject Button(Transform parent,string title,Vector2 pos,Vector2 size,UnityEngine.Events.UnityAction callback)
        {
            var go=Panel(parent,title,pos,size,new Vector2(.5f,.5f),new Color(.19f,.52f,.56f,1f));
            var button=go.AddComponent<Button>();button.onClick.AddListener(callback);
            Label(go.transform,"Text",Vector2.zero,size,new Vector2(.5f,.5f),18).text=title;
            return go;
        }
        static Slider CreateSlider(Transform parent,Vector2 pos,float min,float max)
        {
            var go=new GameObject("Slider",typeof(RectTransform),typeof(Slider));
            Rect(go,parent,pos,new Vector2(190,26),new Vector2(.5f,.5f));
            var bg=Panel(go.transform,"Background",Vector2.zero,new Vector2(190,9),new Vector2(.5f,.5f),new Color(.22f,.39f,.43f,1f));
            var fill=Panel(go.transform,"Fill",Vector2.zero,new Vector2(180,9),new Vector2(.5f,.5f),new Color(.8f,.82f,.48f,1f));
            var handle=Panel(go.transform,"Handle",Vector2.zero,new Vector2(18,27),new Vector2(.5f,.5f),new Color(1f,.95f,.72f,1f));
            var slider=go.GetComponent<Slider>();slider.fillRect=fill.GetComponent<RectTransform>();slider.handleRect=handle.GetComponent<RectTransform>();
            slider.targetGraphic=handle.GetComponent<Image>();slider.minValue=min;slider.maxValue=max;
            return slider;
        }
        static TMP_InputField CreateSeedInput(Transform parent,Vector2 pos)
        {
            var bg=Panel(parent,"Seed Input",pos,new Vector2(245,42),new Vector2(.5f,1),new Color(.19f,.34f,.39f,1f));
            var label=Label(bg.transform,"InputText",Vector2.zero,new Vector2(225,38),new Vector2(.5f,.5f),20);
            var input=bg.AddComponent<TMP_InputField>();
            input.textComponent=label;
            input.contentType=TMP_InputField.ContentType.IntegerNumber;
            return input;
        }
    }
}
