using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Net.NetworkInformation;
using System.Security.Cryptography;
using System.Text;
using UnityEditor;
using UnityEngine;

[FilePath("Library/QHopTrainingConsole.asset",FilePathAttribute.Location.ProjectFolder)]
public sealed class QHopTrainingConsoleState : ScriptableSingleton<QHopTrainingConsoleState>
{
    public string runDirectory="";
    public int seed=20260927,workers=4,batch=16384,previewSeed=92500,extraSteps=0;
    public float simulationSpeed=1f;
    public int trainerPid,previewPid;
    public long trainerStartTicks,previewStartTicks;
    public double trainerLaunchUnix,previewLaunchUnix;
    public double stopRequestedUnix;
    public bool launchPreviewAfterStop;
    public string notice="";
    public void Store(){Save(true);}
}

[InitializeOnLoad]
public sealed class QHopTrainingConsole : EditorWindow
{
    const int MaxWorkers=8;
    const float MaxSimulationSpeed=20f;
    [Serializable] sealed class ReviewStatus
    {
        public string status,candidate,candidate_sha256,approved_at;
        public int preview_episodes;
    }
    [Serializable] sealed class RunStatus
    {
        public string session_id,status,checkpoint,error,message;
        public int pid,stage,updates,passes,global_steps,phase_steps,task_index;
        public float learning_rate,difficulty,shaping,horse_steps_per_second;
        public double updated_at;
        public ReviewStatus review;
        public ApprovalRecord[] approvals;
        public EvaluationStatus last_evaluation;
        public EpisodeSummary recent100;
    }
    [Serializable] sealed class ApprovalRecord
    {
        public int stage,preview_episodes;
        public string candidate_sha256,approved_at;
    }
    [Serializable] sealed class EvaluationStatus
    {
        public int stage,task_index;
        public float completion,fall_free,median_speed,falls_per_100m,
            gait_coverage,correct_cycle_ratio,bad_cycle_ratio,high_hoof_fraction,
            synchrony_cycle_ratio,four_leg_participation,speed_band_fraction,
            worst_morphology_completion,disturbance_recovery,median_time;
        public int collision_rule_errors;
    }
    [Serializable] sealed class PreviewStatus
    {
        public string status,checkpoint,checkpoint_sha256,error;
        public string task_name;
        public int pid,episodes,seed,task_index;
        public double updated_at;
    }
    [Serializable] sealed class CheckpointInfo
    {
        public string checkpoint,sha256,unity_build,physics_fingerprint,reward_hash,course_hash;
        public int stage,batch,workers,seed,updates,global_steps,task_index;
        public float learning_rate;
        public ReviewStatus review;
    }
    [Serializable] sealed class DoctorInfo
    {
        public bool unity_build_contract_matches,unity_schema_matches,cuda_available;
        public string unity_build_sha256,unity_build_error,reward_hash,course_hash;
    }
    [Serializable] sealed class StageGoals { public StageGoal[] stages; public Stage1Task[] stage1Tasks,stage2Tasks; }
    [Serializable] sealed class StageGoal { public string name,goal,metric; }
    [Serializable] sealed class Stage1Task { public string name; public float distance,timeLimit; }
    [Serializable] sealed class EpisodeSummary
    {
        public int episodes,dnf_total;
        public float completion,progress_m,falls,stationary_fraction,reward_progress,
            reward_posture,reward_time,reward_fall,reward_finish,reward_dnf,
            reward_gait,reward_bad_gait,reward_high_hoof,reward_fold,
            reward_participation,reward_coordination,reward_process,reward_cycle,
            reward_order,reward_synchrony,reward_missing,reward_drag,
            correct_cycle_ratio,bad_cycle_ratio,high_hoof_fraction,
            filtered_posture_m,filtered_ground_m,filtered_inactive_m,filtered_boundary_m;
    }

    RunStatus status;
    PreviewStatus preview;
    StageGoals goals;
    double lastRefresh;
    Vector2 scroll;
    static string Root => Directory.GetParent(Application.dataPath).FullName;
    static string Python => Path.Combine(Root,".venv-qhop","Scripts","python.exe");
    static string Player => Path.Combine(Root,"Builds","Training","QHopTraining.exe");
    static QHopTrainingConsoleState State => QHopTrainingConsoleState.instance;
    static string Run => Path.GetFullPath(Path.IsPathRooted(State.runDirectory)?
        State.runDirectory:Path.Combine(Root,State.runDirectory));

    static QHopTrainingConsole()
    {
        EditorApplication.quitting+=OnEditorQuit;
    }
    [MenuItem("QHOP/训练/打开训练控制台",priority=10)]
    public static void Open(){GetWindow<QHopTrainingConsole>("QHOP 训练控制台").Show();}
    [MenuItem("QHOP/训练/开始训练",priority=11)]
    public static void StartFromMenu(){Open();GetWindow<QHopTrainingConsole>().StartNew();}
    [MenuItem("QHOP/训练/结束训练并保存",priority=12)]
    public static void StopFromMenu(){Open();GetWindow<QHopTrainingConsole>().Stop();}
    [MenuItem("QHOP/训练/断点继续",priority=13)]
    public static void ResumeFromMenu(){Open();GetWindow<QHopTrainingConsole>().ResumeLatest();}
    [MenuItem("QHOP/训练/从 checkpoint 继续…",priority=14)]
    public static void ResumeChosenFromMenu(){Open();GetWindow<QHopTrainingConsole>().ResumeChosen();}
    [MenuItem("QHOP/训练/启动阶段验证",priority=15)]
    public static void PreviewFromMenu(){Open();GetWindow<QHopTrainingConsole>().StartPreview();}
    [MenuItem("QHOP/训练/从第一阶段迁移到步态训练…",priority=17)]
    public static void TransferFromMenu(){Open();GetWindow<QHopTrainingConsole>().TransferStage1();}
    [MenuItem("QHOP/训练/人工通过并进入下一阶段",priority=16)]
    public static void ApproveFromMenu(){Open();GetWindow<QHopTrainingConsole>().Approve();}

    void OnEnable()
    {
        if(string.IsNullOrEmpty(State.runDirectory))
        {
            State.runDirectory="Runs/editor-"+DateTime.Now.ToString("yyyyMMdd-HHmmss");
            State.Store();
        }
        var asset=AssetDatabase.LoadAssetAtPath<TextAsset>("Assets/Resources/QHopStageGoals.json");
        if(asset!=null)goals=JsonUtility.FromJson<StageGoals>(asset.text);
        EditorApplication.update+=Tick;
        Refresh();
    }
    void OnDisable(){EditorApplication.update-=Tick;}
    static void OnEditorQuit()
    {
        if(OwnedProcess(State.trainerPid,State.trainerStartTicks))
        {
            var path=Path.Combine(Run,"session-status.json");
            var current=Read<RunStatus>(path);
            if(current!=null && (current.pid==State.trainerPid ||
                State.trainerLaunchUnix>0 && current.updated_at>=State.trainerLaunchUnix-1))
                WriteStopRequest(current.session_id);
        }
        if(OwnedProcess(State.previewPid,State.previewStartTicks))KillTree(State.previewPid);
    }
    static T Read<T>(string path) where T:class
    {
        try
        {
            string text;
            using(var stream=new FileStream(path,FileMode.Open,FileAccess.Read,
                FileShare.ReadWrite|FileShare.Delete))
            using(var reader=new StreamReader(stream,Encoding.UTF8,true))
                text=reader.ReadToEnd();
            return JsonUtility.FromJson<T>(text);
        }
        catch(Exception){return null;}
    }
    static bool OwnedProcess(int pid,long ticks)
    {
        if(pid<=0 || ticks<=0)return false;
        try{using(var process=Process.GetProcessById(pid))
            return !process.HasExited && process.StartTime.Ticks==ticks;}
        catch(Exception){return false;}
    }
    static bool CandidateHashMatches(RunStatus current)
    {
        if(current==null || current.review==null ||
           string.IsNullOrEmpty(current.review.candidate) ||
           string.IsNullOrEmpty(current.review.candidate_sha256) ||
           !File.Exists(current.review.candidate))return false;
        using(var sha=SHA256.Create())
        using(var file=File.OpenRead(current.review.candidate))
            return string.Equals(BitConverter.ToString(sha.ComputeHash(file)).Replace("-", ""),
                current.review.candidate_sha256,StringComparison.OrdinalIgnoreCase);
    }
    static void KillTree(int pid)
    {
        var start=new ProcessStartInfo("taskkill.exe","/T /F /PID "+pid)
        {UseShellExecute=false,CreateNoWindow=true};
        using(var process=Process.Start(start))if(process!=null)process.WaitForExit(10000);
    }
    static double UnixNow(){return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds()/1000.0;}
    static bool Adopt(ref int pid,ref long ticks,int reportedPid,double updatedAt,double launchedAt)
    {
        if(reportedPid<=0 || launchedAt<=0 || updatedAt<launchedAt-1)return false;
        try
        {
            using(var process=Process.GetProcessById(reportedPid))
            {
                if(process.HasExited)return false;
                pid=reportedPid;ticks=process.StartTime.Ticks;return true;
            }
        }
        catch(Exception){return false;}
    }
    static string Quote(string value){return "\""+value.Replace("\"","\\\"")+"\"";}
    static string Args(params string[] values)
    {
        var builder=new StringBuilder();
        foreach(var value in values){if(builder.Length>0)builder.Append(' ');builder.Append(Quote(value));}
        return builder.ToString();
    }
    static ProcessStartInfo PythonStartInfo(bool training,params string[] args)
    {
        var start=new ProcessStartInfo(Python,"-m qhop_train.cli "+Args(args));
        start.WorkingDirectory=Root;
        start.UseShellExecute=false;
        start.CreateNoWindow=true;
        start.EnvironmentVariables["PYTHONPATH"]=Path.Combine(Root,"training");
        var speed=training?State.simulationSpeed:1f;
        if(float.IsNaN(speed) || float.IsInfinity(speed) || speed<1f || speed>MaxSimulationSpeed)
            throw new InvalidOperationException("仿真速度必须在 1–20 倍之间");
        start.EnvironmentVariables["QHOP_TRAIN_TIME_SCALE"]=speed.ToString(CultureInfo.InvariantCulture);
        return start;
    }
    static Process StartPython(bool training,params string[] args)
    {
        var start=PythonStartInfo(training,args);
        var process=Process.Start(start);
        if(process==null)throw new InvalidOperationException("Python process did not start");
        return process;
    }
    static string RunPython(out int code,params string[] args)
    {
        var start=PythonStartInfo(false,args);
        start.RedirectStandardOutput=true;
        start.RedirectStandardError=true;
        using(var process=Process.Start(start))
        {
            if(process==null)throw new InvalidOperationException("Python process did not start");
            var output=process.StandardOutput.ReadToEnd();
            var error=process.StandardError.ReadToEnd();
            if(!process.WaitForExit(120000)){process.Kill();throw new TimeoutException("Python preflight timeout");}
            code=process.ExitCode;
            return code==0?output:error+"\n"+output;
        }
    }
    void Fail(Exception error){State.notice=error.Message;State.Store();Repaint();}
    void Tick()
    {
        if(EditorApplication.timeSinceStartup-lastRefresh<.5)return;
        lastRefresh=EditorApplication.timeSinceStartup;
        Refresh();
        if(State.launchPreviewAfterStop && OwnedProcess(State.trainerPid,State.trainerStartTicks) &&
           status!=null && status.pid==State.trainerPid && status.status!="stopping")
            WriteStopRequest(status.session_id);
        if(State.launchPreviewAfterStop && !OwnedProcess(State.trainerPid,State.trainerStartTicks))
        {
            State.launchPreviewAfterStop=false;State.Store();
            try{StartPreviewNow();}catch(Exception error){Fail(error);}
        }
        if(State.stopRequestedUnix>0 &&
           OwnedProcess(State.trainerPid,State.trainerStartTicks) &&
           UnixNow()-State.stopRequestedUnix>30 &&
           (string.IsNullOrEmpty(State.notice) ||
            !State.notice.StartsWith("安全停止超时",StringComparison.Ordinal)))
        {
            State.notice="安全停止超时；查看 "+Path.Combine(Run,"console.log")+
                " 和 worker-logs，可强制结束本会话进程树";
            State.Store();
        }
        if(State.stopRequestedUnix>0 &&
           !OwnedProcess(State.trainerPid,State.trainerStartTicks))
        {State.stopRequestedUnix=0;State.Store();}
        Repaint();
    }
    void Refresh()
    {
        status=Read<RunStatus>(Path.Combine(Run,"session-status.json"));
        preview=Read<PreviewStatus>(Path.Combine(Run,"preview-status.json"));
        bool changed=false;
        if(status!=null && State.trainerLaunchUnix>0 &&
            Adopt(ref State.trainerPid,ref State.trainerStartTicks,status.pid,
                  status.updated_at,State.trainerLaunchUnix))
        {State.trainerLaunchUnix=0;changed=true;}
        if(preview!=null && State.previewLaunchUnix>0 &&
            Adopt(ref State.previewPid,ref State.previewStartTicks,preview.pid,
                  preview.updated_at,State.previewLaunchUnix))
        {State.previewLaunchUnix=0;changed=true;}
        if(changed)State.Store();
    }
    static void CheckPorts(bool training,int workers)
    {
        var ports=IPGlobalProperties.GetIPGlobalProperties().GetActiveTcpListeners();
        foreach(var endpoint in ports)
        {
            var port=endpoint.Port;
            if(training && port>=5105 && port<5105+workers || !training && port==5905)
                throw new InvalidOperationException("ML-Agents port "+port+" is occupied");
        }
    }
    static DoctorInfo Preflight()
    {
        if(!File.Exists(Python))throw new FileNotFoundException("项目虚拟环境不存在",Python);
        if(!File.Exists(Player))throw new FileNotFoundException("训练 Player 不存在，请先构建",Player);
        int code;
        var output=RunPython(out code,"doctor");
        if(code!=0)throw new InvalidOperationException("doctor 未通过："+output);
        var report=JsonUtility.FromJson<DoctorInfo>(output);
        if(!report.unity_build_contract_matches || !report.unity_schema_matches || !report.cuda_available)
            throw new InvalidOperationException("依赖、GPU 或构建契约不匹配："+output);
        return report;
    }
    static CheckpointInfo Inspect(string path,DoctorInfo doctor)
    {
        int code;
        var output=RunPython(out code,"checkpoint-info","--checkpoint",path);
        if(code!=0)throw new InvalidOperationException(output);
        var info=JsonUtility.FromJson<CheckpointInfo>(output);
        if(info.unity_build!=doctor.unity_build_sha256 ||
           info.reward_hash!=doctor.reward_hash || info.course_hash!=doctor.course_hash)
            throw new InvalidOperationException("Checkpoint 与当前训练 Player、奖励或课程契约不匹配");
        return info;
    }
    void StartNew()
    {
        try
        {
            if(OwnedProcess(State.trainerPid,State.trainerStartTicks))
                throw new InvalidOperationException("已有训练会话正在运行");
            if(OwnedProcess(State.previewPid,State.previewStartTicks))
                throw new InvalidOperationException("请先关闭阶段验证窗口");
            Preflight();CheckPorts(true,State.workers);
            if(File.Exists(Path.Combine(Run,"latest.pt")))
            {
                string directory;
                int suffix=0;
                do
                {
                    directory="Runs/editor-"+DateTime.Now.ToString("yyyyMMdd-HHmmss")+
                        (suffix==0?"":"-"+suffix);
                    suffix++;
                }
                while(Directory.Exists(Path.Combine(Root,directory)));
                State.runDirectory=directory;
                State.Store();
            }
            Directory.CreateDirectory(Run);
            var process=StartPython(true,"--log-file",Path.Combine(Run,"console.log"),
                "train","--output",Run,"--workers",State.workers.ToString(),
                "--batch",State.batch.ToString(),"--seed",State.seed.ToString());
            State.trainerPid=process.Id;State.trainerStartTicks=process.StartTime.Ticks;
            State.trainerLaunchUnix=UnixNow();
            State.stopRequestedUnix=0;
            State.notice="已启动 Stage 1；等待会话状态文件";State.Store();
        }
        catch(Exception error){Fail(error);}
    }
    void TransferStage1(bool adapt=false)
    {
        State.notice="r14 LiDAR/GRU 需要从头训练，不支持旧模型迁移。旧实验保持可读。";
        State.Store();Repaint();
    }
    void StartResume(string checkpoint)
    {
        if(OwnedProcess(State.trainerPid,State.trainerStartTicks))
            throw new InvalidOperationException("已有训练会话正在运行");
        if(OwnedProcess(State.previewPid,State.previewStartTicks))
            throw new InvalidOperationException("请先关闭阶段验证窗口");
        var doctor=Preflight();
        var info=Inspect(checkpoint,doctor);
        if(info.review!=null && info.review.status!="training")
            throw new InvalidOperationException("此阶段待人工审核或已结束；请先验证、通过或选择继续本阶段");
        State.batch=info.batch;State.workers=info.workers>0?info.workers:4;
        State.seed=info.seed;
        CheckPorts(true,State.workers);
        Directory.CreateDirectory(Run);
        var process=StartPython(true,"--log-file",Path.Combine(Run,"console.log"),
            "resume","--checkpoint",checkpoint,"--output",Run,
            "--workers",State.workers.ToString(),"--batch",State.batch.ToString(),
            "--seed",State.seed.ToString(),"--extra-steps",State.extraSteps.ToString());
        State.trainerPid=process.Id;State.trainerStartTicks=process.StartTime.Ticks;
        State.trainerLaunchUnix=UnixNow();
        State.stopRequestedUnix=0;
        State.extraSteps=0;
        State.notice="正在完整恢复；Adam、学习率和课程状态由 checkpoint 读取";
        State.Store();
    }
    void ResumeLatest()
    {
        try
        {
            var path=Path.Combine(Run,"latest.pt");
            if(!File.Exists(path))throw new FileNotFoundException("当前实验没有 latest.pt",path);
            StartResume(path);
        }
        catch(Exception error){Fail(error);}
    }
    void ResumeChosen()
    {
        try
        {
            var chosen=EditorUtility.OpenFilePanel("选择完整恢复 checkpoint",Run,"pt");
            if(string.IsNullOrEmpty(chosen))return;
            var original=Run;
            var latest=Path.Combine(original,"latest.pt");
            if(!string.Equals(Path.GetFullPath(chosen),Path.GetFullPath(latest),StringComparison.OrdinalIgnoreCase))
            {
                var branch=original+"-branch-"+DateTime.Now.ToString("yyyyMMdd-HHmmss");
                State.runDirectory=branch;
                State.notice="较早 checkpoint 将恢复到新实验分支："+branch;
                State.Store();
            }
            StartResume(chosen);
        }
        catch(Exception error){Fail(error);}
    }
    static void AtomicWrite(string path,string text)
    {
        var temporary=path+".tmp-"+Guid.NewGuid().ToString("N");
        File.WriteAllText(temporary,text,new UTF8Encoding(false));
        if(File.Exists(path))File.Replace(temporary,path,null);
        else File.Move(temporary,path);
    }
    static void WriteStopRequest(string sessionId)
    {
        if(string.IsNullOrEmpty(sessionId))return;
        var path=Path.Combine(Run,"session-request.json");
        var text="{\"session_id\":\""+sessionId+"\",\"sequence\":"+
            DateTime.UtcNow.Ticks+",\"command\":\"stop\"}";
        AtomicWrite(path,text);
    }
    void Stop()
    {
        try
        {
            Refresh();
            if(status==null || status.pid!=State.trainerPid ||
               !OwnedProcess(State.trainerPid,State.trainerStartTicks))
                throw new InvalidOperationException("没有可停止的本项目训练会话");
            WriteStopRequest(status.session_id);
            State.stopRequestedUnix=UnixNow();
            State.notice="已请求安全停止；更新完成后保存，采样中的 rollout 将丢弃";
            State.Store();
        }
        catch(Exception error){Fail(error);}
    }
    void ForceStop()
    {
        try
        {
            if(!OwnedProcess(State.trainerPid,State.trainerStartTicks))
                throw new InvalidOperationException("本会话进程已结束");
            KillTree(State.trainerPid);
            State.stopRequestedUnix=0;
            State.notice="已强制结束本会话进程树；仅 latest.pt 的上一个完整更新可恢复";
            State.Store();
        }
        catch(Exception error){Fail(error);}
    }
    void StartPreview()
    {
        try
        {
            if(OwnedProcess(State.previewPid,State.previewStartTicks))
                throw new InvalidOperationException("已有可视化验证窗口");
            if(OwnedProcess(State.trainerPid,State.trainerStartTicks))
            {
                State.launchPreviewAfterStop=true;State.Store();Stop();return;
            }
            StartPreviewNow();
        }
        catch(Exception error){Fail(error);}
    }
    void StartPreviewNow()
    {
        var doctor=Preflight();CheckPorts(false,1);
        Refresh();
        var path=status!=null && status.review!=null && status.review.status=="pending"?
            status.review.candidate:Path.Combine(Run,"latest.pt");
        if(!File.Exists(path))throw new FileNotFoundException("验证 checkpoint 不存在",path);
        Inspect(path,doctor);
        var process=StartPython(false,"--log-file",Path.Combine(Run,"preview-console.log"),
            "preview","--checkpoint",path,"--output",Run,
            "--seed",State.previewSeed.ToString());
        State.previewPid=process.Id;State.previewStartTicks=process.StartTime.Ticks;
        State.previewLaunchUnix=UnixNow();
        State.notice="已启动独立可见的阶段验证 Player";State.Store();
    }
    void Approve()
    {
        try
        {
            Refresh();
            if(OwnedProcess(State.trainerPid,State.trainerStartTicks) ||
               OwnedProcess(State.previewPid,State.previewStartTicks))
                throw new InvalidOperationException("请先结束训练和验证窗口");
            if(status==null || status.review==null || status.review.status!="pending" ||
               status.passes<3 || status.review.preview_episodes<1 ||
               !CandidateHashMatches(status))
                throw new InvalidOperationException("自动评估达标且至少完成一回合可视化验证后才可通过");
            int code;
            var output=RunPython(out code,"approve-stage","--output",Run);
            if(code!=0)throw new InvalidOperationException(output);
            State.notice="人工审核已记录；下一阶段尚未启动，请手动点击断点继续";
            State.Store();Refresh();
        }
        catch(Exception error){Fail(error);}
    }
    void ContinueStage()
    {
        try
        {
            if(OwnedProcess(State.trainerPid,State.trainerStartTicks) ||
               OwnedProcess(State.previewPid,State.previewStartTicks))
                throw new InvalidOperationException("请先结束当前会话");
            int code;
            var output=RunPython(out code,"continue-stage","--output",Run,
                "--extra-steps",State.extraSteps.ToString());
            if(code!=0)throw new InvalidOperationException(output);
            State.notice="候选资格和达标计数已清除；可点击断点继续";State.Store();Refresh();
            State.extraSteps=0;State.Store();
        }
        catch(Exception error){Fail(error);}
    }
    void OnGUI()
    {
        var state=State;
        scroll=EditorGUILayout.BeginScrollView(scroll);
        EditorGUILayout.LabelField("训练会话",EditorStyles.boldLabel);
        EditorGUI.BeginDisabledGroup(OwnedProcess(state.trainerPid,state.trainerStartTicks) ||
            OwnedProcess(state.previewPid,state.previewStartTicks));
        state.runDirectory=EditorGUILayout.TextField("实验目录",state.runDirectory);
        EditorGUI.EndDisabledGroup();
        state.seed=EditorGUILayout.IntField("随机种子",state.seed);
        state.workers=EditorGUILayout.IntSlider("并行进程",state.workers,1,MaxWorkers);
        state.simulationSpeed=EditorGUILayout.Slider("仿真速度（倍）",state.simulationSpeed,1f,MaxSimulationSpeed);
        state.batch=Mathf.Max(1,EditorGUILayout.IntField("Batch",state.batch));
        state.previewSeed=EditorGUILayout.IntField("验证 seed",state.previewSeed);
        state.extraSteps=Mathf.Max(0,EditorGUILayout.IntField("追加阶段步数",state.extraSteps));
        EditorGUILayout.HelpBox("进程数和仿真速度在下次启动训练时生效；调整速度需先结束训练并保存，再断点继续。断点继续沿用 checkpoint 的进程数，阶段验证保持实时。",MessageType.Info);
        if(GUI.changed)state.Store();
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("状态",status==null?"无会话":status.status);
        if(status!=null)
        {
            EditorGUILayout.LabelField("阶段",status.stage.ToString());
            if(status.stage==1 && goals!=null && goals.stage1Tasks!=null &&
               status.task_index>=0 && status.task_index<goals.stage1Tasks.Length)
            {
                var current=goals.stage1Tasks[status.task_index];
                EditorGUILayout.LabelField("当前子任务",current.name+"；"+
                    current.distance.ToString("F0")+" 米 / "+
                    current.timeLimit.ToString("F0")+" 秒；最终目标 20 米 / 20 秒");
            }
            if(status.stage==2 && goals!=null && goals.stage2Tasks!=null &&
               status.task_index>=0 && status.task_index<goals.stage2Tasks.Length)
            {
                var current=goals.stage2Tasks[status.task_index];
                EditorGUILayout.LabelField("当前子任务",current.name+"；"+
                    current.distance.ToString("F0")+" 米 / "+
                    current.timeLimit.ToString("F0")+" 秒");
            }
            EditorGUILayout.LabelField("累计步数 / 阶段步数",status.global_steps+" / "+status.phase_steps);
            var budgets=new[]{3000000,8000000,8000000,12000000,15000000,15000000,10000000};
            if(status.stage>=1 && status.stage<=7)
                EditorGUILayout.LabelField("阶段预算",budgets[status.stage-1].ToString());
            EditorGUILayout.LabelField("PPO 更新 / 学习率",status.updates+" / "+status.learning_rate.ToString("G6"));
            EditorGUILayout.LabelField("有效采样率",status.horse_steps_per_second.ToString("F1")+" 马步/秒");
            int requiredPasses=status.stage==2 && status.task_index<2?2:3;
            EditorGUILayout.LabelField("自动评估连续达标",status.passes+" / "+requiredPasses);
            if(status.recent100!=null && status.recent100.episodes>0)
            {
                var recent=status.recent100;
                EditorGUILayout.LabelField("最近 "+recent.episodes+" 回合",
                    "完赛 "+recent.completion.ToString("P1")+"；前进 "+
                    recent.progress_m.ToString("F2")+" m；摔倒 "+recent.falls.ToString("F2")+
                    "；静止 "+recent.stationary_fraction.ToString("P1")+
                    "；DNF "+recent.dnf_total,EditorStyles.wordWrappedLabel);
                EditorGUILayout.LabelField("平均奖励贡献",
                    "前进 "+recent.reward_progress.ToString("F2")+
                    "；姿态 "+recent.reward_posture.ToString("F2")+
                    "；时间 "+recent.reward_time.ToString("F2")+
                    "；摔倒 "+recent.reward_fall.ToString("F2")+
                    "；完赛 "+recent.reward_finish.ToString("F2")+
                    "；DNF "+recent.reward_dnf.ToString("F2"),EditorStyles.wordWrappedLabel);
                EditorGUILayout.LabelField("步态奖励与约束",
                    "正向步态 "+recent.reward_gait.ToString("F2")+
                    "；错误 "+recent.reward_bad_gait.ToString("F2")+
                    "；高抬蹄 "+recent.reward_high_hoof.ToString("F2")+
                    "；折叠 "+recent.reward_fold.ToString("F2"),EditorStyles.wordWrappedLabel);
                EditorGUILayout.LabelField("步态奖励分项",
                    "参与 "+recent.reward_participation.ToString("F2")+"；协调 "+recent.reward_coordination.ToString("F2")+
                    "；顺序进展 "+recent.reward_process.ToString("F2")+"；完整周期 "+recent.reward_cycle.ToString("F2"),EditorStyles.wordWrappedLabel);
                EditorGUILayout.LabelField("错误惩罚分项",
                    "乱序 "+recent.reward_order.ToString("F2")+"；同步 "+recent.reward_synchrony.ToString("F2")+
                    "；缺腿 "+recent.reward_missing.ToString("F2")+"；拖蹄 "+recent.reward_drag.ToString("F2"),EditorStyles.wordWrappedLabel);
                EditorGUILayout.LabelField("过滤的前进位移",
                    "姿态 "+recent.filtered_posture_m.ToString("F2")+
                    " m；躯干接地 "+recent.filtered_ground_m.ToString("F2")+
                    " m；罚停 "+recent.filtered_inactive_m.ToString("F2")+
                    " m；越界 "+recent.filtered_boundary_m.ToString("F2")+" m",
                    EditorStyles.wordWrappedLabel);
            }
            if(status.last_evaluation!=null && status.last_evaluation.stage==status.stage)
            {
                var evaluation=status.last_evaluation;
                if(status.stage==1 && goals!=null && goals.stage1Tasks!=null &&
                   evaluation.task_index>=0 && evaluation.task_index<goals.stage1Tasks.Length)
                    EditorGUILayout.LabelField("最近评估子任务",
                        goals.stage1Tasks[evaluation.task_index].name);
                if(status.stage==2 && goals!=null && goals.stage2Tasks!=null &&
                   evaluation.task_index>=0 && evaluation.task_index<goals.stage2Tasks.Length)
                    EditorGUILayout.LabelField("最近评估子任务",
                        goals.stage2Tasks[evaluation.task_index].name);
                EditorGUILayout.LabelField("最近评估", "完赛 "+evaluation.completion.ToString("P1")+
                    "；无摔倒 "+evaluation.fall_free.ToString("P1")+
                    "；中位速度 "+evaluation.median_speed.ToString("F2")+" m/s"+
                    "；中位时间 "+evaluation.median_time.ToString("F2")+" s"+
                    "；摔倒/100m "+evaluation.falls_per_100m.ToString("F2"),
                    EditorStyles.wordWrappedLabel);
                EditorGUILayout.LabelField("步态评估",
                    "四腿参与 "+evaluation.four_leg_participation.ToString("P1")+"；同步拍地 "+evaluation.synchrony_cycle_ratio.ToString("P1")+"；速度达标 "+evaluation.speed_band_fraction.ToString("P1")+"；覆盖 "+evaluation.gait_coverage.ToString("P1")+
                    "；正确 "+evaluation.correct_cycle_ratio.ToString("P1")+
                    "；错误 "+evaluation.bad_cycle_ratio.ToString("P1")+
                    "；高抬蹄 "+evaluation.high_hoof_fraction.ToString("P1"));
                if(status.stage==4)EditorGUILayout.LabelField("最差体型组完赛率",
                    evaluation.worst_morphology_completion.ToString("P1"));
                if(status.stage==5)EditorGUILayout.LabelField("扰动恢复率",
                    evaluation.disturbance_recovery.ToString("P1"));
                if(status.stage==6)EditorGUILayout.LabelField("碰撞规则异常",
                    evaluation.collision_rule_errors.ToString());
            }
            EditorGUILayout.LabelField("最近 checkpoint",status.checkpoint??"");
            if(status.review!=null)
            {
                EditorGUILayout.LabelField("人工审核",status.review.status??"未审核");
                EditorGUILayout.LabelField("候选 checkpoint",status.review.candidate??"");
                EditorGUILayout.LabelField("已完成验证回合",status.review.preview_episodes.ToString());
                if(!string.IsNullOrEmpty(status.review.approved_at))
                    EditorGUILayout.LabelField("人工通过时间",status.review.approved_at);
            }
            if(status.approvals!=null && status.approvals.Length>0)
            {
                var approval=status.approvals[status.approvals.Length-1];
                EditorGUILayout.LabelField("最近人工审核","Stage "+approval.stage+
                    "；"+approval.approved_at+"；验证回合 "+approval.preview_episodes,
                    EditorStyles.wordWrappedLabel);
                EditorGUILayout.LabelField("批准的候选 SHA-256",approval.candidate_sha256??"");
            }
            if(!string.IsNullOrEmpty(status.error))EditorGUILayout.HelpBox(status.error,MessageType.Error);
        }
        if(preview!=null)
        {
            EditorGUILayout.LabelField("验证窗口",preview.status+"；"+
                (preview.task_name??"")+"；完成 "+preview.episodes+" 回合");
            EditorGUILayout.LabelField("验证 checkpoint",preview.checkpoint??"");
            if(!string.IsNullOrEmpty(preview.error))EditorGUILayout.HelpBox(preview.error,MessageType.Error);
        }
        if(!string.IsNullOrEmpty(state.notice))EditorGUILayout.HelpBox(state.notice,MessageType.Info);
        EditorGUILayout.Space();
        EditorGUILayout.BeginHorizontal();
        if(GUILayout.Button("开始训练"))StartNew();
        if(GUILayout.Button("结束训练并保存"))Stop();
        if(GUILayout.Button("断点继续"))ResumeLatest();
        EditorGUILayout.EndHorizontal();
        EditorGUILayout.BeginHorizontal();
        if(GUILayout.Button("从 checkpoint 继续…"))ResumeChosen();
        EditorGUILayout.HelpBox("r14：从头训练；旧 checkpoint 不可迁移或续训。",MessageType.Info);

        if(GUILayout.Button("启动阶段验证"))StartPreview();
        EditorGUI.BeginDisabledGroup(status==null || status.review==null ||
            status.review.status!="pending" || status.passes<3 ||
            status.review.preview_episodes<1 || !CandidateHashMatches(status));
        if(GUILayout.Button("人工通过并进入下一阶段"))Approve();
        EditorGUI.EndDisabledGroup();
        EditorGUILayout.EndHorizontal();
        EditorGUILayout.BeginHorizontal();
        if(GUILayout.Button("继续本阶段训练"))ContinueStage();
        if(GUILayout.Button("打开实验目录")){Directory.CreateDirectory(Run);EditorUtility.RevealInFinder(Run);}
        if(GUILayout.Button("查看日志"))EditorUtility.RevealInFinder(Path.Combine(Run,"console.log"));
        EditorGUI.BeginDisabledGroup(OwnedProcess(state.trainerPid,state.trainerStartTicks) ||
            OwnedProcess(state.previewPid,state.previewStartTicks));
        if(GUILayout.Button("构建训练环境"))BuildTraining.Build();
        EditorGUI.EndDisabledGroup();
        EditorGUILayout.EndHorizontal();
        if(OwnedProcess(state.trainerPid,state.trainerStartTicks) &&
            state.stopRequestedUnix>0 && UnixNow()-state.stopRequestedUnix>30 &&
            GUILayout.Button("强制结束本会话进程树（仅安全停止超时使用）"))ForceStop();
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("七阶段目标",EditorStyles.boldLabel);
        if(goals!=null && goals.stages!=null)
            for(int i=0;i<goals.stages.Length;i++)
            {
                var goal=goals.stages[i];
                var stage=i+1;
                var label=status==null || stage>status.stage?"未开始":
                    stage<status.stage?"已通过":
                    status.review!=null && status.review.status=="pending"?"待人工验证":"训练中";
                EditorGUILayout.LabelField(goal.name+"  ["+label+"]",EditorStyles.boldLabel);
                EditorGUILayout.LabelField(goal.goal,EditorStyles.wordWrappedLabel);
                EditorGUILayout.LabelField("自动目标："+goal.metric,EditorStyles.wordWrappedLabel);
            }
        EditorGUILayout.EndScrollView();
    }
}
