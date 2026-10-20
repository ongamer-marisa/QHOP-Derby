"""Compile Unity scripts against locally bundled assemblies when Editor licensing blocks imports."""
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
editor = Path(r"C:\Program Files\Unity\Hub\Editor\6000.0.45f1\Editor\Data")
compiler = Path(r"C:\Program Files\dotnet\sdk\8.0.403\Roslyn\bincore\csc.dll")
engine = editor / "Managed/UnityEngine"
template = editor / "Resources/PackageManager/ProjectTemplates/libcache/com.unity.template.3d-cross-platform-17.0.11/ScriptAssemblies"
refs = list(engine.glob("*.dll"))
refs += list(Path(r"C:\Program Files\dotnet\packs\Microsoft.NETCore.App.Ref\8.0.10\ref\net8.0").glob("*.dll"))
if not any("System.Runtime.dll" in str(ref) for ref in refs):
    refs += list(Path(r"C:\Program Files\dotnet\packs\Microsoft.NETCore.App.Ref").glob("*/ref/net8.0/*.dll"))
refs += [editor / "Managed/UnityEditor/UnityEditor.CoreModule.dll"]
refs += list((editor / "Managed/UnityEditor").glob("*.dll"))
refs += [template / "Unity.TextMeshPro.dll"]
refs += list(template.glob("UnityEngine.UI.dll"))
refs += list(template.glob("Unity.ugui.dll"))
tmp = root / "Temp"
tmp.mkdir(exist_ok=True)
core_only = "--core" in sys.argv
stub_external = "--stubs" in sys.argv
sources = list((root / "Assets/Scripts").rglob("*.cs")) + list((root / "Assets/Editor").rglob("*.cs"))
if core_only:
    sources = [source for source in sources if source.name not in
               ("HorsePolicyRunner.cs", "BuildTraining.cs", "QHopInferenceProbe.cs", "QHopR14Probe.cs",
                "QHopJointProbe.cs", "QHopTrainingConsole.cs", "QHopTrainingMenuProbe.cs")
               and "Training" not in source.parts]
    stub = root / "Temp/PolicyRunnerStub.cs"
    stub.write_text("""namespace QHop {
public sealed class HorsePolicyRunner : UnityEngine.MonoBehaviour {
  public string ModelName => "stub";
  public bool Load(System.Collections.Generic.IReadOnlyList<HorseActor> horses,
    HorseRaceSession race) => false;
  public IHorseController Controller(int index) => new StandingPoseController();
}}""", encoding="utf-8")
    sources.append(stub)
elif stub_external:
    stub = root / "Temp/ExternalMlStubs.cs"
    stub.write_text("""namespace Unity.MLAgents {
  public class Agent : UnityEngine.MonoBehaviour {
    public virtual void CollectObservations(Unity.MLAgents.Sensors.VectorSensor s) {}
    public virtual void OnActionReceived(Unity.MLAgents.Actuators.ActionBuffers a) {}
    public virtual void Heuristic(in Unity.MLAgents.Actuators.ActionBuffers a) {}
    public void RequestDecision() {} public void AddReward(float x) {} public void EndEpisode() {}
  }
  public class Academy { public static Academy Instance => new Academy(); public static bool IsInitialized=>true;
    public EnvironmentParameters EnvironmentParameters => new EnvironmentParameters(); }
  public class EnvironmentParameters { public float GetWithDefault(string x,float y)=>y; }
}
namespace Unity.MLAgents.Sensors { public class VectorSensor { public void AddObservation(float[] x) {} } }
namespace Unity.MLAgents.SideChannels {
  public abstract class SideChannel { protected System.Guid ChannelId {get;set;}
    protected abstract void OnMessageReceived(IncomingMessage message);
    protected void QueueMessageToSend(OutgoingMessage message) {} }
  public class IncomingMessage { public string ReadString()=>""; }
  public class OutgoingMessage : System.IDisposable { public void WriteString(string value) {} public void Dispose() {} }
  public static class SideChannelManager { public static void RegisterSideChannel(SideChannel value) {}
    public static void UnregisterSideChannel(SideChannel value) {} }
}
namespace Unity.MLAgents.Actuators {
  public struct ActionSegment<T> { public T this[int x] { get=>default(T);set{} } }
  public struct ActionBuffers { public ActionSegment<float> ContinuousActions=>new ActionSegment<float>(); }
  public struct ActionSpec { public static ActionSpec MakeContinuous(int x)=>new ActionSpec(); }
}
namespace Unity.MLAgents.Policies {
  public enum BehaviorType { Default }
  public class BrainParameters { public int VectorObservationSize; public int NumStackedVectorObservations;
    public Unity.MLAgents.Actuators.ActionSpec ActionSpec; }
  public class BehaviorParameters : UnityEngine.MonoBehaviour { public string BehaviorName;
    public BehaviorType BehaviorType; public BrainParameters BrainParameters=new BrainParameters(); }
}
namespace Unity.InferenceEngine {
  public class ModelAsset : UnityEngine.Object {}
  public class Model {} public enum BackendType { CPU }
  public class TensorShape { public TensorShape(int a,int b) {} }
  public class Tensor<T> : System.IDisposable { public Tensor(TensorShape s,T[] a) {}
    public T[] DownloadToArray()=>new T[0]; public void Dispose() {} }
  public class Worker : System.IDisposable { public Worker(Model m,BackendType b) {}
    public void SetInput(string name,Tensor<float> x) {}
    public void Schedule() {} public void Schedule(Tensor<float> x) {}
    public object PeekOutput()=>null; public object PeekOutput(string name)=>null;
    public void Dispose() {} }
  public static class ModelLoader { public static Model Load(ModelAsset a)=>new Model(); }
}""", encoding="utf-8")
    sources.append(stub)
else:
    installed = list((root / "Library/ScriptAssemblies").glob("Unity.MLAgents.dll"))
    installed += list((root / "Library/ScriptAssemblies").glob("Unity.InferenceEngine.dll"))
    if len(installed) < 2:
        raise SystemExit("ML-Agents/Inference Engine assemblies are not imported. "
                         "Activate Unity and import packages, or use --core for shared physics code.")
    refs += installed
args = ["/nologo", "/langversion:latest", "/target:library", f"/out:{tmp / 'QHopCheck.dll'}"]
args.extend(f"/reference:{ref}" for ref in refs if ref.exists())
args.extend(str(source) for source in sources)
response=tmp / "compile.rsp"
response.write_text("\n".join('"'+argument+'"' if ' ' in argument else argument for argument in args),encoding="utf-8")
result = subprocess.run(["dotnet",str(compiler),"@"+str(response)], capture_output=True, text=True, encoding="utf-8", errors="replace")
print(result.stdout, result.stderr)
raise SystemExit(result.returncode)
