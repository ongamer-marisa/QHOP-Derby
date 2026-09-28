using System;
using System.IO;
using System.Threading;
using UnityEditor;
using UnityEngine;

// Batchmode integration probe. It starts one tiny editor-controlled session and
// requests safe stop during sampling; it never performs formal training.
public static class QHopTrainingMenuProbe
{
    [Serializable] sealed class Status
    {
        public string status;
        public int updates;
    }
    public static void Run()
    {
        var state=QHopTrainingConsoleState.instance;
        var oldDirectory=state.runDirectory;
        var oldWorkers=state.workers;
        var oldBatch=state.batch;
        var oldSeed=state.seed;
        var oldNotice=state.notice;
        bool success=false;
        try
        {
        state.runDirectory="Runs/editor-menu-probe-"+DateTime.Now.ToString("yyyyMMdd-HHmmss");
        state.workers=1;
        state.batch=4096;
        state.seed=20260927;
        state.Store();
        if(!EditorApplication.ExecuteMenuItem("QHOP/训练/打开训练控制台"))
            throw new InvalidOperationException("QHOP training menu is missing");
        var root=Directory.GetParent(Application.dataPath).FullName;
        var directory=Path.Combine(root,state.runDirectory);
        var statusPath=Path.Combine(directory,"session-status.json");
        QHopTrainingConsole.StartFromMenu();
        bool sampling=false;
        for(int i=0;i<240;i++)
        {
            Thread.Sleep(250);
            if(!File.Exists(statusPath))continue;
            var current=JsonUtility.FromJson<Status>(File.ReadAllText(statusPath));
            if(current.status=="sampling"){sampling=true;break;}
            if(current.status=="failed")break;
        }
        if(!sampling)throw new InvalidOperationException("Editor menu did not start sampling");
        QHopTrainingConsole.StopFromMenu();
        for(int i=0;i<240;i++)
        {
            Thread.Sleep(250);
            var current=JsonUtility.FromJson<Status>(File.ReadAllText(statusPath));
            if(current.status=="stopped")
            {
                if(!File.Exists(Path.Combine(directory,"latest.pt")))
                    throw new InvalidOperationException("Safe stop did not retain a checkpoint");
                Debug.Log("QHOP_MENU_PROBE_PASS updates="+current.updates+" directory="+directory);
                success=true;
                return;
            }
            if(current.status=="failed")break;
        }
        throw new InvalidOperationException("Editor menu safe stop did not finish");
        }
        finally
        {
            if(success)
            {
                state.runDirectory=oldDirectory;
                state.workers=oldWorkers;
                state.batch=oldBatch;
                state.seed=oldSeed;
                state.notice=oldNotice;
                state.trainerPid=0;
                state.trainerStartTicks=0;
                state.trainerLaunchUnix=0;
                state.stopRequestedUnix=0;
                state.Store();
            }
        }
    }
}
