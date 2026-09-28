using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace QHop
{
    public sealed class HorseTrainingAgent : Agent
    {
        public HorseActor Horse { get; private set; }
        public HorseTrainingArea Area { get; private set; }
        readonly HorseObservationHistory history=new HorseObservationHistory();
        int seenFalls;
        public void Bind(HorseTrainingArea area,HorseActor horse)
        {
            Area=area;Horse=horse;
            seenFalls=0;
            history.Reset(HorseTrainingObservation.Frame(horse,
                area.Race.Results[horse.Id],area.TimeRemaining,area.Distance,
                area.Disturbances[horse.Id],area.Recoveries[horse.Id]));
        }
        public void ResetHistory()
        {
            if(Horse!=null && Area!=null)
                history.Reset(HorseTrainingObservation.Frame(Horse,
                    Area.Race.Results[Horse.Id],Area.TimeRemaining,Area.Distance,
                    Area.Disturbances[Horse.Id],Area.Recoveries[Horse.Id]));
        }
        public override void CollectObservations(VectorSensor sensor)
        {
            var frame=HorseTrainingObservation.Frame(Horse,
                Area.Race.Results[Horse.Id],Area.TimeRemaining,Area.Distance,
                Area.Disturbances[Horse.Id],Area.Recoveries[Horse.Id]);
            int falls=Area.Race.Results[Horse.Id].falls;
            if(Horse.Condition==HorseCondition.Standing && falls!=seenFalls)
            {history.Reset(frame);seenFalls=falls;}
            float[] observation=history.Build(frame,HorseTrainingObservation.TerrainGrid(Horse),
                HorseTrainingObservation.OpponentData(Horse,Area.Horses,Area.Race.Results));
            if(observation.Length!=HorseTrainingObservation.Size)
                throw new System.InvalidOperationException("Training observation schema mismatch");
            sensor.AddObservation(observation);
        }
        public override void OnActionReceived(ActionBuffers actions)
        {
            if(Horse==null || !Horse.IsPolicyActive)return;
            var action=new HorseAction();
            for(int i=0;i<HorseAction.Size;i++)
                action.values[i]=Mathf.Clamp(actions.ContinuousActions[i],-1f,1f);
            Horse.SetExternalAction(action);
        }
        public override void Heuristic(in ActionBuffers actionsOut)
        {
            var actions=actionsOut.ContinuousActions;
            for(int i=0;i<HorseAction.Size;i++)actions[i]=i>=HorseAction.AngleCount?.7f:0f;
        }
    }
}
