using System;
using UnityEngine;
using Unity.MLAgents.SideChannels;

namespace QHop
{
    public sealed class QHopEpisodeSideChannel : SideChannel
    {
        public const string Id="4ed887c2-8036-4d8c-b112-07915209a1c4";
        public static int VisionUpdateMin,VisionUpdateMax;
        [Serializable] sealed class VisionDebug {public string kind;public int minimum,maximum;}

        public QHopEpisodeSideChannel()
        {
            ChannelId=new Guid(Id);
        }

        protected override void OnMessageReceived(IncomingMessage message)
        {
            var value=JsonUtility.FromJson<VisionDebug>(message.ReadString());
            if(value!=null && value.kind=="vision_debug")
            {VisionUpdateMin=value.minimum;VisionUpdateMax=value.maximum;}
        }

        public void Send(string json)
        {
            using(var message=new OutgoingMessage())
            {
                message.WriteString(json);
                QueueMessageToSend(message);
            }
        }
    }
}
