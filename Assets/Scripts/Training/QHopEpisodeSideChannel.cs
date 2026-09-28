using System;
using Unity.MLAgents.SideChannels;

namespace QHop
{
    public sealed class QHopEpisodeSideChannel : SideChannel
    {
        public const string Id="4ed887c2-8036-4d8c-b112-07915209a1c4";

        public QHopEpisodeSideChannel()
        {
            ChannelId=new Guid(Id);
        }

        protected override void OnMessageReceived(IncomingMessage message) {}

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
