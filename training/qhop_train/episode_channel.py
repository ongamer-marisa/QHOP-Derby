"""Dedicated Unity episode diagnostics, separate from policy observations."""
from collections import deque
import json
import uuid

from mlagents_envs.side_channel.side_channel import SideChannel


CHANNEL_ID = uuid.UUID("4ed887c2-8036-4d8c-b112-07915209a1c4")


class EpisodeChannel(SideChannel):
    def __init__(self):
        super().__init__(CHANNEL_ID)
        self.events = deque()

    def on_message_received(self, message):
        event = json.loads(message.read_string())
        if event.get("version") != 1 or not event.get("episodeId"):
            raise ValueError("Unknown Unity episode diagnostics version")
        self.events.append(event)

    def drain(self):
        result = list(self.events)
        self.events.clear()
        return result
