"""WebSocket: dashboard stream (telemetry + preview frames)."""

from __future__ import annotations

import json
from typing import Any

from channels.generic.websocket import AsyncWebsocketConsumer


class DashboardConsumer(AsyncWebsocketConsumer):
    async def connect(self) -> None:
        await self.channel_layer.group_add("dashboard", self.channel_name)
        await self.accept()

    async def disconnect(self, code: int) -> None:
        await self.channel_layer.group_discard("dashboard", self.channel_name)

    async def send_telemetry(self, event: dict[str, Any]) -> None:
        payload = event.get("payload", {})
        await self.send(text_data=json.dumps(payload))
