"""Channels WebSocket URL patterns."""

from django.urls import re_path

from api.consumers import DashboardConsumer

websocket_urlpatterns = [
    re_path(r"ws/stream/$", DashboardConsumer.as_asgi()),
]
