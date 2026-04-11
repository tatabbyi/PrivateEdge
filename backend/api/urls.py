"""REST URL routes."""

from django.urls import path

from api.views import ConfigView, HealthView, LogsView, StatusView

urlpatterns = [
    path("config/", ConfigView.as_view()),
    path("status/", StatusView.as_view()),
    path("logs/", LogsView.as_view()),
    path("health/", HealthView.as_view()),
]
