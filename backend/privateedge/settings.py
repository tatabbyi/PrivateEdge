"""Django settings — PrivateEdge control plane (REST + Channels)."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = BASE_DIR.parent
SRC_ROOT = REPO_ROOT / "src"
_src = str(SRC_ROOT)
if _src not in sys.path:
    sys.path.insert(0, _src)

SECRET_KEY = "privateedge-dev-key-change-in-production"
DEBUG = True
ALLOWED_HOSTS: list[str] = ["*"]

# Settings rev 3 — must include auth + UNAUTHENTICATED_USER None for DRF (restart daphne after edits).

# `django.contrib.auth` is required: DRF accesses `request.user` and defaults to
# AnonymousUser unless UNAUTHENTICATED_USER is None. Without `auth`, Permission
# models raise RuntimeError at import time.
INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.auth",
    "django.contrib.sessions",
    "django.contrib.staticfiles",
    "rest_framework",
    "channels",
    "corsheaders",
    "api.apps.ApiConfig",
]

MIDDLEWARE = [
    "corsheaders.middleware.CorsMiddleware",
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
]

ROOT_URLCONF = "privateedge.urls"
WSGI_APPLICATION = "privateedge.wsgi.application"
ASGI_APPLICATION = "privateedge.asgi.application"

TEMPLATES: list[dict] = []

DATABASES = {"default": {"ENGINE": "django.db.backends.sqlite3", "NAME": ":memory:"}}

AUTH_PASSWORD_VALIDATORS: list[dict] = []

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": [
        "rest_framework.renderers.JSONRenderer",
    ],
    "DEFAULT_AUTHENTICATION_CLASSES": [],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.AllowAny",
    ],
    # Avoid importing django.contrib.auth.models when `auth` is misconfigured;
    # unauthenticated requests use request.user = None.
    "UNAUTHENTICATED_USER": None,
}

# In dev, allow any origin so the UI works on LAN IPs / hostnames (not only loopback).
if DEBUG:
    CORS_ALLOW_ALL_ORIGINS = True
else:
    CORS_ALLOWED_ORIGINS: list[str] = []
CORS_ALLOW_CREDENTIALS = True

CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    },
}
