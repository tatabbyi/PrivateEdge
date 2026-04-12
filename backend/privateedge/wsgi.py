"""WSGI entry (optional; ASGI is primary for Channels)."""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "privateedge.settings")

application = get_wsgi_application()
