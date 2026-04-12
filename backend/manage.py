#!/usr/bin/env python
"""Django management CLI."""

import os
import sys
from pathlib import Path

# Ensure the `backend/` directory (this file's parent) is on sys.path so `api`,
# `privateedge`, and `services` resolve regardless of current working directory.
_BACKEND_ROOT = Path(__file__).resolve().parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "privateedge.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Is it installed and available on PYTHONPATH?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
