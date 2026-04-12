from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BACKEND = ROOT / "backend"

for p in (SRC, BACKEND):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
