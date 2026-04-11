# PrivateEdge

PrivateEdge is an on-device, real-time safety layer for livestreamers and content creators. It automatically detects and blocks sensitive, private, or harmful content before it ever reaches the stream.

## Problem

Describe the user-facing problem this project solves (privacy leaks, harmful content, compliance, etc.).

## Architecture

High-level layout:

| Path | Role |
|------|------|
| `src/capture/` | Video, screen, and audio capture |
| `src/models/` | Wrappers for ONNX + QNN Execution Provider |
| `src/policy/` | Rules, thresholds, emotion logic |
| `src/render/` | Masking, blurring, audio muting |
| `src/ui/` | Control panel |
| `src/configs/` | Thresholds and modes (source-side) |
| `models/` | ONNX weights and model configs |
| `configs/` | Thresholds and modes (deployment / root) |
| `scripts/` | Benchmarking and profiling |
| `tests/` | Unit tests for policy and basic I/O |

## Setup

**Python (Arch / PEP 668):** use a virtualenv — do not `pip install` into system Python.

```bash
cd PrivateEdge
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Backend sanity check** (venv is at the **repo root** `.venv/`, not under `backend/`):

From repo root: `source .venv/bin/activate` then `python backend/manage.py check`.

Or after `cd backend`: `source ../.venv/bin/activate` then `python manage.py check` (not `python backend/manage.py` — that path is only valid from the repo root).

If you see `ModuleNotFoundError: No module named 'api'...'`, ensure these files exist: `backend/api/__init__.py`, `backend/api/urls.py`, `backend/api/routing.py`, `backend/api/views.py`, `backend/api/consumers.py`, and that you are using the venv where Django is installed.

**Frontend:** project root is `frontend/` (`package.json`, `index.html`, `src/`). If you only see `node_modules/`, re-copy or pull the repo files, then:

```bash
cd frontend
npm install
npm run dev
```

Place ONNX assets under `models/` per your deployment notes.

## Usage

**UI & hackathon alignment:** See [docs/DESIGN.md](docs/DESIGN.md) and reference PDFs under [docs/references/](docs/references/) (Participant Information Guide + PrivateEdge deck).

**Run:**

1. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
2. Backend (listens on all interfaces): `cd backend && daphne -b 0.0.0.0 -p 8000 privateedge.asgi:application`
3. Frontend: `cd frontend && npm install && npm run dev` → open the **Network** URL Vite prints (or this machine’s hostname/IP on port 5173).

Optional: create `frontend/.env` with `PRIVATEEDGE_API_ORIGIN=http://<host>:8000` if the proxy must reach the API elsewhere (default `http://127.0.0.1:8000`).

Requirements: **ONNX Runtime with QNN Execution Provider on Snapdragon X Elite** for the judged stack; CPU-only dev is possible off-target.

## Limitations

Document known constraints (platform support, latency, model size, false positives/negatives, etc.).
