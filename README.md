# PrivateEdge

PrivateEdge is an edge-first moderation and privacy layer for livestreamers and creators. It analyzes webcam, screen-share, and microphone input locally, then applies policy actions (blur/mute/silent mode) before content leaves the device.

## Why this exists

Live creators can accidentally leak personal information or harmful content in real time (documents on screen, bystanders in frame, spoken PII, toxic speech spikes). PrivateEdge adds a low-latency protective layer that runs on-device and is designed for Snapdragon X Elite + ONNX Runtime QNN EP.

## Hackathon alignment

- **Edge-first execution:** inference and policy loop run locally.
- **ONNX + QNN EP path:** `src/models/runtime.py` prefers `QNNExecutionProvider` when available.
- **Deployment focus:** web dashboard + backend can run on a single laptop.
- **Open source submission:** MIT licensed repository.

## Repository layout

| Path | Role |
|------|------|
| `backend/` | Django + Channels control plane (`/api`, `/ws`) |
| `frontend/` | React + Vite dashboard |
| `src/capture/` | Webcam and screen capture sources |
| `src/inference/` | Vision/audio signal extraction and scoring fusion |
| `src/policy/` | Policy types and decision engine |
| `src/render/` | Protection rendering primitives |
| `src/models/` | ONNX Runtime session setup (QNN/CUDA/CPU fallback) |
| `models/` | ONNX model assets (e.g. `nsfw.onnx`) |
| `tests/` | Unit tests for policy and score logic |
| `scripts/` | Utility scripts, including environment preflight |

## Setup

### Python backend

```bash
py -3 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Run locally

In terminal A:

```bash
.\.venv\Scripts\activate
cd backend
daphne -b 0.0.0.0 -p 8000 privateedge.asgi:application
```

In terminal B:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173` (or the Vite Network URL).

## Snapdragon day-of preflight

Before demoing on the loaner device, run:

```bash
.\.venv\Scripts\activate
python scripts/preflight_env.py --model models/nsfw.onnx
```

Expected:
- `QNNExecutionProvider` appears in `providers`
- `session_ok: true`
- `session_provider` is `QNNExecutionProvider` (or at least non-null)

If QNN is missing, install the Snapdragon-compatible ONNX Runtime build and re-run preflight.

## Test

```bash
.\.venv\Scripts\activate
pytest -q
```

## API overview

- `GET /api/health/` -> service health
- `GET /api/config/` -> runtime config
- `PATCH /api/config/` -> toggle modules and thresholds
- `GET /api/status/` -> latest telemetry
- `GET /api/logs/` -> recent events
- `WS /ws/stream/` -> telemetry + frame payload stream

## Submission checklist (competition)

- [x] Fill team names + emails below.
- [x] Include open-source license (`LICENSE`).
- [x] Provide setup/run instructions.
- [ ] Confirm app runs on Snapdragon X Elite with QNN EP.
- [ ] Add short demo video or screenshots (recommended).
- [x] Provide tests + test command.

### Team

- Cristian Bivol / crisbivol.business@gmail.com:
- David Conroy / davidconroy333@gmail.com:
- Rachel Keaveney / ytrachel0@gmail.com:
- Joan Byrne / tomentose38@gmail.com:

## Current limitations

- Demo blur is currently full-frame when policy triggers, not region-specific masking.
- No production installer yet (run via Python + Node scripts).
- Accuracy depends on available model assets (`models/nsfw.onnx`) and microphone quality.
- `npu_percent` telemetry is estimated, not direct hardware counter integration.
- No persisted storage; runtime state resets on restart.
