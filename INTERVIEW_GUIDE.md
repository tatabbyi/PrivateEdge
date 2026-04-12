# PrivateEdge Interview Guide (In-Depth, Plain English)

## 1) What this project is

PrivateEdge is a **real-time safety and privacy filter** for livestream content.

It takes three input streams:
- webcam video,
- screen-share video,
- microphone audio.

It then:
- detects risk signals (private text, unsafe visuals, risky speech),
- decides what protection to apply (blur, mute, silent protection),
- and exposes everything to a control dashboard so an operator can tune behavior live.

The key philosophy is: **do this locally on-device first** (low latency + privacy), not by sending raw data to a cloud moderation service.

---

## 2) System architecture in one view

Pipeline flow:

1. Capture input (`src/capture`)
2. Infer risk scores (`src/inference`)
3. Merge/fuse scores (`src/inference/scoring.py`)
4. Decide action (`src/policy/engine.py`)
5. Apply protection (`src/render/blur.py`)
6. Push telemetry/frames to UI (`backend/api/consumers.py`)
7. Operator updates config via REST (`backend/api/views.py`)

Back-end and front-end responsibilities:
- **Backend (Django + Channels):** pipeline execution, state, API, WebSocket stream.
- **Frontend (React):** live dashboard for controls + monitoring.

---

## 3) Why each major folder exists

- `backend/`
  - Hosts HTTP and WebSocket endpoints.
  - Owns shared runtime state and the long-running pipeline loop.
  - Makes the app controllable from the dashboard.

- `frontend/`
  - Operator UI for toggles/sliders/devices/modes.
  - Receives live frame + telemetry stream over WebSocket.

- `src/capture/`
  - Hardware interfaces for webcam/screen.
  - Keeps device logic isolated from policy and UI.

- `src/inference/`
  - Converts raw media into structured risk numbers.
  - Keeps detection logic separate from business rules.

- `src/policy/`
  - Rule layer: "given scores, what action should we take?"
  - This separation makes behavior explainable and testable.

- `src/render/`
  - Implements how protection looks (currently blur-centric).
  - Cleanly decouples decisions from visual effect implementation.

- `src/models/`
  - Runtime/provider selection for ONNX sessions.
  - Handles acceleration preference and fallback.

- `tests/`
  - Unit tests for high-risk logic (policy, scoring, text signals).

- `scripts/`, `run_all.ps1`, `run_all.cmd`
  - Environment preflight and one-command local startup.

---

## 4) Backend deep dive (what runs where)

### Startup

- `backend/api/apps.py` starts the processing pipeline when Django app boots.
- This means backend and processing loop can run together for demo simplicity.

### Shared state

- `backend/services/state.py` exposes a singleton state object.
- State includes:
  - runtime config values from UI,
  - latest telemetry,
  - event log,
  - policy context/engine.

Why this pattern:
- Easy to reason about in a single-process demo.
- Fast to update/read from API and pipeline loop.

Tradeoff:
- Not distributed-safe by default (in-memory per process).

### Main loop

- `backend/services/pipeline.py` runs repeatedly:
  - capture frame(s),
  - run inference,
  - merge scores,
  - apply policy,
  - render protected output,
  - broadcast state over WebSocket.

This is the "heart" of the app.

### APIs

- `backend/api/views.py` exposes control and status:
  - config get/patch,
  - status,
  - logs,
  - devices,
  - health/runtime info.

### WebSocket

- `backend/api/consumers.py` pushes live payloads to frontend.
- `backend/api/routing.py` maps `/ws/stream/`.
- Used for low-latency updates without polling.

---

## 5) Frontend deep dive

- Main UI lives in `frontend/src/App.tsx`.
- It does two things in parallel:
  1. Pulls and patches config using REST.
  2. Maintains a persistent WebSocket for live telemetry/frame updates.

UI responsibilities:
- Show current scores/state and stream status.
- Let operator switch modules/modes quickly.
- Let operator tune sensitivity/blur/mute thresholds live.
- Let operator choose capture and virtual output options.

Why React + Vite:
- Fast dev iteration, simple local setup, great for demo UX updates.

---

## 6) Inference and scoring design choices

### Vision/text/audio signals

- `src/inference/vision.py`
  - Computes visual risk signals (faces/documents/nsfw path).

- `src/inference/text_signals.py`
  - Heuristic text risk extraction (PII/toxicity cues).

- `src/inference/audio_worker.py`
  - Handles audio stream processing path.

### Score fusion

- `src/inference/scoring.py`
  - Merges per-source outputs into one score object.
  - Keeps one decision surface for policy instead of scattered logic.

Why this layering matters in interviews:
- You can explain each stage independently.
- You can replace one detector without rewriting policy or UI.

---

## 7) Policy engine: exact value proposition

`src/policy/engine.py` is where moderation behavior becomes deterministic.

Inputs:
- normalized scores from inference layer,
- config thresholds/toggles from runtime state.

Outputs:
- explicit decision object (blur, mute, reason/mode behavior).

Why this is strong:
- Transparent and auditable.
- Testable with unit tests.
- Easier to justify decisions in a product review or safety review.

---

## 8) Model runtime strategy (QNN/CUDA/CPU)

`src/models/runtime.py` chooses ONNX providers in priority order:
1. QNN (Snapdragon/NPU path),
2. CUDA (GPU path),
3. CPU fallback.

Why:
- Performance when specialized hardware exists.
- Reliability when it does not.

Interview framing:
"We designed for graceful degradation rather than hard failure."

---

## 9) OBS and virtual outputs (important clarification)

Current state:
- There are virtual output concepts and naming defaults (for OBS-friendly routing).
- There is **not** direct OBS remote control (no `obs-websocket` scene/recording automation).

How to say this cleanly:
"We integrate at the device layer, not the OBS automation API layer."

---

## 10) Why these tradeoffs were made

### Chosen for speed and reliability

- In-memory state/channels:
  - faster to build and debug in one process.
- Threaded in-process pipeline:
  - fewer moving pieces for local demos.
- Full-frame blur default:
  - simpler and robust compared to brittle region masks under time pressure.

### Accepted limitations

- Not horizontally scalable as-is.
- Some telemetry is estimated rather than hardware-native counters.
- No full production persistence layer.

---

## 11) Testing strategy (what confidence you can claim)

Tests cover core logic where mistakes are expensive:
- `tests/test_policy_engine.py`
  - verifies decision behavior under threshold combinations.
- `tests/test_scoring.py`
  - verifies merge/fusion logic.
- `tests/test_text_signals.py`
  - verifies textual risk signal extraction.

This gives confidence that policy decisions are stable even as internals evolve.

What remains future work:
- broader integration tests (REST + WS + full loop),
- stress/load tests for long sessions.

---

## 12) Interview answers you can memorize

### "What problem does this solve?"

It reduces accidental privacy/safety leaks during livestreaming by applying local, real-time moderation before content is published.

### "Why edge-first?"

Privacy and latency. Local execution avoids sending raw sensitive media out for first-pass moderation.

### "How do you keep decisions explainable?"

We separate detection from policy and use explicit threshold-based decision logic in `PolicyEngine`.

### "Why this stack?"

Django/DRF + Channels gave us fast REST+WS delivery in one backend; React gave rapid dashboard iteration.

### "What would you improve next?"

Region-aware masking, distributed state/channel layer, stronger integration/load testing, and deeper virtual output tooling.

---

## 13) 2-minute interview script (clear English)

"PrivateEdge is a local-first moderation system for livestreaming. We capture webcam, screen-share, and microphone input, compute risk signals in real time, and run a policy engine that decides protection actions like blur and mute before content leaves the machine.

The backend is Django with Channels: REST endpoints manage configuration and status, while WebSockets stream live telemetry and protected preview frames to a React dashboard. The pipeline loop continuously performs capture, inference, score fusion, policy evaluation, and rendering.

We intentionally separated inference from policy. That gives us explainable behavior, easier testing, and safer tuning. For model execution we use ONNX Runtime with hardware-aware fallback: QNN first, then CUDA, then CPU, so the app remains deployable across different machines.

Our main tradeoff was prioritizing reliability and delivery speed for a local demo: in-memory state and a single-process pipeline are simple and fast, but not the final production architecture. Next steps are scaling hardening, richer masking quality, and stronger integration/load testing." 

---

## 14) 30-second version (if interviewer cuts you off)

"PrivateEdge is a real-time, on-device moderation layer for livestreaming. It captures webcam/screen/audio, computes risk scores, applies policy-based blur/mute decisions, and streams results to a React dashboard via Django Channels. We chose explicit policy logic for explainability and ONNX provider fallback for portability. The current design is optimized for low-latency local reliability, with production scaling as the next phase."

