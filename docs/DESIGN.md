# PrivateEdge UI and product alignment

This dashboard matches the **PrivateEdge** control-surface concept: **dual previews** (raw vs policy output), **live edge telemetry** (FPS, latency, NPU), **protection toggles** aligned with `configs/default.yaml`, and **audio / event** panels—without shipping raw media off-device.

## Reference documents (in-repo)

| File | Role |
|------|------|
| `docs/references/PrivateEdge.pdf` | Product / model tradeoffs (e.g. background face detection, identity protection use cases). |
| `docs/references/Participant Information Guide_Snapdragon AI Unplugged at UL.pdf` | Hackathon rules: **ONNX + QNN EP on Snapdragon X Elite**, edge-first submission, README/license requirements. |

## Hackathon criteria reflected in the app

From the **Participant Information Guide**:

- **Technical implementation:** NPU usage %, latency, FPS shown in the header; inference runs locally via the Python pipeline.
- **Use case:** Real-time moderation before content is streamed; toggles map to policy modules (faces, documents, NSFW, audio PII).
- **Deployment:** Run instructions in the project README; UI connects to local REST + WebSocket only.

## Visual system

- **Dark** control surface (`#0b0f14` base) suitable for demo on Copilot+ PC.
- **Accent** blue (`#3d9aed`) for active controls and metrics emphasis.
- **Segoe UI / system UI** stack for native Windows feel on Snapdragon laptops.

## Copy

- Title: **PrivateEdge**
- Subtitle calls out **ONNX Runtime + QNN EP** and **Snapdragon X Elite** to match submission language.

Adjust spacing in `frontend/src/App.css` if judges request larger touch targets on the loaner device.

The UI uses **same-origin** `/api` and `/ws` URLs (current hostname/IP), not hardcoded loopback—see `frontend/vite.config.ts` proxy and optional `PRIVATEEDGE_API_ORIGIN`.
