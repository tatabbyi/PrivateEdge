import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

type Config = {
  face_masking: boolean;
  text_document_blocking: boolean;
  nsfw_detection: boolean;
  /** PyTorch + Hugging Face EfficientNet when no local ONNX NSFW model */
  hf_efficientnet_nsfw: boolean;
  audio_pii_filtering: boolean;
  mode: string;
  detection_sensitivity: number;
  detection_sensitivity_secondary: number;
  blur_strength: number;
  blur_strength_secondary: number;
  mute_sensitivity: number;
  protection_enabled: boolean;
  /** When false, webcam capture and inference for that source are off */
  webcam_enabled: boolean;
  /** When false, screen capture and inference for that source are off */
  screen_share_enabled: boolean;
};

type Telemetry = {
  fps: number;
  latency_ms: number;
  npu_percent: number;
};

/** Vision/audio signals from the pipeline (same WebSocket as frames). */
type LiveScores = {
  p_nsfw: number;
  p_doc: number;
  p_face_other: number;
  p_pii_audio: number;
  anger: number;
};

type WsPayload = {
  kind: string;
  raw_webcam_jpeg?: string;
  raw_screen_jpeg?: string;
  protected_webcam_jpeg?: string;
  protected_screen_jpeg?: string;
  /** False when showing test pattern (real desktop capture failed). */
  screen_capture_live?: boolean;
  telemetry?: Telemetry;
  scores?: LiveScores;
  events?: { message: string; kind: string }[];
  audio?: { id: string; label: string; tone: string }[];
};

/** Empty = same origin (Vite proxy in dev, or UI served with API). Override with VITE_API_BASE. */
const API_BASE =
  (import.meta.env.VITE_API_BASE as string | undefined)?.trim() ?? "";

const CONFIG_FETCH_MS = 12_000;

/** Ensure booleans from API are real booleans (defaults for older responses). */
function normalizeConfig(c: Config): Config {
  return {
    ...c,
    webcam_enabled:
      typeof c.webcam_enabled === "boolean" ? c.webcam_enabled : true,
    screen_share_enabled:
      typeof c.screen_share_enabled === "boolean"
        ? c.screen_share_enabled
        : true,
  };
}

/** WebSocket to current page host (Vite proxies `/ws` → backend). Optional VITE_WS_URL override. */
function wsUrl(): string {
  if (import.meta.env.VITE_WS_URL) return import.meta.env.VITE_WS_URL;
  if (typeof window === "undefined") return "";
  const { protocol, host } = window.location;
  const wsScheme = protocol === "https:" ? "wss:" : "ws:";
  return `${wsScheme}//${host}/ws/stream/`;
}

export function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [configError, setConfigError] = useState<string | null>(null);
  const [telemetry, setTelemetry] = useState<Telemetry>({
    fps: 0,
    latency_ms: 0,
    npu_percent: 0,
  });
  const [liveScores, setLiveScores] = useState<LiveScores | null>(null);
  /** Protected preview only (one tile per source). */
  const [webcamPreview, setWebcamPreview] = useState<string | null>(null);
  const [screenPreview, setScreenPreview] = useState<string | null>(null);
  const [screenCaptureLive, setScreenCaptureLive] = useState<boolean | null>(
    null
  );
  const [events, setEvents] = useState<{ message: string; kind: string }[]>([]);
  const [audioLines, setAudioLines] = useState<
    { id: string; label: string; tone: string }[]
  >([{ id: "ok", label: "Audio OK", tone: "ok" }]);
  const [wsState, setWsState] = useState<"connecting" | "open" | "closed">(
    "connecting"
  );
  const wsRef = useRef<WebSocket | null>(null);

  const loadConfig = useCallback(async () => {
    setConfigError(null);
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), CONFIG_FETCH_MS);
    try {
      const r = await fetch(`${API_BASE}/api/config/`, {
        signal: ctrl.signal,
        headers: { Accept: "application/json" },
      });
      clearTimeout(timer);
      const raw = await r.text();
      if (!r.ok) {
        throw new Error(
          `${r.status} ${r.statusText}${raw ? `: ${raw.slice(0, 120)}` : ""}`
        );
      }
      let j: { config?: Config };
      try {
        j = JSON.parse(raw) as { config?: Config };
      } catch {
        const hint = raw.trimStart().startsWith("<")
          ? " (got HTML, not JSON — is daphne running on :8000 and are you using npm run dev so /api is proxied?)"
          : "";
        throw new SyntaxError(`Invalid JSON from /api/config/${hint}`);
      }
      if (!j?.config) {
        throw new Error("missing config in response");
      }
      setConfig(normalizeConfig(j.config));
    } catch (e) {
      clearTimeout(timer);
      let msg = "Could not load configuration";
      if (e instanceof Error) {
        msg =
          e.name === "AbortError"
            ? `timed out after ${CONFIG_FETCH_MS / 1000}s — is the API on port 8000 running?`
            : e.message;
      }
      setConfigError(msg);
      console.error("loadConfig:", e);
    }
  }, []);

  const patchConfig = useCallback(async (partial: Partial<Config>) => {
    try {
      const r = await fetch(`${API_BASE}/api/config/`, {
        method: "PATCH",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(partial),
      });
      const raw = await r.text();
      if (!r.ok) {
        console.error("patchConfig failed:", r.status, raw.slice(0, 200));
        throw new Error(`${r.status} ${r.statusText}`);
      }
      const j = JSON.parse(raw) as { config?: Config };
      if (j?.config) setConfig(normalizeConfig(j.config));
    } catch (e) {
      console.error("patchConfig:", e);
      throw e;
    }
  }, []);

  useEffect(() => {
    void loadConfig();
  }, [loadConfig]);

  useEffect(() => {
    const url = wsUrl();
    const ws = new WebSocket(url);
    wsRef.current = ws;
    ws.onopen = () => setWsState("open");
    ws.onclose = () => setWsState("closed");
    ws.onerror = () => setWsState("closed");
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as WsPayload;
        if (data.telemetry) setTelemetry(data.telemetry);
        if (data.scores) setLiveScores(data.scores);
        if (data.protected_webcam_jpeg)
          setWebcamPreview(data.protected_webcam_jpeg);
        if (data.protected_screen_jpeg)
          setScreenPreview(data.protected_screen_jpeg);
        if (typeof data.screen_capture_live === "boolean")
          setScreenCaptureLive(data.screen_capture_live);
        if (data.events) setEvents(data.events);
        if (data.audio) setAudioLines(data.audio);
      } catch {
        /* ignore */
      }
    };
    return () => ws.close();
  }, []);

  const toggle = (key: keyof Config) => {
    if (!config) return;
    const cur = config[key];
    if (typeof cur === "boolean") {
      void patchConfig({ [key]: !cur } as Partial<Config>);
    }
  };

  const modeOptions = useMemo(
    () => [
      { value: "emotion_adaptive", label: "Emotion Adaptive Mode" },
      { value: "strict", label: "Strict" },
      { value: "normal", label: "Normal" },
      { value: "minimal", label: "Minimal" },
      { value: "silent_protection", label: "Silent Protection" },
    ],
    []
  );

  if (!config) {
    if (configError) {
      return (
        <div className="app">
          <div className="load-error">
            <p className="placeholder">
              Cannot reach the PrivateEdge API ({configError}).
            </p>
            <p className="placeholder">
              Start the API (bind all interfaces), then retry:
              <br />
              <code>
                cd backend &amp;&amp; daphne -b 0.0.0.0 -p 8000
                privateedge.asgi:application
              </code>
            </p>
            <p>
              <button
                type="button"
                className="btn btn-primary"
                onClick={() => void loadConfig()}
              >
                Retry
              </button>{" "}
              <a href="/api/health/">Check /api/health/</a>
            </p>
          </div>
        </div>
      );
    }
    return (
      <div className="app">
        <p className="placeholder">Loading configuration…</p>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <div className="brand">
          <div className="brand-icon" aria-hidden>
            🛡️
          </div>
          <div>
            <h1>PrivateEdge</h1>
            <span>On-device moderation · ONNX Runtime + QNN EP (Snapdragon X Elite)</span>
          </div>
        </div>
        <div className="metrics">
          <div className="metric">
            <label>FPS</label>
            <strong>{telemetry.fps.toFixed(0)}</strong>
          </div>
          <div className="metric">
            <label>Latency</label>
            <strong>{telemetry.latency_ms.toFixed(0)} ms</strong>
          </div>
          <div className="metric">
            <label>NPU</label>
            <strong>{telemetry.npu_percent.toFixed(0)}%</strong>
          </div>
          {liveScores && (
            <div
              className="metric metric-scores"
              title="Live model signals (NSFW uses HF EfficientNet when that toggle is on and no ONNX NSFW model is loaded)"
            >
              <label>NSFW</label>
              <strong>{liveScores.p_nsfw.toFixed(2)}</strong>
            </div>
          )}
          <div className="battery" title="Local power (telemetry)">
            🔋
          </div>
        </div>
      </header>

      <div className="main-grid">
        <section className="feeds">
          <article className="feed-card">
            <header>Webcam</header>
            <div className="img">
              {config.webcam_enabled ? (
                webcamPreview ? (
                  <img
                    alt="Webcam — protected preview"
                    src={`data:image/jpeg;base64,${webcamPreview}`}
                  />
                ) : (
                  <span className="placeholder">Waiting for frames…</span>
                )
              ) : (
                <span className="placeholder muted">Webcam off</span>
              )}
            </div>
          </article>
          <article className="feed-card">
            <header>Screen share</header>
            {config.screen_share_enabled &&
              screenCaptureLive === false && (
                <p className="feed-warning">
                  Not your real screen — capture failed (test pattern). On
                  Wayland install <code>grim</code>; on X11 ensure{" "}
                  <code>DISPLAY</code> is set. Check the API logs.
                </p>
              )}
            <div className="img">
              {config.screen_share_enabled ? (
                screenPreview ? (
                  <img
                    alt="Screen — protected preview"
                    src={`data:image/jpeg;base64,${screenPreview}`}
                  />
                ) : (
                  <span className="placeholder">Waiting for frames…</span>
                )
              ) : (
                <span className="placeholder muted">Screen share off</span>
              )}
            </div>
          </article>
        </section>

        <aside className="sidebar">
          <h2>Video sources</h2>
          <p className="sidebar-hint">
            Turn sources on or off. Policy uses the max of webcam and screen
            scores when both run.
          </p>
          <div className="toggle-row">
            <span>Webcam</span>
            <button
              type="button"
              className={`switch ${config.webcam_enabled ? "on" : ""}`}
              aria-pressed={config.webcam_enabled}
              onClick={() => toggle("webcam_enabled")}
            />
          </div>
          <div className="toggle-row">
            <span>Screen share</span>
            <button
              type="button"
              className={`switch ${config.screen_share_enabled ? "on" : ""}`}
              aria-pressed={config.screen_share_enabled}
              onClick={() => toggle("screen_share_enabled")}
            />
          </div>
          <h2>Protection settings</h2>
          <div className="toggle-row">
            <span>Protection (blur &amp; mute)</span>
            <button
              type="button"
              className={`switch ${config.protection_enabled ? "on" : ""}`}
              aria-pressed={config.protection_enabled}
              onClick={() => toggle("protection_enabled")}
            />
          </div>
          <div className="toggle-row">
            <span>Face masking</span>
            <button
              type="button"
              className={`switch ${config.face_masking ? "on" : ""}`}
              aria-pressed={config.face_masking}
              onClick={() => toggle("face_masking")}
            />
          </div>
          <div className="toggle-row">
            <span>Text &amp; document blocking</span>
            <button
              type="button"
              className={`switch ${config.text_document_blocking ? "on" : ""}`}
              aria-pressed={config.text_document_blocking}
              onClick={() => toggle("text_document_blocking")}
            />
          </div>
          <div className="toggle-row">
            <span>NSFW detection</span>
            <button
              type="button"
              className={`switch ${config.nsfw_detection ? "on" : ""}`}
              aria-pressed={config.nsfw_detection}
              onClick={() => toggle("nsfw_detection")}
            />
          </div>
          <div className="toggle-row toggle-row--sub">
            <span title="Uses Hugging Face EfficientNet (PyTorch) when no models/nsfw.onnx is present">
              HF EfficientNet NSFW
            </span>
            <button
              type="button"
              className={`switch ${config.hf_efficientnet_nsfw ? "on" : ""}`}
              aria-pressed={config.hf_efficientnet_nsfw}
              onClick={() => toggle("hf_efficientnet_nsfw")}
            />
          </div>
          <div className="toggle-row">
            <span>Audio PII filtering</span>
            <button
              type="button"
              className={`switch ${config.audio_pii_filtering ? "on" : ""}`}
              aria-pressed={config.audio_pii_filtering}
              onClick={() => toggle("audio_pii_filtering")}
            />
          </div>
          <div className="select-wrap">
            <label htmlFor="mode">Mode</label>
            <select
              id="mode"
              value={config.mode}
              onChange={(e) => patchConfig({ mode: e.target.value })}
            >
              {modeOptions.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>
        </aside>
      </div>

      <div className="bottom">
        <div className="panel">
          <h3>Audio status</h3>
          <div className="audio-lines">
            {audioLines.map((a) => (
              <div
                key={a.id + a.label}
                className={`audio-line ${a.tone}`}
              >
                {a.tone === "ok" ? "✓" : a.tone === "warn" ? "⚠" : "⛔"}{" "}
                {a.label}
              </div>
            ))}
          </div>
          <h3 style={{ marginTop: "0.75rem" }}>Recent events</h3>
          <ul className="events">
            {events.length === 0 && (
              <li>No policy events yet — detections appear here.</li>
            )}
            {events.map((e, i) => (
              <li key={i}>{e.message}</li>
            ))}
          </ul>
        </div>

        <div className="panel sliders">
          <div className="slider-row">
            <label>
              <span>Detection sensitivity (A)</span>
              <span>{config.detection_sensitivity.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={config.detection_sensitivity}
              onChange={(e) =>
                patchConfig({
                  detection_sensitivity: parseFloat(e.target.value),
                })
              }
            />
          </div>
          <div className="slider-row">
            <label>
              <span>Detection sensitivity (B)</span>
              <span>{config.detection_sensitivity_secondary.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={config.detection_sensitivity_secondary}
              onChange={(e) =>
                patchConfig({
                  detection_sensitivity_secondary: parseFloat(
                    e.target.value
                  ),
                })
              }
            />
          </div>
          <div className="slider-row">
            <label>
              <span>Blur strength (A)</span>
              <span>{config.blur_strength.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={config.blur_strength}
              onChange={(e) =>
                patchConfig({ blur_strength: parseFloat(e.target.value) })
              }
            />
          </div>
          <div className="slider-row">
            <label>
              <span>Blur strength (B)</span>
              <span>{config.blur_strength_secondary.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={config.blur_strength_secondary}
              onChange={(e) =>
                patchConfig({
                  blur_strength_secondary: parseFloat(e.target.value),
                })
              }
            />
          </div>
          <div className="slider-row">
            <label>
              <span>Mute sensitivity</span>
              <span>{config.mute_sensitivity.toFixed(2)}</span>
            </label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={config.mute_sensitivity}
              onChange={(e) =>
                patchConfig({ mute_sensitivity: parseFloat(e.target.value) })
              }
            />
          </div>
        </div>

        <div className="actions">
          <div className={`conn ${wsState === "open" ? "ok" : "bad"}`}>
            WebSocket: {wsState}
          </div>
        </div>
      </div>
    </div>
  );
}
