import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

type Config = {
  face_masking: boolean;
  text_document_blocking: boolean;
  nsfw_detection: boolean;
  audio_pii_filtering: boolean;
  profanity_bleep_enabled: boolean;
  middle_finger_censoring: boolean;
  mode: string;
  detection_sensitivity: number;
  blur_strength: number;
  mute_sensitivity: number;
  protection_enabled: boolean;
  /** When false, webcam capture and inference for that source are off */
  webcam_enabled: boolean;
  /** When false, screen capture and inference for that source are off */
  screen_share_enabled: boolean;
  /** Null = auto/default, integer = explicit OpenCV camera index */
  webcam_index: number | null;
  /** Null = OS default, integer = explicit microphone input index */
  mic_device_index: number | null;
  virtual_webcam_enabled: boolean;
  virtual_screenshare_enabled: boolean;
  virtual_audio_enabled: boolean;
  virtual_webcam_device_name: string;
  virtual_screenshare_device_name: string;
  virtual_audio_output_device: string;
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
  p_obscene_gesture: number;
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

type DeviceOption = {
  id: number;
  label: string;
};

type DevicesResponse = {
  video_inputs?: DeviceOption[];
  audio_inputs?: DeviceOption[];
  audio_outputs?: DeviceOption[];
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
    webcam_index:
      typeof c.webcam_index === "number" ? Math.trunc(c.webcam_index) : null,
    mic_device_index:
      typeof c.mic_device_index === "number"
        ? Math.trunc(c.mic_device_index)
        : null,
    virtual_webcam_enabled:
      typeof c.virtual_webcam_enabled === "boolean"
        ? c.virtual_webcam_enabled
        : true,
    virtual_screenshare_enabled:
      typeof c.virtual_screenshare_enabled === "boolean"
        ? c.virtual_screenshare_enabled
        : false,
    virtual_audio_enabled:
      typeof c.virtual_audio_enabled === "boolean" ? c.virtual_audio_enabled : false,
    virtual_webcam_device_name:
      typeof c.virtual_webcam_device_name === "string"
        ? c.virtual_webcam_device_name
        : "OBS Virtual Camera",
    virtual_screenshare_device_name:
      typeof c.virtual_screenshare_device_name === "string"
        ? c.virtual_screenshare_device_name
        : "privateedge-screenshare",
    virtual_audio_output_device:
      typeof c.virtual_audio_output_device === "string"
        ? c.virtual_audio_output_device
        : "privateedge-audio",
    profanity_bleep_enabled:
      typeof c.profanity_bleep_enabled === "boolean"
        ? c.profanity_bleep_enabled
        : true,
    middle_finger_censoring:
      typeof c.middle_finger_censoring === "boolean"
        ? c.middle_finger_censoring
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
  const [videoInputs, setVideoInputs] = useState<DeviceOption[]>([]);
  const [audioInputs, setAudioInputs] = useState<DeviceOption[]>([]);
  const [audioOutputs, setAudioOutputs] = useState<DeviceOption[]>([]);
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

  const loadDevices = useCallback(async () => {
    try {
      const r = await fetch(`${API_BASE}/api/devices/`, {
        headers: { Accept: "application/json" },
      });
      if (!r.ok) return;
      const j = (await r.json()) as DevicesResponse;
      setVideoInputs(Array.isArray(j.video_inputs) ? j.video_inputs : []);
      setAudioInputs(Array.isArray(j.audio_inputs) ? j.audio_inputs : []);
      setAudioOutputs(Array.isArray(j.audio_outputs) ? j.audio_outputs : []);
    } catch {
      // Keep UI usable even if device probing endpoint fails.
    }
  }, []);

  useEffect(() => {
    void loadConfig();
  }, [loadConfig]);

  useEffect(() => {
    void loadDevices();
  }, [loadDevices]);

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
              title="Live NSFW signal score"
            >
              <label>NSFW</label>
              <strong>{liveScores.p_nsfw.toFixed(2)}</strong>
            </div>
          )}
          {liveScores && (
            <div className="metric metric-scores" title="Middle-finger gesture score">
              <label>Gesture</label>
              <strong>{liveScores.p_obscene_gesture.toFixed(2)}</strong>
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
          <div className="select-wrap">
            <label htmlFor="webcam-index">Webcam input</label>
            <select
              id="webcam-index"
              value={config.webcam_index ?? -1}
              onChange={(e) =>
                patchConfig({
                  webcam_index:
                    Number(e.target.value) < 0
                      ? null
                      : Number(e.target.value),
                })
              }
            >
              <option value={-1}>Auto / default</option>
              {videoInputs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>
          <div className="select-wrap">
            <label htmlFor="mic-device-index">Microphone input</label>
            <select
              id="mic-device-index"
              value={config.mic_device_index ?? -1}
              onChange={(e) =>
                patchConfig({
                  mic_device_index:
                    Number(e.target.value) < 0
                      ? null
                      : Number(e.target.value),
                })
              }
            >
              <option value={-1}>Default system microphone</option>
              {audioInputs.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.label}
                </option>
              ))}
            </select>
          </div>
          <h2>Virtual outputs</h2>
          <div className="toggle-row">
            <span>Virtual webcam</span>
            <button
              type="button"
              className={`switch ${config.virtual_webcam_enabled ? "on" : ""}`}
              aria-pressed={config.virtual_webcam_enabled}
              onClick={() => toggle("virtual_webcam_enabled")}
            />
          </div>
          <div className="toggle-row">
            <span>Virtual screenshare</span>
            <button
              type="button"
              className={`switch ${config.virtual_screenshare_enabled ? "on" : ""}`}
              aria-pressed={config.virtual_screenshare_enabled}
              onClick={() => toggle("virtual_screenshare_enabled")}
            />
          </div>
          <div className="toggle-row">
            <span>Virtual audio</span>
            <button
              type="button"
              className={`switch ${config.virtual_audio_enabled ? "on" : ""}`}
              aria-pressed={config.virtual_audio_enabled}
              onClick={() => toggle("virtual_audio_enabled")}
            />
          </div>
          <div className="select-wrap">
            <label htmlFor="virtual-webcam-name">Virtual webcam device name</label>
            <input
              id="virtual-webcam-name"
              value={config.virtual_webcam_device_name}
              onChange={(e) =>
                patchConfig({ virtual_webcam_device_name: e.target.value })
              }
            />
          </div>
          <div className="select-wrap">
            <label htmlFor="virtual-screen-name">Virtual screenshare device name</label>
            <input
              id="virtual-screen-name"
              value={config.virtual_screenshare_device_name}
              onChange={(e) =>
                patchConfig({ virtual_screenshare_device_name: e.target.value })
              }
            />
          </div>
          <div className="select-wrap">
            <label htmlFor="virtual-audio-name">Virtual audio output device</label>
            <input
              id="virtual-audio-name"
              list="audio-output-options"
              value={config.virtual_audio_output_device}
              onChange={(e) =>
                patchConfig({ virtual_audio_output_device: e.target.value })
              }
            />
            <datalist id="audio-output-options">
              {audioOutputs.map((d) => (
                <option key={d.id} value={d.label.replace(/\s+\(\d+ch\)$/, "")} />
              ))}
            </datalist>
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
          <div className="toggle-row">
            <span>Audio PII filtering</span>
            <button
              type="button"
              className={`switch ${config.audio_pii_filtering ? "on" : ""}`}
              aria-pressed={config.audio_pii_filtering}
              onClick={() => toggle("audio_pii_filtering")}
            />
          </div>
          <div className="toggle-row">
            <span>Profanity bleep (audio)</span>
            <button
              type="button"
              className={`switch ${config.profanity_bleep_enabled ? "on" : ""}`}
              aria-pressed={config.profanity_bleep_enabled}
              onClick={() => toggle("profanity_bleep_enabled")}
            />
          </div>
          <div className="toggle-row">
            <span>Middle-finger censoring</span>
            <button
              type="button"
              className={`switch ${config.middle_finger_censoring ? "on" : ""}`}
              aria-pressed={config.middle_finger_censoring}
              onClick={() => toggle("middle_finger_censoring")}
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
              <span>Detection sensitivity</span>
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
              <span>Blur strength</span>
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
