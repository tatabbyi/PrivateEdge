import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

/** Dev/preview proxy: set PRIVATEEDGE_API_ORIGIN in .env. Use 127.0.0.1 (Node often fails proxying to 0.0.0.0). */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const apiOrigin =
    env.PRIVATEEDGE_API_ORIGIN ||
    process.env.PRIVATEEDGE_API_ORIGIN ||
    "http://127.0.0.1:8000";
  const wsOrigin = apiOrigin.replace(/^http/, "ws");

  const proxy = {
    "/api": { target: apiOrigin, changeOrigin: true },
    "/ws": { target: wsOrigin, ws: true },
  };

  return {
    plugins: [react()],
    server: {
      port: 5173,
      host: true,
      proxy,
    },
    preview: {
      host: true,
      proxy,
    },
  };
});
