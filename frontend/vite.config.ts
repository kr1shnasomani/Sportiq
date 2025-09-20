import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), "");
  const devPort = Number(env.VITE_DEV_PORT || "8080");
  const backendUrl = env.VITE_BACKEND_URL || "http://localhost:8000";

  return {
    server: {
      host: "::",
      port: devPort,
      proxy: {
        "/api": {
          target: backendUrl,
          changeOrigin: true,
        },
        "/health": {
          target: backendUrl,
          changeOrigin: true,
        },
      },
    },
    plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
  };
});
