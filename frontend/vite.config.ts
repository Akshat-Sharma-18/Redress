import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Proxying rather than pointing the client at http://localhost:8000
    // directly: same-origin requests in development mean the deployed build,
    // where the API is served from the same origin, exercises the same code
    // path. A CORS-only dev setup hides same-origin bugs until deploy.
    proxy: {
      "/api": {
        target: process.env.REDRESS_API ?? "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
