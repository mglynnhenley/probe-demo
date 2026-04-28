import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const MODAL_URL = "https://mglynnhenley--probe-api.modal.run";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: MODAL_URL,
        changeOrigin: true,
        secure: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
