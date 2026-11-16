import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { gallery } from "./server/gallery";

export default defineConfig({
  plugins: [react(), gallery(path.resolve(import.meta.dirname, ".."))],
  server: { host: "127.0.0.1", port: 5194, fs: { allow: [import.meta.dirname] } },
  preview: { host: "127.0.0.1", port: 5274 },
});
