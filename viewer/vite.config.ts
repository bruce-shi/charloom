import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
let target = process.env.CHARLOOM_SERVER_URL;
if (!target) {
  try {
    target = JSON.parse(
      fs.readFileSync(
        path.join(
          process.env.CHARLOOM_DATA_DIR ?? path.join(os.homedir(), ".charloom"),
          "server.json",
        ),
        "utf8",
      ),
    ).url;
  } catch {}
}
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    proxy: {
      "/api/local/v1": {
        target: target ?? "http://127.0.0.1:5274",
        changeOrigin: true,
        ws: true,
      },
      "/files/": {
        target: target ?? "http://127.0.0.1:5274",
        changeOrigin: true,
      },
    },
  },
});
