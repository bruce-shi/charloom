import { randomBytes, timingSafeEqual } from "node:crypto";
import { createReadStream } from "node:fs";
import { realpath, stat } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import type { AddressInfo } from "node:net";
import type { Plugin, PreviewServer, ViteDevServer } from "vite";

const types: Record<string, string> = {
  ".glb": "model/gltf-binary", ".json": "application/json", ".png": "image/png",
  ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".md": "text/plain",
};

/** Serve the existing catalog and its artifacts with a per-process loopback session. */
export function gallery(root: string): Plugin {
  function configure(server: ViteDevServer | PreviewServer) {
    const token = randomBytes(32).toString("hex");
    const http = server.httpServer;
    if (!http) throw Error("Gallery requires a local HTTP server");
    http.once("listening", () => {
      const { port } = http.address() as AddressInfo;
      console.log(`Charloom Gallery: http://127.0.0.1:${port}/assets#auth=${token}`);
    });
    server.middlewares.use(async (req, res, next) => {
      const uri = (req.url ?? "/").split("?")[0];
      if (!uri.startsWith("/files/") && !uri.startsWith("/api/")) return next();
      const end = (code: number) => { res.statusCode = code; res.end(); };
      const { port } = http.address() as AddressInfo;
      const hosts = [`127.0.0.1:${port}`, `localhost:${port}`];
      if (!hosts.includes(req.headers.host ?? "") ||
          (req.headers.origin && !hosts.some(host => req.headers.origin === `http://${host}`))) return end(403);
      const cookie = `charloom_session_${port}`;
      const supplied = req.headers.authorization?.replace(/^Bearer /, "") ??
        req.headers.cookie?.split(";").map(v => v.trim()).find(v => v.startsWith(cookie + "="))?.slice(cookie.length + 1);
      const bytes = Buffer.from(supplied ?? "");
      if (bytes.length !== token.length || !timingSafeEqual(bytes, Buffer.from(token))) return end(401);
      res.setHeader("Cache-Control", "no-store");
      if (uri === "/api/local/v1/session" && req.method === "POST" && req.headers["x-charloom-protocol"] === "1") {
        res.setHeader("Set-Cookie", `${cookie}=${token}; HttpOnly; SameSite=Strict; Path=/`);
        return end(204);
      }
      if (!uri.startsWith("/files/")) return end(404);
      if (req.method !== "GET" && req.method !== "HEAD") return end(405);
      try {
        const [, , folder, ...parts] = decodeURIComponent(uri).split("/");
        if (!["out", "fixtures"].includes(folder) || parts.some(p => !p || p === "." || p === "..")) return end(403);
        const base = await realpath(resolve(root, folder));
        const file = await realpath(resolve(base, ...parts));
        const mime = types[extname(file).toLowerCase()];
        if (!file.startsWith(base + sep) || !mime) return end(403);
        const info = await stat(file);
        if (!info.isFile()) return end(404);
        res.setHeader("Content-Type", mime);
        res.setHeader("Content-Length", info.size);
        if (req.method === "HEAD") return res.end();
        const stream = createReadStream(file);
        res.on("close", () => stream.destroy());
        stream.on("error", () => res.destroy()).pipe(res);
      } catch {
        end(404);
      }
    });
  }
  return { name: "charloom-gallery", configureServer: configure, configurePreviewServer: configure };
}
