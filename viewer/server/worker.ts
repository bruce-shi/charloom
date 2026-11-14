// Private native-job adapter owned by `charloom serve`. The public transport,
// authentication, project persistence and event stream live in charloom-local.
import {
  createServer,
  type IncomingMessage,
  type ServerResponse,
} from "node:http";
import { animationJobs } from "./animation";
import fs from "node:fs";
const root = process.env.CHARLOOM_SOURCE_ROOT!,
  storage = process.env.CHARLOOM_JOB_DIR!;
if (!root || !storage || !process.env.CHARLOOM_WORKER_TOKEN)
  throw Error("Use charloom serve to start this worker");
fs.mkdirSync(storage, { recursive: true });
type Middleware = (
  req: IncomingMessage,
  res: ServerResponse,
  next: () => void,
) => void;
const handlers: Middleware[] = [];
const server = createServer((req, res) => {
  if (
    req.headers.authorization !==
    "Bearer " + process.env.CHARLOOM_WORKER_TOKEN
  ) {
    res.writeHead(403).end();
    return;
  }
  if (req.url === "/health") {
    res
      .writeHead(200, { "Content-Type": "application/json" })
      .end('{"ready":true}');
    return;
  }
  let index = 0;
  const next = () => {
    const handler = handlers[index++];
    if (handler) handler(req, res, next);
    else res.writeHead(404).end();
  };
  next();
});
const adapter = {
  middlewares: { use: (fn: Middleware) => handlers.push(fn) },
  httpServer: server,
};
for (const plugin of [animationJobs(root, storage)]) {
  const hook = plugin.configureServer;
  if (typeof hook === "function") hook.call({} as any, adapter as any);
}
server.listen(Number(process.env.CHARLOOM_WORKER_PORT), "127.0.0.1");
function stop() {
  server.close();
  setTimeout(() => process.exit(0), 200).unref();
}
process.on("SIGTERM", stop);
process.on("SIGINT", stop);

// A hard-killed authority must not leave native workers running indefinitely.
const authority = process.ppid;
setInterval(() => {
  if (process.ppid !== authority) stop();
}, 1000).unref();
