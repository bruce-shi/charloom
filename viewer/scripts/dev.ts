// One local authority in development too. Vite only serves hot-reloaded UI.
import path from "node:path";
import os from "node:os";
const root = path.resolve(import.meta.dirname, "../.."),
  data = process.env.CHARLOOM_DATA_DIR ?? path.join(os.homedir(), ".charloom");
const binary =
  process.env.CHARLOOM_CLI ?? path.join(root, "target/iterate/charloom");
await Bun.$`cargo build -p charloom-cli --profile iterate`.cwd(root);
const service = Bun.spawn(
  [
    binary,
    "serve",
    "--data-dir",
    data,
    "--port",
    "0",
    "--dev-origin",
    "http://127.0.0.1:5194",
  ],
  { cwd: root, stdout: "inherit", stderr: "inherit" },
);
try {
  let info: any;
  for (let i = 0; i < 100; i++) {
    try {
      const p = await Bun.file(path.join(data, "server.json")).json();
      if (p.pid === service.pid) {
        info = p;
        break;
      }
    } catch {}
    await Bun.sleep(50);
  }
  if (!info)
    throw Error(
      "本地服务无法启动。若该数据目录已有服务，请使用 charloom ide，或停止该服务后启动开发模式。",
    );
  console.log(
    `\nCharloom IDE: http://127.0.0.1:5194/#auth=${info.auth_token}\n`,
  );
  const vite = Bun.spawn(
    ["bunx", "vite", "--host", "127.0.0.1", "--port", "5194", "--strictPort"],
    {
      cwd: path.join(root, "viewer"),
      env: { ...process.env, CHARLOOM_SERVER_URL: info.url },
      stdout: "inherit",
      stderr: "inherit",
    },
  );
  const stop = () => {
    vite.kill();
    service.kill();
  };
  process.once("SIGINT", stop);
  process.once("SIGTERM", stop);
  await vite.exited;
} finally {
  service.kill();
  await service.exited;
}
