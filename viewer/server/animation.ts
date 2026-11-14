import type { Plugin } from "vite";
import { retargetSettings } from "./animation-profile";
import type { IncomingMessage, ServerResponse } from "node:http";
import { spawn, type ChildProcess } from "node:child_process";
import fs from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { createHash, randomUUID } from "node:crypto";

export type MotionAsset = {
  id: string;
  title: string;
  source: string;
  file: string;
  clips: { name: string; duration: number }[];
};
export type MotionJob = {
  id: string;
  status: "running" | "complete" | "failed" | "cancelled";
  stage: string;
  title: string;
  clip: string;
  duration: number;
  sourceId?: string;
  baseJobId?: string;
  character?: string;
  report?: any;
  error?: string;
};
export type MotionContact = {
  joint: string;
  position: [number, number, number];
  start: number;
  end: number;
  chain_root?: string;
};
const uuid = /^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$/;
export function validContacts(
  value: unknown,
  duration: number,
): MotionContact[] {
  if (!Array.isArray(value) || !value.length || value.length > 16)
    throw new Error("请选择 1–16 个 IK 控制点");
  const seen = new Set<string>();
  return value.map((c) => {
    if (
      !c ||
      typeof c.joint !== "string" ||
      !c.joint ||
      c.joint.length > 200 ||
      seen.has(c.joint) ||
      !Array.isArray(c.position) ||
      c.position.length !== 3 ||
      !c.position.every(
        (n: unknown) =>
          typeof n === "number" && Number.isFinite(n) && Math.abs(n) < 10000,
      ) ||
      !Number.isFinite(c.start) ||
      !Number.isFinite(c.end) ||
      c.start < 0 ||
      c.end <= c.start ||
      c.end > duration + 1e-6 ||
      (c.chain_root !== undefined &&
        (typeof c.chain_root !== "string" || c.chain_root.length > 200))
    )
      throw new Error("IK 控制点或作用区间无效");
    seen.add(c.joint);
    return {
      joint: c.joint,
      position: c.position,
      start: c.start,
      end: c.end,
      ...(c.chain_root ? { chain_root: c.chain_root } : {}),
    };
  });
}
export function animationJobs(root: string, storage = path.join(root, "out/workspace")): Plugin {
  const directory = path.join(storage, "animation"),
    binary = process.env.CHARLOOM_CLI ?? path.join(root, "target/release/charloom");
  const jobs = new Map<string, MotionJob>(),
    processes = new Map<string, ChildProcess>();
  let library: Promise<MotionAsset[]> | undefined;
  const reply = (res: ServerResponse, status: number, value: unknown) => {
    res.writeHead(status, {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
    });
    res.end(JSON.stringify(value));
  };
  const run = (id: string, args: string[]) =>
    new Promise<string>((resolve, reject) => {
      const child = spawn(binary, args, {
        cwd: root,
        stdio: ["ignore", "pipe", "pipe"],
      });
      processes.set(id, child);
      let output = "",
        error = "",
        timedOut = false;
      child.stdout!.on("data", (c) => {
        output = (output + c).slice(-2_000_000);
      });
      child.stderr!.on("data", (c) => {
        error = (error + c).slice(-20000);
      });
      const timer = setTimeout(() => {
        timedOut = true;
        child.kill("SIGTERM");
      }, 600000);
      const cleanup = () => {
        clearTimeout(timer);
        if (processes.get(id) === child) processes.delete(id);
      };
      child.once("error", (e) => {
        cleanup();
        reject(e);
      });
      child.once("close", (code) => {
        cleanup();
        if (code === 0) return resolve(output);
        try {
          const e = JSON.parse(output).error;
          error = typeof e === "string" ? e : (e?.message ?? error);
        } catch {
          /* Native plain text. */
        }
        reject(
          new Error(
            timedOut
              ? "动作计算超时"
              : (error || output || "动作计算失败").slice(0, 1600),
          ),
        );
      });
    });
  const inspect = async (file: string) =>
    JSON.parse(await run(randomUUID(), ["inspect", file, "--json"]));
  const clips = (info: any) =>
    (info.clips ?? [])
      .filter(
        (c: any) => c.tracks > 0 && c.time_range_s?.[1] > c.time_range_s?.[0],
      )
      .map((c: any) => ({
        name: c.name as string,
        duration: c.time_range_s[1] - c.time_range_s[0],
      }));
  const getLibrary = () =>
    (library ??= (async () => {
      const manifest = JSON.parse(
        await fs.readFile(
          path.join(root, "fixtures/mixamo/manifest.json"),
          "utf8",
        ),
      );
      const titles: Record<string, string> = {
        Running: "跑步",
        Walking: "走路",
        Waving: "挥手",
        "Two Hand Club Combo": "双手棍击",
      };
      return Promise.all(
        manifest.assets.map(async (asset: any) => {
          if (
            path.basename(asset.file) !== asset.file ||
            !asset.file.endsWith(".fbx")
          )
            throw new Error("动作清单路径无效");
          const file = path.join(root, "fixtures/mixamo", asset.file),
            data = await fs.readFile(file);
          if (createHash("sha256").update(data).digest("hex") !== asset.sha256)
            throw new Error("本地 Mixamo 动作与来源记录不一致");
          return {
            id: path.parse(asset.file).name,
            title: titles[asset.animation] ?? asset.animation,
            source: "Mixamo · " + asset.animation,
            file,
            clips: clips(await inspect(file)),
          };
        }),
      );
    })().catch((e) => {
      library = undefined;
      throw e;
    }));
  const listLibrary = async () => {
    const assets = [...await getLibrary()];
    for (const base of [path.join(directory, "sources"), path.join(root, "out/workspace/animation/sources")]) {
      for (const id of await fs.readdir(base).catch(() => [] as string[])) {
        if (!uuid.test(id)) continue;
        try { const asset = JSON.parse(await fs.readFile(path.join(base, id, "asset.json"), "utf8")) as MotionAsset; if (!assets.some(a => a.id === asset.id)) assets.push(asset); } catch {}
      }
    }
    return assets;
  };
  const source = async (id: string): Promise<MotionAsset> => {
    if (id.startsWith("upload-") && uuid.test(id.slice(7)))
      return JSON.parse(await fs.readFile(path.join(directory, "sources", id.slice(7), "asset.json"), "utf8").catch(() => fs.readFile(path.join(root, "out/workspace/animation/sources", id.slice(7), "asset.json"), "utf8")));
    const asset = (await getLibrary()).find((a) => a.id === id);
    if (!asset) throw new Error("没有找到这个动作");
    return asset;
  };
  const folder = (id: string) => {
    const current = path.join(directory, "jobs", id), legacy = path.join(root, "out/workspace/animation/jobs", id);
    return !existsSync(current) && existsSync(legacy) ? legacy : current;
  };
  const save = async (job: MotionJob) => {
    jobs.set(job.id, job);
    await fs.writeFile(
      path.join(folder(job.id), "job.json"),
      JSON.stringify(job),
    );
  };
  const read = async (id: string): Promise<MotionJob> => {
    if (!uuid.test(id)) throw new Error("动作任务无效");
    const job =
      jobs.get(id) ??
      JSON.parse(await fs.readFile(path.join(folder(id), "job.json"), "utf8"));
    if (job.status === "running" && !jobs.has(id))
      return { ...job, status: "failed", error: "服务已重启，请重新适配动作" };
    return job;
  };
  const compute = async (job: MotionJob, args: string[]) => {
    try {
      await run(job.id, args);
      if (jobs.get(job.id)?.status === "cancelled") return;
      const report = JSON.parse(
        await fs.readFile(
          path.join(folder(job.id), "result/report.json"),
          "utf8",
        ),
      );
      await save({
        ...job,
        status: "complete",
        stage: "可以播放检查",
        report,
        character: `${process.env.CHARLOOM_JOB_URL ?? "/files/out/workspace"}/animation/jobs/${job.id}/result/character.glb`,
      });
    } catch (e) {
      if (jobs.get(job.id)?.status !== "cancelled")
        await save({
          ...job,
          status: "failed",
          stage: "需要调整",
          error: e instanceof Error ? e.message : String(e),
        });
    }
  };
  const body = async (req: IncomingMessage, max: number) => {
    const parts: Buffer[] = [];
    let size = 0;
    for await (const part of req) {
      const b = Buffer.from(part);
      size += b.length;
      if (size > max) throw new Error("文件超过单次 256 MB 限制");
      parts.push(b);
    }
    return Buffer.concat(parts);
  };
  const form = async (req: IncomingMessage) => {
    if (!req.headers["content-type"]?.startsWith("multipart/form-data"))
      throw new Error("请选择模型或动作文件");
    return new Request("http://localhost", {
      method: "POST",
      headers: { "Content-Type": req.headers["content-type"] },
      body: await body(req, 256 * 1024 * 1024),
    }).formData();
  };
  const serve = async (
    req: IncomingMessage,
    res: ServerResponse,
    next: () => void,
  ) => {
    const url = new URL(req.url ?? "/", "http://localhost"),
      base = "/api/animations";
    if (!url.pathname.startsWith(base + "/")) return next();
    try {
      if (req.method === "GET" && url.pathname === base + "/library") {
        const assets = await listLibrary();
        return reply(res, 200, { assets: assets.map(({ file, ...a }) => a) });
      }
      const match = url.pathname.match(
        /^\/api\/animations\/jobs\/([a-f0-9-]{36})$/,
      );
      if (req.method === "GET" && match)
        return reply(res, 200, await read(match[1]));
      if (req.headers.origin !== `http://${req.headers.host}`)
        return reply(res, 403, { error: "请从当前工作区发起动作处理" });
      if (req.method === "DELETE" && match) {
        const job = await read(match[1]);
        if (job.status === "running") {
          await save({ ...job, status: "cancelled", stage: "已取消" });
          processes.get(job.id)?.kill("SIGTERM");
        }
        return reply(res, 200, { status: "cancelled" });
      }
      if (req.method !== "POST")
        return reply(res, 405, { error: "不支持的动作请求" });
      if ([...jobs.values()].filter((j) => j.status === "running").length >= 2)
        return reply(res, 409, { error: "已有动作正在计算，请稍后重试" });
      await fs.access(binary).catch(() => {
        throw new Error("请先运行 bun run build:native");
      });
      if (url.pathname === base + "/sources") {
        const data = await form(req),
          file = data.get("motion");
        if (
          !file ||
          typeof file === "string" ||
          !/\.(fbx|glb)$/i.test(file.name)
        )
          throw new Error("请选择 Mixamo FBX 或含动作的 GLB");
        const id = randomUUID(),
          dest = path.join(directory, "sources", id);
        await fs.mkdir(dest, { recursive: true });
        const local = path.join(
          dest,
          "motion" + path.extname(file.name).toLowerCase(),
        );
        await fs.writeFile(local, Buffer.from(await file.arrayBuffer()));
        const asset: MotionAsset = {
          id: "upload-" + id,
          title: path.parse(file.name).name,
          source: "本地导入",
          file: local,
          clips: clips(await inspect(local)),
        };
        if (!asset.clips.length) throw new Error("文件没有可播放的骨骼动作");
        await fs.writeFile(
          path.join(dest, "asset.json"),
          JSON.stringify(asset),
        );
        const { file: _, ...publicAsset } = asset;
        return reply(res, 201, publicAsset);
      }
      if (url.pathname !== base + "/jobs")
        return reply(res, 404, { error: "没有这个动作接口" });
      const id = randomUUID();
      await fs.mkdir(folder(id), { recursive: true });
      if (req.headers["content-type"]?.startsWith("application/json")) {
        const input = JSON.parse((await body(req, 65536)).toString());
        const prior = await read(input.baseJobId);
        if (prior.status !== "complete") throw new Error("请先完成动作适配");
        const contacts = input.surface_floor && Array.isArray(input.contacts) && !input.contacts.length
          ? [] : validContacts(input.contacts, prior.duration);
        const name = prior.clip + "_IK_" + id.slice(0, 6),
          target = path.join(folder(prior.id), "result/character.glb");
        const recipe = {
          schema_version: "0.1",
          asset: target,
          name,
          duration: prior.duration,
          fps: input.surface_floor ? 120 : 30,
          layers: [{ clip: prior.clip }],
          contacts,
          ...(input.surface_floor ? { surface_floor: input.surface_floor } : {}),
          solver: {
            iterations: 100,
            tolerance_m: 0.002,
            posture_weight: 0.00001,
          },
        };
        const recipePath = path.join(folder(id), "recipe.json");
        await fs.writeFile(recipePath, JSON.stringify(recipe));
        const job: MotionJob = {
          id,
          status: "running",
          stage: "正在烘焙 IK 修正…",
          title: prior.title,
          clip: name,
          duration: prior.duration,
          baseJobId: prior.id,
          sourceId: prior.sourceId,
        };
        await save(job);
        void compute(job, [
          "build",
          recipePath,
          "--out",
          path.join(folder(id), "result"),
          "--json",
        ]);
        return reply(res, 202, job);
      }
      const data = await form(req),
        model = data.get("model"),
        asset = await source(String(data.get("source"))),
        clip = asset.clips.find((c) => c.name === data.get("clip"));
      if (!model || typeof model === "string" || !clip)
        throw new Error("缺少角色或动作片段");
      const bytes = Buffer.from(await model.arrayBuffer());
      if (
        bytes.length < 12 ||
        bytes.toString("ascii", 0, 4) !== "glTF" ||
        bytes.readUInt32LE(4) !== 2 ||
        bytes.readUInt32LE(8) !== bytes.length
      )
        throw new Error("角色必须是有效 GLB");
      const target = path.join(folder(id), "target.glb");
      await fs.writeFile(target, bytes);
      const [targetInfo, mappingText] = await Promise.all([
        inspect(target),
        run(randomUUID(), ["map", asset.file, target, "--json"]),
      ]);
      const settings = retargetSettings(
        targetInfo.joints,
        JSON.parse(mappingText),
      );
      const name = asset.title + "_" + id.slice(0, 6),
        configPath = path.join(folder(id), "config.json");
      await fs.writeFile(
        configPath,
        JSON.stringify({
          fps: 30,
          name,
          root_motion: data.get("rootMotion") === "true",
          ...settings,
          align_reference_chains: true,
        }),
      );
      const job: MotionJob = {
        id,
        status: "running",
        stage: "正在匹配骨骼并适配动作…",
        title: asset.title,
        clip: name,
        duration: clip.duration,
        sourceId: asset.id,
      };
      await save(job);
      void compute(job, [
        "retarget",
        asset.file,
        target,
        "--clip",
        clip.name,
        "--config",
        configPath,
        "--out",
        path.join(folder(id), "result"),
        "--json",
      ]);
      return reply(res, 202, job);
    } catch (e) {
      reply(res, 400, { error: e instanceof Error ? e.message : String(e) });
    }
  };
  return {
    name: "local-animation-jobs",
    configureServer(server) {
      server.middlewares.use(serve);
      server.httpServer?.once("close", () =>
        processes.forEach((p) => p.kill()),
      );
    },
    configurePreviewServer(server) {
      server.middlewares.use(serve);
      server.httpServer?.once("close", () =>
        processes.forEach((p) => p.kill()),
      );
    },
  };
}
