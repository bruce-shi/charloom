import { Realtime, SocketUnavailable, SocketReplyError } from "./realtime";
import type { WireSocketServer } from "./generated";
import type { ProjectManifest } from "./project";
import {
  applyChanges,
  changes,
  type Asset,
  type Change,
  type LocalRequest,
  type NativeTask,
  type Project,
  type Receipt,
  type Summary,
} from "./protocol";
export class LocalError extends Error {
  constructor(
    public status: number,
    public detail: any,
  ) {
    super(detail?.error?.message ?? `本地服务返回 ${status}`);
  }
}
type Batch = {
  prepare?: () => Promise<Record<string, Asset>>;
  changes?: Change[];
  assets?: Record<string, Asset>;
  request_id: string;
};
type Channel = {
  staged: number;
  local: ProjectManifest;
  pending: Batch[];
  busy: boolean;
  blocked: boolean;
  conflict: boolean;
  waiters: { resolve: () => void; reject: (e: Error) => void }[];
};
export class LocalClient {
  readonly actor = "web:" + crypto.randomUUID();
  readonly records = new Map<string, Project>();
  private ready?: Promise<any>;
  private channels = new Map<string, Channel>();
  private listeners = new Set<(p: Project | undefined, r?: Receipt) => void>();
  private libraryListeners = new Set<(library: string) => void>();
  subscribeLibrary(fn: (library: string) => void) {
    this.libraryListeners.add(fn);
    return () => {
      this.libraryListeners.delete(fn);
    };
  }
  private readonly realtime: Realtime;
  private notices: WireSocketServer[] = [];
  private draining = false;
  transport: "websocket" | "offline" = "offline";
  private token?: string;
  connected = false;
  error = "";
  constructor(
    readonly base = "/api/local/v1",
    token?: string,
  ) {
    this.token = token;
    this.realtime = new Realtime({
      base: () => this.base,
      token: () => this.token,
      prepare: () => this.connect(),
      event: (frame) => {
        if (this.notices.length >= 128) this.notices = [{ type: "resync" }];
        else this.notices.push(frame);
        void this.drainNotices();
      },
      state: (connected, error) => {
        this.connected = connected;
        this.transport = connected ? "websocket" : "offline";
        this.error = connected ? "" : (error ?? "");
        this.emit();
      },
    });
  }
  subscribe(fn: (p: Project | undefined, r?: Receipt) => void) {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }
  private emit(p?: Project, r?: Receipt) {
    this.listeners.forEach((fn) => fn(p, r));
  }
  async connect() {
    return (this.ready ??= (async () => {
      if (typeof window !== "undefined") {
        const auth = new URLSearchParams(window.location.hash.slice(1)).get(
          "auth",
        );
        if (auth) {
          const r = await fetch(this.base + "/session", {
            method: "POST",
            credentials: "same-origin",
            headers: {
              "x-charloom-protocol": "1",
              authorization: `Bearer ${auth}`,
            },
          });
          if (!r.ok) throw new LocalError(r.status, await r.json());
          window.history.replaceState(
            window.history.state,
            "",
            window.location.pathname + window.location.search,
          );
        }
      }
      const r = await fetch(this.base + "/health", {
        credentials: "same-origin",
        headers: this.headers(),
      });
      if (!r.ok) throw new LocalError(r.status, await r.json());
      const info = await r.json();
      if (info.protocol !== 1) throw Error("本地服务版本不兼容");
      this.connected = true;
      this.error = "";
      return info;
    })().catch((e) => {
      this.ready = undefined;
      this.connected = false;
      this.error = e instanceof Error ? e.message : String(e);
      this.emit();
      throw e;
    }));
  }
  private headers() {
    return {
      "x-charloom-protocol": "1",
      ...(this.token ? { authorization: `Bearer ${this.token}` } : {}),
    };
  }
  async call<T>(request: LocalRequest, signal?: AbortSignal): Promise<T> {
    await this.connect();
    const body = JSON.stringify(request);
    // Snapshot reads and bulk geometry use HTTP; small commands and acks share
    // the same WebSocket as project/library notifications.
    if (
      !(request.action === "read" && !request.pointer) &&
      new TextEncoder().encode(body).byteLength <= 1024 * 1024
    ) {
      try {
        return await this.realtime.request<T>(request, signal);
      } catch (error) {
        if (error instanceof SocketReplyError)
          throw new LocalError(error.status, error.detail);
        if (!(error instanceof SocketUnavailable) || signal?.aborted)
          throw error;
      }
    }
    const response = await fetch(this.base + "/action", {
      method: "POST",
      credentials: "same-origin",
      headers: { ...this.headers(), "content-type": "application/json" },
      body,
      signal,
    });
    const result = await response.json();
    if (!response.ok) throw new LocalError(response.status, result);
    return result;
  }
  async upload(bytes: Uint8Array) {
    await this.connect();
    const r = await fetch(this.base + "/blobs", {
      method: "POST",
      credentials: "same-origin",
      headers: {
        ...this.headers(),
        "content-type": "application/octet-stream",
      },
      body: bytes as Uint8Array<ArrayBuffer>,
    });
    if (!r.ok) throw new LocalError(r.status, await r.json());
    return (await r.json()).blob as string;
  }
  async bytes(blob: string) {
    await this.connect();
    const r = await fetch(this.base + "/blobs/" + blob, {
      credentials: "same-origin",
      headers: this.headers(),
    });
    if (!r.ok) throw Error("读取本地资源失败");
    return new Uint8Array(await r.arrayBuffer());
  }
  async list() {
    return this.call<Summary[]>({ action: "list" });
  }
  async read(id: string) {
    const p = await this.call<Project>({ action: "read", id });
    const cached = this.records.get(id);
    if (cached?.revision === p.revision && cached.updated_at === p.updated_at) return cached;
    this.records.set(id, p);
    return p;
  }
  async create(
    id: string,
    name: string,
    document?: ProjectManifest,
    assets?: Record<string, Asset>,
  ) {
    const p = await this.call<Project>({
      action: "create",
      id,
      name,
      document,
      assets,
    });
    this.records.set(id, p);
    return p;
  }
  private channel(id: string) {
    let c = this.channels.get(id);
    if (!c) {
      const p = this.records.get(id);
      if (!p) throw Error("本地项目尚未打开");
      c = {
        staged: 0,
        local: p.document,
        pending: [],
        busy: false,
        blocked: false,
        conflict: false,
        waiters: [],
      };
      this.channels.set(id, c);
    }
    return c;
  }
  pending(id: string) {
    const c = this.channels.get(id);
    return !!c && (c.staged > 0 || c.busy || c.pending.length > 0);
  }
  conflict(id: string) {
    return this.channels.get(id)?.conflict ?? false;
  }
  hold(id: string) {
    const c = this.records.has(id) ? this.channel(id) : undefined;
    if (c) c.staged++;
    return () => {
      if (c) c.staged--;
    };
  }
  async save(
    id: string,
    next: ProjectManifest,
    prepare?: () => Promise<Record<string, Asset>>,
  ) {
    const c = this.channel(id),
      ops = changes(c.local, next);
    c.local = next;
    if (!prepare && !ops.length && !c.pending.length && !c.busy) return;
    if (ops.length || prepare)
      c.pending.push({
        changes: ops,
        prepare,
        request_id: crypto.randomUUID(),
      });
    return this.wait(id, c);
  }
  async resources(id: string, assets: Record<string, Asset>) {
    const c = this.channel(id);
    c.pending.push({ assets, request_id: crypto.randomUUID() });
    return this.wait(id, c);
  }
  private wait(id: string, c: Channel) {
    if (c.blocked)
      return Promise.reject(
        Error(
          c.conflict
            ? "项目已被其他客户端修改，本地修改仍保留，请选择如何继续"
            : "本地服务暂时不可用，修改仍在等待保存",
        ),
      );
    const p = new Promise<void>((resolve, reject) =>
      c.waiters.push({ resolve, reject }),
    );
    void this.flush(id, c);
    return p;
  }
  private async flush(id: string, c: Channel) {
    if (c.busy || c.blocked) return;
    c.busy = true;
    try {
      while (c.pending.length) {
        if (c.blocked)
          throw new LocalError(409, {
            error: { message: "项目已被其他客户端修改，本地修改仍保留" },
          });
        let batch = c.pending.shift()!;
        if (batch.changes && !batch.prepare) {
          while (c.pending[0]?.changes && !c.pending[0].prepare)
            batch = {
              request_id: batch.request_id,
              changes: [...batch.changes!, ...c.pending.shift()!.changes!],
            };
        }
        const record = this.records.get(id)!;
        try {
          const assets = { ...batch.assets, ...(await batch.prepare?.()) };
          if (c.blocked)
            throw new LocalError(409, {
              error: { message: "项目已被其他客户端修改，本地修改仍保留" },
            });
          if (
            batch.changes &&
            !batch.changes.length &&
            !Object.keys(assets).length
          )
            continue;
          const receipt = await this.call<Receipt>(
            batch.changes
              ? {
                  action: "edit",
                  id,
                  expected_revision: record.revision,
                  changes: batch.changes,
                  assets: Object.keys(assets).length ? assets : undefined,
                  actor: this.actor,
                  request_id: batch.request_id,
                }
              : {
                  action: "resources",
                  id,
                  expected_revision: record.revision,
                  assets,
                  actor: this.actor,
                },
          );
          const latest = this.records.get(id)!;
          if (receipt.revision > latest.revision) {
            const document = applyChanges(
              latest.document,
              receipt.changes.length ? receipt.changes : (batch.changes ?? []),
            );
            this.records.set(id, {
              ...latest,
              name: document.name,
              revision: receipt.revision,
              updated_at: receipt.updated_at,
              assets: { ...latest.assets, ...assets },
              document,
            });
          }
        } catch (e) {
          c.pending.unshift(batch);
          throw e;
        }
      }
      if (c.blocked)
        throw new LocalError(409, {
          error: { message: "项目已被其他客户端修改，本地修改仍保留" },
        });
       
      c.waiters.splice(0).forEach((w) => w.resolve());
    } catch (e) {
      c.blocked = true;
      c.conflict = e instanceof LocalError && e.status === 409;
      const error = e instanceof Error ? e : Error(String(e));
      c.waiters.splice(0).forEach((w) => w.reject(error));
      this.error = c.conflict
        ? "其他客户端更新了项目；本地修改保留，尚未覆盖服务器"
        : error.message;
      this.emit(this.records.get(id));
    } finally {
      c.busy = false;
    }
  }
  async resolve(id: string, keep: boolean) {
    const c = this.channel(id);
    if (c.busy) throw Error("正在保存，请稍后再试");
    const p = await this.read(id);
    c.blocked = false;
    c.conflict = false;
    if (!keep) {
      c.pending = [];
      c.local = p.document;
      this.error = "";
      this.emit(p);
      return;
    }
    const done = this.wait(id, c);
    await done;
    this.error = "";
    this.emit(this.records.get(id));
  }
  async discardAndRead(id: string) {
    const p = await this.read(id),
      c = this.channel(id);
    if (c.busy) throw Error("正在保存，请稍后再试");
    c.pending = [];
    c.blocked = false;
    c.conflict = false;
    c.local = p.document;
    return p;
  }
  start() {
    this.realtime.start();
  }
  stop() {
    this.realtime.stop();
  }
  private async drainNotices() {
    if (this.draining) return;
    this.draining = true;
    try {
      while (this.notices.length) {
        const frame = this.notices.shift()!;
        if (frame.type === "resync") await this.resync();
        else if (frame.type === "project_changed")
          await this.event(frame.receipt);
        else if (frame.type === "library_changed")
          this.libraryListeners.forEach((fn) => fn(frame.library));
      }
    } catch (error) {
      this.error = String(error);
      this.emit();
    } finally {
      this.draining = false;
    }
  }
  private async resync() {
    const summaries = new Map((await this.list()).map(p => [p.id,p]));
    for (const id of this.records.keys()) {
      const c = this.channels.get(id);
      while (c?.busy) await new Promise((r) => setTimeout(r, 10));
      const before = this.records.get(id)!;
      if (summaries.get(id)?.revision === before.revision) {
        if (c?.blocked && !c.conflict) { c.blocked = false; void this.flush(id,c); }
        continue;
      }
      const next = await this.read(id);
      if (next.revision !== before.revision) {
        if (c && (c.staged > 0 || c.pending.length || c.busy)) {
          c.blocked = true;
          c.conflict = true;
        } else if (c) c.local = next.document;
        this.emit(next);
      } else if (c?.blocked && !c.conflict) {
        c.blocked = false;
        void this.flush(id, c);
      }
    }
    this.emit();
    this.libraryListeners.forEach((fn) => fn("motion"));
  }
  private async event(receipt: Receipt) {
    if (receipt.actor === this.actor) {
      this.emit(undefined, receipt);
      return;
    }
    const previous = this.records.get(receipt.project_id);
    if (!previous) {
      this.emit(undefined, receipt);
      return;
    }
    if (receipt.revision <= previous.revision) return;
    const c = this.channels.get(receipt.project_id);
    if (c && (c.staged > 0 || c.pending.length || c.busy)) {
      c.blocked = true;
      c.conflict = true;
    }
    const next =
      receipt.assets_changed || receipt.revision !== previous.revision + 1
        ? await this.read(receipt.project_id)
        : {
            ...previous,
            revision: receipt.revision,
            updated_at: receipt.updated_at,
            document: applyChanges(previous.document, receipt.changes),
          };
    next.name = next.document.name;
    this.records.set(next.id, next);
    if (c && !c.blocked) c.local = next.document;
    this.emit(next, receipt);
  }
}
export const localClient = new LocalClient();

export const nativeTask = <T = any>(task: NativeTask, signal?: AbortSignal) =>
  localClient.call<T>({ action: "task", task }, signal);
