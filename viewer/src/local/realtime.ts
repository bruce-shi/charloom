import type {
  WireRequest,
  WireSocketClient,
  WireSocketServer,
} from "./generated";

export class SocketUnavailable extends Error {}
export class SocketReplyError extends Error {
  constructor(
    readonly status: number,
    readonly detail: any,
  ) {
    super(detail?.error?.message ?? "实时请求失败");
  }
}
type Peer = { socket: WebSocket; closed: Promise<void> };
type Pending = {
  peer: Peer;
  resolve: (value: any) => void;
  reject: (error: Error) => void;
  finish: () => void;
};
type Options = {
  base: () => string;
  token: () => string | undefined;
  prepare: () => Promise<unknown>;
  event: (frame: WireSocketServer) => void;
  state: (connected: boolean, error?: string) => void;
};
/** One bounded duplex connection. HTTP remains available for bulk assets and snapshots. */
export class Realtime {
  private peer?: Peer;
  private opening?: Promise<Peer>;
  private connecting?: WebSocket;
  private running = false;
  private epoch = 0;
  private pending = new Map<string, Pending>();
  constructor(private readonly options: Options) {}
  start() {
    if (this.running) return;
    this.running = true;
    void this.watch(++this.epoch);
  }
  stop() {
    this.running = false;
    this.epoch++;
    this.peer?.socket.close();
    this.connecting?.close();
    this.opening = undefined;
  }
  private open() {
    if (this.peer?.socket.readyState === WebSocket.OPEN)
      return Promise.resolve(this.peer);
    if (this.opening) return this.opening;
    const pending = this.createPeer(this.epoch);
    this.opening = pending;
    void pending
      .finally(() => {
        if (this.opening === pending) this.opening = undefined;
      })
      .catch(() => {});
    return pending;
  }
  private async createPeer(epoch: number): Promise<Peer> {
    await this.options.prepare();
    if (!this.running || epoch !== this.epoch)
      throw new SocketUnavailable("实时连接已停止");
    if (typeof WebSocket === "undefined")
      throw new SocketUnavailable("当前客户端不支持 WebSocket");
    const url = new URL(
      this.options.base() + "/ws",
      typeof window !== "undefined" ? window.location.href : undefined,
    );
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    const token = this.options.token();
    const socket = new WebSocket(
      url,
      token ? ["charloom.v1", "charloom.auth." + token] : ["charloom.v1"],
    );
    this.connecting = socket;
    let close!: () => void;
    const peer = {
      socket,
      closed: new Promise<void>((resolve) => {
        close = resolve;
      }),
    };
    return new Promise<Peer>((resolve, reject) => {
      let opened = false;
      const timeout = setTimeout(() => {
        reject(new SocketUnavailable("实时连接暂时不可用"));
        socket.close();
      }, 3000);
      socket.onopen = () => {
        clearTimeout(timeout);
        opened = true;
        this.peer = peer;
        if (this.connecting === socket) this.connecting = undefined;
        this.options.state(true);
        resolve(peer);
      };
      socket.onmessage = (event) => {
        let frame: WireSocketServer;
        try {
          frame = JSON.parse(String(event.data));
        } catch {
          socket.close(1002, "Invalid protocol JSON");
          return;
        }
        if (frame.type === "result" || frame.type === "error") {
          const request = frame.id ? this.pending.get(frame.id) : undefined;
          if (request?.peer === peer) {
            this.pending.delete(frame.id!);
            request.finish();
            frame.type === "result"
              ? request.resolve(frame.value)
              : request.reject(
                  new SocketReplyError(frame.status, frame.detail),
                );
          } else if (frame.type === "error" && !frame.id)
            this.options.state(
              false,
              (frame.detail as any)?.error?.message ?? "实时协议错误",
            );
        } else if (frame.type === "ready") {
          if (frame.protocol !== 1)
            socket.close(1002, "Protocol version mismatch");
        } else this.options.event(frame);
      };
      socket.onerror = () => {
        if (!opened) reject(new SocketUnavailable("实时连接暂时不可用"));
      };
      socket.onclose = () => {
        clearTimeout(timeout);
        close();
        if (this.connecting === socket) this.connecting = undefined;
        if (!opened) reject(new SocketUnavailable("实时连接暂时不可用"));
        if (this.peer === peer) {
          this.peer = undefined;
          this.options.state(false, "实时连接断开，正在重连");
        }
        for (const [id, request] of this.pending)
          if (request.peer === peer) {
            this.pending.delete(id);
            request.finish();
            request.reject(Error("实时连接中断，尚未确认的修改已保留"));
          }
      };
    });
  }
  async request<T>(request: WireRequest, signal?: AbortSignal): Promise<T> {
    if (signal?.aborted) throw new DOMException("操作已取消", "AbortError");
    this.start();
    const peer = await this.open();
    if (signal?.aborted) throw new DOMException("操作已取消", "AbortError");
    if (peer.socket.readyState !== WebSocket.OPEN)
      throw new SocketUnavailable("实时连接暂时不可用");
    if (peer.socket.bufferedAmount > 4 * 1024 * 1024 || this.pending.size >= 32)
      throw Error("实时队列繁忙，修改仍待保存");
    const id = crypto.randomUUID(),
      frame: WireSocketClient = { type: "action", id, request };
    return new Promise<T>((resolve, reject) => {
      const abort = () => {
        this.pending.delete(id);
        finish();
        reject(new DOMException("操作已取消", "AbortError"));
      };
      const timeout = setTimeout(() => {
        this.pending.delete(id);
        finish();
        reject(Error("实时请求超时，请核对保存结果"));
      }, 1800_000);
      const finish = () => {
        clearTimeout(timeout);
        signal?.removeEventListener("abort", abort);
      };
      this.pending.set(id, { peer, resolve, reject, finish });
      signal?.addEventListener("abort", abort, { once: true });
      try {
        peer.socket.send(JSON.stringify(frame));
      } catch (error) {
        this.pending.delete(id);
        finish();
        reject(error);
      }
    });
  }
  private async watch(epoch: number) {
    while (this.running && epoch === this.epoch) {
      try {
        const peer = await this.open();
        await peer.closed;
      } catch (error) {
        if (epoch === this.epoch && this.running)
          this.options.state(false, String(error));
      }
      if (this.running && epoch === this.epoch)
        await new Promise((resolve) => setTimeout(resolve, 1000));
    }
  }
}
