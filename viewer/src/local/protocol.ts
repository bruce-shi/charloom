import type { ProjectManifest } from "./project";
import type {
  WireRequest,
  WireProject,
  WireReceipt,
  WireChange,
  WireAsset,
  WireTask,
} from "./generated";
export type Change = WireChange;
export type Asset = WireAsset;
export type Project = Omit<WireProject, "document"> & {
  document: ProjectManifest;
};
export type Summary = Pick<Project, "id" | "name" | "revision" | "updated_at">;
export type Receipt = WireReceipt;
export type NativeTask = WireTask;
export type LocalRequest = WireRequest;
export type Directory = {
  path: string;
  parent: string | null;
  entries: { name: string; path: string; is_dir: boolean }[];
};

const token = (s: string) => s.replace(/~/g, "~0").replace(/\//g, "~1");
export function equal(a: any, b: any): boolean {
  if (a === b) return true;
  if (
    !a ||
    !b ||
    typeof a !== "object" ||
    typeof b !== "object" ||
    Array.isArray(a) !== Array.isArray(b)
  )
    return false;
  const ak = Object.keys(a),
    bk = Object.keys(b);
  return (
    ak.length === bk.length &&
    ak.every((k) => Object.hasOwn(b, k) && equal(a[k], b[k]))
  );
}
/** Identity skips unchanged geometry; large changed arrays travel as one value. */
export function changes(before: any, after: any, path = ""): Change[] {
  if (before === after) return [];
  if (Array.isArray(before) && Array.isArray(after)) {
    if (equal(before, after)) return [];
    if (before.length !== after.length || after.length > 256)
      return [{ op: "set", path, value: after }];
    return after.flatMap((v, i) => changes(before[i], v, `${path}/${i}`));
  }
  if (
    before &&
    after &&
    typeof before === "object" &&
    typeof after === "object" &&
    !Array.isArray(before) &&
    !Array.isArray(after)
  ) {
    return [
      ...Object.keys(before)
        .filter((k) => before[k] !== undefined && (!(k in after) || after[k] === undefined))
        .map((k) => ({ op: "remove" as const, path: `${path}/${token(k)}` })),
      ...Object.keys(after)
        .filter((k) => after[k] !== undefined)
        .flatMap((k) => changes(before[k], after[k], `${path}/${token(k)}`)),
    ];
  }
  return [{ op: "set", path, value: after }];
}
export function applyChanges<T>(value: T, ops: Change[]): T {
  let result = value;
  for (const op of ops) {
    const parts = op.path
      .slice(1)
      .split("/")
      .map((p) => p.replace(/~1/g, "/").replace(/~0/g, "~"));
    if (
      !op.path.startsWith("/") ||
      parts.some((p) => ["__proto__", "prototype", "constructor"].includes(p))
    )
      throw Error("无效的项目修改路径");
    const update = (node: any, at: number): any => {
      if (!node || typeof node !== "object")
        throw Error("项目字段的上级不存在");
      const copy = Array.isArray(node) ? [...node] : { ...node },
        key = parts[at];
      if (at === parts.length - 1) {
        if (op.op === "set") {
          if (Array.isArray(copy) && key === "-") copy.push(op.value);
          else copy[key as any] = op.value;
        } else if (Array.isArray(copy)) copy.splice(Number(key), 1);
        else delete copy[key];
      } else copy[key as any] = update(node[key], at + 1);
      return copy;
    };
    result = update(result, 0);
  }
  return result;
}
