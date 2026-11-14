import type { CharacterCase, RunMetrics } from "./data";

/** Result runs (scripts/mh-ide-sync.py) newest first. */
export const resultRuns = (cases: CharacterCase[]) =>
  cases
    .filter((c) => c.suite === "mh")
    .sort((a, b) => (b.timestamp ?? "").localeCompare(a.timestamp ?? ""));

export const characterOf = (c: CharacterCase) =>
  c.character ?? c.title.split(" ")[0];

export const runLabel = (c: CharacterCase) =>
  [c.tag ?? c.id, c.round].filter(Boolean).join(" · ");

export const shortTime = (iso?: string) =>
  iso
    ? new Date(iso).toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
      })
    : "";

type Metric = {
  key: keyof RunMetrics;
  label: string;
  unit: string;
  digits: number;
  /** Values above this are flagged. */
  warn?: number;
};
export const METRICS: Metric[] = [
  { key: "eyeMm", label: "眼部残差", unit: "mm", digits: 1, warn: 2 },
  { key: "skinP95Mm", label: "皮肤 p95", unit: "mm", digits: 1, warn: 10 },
  { key: "walkPct", label: "走路穿模", unit: "%", digits: 2, warn: 1 },
  { key: "faceBumps", label: "面部凸起", unit: "", digits: 0, warn: 100 },
  { key: "newFolds", label: "新折痕", unit: "", digits: 0, warn: 20 },
  { key: "inverted", label: "翻转三角形", unit: "", digits: 0, warn: 0 },
  { key: "edgeMax", label: "最大边拉伸", unit: "×", digits: 2, warn: 4 },
  { key: "seconds", label: "耗时", unit: "s", digits: 0 },
];
export const CARD_METRICS = METRICS.slice(0, 5);

export function metricText(m: RunMetrics | undefined, metric: Metric) {
  const v = m?.[metric.key];
  return v === null || v === undefined ? "—" : v.toFixed(metric.digits);
}
export function metricWarn(m: RunMetrics | undefined, metric: Metric) {
  const v = m?.[metric.key];
  return metric.warn !== undefined && typeof v === "number" && v > metric.warn;
}
