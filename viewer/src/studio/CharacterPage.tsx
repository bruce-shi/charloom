import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Link,
  useNavigate,
  useParams,
  useSearchParams,
} from "react-router";
import { ArrowLeft, Focus, Pause, Play, RotateCcw, Square } from "lucide-react";
import { useStudio, type Point3 } from "./data";
import {
  AnimationClock,
  ModelViewport,
  invalidateCamera,
  type Bars,
  type CameraPose,
  type ClipKind,
  type ViewSettings,
} from "./Viewport";
import { characterOf, resultRuns, runLabel, shortTime } from "./results";
import { CharacterSidebar } from "./CharacterSidebar";
import { EXPRESSIONS, moodAt } from "./faceClips";

const VIEWS = ["正面", "侧面", "背面", "透视"];
/** Left viewport: the source scan, or another run of the same character. */
const SCAN = "scan";

/** MakeHuman's names for the body-shape targets the pipeline exports
 * (sizing.rs, the prior's `body-shape.glb`), grouped as MakeHuman's
 * modelling tabs; a target missing here is listed under 其他. */
const GROUPS: [string, [string, string][]][] = [
  ["整体", [["build-weight", "体重"], ["build-muscle", "肌肉"]]],
  [
    "躯干",
    [
      ["stomach-pregnant", "腹部"],
      ["waist-circ", "腰围"],
      ["hips-circ", "臀围"],
      ["bust-circ", "胸围"],
      ["breast-size", "胸部大小"],
      ["breast-firmness", "胸部挺度"],
      ["pectoral", "胸肌"],
      ["dorsi", "背阔肌"],
      ["buttocks-volume", "臀部"],
      ["torso-scale-depth", "躯干厚度"],
      ["torso-scale-horiz", "躯干宽度"],
    ],
  ],
  [
    "四肢",
    [
      ["upperarm-circ", "上臂围"],
      ["thigh-circ", "大腿围"],
      ["calf-circ", "小腿围"],
      ["hand-scale", "手大小"],
      ["foot-scale", "脚大小"],
    ],
  ],
  [
    "头颈",
    [
      ["neck-circ", "颈围"],
      ["head-scale-horiz", "头宽"],
      ["head-scale-vert", "头高"],
      ["head-scale-depth", "头深"],
    ],
  ],
  [
    "面部",
    [
      ["nose-volume", "鼻子"],
      ["mouth-scale-horiz", "嘴宽"],
      ["mouth-scale-vert", "嘴高"],
      ["eye-scale", "眼睛"],
      ["cheek-volume", "脸颊"],
    ],
  ],
];

/** One mh-transfer result: the character in one viewport, its model
 * versions (dressed with the walk, clothes only, ...) and MakeHuman's
 * body-shape bars, which move the body and the clothes together. */
export default function CharacterPage() {
  const { cases, error } = useStudio(),
    { characterId } = useParams(),
    [params, setParams] = useSearchParams(),
    navigate = useNavigate();
  const selected = cases.find((c) => c.id === characterId);
  const siblings = useMemo(
    () =>
      selected
        ? resultRuns(cases).filter((c) => characterOf(c) === characterOf(selected))
        : [],
    [cases, selected],
  );
  const [settings, setSettings] = useState<ViewSettings>({
      clay: false,
      wireframe: false,
      grid: true,
    }),
    [viewName, setViewName] = useState("正面"),
    [available, setAvailable] = useState<string[] | null>(null),
    [bars, setBars] = useState<Bars>({}),
    [face, setFace] = useState<Point3 | null>(null);
  // Walk or face test; kept when the run or model changes.
  const clip = useRef<ClipKind>("walk");
  const stageId = params.get("stage");
  const stage = selected?.stages?.find((s) => s.id === stageId) ?? selected?.stages?.[0];
  const shared = useRef<CameraPose>({
    position: [0, 0.02, 3.65],
    target: [0, 0, 0],
    version: 0,
    driver: null,
  });
  const leftId = params.get("left") ?? SCAN;
  const leftRun = leftId === SCAN ? undefined : siblings.find((c) => c.id === leftId);
  const leftStage =
    leftRun?.stages?.find((s) => s.id === stage?.id) ?? leftRun?.stages?.[0];
  // One playhead for both sides, so two runs walk in step.
  const clock = useMemo(() => {
    const c = new AnimationClock();
    c.clip = clip.current;
    return c;
  }, [characterId, stage?.file, leftStage?.file]);
  useEffect(() => {
    setAvailable(null);
    setFace(null);
  }, [stage?.file]);
  const center = selected?.bodyCenter ?? ([0, 0.875, 0] as Point3);
  const camera = useCallback(
    (direction: string) => {
      const d = 3.65;
      const offset: Point3 =
        direction === "侧面"
          ? [d, 0, 0]
          : direction === "背面"
            ? [0, 0, -d]
            : direction === "透视"
              ? [d * 0.63, 0.12, d * 0.78]
              : [0, 0.02, d];
      shared.current = {
        position: offset,
        target: [0, 0, 0],
        version: shared.current.version + 1,
        driver: null,
      };
      invalidateCamera(shared);
      setViewName(direction);
    },
    [],
  );
  /** Close-up of the face, from the front or three-quarter. */
  const faceCamera = useCallback(
    (direction: "面部" | "面部 3/4") => {
      if (!face) return;
      const d = 0.8,
        a = direction === "面部" ? 0 : Math.PI / 5;
      shared.current = {
        position: [face[0] + d * Math.sin(a), face[1], face[2] + d * Math.cos(a)],
        target: face,
        version: shared.current.version + 1,
        driver: null,
      };
      invalidateCamera(shared);
      setViewName(direction);
    },
    [face],
  );
  const setClip = useCallback(
    (kind: ClipKind) => {
      clip.current = kind;
      clock.setClip(kind);
      if (kind === "walk") camera("正面");
      else faceCamera("面部");
    },
    [clock, camera, faceCamera],
  );
  useEffect(() => {
    camera("正面");
    setBars({});
  }, [characterId, camera]);
  // A new run or model opened during the face test: back on its face.
  useEffect(() => {
    if (face && clip.current !== "walk") faceCamera("面部");
  }, [face, faceCamera]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input,textarea,select,dialog")) return;
      if (e.key.toLowerCase() === "f" || e.key === "1") camera("正面");
      if (e.key === "3") camera("侧面");
      if (e.key === "7") camera("背面");
      if (e.key === " ") {
        e.preventDefault();
        clock.playing ? clock.pause() : clock.play();
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [camera, clock]);
  if (error)
    return (
      <div className="studio-empty" role="alert">
        {error}
      </div>
    );
  if (!cases.length) return <div className="studio-empty">正在读取结果…</div>;
  if (!selected)
    return (
      <div className="studio-empty">
        没有找到这个结果。<Link to="/assets">返回结果库</Link>
      </div>
    );
  const index = siblings.findIndex((c) => c.id === selected.id);
  const goRun = (id: string) =>
    navigate(`/characters/${encodeURIComponent(id)}/editor?${params.toString()}`);
  return (
    <div className="character-editor inspector-hidden result-preview">
      <div className="character-workspace-header">
        <Link className="result-back" to="/assets" title="返回结果库">
          <ArrowLeft size={14} />
        </Link>
        <strong className="character-name">{characterOf(selected)}</strong>
        <label className="run-picker" title="这个角色的所有结果，最新在前">
          结果
          <select
            aria-label="切换结果"
            value={selected.id}
            onChange={(e) => goRun(e.target.value)}
          >
            {siblings.map((c) => (
              <option key={c.id} value={c.id}>
                {runLabel(c)} · {shortTime(c.timestamp)}
                {c.change ? ` · ${c.change}` : ""}
              </option>
            ))}
          </select>
        </label>
        <button
          className="icon-button"
          aria-label="较新的结果"
          disabled={index <= 0}
          onClick={() => goRun(siblings[index - 1].id)}
        >
          ‹
        </button>
        <button
          className="icon-button"
          aria-label="较早的结果"
          disabled={index < 0 || index >= siblings.length - 1}
          onClick={() => goRun(siblings[index + 1].id)}
        >
          ›
        </button>
      </div>
      <div className="character-editor-content">
        <div className="editor-toolbar">
          <div className="tool-group view-buttons">
            {VIEWS.map((v) => (
              <button key={v} aria-pressed={viewName === v} onClick={() => camera(v)}>
                {v}
              </button>
            ))}
            {(["面部", "面部 3/4"] as const).map((v) => (
              <button
                key={v}
                aria-pressed={viewName === v}
                disabled={!face}
                title="面部特写"
                onClick={() => faceCamera(v)}
              >
                {v}
              </button>
            ))}
          </div>
          <button className="tool-button" title="全身居中 F" onClick={() => camera("正面")}>
            <Focus size={16} />
            <span>找回角色</span>
          </button>
          <div className="toolbar-spacer" />
          <label className="field-inline">
            左侧
            <select
              aria-label="左侧对比"
              value={leftRun ? leftRun.id : SCAN}
              onChange={(e) =>
                setParams(
                  (p) => {
                    if (e.target.value === SCAN) p.delete("left");
                    else p.set("left", e.target.value);
                    return p;
                  },
                  { replace: true },
                )
              }
            >
              <option value={SCAN}>原扫描</option>
              {siblings
                .filter((c) => c.id !== selected.id)
                .map((c) => (
                  <option key={c.id} value={c.id}>
                    {runLabel(c)} · {shortTime(c.timestamp)}
                  </option>
                ))}
            </select>
          </label>
          <label className="field-inline">
            模型
            <select
              aria-label="模型"
              value={stage?.id ?? ""}
              onChange={(e) =>
                setParams(
                  (p) => {
                    p.set("stage", e.target.value);
                    return p;
                  },
                  { replace: true },
                )
              }
            >
              {selected.stages?.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.label}
                </option>
              ))}
            </select>
          </label>
          {(
            [
              ["clay", "灰模"],
              ["wireframe", "线框"],
              ["grid", "地面"],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              className="tool-button"
              aria-pressed={settings[key]}
              onClick={() => setSettings((s) => ({ ...s, [key]: !s[key] }))}
            >
              {label}
            </button>
          ))}
        </div>
        <div className="editor-body">
          <div className="editor-main">
            <div className="viewport-grid">
              <div className="source-viewport">
                {leftRun && leftStage ? (
                  <ModelViewport
                    key={"run-" + leftRun.id}
                    id="original"
                    title={runLabel(leftRun)}
                    tag={shortTime(leftRun.timestamp)}
                    file={leftStage.file}
                    center={leftRun.bodyCenter ?? center}
                    shared={shared}
                    settings={settings}
                    clock={clock}
                    parts={leftStage.parts}
                  />
                ) : (
                  <ModelViewport
                    key="scan"
                    id="original"
                    title="原扫描"
                    tag="SOURCE"
                    file={selected.sourceFile ?? ""}
                    center={center}
                    shared={shared}
                    settings={settings}
                    matrix={selected.sourceMatrix}
                  />
                )}
              </div>
              <ModelViewport
                id="fitted"
                title={runLabel(selected)}
                tag={stage?.label ?? ""}
                file={stage?.file ?? ""}
                center={center}
                shared={shared}
                settings={settings}
                clock={clock}
                parts={stage?.parts}
                bars={bars}
                onBars={setAvailable}
                onFace={setFace}
              />
            </div>
            <WalkBar clock={clock} setClip={setClip} />
          </div>
          <CharacterSidebar key={selected.id} renders={selected.renders}>
            <ShapePanel available={available} bars={bars} setBars={setBars} />
          </CharacterSidebar>
        </div>
      </div>
    </div>
  );
}

function ShapePanel({
  available,
  bars,
  setBars,
}: {
  available: string[] | null;
  bars: Bars;
  setBars: (f: (b: Bars) => Bars) => void;
}) {
  const names = available ?? [];
  const known = new Set(GROUPS.flatMap(([, items]) => items.map(([n]) => n)));
  const groups: [string, [string, string][]][] = [
    ...GROUPS.map(
      ([g, items]) =>
        [g, items.filter(([n]) => names.includes(n))] as [string, [string, string][]],
    ),
    ["其他", names.filter((n) => !known.has(n)).map((n) => [n, n] as [string, string])],
  ];
  const changed = Object.values(bars).some((v) => v !== 0);
  return (
    <div className="shape-panel">
      <div className="shape-panel-title">
        <strong>MakeHuman 体型</strong>
        <button
          className="tool-button"
          disabled={!changed}
          title="所有滑杆回到 0（拟合结果）"
          onClick={() => setBars(() => ({}))}
        >
          <RotateCcw size={13} />
          <span>复位</span>
        </button>
      </div>
      {available === null ? (
        <p className="shape-panel-empty">正在读取模型的体型目标…</p>
      ) : !available.length ? (
        <p className="shape-panel-empty">
          这个结果没有体型滑杆。用 size_morphs 打开的运行（x29 起）才带 MakeHuman
          体型目标。
        </p>
      ) : (
        groups
          .filter(([, items]) => items.length)
          .map(([group, items]) => (
            <section key={group} className="shape-group">
              <h3>{group}</h3>
              {items.map(([name, label]) => {
                const value = bars[name] ?? 0;
                return (
                  <label key={name} className="shape-bar" title={name}>
                    <span>{label}</span>
                    <input
                      type="range"
                      min={-1}
                      max={1}
                      step={0.01}
                      value={value}
                      aria-label={label}
                      onChange={(e) => {
                        const v = Number(e.target.value);
                        setBars((b) => ({ ...b, [name]: Math.abs(v) < 0.015 ? 0 : v }));
                      }}
                      onDoubleClick={() => setBars((b) => ({ ...b, [name]: 0 }))}
                    />
                    <output className={value ? "" : "muted"}>
                      {(value * 100).toFixed(0)}
                    </output>
                  </label>
                );
              })}
            </section>
          ))
      )}
    </div>
  );
}

function WalkBar({
  clock,
  setClip,
}: {
  clock: AnimationClock;
  setClip: (kind: ClipKind) => void;
}) {
  const [, redraw] = useState(0);
  const slider = useRef<HTMLInputElement>(null),
    label = useRef<HTMLSpanElement>(null),
    steps = useRef<HTMLDivElement>(null);
  const face = clock.clip === "face",
    talk = clock.clip === "talk";
  // One expression per second: held for the first half, then blended into
  // the next (faceClips.ts).
  const count = face ? EXPRESSIONS.length : 0;
  const mood = useRef<HTMLSpanElement>(null);
  useEffect(() => clock.subscribe(() => redraw((n) => n + 1)), [clock]);
  useEffect(() => {
    let frame = 0;
    const tick = () => {
      const t = clock.time();
      if (slider.current) slider.current.value = String(t);
      if (label.current)
        label.current.textContent = `${t.toFixed(2)} / ${clock.duration.toFixed(2)} s`;
      const current = clock.enabled && count ? Math.round(t) % count : -1;
      steps.current?.querySelectorAll("button").forEach((b, k) => {
        b.setAttribute("aria-pressed", String(k === current));
      });
      if (mood.current) mood.current.textContent = clock.enabled ? moodAt(t) : "";
      if (clock.playing) frame = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(frame);
  });
  return (
    <div className="stage-bottom walk-bar">
      <div className="tool-group clip-buttons">
        {(
          [
            ["walk", "走路"],
            ["face", "表情"],
            ["talk", "说话"],
          ] as const
        ).map(([kind, name]) => (
          <button
            key={kind}
            aria-pressed={clock.clip === kind}
            disabled={!clock.has(kind)}
            title={
              clock.has(kind)
                ? kind === "face"
                  ? "表情测试：逐个检查面部是否贴合 MakeHuman 面部骨骼"
                  : kind === "talk"
                    ? "说话：边说话边变换表情，带眨眼、视线和头部动作"
                    : "走路动画"
                : "这个模型没有这段动画"
            }
            onClick={() => setClip(kind)}
          >
            {name}
          </button>
        ))}
      </div>
      <button
        aria-label={clock.playing ? "暂停动画" : "播放动画"}
        title="播放 / 暂停（空格）"
        onClick={() => (clock.playing ? clock.pause() : clock.play())}
        disabled={!clock.duration}
      >
        {clock.playing ? <Pause size={13} /> : <Play size={13} />}
        {clock.playing ? "暂停" : "播放"}
      </button>
      <input
        ref={slider}
        type="range"
        aria-label="动画时间"
        min={0}
        max={clock.duration || 1}
        step={0.01}
        disabled={!clock.duration}
        onChange={(e) => {
          clock.pause();
          clock.seek(Number(e.target.value));
        }}
      />
      <span ref={label} className="walk-time" />
      {face && (
        <div ref={steps} className="tool-group expression-steps" aria-label="表情">
          {Array.from({ length: count }, (_, k) => (
            <button
              key={k}
              title="停在这个表情"
              onClick={() => {
                clock.pause();
                clock.seek(k + 0.25);
              }}
            >
              {EXPRESSIONS[k][0]}
            </button>
          ))}
        </div>
      )}
      {talk && <span ref={mood} className="talk-mood" title="当前情绪" />}
      <button title="回到静止姿势" onClick={() => clock.rest()} disabled={!clock.enabled}>
        <Square size={11} />
        静止姿势
      </button>
    </div>
  );
}
