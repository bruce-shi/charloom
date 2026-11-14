import {
  Component,
  Suspense,
  memo,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import {
  AnimationMixer,
  Group,
  Vector3,
  Mesh,
  MeshStandardMaterial,
  SkinnedMesh,
} from "three";
import { clone } from "three/addons/utils/SkeletonUtils.js";
import type { OrbitControls as Controls } from "three-stdlib";
import type { Point3 } from "./data";
import { faceClips } from "./faceClips";

export type CameraPose = {
  position: Point3;
  target: Point3;
  version: number;
  driver: string | null;
};
// Static viewports render only when edited. Camera changes invalidate both views.
const cameraListeners = new WeakMap<object, Set<() => void>>();
export function invalidateCamera(shared: React.RefObject<CameraPose>) {
  cameraListeners.get(shared)?.forEach((invalidate) => invalidate());
}
export type ViewSettings = {
  clay: boolean;
  wireframe: boolean;
  grid: boolean;
};
/** MakeHuman body-shape bars, -1..1 per target name (`build-weight`):
 * positive drives the morph `NAME+`, negative `NAME-` (sizing.rs). */
export type Bars = Record<string, number>;
/** What a mesh is (`charloom_part` extras: "garment", "hair"); the body,
 * eyes and other MakeHuman parts have none. */
const partOf = (n: Mesh) => (n.userData.charloom_part as string | undefined) ?? "body";
/** Bar names (without the sign) of a loaded model, the body's order first. */
function barNames(scene: Group | Mesh | any): string[] {
  const names: string[] = [];
  const add = (dict?: Record<string, number>) => {
    if (!dict) return;
    for (const key of Object.keys(dict).sort((a, b) => dict[a] - dict[b])) {
      const m = /^(.*)[+-]$/.exec(key);
      if (m && !names.includes(m[1])) names.push(m[1]);
    }
  };
  const meshes: Mesh[] = [];
  scene.traverse((n: any) => {
    if (n instanceof Mesh) meshes.push(n);
  });
  meshes.sort((a, b) => Number(partOf(b) === "body") - Number(partOf(a) === "body"));
  meshes.forEach((m) => add(m.morphTargetDictionary));
  return names;
}
 
export type ClipKind = "walk" | "face" | "talk";
const KINDS: ClipKind[] = ["walk", "face", "talk"];
const FACE_CLIP = "face_test";
/** One playhead shared by every viewport that shows an animated result, so
 * two runs compared side by side walk in step. Until played or scrubbed the
 * models stay in their rest pose. */
export class AnimationClock {
  enabled = false;
  playing = false;
  clip: ClipKind = "walk";
  private durations: Record<ClipKind, number> = { walk: 0, face: 0, talk: 0 };
  get duration() {
    return this.durations[this.clip];
  }
  has(kind: ClipKind) {
    return this.durations[kind] > 0;
  }
  /** Switch clip, back at its start and in the rest pose. */
  setClip(kind: ClipKind) {
    this.clip = kind;
    this.rest();
  }
  private offset = 0;
  private startedAt = 0;
  private listeners = new Set<() => void>();
  time() {
    const t = this.playing
      ? this.offset + (performance.now() - this.startedAt) / 1000
      : this.offset;
    return this.duration ? t % this.duration : t;
  }
  play() {
    if (this.playing) return;
    this.offset = this.time();
    this.startedAt = performance.now();
    this.playing = this.enabled = true;
    this.emit();
  }
  pause() {
    this.offset = this.time();
    this.playing = false;
    this.emit();
  }
  seek(t: number) {
    this.offset = t;
    this.startedAt = performance.now();
    this.enabled = true;
    this.emit();
  }
  rest() {
    this.playing = this.enabled = false;
    this.offset = 0;
    this.emit();
  }
  setDuration(kind: ClipKind, d: number) {
    if (d > this.durations[kind]) {
      this.durations[kind] = d;
      this.emit();
    }
  }
  subscribe(fn: () => void) {
    this.listeners.add(fn);
    return () => {
      this.listeners.delete(fn);
    };
  }
  private emit() {
    this.listeners.forEach((fn) => fn());
  }
}
const CAMERA = {
  position: [0, 0.02, 3.65] as Point3,
  fov: 36,
  near: 0.001,
  far: 30,
};
class SceneBoundary extends Component<
  { children: React.ReactNode },
  { error: string }
> {
  state = { error: "" };
  static getDerivedStateFromError(e: Error) {
    return { error: e.message };
  }
  render() {
    return this.state.error ? (
      <div className="scene-error" role="alert">
        模型未能加载<p>{this.state.error}</p>
      </div>
    ) : (
      this.props.children
    );
  }
}
const AssetObject = memo(function AssetObject({
  file,
  center,
  settings,
  ready,
  matrix,
  clock,
  parts,
  bars,
  onBars,
  onFace,
}: {
  file: string;
  center: Point3;
  settings: ViewSettings;
  ready: () => void;
  matrix?: number[];
  clock?: AnimationClock;
  parts?: string[];
  bars?: Bars;
  onBars?: (names: string[]) => void;
  onFace?: (point: Point3) => void;
}) {
  const gltf = useGLTF(file);
  const invalidate = useThree((state) => state.invalidate);
  const scene = useMemo(() => {
    const object = clone(gltf.scene);
    if (matrix?.length === 16) {
      // The scan placed in the result's working frame (column-major, as glTF).
      const framed = new Group();
      framed.matrixAutoUpdate = false;
      framed.matrix.fromArray(matrix);
      framed.add(object);
      object.updateMatrix();
      framed.updateMatrixWorld(true);
    }
    object.traverse((n) => {
      if (n instanceof Mesh) {
        n.geometry = n.geometry.clone();
        n.frustumCulled = false;
        n.userData.originalVisible = n.visible;
        const materials = Array.isArray(n.material) ? n.material : [n.material];
        n.userData.originalMaterials = materials;
        const copies = materials.map((m) => m.clone());
        n.material = Array.isArray(n.material) ? copies : copies[0];
      }
    });
    return object.parent ?? object;
  }, [gltf, matrix]);
  const mixer = useMemo(() => {
    if (!clock) return undefined;
    const m = new AnimationMixer(scene);
    const walk = gltf.animations.find((clip) => clip.name === "walking")
      ?? gltf.animations.find((clip) => clip.name !== FACE_CLIP && clip.name !== "face_talk");
    const embeddedFace = gltf.animations.find((clip) => clip.name === FACE_CLIP);
    const embeddedTalk = gltf.animations.find((clip) => clip.name === "face_talk");
    const { face, talk } = embeddedFace && embeddedTalk
      ? { face: embeddedFace, talk: embeddedTalk }
      : faceClips(scene);
    return {
      mixer: m,
      actions: {
        walk: walk && m.clipAction(walk),
        face: face && m.clipAction(face),
        talk: talk && m.clipAction(talk),
      },
    };
  }, [scene, gltf, clock]);
  useEffect(() => {
    if (!mixer || !clock) return;
    for (const kind of KINDS) {
      const action = mixer.actions[kind];
      if (action) clock.setDuration(kind, action.getClip().duration);
    }
    let active: ClipKind | null = null;
    const sync = () => {
      const wanted = clock.enabled ? clock.clip : null;
      if (wanted !== active) {
        mixer.mixer.stopAllAction();
        scene.traverse((n) => {
          if (n instanceof SkinnedMesh) n.skeleton.pose();
        });
        if (wanted) mixer.actions[wanted]?.reset().play();
        active = wanted;
      }
      invalidate();
    };
    sync();
    const unsubscribe = clock.subscribe(sync);
    return () => {
      unsubscribe();
      mixer.mixer.stopAllAction();
      mixer.mixer.uncacheRoot(scene);
    };
  }, [mixer, clock, scene, invalidate]);
  useFrame(() => {
    const action = clock?.enabled && mixer?.actions[clock.clip];
    if (!mixer || !clock || !action) return;
    mixer.mixer.setTime(clock.time() % action.getClip().duration);
    if (clock.playing) invalidate();
  });
  // Where the face is in the viewport (between the eye bones, else the
  // head bone), for the close-up camera.
  useEffect(() => {
    if (!onFace) return;
    const at = (name: string) => {
      // three drops the dot from node names ("eye.L" -> "eyeL").
      const node =
        scene.getObjectByName(name) ?? scene.getObjectByName(name.replace(".", ""));
      return node?.getWorldPosition(new Vector3());
    };
    scene.updateWorldMatrix(true, true);
    const l = at("eye.L"),
      r = at("eye.R"),
      p = l && r ? l.add(r).multiplyScalar(0.5) : at("head");
    if (p) onFace(p.toArray());
  }, [scene, onFace]);
  useEffect(() => {
    scene.traverse((n) => {
      if (!(n instanceof Mesh)) return;
      const hair = partOf(n) === "hair";
      n.visible =
        n.userData.originalVisible && (!parts || parts.includes(partOf(n)));
      const originals = n.userData.originalMaterials;
      const mats = Array.isArray(n.material) ? n.material : [n.material];
      mats.forEach((m, i) => {
        if (m instanceof MeshStandardMaterial) {
          m.map = settings.clay ? null : originals[i].map;
          m.color.set(
            settings.clay ? (hair ? "#6b737c" : "#bfc9c6") : originals[i].color,
          );
          m.wireframe = settings.wireframe;
          m.roughness = Math.max(0.65, originals[i].roughness ?? 0.8);
          m.needsUpdate = true;
        }
      });
    });
    invalidate();
  }, [scene, settings.clay, settings.wireframe, parts, invalidate]);
  useEffect(() => {
    onBars?.(barNames(scene));
  }, [scene, onBars]);
  // Body and clothes carry the same named targets, so one value moves both.
  useEffect(() => {
    scene.traverse((n) => {
      if (!(n instanceof Mesh) || !n.morphTargetDictionary || !n.morphTargetInfluences)
        return;
      const influences = n.morphTargetInfluences;
      influences.fill(0);
      for (const [name, value] of Object.entries(bars ?? {})) {
        const i = n.morphTargetDictionary[`${name}${value >= 0 ? "+" : "-"}`];
        if (i !== undefined) influences[i] = Math.abs(value);
      }
    });
    invalidate();
  }, [scene, bars, invalidate]);
  useEffect(() => {
    ready();
    return () => {
      scene.traverse((n) => {
        if (n instanceof Mesh) {
          n.geometry.dispose();
          for (const m of Array.isArray(n.material) ? n.material : [n.material])
            m.dispose();
        }
      });
    };
  }, [scene, ready]);
  return (
    <group position={center.map((v) => -v) as Point3}>
      <primitive object={scene} dispose={null} />
    </group>
  );
});
function LinkedControls({
  id,
  shared,
}: {
  id: string;
  shared: React.RefObject<CameraPose>;
}) {
  const controls = useRef<Controls>(null),
    seen = useRef(-1),
    applying = useRef(false);
  const { camera, invalidate } = useThree();
  useEffect(() => {
    let listeners = cameraListeners.get(shared);
    if (!listeners) {
      listeners = new Set();
      cameraListeners.set(shared, listeners);
    }
    listeners.add(invalidate);
    invalidate();
    return () => {
      listeners.delete(invalidate);
    };
  }, [shared, invalidate]);
  useFrame(() => {
    const c = controls.current,
      s = shared.current;
    if (!c || seen.current === s.version) return;
    applying.current = true;
    camera.position.fromArray(s.position);
    c.target.fromArray(s.target);
    c.update();
    seen.current = s.version;
    applying.current = false;
  });
  return (
    <OrbitControls
      ref={controls}
      makeDefault
      enableDamping={false}
      minDistance={0.12}
      maxDistance={7}
      onStart={() => {
        shared.current.driver = id;
      }}
      onChange={() => {
        const c = controls.current,
          s = shared.current;
        if (!c || applying.current || s.driver !== id) return;
        s.position = camera.position.toArray();
        s.target = c.target.toArray();
        s.version++;
        seen.current = s.version;
        invalidateCamera(shared);
      }}
    />
  );
}
export function ModelViewport({
  id,
  title,
  tag,
  file,
  center,
  shared,
  settings,
  matrix,
  clock,
  parts,
  bars,
  onBars,
  onFace,
}: {
  id: string;
  title: string;
  tag: string;
  file: string;
  center: Point3;
  shared: React.RefObject<CameraPose>;
  settings: ViewSettings;
  matrix?: number[];
  clock?: AnimationClock;
  parts?: string[];
  bars?: Bars;
  onBars?: (names: string[]) => void;
  onFace?: (point: Point3) => void;
}) {
  const [loaded, setLoaded] = useState("");
  const ready = useCallback(() => setLoaded(file), [file]);
  return (
    <section
      className="model-viewport"
      aria-label={title}
      data-model-file={file}
    >
      <div className="model-label">
        <span className={`model-dot ${id}`} />
        <strong>{title}</strong>
        <span>{tag}</span>
      </div>
      <div className="scene-canvas">
        <SceneBoundary key={file}>
          <Canvas
            camera={CAMERA}
            dpr={[1, 1.5]}
            gl={{ antialias: true }}
            frameloop="demand"
          >
            <color attach="background" args={["#20262b"]} />
            <hemisphereLight args={["#fff8ed", "#718a9b", 1.5]} />
            <directionalLight position={[2, 4, 4]} intensity={2.5} />
            <directionalLight position={[-3, 1, 2]} intensity={1.1} />
            <directionalLight position={[1, 2, -3]} intensity={1.4} />
            {settings.grid && (
              <gridHelper
                args={[8, 40, "#4a565b", "#30393f"]}
                position={[0, -center[1] - 0.005, 0]}
              />
            )}
            <Suspense fallback={null}>
              <AssetObject
                file={file}
                center={center}
                settings={settings}
                ready={ready}
                matrix={matrix}
                clock={clock}
                parts={parts}
                bars={bars}
                onBars={onBars}
                onFace={onFace}
              />
            </Suspense>
            <LinkedControls id={id} shared={shared} />
          </Canvas>
        </SceneBoundary>
        {loaded !== file && (
          <div className="scene-loading" role="status">
            <span />
            正在加载模型…
          </div>
        )}
      </div>

    </section>
  );
}
