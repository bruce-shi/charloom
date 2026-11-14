// Face review clips, built in the viewer on MakeHuman's face rig so every
// result can play them, whatever clips its file carries:
//   face: the test expressions of crates/charloom-mh/src/face_test.rs, one
//         per second (keep the two tables the same);
//   talk: a 12 s loop of someone talking while their mood changes, the
//         mouth shaped by each sound (lip sync), with blinks, glances and
//         a little head movement.
import {
  AnimationClip,
  type KeyframeTrack,
  type Mesh,
  NumberKeyframeTrack,
  Object3D,
  Quaternion,
  QuaternionKeyframeTrack,
  Vector3,
  VectorKeyframeTrack,
} from "three";

/** What the face is doing at one moment. 0 is rest; jaw, glance and head
 * values are radians, the others 0..1. */
type Face = {
  jaw: number;
  smile: number;
  browUp: number;
  frown: number;
  blink: number;
  /** Mouth shape, -1..1: corners apart (+) or drawn together (-), upper
   * lip raised, lower lip raised, lips pushed forward. */
  lipWide: number;
  lipUpper: number;
  lipLower: number;
  lipForward: number;
  /** Glance to the character's left, and up. */
  eyeLeft: number;
  eyeUp: number;
  headLeft: number;
  headUp: number;
  headRoll: number;
};
const REST: Face = {
  jaw: 0,
  smile: 0,
  browUp: 0,
  frown: 0,
  blink: 0,
  lipWide: 0,
  lipUpper: 0,
  lipLower: 0,
  lipForward: 0,
  eyeLeft: 0,
  eyeUp: 0,
  headLeft: 0,
  headUp: 0,
  headRoll: 0,
};

/** The test expressions, in face_test.rs order. */
export const EXPRESSIONS: [string, Partial<Face>][] = [
  ["中性", {}],
  ["张嘴", { jaw: 0.3 }],
  ["微笑", { smile: 1 }],
  ["抬眉", { browUp: 1 }],
  ["皱眉", { frown: 1 }],
  ["眨眼", { blink: 1 }],
  ["向左看", { eyeLeft: 0.35 }],
  ["向上看", { eyeUp: 0.3 }],
];

const X = new Vector3(1, 0, 0),
  Y = new Vector3(0, 1, 0),
  Z = new Vector3(0, 0, 1);
/** A turn of one joint: about its own axis, or about the model's axis
 * (X to the character's left, Y up, Z forward). The eye bones are rolled
 * and mirrored, so the eyes and the head turn about the model's axes. */
type Turn = [axis: Vector3, model: boolean, angle: (f: Face) => number];
const side = (name: string, turns: Turn[]): [string, Turn[]][] => [
  [`${name}.L`, turns],
  [`${name}.R`, turns],
];
/** face_test.rs angles: smile turns levator03 by -0.5 and risorius03 by
 * -0.6, and so on. */
const RIG: [string, Turn[]][] = [
  ["jaw", [[X, false, (f) => f.jaw]]],
  ...side("levator03", [[X, false, (f) => -0.5 * f.smile]]),
  ...side("risorius03", [[X, false, (f) => -0.6 * f.smile + 0.6 * f.frown]]),
  ...side("oculi01", [[X, false, (f) => -0.5 * f.browUp]]),
  ...side("oculi02", [[X, false, (f) => -0.5 * f.browUp + 0.4 * f.frown]]),
  ...side("orbicularis03", [[X, false, (f) => -0.5 * f.blink]]),
  ...side("eye", [
    [Y, true, (f) => f.eyeLeft],
    [X, true, (f) => -f.eyeUp],
  ]),
  [
    "head",
    [
      [Y, true, (f) => f.headLeft],
      [X, true, (f) => -f.headUp],
      [Z, true, (f) => f.headRoll],
    ],
  ],
];

 
type Slide = (f: Face) => [number, number, number];
const upper: Slide = (f) => [(f.lipWide < 0 ? 0.05 : 0.03) * f.lipWide, 0.04 * f.lipUpper, 0.06 * f.lipForward],
  lower: Slide = (f) => [(f.lipWide < 0 ? 0.05 : 0.03) * f.lipWide, 0.05 * f.lipLower, 0.06 * f.lipForward];
// Rounded lips draw the corners in further than a wide mouth spreads them.
const corner: Slide = (f) => [
  (f.lipWide < 0 ? 0.1 : 0.06) * f.lipWide,
  0,
  0.03 * f.lipForward,
];
const SLIDES: [string, Slide][] = [
  ["oris05", (f) => [0, upper(f)[1], upper(f)[2]]],
  ["oris03.L", upper],
  ["oris03.R", upper],
  ["oris01", (f) => [0, lower(f)[1], lower(f)[2]]],
  ["oris07.L", lower],
  ["oris07.R", lower],
  ["levator05.L", corner],
  ["levator05.R", corner],
];

const smooth = (u: number) => {
  const x = Math.min(1, Math.max(0, u));
  return x * x * (3 - 2 * x);
};
const mix = (a: Face, b: Face, u: number) => {
  const out = { ...a };
  for (const k of Object.keys(a) as (keyof Face)[]) out[k] = a[k] + (b[k] - a[k]) * u;
  return out;
};

/** Each expression held for half a second, half a second to the next. */
function testFace(t: number): Face {
  const pose = (k: number): Face => ({ ...REST, ...EXPRESSIONS[k]?.[1] });
  const k = Math.floor(t);
  return mix(pose(k), pose(k + 1), smooth((t - k - 0.5) * 2));
}

export const TALK_SECONDS = 12;
/** The mood through the talk: [from second, name, smile, brows up, frown]. */
const MOODS: [number, string, number, number, number][] = [
  [0, "平静", 0.15, 0, 0],
  [2.7, "微笑", 0.8, 0.15, 0],
  [5.5, "惊讶", 0.1, 0.9, 0],
  [7.7, "皱眉", 0, 0, 0.8],
  [10.1, "微笑", 0.6, 0.2, 0],
  [11.4, "平静", 0.15, 0, 0],
];
export const moodAt = (t: number) =>
  MOODS.reduce((name, m) => (t >= m[0] ? m[1] : name), MOODS[0][1]);
/** Mouth shapes by sound: [jaw, wide, upper lip, lower lip, forward]. */
const SOUNDS: Record<string, [number, number, number, number, number]> = {
  A: [0.19, 0.1, 0, 0, 0], // father
  E: [0.09, 1, 0.5, -0.3, 0], // bed, see
  I: [0.05, 0.8, 0.3, 0, 0], // sit
  O: [0.14, -0.8, 0, 0, 0.7], // go
  U: [0.06, -1, 0, 0, 1], // you
  M: [0, 0, -0.3, 0.3, 0], // m b p: lips together
  F: [0.04, 0, 0.4, 0.9, 0], // f v: lower lip to the upper teeth
  S: [0.04, 0.5, 0.3, -0.2, 0], // s z t d
  L: [0.09, 0.2, 0, 0, 0], // l n
};
/** Spoken phrases, with a breath between them: [from, to, sounds]. */
const PHRASES: [number, number, string][] = [
  [0.3, 2.2, "SIAOMEIUAFA"],
  [2.9, 5.0, "UOMELIASOUEMA"],
  [5.7, 7.2, "AOUASEMA"],
  [8.0, 9.7, "MUSIFALEOMA"],
  [10.3, 11.5, "EIAMOLU"],
];
/** The mouth through the loop: [second, jaw, wide, upper, lower, forward],
 * one key at the middle of each sound (vowels take twice a consonant's
 * time) and a closed mouth around each phrase. */
const MOUTH = (() => {
  const vowel = (c: string) => "AEIOU".includes(c),
    stress = [0.8, 1.1, 0.7, 1, 0.9];
  const keys: number[][] = [[0, 0, 0, 0, 0, 0]];
  let n = 0;
  for (const [from, to, sounds] of PHRASES) {
    const unit =
      (to - from - 0.2) / [...sounds].reduce((sum, c) => sum + (vowel(c) ? 2 : 1), 0);
    let t = from + 0.1;
    keys.push([from, 0, 0, 0, 0, 0]);
    for (const c of sounds) {
      const length = unit * (vowel(c) ? 2 : 1),
        [jaw, ...lips] = SOUNDS[c];
      keys.push([t + length / 2, jaw * (vowel(c) ? stress[n++ % stress.length] : 1), ...lips]);
      t += length;
    }
    keys.push([to, 0, 0, 0, 0, 0]);
  }
  keys.push([TALK_SECONDS, 0, 0, 0, 0, 0]);
  return keys;
})();
const BLINKS = [1.1, 3.3, 5.75, 6.9, 8.6, 10.2, 11.3];
/** Glances: [second, left, up], each reached in a quick jump. */
const GLANCES: [number, number, number][] = [
  [0, 0, 0],
  [1.1, 0.1, 0.03],
  [2.6, 0, 0],
  [4.2, -0.12, -0.04],
  [5.75, 0, 0.08],
  [7.6, 0.06, -0.1],
  [9.0, -0.05, -0.08],
  [10.2, 0, 0],
];

function talkFace(t: number): Face {
  const f = { ...REST };
  // Mood: each one eases in over 0.6 s.
  let mood = MOODS[0].slice(2) as number[];
  for (const m of MOODS.slice(1)) {
    const u = smooth((t - m[0]) / 0.6);
    mood = mood.map((v, i) => v + ((m[i + 2] as number) - v) * u);
  }
  [f.smile, f.browUp, f.frown] = mood;
  // Speech: the mouth eases from each sound's shape to the next; a
  // surprised face keeps the mouth a little open.
  const next = Math.max(1, MOUTH.findIndex((k) => k[0] > t)),
    a = MOUTH[next - 1],
    b = MOUTH[next],
    u = smooth((t - a[0]) / (b[0] - a[0]));
  [f.jaw, f.lipWide, f.lipUpper, f.lipLower, f.lipForward] = a
    .slice(1)
    .map((v, i) => v + (b[i + 1] - v) * u);
  // Wide-open sounds lift the brows a little.
  f.browUp += Math.max(0, f.jaw - 0.12);
  f.jaw += 0.05 * mood[1];
  for (const b of BLINKS)
    f.blink = Math.max(f.blink, smooth((t - b) / 0.08) * smooth((b + 0.24 - t) / 0.16));
  // A smile narrows the eyes a little.
  f.blink = Math.max(f.blink, 0.15 * f.smile);
  for (let i = 1; i < GLANCES.length; i++) {
    const u = smooth((t - GLANCES[i][0]) / 0.07);
    f.eyeLeft += (GLANCES[i][1] - GLANCES[i - 1][1]) * u;
    f.eyeUp += (GLANCES[i][2] - GLANCES[i - 1][2]) * u;
  }
  // The head keeps moving a little; all of it repeats with the loop.
  const w = (2 * Math.PI * t) / TALK_SECONDS;
  f.headLeft = 0.05 * Math.sin(2 * w) + 0.03 * Math.sin(3 * w + 1);
  f.headUp = 0.035 * Math.sin(3 * w) + 0.03 * mood[1] - 0.04 * mood[2];
  f.headRoll = 0.03 * Math.sin(w) + 0.02 * Math.sin(4 * w + 2);
  return f;
}

/** `face(t)` sampled on the scene's face joints (the scene in its rest
 * pose). Undefined when the model has no MakeHuman face rig. */
function clip(
  scene: Object3D,
  name: string,
  seconds: number,
  face: (t: number) => Face,
): AnimationClip | undefined {
  // three drops the dot from node names ("eye.L" -> "eyeL").
  const node = (n: string) =>
    scene.getObjectByName(n) ?? scene.getObjectByName(n.replaceAll(".", ""));
  if (!node("jaw")) return undefined;
  const fps = 30,
    frames = Math.round(seconds * fps);
  const times = Array.from({ length: frames + 1 }, (_, i) => i / fps);
  const faces = times.map((t) => face(t % seconds));
  const tracks: KeyframeTrack[] = [];
  scene.traverse((object) => {
    const mesh = object as Mesh;
    const blink = mesh.morphTargetDictionary?.eyeBlink;
    if (blink !== undefined) {
      tracks.push(new NumberKeyframeTrack(
        `${mesh.uuid}.morphTargetInfluences[${blink}]`,
        times,
        faces.map((f) => f.blink),
      ));
    }
  });
  const eyes = [node("eye.L"), node("eye.R")].map((e) => e?.getWorldPosition(new Vector3()));
  const span = eyes[0] && eyes[1] ? eyes[0].distanceTo(eyes[1]) : 0;
  // A reconstructed mouth can have less than a millimetre between the lips.
  // Limit opposing closure slides to its measured neutral clearance.
  const clearance: unknown = node("jaw")?.userData.charloomLipClearanceM;
  if (span > 0 && typeof clearance === "number" && clearance >= 0) {
    for (const f of faces) {
      const approach = span * (0.05 * f.lipLower - 0.04 * f.lipUpper);
      if (approach <= clearance) continue;
      if (f.lipUpper < 0 && f.lipLower > 0) {
        const scale = clearance / approach;
        f.lipUpper *= scale;
        f.lipLower *= scale;
      } else {
        // On F and V both lips rise. Limit how far the lower lip overtakes
        // the upper lip on rigs whose eye span exceeds their mouth width.
        f.lipLower -= (approach - clearance) / (0.05 * span);
      }
    }
  }
  for (const [joint, slide] of span ? SLIDES : []) {
    const bone = node(joint);
    if (!bone?.parent) continue;
    const rest = bone.position.clone(),
      toLocal = bone.parent.getWorldQuaternion(new Quaternion()).invert(),
      mirror = joint.endsWith(".R") ? -1 : 1,
      d = new Vector3();
    const values: number[] = [];
    let moved = false;
    for (const f of faces) {
      const [x, y, z] = slide(f);
      if (x || y || z) moved = true;
      d.set(x * mirror, y, z).multiplyScalar(span).applyQuaternion(toLocal).add(rest);
      values.push(d.x, d.y, d.z);
    }
    if (moved) tracks.push(new VectorKeyframeTrack(`${bone.name}.position`, times, values));
  }
  for (const [joint, turns] of RIG) {
    const bone = node(joint);
    if (!bone) continue;
    const rest = bone.quaternion.clone();
    const toLocal = bone.getWorldQuaternion(new Quaternion()).invert();
    const axes = turns.map(([axis, model]) =>
      model ? axis.clone().applyQuaternion(toLocal).normalize() : axis,
    );
    const values: number[] = [],
      q = new Quaternion(),
      turn = new Quaternion();
    let moved = false;
    for (const f of faces) {
      q.copy(rest);
      turns.forEach(([, , angle], i) => {
        const a = angle(f);
        if (a) moved = true;
        q.multiply(turn.setFromAxisAngle(axes[i], a));
      });
      values.push(q.x, q.y, q.z, q.w);
    }
    if (moved) tracks.push(new QuaternionKeyframeTrack(`${bone.name}.quaternion`, times, values));
  }
  return tracks.length ? new AnimationClip(name, seconds, tracks) : undefined;
}

export function faceClips(scene: Object3D) {
  scene.updateWorldMatrix(true, true);
  return {
    face: clip(scene, "face_test", EXPRESSIONS.length, testFace),
    talk: clip(scene, "face_talk", TALK_SECONDS, talkFace),
  };
}
