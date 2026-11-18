// Face review clips, built in the viewer on MakeHuman's face rig so every
// result can play them, whatever clips its file carries:
//   face: one expression per second;
//   talk: a 12 s loop with phrase stress, the
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
  browBias: number;
  frown: number;
  blink: number;
  /** Mouth shape, -1..1: corners apart (+) or drawn together (-), upper
   * lip raised, lower lip raised, lips pushed forward. */
  lipWide: number;
  lipUpper: number;
  lipLower: number;
  lipForward: number;
  labiodental: number;
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
  browBias: 0,
  frown: 0,
  blink: 0,
  lipWide: 0,
  lipUpper: 0,
  lipLower: 0,
  lipForward: 0,
  labiodental: 0,
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
];

/** Distribute the spoken gesture through the head's actual neck ancestors.
 * A head-only rig receives the whole turn. Model axes keep rolled bones aligned. */
function speakingTurns(head: Object3D): [string, Turn[]][] {
  const shares = new Map([["neck03", .22], ["neck02", .15], ["neck01", .08]]);
  const chain: [string, number][] = [[head.name, .55]];
  for (let parent = head.parent; parent; parent = parent.parent) {
    const share = shares.get(parent.name);
    if (share !== undefined) chain.push([parent.name, share]);
  }
  const total = chain.reduce((sum, [, weight]) => sum + weight, 0);
  return chain.map(([name, weight]) => [name, [
    [Y, true, (f) => f.headLeft * weight / total],
    [X, true, (f) => -f.headUp * weight / total],
    [Z, true, (f) => f.headRoll * weight / total],
  ]]);
}

 
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
/** The demo's phoneme score. Capital vowels carry phrase stress; both the
 * mouth and the surrounding tissue read the same events. Times are seconds. */
const PHRASES: [number, number, string][] = [
  [0.3, 2.2, "siAomeiUafa"],
  [2.9, 5.0, "uomEliasOuema"],
  [5.7, 7.2, "aOuasEma"],
  [8.0, 9.7, "musIfalEoma"],
  [10.3, 11.5, "eiAmolu"],
];

export function phraseAt(t: number) {
  const i = PHRASES.findIndex(([from, to]) => from <= t && t <= to);
  return i < 0 ? "停顿" : `第 ${i + 1} 句`;
}

const SPEECH = PHRASES.flatMap(([from, to, score], phrase) => {
  const lengths = [...score].map(c => /[aeiou]/i.test(c) ? 2 : 1);
  const unit = (to - from - .2) / lengths.reduce<number>((sum, n) => sum + n, 0);
  let start = from + .1;
  return [...score].map((c, i) => {
    const duration = lengths[i] * unit;
    const cue = { sound: c.toUpperCase(), time: start + duration / 2, duration,
      stressed: /[AEIOU]/.test(c), vowel: lengths[i] === 2, phrase };
    start += duration;
    return cue;
  });
});
export const speechContacts = SPEECH.filter(c => c.sound === "M" || c.sound === "F");
export const speechAccents = SPEECH.filter(c => c.stressed);

const MOUTH = (() => {
  const rest = [0, 0, 0, 0, 0, 0];
  const keys = [0, ...PHRASES.flatMap(([from, to]) => [from, to]), TALK_SECONDS]
    .map(t => [t, ...rest]);
  for (const cue of SPEECH) {
    const [jaw, ...lips] = SOUNDS[cue.sound];
    const gain = cue.vowel ? (cue.stressed ? .92 : .66) : 1;
    const pose = [jaw * gain, ...lips, cue.sound === "F" ? 1 : 0];
    if (cue.sound === "M" || cue.sound === "F") {
      keys.push([cue.time - cue.duration * .22, ...pose], [cue.time + cue.duration * .22, ...pose]);
    } else {
      keys.push([cue.time, ...pose]);
    }
  }
  return keys.sort((a, b) => a[0] - b[0]);
})();

// Two short glances during phrase breaks, returning to the listener.
const GLANCES: [number, number, number][] = [
  [PHRASES[0][1], .075, -.025], [PHRASES[1][0], 0, 0],
  [PHRASES[2][1], -.085, .035], [PHRASES[3][0], 0, 0],
];
// One principal accent per phrase carries the head beat. Some phrases
// settle into a nod, others open with a lift. The face keeps the finer accents.
const HEAD_BEATS = PHRASES.flatMap((_, phrase) => {
  const accents = speechAccents.filter(cue => cue.phrase === phrase);
  return phrase % 2 ? accents.slice(-1) : accents.slice(0, 1);
});

const BLINKS = [PHRASES[0][1], PHRASES[2][0] - .27, PHRASES[3][1], PHRASES[4][1]];

function pulse(time: number, peak: number, attack: number, release: number) {
  return smooth(1 + (time - peak) / attack) * smooth(1 - (time - peak) / release);
}

function talkFace(t: number): Face {
  const f = { ...REST };
  const next = MOUTH.findIndex(k => k[0] > t);
  const a = MOUTH[next - 1], b = MOUTH[next];
  const u = smooth((t - a[0]) / (b[0] - a[0]));
  [f.jaw, f.lipWide, f.lipUpper, f.lipLower, f.lipForward, f.labiodental] =
    a.slice(1).map((v, i) => v + (b[i + 1] - v) * u);
  // The brow leads a stressed vowel. The cheek peaks just after it and
  // releases more slowly. Neither expression layer drives the jaw.
  for (const cue of speechAccents) {
    const brow = pulse(t, cue.time - .065, .17, .30);
    const cheek = pulse(t, cue.time + .045, .15, .32);
    const spread = Math.max(0, SOUNDS[cue.sound][1]);
    f.browUp += .30 * brow;
    f.browBias += (cue.phrase % 2 ? -.055 : .055) * brow;
    f.smile += (.12 + .11 * spread) * cheek;
  }
  for (const start of BLINKS)
    f.blink = Math.max(f.blink, smooth((t - start) / .075) * smooth((start + .23 - t) / .12));
  let lastLeft = 0, lastUp = 0;
  for (const [start, left, up] of GLANCES) {
    const blend = smooth((t - start) / .09);
    const follow = smooth((t - start - .09) / .42);
    f.headLeft += (left - lastLeft) * follow;
    f.headRoll -= .26 * (left - lastLeft) * follow;
    f.eyeLeft += (left - lastLeft) * blend;
    f.eyeUp += (up - lastUp) * blend;
    lastLeft = left;
    lastUp = up;
  }
  for (const beat of HEAD_BEATS) {
    const { phrase } = beat;
    const prepare = pulse(t, beat.time - .15, .32, .20);
    const land = pulse(t, beat.time + .08, .24, .46);
    const direction = phrase === 2 ? -1 : 1;
    f.headUp += direction * (.031 * prepare - .074 * land);
    const lean = pulse(t, beat.time + .13, .60, .76);
    f.headRoll += (phrase % 2 ? -.030 : .020) * lean;
    f.headLeft += (phrase % 2 ? .042 : -.035) * lean;
  }
  return f;
}

const TISSUE_SHAPES: [string, (f: Face, talking: boolean) => number][] = [
  ["eyeBlink", (f) => f.blink],
  ["gazeLeft", (f) => f.eyeLeft / .35],
  ["gazeUp", (f) => f.eyeUp / .3],
  ["lipLabiodental", (f) => f.labiodental],
  ["browRaiseL", (f) => f.browUp + f.browBias],
  ["browRaiseR", (f) => f.browUp - f.browBias],
  ["browKnit", (f) => f.frown],
  ["cheekSmile", (f) => f.smile],
  ["cheekSquint", (f) => f.smile * (1 - f.blink)],
  ["eyeWiden", (f, talking) => f.browUp * (talking ? .42 : 1) * (1 - f.blink)],
  ["mouthFrown", (f) => f.frown],
  ["cheekStretch", (f, talking) => Math.max(0, f.lipWide) * (talking ? .45 : 1)],
  ["cheekPucker", (f, talking) => Math.max(0, -f.lipWide) * (talking ? .55 : 1)],
  ["cheekJawOpen", (f) => Math.min(1, f.jaw / .25)],
];

/** `face(t)` sampled on the scene's face joints (the scene in its rest
 * pose) or its calibrated facial morph targets. */
function clip(
  scene: Object3D,
  name: string,
  seconds: number,
  face: (t: number) => Face,
): AnimationClip | undefined {
  // three drops the dot from node names ("eye.L" -> "eyeL").
  const node = (n: string) =>
    scene.getObjectByName(n) ?? scene.getObjectByName(n.replaceAll(".", ""));
  let hasMorphFace = false;
  scene.traverse(object => {
    if ((object as Mesh).morphTargetDictionary?.cheekJawOpen !== undefined) hasMorphFace = true;
  });
  if (!node("jaw") && !hasMorphFace) return undefined;
  const fps = 30,
    frames = Math.round(seconds * fps);
  const times = Array.from({ length: frames + 1 }, (_, i) => i / fps);
  if (name === "face_talk") {
    for (const { time } of speechContacts) {
      if (!times.some(t => Math.fround(t) === Math.fround(time))) times.push(time);
    }
    times.sort((a, b) => a - b);
  }
  const faces = times.map((t) => face(t % seconds));
  const tracks: KeyframeTrack[] = [];
  scene.traverse((object) => {
    const mesh = object as Mesh;
    for (const [shape, value] of TISSUE_SHAPES) {
      const index = mesh.morphTargetDictionary?.[shape];
      if (index === undefined) continue;
      tracks.push(new NumberKeyframeTrack(
        `${mesh.uuid}.morphTargetInfluences[${index}]`,
        times,
        faces.map(f => value(f, name === "face_talk")),
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
  // A source-specific F/V calibration seats the lower lip against the incisal edge.
  const offset: unknown = node("jaw")?.userData.charloomLabiodentalOffsetM;
  const labiodental = Array.isArray(offset) && offset.length === 2 && offset.every(Number.isFinite)
    ? new Vector3(0, offset[0], offset[1]) : undefined;
  for (const [joint, slide] of span ? SLIDES : []) {
    const bone = node(joint);
    if (!bone?.parent) continue;
    const rest = bone.position.clone(),
      toLocal = bone.parent.getWorldQuaternion(new Quaternion()).invert(),
      mirror = joint.endsWith(".R") ? -1 : 1,
      d = new Vector3();
    const lowerLip = joint === "oris01" || joint.startsWith("oris07.");
    const values: number[] = [];
    let moved = false;
    for (const f of faces) {
      const [x, y, z] = slide(f);
      d.set(x * mirror, y, z).multiplyScalar(span);
      if (lowerLip && labiodental) d.addScaledVector(labiodental, f.labiodental);
      if (d.lengthSq() > 0) moved = true;
      d.applyQuaternion(toLocal).add(rest);
      values.push(d.x, d.y, d.z);
    }
    if (moved) tracks.push(new VectorKeyframeTrack(`${bone.name}.position`, times, values));
  }
  const head = node("head") ?? (hasMorphFace ? node("Head") : undefined);
  const rig = name === "face_talk" && head && head.userData.charloomSpeechHeadMotion !== false
    ? [...RIG, ...speakingTurns(head)] : RIG;
  for (const [joint, turns] of rig) {
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
