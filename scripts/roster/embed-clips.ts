// Evaluate the same Three.js face controls the Gallery uses. Python writes the
// animation accessors and preserves the rest of the GLB.
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { GLTFLoader } from "../../viewer/node_modules/three/examples/jsm/loaders/GLTFLoader.js";
import { faceClips } from "../../viewer/src/studio/faceClips.ts";

globalThis.self = globalThis;
globalThis.ProgressEvent = class extends Event {
  constructor(type, init) {
    super(type);
    Object.assign(this, init);
  }
};

const folder = process.env.CHARLOOM_HEAD_DIR || "out/characters";
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "charloom-face-clips-"));

async function embed(model) {
  const bytes = fs.readFileSync(model);
  const end = 20 + bytes.readUInt32LE(12);
  const doc = JSON.parse(bytes.subarray(20, end));
  // Headless evaluation needs geometry, bones and morph targets only.
  delete doc.materials;
  delete doc.images;
  delete doc.textures;
  for (const mesh of doc.meshes) for (const primitive of mesh.primitives) delete primitive.material;
  const binary = bytes.subarray(end + 8, end + 8 + doc.buffers[0].byteLength);
  doc.buffers[0].uri = "data:application/octet-stream;base64," + binary.toString("base64");
  const { scene, parser } = await new GLTFLoader().parseAsync(JSON.stringify(doc), "");
  const { face, talk } = faceClips(scene);
  if (!face || !talk) throw Error("Missing face rig in " + model);
  const clips = [face, talk].map(clip => {
    const tracks = [];
    const morphsByNode = new Map();
    for (const track of clip.tracks) {
      const [objectName, property] = track.name.split(".");
      const object = scene.getObjectByName(objectName) ?? scene.getObjectByProperty("uuid", objectName);
      const node = parser.associations.get(object)?.nodes;
      if (node === undefined) throw Error("Cannot bind " + track.name);
      const times = Array.from(track.times);
      if (property.startsWith("morphTargetInfluences[")) {
        const index = Number(property.slice(property.indexOf("[") + 1, -1));
        // glTF animates all weights on a node in one channel. The viewer
        // builds individual shape tracks on a shared sample clock.
        let channel = morphsByNode.get(node);
        if (!channel) {
          channel = { node, interpolation: "linear", times,
            values: times.map(() => [...object.morphTargetInfluences]) };
          morphsByNode.set(node, channel);
        }
        if (channel.times.length !== times.length || times.some((t, i) => t !== channel.times[i]))
          throw Error("Different sample times for morphs on " + object.name);
        track.values.forEach((value, frame) => { channel.values[frame][index] = value; });
      } else {
        const size = track.getValueSize();
        if (property !== "quaternion" && property !== "position") throw Error("Unknown face channel " + property);
        const values = times.map((_, i) => {
          const data = Array.from(track.values.slice(i * size, (i + 1) * size));
          if (size === 3) data.push(0);
          return data;
        });
        tracks.push({ joint: node, property: property === "quaternion" ? "rotation" : "translation",
          interpolation: "linear", times, values });
      }
    }
    return { name: clip.name, tracks, morph_tracks: [...morphsByNode.values()] };
  });
  const file = path.join(temp, "clips.json");
  fs.writeFileSync(file, JSON.stringify(clips));
  execFileSync(process.env.CHARLOOM_CV_PYTHON || "python3",
    [path.join(import.meta.dir, "write_face_clips.py"), model, file], { stdio: "inherit" });
}
try {
  const args = process.argv.slice(2).filter(arg => arg !== "--python-writer");
  if (args[0] === "--models") {
    for (const file of args.slice(1)) await embed(file);
  } else {
    for (const name of args)
      for (const file of ["3-skin/body.glb", "4-garments/character.glb"])
        await embed(path.join(folder, name, file));
  }
} finally {
  fs.rmSync(temp, { recursive: true });
}
