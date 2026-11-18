/** Independent Three.js skeletal evaluation against Rust samples; no renderer. */
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { AnimationMixer, LoopOnce, Mesh, SkinnedMesh, Vector3 } from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const root = path.resolve(import.meta.dirname, "../..");
const cases = [
  ["out/hand-rig-v2/character.glb", "probe_chain_2"],
  ["out/unknown-rig-v2/character.glb", "probe_chain_4"],
  ["fixtures/local-myth/crossbow-guard.glb", "walk"],
  ["fixtures/local-myth/longfeather-guardian.glb", "move"],
  ["fixtures/local-myth/ford-scout.glb", "move"],
  ["out/myth-starbow-retarget/character.glb", "retargeted"],
  ["out/examples/mixamo-club-v4/bound/character.glb", "club_combo_bound"],
  ["fixtures/local-myth/starbow-adept.glb", "walk"],
  ["out/examples/myth-bow-grip-v1/character.glb", "bow_grip_check"],
  ["out/examples/myth-crossbow-grip-v1/character.glb", "crossbow_grip_check"],
  ...["quaternius","myth-crossbow"].flatMap(target=>
    ["walking","running","waving"].map(motion=>[
      `out/examples/mixamo-comparisons-v1/${target}/waving/character.glb`, `mixamo_${motion}`
    ])
  ),
];
const results = [];
for (const [relative, clipName] of cases) {
  if(process.argv[2] && !relative.includes(process.argv[2]))continue;
  const file = path.join(root, relative);
  if (!fs.existsSync(file)) throw new Error(`Required fixture missing: ${file}`);
  const bytes = fs.readFileSync(file);
  const jsonLength = bytes.readUInt32LE(12);
  const json = JSON.parse(bytes.subarray(20, 20 + jsonLength).toString());
  // No graphics or external resources are needed for the numerical comparison.
  delete json.materials; delete json.textures; delete json.images;
  for (const mesh of json.meshes ?? []) for (const primitive of mesh.primitives) delete primitive.material;
  const encoded = Buffer.from(JSON.stringify(json));
  const padded = Buffer.alloc(Math.ceil(encoded.length / 4) * 4, 32); encoded.copy(padded);
  const binary = bytes.subarray(28 + jsonLength);
  const glb = Buffer.alloc(28 + padded.length + binary.length);
  glb.write("glTF");glb.writeUInt32LE(2,4);glb.writeUInt32LE(glb.length,8);
  glb.writeUInt32LE(padded.length,12);glb.write("JSON",16);padded.copy(glb,20);
  glb.writeUInt32LE(binary.length,20+padded.length);glb.write("BIN\0",24+padded.length);binary.copy(glb,28+padded.length);
  const loaded = await new GLTFLoader().parseAsync(glb.buffer.slice(glb.byteOffset,glb.byteOffset+glb.length), "");
  const clip = loaded.animations.find(a=>a.name===clipName);
  if (!clip) throw new Error(`clip missing: ${relative}/${clipName}`);
  const meshes = new Map<string,Mesh>();
  loaded.scene.traverse(node=>{
    if (!(node instanceof Mesh)) return;
    const association = loaded.parser.associations.get(node);
    let parent = node.parent;
    let nodeId = association?.nodes;
    while(nodeId===undefined && parent) { nodeId=loaded.parser.associations.get(parent)?.nodes;parent=parent.parent; }
    if (nodeId!==undefined) meshes.set(`${nodeId}:${association?.primitives ?? 0}`,node);
  });
  let maxError=0, count=0;let worst:unknown=null;
  const mixer = new AnimationMixer(loaded.scene);
  const action = mixer.clipAction(clip).setLoop(LoopOnce,1);action.clampWhenFinished=true;action.play();
  for(const fraction of [0,.37,.81]) {
    const time=clip.duration*fraction;
    action.reset().play();mixer.setTime(time);loaded.scene.updateMatrixWorld(true);
    loaded.scene.traverse(node=>{if(node instanceof SkinnedMesh)node.skeleton.update();});
    const sampled=JSON.parse(execFileSync(process.env.CHARLOOM_CV_PYTHON??path.join(root,"out/tools/cv/.venv/bin/python"),[path.join(root,"scripts/animate.py"),"sample",file,"--surface","--clip",clipName,"--time",String(time)],{encoding:"utf8",maxBuffer:16*1024*1024}));
    for(const mesh of sampled.meshes) {
      const object=meshes.get(`${mesh.node}:${mesh.primitive}`);
      if(!object)throw new Error(`mesh missing: ${mesh.node}:${mesh.primitive}`);
      if(object.geometry.attributes.position.count!==mesh.vertex_count)throw new Error("vertex count mismatch");
      for(const sample of mesh.samples){
        const point=object.getVertexPosition(sample.index,new Vector3()).applyMatrix4(object.matrixWorld);
        const error=point.distanceTo(new Vector3(...sample.position));
        if(!Number.isFinite(error))throw new Error("non-finite error");
        if(error>maxError){maxError=error;worst={time,node:mesh.node,primitive:mesh.primitive,vertex:sample.index,python:sample.position,three:point.toArray(),object:object.name,skin:object instanceof SkinnedMesh,morph:object.morphTargetInfluences};}count++;
      }
    }
  }
  results.push({file:relative,clip:clipName,sampled_vertices:count,max_world_error_m:maxError,worst});
  mixer.stopAllAction();mixer.uncacheRoot(loaded.scene);
}
const report={status:results.length>0 && results.every(r=>r.max_world_error_m<=1e-4)?"passed":"failed",method:"Python f64 skeletal LBS compared with Three.js GLTFLoader and AnimationMixer; three phases and up to 32 vertices per primitive",cases:results};
fs.mkdirSync(path.join(root,"out/verification"),{recursive:true});
fs.writeFileSync(path.join(root,"out/verification/three-pose-comparison.json"),JSON.stringify(report,null,2)+"\n");
console.log(JSON.stringify(report,null,2));
if(report.status!=="passed")process.exitCode=1;
