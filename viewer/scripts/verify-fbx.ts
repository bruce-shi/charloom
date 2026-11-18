/** Independent Mixamo FBX evaluation: Three.js FBXLoader versus ufbx/Rust. */
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { AnimationMixer, LoadingManager, LoopOnce, Matrix4, Quaternion, Texture, Vector3 } from "three";
import { FBXLoader } from "three/addons/loaders/FBXLoader.js";

const root = path.resolve(import.meta.dirname, "../..");
// Rendering is not required. Avoid attempting to load textures from source paths.
(globalThis as any).window = { URL, innerWidth: 100, innerHeight: 100 };
const manager = new LoadingManager();
manager.addHandler(/.*/, {path:"",setPath(value:string){this.path=value;return this;},load(){return new Texture();}} as any);
const cases=[];
for (const name of ["walking","running","waving","two-hand-club-combo"]) {
  const file=path.join(root,"fixtures/mixamo",`${name}.fbx`);
  const bytes=fs.readFileSync(file);
  const scene=new FBXLoader(manager).parse(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.length),"");
  const clip=scene.animations.find(c=>c.name==="mixamo.com");
  if(!clip)throw new Error(`Missing source animation: ${name}`);
  const normalize=(n:string)=>n.replace(/[^a-zA-Z0-9]/g,"").toLowerCase();
  const nodes=new Map<string,any>();
  scene.traverse(node=>{if((node as any).isBone){const key=normalize(node.name);const previous=nodes.get(key);if(previous){
    // FBXLoader creates identity children with the same ID for additional skins.
    // Compare the animated outer node, and reject any genuinely ambiguous node.
    if(node.parent===previous && (node as any).ID===previous.ID && node.position.length()<1e-10 && node.quaternion.angleTo(new Quaternion())<1e-10) return;
    throw new Error(`Ambiguous FBX bone: ${node.name}`);
  }nodes.set(key,node);}});
  const mixer=new AnimationMixer(scene);const action=mixer.clipAction(clip).setLoop(LoopOnce,1);action.clampWhenFinished=true;action.play();
  let maxPosition=0,maxRotation=0,samples=0;let worst:unknown;
  for(const fraction of [0,.07,.19,.37,.51,.71,.87,.98]) {
    const time=Math.round(clip.duration*fraction*30)/30;
    action.reset().play();mixer.setTime(time);scene.updateMatrixWorld(true);
    const actual=JSON.parse(execFileSync(process.env.CHARLOOM_CV_PYTHON??path.join(root,"out/tools/cv/.venv/bin/python"),[path.join(root,"scripts/animate.py"),"sample",file,"--clip","mixamo.com","--time",String(time)],{encoding:"utf8",maxBuffer:32*1024*1024}));
    for(let i=0;i<actual.names.length;i++) {
      const node=nodes.get(normalize(actual.names[i]));if(!node)continue;
      // These Mixamo downloads declare centimeters. Blender normalizes to meters.
      const point=node.getWorldPosition(new Vector3()).multiplyScalar(.01);
      const error=point.distanceTo(new Vector3(...actual.world[i].slice(0,3).map((row:number[])=>row[3])));
      const q=new Quaternion();new Matrix4().fromArray(actual.world[i].flat()).transpose().decompose(new Vector3(),q,new Vector3());
      const rotationError=node.getWorldQuaternion(new Quaternion()).angleTo(q);
      if(error>maxPosition){maxPosition=error;worst={time,joint:actual.names[i],python:actual.world[i].slice(0,3).map((row:number[])=>row[3]),three:point.toArray()};}
      maxRotation=Math.max(maxRotation,rotationError);samples++;
    }
  }
  cases.push({file:path.relative(root,file),duration_s:clip.duration,compared_joint_samples:samples,max_position_error_m:maxPosition,max_rotation_error_rad:maxRotation,worst});
  mixer.stopAllAction();mixer.uncacheRoot(scene);
}
const report={status:cases.every(c=>c.compared_joint_samples>100&&c.max_position_error_m<.001&&c.max_rotation_error_rad<.005)?"passed":"failed",method:"Independent Three.js FBXLoader, 8 source-key times per motion, centimeter-to-meter normalization",cases};
fs.writeFileSync(path.join(root,"out/verification/fbx-comparison.json"),JSON.stringify(report,null,2)+"\n");
console.log(JSON.stringify(report,null,2));
if(report.status!=="passed")process.exitCode=1;
