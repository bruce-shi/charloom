import fs from 'node:fs';
import path from 'node:path';
import { AnimationMixer,Vector3 } from '../../viewer/node_modules/three/build/three.module.js';
import { GLTFLoader } from '../../viewer/node_modules/three/examples/jsm/loaders/GLTFLoader.js';
import { faceClips } from '../../viewer/src/studio/faceClips.ts';
globalThis.self=globalThis;
globalThis.ProgressEvent=class extends Event{constructor(type,init){super(type);Object.assign(this,init);}};
for (const name of process.argv.slice(2)){
 const dir=path.join((process.env.CHARLOOM_TRANSFER_DIR || "out/transfers"),'final',name), curves=JSON.parse(fs.readFileSync(dir+'/mouth-curves.json','utf8'));
 const buf=fs.readFileSync(dir+'/3-skin/body.glb'),len=buf.readUInt32LE(12),doc=JSON.parse(buf.subarray(20,20+len).toString());const bodyIndex=doc.meshes.findIndex(m=>m.name==='body');delete doc.materials;delete doc.images;delete doc.textures;
 for(const m of doc.meshes)for(const p of m.primitives)delete p.material;
 doc.buffers[0].uri='data:application/octet-stream;base64,'+buf.subarray(28+len).toString('base64');
 const g=await new GLTFLoader().parseAsync(JSON.stringify(doc),'');const body={};
 g.scene.traverse(o=>{const a=g.parser.associations.get(o);if(o.isSkinnedMesh && a?.meshes===bodyIndex)body[a.primitives??0]=o;});
 const head=g.scene.getObjectByName('head');g.scene.updateMatrixWorld(true);const frame=head.matrixWorld.clone();
 const reference=[];
 for(const f of [.3,.5,.7]){
  const pair=[];
  for(const [s,side] of ['upper','lower'].entries()){
   const list=curves[side];const ix=Math.round((curves.upper.length-1)*f);const x=curves.upper[ix][0];const sorted=[...list].sort((a,b)=>Math.abs(a[0]-x)-Math.abs(b[0]-x));const match=sorted[0];const prim=name.startsWith('kawaii-')?s+1:['goblin','brute'].includes(name)?1:0;
   const mesh=body[prim];if(!mesh)throw Error(name+' '+prim+' missing');const pos=mesh.geometry.attributes.position;let ii=0, best=Infinity;
   for(let i=0;i<pos.count;i++){const d=(pos.getX(i)-match[0])**2+(pos.getY(i)-match[1])**2+(pos.getZ(i)-match[2])**2;if(d<best){ii=i;best=d}}
   pair.push({mesh,ii,residual:Math.sqrt(best)*1000})
  }
  reference.push(pair)
 }
 const mixer=new AnimationMixer(g.scene);const clip=faceClips(g.scene).talk;mixer.clipAction(clip).play();
 const actual=({mesh,ii})=>head.worldToLocal(mesh.getVertexPosition(ii,new Vector3()).applyMatrix4(mesh.matrixWorld)).applyMatrix4(frame);
 const frames=[];for(let i=0;i<=360;i++){mixer.setTime(i/30);g.scene.updateMatrixWorld(true);frames.push(reference.map(([u,l])=>(actual(u).y-actual(l).y)*1000))}
 const summary=reference.map((pair,k)=>{const s=frames.map(v=>v[k]);return {neutral:s[0],min:Math.min(...s),minTime:s.indexOf(Math.min(...s))/30,max:Math.max(...s),maxTime:s.indexOf(Math.max(...s))/30,traceToVertexMm:pair.map(p=>p.residual),sameVertex:pair[0].mesh===pair[1].mesh&&pair[0].ii===pair[1].ii}});
 const result={samples:frames.length,landmarks:'nearest export rim vertices, y separation after removing head pose',at: [.3,.5,.7],aperturesMm:summary};
 fs.writeFileSync(dir+'/speech-check.json',JSON.stringify(result,null,2));console.log(name,JSON.stringify(summary[1]));
}
