import fs from 'node:fs';
import {SkinnedMesh,Skeleton,AnimationMixer,Vector3} from '../../viewer/node_modules/three/build/three.module.js';
import {GLTFLoader} from '../../viewer/node_modules/three/examples/jsm/loaders/GLTFLoader.js';
globalThis.self=globalThis;
globalThis.ProgressEvent=class extends Event{constructor(t,i){super(t);Object.assign(this,i)}};
async function read(file){
 const b=fs.readFileSync(file.slice(7)),len=b.readUInt32LE(12),doc=JSON.parse(b.subarray(20,20+len));delete doc.images;delete doc.materials;delete doc.textures;for(const mesh of doc.meshes)for(const p of mesh.primitives)delete p.material;
 doc.buffers[0].uri='data:application/octet-stream;base64,'+b.subarray(28+len).toString('base64');
 return new GLTFLoader().parseAsync(JSON.stringify(doc),'');
}
const rows=[];
for(const row of JSON.parse(fs.readFileSync('out/review/ide/results.json','utf8')).runs){
 const stage=k=>row.stages.find(s=>s.id===k).file;
 const body=await read(stage('body')),dressed=await read(stage('character'));
 let skin;const garments=[];body.scene.traverse(n=>{if(n.isSkinnedMesh)skin??=n.skeleton});
 dressed.scene.traverse(n=>{if(n.isSkinnedMesh&&n.userData.charloom_part==='garment')garments.push(n)});
 if(!garments.length)throw Error(row.alias+' missing garments');
 const boneMap=new Map(skin.bones.map(b=>[b.name,b]));
 // Move all of this character's outfit onto its bare stage, reusing the actual
 // body bones. Keep the garment bind transforms, vertex data and skin weights.
 body.scene.updateMatrixWorld(true);dressed.scene.updateMatrixWorld(true);
 const items=[];
 for(const garment of garments){
  const targetBones=garment.skeleton.bones.map(b=>{const bone=boneMap.get(b.name);if(!bone)throw Error(row.alias+' missing '+b.name);return bone;});
  const item=new SkinnedMesh(garment.geometry,garment.material);
  body.scene.add(item);item.matrix.copy(body.scene.matrixWorld.clone().invert().multiply(garment.matrixWorld));item.matrix.decompose(item.position,item.quaternion,item.scale);item.updateMatrixWorld(true);
  item.bind(new Skeleton(targetBones,garment.skeleton.boneInverses.map(m=>m.clone())),garment.bindMatrix.clone());
  items.push({item,garment});
 }
 const a=new AnimationMixer(body.scene),b=new AnimationMixer(dressed.scene);
 const clip=dressed.animations.find(a=>a.name==='walking');if(!clip)throw Error(row.alias+' missing walk');a.clipAction(clip).play();b.clipAction(clip).play();
 let maxMm=0,total=0;
 for(const phase of [0,.13,.41,.67,.89]){
  a.setTime(phase*clip.duration);b.setTime(phase*clip.duration);body.scene.updateMatrixWorld(true);dressed.scene.updateMatrixWorld(true);
  for(const {item,garment} of items){
   const count=item.geometry.attributes.position.count;for(let i=0;i<count;i+=Math.max(1,Math.floor(count/31))){
    const p=item.getVertexPosition(i,new Vector3()).applyMatrix4(item.matrixWorld),q=garment.getVertexPosition(i,new Vector3()).applyMatrix4(garment.matrixWorld);
    maxMm=Math.max(maxMm,1000*p.distanceTo(q));total++;
   }
  }
 }
 for(const {item} of items)body.scene.remove(item);
 rows.push({character:row.character,items:garments.map(m=>m.name),samples:total,maxDeviationMm:maxMm,pass:maxMm<.1});
 console.log(row.alias,items.length,maxMm.toFixed(4));
}
fs.writeFileSync((process.env.CHARLOOM_HEAD_DIR || 'out/review/kawaii-head-fit')+'/wardrobe-check.json',JSON.stringify(rows,null,2));
if(rows.some(x=>!x.pass))process.exitCode=1;
