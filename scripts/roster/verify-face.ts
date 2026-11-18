import fs from 'node:fs';
import path from 'node:path';
import { AnimationMixer,Quaternion,Triangle,Vector3 } from '../../viewer/node_modules/three/build/three.module.js';
import { GLTFLoader } from '../../viewer/node_modules/three/examples/jsm/loaders/GLTFLoader.js';
import {speechAccents,speechContacts} from '../../viewer/src/studio/faceClips.ts';
globalThis.self=globalThis;
globalThis.ProgressEvent=class extends Event{constructor(type,init){super(type);Object.assign(this,init);}};
// A rigid sphere occupies the same volume at every gaze. Measure clearance
// against the animated outer skin, excluding the intentionally cut socket walls.
function paintedEyeClearance(scene, head, side, control) {
  const bone = scene.getObjectByName('eye' + side.toUpperCase());
  const center = bone.getWorldPosition(new Vector3());
  const eye = scene.getObjectByName('eye_' + side);
  const point = new Vector3();
  for (let i = 0; i < eye.geometry.attributes.position.count; i++) {
    eye.getVertexPosition(i, point).applyMatrix4(eye.matrixWorld);
    if (Math.abs(point.distanceTo(center) - control.radius) > control.radius * .001)
      throw Error('Painted eye must rotate on its fitted sphere: ' + side);
  }
  const geometry = head.geometry, position = geometry.attributes.position, uv = geometry.attributes.uv;
  const index = geometry.index.array, triangles = [], vertices = new Map();
  const reach = control.radius * 2.2;
  for (let i = 0; i < index.length; i += 3) {
    const ids = [index[i], index[i + 1], index[i + 2]];
    if (ids.some(id => uv.getX(id) > control.sourceMaxU + 1e-6)) continue;
    const p = ids.map(id => new Vector3().fromBufferAttribute(position, id).applyMatrix4(head.matrixWorld));
    if (['x','y','z'].some(k => p.every(v => v[k] > center[k] + reach) || p.every(v => v[k] < center[k] - reach))) continue;
    for (const id of ids) if (!vertices.has(id)) vertices.set(id, new Vector3());
    triangles.push(new Triangle(...ids.map(id => vertices.get(id))));
  }
  if (!triangles.length) throw Error('Missing exterior skin around eye ' + side);
  const nearest = new Vector3();
  return () => {
    bone.getWorldPosition(center);
    for (const [id, p] of vertices) head.getVertexPosition(id, p).applyMatrix4(head.matrixWorld);
    let distance = Infinity;
    for (const tri of triangles) distance = Math.min(distance, tri.closestPointToPoint(center, nearest).distanceToSquared(center));
    return (Math.sqrt(distance) - control.radius) * 1000;
  };
}
for(const name of process.argv.slice(2)){
 const dir=path.join((process.env.CHARLOOM_HEAD_DIR || "out/characters"),name),controls=JSON.parse(fs.readFileSync(dir+'/controls.json','utf8'));const result={};
 for(const [stage,rel] of [['body','3-skin/body.glb'],['character','4-garments/character.glb']]){
 const bytes=fs.readFileSync(dir+'/'+rel),len=bytes.readUInt32LE(12),doc=JSON.parse(bytes.subarray(20,20+len));delete doc.images;delete doc.materials;delete doc.textures;for(const m of doc.meshes)for(const p of m.primitives)delete p.material;
 doc.buffers[0].uri='data:application/octet-stream;base64,'+bytes.subarray(28+len).toString('base64');const g=await new GLTFLoader().parseAsync(JSON.stringify(doc),'');const scene=g.scene;
 const mesh=n=>scene.getObjectByName(n),head=mesh(controls.faceMesh ?? 'fitted_head'),collar=mesh(controls.collarMesh ?? 'fitted_head'),body=mesh('body'),skull=mesh('head');
 scene.updateMatrixWorld(true);const restFrame=skull.matrixWorld.clone();
 const restHeadTurn=skull.getWorldQuaternion(new Quaternion()), headTurn=new Quaternion();
 const pos=(m,i)=>m.getVertexPosition(i,new Vector3()).applyMatrix4(m.matrixWorld);
 const sample=(m,binding)=>typeof binding==='number' ? pos(m,binding) : binding.vertices.reduce((point,index,k)=>point.addScaledVector(pos(m,index),binding.weights[k]),new Vector3());
 const mixer=new AnimationMixer(scene);const clips={face:g.animations.find(a=>a.name==='face_test'),talk:g.animations.find(a=>a.name==='face_talk')};if(!clips.face||!clips.talk)throw Error('missing exported facial clips');let last;
 function at(clip,t){if(last!==clip){mixer.stopAllAction();mixer.clipAction(clip).reset().play();last=clip;}mixer.setTime(t);scene.updateMatrixWorld(true);}
 function aperture(key){const c=controls[key],k=Math.floor(c.upper.length/2);
 const a=skull.worldToLocal(sample(head,c.upper[k])).applyMatrix4(restFrame),b=skull.worldToLocal(sample(head,c.lower[k])).applyMatrix4(restFrame);return (a.y-b.y)*1000;}
 const qa={};
 if (controls.binding === 'scan_surface') {
   const rootJoint = head.skeleton.bones.findIndex(bone => bone.name === 'root');
   const joints = head.geometry.getAttribute('skinIndex');
   const weights = head.geometry.getAttribute('skinWeight');
   let maxRootWeight = 0;
   for (let vertex = 0; vertex < joints.count; vertex++) {
     for (let slot = 0; slot < joints.itemSize; slot++) {
       if (joints.getComponent(vertex, slot) === rootJoint) {
         maxRootWeight = Math.max(maxRootWeight, weights.getComponent(vertex, slot));
       }
     }
   }
   if (maxRootWeight > 1e-6) throw Error(name+' facial skin follows the rig root: '+maxRootWeight);
 }
 at(clips.face,.25);qa.openMm=['l','r'].map(aperture);at(clips.face,5.25);qa.blinkMm=['l','r'].map(aperture);
 for(const e of qa.blinkMm)if(Math.abs(e)>(name.startsWith("kawaii-") ? .1 : .2))throw Error(name+' blink '+e);
 // Check the embedded clip on reconstructed heads with expression shapes.
 // A separate glTF weights channel per shape would overwrite its neighbours.
 if(head.morphTargetDictionary?.cheekSmile !== undefined){
   const shapes=head.morphTargetDictionary,weights=head.morphTargetInfluences;
   const modes=[['cheekSmile',2.25],['browRaiseL',3.25],['browRaiseR',3.25],['browKnit',4.25]];
   for(const [shape,time] of modes){
     at(clips.face,time);
     if(!(weights[shapes[shape]]>.99))throw Error(name+' missing facial pose '+shape);
   }
   qa.bilabialGapMm = [];
   for (const cue of speechContacts.filter(c => c.sound === 'M')) {
     at(clips.talk,cue.time);
     const gap = aperture('mouth');
     if (Math.abs(gap) > .2) throw Error(name+' unsealed lips at '+cue.time+': '+gap);
     qa.bilabialGapMm.push(gap);
   }
   for (const cue of speechAccents) {
     at(clips.talk,cue.time - .065);
     const earlyBrow = weights[shapes.browRaiseL], earlyCheek = weights[shapes.cheekSmile];
     at(clips.talk,cue.time + .045);
     if (!(earlyBrow > weights[shapes.browRaiseL] && weights[shapes.cheekSmile] > earlyCheek))
       throw Error(name+' missing brow and cheek response at '+cue.time);
   }
 }
 at(clips.face,.25);
 const eyes = Object.entries(controls.paintedEyes ?? {}).map(([side, control]) =>
   [side, paintedEyeClearance(scene,head,side,control)]);
 if (eyes.length) {
   qa.eyeClearanceMm = Object.fromEntries(eyes.map(([side]) => [side, Infinity]));
   for (const clip of [clips.face, clips.talk]) {
     for (let frame = 0; frame <= Math.ceil(clip.duration * 30); frame++) {
       at(clip,frame / 30);
       for (const [side, measure] of eyes) qa.eyeClearanceMm[side] = Math.min(qa.eyeClearanceMm[side],measure());
     }
   }
   for (const [side, clearance] of Object.entries(qa.eyeClearanceMm))
     if (clearance < -.01) throw Error(name+' '+side+' eye intersects skin: '+clearance+' mm');
 }
 const walk=g.animations.find(a=>a.name==='walking');if(!walk)throw Error('missing walk');
 const seamPairs = [];
 if (stage === 'character' && controls.dressedNeck) {
   const neck = mesh('source_neck');
   for (const strip of controls.dressedNeck) {
     const object = mesh(strip.mesh);
     const other = object.isSkinnedMesh ? object : object.children.filter(child => child.isSkinnedMesh)[strip.primitive];
     for (const [i, j] of strip.pairs) seamPairs.push([neck, i, other, j]);
   }
 } else {
   for (const [k, h] of controls.collar.head.entries()) seamPairs.push([collar, h, body, controls.collar.body[k]]);
 }
 let gap=0,lo=Infinity,hi=-Infinity,peakHeadTurn=0;
 for(const clip of [walk,clips.talk]){
  for(let i=0;i<=120;i++){
   at(clip,clip.duration*i/121);
   for(const [a,i,b,j] of seamPairs)gap=Math.max(gap,pos(a,i).distanceTo(pos(b,j))*1000);
   if(clip===clips.talk){
     const v=aperture('mouth');lo=Math.min(lo,v);hi=Math.max(hi,v);
     peakHeadTurn=Math.max(peakHeadTurn,skull.getWorldQuaternion(headTurn).angleTo(restHeadTurn));
   }
  }
 }
 qa.speechHeadTurnDegrees=peakHeadTurn*180/Math.PI;
 if(skull.userData.charloomSpeechHeadMotion !== false && qa.speechHeadTurnDegrees < 1)
   throw Error(name+' stationary head during speech');
 qa.neckMaxMm=gap;qa.dialogueMm=[lo,hi];qa.walkTracks=walk.tracks.length;qa.sha256=new Bun.CryptoHasher('sha256').update(bytes).digest('hex');qa.garments=[];scene.traverse(m=>{if(m.isSkinnedMesh&&m.userData.charloom_part==='garment')qa.garments.push(m.name);});
 if(gap>.05||hi-lo<3)throw Error(name+' '+JSON.stringify(qa));result[stage]=qa;
 }
 fs.writeFileSync(dir+'/animated-checks.json',JSON.stringify(result,null,2));console.log(name,JSON.stringify(result.character));
}
