/** Validate the exported artifact's grip constraints using only Three.js. */
import fs from "node:fs";
import path from "node:path";
import {AnimationMixer,Bone,LoopOnce,Matrix4,Quaternion,Vector3} from "three";
import {GLTFLoader} from "three/addons/loaders/GLTFLoader.js";
const root=path.resolve(import.meta.dirname,"../..");
const folder=path.resolve(root,process.argv[2]??"out/examples/mixamo-club");
const file=path.join(folder,"bound/character.glb");
const report=JSON.parse(fs.readFileSync(path.join(folder,"bound/report.json"),"utf8"));
const binding=report.attachment_calibration[0];
const original=fs.readFileSync(file),jsonLength=original.readUInt32LE(12);
const data=JSON.parse(original.subarray(20,20+jsonLength).toString());
delete data.materials;delete data.textures;delete data.images;
for(const mesh of data.meshes??[])for(const primitive of mesh.primitives)delete primitive.material;
const encoded=Buffer.from(JSON.stringify(data));const padded=Buffer.alloc(Math.ceil(encoded.length/4)*4,32);encoded.copy(padded);
const binary=original.subarray(28+jsonLength),bytes=Buffer.alloc(28+padded.length+binary.length);
bytes.write("glTF");bytes.writeUInt32LE(2,4);bytes.writeUInt32LE(bytes.length,8);bytes.writeUInt32LE(padded.length,12);bytes.write("JSON",16);padded.copy(bytes,20);bytes.writeUInt32LE(binary.length,20+padded.length);bytes.write("BIN\0",24+padded.length);binary.copy(bytes,28+padded.length);
const loaded=await new GLTFLoader().parseAsync(bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength),"");
const toMatrix=(t:any)=>new Matrix4().compose(new Vector3(...t.translation),new Quaternion(...t.rotation),new Vector3(...t.scale));
const offset=toMatrix(binding.driver_offset),secondaryOffset=toMatrix(binding.followers[0].offset);
const primary=loaded.scene.getObjectByName(binding.driver)!,secondary=loaded.scene.getObjectByName("hand_l")!,prop=loaded.scene.getObjectByName(binding.node)!;
if(!primary||!secondary||!prop)throw new Error("Missing exported grip node");
loaded.scene.updateMatrixWorld(true);
const boneSegments:{bone:Bone;length:number}[]=[];
loaded.scene.traverse(node=>{
  if(node instanceof Bone && node.parent instanceof Bone && node.name!=="pelvis") {
    const length=node.getWorldPosition(new Vector3()).distanceTo(node.parent.getWorldPosition(new Vector3()));
    if(length>1e-8)boneSegments.push({bone:node,length});
  }
});
const mixer=new AnimationMixer(loaded.scene),cases=[];
for(const name of ["club_primary_only","club_combo_bound"]) {
  mixer.stopAllAction();
  const clip=loaded.animations.find(c=>c.name===name)!;if(!clip)throw new Error(`Missing clip ${name}`);
  const action=mixer.clipAction(clip).setLoop(LoopOnce,1);action.clampWhenFinished=true;action.reset().play();
  const times=Array.from(clip.tracks[0].times);const samples=times.flatMap((time,i)=>i===times.length-1?[{time,key:true}]:[{time,key:true},{time:(time+times[i+1])/2,key:false}]);
  let primaryMax=0,secondaryKeyMax=0,secondaryBetweenMax=0,rotationKeyMax=0,rotationBetweenMax=0,boneLengthDrift=0;let worst:unknown;
  for(const {time,key} of samples) {
    action.reset().play();mixer.setTime(time);loaded.scene.updateMatrixWorld(true);
    const expected=primary.matrixWorld.clone().multiply(offset),leftExpected=prop.matrixWorld.clone().multiply(secondaryOffset);
    const primaryError=new Vector3().setFromMatrixPosition(expected).distanceTo(new Vector3().setFromMatrixPosition(prop.matrixWorld));
    const secondaryError=new Vector3().setFromMatrixPosition(leftExpected).distanceTo(new Vector3().setFromMatrixPosition(secondary.matrixWorld));
    const expectedQ=new Quaternion();leftExpected.decompose(new Vector3(),expectedQ,new Vector3());
    const angle=expectedQ.angleTo(secondary.getWorldQuaternion(new Quaternion()));
    for(const {bone,length} of boneSegments) {
      const current=bone.getWorldPosition(new Vector3()).distanceTo(bone.parent!.getWorldPosition(new Vector3()));
      boneLengthDrift=Math.max(boneLengthDrift,Math.abs(current-length)/length);
    }
    primaryMax=Math.max(primaryMax,primaryError);
    if(key){if(secondaryError>secondaryKeyMax)worst={time,position_error_m:secondaryError};secondaryKeyMax=Math.max(secondaryKeyMax,secondaryError);rotationKeyMax=Math.max(rotationKeyMax,angle);}
    else {secondaryBetweenMax=Math.max(secondaryBetweenMax,secondaryError);rotationBetweenMax=Math.max(rotationBetweenMax,angle);}
  }
  cases.push({clip:name,baked_frames:times.length,independent_samples:samples.length,max_primary_attachment_error_m:primaryMax,max_secondary_key_error_m:secondaryKeyMax,max_secondary_between_key_error_m:secondaryBetweenMax,max_secondary_key_rotation_error_rad:rotationKeyMax,max_secondary_between_key_rotation_error_rad:rotationBetweenMax,checked_bone_segments:boneSegments.length,max_relative_bone_length_drift:boneLengthDrift,worst});
}
const bound=cases[1];
const result={status:bound.max_secondary_key_error_m<=.0011&&bound.max_secondary_between_key_error_m<=.005&&bound.max_secondary_key_rotation_error_rad<=.011&&bound.max_primary_attachment_error_m<=.005&&bound.max_relative_bone_length_drift<=1e-4?"passed":"failed",
  method:"Independent Three.js GLTFLoader/AnimationMixer over every exported key and interval midpoint; positions in meters; compare primary-only baseline with both-hand solve",cases,
  visual_review:"pending",limits:["Grip frames are calibrated geometric targets, not a mesh collision or finger pressure test.","This is kinematic animation; impact physics and automatic grasp inference are not evaluated."]};
fs.writeFileSync(path.join(folder,"independent-grip-validation.json"),JSON.stringify(result,null,2)+"\n");console.log(JSON.stringify(result,null,2));
if(result.status!=="passed")process.exitCode=1;
