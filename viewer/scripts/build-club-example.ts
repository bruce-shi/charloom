/** End-to-end: downloaded motion + independent human + independent rigid club. */
import fs from "node:fs";
import path from "node:path";
import {execFileSync} from "node:child_process";
import {Matrix4,Quaternion,Vector3} from "three";

const root=path.resolve(import.meta.dirname,"../..");
const output=path.resolve(root,process.argv[2]??"out/examples/mixamo-club");
if(fs.existsSync(output))throw new Error("Choose a new example output directory");
fs.mkdirSync(output,{recursive:true});
const cli=(...args:string[])=>JSON.parse(execFileSync(process.env.CHARLOOM_CV_PYTHON??path.join(root,"out/tools/cv/.venv/bin/python"),[path.join(root,"scripts/animate.py"),...args],{cwd:root,encoding:"utf8",maxBuffer:64*1024*1024}));
const source=path.join(root,"fixtures/mixamo/two-hand-club-combo.fbx");
const target=path.join(root,"fixtures/quaternius/superhero-male.glb");
const prop=path.join(root,"fixtures/generated/training-club.glb");
const config={fps:120,name:"mixamo_club_body",align_reference_chains:true,min_target_coverage:.9};
const configPath=path.join(output,"retarget-config.json");fs.writeFileSync(configPath,JSON.stringify(config,null,2)+"\n");
const body=cli("retarget",source,target,"--clip","mixamo.com","--config",configPath,"--out",path.join(output,"body"));
const bodyPath=path.join(output,"body/character.glb");
const inspection=cli("inspect",bodyPath);
const duration=inspection.clips[config.name][1];
const calibration=cli("sample",bodyPath,"--clip",config.name,"--time","0");
const id=(name:string)=>{const i=calibration.names.indexOf(name);if(i<0)throw new Error(`Joint missing: ${name}`);return i;};
const point=(name:string)=>new Vector3(...calibration.world[id(name)].slice(0,3).map((row:number[])=>row[3]));
// Use the proximal/middle finger region as an explicit, reproducible initial
// grip estimate. This is a fixture calibration, not automatic grasp inference.
const palm=(side:string)=>point(`hand_${side}`).lerp(point(`middle_01_${side}`),.75).add(point(`middle_02_${side}`).sub(point(`middle_01_${side}`)).multiplyScalar(.35));
const right=palm("r"),left=palm("l");
const y=left.clone().sub(right).normalize();
const thumbSide=point("thumb_02_r").sub(point("pinky_02_r")).add(point("thumb_02_l").sub(point("pinky_02_l")));
if(y.dot(thumbSide)<0)y.negate();
const auxiliary=Math.abs(y.y)<.8?new Vector3(0,1,0):new Vector3(0,0,1);
const x=auxiliary.cross(y).normalize();const z=x.clone().cross(y).normalize();
const matrix=new Matrix4().makeBasis(x,y,z).setPosition(right.clone().add(left).multiplyScalar(.5).addScaledVector(y,-.1));
const position=new Vector3(),rotation=new Quaternion(),scale=new Vector3();matrix.decompose(position,rotation,scale);
const recipe={schema_version:"0.1",asset:bodyPath,name:"club_combo_bound",duration,fps:120,
  layers:[{clip:config.name}],props:[{name:"training_club",path:prop,transform:{translation:position.toArray(),rotation:rotation.toArray(),scale:scale.toArray()}}],
  attachments:[{node:"training_club",driver:"hand_r",driver_chain_root:"upperarm_r",max_driver_translation_m:.25,start:0,end:duration,reference_clip:config.name,reference_time:0,
    followers:[{joint:"hand_l",chain_root:"upperarm_l",orientation:true}],release:"hold_world"}],
  solver:{iterations:180,tolerance_m:.0003,rotation_tolerance_rad:.004,damping:.01,posture_weight:.0000001}};
const primaryRecipe={...recipe,name:"club_primary_only",attachments:recipe.attachments.map(a=>({...a,driver_chain_root:undefined,followers:[]}))};
const primaryPath=path.join(output,"primary-only.recipe.json");fs.writeFileSync(primaryPath,JSON.stringify(primaryRecipe,null,2)+"\n");
cli("build",primaryPath,"--out",path.join(output,"primary-only"));
recipe.asset=path.join(output,"primary-only/character.glb");
recipe.props=[];
const recipePath=path.join(output,"recipe.json");fs.writeFileSync(recipePath,JSON.stringify(recipe,null,2)+"\n");
const result=cli("build",recipePath,"--out",path.join(output,"bound"));
const report={source_motion:source,human_model:target,weapon_model:prop,duration_s:duration,
  target_deform_coverage:body.coverage,mapped_joints:body.mapped_joints,
  initial_palm_separation_m:right.distanceTo(left),max_grip_residual_m:Math.max(0,...Object.values<number>(result.peak_position_error_m)),
  grip_nonconverged_frames:result.grip_nonconverged_frames,frames:result.frames,
  max_primary_hand_correction_m:result.max_driver_correction_m,
  output:path.join(output,"bound/character.glb"),quality_status:"needs_review",visual_review:"pending"};
fs.writeFileSync(path.join(output,"summary.json"),JSON.stringify(report,null,2)+"\n");
console.log(JSON.stringify(report,null,2));
