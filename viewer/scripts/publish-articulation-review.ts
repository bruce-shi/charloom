/** Adopt only the explicitly verified review manifest into the local viewer. */
import fs from'node:fs';import path from'node:path';import crypto from'node:crypto';
const root=path.resolve(import.meta.dirname,'../..'),folder=path.resolve(process.argv[2]),manifest=JSON.parse(fs.readFileSync(path.join(folder,'regression.json'),'utf8'));
const hash=(file:string)=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex'),url=(file:string)=>'/files/'+path.relative(root,file).split(path.sep).join('/');
const expected=['crossbow','starbow'].flatMap(model=>['walk','run','strike','strike-club','strike-polearm','carry-walk','carry-run'].map(m=>`${model}-${m}`));
if(manifest.cases.length!==14||new Set(manifest.cases.map((c:any)=>c.id)).size!==14)throw new Error('Expected exactly 14 final cases');
for(const id of expected){const c=manifest.cases.find((c:any)=>c.id===id);if(!c||c.independent_validation?.status!=='passed'||!c.visual_evidence?.length)throw new Error('Missing independent or visual review '+id);if(hash(c.file)!==c.sha256)throw new Error('Artifact changed '+id);}
const catalogPath=path.join(root,'viewer/public/catalog.json'),matrixPath=path.join(root,'out/examples/motion-matrix-current.json');
if(fs.existsSync(path.join(folder,'preview-adoption.json')))throw new Error('Already adopted; inspect existing provenance');
fs.copyFileSync(catalogPath,path.join(folder,'catalog-before.json'));fs.copyFileSync(matrixPath,path.join(folder,'matrix-before.json'));
const oldCatalog=JSON.parse(fs.readFileSync(catalogPath,'utf8')),catalog=oldCatalog.filter((e:any)=>!e.id.startsWith('audit-')&&!e.id.startsWith('articulation-'));
for(const model of ['crossbow','starbow'])for(const kind of ['body','carry','strike-club','strike-polearm']){
 const entry=catalog.find((e:any)=>e.id===`matrix-${model}-${kind}`);if(!entry)throw new Error('Missing catalog group');
 const ids=kind==='body'?['walk','run','strike']:kind==='carry'?['carry-walk','carry-run']:[kind];
 const cases=ids.map(m=>manifest.cases.find((c:any)=>c.id===`${model}-${m}`));
 entry.variants=cases.map((c:any)=>({name:c.clip,file:url(c.file),report:url(c.report),duration:c.duration_s,groundHeight:JSON.parse(fs.readFileSync(c.constraint_recipe??c.recipe,"utf8")).surface_floor?.height_m}));entry.file=entry.variants[0].file;entry.report=url(path.join(folder,'regression.json'));entry.clips=cases.map((c:any)=>c.clip);entry.defaultClip=entry.clips[0];entry.note='已更新身体避让、关节范围、手指形状与护腕刚性。切换动作并查看手部近景，等待本轮统一验收。';
}
const matrix=JSON.parse(fs.readFileSync(matrixPath,'utf8'));
for(let i=0;i<matrix.cases.length;i++){const previous=matrix.cases[i],c=manifest.cases.find((r:any)=>r.id===previous.id);if(!c)continue;matrix.cases[i]={...previous,file:c.file,viewer_file:c.file,clip:c.clip,report:c.report,frames:c.frames??previous.frames,independent_validation:c.independent_validation,visual_review:'agent_sampled_review_pending_user',visual_evidence:c.visual_evidence,source_sha256:c.source_sha256,artifact_sha256:c.sha256,review_manifest:path.join(folder,'regression.json')};}
const untouched=matrix.cases.filter((c:any)=>!expected.includes(c.id));const before=JSON.parse(fs.readFileSync(path.join(folder,'matrix-before.json'),'utf8'));for(const c of untouched)if(JSON.stringify(c)!==JSON.stringify(before.cases.find((b:any)=>b.id===c.id)))throw new Error('Unrelated case changed');
matrix.user_visual_review='pending_unified_review';matrix.runtime_adoption=false;matrix.review_notes='14 crossbow/starbow cases updated for seven reported issues; the other ten case records are unchanged. Sampled agent views and numerical checks are not user approval.';matrix.validation_scope={...matrix.validation_scope,articulation_review:path.join(folder,'regression.json')};
fs.writeFileSync(catalogPath,JSON.stringify(catalog,null,2)+'\n');fs.writeFileSync(matrixPath,JSON.stringify(matrix,null,2)+'\n');
const adopted=[];for(const c of manifest.cases){const response=await fetch('http://127.0.0.1:5194'+url(c.file));if(!response.ok)throw new Error('HTTP failed '+c.id);const actual=crypto.createHash('sha256').update(Buffer.from(await response.arrayBuffer())).digest('hex');if(actual!==c.sha256)throw new Error('Served artifact mismatch '+c.id);adopted.push({id:c.id,url:url(c.file),sha256:actual});}
const result={status:'local_preview_updated',cases:adopted,unchanged_cases:untouched.map((c:any)=>c.id),user_approval:'pending',game_runtime_adoption:false};fs.writeFileSync(path.join(folder,'preview-adoption.json'),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify(result));
