"""Measure a frozen body cohort and fixed reverse samples; updates diagnostic report.

Python diagnostic only; no native pipeline or animation validation.
"""
import sys,json,numpy as np,importlib.util
from pathlib import Path
import igl
ROOT=Path(__file__).resolve().parent.parent;sp=importlib.util.spec_from_file_location('l',ROOT/'scripts/mh-body-fit-lab.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
for arg in sys.argv[1:]:
 out=Path(arg);r=json.loads((out/'report.json').read_text());d=np.load(out/'surface.npz');sv=d['source_positions'];sf=d['source_triangles'].astype(np.int64);st=igl.AABB();st.init(sv,sf);baseg=m.lab.Glb(Path(r['input'])/'3-skin/body.glb');bp=next(z['primitives'][0]for z in baseg.doc['meshes']if z.get('name')=='body');raw=baseg.array(bp['attributes']['POSITION']).astype(float);tri=baseg.array(bp['indices']).reshape(-1,3).astype(np.int64);bodyface=d['free'][d['export_to_control']][tri].any(1);unionv=[raw];uniont=[tri];classes=[bodyface]
 for mesh in baseg.doc['meshes']:
  if mesh.get('name','').startswith(('scan_hand','wrist_bridge')):
   for pr in mesh['primitives']:
    uv=baseg.array(pr['attributes']['POSITION']).astype(float);ut=baseg.array(pr['indices']).reshape(-1,3).astype(np.int64);uniont.append(ut+sum(len(z)for z in unionv));unionv.append(uv);classes.append(np.zeros(len(ut),bool))
 unionv=np.concatenate(unionv);uniont=np.concatenate(uniont);classes=np.concatenate(classes);bt=igl.AABB();bt.init(unionv,uniont);dd,hit,_=bt.squared_distance(unionv,uniont,sv[sf].mean(1));target=sf[classes[hit]];q=sv[target];area=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1);rng=np.random.default_rng(20261113);pick=rng.choice(len(target),12000,p=area/area.sum());uv=rng.random((12000,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1];samples=q[pick,0]+uv[:,:1]*(q[pick,1]-q[pick,0])+uv[:,1:]*(q[pick,2]-q[pick,0]);stats={}
 for label,path in [('before',Path(r['input'])/'3-skin/body.glb'),('after',out/'3-skin/body.glb')]:
  g=m.lab.Glb(path);p=next(z['primitives'][0]for z in g.doc['meshes']if z.get('name')=='body');v=g.array(p['attributes']['POSITION']).astype(float);_,first=np.unique(d['export_to_control'],return_index=True);x=v[first];dd,_,_=st.squared_distance(sv,sf,x[d['cohort']]);tree=igl.AABB();tree.init(v,tri);rd,_,_=tree.squared_distance(v,tri,samples);q=v[tri];ar=np.linalg.norm(np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),axis=1);edge2=sum(((q[:,i]-q[:,j])**2).sum(1)for i,j in [(0,1),(1,2),(2,0)]);quality=2*np.sqrt(3)*ar/np.maximum(edge2,1e-30)
  stats[label]={'forward_fixed_cohort_vertices':len(d['cohort']),'forward_p50_mm':float(np.median(np.sqrt(dd))*1000),'forward_p95_mm':float(np.percentile(np.sqrt(dd),95)*1000),'reverse_fixed_area_samples':len(samples),'reverse_p50_mm':float(np.median(np.sqrt(rd))*1000),'reverse_p95_mm':float(np.percentile(np.sqrt(rd),95)*1000),'active_body_degenerate_triangles':int(((ar<1e-12)&bodyface).sum()),'active_body_triangle_quality_p05':float(np.percentile(quality[bodyface],5)),'active_body_fold_edges':m.lab.fold_edges(x,d['rest'],d['triangles'],d['free'])}
 r['independent_body_metrics']={'selection':'Same baseline-defined vertex cohort and source skin area samples before/after; source triangles assigned to an active baseline body face, with scan hands and bridges included as fixed classification exclusions. No distance-based rejection of reverse samples.','results':stats};(out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(out.name,json.dumps(stats),flush=True)
