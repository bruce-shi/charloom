"""Blender static body audit on actual exported float32 triangles.

Invoke with -- DIAGNOSTIC_DIRECTORY; writes geometry-audit.json.
Reports strict noncoplanar crossings touching the selected body only.
"""
import json,numpy as np,sys
from pathlib import Path
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/"scripts"))
from roster.mesh import Glb

def segment_hits_triangle(start, end, triangle):
    direction=end-start
    e1,e2=triangle[1]-triangle[0],triangle[2]-triangle[0]
    h=np.cross(direction,e2);det=np.dot(e1,h)
    if abs(det)<1e-12:return False
    s=start-triangle[0];u=np.dot(s,h)/det
    q=np.cross(s,e1);v=np.dot(direction,q)/det;t=np.dot(e2,q)/det
    return 1e-6<u and 1e-6<v and u+v<1-1e-6 and 1e-6<t<1-1e-6
for arg in sys.argv[sys.argv.index('--')+1:]:
 out=Path(arg);r=json.loads((out/'report.json').read_text());d=np.load(out/'surface.npz');g=Glb(Path(r['input'])/'3-skin/body.glb');p=next(m['primitives'][0]for m in g.doc['meshes']if m.get('name')=='body');orig=g.array(p['attributes']['POSITION']).astype(float);tri=g.array(p['indices']).reshape(-1,3);free=d['free'][d['export_to_control']];_,weld=np.unique(np.round(orig,6),axis=0,return_inverse=True);active=free[tri].any(1)
 def pairs(v):
  tree=BVHTree.FromPolygons(v.tolist(),tri.tolist(),all_triangles=True,epsilon=0);result=[]
  for i,j in tree.overlap(tree):
   if i>=j or not(active[i]or active[j])or np.intersect1d(weld[tri[i]],weld[tri[j]]).size:continue
   a,z=v[tri[i]],v[tri[j]]
   if any(segment_hits_triangle(a[k],a[(k+1)%3],z)or segment_hits_triangle(z[k],z[(k+1)%3],a)for k in range(3)):result.append((int(i),int(j)))
  return result
 ex=Glb(out/'3-skin/body.glb');ep=next(m['primitives'][0]for m in ex.doc['meshes']if m.get('name')=='body');v=ex.array(ep['attributes']['POSITION']).astype(float);assert np.array_equal(tri,ex.array(ep['indices']).reshape(-1,3));np.testing.assert_array_equal(orig[~free],v[~free]);before=pairs(orig);after=pairs(v);new=set(after)-set(before)
 result={'scope':'Actual float32 GLB; strict noncoplanar crossings where at least one triangle touches active body vertices. Baseline seam welding; head/hands/feet excluded when both triangles fixed. No coplanar/tangent or continuous collision proof.','before_pairs':before,'after_pairs':after,'before_count':len(before),'after_count':len(after),'new_count':len(new),'fixed_export_vertices_bitwise_unchanged':True,'topology_unchanged':True}
 (out/'geometry-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(out.name,{k:result[k]for k in ['before_count','after_count','new_count']},flush=True)
