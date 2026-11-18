"""Copy a reviewed body shape/weights into a compatible bare or dressed export.
PY copy_body.py BODY.glb TARGET.glb --out NEW.glb
Keeps target triangles, materials, other meshes and animations; rejects changed
vertex order, skeleton rest transforms or body morph targets. Recheck clothing.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from animation.surface import Surface
from roster.mesh_output import add,finish,normals
from refit_sections import world_to_local


def body(surface):
    matches=[(i,p) for i,p in enumerate(surface.parts) if surface.scene.names[p.node]=='body']
    if len(matches)!=1: raise ValueError('Expected exactly one body primitive')
    return matches[0]


def copy_body(source,target,out):
    if out.exists():raise FileExistsError(out)
    donor,recipient=Surface(source),Surface(target)
    di,dp=body(donor);ri,rp=body(recipient)
    if len(dp.positions)!=len(rp.positions) or dp.deltas or rp.deltas:
        raise ValueError('Body vertex count changed or has morphs; rebuild correspondences')
    faces={tuple(x) for x in dp.faces}
    if any(tuple(x) not in faces for x in rp.faces):
        raise ValueError('Target must use the same body vertex order and a subset of donor faces')
    dn=[donor.scene.names[i] for i in dp.bones];rn=[recipient.scene.names[i] for i in rp.bones]
    if len(set(dn))!=len(dn) or len(set(rn))!=len(rn) or set(dn)-set(rn):
        raise ValueError('Body bone names are ambiguous or incompatible')
    dw=donor.scene.world(donor.scene.rest);rw=recipient.scene.world(recipient.scene.rest)
    mapping=np.array([rn.index(name) for name in dn])
    if not np.allclose(dw[dp.bones],rw[rp.bones[mapping]],atol=1e-5):
        raise ValueError('Skeleton rest transforms differ; do not transplant across fits')
    rp.palette=mapping[dp.palette];rp.weights=dp.weights.copy()
    local=world_to_local(recipient,rp,donor.rest[di])
    g=recipient.scene.glb;attr=rp.prim['attributes']
    for k in list(attr):
        if k.startswith(('JOINTS_','WEIGHTS_')):del attr[k]
    if rp.palette.shape[1]>4:raise ValueError('Expected at most four body influences')
    pad=4-rp.palette.shape[1]
    attr['JOINTS_0']=add(g,np.pad(rp.palette,((0,0),(0,pad))).astype('<u2'),'VEC4',5123)
    attr['WEIGHTS_0']=add(g,np.pad(rp.weights,((0,0),(0,pad))).astype('<f4'),'VEC4')
    attr['POSITION']=add(g,local.astype('<f4'),'VEC3')
    _,ix,weld=np.unique(local.round(6),axis=0,return_index=True,return_inverse=True)
    attr['NORMAL']=add(g,normals(local[ix],weld[rp.faces])[0][weld].astype('<f4'),'VEC3')
    out.parent.mkdir(parents=True,exist_ok=True);finish(g,out)
    check=Surface(out);ci,cp=body(check)
    error=float(np.linalg.norm(check.rest[ci]-donor.rest[di],axis=1).max()*1000)
    if error>.01:raise ValueError(f'Copied body differs by {error} mm')
    return {'max_rest_error_mm':error,'body_triangles':len(cp.faces),'status':'body_copied_requires_garment_motion_review'}

if __name__=='__main__':
    import json
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('body',type=Path);p.add_argument('target',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(copy_body(a.body,a.target,a.out)))
