"""Bind reviewed surface points and embedded anatomical centres to export vertices.
PY bind_landmarks.py CONFIG.json --out MODEL.json
CONFIG includes mesh, points, regions and tolerance_mm. Points are reviewed on
the model being bound, not copied from fitted targets to manufacture agreement.
"""
import argparse
from pathlib import Path
import numpy as np
import igl
from scipy.spatial import cKDTree
from measurement_common import read,write,load_mesh,face_selection,digest


def embedded_binding(vertices, ids, point, count, softness):
    if count<4 or softness<=0:raise ValueError('Need >=4 neighbours and positive softness_m')
    distance,local=cKDTree(vertices[ids]).query(point,k=min(count,len(ids)))
    indices=ids[np.atleast_1d(local)]
    dx=vertices[indices]-point;design=np.c_[np.ones(len(indices)),dx/softness]
    weights=1/(np.atleast_1d(distance)**2+softness**2)
    normal=design.T@(weights[:,None]*design)
    if np.linalg.cond(normal)>1e10:raise ValueError('Anatomical neighbourhood cannot support a stable embedded centre')
    coefficients=np.linalg.solve(normal,np.eye(4)[0])@(design.T*weights)
    return {'vertices':indices.tolist(),'coefficients':coefficients.tolist()}


def run(config_path,out):
    cfg=read(config_path)
    if out.exists():raise FileExistsError(out)
    surface,part,vertices,topology=load_mesh(config_path.parent,cfg['mesh'])
    bindings={};report={}
    for name,point in cfg['points'].items():
        xyz=np.asarray(point['xyz_m'],float)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all() or not point.get('evidence'):raise ValueError('Provide reviewed xyz_m and evidence for every point')
        faceids=face_selection(part,surface,point['region'])
        if not len(faceids):raise ValueError('Empty landmark region')
        if point['kind']=='surface':
            distances,selected,closest=igl.point_mesh_squared_distance(xyz[None],vertices,part.faces[faceids])
            error=float(np.sqrt(distances[0])*1000)
            if error>point['max_snap_mm']:raise ValueError(f'{name} is {error:.3f} mm from its selected surface')
            face=int(faceids[selected[0]]);triangle=vertices[part.faces[face]]
            bary=igl.barycentric_coordinates(closest,*(triangle[i][None] for i in range(3)))[0]
            bindings[name]={'triangle':face,'barycentric':bary.tolist()};actual=closest[0]
        elif point['kind']=='embedded':
            ids=np.unique(part.faces[faceids]);binding=embedded_binding(vertices,ids,xyz,point.get('neighbors',90),point.get('softness_m',.008))
            bindings[name]=binding;actual=np.array(binding['coefficients'])@vertices[binding['vertices']]
            error=float(np.linalg.norm(actual-xyz)*1000)
        else:raise ValueError('Point kind must be surface or embedded')
        report[name]={'reviewed_xyz_m':xyz.tolist(),'bound_xyz_m':actual.tolist(),'residual_mm':error,'evidence':point['evidence']}
    record={k:cfg[k] for k in ('mesh','regions','tolerance_mm')}
    record.update(landmark_bindings=bindings,binding_evidence=report,triangle_sha256=topology,config_sha256=digest(config_path))
    record['mesh']=dict(record['mesh'],triangle_sha256=topology)
    write(out,record)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.config.resolve(),a.out)
