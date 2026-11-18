"""Measure chest/axilla motion in the rib-cage frame and write explicit raise clips.
PY check_pose_regions.py CONFIG.json --out NEW_DIRECTORY
Region vertices are reviewed on the hash-bound body. Each test pose rotates one
named bone around an explicit world axis while keeping the rest of the rig fixed.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'skills/charloom-body-measure/scripts'))
from measurement_common import read,write,load_mesh,digest,unit
from roster.write_face_clips import write_clips


def raised_pose(scene,bone,axis,degrees):
    pose=scene.rest.copy();joint=scene.joint(bone);world=scene.world(pose)
    parent=scene.parents.get(joint)
    basis=world[parent,:3,:3] if parent is not None else np.eye(3)
    basis=basis/np.linalg.norm(basis,axis=0)
    parent_rotation=Rotation.from_matrix(basis)
    delta=Rotation.from_rotvec(unit(axis)*np.deg2rad(degrees))
    pose[joint,3:7]=(parent_rotation.inv()*delta*parent_rotation*Rotation.from_quat(pose[joint,3:7])).as_quat()
    return pose,joint


def local_points(points,frame):
    return np.linalg.solve(frame[:3,:3],(points-frame[:3,3]).T).T


def run(path,out):
    cfg=read(path)
    if out.exists():raise FileExistsError(out)
    if any(k in cfg['mesh'] for k in ('world_from_model','clip')):raise ValueError('Use untransformed rest export')
    surf,part,rest,topology=load_mesh(path.parent,cfg['mesh']);scene=surf.scene
    slot=next(i for i,p in enumerate(surf.parts) if p is part)
    reference=scene.joint(cfg['reference_bone']);world=scene.world(scene.rest)
    resting=local_points(rest,world[reference]);regions={}
    for name,spec in cfg['regions'].items():
        ids=np.unique(np.asarray(spec['vertices'],int))
        if not len(ids) or ids.min()<0 or ids.max()>=len(rest) or not spec.get('evidence'):raise ValueError('Provide reviewed region vertices and evidence')
        edges=part.edges[np.isin(part.edges,ids).all(1)]
        if not len(edges):raise ValueError('Region must contain mesh edges, not isolated markers')
        regions[name]=(ids,edges,np.linalg.norm(resting[edges[:,0]]-resting[edges[:,1]],axis=1))
    if not regions:raise ValueError('Select chest/axilla regions')
    samples=[];clips=[]
    for spec in cfg['poses']:
        if spec['name'] in scene.clips:
            raise ValueError('Diagnostic pose name would replace an existing clip')
        pose,joint=raised_pose(scene,spec['bone'],spec['axis_world'],spec['degrees'])
        samples.append((spec['name'],None,pose,{}))
        clips.append({'name':spec['name'],'tracks':[{'joint':joint,'property':'rotation','times':np.array([0.,1.]),'values':np.array([scene.rest[joint,3:7],pose[joint,3:7]])}],'morph_tracks':[]})
    for clip in cfg.get('clips',[]):
        for time in np.linspace(*scene.interval(clip),cfg.get('samples_per_clip',33)):
            pose,morphs=scene.sample(clip,float(time));samples.append((clip,float(time),pose,morphs))
    results=[]
    for name,time,pose,morphs in samples:
        vertices=surf.evaluate(pose,morphs)[slot];frame=scene.world(pose)[reference]
        points=local_points(vertices,frame);metrics={}
        for region,(ids,edges,baseline) in regions.items():
            displacement=np.linalg.norm(points[ids]-resting[ids],axis=1)*1000
            lengths=np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1)
            growth=(lengths-baseline)*1000;ratio=lengths/np.maximum(baseline,1e-12)
            worst=int(np.argmax(growth))
            metrics[region]={'vertices':len(ids),'max_displacement_mm':float(displacement.max()),'p95_displacement_mm':float(np.percentile(displacement,95)),
               'max_edge_growth_mm':float(growth.max()),'max_edge_ratio':float(ratio.max()),'worst_edge':edges[worst].tolist()}
        results.append({'pose':name,'time':time,'regions':metrics})
    out.mkdir(parents=True)
    scene.glb.write(out/'pose-check.glb')
    if clips:write_clips(out/'pose-check.glb',clips)
    write(out/'pose-regions.json',{'source':cfg['mesh'],'config_sha256':digest(path),'triangle_sha256':topology,
          'reference_bone':cfg['reference_bone'],'regions':cfg['regions'],'samples':results,'status':'requires_region_and_visual_review'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.config.resolve(),a.out)
