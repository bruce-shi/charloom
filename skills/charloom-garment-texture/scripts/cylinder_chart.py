"""Project reviewed original-scan triangles to a continuous cylindrical material chart.
PY cylinder_chart.py CONFIG.json --out NEW_DIRECTORY
The chart preserves observed UV colours. Hidden areas remain explicitly empty;
this command does not synthesize missing cloth or author replacement geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'scripts'))
from transfer.scan import Scan,transform_points
from transfer.guide import rasterize


def subdivide(tri,bary,ids):
    a,b,c=tri.transpose(1,0,2);ab,bc,ca=(a+b)/2,(b+c)/2,(c+a)/2
    u,v,w=bary.transpose(1,0,2);uv,vw,wu=(u+v)/2,(v+w)/2,(w+u)/2
    ts=[(a,ab,ca),(ab,b,bc),(ca,bc,c),(ab,bc,ca)]
    bs=[(u,uv,wu),(uv,v,vw),(wu,vw,w),(uv,vw,wu)]
    return np.concatenate([np.stack(x,1) for x in ts]),np.concatenate([np.stack(x,1) for x in bs]),np.tile(ids,4)


def run(path,out):
    cfg=json.loads(path.read_text());source=Path(cfg['source'])
    if not source.is_absolute():source=path.parent/source
    rawhash=hashlib.sha256(source.read_bytes()).hexdigest()
    if rawhash!=cfg['source_sha256']:raise ValueError('Source changed')
    if out.exists():raise FileExistsError(out)
    scan=Scan(source)
    frame=np.asarray(cfg['source_to_working'],float);center=np.asarray(cfg['center_m'],float)
    if frame.shape!=(4,4) or center.shape!=(3,) or not np.isfinite(frame).all() or not np.isfinite(center).all():raise ValueError('Invalid working frame or cylinder centre')
    ids=[]
    for a,b in cfg['triangle_ranges']:
        if not 0<=a<b<=len(scan.faces):raise ValueError('Invalid triangle range')
        ids.extend(range(a,b))
    ids=np.unique(ids)
    if not len(ids):raise ValueError('Select reviewed source triangles')
    vertices=transform_points(scan.vertices,frame);tri=vertices[scan.faces[ids]]
    bary=np.tile(np.eye(3),(len(ids),1,1))
    levels=cfg.get('subdivisions',2)
    if not isinstance(levels,int) or not 0<=levels<=4:raise ValueError('subdivisions must be 0..4')
    for _ in range(levels):tri,bary,ids=subdivide(tri,bary,ids)
    delta=tri-center;theta=np.unwrap(np.arctan2(delta[:,:,0],delta[:,:,2]),axis=1)
    radius=np.linalg.norm(delta[:,:,[0,2]],axis=2)
    height=tri[:,:,1]-cfg.get('height_radius_slope',0)*radius
    lo,hi=cfg['height_range_m'];width,h=cfg['size']
    if hi<=lo or not (16<=width<=4096 and 16<=h<=4096):raise ValueError('Invalid chart extent or size')
    charts=[];bs=[];sources=[]
    for shift in (-2*np.pi,0,2*np.pi):
        pts=np.stack([(theta+shift+np.pi)*width/(2*np.pi),(hi-height)*h/(hi-lo),-radius],axis=2)
        keep=(pts[:,:,0].min(1)<width)&(pts[:,:,0].max(1)>0)
        charts.append(pts[keep]);bs.append(bary[keep]);sources.append(ids[keep])
    chart=np.concatenate(charts);weights=np.concatenate(bs);face_ids=np.concatenate(sources)
    index,blend=rasterize(chart.reshape(-1,3),np.arange(chart.size//3).reshape(-1,3),width,h)
    valid=index>=0
    source_bary=np.einsum('ij,ijk->ik',blend[valid],weights[index[valid]])
    colors=scan.colors(face_ids[index[valid]],source_bary)
    image=np.zeros((h,width,4),dtype=np.uint8);image[valid,:3]=np.rint(colors*255).clip(0,255).astype(np.uint8);image[valid,3]=255
    out.mkdir(parents=True);Image.fromarray(image).save(out/'observed.png');Image.fromarray(valid.astype(np.uint8)*255).save(out/'coverage.png')
    pixels=np.full((h,width),-1,dtype=np.int64);pixels[valid]=face_ids[index[valid]]
    np.savez_compressed(out/'source-pixels.npz',triangle_ids=pixels,valid_barycentric=source_bary)
    (out/'chart.json').write_text(json.dumps({'source_sha256':rawhash,'source_hash':scan.source_hash,'configuration':cfg,'coverage_fraction':float(valid.mean()),'status':'observed_chart_requires_region_review'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.config.resolve(),a.out)
