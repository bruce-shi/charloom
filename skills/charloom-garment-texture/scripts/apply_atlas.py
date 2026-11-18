"""Assign a reviewed atlas to explicit primitives that already have continuous UVs.
PY apply_atlas.py CONFIG.json --out NEW.glb
No geometry, skin weights or existing UV coordinates are changed.
"""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'skills/charloom-body-measure/scripts'))
from measurement_common import read,checked_file,digest,write
from roster.mesh import Glb
from roster.mesh_output import texture,finish


def run(path,out):
    cfg=read(path);base=path.parent
    source=checked_file(base,cfg['model']);atlas=checked_file(base,cfg['atlas'])
    if out.exists():raise FileExistsError(out)
    g=Glb(source);im=np.asarray(Image.open(atlas).convert('RGBA'))
    selected=[]
    for spec in cfg['primitives']:
        nodes=[n for n in g.doc['nodes'] if n.get('name')==spec['node'] and 'mesh' in n]
        if len(nodes)!=1:raise ValueError('Select one exact mesh node')
        primitive=g.doc['meshes'][nodes[0]['mesh']]['primitives'][spec['primitive']]
        uv=g.array(primitive['attributes']['TEXCOORD_0'])
        if not np.isfinite(uv).all():raise ValueError('Invalid UVs')
        if not spec.get('uv_evidence'):raise ValueError('Record reviewed continuous UV chart evidence')
        selected.append(primitive)
    if not selected:raise ValueError('Select at least one primitive')
    material=texture(g,im,cfg.get('material_name','Reviewed garment atlas'))
    # Keep each original material's shading settings; replace only the albedo.
    new_texture=g.doc['materials'][material]['pbrMetallicRoughness']['baseColorTexture']
    for primitive in selected:
        original=deepcopy(g.doc['materials'][primitive['material']]) if 'material' in primitive else {}
        original.setdefault('pbrMetallicRoughness',{})['baseColorTexture']=deepcopy(new_texture)
        original['name']=cfg.get('material_name','Reviewed garment atlas')
        primitive['material']=len(g.doc['materials']);g.doc['materials'].append(original)
    out.parent.mkdir(parents=True,exist_ok=True);finish(g,out)
    write(out.with_suffix('.atlas.json'),{'source_sha256':digest(source),'atlas_sha256':digest(atlas),'primitives':cfg['primitives'],'output_sha256':digest(out),'status':'requires_textured_motion_review'})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();run(a.config.resolve(),a.out)
