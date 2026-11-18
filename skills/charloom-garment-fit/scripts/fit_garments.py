"""Inspect or apply measured per-piece garment adjustments in Blender.
blender -b --factory-startup --python-exit-code 1 --python fit_garments.py -- MODEL.glb --inventory OUT.json
blender -b --factory-startup --python-exit-code 1 --python fit_garments.py -- MODEL.glb --config EDITS.json --out NEW_DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import bpy
from mathutils import Matrix
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'blender'))
from charloom import api


def main(argv):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',type=Path);p.add_argument('--inventory',type=Path)
    p.add_argument('--config',type=Path);p.add_argument('--out',type=Path)
    a=p.parse_args(argv)
    if bool(a.config)!=bool(a.out) or not (a.config or a.inventory):p.error('Choose --inventory or --config with --out')
    for output in (a.out,a.inventory):
        if output and output.exists():raise FileExistsError(output)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    rig,body=api.import_character(a.model)
    meshes={o.name:o for o in bpy.context.scene.objects if o.type=='MESH' and o.find_armature()==rig}
    inventory={'file':str(a.model.resolve()),'sha256':hashlib.sha256(a.model.read_bytes()).hexdigest(),
       'coordinate_system':'Blender bone-local metres; bone.matrix_local included',
       'body':body.name if body else None,'meshes':{n:{'vertices':len(o.data.vertices),'polygons':len(o.data.polygons),
       'materials':[s.material.name if s.material else None for s in o.material_slots]} for n,o in meshes.items()},
       'bones':{b.name:[list(row) for row in b.matrix_local] for b in rig.data.bones}}
    if a.inventory:
        a.inventory.parent.mkdir(parents=True,exist_ok=True);a.inventory.write_text(json.dumps(inventory,indent=2))
    if not a.config:return
    cfg=json.loads(a.config.read_text())
    if cfg['source_sha256']!=inventory['sha256']:raise ValueError('Garment configuration belongs to a different export')
    rig.data.pose_position='REST';bpy.context.view_layer.update()
    records=[]
    for op in cfg['pieces']:
        selected=[meshes[n] for n in op['objects']]
        if not selected or body in selected:raise ValueError('Select garment objects, not body')
        mode=op.get('bind','preserve')
        if mode not in ('preserve','body','rigid'):raise ValueError('bind must be preserve, body or rigid')
        if 'matrix' in op:
            if not op.get('measurement_evidence'):raise ValueError('Measured transforms require measurement_evidence')
            transform=Matrix(op['matrix'])
            if len(transform)!=4 or len(transform[0])!=4 or abs(transform.determinant())<1e-10 or any(abs(transform[3][i]-[0,0,0,1][i])>1e-8 for i in range(4)):
                raise ValueError('matrix must be a nonsingular affine 4x4')
            frame=rig.matrix_world @ rig.data.bones[op['bone']].matrix_local
            for obj in selected:
                m=obj.matrix_world.inverted() @ frame @ transform @ frame.inverted() @ obj.matrix_world
                if obj.data.shape_keys:raise ValueError('Review garment morphs before applying a rest-shape transform')
                ids=op.get('vertices',{}).get(obj.name,list(range(len(obj.data.vertices))))
                for i in ids:obj.data.vertices[i].co=m @ obj.data.vertices[i].co
                obj.data.update()
        if mode=='body':api.bind_clothing(body,rig,selected)
        if mode=='rigid':
            bone=op['bone']
            if bone not in rig.data.bones:raise ValueError(f'Unknown rigid mount {bone}')
            for obj in selected:
                for group in list(obj.vertex_groups):
                    if group.name in rig.data.bones:obj.vertex_groups.remove(group)
                group=obj.vertex_groups.get(bone) or obj.vertex_groups.new(name=bone)
                group.add(list(range(len(obj.data.vertices))),1.,'REPLACE')
        records.append({'objects':op['objects'],'bind':mode,'evidence':op.get('measurement_evidence')})
    rig.data.pose_position='POSE';bpy.context.view_layer.update()
    a.out.mkdir(parents=True)
    bpy.ops.wm.save_as_mainfile(filepath=str((a.out/'character.blend').resolve()),compress=True)
    api.export_character(rig,a.out/'character.glb')
    (a.out/'garments.json').write_text(json.dumps({'source':inventory['file'],'source_sha256':inventory['sha256'],
       'operations':records,'status':'requires_section_and_motion_review'},indent=2))

if __name__=='__main__':main(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
