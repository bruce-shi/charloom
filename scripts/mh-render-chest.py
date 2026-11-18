"""Sport-bra close-ups of a transfer's 3-skin/body.glb.

Blender -b --factory-startup --python scripts/mh-render-chest.py -- RUN_DIR
python3 scripts/mh-render-chest.py RUN_DIR --cpu

The CPU option uses charloom mh-render --only chest, without compiling.
Both backends frame the sport-bra mesh at rest and render textured and clay
views. Files include the backend name; CPU pictures are not EEVEE renders.
"""
import argparse
import os
import json
from pathlib import Path
import subprocess
import sys


def main():
    args = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--cpu', action='store_true')
    opt = parser.parse_args(args)
    run = opt.run.resolve()
    body = run / '3-skin/body.glb'
    if not body.is_file():
        parser.error(f'missing {body}')
    if opt.cpu:
        python = os.getenv('CHARLOOM_CV_PYTHON',str(Path(__file__).resolve().parents[1]/'out/tools/cv/.venv/bin/python'))
        result = subprocess.run([python, str(Path(__file__).with_name('render-character.py')), str(run), '--only', 'chest'],
                                capture_output=True, text=True, check=True)
        report = json.loads(result.stdout)
        if report['images'] != 10:
            raise RuntimeError('expected ten sport-bra close-ups; body.glb may have no sport bra')
        (run / 'renders/chest-backend.json').write_text(json.dumps({
            'backend': 'python-cpu', 'source': str(body), 'render': report,
        }, indent=2) + '\n')
        print(result.stdout)
        return

    import bpy
    import math
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=str(body))
    scene = bpy.context.scene
    for ob in scene.objects:
        if ob.type == 'ARMATURE':
            ob.data.pose_position = 'REST'
    bpy.context.view_layer.update()
    bra = [ob for ob in scene.objects if ob.type == 'MESH'
           and 'underwear_punkduck_sport_bra' in ob.name]
    if not bra:
        raise RuntimeError('body.glb has no punkduck sport bra to frame')
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for ob in bra:
        ev = ob.evaluated_get(depsgraph)
        mesh = ev.to_mesh()
        points.extend(ev.matrix_world @ v.co for v in mesh.vertices)
        ev.to_mesh_clear()
    lo = Vector(tuple(min(p[i] for p in points) for i in range(3)))
    hi = Vector(tuple(max(p[i] for p in points) for i in range(3)))
    center = (lo + hi) / 2
    size = max(hi - lo) * 1.2
    out = run / 'renders'
    out.mkdir(exist_ok=True)
    engines = {e.identifier for e in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items}
    scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in engines else 'BLENDER_EEVEE'
    scene.render.resolution_x = scene.render.resolution_y = 1024
    scene.render.resolution_percentage = 100
    world = bpy.data.worlds.new('chest-world')
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs[0].default_value = (0.35, 0.37, 0.4, 1)
    world.node_tree.nodes['Background'].inputs[1].default_value = 0.8
    sun = bpy.data.objects.new('chest-sun', bpy.data.lights.new('chest-sun', 'SUN'))
    scene.collection.objects.link(sun)
    sun.data.energy = 3.0
    sun.rotation_euler = (math.radians(50), 0, math.radians(20))
    camera = bpy.data.objects.new('chest-camera', bpy.data.cameras.new('chest-camera'))
    scene.collection.objects.link(camera)
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = size
    scene.camera = camera
    clay = bpy.data.materials.new('chest-clay')
    clay.diffuse_color = (0.55, 0.55, 0.55, 1)
    paths = []
    for mode in ('textured', 'clay'):
        bpy.context.view_layer.material_override = clay if mode == 'clay' else None
        for name, degrees in [('front', 0), ('quarter', 45), ('side', 90),
                              ('back-quarter', 135), ('back', 180)]:
            angle = math.radians(degrees)
            camera.location = center + Vector((math.sin(angle), -math.cos(angle), 0)) * 5
            camera.rotation_euler = (math.radians(90), 0, angle)
            path = out / f'chest-eevee-{mode}-{name}.png'
            scene.render.filepath = str(path)
            bpy.ops.render.render(write_still=True)
            paths.append(path.name)
    (out / 'chest-backend.json').write_text(json.dumps({
        'backend': 'blender-eevee', 'source': str(body),
        'camera_center_blender': list(center), 'ortho_size_m': size,
        'files': paths,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
