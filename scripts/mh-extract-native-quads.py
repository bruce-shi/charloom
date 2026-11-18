"""Read a matching MPFB editable.blend without saving it; export native quads.

Blender background invocation: --python THIS_FILE -- TEMPLATE.blend OUTPUT.npz
Keeps the helper mask, disables generated Delete masks, and preserves source
vertex IDs through the evaluated mesh. Coordinates are glTF Y-up metres.
"""
import sys
from pathlib import Path

import bpy
import numpy as np

template, output = [Path(p).resolve() for p in sys.argv[sys.argv.index('--') + 1:]]
if output.exists():
    raise ValueError('Use a new output file')
bpy.ops.wm.open_mainfile(filepath=str(template), use_scripts=False)
candidates = [o for o in bpy.context.scene.objects
              if o.type == 'MESH' and len(o.data.vertices) == 19158]
if len(candidates) != 1:
    raise ValueError('Expected one unmodified 19158-vertex MakeHuman base')
obj = candidates[0]
for modifier in obj.modifiers:
    if modifier.type == 'MASK' and modifier.name.startswith('Delete.'):
        modifier.show_viewport = False
        modifier.show_render = False
name = 'original_vertex_id'
if name in obj.data.attributes:
    obj.data.attributes.remove(obj.data.attributes[name])
attribute = obj.data.attributes.new(name, 'INT', 'POINT')
attribute.data.foreach_set('value', np.arange(len(obj.data.vertices), dtype=np.int32))
bpy.context.view_layer.update()
evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
mesh = evaluated.to_mesh()
try:
    if not all(len(p.vertices) == 4 for p in mesh.polygons):
        raise ValueError('Evaluated mesh is not entirely quads')
    positions = np.array([evaluated.matrix_world @ v.co for v in mesh.vertices])
    positions = positions[:, [0, 2, 1]]
    positions[:, 2] *= -1
    quads = np.array([list(p.vertices) for p in mesh.polygons], dtype=np.int32)
    original_ids = np.array([a.value for a in mesh.attributes[name].data], dtype=np.int32)
    if len(np.unique(original_ids)) != len(positions) or not np.isfinite(positions).all():
        raise ValueError('Invalid evaluated correspondence')
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, positions=positions, quads=quads, original_ids=original_ids)
    print(f'Exported {len(positions)} vertices and {len(quads)} native quads to {output}')
finally:
    evaluated.to_mesh_clear()
