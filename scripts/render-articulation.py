"""Offline independent GLB preview. Input must contain one selected animation."""
import bpy, sys, math, json
from pathlib import Path
from mathutils import Vector
args=sys.argv[sys.argv.index('--')+1:]
source,output=args[:2]; times=[float(x) for x in args[2].split(',')]; focus=args[3] if len(args)>3 else 'body'
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene; scene.render.fps=30
bpy.ops.import_scene.gltf(filepath=str(Path(source).resolve()))
if focus.endswith('-bind'):
 for arm in [o for o in scene.objects if o.type=='ARMATURE']:arm.data.pose_position='REST'
if 'solo' in focus:
 for obj in scene.objects:
  if obj.type=='MESH' and not ('hand' in obj.name.lower() and not any(w in obj.name.lower() for w in ['weapon','club','polearm'])):obj.hide_render=True
for obj in scene.objects:
 for mod in obj.modifiers:
  if mod.type=='ARMATURE':mod.use_deform_preserve_volume=False
scene.render.engine='BLENDER_EEVEE';scene.render.resolution_x=1000;scene.render.resolution_y=1000;scene.render.resolution_percentage=100
scene.world=bpy.data.worlds.new('Studio');scene.world.use_nodes=True;scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.65,.68,.7,1);scene.world.node_tree.nodes['Background'].inputs[1].default_value=.7
for pos,energy,size in [((3,-4,6),1000,5),((-3,-1,3),600,4)]:
 bpy.ops.object.light_add(type='AREA',location=pos);bpy.context.object.data.energy=energy;bpy.context.object.data.shape='DISK';bpy.context.object.data.size=size
bpy.ops.mesh.primitive_plane_add(size=200,location=(0,0,-.002));floor=bpy.context.object;floor.name='Preview ground';material=bpy.data.materials.new('Ground');material.diffuse_color=(.19,.21,.23,1);floor.data.materials.append(material)
bpy.ops.object.camera_add();camera=bpy.context.object;scene.camera=camera;camera.data.type='ORTHO'
root=Path(output);root.mkdir(parents=True,exist_ok=True)
for t in times:
 scene.frame_set(int(t*30),subframe=t*30-int(t*30));bpy.context.view_layer.update()
 target=Vector((0,0,1.0));scale=2.8;offset=Vector((3,-4,2.3))
 for armature in [o for o in scene.objects if o.type=='ARMATURE']:
  hip=next((armature.pose.bones.get(n) for n in ['Hips','pelvis'] if armature.pose.bones.get(n)),None)
  if hip is not None:
   target=armature.matrix_world@hip.head+Vector((0,0,.08));break
 if focus!='body':
  names=['LeftHand','hand_l'] if focus.startswith('left') else ['RightHand','hand_r']
  found=[o for o in scene.objects if o.type=='ARMATURE' and any(n in o.pose.bones for n in names)]
  name=next((n for n in names if found and n in found[0].pose.bones),names[0])
  if not found:raise RuntimeError('Missing hand bone '+name)
  target=found[0].matrix_world@(found[0].data.bones[name].head_local if focus.endswith('-bind') else found[0].pose.bones[name].head);scale=.48;offset=Vector((-.32 if focus.startswith('right') else .32,-.45,.15))
 if '-palm' in focus or '-back' in focus:
  arm=found[0];prefix='Right' if focus.startswith('right') else 'Left'
  def bone_point(digit):
   names=[prefix+'Hand'+digit+'1','HC_'+prefix+'_'+('little' if digit=='Pinky' else digit.lower())+'_1',('pinky' if digit=='Pinky' else digit.lower())+'_01_'+('r' if prefix=='Right' else 'l')]
   bone=next((arm.pose.bones.get(n) for n in names if arm.pose.bones.get(n)),None)
   if bone is None:raise RuntimeError('Missing palm landmark '+digit)
   return arm.matrix_world@(arm.data.bones[bone.name].head_local if focus.endswith('-bind') else bone.head)
  index,little,middle=bone_point('Index'),bone_point('Pinky'),bone_point('Middle')
  normal=(index-target).cross(little-target).normalized()
  target=target.lerp(middle,.7);offset=normal*(.45 if '-palm' in focus else -.45);scale=.32
 camera.location=target+offset;camera.rotation_euler=(target-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.ortho_scale=scale
 scene.render.filepath=str(root/f'{focus}-{t:.3f}.png');bpy.ops.render.render(write_still=True)
print(json.dumps({'source':source,'focus':focus,'times':times,'output':output}))
