"""Run from Blender 2.83 with --factory-startup --disable-autoexec --python.

The original .blend and license stay untouched. This script exports a game rig
with Blender-only control bones, constraints and drivers removed.
"""
import bpy
from pathlib import Path

project = Path(__file__).resolve().parents[1]
target = project / "Assets" / "Art" / "Horse" / "Source"
source = target / "Horse.blend"
if not source.is_file():
    raise FileNotFoundError("Bundled horse source is missing: " + str(source))

bpy.ops.wm.open_mainfile(filepath=str(source), load_ui=False)
if bpy.context.object and bpy.context.object.mode != 'OBJECT':
    bpy.ops.object.mode_set(mode='OBJECT')
armature = bpy.data.objects["Armature"]
mesh = bpy.data.objects["Cylinder"]

for obj in list(bpy.data.objects):
    if obj not in (armature, mesh):
        bpy.data.objects.remove(obj, do_unlink=True)

if armature.animation_data:
    for driver in list(armature.animation_data.drivers):
        armature.driver_remove(driver.data_path, driver.array_index)
for pose_bone in armature.pose.bones:
    for constraint in list(pose_bone.constraints):
        pose_bone.constraints.remove(constraint)

bpy.ops.object.select_all(action='DESELECT')
mesh.select_set(True)
bpy.context.view_layer.objects.active = mesh
for modifier in list(mesh.modifiers):
    if modifier.type == 'MIRROR':
        bpy.ops.object.modifier_apply(modifier=modifier.name)

# Keep named morph endpoints in the FBX for visual QA and later art iteration.
# Runtime dimensions are applied by the same config that constructs the joints.
mesh.shape_key_add(name="Basis", from_mix=False)
for name, axis, factor in (
    ("BulkThin", "bulk", -.15), ("BulkFat", "bulk", .15),
    ("FrontLegShort", "front", -.15), ("FrontLegLong", "front", .15),
    ("BackLegShort", "back", -.15), ("BackLegLong", "back", .15),
    ("BodyShort", "body", -.15), ("BodyLong", "body", .15),
    ("FrontMuscle", "front_muscle", .10), ("BackMuscle", "back_muscle", .10)):
    key=mesh.shape_key_add(name=name, from_mix=False)
    for i,vertex in enumerate(mesh.data.vertices):
        x,y,z=vertex.co
        if axis=="bulk":
            torso=max(0.0,min(1.0,(z-2.0)/1.2))
            key.data[i].co.x=x*(1+factor*torso)
        elif axis in ("front","back"):
            region=y < -.6 if axis=="front" else y > .6
            if region and z < 3.6:
                key.data[i].co.z=3.6+(z-3.6)*(1+factor)
        elif axis=="body":
            if y < -3.5: shift=-2.1*factor
            elif y > .5: shift=1.9*factor
            else: shift=(y+1.4)*factor
            key.data[i].co.y=y+shift
        else:
            region=y < -.6 if axis=="front_muscle" else y > .6
            if region and z < 3.6:
                center=.6 if x>=0 else -.6
                key.data[i].co.x=center+(x-center)*(1+factor)

# Do not export non-deforming Blender control bones. They cannot run in Unity.
bpy.ops.object.select_all(action='DESELECT')
armature.select_set(True)
bpy.context.view_layer.objects.active = armature
bpy.ops.object.mode_set(mode='EDIT')
for bone in list(armature.data.edit_bones):
    if not bone.use_deform:
        armature.data.edit_bones.remove(bone)
bpy.ops.object.mode_set(mode='OBJECT')

# The source's mirrored mesh is intentionally low-poly. Shape keys are authored
# in game space by the runtime morph processor so physics pivots follow exactly.
bpy.ops.object.select_all(action='DESELECT')
armature.select_set(True)
mesh.select_set(True)
bpy.context.view_layer.objects.active = armature
output = project / "Assets" / "Resources" / "HorseGame.fbx"
output.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.export_scene.fbx(
    filepath=str(output), use_selection=True, object_types={'ARMATURE', 'MESH'},
    apply_scale_options='FBX_SCALE_ALL', global_scale=0.36,
    axis_forward='-Y', axis_up='Z', add_leaf_bones=False,
    use_armature_deform_only=True, bake_anim=False,
    mesh_smooth_type='OFF', path_mode='AUTO')
print("HORSE_EXPORT:", output, output.stat().st_size)
