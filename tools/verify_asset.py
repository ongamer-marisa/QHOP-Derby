"""Run with Blender --background --factory-startup --disable-autoexec --python."""
from pathlib import Path
import bpy

project=Path(__file__).resolve().parents[1]
bpy.ops.import_scene.fbx(filepath=str(project/"Assets"/"Resources"/"HorseGame.fbx"))
horses=[o for o in bpy.data.objects if o.type=="MESH" and len(o.data.vertices)>100]
assert len(horses)==1, "Expected one game horse mesh"
mesh=horses[0].data
armatures=[o for o in bpy.data.objects if o.type=="ARMATURE"]
assert len(armatures)==1 and len(armatures[0].data.bones)==40
assert sum(len(face.vertices)-2 for face in mesh.polygons)==964
assert len(mesh.materials)==4
required={"BulkThin","BulkFat","FrontLegShort","FrontLegLong",
          "BackLegShort","BackLegLong","BodyShort","BodyLong",
          "FrontMuscle","BackMuscle"}
keys={key.name for key in mesh.shape_keys.key_blocks}
assert required.issubset(keys),required-keys
assert all("Ctrl_" not in bone.name and "IK_" not in bone.name for bone in armatures[0].data.bones)
for name in required:
    key=mesh.shape_keys.key_blocks[name]
    basis=mesh.shape_keys.key_blocks["Basis"]
    assert any((a.co-b.co).length>1e-4 for a,b in zip(key.data,basis.data)),name+" has no shape change"
print("HORSE_ASSET_VERIFIED: 40 bones, 964 triangles, 4 materials, 10 morph endpoints")
