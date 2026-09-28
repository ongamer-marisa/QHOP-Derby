"""Make scene assets with runtime bootstrapping when Unity Editor is unlicensed."""
from pathlib import Path
import uuid
import re

root=Path(__file__).resolve().parents[1]
scenes=root/"Assets"/"Scenes"
sample=(scenes/"SampleScene.unity").read_text(encoding="utf-8")
settings=root/"ProjectSettings"/"EditorBuildSettings.asset"
guids={}
for name in ("DerbyStraight","AttributeLab"):
    scene_path=scenes/(name+".unity")
    meta_path=scenes/(name+".unity.meta")
    if scene_path.exists() and meta_path.exists():
        existing=re.search(r"^guid: ([0-9a-f]{32})$",meta_path.read_text(encoding="utf-8"),re.MULTILINE)
        assert existing is not None
        guid=existing.group(1)
        guids[name]=guid
        continue
    guid=uuid.uuid4().hex
    guids[name]=guid
    scene_path.write_text(sample,encoding="utf-8")
    meta_path.write_text(
        "fileFormatVersion: 2\nguid: "+guid+"\nDefaultImporter:\n  externalObjects: {}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n",encoding="utf-8")
text=settings.read_text(encoding="utf-8")
text=re.sub(r"  m_Scenes:\n(?:  - enabled:.*?\n    path:.*?\n    guid:.*?\n)+",
    "  m_Scenes:\n"+"".join(f"  - enabled: 1\n    path: Assets/Scenes/{name}.unity\n    guid: {guids[name]}\n" for name in guids),text)
settings.write_text(text,encoding="utf-8")
print(guids)
