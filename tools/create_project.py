"""Copy the locally installed Unity URP template without its generated cache."""
from pathlib import Path
import tarfile
import shutil

root = Path(__file__).resolve().parents[1]
archive = Path(r"C:\Program Files\Unity\Hub\Editor\6000.0.45f1\Editor\Data\Resources\PackageManager\ProjectTemplates\com.unity.template.3d-cross-platform-17.0.11.tgz")
prefix = "package/ProjectData~/"
allowed = ("Assets/", "Packages/", "ProjectSettings/")
with tarfile.open(archive, "r:gz") as source:
    for member in source:
        if not member.isfile() or not member.name.startswith(prefix):
            continue
        relative = member.name[len(prefix):]
        if not relative.startswith(allowed):
            continue
        destination = root / relative
        if destination.exists():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.extractfile(member) as input_file, destination.open("wb") as output_file:
            shutil.copyfileobj(input_file, output_file)
print("Unity project template copied to", root)
