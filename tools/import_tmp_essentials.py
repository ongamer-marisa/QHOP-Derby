"""Install the Unity-bundled TextMesh Pro essentials with their original GUIDs."""
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parents[1]
package = next((root / "Library" / "PackageCache").glob(
    "com.unity.ugui@*/Package Resources/TMP Essential Resources.unitypackage"
))

with tarfile.open(package, "r:gz") as archive:
    files = {}
    for member in archive:
        if not member.isfile():
            continue
        parts = Path(member.name).parts
        if len(parts) != 2 or parts[1] not in ("pathname", "asset", "asset.meta"):
            continue
        files.setdefault(parts[0], {})[parts[1]] = archive.extractfile(member).read()

    installed = 0
    for entry in files.values():
        if "pathname" not in entry:
            continue
        path = entry["pathname"].decode("utf-8").strip().replace("\\", "/")
        if not path.startswith("Assets/TextMesh Pro") or ".." in Path(path).parts:
            raise ValueError(f"Unexpected package path: {path}")
        destination = root / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if "asset" in entry and not destination.exists():
            destination.write_bytes(entry["asset"])
            installed += 1
        if "asset.meta" in entry:
            meta = Path(str(destination) + ".meta")
            if not meta.exists():
                meta.write_bytes(entry["asset.meta"])

print(f"Imported {installed} TextMesh Pro essential assets from {package}")
