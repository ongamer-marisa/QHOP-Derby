"""Verify that the executable was built from the current observation/physics contract."""
import hashlib
import json
from pathlib import Path

from .schema import ACTION, OBSERVATION, PHYSICS_FINGERPRINT, VERSION, LIDAR_CONTRACT, RIG_CONTRACT, MODEL_ARCHITECTURE
from .curriculum import REWARD_HASH, COURSE_HASH


def verify(executable):
    executable=Path(executable)
    sidecar=executable.with_suffix(".contract.json")
    contract=json.loads(sidecar.read_text(encoding="utf-8"))
    expected={"schema_version":VERSION,"observation_size":OBSERVATION,
              "action_size":ACTION,"physics_fingerprint":PHYSICS_FINGERPRINT,
              "reward_hash":REWARD_HASH,"course_hash":COURSE_HASH,
              "lidar_contract":LIDAR_CONTRACT,"rig_contract":RIG_CONTRACT,"model_architecture":MODEL_ARCHITECTURE}
    for name,value in expected.items():
        if contract.get(name)!=value:
            raise ValueError(f"Unity build {name} mismatch: {contract.get(name)} != {value}")
    data=executable.with_name(executable.stem+"_Data")
    files={"executable_sha256":executable,
           "assembly_sha256":data/"Managed/Assembly-CSharp.dll",
           "resources_sha256":data/"resources.assets",
           "scene_sha256":data/"level0"}
    verified=[]
    for field,path in files.items():
        digest=hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda:handle.read(1024*1024),b""):
                digest.update(chunk)
        result=digest.hexdigest()
        if contract.get(field)!=result:
            raise ValueError(f"Unity {field} differs from its build contract")
        verified.append(result)
    return hashlib.sha256("".join(verified).encode("ascii")).hexdigest()
