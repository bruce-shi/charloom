"""Fingerprint a GLB's character joint hierarchy, bind pose and skinning data."""

import hashlib
import json
import numpy as np

from transfer.scan import transforms


def signature(g):
    world, parents = transforms(g.doc)
    nodes = g.doc["nodes"]
    joints = sorted({i for skin in g.doc.get("skins", []) for i in skin["joints"]})
    # Include non-skin ancestors so a changed armature transform is observable.
    ancestry = set(joints)
    for i in joints:
        while i in parents:
            i = parents[i]
            ancestry.add(i)
    hierarchy = hashlib.sha256()
    skinning = hashlib.sha256()

    def record(h, metadata):
        h.update(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode() + b"\n"
        )

    def array(h, data):
        values = np.ascontiguousarray(data)
        record(h, (values.shape, values.dtype.str))
        h.update(values.tobytes())

    for i in sorted(ancestry):
        record(hierarchy, (i, nodes[i].get("name"), parents.get(i), i in joints))
        # Allow matrix-to-TRS serialization at glTF float precision.
        bind = world[i].astype("<f4")
        bind[bind == 0] = 0
        array(hierarchy, bind)
    for skin in g.doc.get("skins", []):
        record(skinning, (skin["joints"], skin.get("skeleton")))
        if "inverseBindMatrices" in skin:
            array(skinning, g.array(skin["inverseBindMatrices"]))
    for i, node in enumerate(nodes):
        if "skin" not in node:
            continue
        record(skinning, (i, node["skin"], node["mesh"]))
        for prim in g.doc["meshes"][node["mesh"]]["primitives"]:
            for attr, acc in sorted(prim["attributes"].items()):
                if attr.startswith(("JOINTS_", "WEIGHTS_")):
                    record(
                        skinning,
                        (attr, g.doc["accessors"][acc].get("normalized", False)),
                    )
                    array(skinning, g.array(acc))
    return dict(
        joint_count=len(joints),
        rig_sha256=hierarchy.hexdigest(),
        skin_sha256=skinning.hexdigest(),
    )
