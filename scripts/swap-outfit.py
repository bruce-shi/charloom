"""Fit an assembled Charloom outfit to another character on the same MakeHuman scaffold."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from transfer.cloth import rig_for
from roster.mesh import Glb
from roster.mesh_output import add, finish, idx, material, normals


def body(g):
    node = next(n for n in g.doc["nodes"] if n.get("name") == "body" and "mesh" in n)
    return node, g.doc["meshes"][node["mesh"]]["primitives"][0]


def geometry(g, prim):
    return (
        g.array(prim["attributes"]["POSITION"]).astype(float),
        g.array(prim["indices"]).reshape(-1, 3).astype(np.int64),
    )


class BodyMap:
    def __init__(self, source, target, faces):
        # Only surface anchors participate. UV seam copies share a position.
        used = np.unique(faces)
        self.source, ids = np.unique(source[used], axis=0, return_index=True)
        self.target = target[used[ids]]
        self.tree = cKDTree(self.source)
        self.height = np.ptp(source[:, 1])

    def map(self, points):
        """Fit a regularized local affine displacement from corresponding body anchors."""
        distances, ids = self.tree.query(points, k=min(48, len(self.source)))
        a, b = self.source[ids], self.target[ids]
        radius = np.maximum(distances[:, -1:], self.height * 0.006)
        weights = np.exp(-3.0 * (distances / radius) ** 2)
        weights /= weights.sum(axis=1, keepdims=True)
        mean = np.einsum("ij,ijk->ik", weights, a)
        shift = np.einsum("ij,ijk->ik", weights, b - a)
        offsets = a - mean[:, None, :]
        weighted = (weights[:, :, None] * offsets).transpose(0, 2, 1)
        cov = weighted @ offsets
        # Unobserved directions near a planar or narrow limb keep their offsets.
        cov += np.eye(3)[None] * (0.04 * radius[:, :, None] ** 2)
        gradient = np.linalg.solve(cov, weighted @ (b - a - shift[:, None, :]))
        return points + shift + np.einsum("ij,ijk->ik", points - mean, gradient)


def bind_to_regions(points, anchors, joints, weights, names, prefixes):
    """Interpolate the recipient's body weights within the reviewed attachment region."""
    names = np.asarray(names)
    allowed = np.flatnonzero([name.startswith(tuple(prefixes)) for name in names])
    if not len(allowed):
        raise ValueError(f"No recipient joints matched {prefixes}")
    field = np.zeros((len(anchors), len(names)), dtype=np.float32)
    np.add.at(field, (np.arange(len(anchors))[:, None], joints), weights)
    field = field[:, allowed]
    total = field.sum(axis=1)
    selected = total > 0.5
    if not selected.any():
        raise ValueError(f"No body anchors for {prefixes}")
    anchors = anchors[selected]
    field = field[selected] / total[selected, None]
    tree = cKDTree(anchors)
    palette, influence = [], []
    for chunk in np.array_split(points, len(points) // 4096 + 1):
        dist, near = tree.query(chunk, k=min(32, len(anchors)))
        dist, near = dist.reshape(len(chunk), -1), near.reshape(len(chunk), -1)
        radius = np.maximum(dist[:, -1:], 0.012 * np.ptp(anchors[:, 1]))
        coeff = np.exp(-2.5 * (dist / np.maximum(radius, 1e-9)) ** 2)
        mixed = np.einsum("ij,ijk->ik", coeff, field[near])
        top = np.argsort(-mixed, axis=1, kind="stable")[:, :4]
        amounts = np.take_along_axis(mixed, top, axis=1)
        pad = 4 - top.shape[1]
        palette.append(np.pad(allowed[top], ((0, 0), (0, pad))))
        influence.append(
            np.pad(amounts / amounts.sum(1, keepdims=True), ((0, 0), (0, pad)))
        )
    return np.vstack(palette).astype("<u2"), np.vstack(influence).astype("<f4")


def inverse_bind(model, palette):
    if "inverseBindMatrices" not in palette:
        return np.repeat(np.eye(4)[None], len(palette["joints"]), axis=0)
    return (
        model.array(palette["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1)
    )


def bind_rotation(matrix):
    basis = matrix[:3, :3]
    return Rotation.from_matrix(basis / np.linalg.norm(basis, axis=0))


def rigid_fit(points, mounts, remap, src_inverse, dst_inverse):
    """Move each hard shell between bind frames while preserving its dimensions."""
    output = np.empty_like(points)
    for joint in np.unique(mounts):
        if remap[joint] < 0:
            raise ValueError("Recipient lacks the rigid garment mount")
        source = np.linalg.inv(src_inverse[joint])
        target = np.linalg.inv(dst_inverse[remap[joint]])

        # Remove scale from the bind frames so even a long prop keeps its length.
        rotation = bind_rotation(target) * bind_rotation(source).inv()
        rows = mounts == joint
        output[rows] = target[:3, 3] + rotation.apply(points[rows] - source[:3, 3])
    return output


def face_keys(faces, count):
    a, b, c = np.sort(faces, axis=1).T
    return (a * count + b) * count + c


def context(folder):
    report = json.loads((folder / "report.json").read_text())
    transfer = Path(report["input_transfer"])
    if not transfer.exists() and (folder / "transfer").is_dir():
        transfer = folder / "transfer"
    config = json.loads((transfer / "config.json").read_text())
    guide = json.loads(Path(config["guidance"]).read_text())
    if guide["source_hash"] != report["source_hash"]:
        raise ValueError("Guide does not match the assembled character")
    proxy = Glb(transfer / "3-skin/body.glb")
    _, prim = body(proxy)
    p, f = geometry(proxy, prim)
    library = Path(config["guidance"]).parent / guide.get(
        "garment_preset_library", "garment-presets"
    )
    return guide, p, f, proxy.array(prim["attributes"]["TEXCOORD_0"]), library


def swap(donor, recipient, output, skin_review=None):
    src, source, sf, suv, library = context(donor)
    dst, target, tf, tuv, _ = context(recipient)
    if not np.array_equal(sf, tf) or not np.array_equal(suv, tuv):
        raise ValueError("Outfit transfer needs corresponding body topology and UVs")
    mapping = BodyMap(source, target, sf)
    donor_file = donor / "4-garments/character.glb"
    exclusions = {}
    bindings = {}
    if skin_review is not None:
        review = json.loads(Path(skin_review).read_text())
        if hashlib.sha256(donor_file.read_bytes()).hexdigest() != review["sha256"]:
            raise ValueError("Skin review belongs to a different donor export")
        exclusions = dict(review["exclude"])
        bindings = dict(review.get("binding", {}))
    check_skin = bool(exclusions)
    source_dressed = Glb(donor_file)
    target_dressed = Glb(recipient / "4-garments/character.glb")
    source_bare = Glb(donor / "3-skin/body.glb")
    target_bare = Glb(recipient / "3-skin/body.glb")
    _, src_body = body(source_dressed)
    _, src_full = body(source_bare)
    target_node, dst_body = body(target_dressed)
    _, dst_full = body(target_bare)
    hidden = np.setdiff1d(
        face_keys(geometry(source_bare, src_full)[1], len(source)),
        face_keys(geometry(source_dressed, src_body)[1], len(source)),
    )
    _, target_faces = geometry(target_bare, dst_full)
    keep = ~np.isin(face_keys(target_faces, len(target)), hidden)
    skin_samples, cloth_samples = [], []

    def pieces(guide):
        return {
            p["name"]
            for p in guide["garments"]
            if p.get("label", "garment") == "garment"
        }

    source_pieces, target_pieces = pieces(src), pieces(dst)
    rigid = {}
    for item in src["garments"]:
        if item["name"] in source_pieces:
            rig, _ = rig_for(item, library)
            if rig["mode"] == "rigid":
                rigid[item["name"]] = rig
    for node in target_dressed.doc["nodes"]:
        if node.get("name") in target_pieces:
            node.pop("mesh", None)
            node.pop("skin", None)
    target_skin = target_dressed.doc["skins"][target_node["skin"]]
    target_names = {
        target_dressed.doc["nodes"][j]["name"]: i
        for i, j in enumerate(target_skin["joints"])
    }
    # Appending GLB data can relocate the mutable backing buffer. Keep the
    # recipient arrays used by later primitives independent of that buffer.
    target_positions = geometry(target_dressed, dst_body)[0]
    body_joints = target_dressed.array(dst_body["attributes"]["JOINTS_0"]).copy()
    body_weights = target_dressed.array(dst_body["attributes"]["WEIGHTS_0"]).copy()
    if body_weights.dtype.kind in "ui":
        body_weights = body_weights.astype(float) / np.iinfo(body_weights.dtype).max
    joint_names = [
        target_dressed.doc["nodes"][j]["name"] for j in target_skin["joints"]
    ]
    materials = {}
    copied = []
    for node in source_dressed.doc["nodes"]:
        name = node.get("name")
        if "mesh" not in node or name not in source_pieces:
            continue
        if any(k in node for k in ("matrix", "translation", "rotation", "scale")):
            raise ValueError(f"{name}: expected a rest-space garment")
        skin = source_dressed.doc["skins"][node["skin"]]
        remap = np.array(
            [
                target_names.get(source_dressed.doc["nodes"][j]["name"], -1)
                for j in skin["joints"]
            ]
        )
        primitives = []
        for part, prim in enumerate(
            source_dressed.doc["meshes"][node["mesh"]]["primitives"]
        ):
            if prim.get("targets") or prim.get("mode", 4) != 4:
                raise ValueError(
                    f"{name}: expected a triangle garment without corrective shape keys"
                )
            p, f = geometry(source_dressed, prim)
            if check_skin:
                excluded = np.zeros(len(f), dtype=bool)
                ids = np.asarray(exclusions.pop(f"{name}/{part}", []), dtype=np.int64)
                if np.any((ids < 0) | (ids >= len(f))):
                    raise ValueError(f"Invalid reviewed triangle for {name}")
                excluded[ids] = True
                centers = p[f].mean(axis=1)
                skin_samples.append(centers[excluded])
                cloth_samples.append(centers[~excluded])
                f = f[~excluded]
                if not len(f):
                    continue
            used, inv = np.unique(f, return_inverse=True)
            f = inv.reshape(-1, 3)
            # Work in batches for dense scans. Only vertices used by this piece travel.
            rigid_joints = None
            if name in rigid:
                attrs_in = prim["attributes"]
                authored = source_dressed.array(attrs_in["JOINTS_0"])[used]
                weights_in = source_dressed.array(attrs_in["WEIGHTS_0"])[used]
                bone = rigid[name].get("bone")
                if bone:
                    names = [
                        source_dressed.doc["nodes"][j]["name"] for j in skin["joints"]
                    ]
                    mounts = np.full(len(used), names.index(bone))
                else:
                    # The donor binding resolves each hard piece to its own joint.
                    if np.any(np.count_nonzero(weights_in, axis=1) != 1):
                        raise ValueError(f"{name}: rigid piece needs a resolved mount")
                    mounts = authored[np.arange(len(used)), weights_in.argmax(axis=1)]

                mapped = rigid_fit(
                    p[used],
                    mounts,
                    remap,
                    inverse_bind(source_dressed, skin),
                    inverse_bind(target_dressed, target_skin),
                )
                rigid_joints = remap[mounts]
            else:
                mapped = np.vstack(
                    [
                        mapping.map(p[ids])
                        for ids in np.array_split(used, 1 + len(used) // 4096)
                    ]
                )
            attrs = {
                "POSITION": add(target_dressed, mapped.astype("<f4"), "VEC3"),
                "NORMAL": add(
                    target_dressed, normals(mapped, f)[0].astype("<f4"), "VEC3"
                ),
            }
            if rigid_joints is not None:
                ji = np.zeros((len(used), 4), dtype="<u2")
                wt = np.zeros((len(used), 4), dtype="<f4")
                ji[:, 0] = rigid_joints
                wt[:, 0] = 1
                attrs["JOINTS_0"] = add(target_dressed, ji, "VEC4", 5123)
                attrs["WEIGHTS_0"] = add(target_dressed, wt, "VEC4")
            elif name in bindings:
                ji, wt = bind_to_regions(
                    mapped,
                    target_positions,
                    body_joints,
                    body_weights,
                    joint_names,
                    bindings[name],
                )
                attrs["JOINTS_0"] = add(target_dressed, ji, "VEC4", 5123)
                attrs["WEIGHTS_0"] = add(target_dressed, wt, "VEC4")
            for key, acc in prim["attributes"].items():
                if (name in bindings or rigid_joints is not None) and key.startswith(
                    ("JOINTS_", "WEIGHTS_")
                ):
                    continue
                if key in ("POSITION", "NORMAL", "TANGENT"):
                    continue
                info = source_dressed.doc["accessors"][acc]
                values = source_dressed.array(acc)[used]
                kind = info["componentType"]
                if key.startswith("JOINTS_"):
                    values = remap[values]
                    weights = source_dressed.array(
                        prim["attributes"]["WEIGHTS_" + key[7:]]
                    )[used]
                    if np.any((values < 0) & (weights > 0)):
                        raise ValueError(
                            f"{name}: target lacks a weighted garment joint"
                        )
                    values = np.maximum(values, 0).astype("<u2")
                    kind = 5123
                attrs[key] = add(target_dressed, values, info["type"], kind)
                if info.get("normalized"):
                    target_dressed.doc["accessors"][attrs[key]]["normalized"] = True
            mid = prim.get("material", -1)
            if mid not in materials:
                materials[mid] = material(target_dressed, source_dressed, prim)
            primitives.append(
                dict(
                    indices=idx(target_dressed, f),
                    attributes=attrs,
                    material=materials[mid],
                )
            )
        if not primitives:
            continue
        mesh_id = len(target_dressed.doc["meshes"])
        node_id = len(target_dressed.doc["nodes"])
        target_dressed.doc["meshes"].append(
            dict(name=name, primitives=primitives, extras=dict(charloom_part="garment"))
        )
        target_dressed.doc["nodes"].append(
            dict(
                name=name,
                mesh=mesh_id,
                skin=target_node["skin"],
                extras=dict(charloom_part="garment"),
            )
        )
        target_dressed.doc["scenes"][target_dressed.doc.get("scene", 0)][
            "nodes"
        ].append(node_id)
        copied.append(name)
    if exclusions:
        raise ValueError(f"Unknown donor primitives in skin review: {list(exclusions)}")
    if any(len(p) for p in skin_samples) and copied:
        # Reopen skin on the recipient wherever the donor mask hid anatomy
        # that was mistakenly included in a garment during segmentation.
        centers = source[target_faces].mean(axis=1)
        skin_dist = cKDTree(np.vstack(skin_samples)).query(centers)[0]
        cloth_dist = cKDTree(np.vstack(cloth_samples)).query(centers)[0]
        keep |= skin_dist < cloth_dist
    dst_body["indices"] = idx(target_dressed, target_faces[keep])
    if not copied:
        raise ValueError("No donor garments matched the calibrated guide")
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        path = Path(tmp) / "character.glb"
        finish(target_dressed, path)
        os.link(path, output)
    return dict(
        donor=donor.name,
        recipient=recipient.name,
        garments=copied,
        hidden_body_faces=int((~keep).sum()),
        output=str(output.resolve()),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ("donor", "recipient", "output"):
        parser.add_argument(arg, type=Path)
    parser.add_argument(
        "--skin-review",
        "--fit-review",
        type=Path,
        help="Reviewed donor skin triangle exclusions, keyed by mesh name and primitive index",
    )
    args = parser.parse_args()
    print(json.dumps(swap(args.donor, args.recipient, args.output, args.skin_review)))
