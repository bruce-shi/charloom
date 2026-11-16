"""Evaluate glTF node transforms and animation curves in their native Y-up frame."""

import numpy as np
from scipy.spatial.transform import Rotation

from roster.mesh import Glb


def blend_quats(a, b, weight):
    a = Rotation.from_quat(a)
    delta = a.inv() * Rotation.from_quat(b)
    return (a * Rotation.from_rotvec(delta.as_rotvec() * weight)).as_quat()


def decompose(m):
    scale = np.linalg.norm(m[:3, :3], axis=0)
    if min(scale) <= 1e-12:
        raise ValueError("Cannot decompose a singular node transform")
    if np.linalg.det(m[:3, :3]) < 0:
        scale[0] *= -1
    return np.r_[m[:3, 3], Rotation.from_matrix(m[:3, :3] / scale).as_quat(), scale]


def matrices(pose):
    result = np.broadcast_to(np.eye(4), (*pose.shape[:-1], 4, 4)).copy()
    result[:, :3, :3] = (
        Rotation.from_quat(pose[:, 3:7]).as_matrix() * pose[:, None, 7:10]
    )
    result[:, :3, 3] = pose[:, :3]
    return result


class Scene:
    def __init__(self, path):
        self.glb = g = Glb(path)
        nodes = g.doc["nodes"]
        self.names = [n.get("name", "") for n in nodes]
        self.parents = {
            j: i for i, n in enumerate(nodes) for j in n.get("children", [])
        }
        self.order = []
        seen = set()
        todo = [i for i in range(len(nodes)) if i not in self.parents]
        while todo:
            i = todo.pop()
            if i in seen:
                raise ValueError("A glTF node has multiple parents or a cycle")
            seen.add(i)
            self.order.append(i)
            todo.extend(nodes[i].get("children", []))
        if len(set(self.order)) != len(nodes) or len(self.order) != len(nodes):
            raise ValueError("Invalid node hierarchy")
        self.rest = np.array(
            [
                decompose(np.array(n["matrix"]).reshape(4, 4).T)
                if "matrix" in n
                else [
                    *n.get("translation", [0] * 3),
                    *n.get("rotation", [0, 0, 0, 1]),
                    *n.get("scale", [1] * 3),
                ]
                for n in nodes
            ],
            dtype=float,
        )
        self.clips = {}
        for animation in g.doc.get("animations", []):
            name = animation.get("name", str(len(self.clips)))
            if name in self.clips:
                raise ValueError(f"Duplicate animation name {name}")
            channels = {}
            for channel in animation["channels"]:
                sampler = animation["samplers"][channel["sampler"]]
                times = g.array(sampler["input"]).ravel().astype(float)
                interpolation = sampler.get("interpolation", "LINEAR")
                factor = 3 if interpolation == "CUBICSPLINE" else 1
                values = (
                    g.array(sampler["output"])
                    .astype(float)
                    .reshape(len(times), factor, -1)
                )
                target = channel["target"]
                channels[target["node"], target["path"]] = times, values, interpolation
            self.clips[name] = channels

    def joint(self, value):
        if isinstance(value, int) and 0 <= value < len(self.names):
            return value
        ids = [i for i, name in enumerate(self.names) if name == value]
        if len(ids) != 1:
            raise ValueError(f"Expected one node named {value!r}, found {len(ids)}")
        return ids[0]

    def world(self, pose):
        result = matrices(pose)
        for i in self.order:
            if i in self.parents:
                result[i] = result[self.parents[i]] @ result[i]
        return result

    def interval(self, clip):
        channels = self.clips[clip].values()
        return min(t[0] for t, _, _ in channels), max(t[-1] for t, _, _ in channels)

    def sample(self, clip, time):
        pose = self.rest.copy()
        morphs = {}
        for (node, prop), (times, values, mode) in self.clips[clip].items():
            i = max(0, np.searchsorted(times, time, side="right") - 1)
            j = min(len(times) - 1, i + 1)
            dt = times[j] - times[i]
            w = np.clip((time - times[i]) / dt, 0, 1) if dt else 0
            if mode == "CUBICSPLINE":
                a, b = values[i], values[j]
                val = (
                    (2 * w**3 - 3 * w * w + 1) * a[1]
                    + (w**3 - 2 * w * w + w) * dt * a[2]
                    + (-2 * w**3 + 3 * w * w) * b[1]
                    + (w**3 - w * w) * dt * b[0]
                )
            elif mode == "STEP" or w == 0:
                val = values[i, 0]
            elif mode != "LINEAR":
                raise ValueError(f"Unknown glTF interpolation {mode}")
            elif prop == "rotation":
                val = blend_quats(values[i, 0], values[j, 0], w)
            else:
                val = (1 - w) * values[i, 0] + w * values[j, 0]
            if prop == "weights":
                morphs[node] = val
            else:
                if prop == "rotation":
                    val = val / np.linalg.norm(val)
                pose[node, SLICES[prop]] = val
        return pose, morphs

    def morph_rest(self, node):
        n = self.glb.doc["nodes"][node]
        mesh = self.glb.doc["meshes"][n["mesh"]]
        return np.array(
            n.get(
                "weights",
                mesh.get(
                    "weights", [0.0] * len(mesh["primitives"][0].get("targets", []))
                ),
            )
        )


SLICES = {"translation": slice(0, 3), "rotation": slice(3, 7), "scale": slice(7, 10)}
