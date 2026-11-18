"""Restore binding on an existing cut mouth with its measured upper and lower rims."""

from .mesh_output import add
import numpy as np
from scipy.interpolate import PchipInterpolator

from .mesh import Glb
from .observed_binding import smooth


def apply(model, curves):
    g = Glb(model)
    skin = g.doc["skins"][0]
    joints = [g.doc["nodes"][j]["name"] for j in skin["joints"]]
    arcs = []
    for key in ("upper", "lower"):
        pts = np.asarray(curves[key], float)
        x, ids = np.unique(pts[:, 0], return_index=True)
        arcs.append(PchipInterpolator(x, pts[ids, 1:], axis=0))
    start, end = max(a.x[0] for a in arcs), min(a.x[-1] for a in arcs)
    if start >= end:
        raise ValueError("Measured lip rims do not overlap")
    grid = np.linspace(start, end, 128)
    upper, lower = [a(grid) for a in arcs]
    middle = (upper[:, 0] + lower[:, 0]) / 2
    gap = np.maximum(
        upper[:, 0] - lower[:, 0], 0.0002 * np.sin(np.linspace(0, np.pi, 128))
    )
    gap[[0, -1]] = 0
    radius = (end - start) / 2
    cx, cy, cz = (start + end) / 2, middle[63:65].mean(), upper[63:65, 1].mean()
    jaw, head = joints.index("jaw"), joints.index("head")
    g.doc["nodes"][skin["joints"][jaw]].setdefault("extras", {})[
        "charloomLipClearanceM"
    ] = float(gap[63:65].mean())
    lower_frame = None
    changed = {}
    for name in ("body", "beard", "teeth", "tongue"):
        for mesh in [m for m in g.doc["meshes"] if m.get("name") == name]:
            for prim in mesh["primitives"]:
                attr = prim["attributes"]
                pos = g.array(attr["POSITION"]).astype(float)
                ji = g.array(attr["JOINTS_0"]).copy()
                if name in ("teeth", "tongue"):
                    normals = g.array(attr["NORMAL"]).astype(float)
                    for joint in [jaw] if name == "tongue" else [head, jaw]:
                        mask = (
                            ji[:, 0] == joint
                            if name != "tongue"
                            else np.ones(len(ji), bool)
                        )
                        p = pos[mask]
                        if len(p) < 2:
                            continue
                        if name != "tongue":
                            centered = p - p.mean(0)
                            theta = np.arctan(
                                (centered[:, 1] @ centered[:, 2])
                                / max(centered[:, 2] @ centered[:, 2], 1e-15)
                            )
                            c, s = np.cos(theta), np.sin(theta)
                            mat = np.array(
                                [
                                    [0.78, 0, 0],
                                    [0, 0.70 * c, -0.70 * s],
                                    [0, 0.72 * s, 0.72 * c],
                                ]
                            )
                            q = p @ mat.T
                            front = q[q[:, 2] >= np.quantile(q[:, 2], 0.85)]
                            tip = np.array(
                                [
                                    np.median(q[:, 0]),
                                    np.quantile(
                                        front[:, 1], 0.035 if joint == head else 0.965
                                    ),
                                    q[:, 2].max(),
                                ]
                            )
                            offset = (
                                np.array(
                                    [
                                        cx,
                                        cy + (0.0016 if joint == head else -0.0035),
                                        cz - (0.012 if joint == head else 0.016),
                                    ]
                                )
                                - tip
                            )
                            if joint == jaw:
                                lower_frame = mat, offset
                        else:
                            if lower_frame is None:
                                raise ValueError(
                                    "A tongue needs its corresponding lower dental arch"
                                )
                            mat, offset = lower_frame
                        q = p @ mat.T + offset
                        q[:, 1] += 0.65 * (np.interp(q[:, 0], grid, middle) - cy)
                        q[:, 2] += 0.82 * (np.interp(q[:, 0], grid, upper[:, 1]) - cz)
                        pos[mask] = q
                        norm = normals[mask] @ np.linalg.inv(mat)
                        normals[mask] = norm / np.maximum(
                            np.linalg.norm(norm, axis=1)[:, None], 1e-12
                        )
                    attr["POSITION"] = add(g, pos.astype("<f4"), "VEC3")
                    attr["NORMAL"] = add(g, normals.astype("<f4"), "VEC3")
                    continue
                x, y, z = pos.T
                center = np.interp(x, grid, middle)
                h = np.interp(x, grid, gap)
                top, bottom = center + 0.5 * h, center - 0.5 * h
                front = np.interp(x, grid, upper[:, 1])
                u = np.abs(x - cx) / radius
                mask = (
                    smooth((1.65 - u) / 0.5)
                    * smooth((top + 0.011 - y) / 0.007)
                    * smooth((y - bottom + 0.085) / 0.035)
                    * smooth((z - front + 0.053) / 0.012)
                )
                nasal = np.zeros_like(x)
                nose = curves.get("nose_tip")
                if (
                    name == "body"
                    and nose
                    and nose[2] - np.interp(nose[0], grid, upper[:, 1]) > radius * 0.5
                ):
                    nasal = (
                        smooth((z - front - 0.008) / (radius * 0.22))
                        * smooth((1.2 - u) / 0.35)
                        * smooth(
                            (y - np.minimum(nose[1], bottom) + radius * 0.40)
                            / (radius * 0.18)
                        )
                    )
                live = (mask > 0) | (nasal > 0)
                weights = g.array(attr["WEIGHTS_0"]).copy()
                weights = (
                    weights.astype(float) / np.iinfo(weights.dtype).max
                    if weights.dtype.kind in "ui"
                    else weights.astype(float)
                )
                dense = np.zeros((len(x), len(joints)))
                lip = smooth(
                    1 - np.abs(y - center) / (0.018 if name == "body" else 0.012)
                )
                active = lip * smooth((z - front + 0.043) / 0.025)
                fraction = smooth((y - bottom) / np.maximum(h, 1e-6))
                corner = smooth((u - 0.55) / 0.45)
                lateral = smooth(u / 0.64)
                rows = np.arange(len(x))
                side = lambda stem: np.where(
                    x > cx, joints.index(stem + ".L"), joints.index(stem + ".R")
                )
                dense[rows, side("levator05")] = active * corner
                for mid, outer, f in [
                    ("oris05", "oris03", fraction),
                    ("oris01", "oris07", 1 - fraction),
                ]:
                    amount = active * (1 - corner) * f
                    dense[:, joints.index(mid)] += amount * (1 - lateral)
                    dense[rows, side(outer)] += amount * lateral
                jaw_share = (1 - fraction) * (0.40 + 0.30 * lip)
                dense[:, jaw] = (1 - active) * jaw_share
                dense[:, head] = (1 - active) * (1 - jaw_share)
                dense *= mask[:, None]
                np.add.at(dense, (rows[:, None], ji), weights * (1 - mask[:, None]))
                dense *= 1 - nasal[:, None]
                dense[:, head] += nasal
                rank = np.argsort(-dense[live], axis=1, kind="stable")[:, :4]
                wt = np.take_along_axis(dense[live], rank, axis=1)
                ji[live] = rank
                weights[live] = wt / wt.sum(1, keepdims=True)
                attr["JOINTS_0"] = add(g, ji.astype("<u2"), "VEC4", 5123)
                attr["WEIGHTS_0"] = add(g, weights.astype("<f4"), "VEC4")
                if name == "body" and "COLOR_0" in attr:
                    colors = g.array(attr["COLOR_0"])
                    colors = (
                        colors.astype(float) / np.iinfo(colors.dtype).max
                        if colors.dtype.kind in "ui"
                        else colors.astype(float)
                    )
                    lining = (
                        smooth((front - z - 0.001) / 0.003)
                        * smooth((0.0015 - np.abs(y - center)) / 0.001)
                        * smooth((1.0 - u) / 0.03)
                        * (z > front - 0.045)
                    )
                    colors[:, :3] += (
                        np.array([0.077, 0.021, 0.024]) - colors[:, :3]
                    ) * lining[:, None]
                    attr["COLOR_0"] = add(
                        g, colors.astype("<f4"), "VEC" + str(colors.shape[1])
                    )
                changed[name] = changed.get(name, 0) + int(live.sum())
    g.doc["buffers"][0]["byteLength"] = len(g.blob)
    g.write(model)
    return changed
