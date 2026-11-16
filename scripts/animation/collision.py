"""Capsule clearances fitted from skinned vertices and evaluated on posed bone segments."""

import numpy as np

from .articulation import unit


def capsule_setup(surface, config):
    scene = surface.scene
    world = scene.world(scene.rest)
    names = [s["name"] for s in config["capsules"]]
    if len(names) != len(set(names)):
        raise ValueError("Each collision capsule needs a unique name")
    caps = []
    for spec in config["capsules"]:
        a, b = [scene.joint(spec[k]) for k in ("start", "end")]
        c = dict(spec, start=a, end=b)
        if c.get("chain_root") is not None:
            c["chain_root"] = scene.joint(c["chain_root"])
        axis = world[b, :3, 3] - world[a, :3, 3]
        if c.get("radius_m") is None:
            skin = [scene.joint(n) for n in spec["skin_joints"]]
            pieces = []
            for part, points in zip(surface.parts, surface.rest):
                if part.weights is None:
                    continue
                mask = (np.isin(part.bones[part.palette], skin) * part.weights).sum(
                    1
                ) >= 0.5
                pieces.extend(points[mask] - world[a, :3, 3])
            offsets = np.asarray(pieces).reshape(-1, 3)
            t = offsets @ axis / (axis @ axis)
            lo, hi = spec.get("interval", (0.0, 1.0))
            selected = (t >= lo) & (t <= hi)
            radial = (offsets - t[:, None] * axis)[selected]
            if not len(radial):
                raise ValueError(f"No skin for capsule {c['name']}")
            q = spec.get("radius_quantile", 0.95)
            c["radius_m"] = float(np.quantile(np.linalg.norm(radial, axis=1), q))
            if spec.get("lateral_reference"):
                i, j = [scene.joint(n) for n in spec["lateral_reference"]]
                y = unit(axis)
                lateral = world[i, :3, 3] - world[j, :3, 3]
                x = unit(lateral - y * (lateral @ y))
                z = np.cross(y, x)
                c["cross_section"] = dict(
                    lateral_axis_local=unit(np.linalg.solve(world[a, :3, :3], x)),
                    radii_m=np.quantile(
                        np.abs(radial @ np.stack([x, z], axis=1)), q, axis=0
                    ),
                )
        caps.append(c)
    return dict(
        config,
        capsules=caps,
        pairs=[[names.index(a), names.index(b)] for a, b in config["pairs"]],
    )


def nearest(a, b, c, d):
    """Minimize squared distance over the closed unit square for two line segments."""
    u, v, w = b - a, d - c, a - c
    uu, vv, uv = u @ u, v @ v, u @ v
    candidates = [
        (0.0, np.clip(w @ v / max(vv, 1e-20), 0, 1)),
        (1.0, np.clip((b - c) @ v / max(vv, 1e-20), 0, 1)),
        (np.clip(-w @ u / max(uu, 1e-20), 0, 1), 0.0),
        (np.clip((d - a) @ u / max(uu, 1e-20), 0, 1), 1.0),
    ]
    det = uu * vv - uv * uv
    if det > 1e-18:
        s = (uv * (w @ v) - vv * (w @ u)) / det
        t = (uu * (w @ v) - uv * (w @ u)) / det
        if 0 <= s <= 1 and 0 <= t <= 1:
            candidates.append((s, t))
    return min(candidates, key=lambda st: np.linalg.norm(w + u * st[0] - v * st[1]))


def radius(c, world, normal):
    section = c.get("cross_section")
    if not section:
        return c["radius_m"]
    y = unit(world[c["end"], :3, 3] - world[c["start"], :3, 3])
    lateral = world[c["start"], :3, :3] @ section["lateral_axis_local"]
    x = unit(lateral - y * (y @ lateral))
    rx, rz = section["radii_m"]
    return np.linalg.norm(
        [rx * (normal @ x), rz * (normal @ np.cross(y, x)), min(rx, rz) * (y @ normal)]
    )


def contacts(config, world):
    caps = config["capsules"]
    for ai, bi in config["pairs"]:
        a, b = caps[ai], caps[bi]
        ends = []
        for cap in (a, b):
            origin = world[cap["start"], :3, 3]
            axis = world[cap["end"], :3, 3] - origin
            ends.extend(origin + t * axis for t in cap.get("interval", (0.0, 1.0)))
        s, t = nearest(*ends)
        v = (1 - s) * ends[0] + s * ends[1] - (1 - t) * ends[2] - t * ends[3]
        distance = np.linalg.norm(v)
        normal = v / distance if distance > 1e-9 else np.array([1.0, 0.0, 0.0])
        gap = distance - radius(a, world, normal) - radius(b, world, normal)
        margin = config.get("margin_m", 0.0)
        if gap < margin:

            def fraction(c, t):
                lo, hi = c.get("interval", (0.0, 1.0))
                return lo + (hi - lo) * t

            yield a, b, fraction(a, s), fraction(b, t), normal, margin - gap
