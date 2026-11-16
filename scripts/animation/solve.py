"""Damped multi-effector pose solve in glTF meters, with analytic rotation Jacobians."""

import numpy as np
from scipy.spatial.transform import Rotation

from .scene import decompose
from .articulation import limited, two_bone
from .collision import contacts


def solve(scene, pose, effectors, config):
    collision = config.get("collision")
    if not effectors and not collision:
        return pose, []
    chains, targets = [], []
    for e in effectors:
        idx = scene.joint(e["joint"])
        root = scene.joint(e["chain_root"]) if e.get("chain_root") is not None else None
        chain = []
        i = idx
        while i is not None:
            chain.append(i)
            if i == root:
                break
            i = scene.parents.get(i)
        if root is not None and root != chain[-1]:
            raise ValueError(f"Chain root must be an ancestor of {e['joint']}")
        chains.append(chain)
        targets.append(
            (
                np.asarray(e["position"], float),
                np.asarray(e.get("position_axes", [True] * 3), bool),
                Rotation.from_quat(e["rotation"])
                if e.get("rotation") is not None
                else None,
            )
        )
    collision_chains = {}
    for cap in (collision or {}).get("capsules", []):
        root = cap.get("chain_root")
        if root is None:
            continue
        for endpoint in ("start", "end"):
            j = cap[endpoint]
            path = []
            while j is not None:
                path.append(j)
                if j == root:
                    break
                j = scene.parents.get(j)
            if path[-1] != root:
                raise ValueError("Collision chain root must be an ancestor")
            collision_chains[cap["name"], endpoint] = path
    joints = sorted(set().union(*chains, *collision_chains.values()))
    slots = {j: 3 * i for i, j in enumerate(joints)}
    free_roots = (
        sorted({c[-1] for c in chains if c[-1] not in scene.parents})
        if config.get("root_translation")
        else []
    )
    n = 3 * (len(joints) + len(free_roots))
    limits = {scene.joint(lim["joint"]): lim for lim in config.get("limits", [])}
    articulations = {
        scene.joint(a["joint"]): a for a in config.get("articulations", [])
    }
    initial = pose
    pose = pose.copy()
    for e, chain, (position, axes, _) in zip(effectors, chains, targets):
        if len(chain) == 3 and axes.all() and config.get("warm_start_two_bone", True):
            two_bone(scene, pose, chain, position, e.get("pole"))
    for j, spec in articulations.items():
        pose[j, 3:7] = limited(Rotation.from_quat(pose[j, 3:7]), spec).as_quat()
    used = set(config) - {
        "iterations",
        "damping",
        "tolerance_m",
        "rotation_tolerance_rad",
        "max_step_rad",
        "root_translation",
        "limits",
        "posture_weight",
        "articulations",
        "warm_start_two_bone",
        "collision",
    }
    if used:
        raise ValueError(f"Unsupported solver options: {sorted(used)}")
    for iteration in range(1 + config.get("iterations", 80)):
        world = scene.world(pose)
        blocks, residuals, report = [], [], []
        for e, chain, (position, axes, rotation) in zip(effectors, chains, targets):
            i = chain[0]
            err = position - world[i, :3, 3]
            rot_err = (
                (
                    rotation * Rotation.from_quat(decompose(world[i])[3:7]).inv()
                ).as_rotvec()
                if rotation
                else None
            )
            weights = np.sqrt(
                [e.get("position_weight", 1.0), e.get("rotation_weight", 1.0)]
            )
            jac = np.zeros((3, n))
            rjac = np.zeros((3, n))
            for j in chain:
                start = slots[j]
                basis = Rotation.from_quat(decompose(world[j])[3:7]).as_matrix()
                jac[:, start : start + 3] = np.cross(
                    basis.T, world[i, :3, 3] - world[j, :3, 3]
                ).T
                rjac[:, start : start + 3] = basis
            if chain[-1] in free_roots:
                col = 3 * (len(joints) + free_roots.index(chain[-1]))
                jac[:, col : col + 3] = np.eye(3)
            blocks.append(jac[axes] * weights[0])
            residuals.append(err[axes] * weights[0])
            if rotation:
                blocks.append(rjac * weights[1])
                residuals.append(rot_err * weights[1])
            report.append(
                dict(
                    joint=scene.names[i],
                    position_error_m=float(np.linalg.norm(err[axes])),
                    rotation_error_rad=float(np.linalg.norm(rot_err))
                    if rotation
                    else 0.0,
                )
            )
        deepest = 0.0
        for a, b, s, t, normal, depth in contacts(
            collision or {"capsules": [], "pairs": []}, world
        ):
            deepest = max(deepest, depth)
            row = np.zeros(n)
            for cap, frac, sign in ((a, s, 1.0), (b, t, -1.0)):
                for endpoint, factor in (("start", 1.0 - frac), ("end", frac)):
                    idx = cap[endpoint]
                    for j in collision_chains.get((cap["name"], endpoint), []):
                        basis = Rotation.from_quat(decompose(world[j])[3:7]).as_matrix()
                        row[slots[j] : slots[j] + 3] += (
                            sign
                            * factor
                            * (
                                np.cross(basis.T, world[idx, :3, 3] - world[j, :3, 3])
                                @ normal
                            )
                        )
            weight = np.sqrt(collision.get("weight", 100.0))
            blocks.append(row[None, :] * weight)
            residuals.append(np.array([depth * weight]))
        if deepest <= (collision or {}).get("tolerance_m", 0.001) and all(
            r["position_error_m"] <= config.get("tolerance_m", 0.001)
            and r["rotation_error_rad"] <= config.get("rotation_tolerance_rad", 0.01)
            for r in report
        ):
            break
        if iteration == config.get("iterations", 80):
            break
        matrix, error = np.vstack(blocks), np.concatenate(residuals)
        damping = max(1e-6, config.get("damping", 0.03))
        regularizer = config.get("posture_weight", 0.00001)
        bias = np.zeros(n)
        for j in joints:
            bias[slots[j] : slots[j] + 3] = (
                Rotation.from_quat(pose[j, 3:7]).inv()
                * Rotation.from_quat(initial[j, 3:7])
            ).as_rotvec()
        step = bias * regularizer / (regularizer + damping**2)
        step += matrix.T @ np.linalg.solve(
            matrix @ matrix.T + np.eye(len(error)) * (damping**2 + regularizer),
            error - matrix @ step,
        )
        for j in joints:
            change = step[slots[j] : slots[j] + 3]
            length = np.linalg.norm(change)
            cap = config.get("max_step_rad", 0.25)
            rot = Rotation.from_quat(pose[j, 3:7]) * Rotation.from_rotvec(
                change * min(1.0, cap / max(length, 1e-12))
            )
            if j in limits:
                ref = Rotation.from_quat(
                    limits[j].get("reference_rotation", scene.rest[j, 3:7])
                )
                diff = (ref.inv() * rot).as_rotvec()
                rot = ref * Rotation.from_rotvec(
                    diff
                    * min(
                        1.0,
                        limits[j]["max_angle_rad"] / max(np.linalg.norm(diff), 1e-12),
                    )
                )
            if j in articulations:
                rot = limited(rot, articulations[j])
            pose[j, 3:7] = rot.as_quat()
        for index, j in enumerate(free_roots, len(joints)):
            pose[j, :3] += step[3 * index : 3 * index + 3]
    return pose, report
