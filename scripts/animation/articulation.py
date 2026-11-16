"""Two-bone reach, elbow and wrist limits, and axial twist transfer."""

import numpy as np
from scipy.spatial.transform import Rotation

from .scene import decompose


def unit(vector):
    vector = np.asarray(vector, float)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-10:
        raise ValueError("A rotation axis or bone has zero length")
    return vector / length


def between(source, destination):
    a, b = unit(source), unit(destination)
    axis = np.cross(a, b)
    angle = np.arctan2(np.linalg.norm(axis), np.dot(a, b))
    if np.linalg.norm(axis) < 1e-10:
        if angle < 1e-8:
            return Rotation.identity()
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
    return Rotation.from_rotvec(angle * unit(axis))


def twist_angle(rot, axis):
    q = rot.as_quat()
    return (2 * np.arctan2(q[:3] @ axis, q[3]) + np.pi) % (2 * np.pi) - np.pi


def limited(rot, spec):
    reference = Rotation.from_quat(spec["reference_rotation"])
    axis = unit(spec["twist_axis"])
    delta = reference.inv() * rot
    hinge = spec.get("hinge")
    if hinge:
        normal = unit(hinge["axis_parent"])
        a, b = reference.apply(axis), rot.apply(axis)
        theta = np.arctan2(normal @ np.cross(a, b), a @ b)
        swing = (
            Rotation.from_rotvec(normal * np.clip(theta, *hinge["range_rad"]))
            * reference
        )
    else:
        twist = Rotation.from_rotvec(axis * twist_angle(delta, axis))
        v = (delta * twist.inv()).as_rotvec()
        swing = reference * Rotation.from_rotvec(
            v * min(1.0, spec["max_swing_rad"] / max(1e-12, np.linalg.norm(v)))
        )
    angle = np.clip(twist_angle(swing.inv() * rot, axis), *spec["twist_range_rad"])
    return swing * Rotation.from_rotvec(axis * angle)


def arm_limits(scene, spec):
    upper, elbow, wrist = [
        scene.joint(spec[key]) for key in ("upper_arm", "forearm", "wrist")
    ]
    fingers = [scene.joint(n) for n in spec["palm_landmarks"]]
    if (
        scene.parents.get(elbow) != upper
        or scene.parents.get(wrist) != elbow
        or len(fingers) != 3
        or any(scene.parents.get(i) != wrist for i in fingers)
    ):
        raise ValueError(
            "Arm articulation needs a direct arm chain and three palm landmarks"
        )
    incoming, axis = unit(scene.rest[elbow, :3]), unit(scene.rest[wrist, :3])
    reference = Rotation.from_quat(scene.rest[elbow, 3:7])
    outgoing = reference.apply(axis)
    hinge = unit(np.cross(incoming, outgoing))
    neutral = between(outgoing, incoming) * reference
    palm_axis = unit(scene.rest[fingers[1], :3])
    palm_normal = unit(
        np.cross(scene.rest[fingers[0], :3] - scene.rest[fingers[2], :3], palm_axis)
    )
    wrist_rest = Rotation.from_quat(scene.rest[wrist, 3:7])
    wrist_neutral = between(wrist_rest.apply(palm_axis), axis) * wrist_rest
    normal = (neutral * wrist_neutral).apply(palm_normal)
    desired = hinge * spec["neutral_palm_hinge_sign"]
    angle = np.arctan2(incoming @ np.cross(normal, desired), normal @ desired)
    neutral = Rotation.from_rotvec(angle * incoming) * neutral
    return [
        dict(
            joint=elbow,
            reference_rotation=neutral.as_quat(),
            twist_axis=axis,
            twist_range_rad=spec["forearm_twist_range_rad"],
            max_swing_rad=spec["max_flexion_rad"],
            hinge=dict(axis_parent=hinge, range_rad=[0.0, spec["max_flexion_rad"]]),
        ),
        dict(
            joint=wrist,
            reference_rotation=wrist_neutral.as_quat(),
            twist_axis=palm_axis,
            twist_range_rad=spec["wrist_twist_range_rad"],
            max_swing_rad=spec["wrist_swing_limit_rad"],
        ),
    ]


def distribute_twist(scene, pose, child, reference, fraction=1.):
    parent = scene.parents[child]
    if (
        not np.allclose(pose[parent, 7:], pose[parent, 7], atol=1e-7)
        or pose[parent, 7] <= 0
    ):
        raise ValueError("Twist distribution needs a positive uniform parent scale")
    current = Rotation.from_quat(pose[child, 3:7])
    axis = unit(pose[child, :3])
    delta = current * Rotation.from_quat(reference).inv()
    roll = Rotation.from_rotvec(axis * twist_angle(delta, axis) * fraction)
    pose[parent, 3:7] = (Rotation.from_quat(pose[parent, 3:7]) * roll).as_quat()
    pose[child, 3:7] = (roll.inv() * current).as_quat()


def aim(scene, pose, joint, child, target):
    world = scene.world(pose)
    origin = world[joint, :3, 3]
    if np.linalg.norm(target - origin) < 1e-10:
        return
    correction = between(world[child, :3, 3] - origin, target - origin)
    global_rot = Rotation.from_quat(decompose(world[joint])[3:7])
    pose[joint, 3:7] = (
        Rotation.from_quat(pose[joint, 3:7])
        * global_rot.inv()
        * correction
        * global_rot
    ).as_quat()


def two_bone(scene, pose, chain, target, pole):
    tip, mid, root = chain
    w = scene.world(pose)
    a, b, c = w[[root, mid, tip], :3, 3]
    u, v = np.linalg.norm(b - a), np.linalg.norm(c - b)
    dist = np.linalg.norm(target - a)
    if min(u, v, dist) < 1e-9:
        return
    axis = (target - a) / dist
    delta = (np.array(pole) if pole is not None else b) - a
    plane = delta - axis * (delta @ axis)
    if np.linalg.norm(plane) < 1e-8:
        plane = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))])
    reach = np.clip(dist, max(abs(u - v), 1e-9), u + v)
    x = (reach**2 + u**2 - v**2) / (2 * reach)
    y = np.sqrt(max(0.0, u * u - x * x))
    aim(scene, pose, root, mid, a + x * axis + y * unit(plane))
    aim(scene, pose, mid, tip, a + reach * axis)
