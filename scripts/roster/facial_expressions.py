"""Facial tissue shapes for reconstructed, textured scan heads in working metres."""

import numpy as np
from common import add, attr


def smooth(value):
    t = np.clip(value, 0., 1.)
    return t * t * (3. - 2. * t)


def expression_fields(points, curves):
    """Use the traced mouth and orbital rims to place compact muscle fields."""
    x, y, _ = points.T
    xs, upper, lower, depth = curves["mouth"]
    mouth = np.column_stack((xs, (upper + lower) * .5, depth))
    width = np.ptp(xs)
    result = {}

    def region(center, radii):
        distance2 = np.sum(((points - center) / radii) ** 2, axis=1)
        return np.maximum(1. - distance2, 0.) ** 2

    def move(name, influence, vector):
        if name not in result:
            result[name] = np.zeros_like(points)
        result[name] += influence[:, None] * vector

    for side, sign, corner in [("l", 1, mouth[-1]), ("r", -1, mouth[0])]:
        ex, top, bottom, ez = curves[side]
        eye_width = np.ptp(ex)
        upper_lid = np.interp(x, ex, top)
        lower_lid = np.interp(x, ex, bottom)
        eye = np.array([ex.mean(), top.max(), np.median(ez)])
        inner = 0 if sign > 0 else -1
        # The brow and forehead move together and taper to rest at the lid rim.
        brow = eye + eye_width * np.array([0., .53, -.04])
        forehead = smooth((y - upper_lid) / (.26 * eye_width))
        lift = forehead * region(brow, eye_width * np.array([1.40, 1.22, .95]))
        move("browRaise" + side.upper(), lift, eye_width * np.array([.025 * sign, .255, -.035]))
        glabella = np.array([ex[inner], top[inner] + .62 * eye_width, ez[inner]])
        knit = forehead * region(glabella, eye_width * np.array([1.05, .91, .9]))
        move("browKnit", knit, eye_width * np.array([-.14 * sign, -.14, .055]))

        # A smile lifts the lower lid and the pad beneath it. Surprise lifts
        # the upper lid. Both release into the separate full-closure blink.
        aperture = np.maximum(upper_lid - lower_lid, .10 * eye_width)
        lower_margin = smooth((upper_lid - y) / aperture)
        upper_margin = smooth((y - lower_lid) / aperture)
        under_eye = eye.copy()
        under_eye[1] = bottom.min() - .08 * eye_width
        lower_pad = lower_margin * region(under_eye, eye_width * np.array([.85, .90, .72]))
        move("cheekSquint", lower_pad, eye_width * np.array([.012 * sign, .090, .045]))
        lid = upper_margin * region(eye, eye_width * np.array([.67, .57, .55]))
        move("eyeWiden", lid, eye_width * np.array([0., .057, .012]))

        # Pull both margins at the commissure together. Spread traction toward
        # the malar pad, leaving the central lips free to close on M/B/P or F/V.
        lateral = smooth((sign * (x - mouth[:, 0].mean()) / width - .28) / .18)
        above_chin = smooth((y - corner[1] + .32 * width) / (.25 * width))
        below_eye = smooth((lower_lid - y) / (.34 * eye_width))
        face = lateral * above_chin * below_eye
        at_corner = face * region(corner, width * np.array([.60, .54, .63]))
        cheek = corner + (eye - corner) * .57 + width * np.array([.24 * sign, 0., -.08])
        at_cheek = face * region(cheek, width * np.array([.69, .93, .67]))
        move("cheekSmile", at_corner, width * np.array([.113 * sign, .145, -.048]))
        move("cheekSmile", at_cheek, width * np.array([.036 * sign, .118, .094]))
        move("mouthFrown", at_corner, width * np.array([-.022 * sign, -.085, .012]))
        move("cheekStretch", at_corner, width * np.array([.073 * sign, .020, -.020]))
        move("cheekStretch", at_cheek, width * np.array([.030 * sign, -.011, -.040]))
        move("cheekPucker", at_corner, width * np.array([-.088 * sign, 0., .085]))
        move("cheekPucker", at_cheek, width * np.array([-.035 * sign, 0., -.060]))
        move("cheekJawOpen", at_cheek, width * np.array([-.016 * sign, -.067, -.025]))
    return result


def add_expression_targets(model, curves):
    # The same field carries detached eyebrow and lash geometry with the skin.
    for node in model.doc["nodes"]:
        if "mesh" not in node or node.get("name") not in {"fitted_head", "source_eyebrows", "source_lashes"}:
            continue
        mesh = model.doc["meshes"][node["mesh"]]
        primitive, = mesh["primitives"]
        positions = attr(model, primitive, "POSITION").astype(float)
        normal = attr(model, primitive, "NORMAL").astype(float)
        fields = expression_fields(positions, curves)
        # Transform the authored normals by the local deformation gradient.
        # This retains smooth shading across coincident vertices at UV seams.
        step = np.ptp(curves["mouth"][0]) * .001
        derivatives = []
        for axis in np.eye(3) * step:
            plus = expression_fields(positions + axis, curves)
            minus = expression_fields(positions - axis, curves)
            derivatives.append({key: (plus[key] - minus[key]) / (2 * step) for key in fields})

        targets = primitive.setdefault("targets", [])
        names = mesh.setdefault("extras", {}).setdefault("targetNames", [])
        weights = mesh.setdefault("weights", [0.] * len(targets))
        for name, displacement in fields.items():
            if not np.any(displacement):
                continue
            jacobian = np.stack([d[name] for d in derivatives], axis=2) + np.eye(3)
            posed = np.linalg.solve(jacobian.transpose(0, 2, 1), normal[:, :, None])[:, :, 0]
            posed /= np.maximum(np.linalg.norm(posed, axis=1, keepdims=True), 1.e-12)
            normal_delta = posed - normal
            normal_delta[~np.any(displacement, axis=1)] = 0.
            for target in targets:
                if "NORMAL" not in target:
                    target["NORMAL"] = add(model, np.zeros_like(normal, dtype="<f4"), "VEC3")
            targets.append({
                "POSITION": add(model, displacement.astype("<f4"), "VEC3"),
                "NORMAL": add(model, normal_delta.astype("<f4"), "VEC3"),
            })
            names.append(name)
            weights.append(0.)
