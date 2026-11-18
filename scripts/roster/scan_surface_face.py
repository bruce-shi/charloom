"""Bind facial morphs on retained scan triangles, preserving their rest surface and UVs."""

from functools import lru_cache

import igl
import numpy as np

from common import add, attr, idx


@lru_cache(None)
def tessellation(divisions):
    pairs = [(i, j) for i in range(divisions + 1) for j in range(divisions + 1 - i)]
    lookup = {pair: index for index, pair in enumerate(pairs)}
    barycentric = np.array(
        [(i / divisions, j / divisions, 1 - (i + j) / divisions) for i, j in pairs]
    )
    faces = []
    for i in range(divisions):
        for j in range(divisions - i):
            faces.append([lookup[i, j], lookup[i + 1, j], lookup[i, j + 1]])
            if i + j + 1 < divisions:
                faces.append([lookup[i + 1, j], lookup[i + 1, j + 1], lookup[i, j + 1]])
    return barycentric, np.asarray(faces)


def expression_fields(points, side, curves):
    """Evaluate the established scan-surface expressions in working metres."""
    x, y, _ = points.T
    mouth_x, mouth_top, mouth_bottom, _ = curves["mouth"]
    seam = np.interp(x, mouth_x, (mouth_top + mouth_bottom) * 0.5)
    width = np.ptp(mouth_x)
    center = (mouth_x[0] + mouth_x[-1]) * 0.5
    u = (x - center) / (width * 0.5)
    v = (seam - y) / width
    lower = side > 0
    result = {}

    def taper(value):
        return np.maximum(0.0, 1.0 - value * value) ** 2

    def move(name, dx=0.0, dy=0.0, dz=0.0):
        values = np.stack(
            np.broadcast_arrays(
                dx + np.zeros_like(x), dy + np.zeros_like(x), dz + np.zeros_like(x)
            ),
            axis=1,
        )
        result.setdefault(name, np.zeros_like(points))[:] += values

    belly = np.maximum(0.0, 1.0 - u * u) ** 0.8
    move(
        "cheekJawOpen",
        dy=-width * 0.43 * belly * np.where(lower, 0.80, -0.20) * taper(v / 1.02),
        dz=-width * 0.038 * belly * taper(v / 0.9),
    )
    mouth = taper(u / 1.60) * taper(v / 0.65)
    move("cheekStretch", dx=width * 0.48 * u * mouth)
    move("cheekPucker", dx=-width * 0.65 * u * mouth, dz=width * 0.18 * mouth)
    cheek = (
        np.minimum(abs(u), 1.0) ** 1.55
        * taper(v / 0.80)
        * taper(np.maximum(abs(u) - 1.0, 0.0) / 0.95)
    )
    move(
        "cheekSmile",
        width * 0.12 * np.sign(u) * cheek,
        width * 0.13 * cheek,
        0.02 * width * cheek,
    )
    move("mouthFrown", -width * 0.04 * np.sign(u) * cheek, -width * 0.15 * cheek)
    move(
        "lipLabiodental",
        dy=width * 0.046 * belly * lower * taper(v / 0.35),
        dz=-width * 0.025 * belly * lower * taper(v / 0.35),
    )
    for key in ("l", "r"):
        xs, upper, bottom, _ = curves[key]
        eye_width = np.ptp(xs)
        eye_center = (xs[0] + xs[-1]) * 0.5
        lateral = (x - eye_center) / eye_width
        top_y = np.interp(x, xs, upper)
        bottom_y = np.interp(x, xs, bottom)
        span = np.maximum(eye_width * 0.025, top_y - bottom_y)
        contact = top_y - 0.73 * span
        # A sampled eyelid margin usually lies inside a triangle. Extend the
        # same closure plane just beyond it so barycentric interpolation of
        # surrounding vertices brings both painted margins into contact.
        support = eye_width * 0.12
        travel = contact - np.clip(y, bottom_y - support, top_y + support)
        outside = np.maximum(bottom_y - y, y - top_y).clip(0)
        influence = taper(outside / (eye_width * 0.48)) * taper(
            np.maximum(abs(lateral) - 0.48, 0.0) / 0.15
        )
        margin = np.maximum(outside - support, 0)
        blink = taper(margin / (eye_width * 0.18)) * taper(
            np.maximum(abs(lateral) - 0.48, 0.0) / 0.15
        )
        move("eyeBlink", dy=travel * blink * 0.997)
        move("eyeWiden", dy=-travel * influence * 0.17)
        move("cheekSquint", dy=travel * influence * 0.17)
        gaze = taper(lateral / 0.52) * taper(
            (y - (top_y + bottom_y) * 0.5) / (0.61 * span + eye_width * 0.025)
        )
        move("gazeLeft", dx=eye_width * 0.13 * gaze)
        move("gazeUp", dy=span * 0.19 * gaze)
        brow_y = np.max(upper) + eye_width * 0.35
        brow = taper((x - eye_center) / (0.66 * eye_width)) * taper(
            (y - brow_y) / (0.32 * eye_width)
        )
        move("browRaise" + key.upper(), dy=eye_width * 0.12 * brow)
        move(
            "browKnit",
            dx=-np.sign(eye_center - center) * eye_width * 0.09 * brow,
            dy=-eye_width * 0.055 * brow,
        )
    return result


def bind_face(model, curves, source_vertices, source_faces, *, observed_faces=None, neck_plane=None):
    node = next(n for n in model.doc["nodes"] if n.get("name") == "fitted_head")
    (primitive,) = model.doc["meshes"][node["mesh"]]["primitives"]
    original = {
        key: attr(model, primitive, key).copy() for key in primitive["attributes"]
    }
    vertices = original["POSITION"].astype(float)
    faces = model.array(primitive["indices"]).reshape(-1, 3).astype(int)
    mouth_x, upper, lower, _ = curves["mouth"]
    mouth_y = (upper + lower) * 0.5
    width = np.ptp(mouth_x)
    eye_x = np.r_[curves["l"][0], curves["r"][0]]
    eye_y = np.r_[curves["l"][1], curves["r"][1]]
    low = np.array([eye_x.min() - width * 0.42, mouth_y.min() - width])
    high = np.array([eye_x.max() + width * 0.42, eye_y.max() + width * 0.8])
    bounds = vertices[faces, :2]
    selected = ((bounds.max(1) >= low) & (bounds.min(1) <= high)).all(1)
    if observed_faces is not None:
        observed_faces = np.asarray(observed_faces, dtype=bool)
        if observed_faces.shape != selected.shape:
            raise ValueError("Source-face ancestry must match the assembled head topology")
        # Animated facial patches follow the scan. Neck bridges and caps
        # remain in the rigid head mesh with their original material.
        selected &= observed_faces
    if not selected.any():
        raise ValueError("No observed scan triangles in the facial binding region")
    rows = np.flatnonzero(selected)
    centers = vertices[faces[rows]].mean(1)
    origins = centers.copy()
    origins[:, 2] = vertices[:, 2].max() + 1
    tree = igl.AABB()
    tree.init(vertices, faces)
    hit, depth, _ = tree.intersect_ray_first(
        vertices, faces, origins, np.tile([0.0, 0.0, -1.0], (len(origins), 1))
    )
    selected[rows] &= (hit >= 0) & (
        abs(origins[:, 2] - depth - centers[:, 2]) < width * 0.60
    )
    source_rows, samples, sides, triangles, seams = [], [], [], [], []

    def append(row, barycentric, side):
        index = len(source_rows)
        source_rows.append(row)
        samples.append(barycentric)
        sides.append(side)
        return index

    for row in np.flatnonzero(selected):
        points = vertices[faces[row]]
        divisions = int(
            np.clip(np.ceil(np.ptp(points[:, :2], axis=0).max() / (width / 22)), 1, 48)
        )
        barycentric, subfaces = tessellation(divisions)
        points = barycentric @ points
        offset = np.interp(points[:, 0], mouth_x, mouth_y) - points[:, 1]
        base = len(source_rows)
        for bary, value in zip(barycentric, offset):
            append(row, bary, 1 if value >= 0 else -1)
        for triangle in subfaces:
            values = offset[triangle]
            xs = points[triangle, 0]
            crossing = (
                values.min() < 0 < values.max()
                and xs.max() >= mouth_x[0]
                and xs.min() <= mouth_x[-1]
            )
            if not crossing:
                triangles.append(triangle + base)
                continue
            edge_vertices = {}
            for sign in (1, -1):
                polygon, boundary = [], []
                for a, b in zip(triangle, np.roll(triangle, -1)):
                    if sign * offset[a] >= 0:
                        polygon.append(a + base)
                    if (offset[a] >= 0) != (offset[b] >= 0):
                        fraction = offset[a] / (offset[a] - offset[b])
                        bary = (
                            barycentric[a] * (1 - fraction) + barycentric[b] * fraction
                        )
                        index = append(row, bary, sign)
                        polygon.append(index)
                        boundary.append(index)
                for j in range(1, len(polygon) - 1):
                    triangles.append([polygon[0], polygon[j], polygon[j + 1]])
                edge_vertices[sign] = boundary
            if len(edge_vertices[1]) == len(edge_vertices[-1]) == 2:
                seams.append([edge_vertices[-1], edge_vertices[1]])
    rows = np.asarray(source_rows)
    barycentric = np.asarray(samples)
    source_triangles = faces[rows]
    interpolate = lambda values: np.einsum(
        "ij,ijk->ik", barycentric, values[source_triangles]
    )
    positions = interpolate(vertices)
    fields = expression_fields(positions, np.asarray(sides), curves)
    if neck_plane is not None:
        normal, offset = neck_plane
        clearance = positions @ normal - offset
        mouth = np.column_stack((curves["mouth"][0], curves["mouth"][1], curves["mouth"][3]))
        width = float(np.min(mouth @ normal - offset)) * .5
        if width <= 0:
            raise ValueError("Face expression boundary crosses the observed lips")
        blend = np.clip(clearance / width, 0, 1)
        for values in fields.values():
            values *= (blend**2 * (3 - 2*blend))[:, None]
    bone_count = len(model.doc["skins"][node["skin"]]["joints"])
    bone_weights = np.zeros((len(rows), bone_count))
    for corner in range(3):
        joints = original["JOINTS_0"][source_triangles[:, corner]]
        weights = (
            original["WEIGHTS_0"][source_triangles[:, corner]]
            * barycentric[:, corner, None]
        )
        np.add.at(bone_weights, (np.arange(len(rows))[:, None], joints), weights)
    joints = np.argsort(-bone_weights, axis=1, kind="stable")[:, :4]
    weights = np.take_along_axis(bone_weights, joints, axis=1)
    weights /= weights.sum(1, keepdims=True)
    attributes = {
        key: interpolate(value)
        for key, value in original.items()
        if key not in ("JOINTS_0", "WEIGHTS_0")
    }
    attributes.update(POSITION=positions, JOINTS_0=joints, WEIGHTS_0=weights)
    triangles = np.asarray(triangles)
    primitive["indices"] = idx(model, faces[~selected])
    primitive.pop("targets", None)
    mesh = model.doc["meshes"][node["mesh"]]
    mesh.pop("weights", None)
    mesh.get("extras", {}).pop("targetNames", None)

    def put(name, data, topology, targets, material):
        output = {}
        for key, values in data.items():
            joint = key.startswith("JOINTS_")
            output[key] = add(
                model,
                values.astype("<u2" if joint else "<f4"),
                "VEC" + str(values.shape[1]),
                5123 if joint else 5126,
            )
        shape_names = list(targets)
        shape_targets = [
            {"POSITION": add(model, targets[key].astype("<f4"), "VEC3")}
            for key in shape_names
        ]
        mesh_id = len(model.doc["meshes"])
        model.doc["meshes"].append(
            dict(
                name=name,
                primitives=[
                    dict(
                        attributes=output,
                        indices=idx(model, topology),
                        material=material,
                        targets=shape_targets,
                    )
                ],
                extras=dict(targetNames=shape_names),
                weights=[0.0] * len(shape_targets),
            )
        )
        node_id = len(model.doc["nodes"])
        model.doc["nodes"].append(
            {
                **node,
                "name": name,
                "mesh": mesh_id,
                "weights": [0.0] * len(shape_targets),
            }
        )
        original_id = model.doc["nodes"].index(node)
        parent = next(
            (n for n in model.doc["nodes"] if original_id in n.get("children", [])),
            None,
        )
        children = (
            parent["children"]
            if parent
            else model.doc["scenes"][model.doc.get("scene", 0)]["nodes"]
        )
        children.append(node_id)

    put("FaceSurface", attributes, triangles, fields, primitive["material"])
    # Bind the oral lining to the actual split edge, including its local depth.
    seams = np.asarray(seams)
    if not len(seams):
        raise ValueError("The traced lip seam does not split the retained scan surface")
    seam_points = positions[seams[:, 0]]
    lip_upper, lip_lower, lip_mix = [], [], []
    for x in np.linspace(mouth_x[0], mouth_x[-1], 65):
        x = np.clip(
            x, seam_points[:, :, 0].min() + 1.0e-7, seam_points[:, :, 0].max() - 1.0e-7
        )
        eligible = np.flatnonzero(
            (seam_points[:, :, 0].min(1) <= x) & (seam_points[:, :, 0].max(1) >= x)
        )
        segment = seam_points[eligible]
        ratio = np.clip(
            (x - segment[:, 0, 0]) / (segment[:, 1, 0] - segment[:, 0, 0] + 1.0e-20),
            0,
            1,
        )
        point = segment[:, 0] * (1 - ratio[:, None]) + segment[:, 1] * ratio[:, None]
        front = point[:, 2].argmax()
        lip_upper.append(seams[eligible[front], 0])
        lip_lower.append(seams[eligible[front], 1])
        lip_mix.append([1 - ratio[front], ratio[front]])
    lip_upper, lip_lower, lip_mix = (
        np.asarray(lip_upper),
        np.asarray(lip_lower),
        np.asarray(lip_mix),
    )

    def edge_values(values, indices):
        return np.einsum("ij,ijk->ik", lip_mix, values[indices])

    lip = edge_values(positions, lip_upper)
    layers = np.array([0.0, 0.10, 0.18, 0.185, 0.75, 1.0])
    oral_positions = np.repeat(lip, len(layers), axis=0)
    layer = np.tile(layers, len(lip))
    oral_positions[:, 2] -= 0.0013 + width * 0.11 * np.sin(layer * np.pi)
    oral_fields = {}
    for key, values in fields.items():
        up = np.repeat(edge_values(values, lip_upper), len(layers), axis=0)
        down = np.repeat(edge_values(values, lip_lower), len(layers), axis=0)
        oral_fields[key] = up * (1 - layer[:, None]) + down * layer[:, None]
    oral_dense = np.repeat(edge_values(bone_weights, lip_upper), len(layers), axis=0)
    oral_joints = np.argsort(-oral_dense, axis=1, kind="stable")[:, :4]
    oral_weights = np.take_along_axis(oral_dense, oral_joints, axis=1)
    oral_weights /= oral_weights.sum(1, keepdims=True)
    oral_faces = []
    for ring in range(len(lip) - 1):
        for section in range(len(layers) - 1):
            a = ring * len(layers) + section
            oral_faces.extend(
                [
                    [a, a + 1, a + len(layers)],
                    [a + len(layers), a + 1, a + len(layers) + 1],
                ]
            )
    colors = np.array(
        [
            [0.060, 0.009, 0.015],
            [0.46, 0.36, 0.29],
            [0.40, 0.31, 0.27],
            [0.042, 0.006, 0.011],
            [0.086, 0.013, 0.022],
            [0.16, 0.027, 0.041],
        ]
    )
    oral_material = len(model.doc["materials"])
    model.doc["materials"].append(
        dict(
            name="Mouth interior",
            doubleSided=True,
            pbrMetallicRoughness=dict(metallicFactor=0, roughnessFactor=1),
        )
    )
    put(
        "MouthInterior",
        dict(
            POSITION=oral_positions,
            NORMAL=np.tile([0.0, 0.0, 1.0], (len(layer), 1)),
            COLOR_0=np.tile(colors, (len(lip), 1)),
            JOINTS_0=oral_joints,
            WEIGHTS_0=oral_weights,
        ),
        np.asarray(oral_faces),
        oral_fields,
        oral_material,
    )
    controls = {
        "binding": "scan_surface",
        "faceMesh": "FaceSurface",
        "collarMesh": "fitted_head",
        "mouth": {"upper": [], "lower": []},
    }
    for label, indices in [("upper", lip_upper), ("lower", lip_lower)]:
        controls["mouth"][label] = [
            dict(vertices=ids.tolist(), weights=mix.tolist())
            for ids, mix in zip(indices, lip_mix)
        ]
    for side in ("l", "r"):
        xs, top, bottom, depth = curves[side]
        center = len(xs) // 2
        probes = np.array(
            [
                [xs[center], top[center], depth[center]],
                [xs[center], bottom[center], depth[center]],
            ]
        )
        _, selected_faces, closest = igl.point_mesh_squared_distance(
            probes, positions, triangles
        )
        selected_triangles = triangles[selected_faces]
        bary = igl.barycentric_coordinates(
            closest, *positions[selected_triangles].transpose(1, 0, 2)
        )
        controls[side] = {
            label: [dict(vertices=ids.tolist(), weights=mix.tolist())]
            for label, ids, mix in zip(("upper", "lower"), selected_triangles, bary)
        }
    distance2, _, _ = igl.point_mesh_squared_distance(
        positions, source_vertices, source_faces
    )
    deviation = float(np.sqrt(distance2.max()) * 1000)
    if deviation > 0.05:
        raise ValueError(
            f"Retained facial surface differs from the source by {deviation:.4f} mm"
        )
    controls["sourceSurface"] = dict(
        selected_triangles=int(selected.sum()),
        generated_triangles=len(triangles),
        neutral_max_distance_mm=deviation,
    )
    return controls
