"""Carry source-triangle corner attributes through head cuts and UV seams."""

import igl
import numpy as np


def map_corners(
    vertices,
    faces,
    ancestors,
    source_positions,
    source_faces,
    source_uv,
    source_normals,
    generated_normals,
    colors,
    atlas_width,
    collar_uv=None,
):
    """Keep the original vertex indices for controls, duplicating only corners with different attributes."""
    inherited = ancestors >= 0
    uv = np.zeros((*faces.shape, 2), dtype=np.float32)
    uv[..., 0] = (atlas_width + 7.5) / (atlas_width + 16)
    normal = generated_normals[faces].astype(np.float32)
    tint = colors[faces].astype(np.float32)
    triangles = source_faces[ancestors[inherited]]
    points = source_positions[triangles]
    query = vertices[faces[inherited]].reshape(-1, 3)
    bary = igl.barycentric_coordinates(
        query, *(a.repeat(3, axis=0) for a in points.transpose(1, 0, 2))
    )
    weights = bary.reshape(-1, 3, 3)
    if not np.isfinite(weights).all():
        raise ValueError("Degenerate source triangle in head texture correspondence")
    tex = weights @ source_uv[triangles]
    tex[..., 0] *= atlas_width / (atlas_width + 16)
    uv[inherited] = tex
    n = weights @ source_normals[triangles]
    normal[inherited] = n / np.maximum(
        np.linalg.norm(n, axis=2, keepdims=True), 1.0e-12
    )
    tint[inherited] = 1
    if collar_uv is not None:
        collar, projection = collar_uv
        uv[collar] = projection[faces[collar]]
        uv[collar, :, 0] *= atlas_width / (atlas_width + 16)
        tint[collar] = 1
    # Rounding just absorbs barycentric error at shared edges in the float32 solid.
    rows = np.column_stack(
        (
            faces.ravel(),
            uv.reshape(-1, 2).round(6),
            normal.reshape(-1, 3).round(5),
            tint.reshape(-1, 3).round(6),
        )
    )
    unique, inv = np.unique(rows, axis=0, return_inverse=True)
    ids = unique[:, 0].astype(int)
    first = np.r_[True, ids[1:] != ids[:-1]]
    if not np.array_equal(ids[first], np.arange(len(vertices))):
        raise ValueError("Unreferenced head vertices in texture assembly")
    order = np.r_[np.flatnonzero(first), np.flatnonzero(~first)]
    remap = np.argsort(order)
    return (
        ids[order],
        remap[inv].reshape(-1, 3),
        *(unique[order, i:j] for i, j in ((1, 3), (3, 6), (6, 9))),
    )


def inside_source_parts(config, source_hash, face_count):
    if config["source_hash"] != source_hash:
        raise ValueError("Facial surface review belongs to a different source")
    if ("upper_lip" in config) != ("lower_lip" in config):
        raise ValueError("Painted smiles require both observed lip margins")
    parts = {}
    owners = np.zeros(face_count, bool)
    for name, ranges in config.get("parts", {}).items():
        mask = np.zeros(face_count, bool)
        for start, end in ranges:
            if not 0 <= start < end <= face_count:
                raise ValueError(f"Invalid source face range for {name}")
            mask[start:end] = True
        if not mask.any() or (owners & mask).any():
            raise ValueError(f"Empty or overlapping facial source selection for {name}")
        owners |= mask
        parts[name] = np.flatnonzero(mask)
    return parts


def fit_painted_globe(xy, surface_z, radius, skin, faces, tree):
    """Seat a sphere inside the uncut skin so its envelope is invariant under gaze."""
    # Search posteriorly from the observed face. Checking the nearest triangle
    # over the whole sphere also covers the thinner skin beyond the canthi.
    clearance = radius * .045
    required = (radius + clearance) ** 2
    depths = surface_z - radius * np.linspace(0., 3., 61)
    def distances(depths):
        points = np.column_stack((np.tile(xy, (len(depths), 1)), depths))
        return tree.squared_distance(skin, faces, points)[0]
    inside = np.flatnonzero(distances(depths) >= required)
    if not len(inside) or inside[0] == 0:
        raise ValueError("Cannot seat a rotating eye behind the measured facial surface")
    i = inside[0]
    front, back = depths[i - 1], depths[i]
    for _ in range(14):
        middle = (front + back) * .5
        if distances([middle])[0] >= required:
            back = middle
        else:
            front = middle
    return np.r_[xy, back]


def observed_globe(x, upper, lower, front, xy, radius):
    """Estimate the true pivot of a curved painted eye facing out to the side."""
    if np.ptp(front(x, (upper + lower) * .5)) < .4 * np.ptp(x):
        return xy, radius
    t = np.linspace(.08, .92, 9)[:, None]
    y = (upper * t + lower * (1 - t)).ravel()
    x = np.tile(x, len(t))
    points = np.c_[x, y, front(x, y)]
    origin = points.mean(0)
    q = points - origin
    fit = np.linalg.lstsq(np.c_[2*q, np.ones(len(q))], (q*q).sum(1), rcond=None)[0]
    center = origin + fit[:3]
    distances = np.linalg.norm(points - center, axis=1)
    fitted_radius = float(distances.mean())
    if (center[2] < points[:, 2].min() and radius * .75 < fitted_radius < radius * 2
            and np.quantile(abs(distances - fitted_radius), .95) < .08 * fitted_radius):
        return center[:2], fitted_radius
    return xy, radius


def painted_eye(center, radius, arc, surface, image, sphere):
    """Transfer the visible painted eye onto its own gaze-driven globe at full texture resolution."""
    xs, upper, lower, _ = arc
    size = 384
    axis = (np.arange(size) + 0.5) / size * 2 - 1
    x, y = np.meshgrid(center[0] + axis * radius, center[1] + axis * radius)

    def opening(x, y):
        return np.minimum.reduce(
            [
                x - xs[0],
                xs[-1] - x,
                np.interp(x, xs, upper) - y,
                y - np.interp(x, xs, lower),
            ]
        )

    distance = opening(x, y)
    mask = distance >= 0
    _, uv = surface(x[mask], y[mask])
    from scipy.ndimage import map_coordinates

    pixels = np.column_stack(
        [
            map_coordinates(
                image[..., k],
                (uv[:, 1] * image.shape[0] - 0.5, uv[:, 0] * image.shape[1] - 0.5),
                order=1,
                mode="nearest",
            )
            for k in range(3)
        ]
    )
    # Extend the colour of the visible sclera around the globe so rotating the
    # iris exposes a consistent surface under the stationary eyelids.
    sclera = np.percentile(pixels, 86, axis=0)
    patch = np.broadcast_to(sclera, (size, size, 3)).astype(np.uint8).copy()
    patch[mask] = pixels
    return sphere[:, :2] * 0.5 + 0.5, patch


def support_chin(positions, weights, mouth, bite_pitch=0.):
    """Blend lip motion into the commissures and rotate the chin with the lower arch."""
    xs, top, bottom, z = mouth
    width = np.ptp(xs)
    u = np.abs(positions[:, 0] - (xs[0] + xs[-1]) * 0.5) / (width * 0.5)
    # Both lips meet at each commissure. Their opening must taper together.
    # Preserve the four joint slots returned by the volume face binder.
    taper = np.clip((u - 0.30) / 0.82, 0, 1)
    retained = 1 - 0.88 * taper**2 * (3 - 2 * taper)
    weights[:, 1:] *= retained[:, None]
    weights[:, 0] = 1 - weights[:, 1:].sum(axis=1)

    def smooth(x):
        t = np.clip(x, 0, 1)
        return t * t * (3 - 2 * t)

    x = positions[:, 0]
    below = np.interp(x, xs, (top + bottom) * 0.5) - positions[:, 1]
    behind = np.interp(x, xs, z) - positions[:, 2]
    below += bite_pitch * behind
    chin = (
        smooth(below / (0.30 * width))
        * smooth((1.35 * width - below) / (0.50 * width))
        * smooth((2.1 - u) / 0.95)
        * smooth((0.94 * width - behind) / (0.49 * width))
    )
    follow = 0.88 * chin * weights[:, 0]
    weights[:, 0] -= follow
    weights[:, 3] += follow


def seat_dental_arches(positions, joints, weights, head_index, mouth):
    """Seat the fitted incisor edges behind the measured lips of a painted smile."""
    xs, top, bottom, _ = mouth
    mid = (xs[0] + xs[-1]) / 2
    width = np.ptp(xs)
    central = np.abs(positions[:, 0] - mid) < width * 0.13
    upper = joints[np.arange(len(joints)), weights.argmax(axis=1)] == head_index
    shifts = []
    for arch, curve, direction in [(upper, top, 1), (~upper, bottom, -1)]:
        samples = positions[arch & central]
        if not len(samples):
            raise ValueError("No fitted incisors near the measured mouth centre")
        anterior = samples[samples[:, 2] > samples[:, 2].max() - width * 0.10]
        edge = anterior[:, 1].min() if direction == 1 else anterior[:, 1].max()
        # At rest the lower incisal edge sits just behind the upper teeth.
        target = np.interp(mid, xs, curve) - width * (
            0.055 if direction == 1 else 0.085
        )
        shifts.append(float(target - edge))
        positions[arch, 1] += shifts[-1]
    return shifts


def lid_contacts(vertices, faces, eye_vertices, eye_faces, arc):
    """Sample the front-view boundary between actual skin and visible eyeball."""
    xs, top, bottom, _ = arc
    width = np.ptp(xs)
    x = np.linspace(xs[0], xs[-1], 65)
    y = np.linspace(bottom.min() - width * 0.12, top.max() + width * 0.12, 513)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    origins = (
        np.c_[xx.ravel(), yy.ravel(), np.full(xx.size, max(vertices[:, 2]) + 1.0)]
        * 1000
    )
    directions = np.tile([0.0, 0.0, -1.0], (len(origins), 1))

    def hits(v, f):
        tree = igl.AABB()
        points = np.ascontiguousarray(v * 1000, dtype=float)
        tree.init(points, f)
        return tree.intersect_ray_first(points, f, origins, directions)

    skin, depth, bary = hits(vertices, faces)
    eye, eye_depth, _ = hits(eye_vertices, eye_faces)
    visible = ((eye >= 0) & ((skin < 0) | (eye_depth < depth))).reshape(xx.shape)
    valid = np.where(visible.sum(axis=1) >= 4)[0]
    if len(valid) < len(x) // 3:
        raise ValueError(
            "The reviewed eye is insufficiently visible through its socket"
        )
    low = np.argmax(visible[valid], axis=1) - 1
    high = len(y) - np.argmax(visible[valid, ::-1], axis=1)
    samples = valid[:, None] * len(y) + np.column_stack((high, low))
    if low.min() < 0 or high.max() >= len(y) or (skin[samples] < 0).any():
        raise ValueError("Could not identify the visible skin around the reviewed eye")
    bc = bary[samples]
    weights = np.concatenate([1 - bc.sum(axis=2, keepdims=True), bc], axis=2)
    tris = skin[samples]
    corners = faces[tris]
    pts = (weights[..., None] * vertices[corners]).sum(axis=2)
    return {
        "x": x[valid],
        "triangles": tris,
        "corners": corners,
        "weights": weights,
        "points": pts,
    }


def insert_lid_controls(v, f, ancestors, contact):
    """Place midpoint controls on the observed lid surfaces, even on a sparse triangulation."""
    mid = np.argmin(np.abs(contact["x"] - np.mean(contact["x"][[0, -1]])))
    ids = []
    for tri, point in zip(contact["triangles"][mid], contact["points"][mid]):
        a, b, c = f[tri]
        i = len(v)
        v = np.vstack((v, point))
        f[tri] = [a, b, i]
        f = np.vstack((f, [b, c, i], [c, a, i]))
        ancestors = np.r_[ancestors, ancestors[tri], ancestors[tri]]
        ids.append(i)
    return v, f, ancestors, ids, mid


def close_authored_lids(v, delta, arc, contact):
    """Close the two measured margins along their full visible width."""
    xs, upper, lower, _ = arc
    width = np.ptp(xs)
    use = (v[:, 0] > xs[0] - width * 0.19) & (v[:, 0] < xs[-1] + width * 0.19)
    # Stop eyelid skin motion before the eyebrow. Keep the orbital crease.
    dist = np.maximum(
        v[:, 1] - np.interp(v[:, 0], xs, upper), np.interp(v[:, 0], xs, lower) - v[:, 1]
    )
    blend = np.clip((width * 0.47 - dist) / (width * 0.29), 0, 1)
    delta[use] *= (blend[use] ** 2 * (3 - 2 * blend[use]))[:, None]
    motion = (contact["weights"][..., None] * delta[contact["corners"]]).sum(axis=2)
    gap = contact["points"][:, 0, 1] - contact["points"][:, 1, 1]
    travel = motion[:, 1, 1] - motion[:, 0, 1]
    valid = (gap > width * 0.07) & (travel > width * 0.05)
    if not valid.any():
        raise ValueError("No measurable closure of the reviewed eyelid margins")
    factor = np.clip(gap[valid] / travel[valid], 0.65, 1.35)
    delta[use, 1] *= np.interp(v[use, 0], contact["x"][valid], factor)
    # Scaling at vertices does not commute with barycentric interpolation on
    # sparse lid triangles. Solve the residual along both measured margins.
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr

    weights = contact["weights"] * np.array([1., -1.])[None, :, None]
    closure = coo_matrix(
        (weights.ravel(), (np.repeat(np.arange(len(gap)), 6), contact["corners"].ravel())),
        shape=(len(gap), len(v)),
    ).tocsr()
    residual = gap + closure @ delta[:, 1]
    correction = lsqr(closure, -residual, damp=.001, atol=1.e-10, btol=1.e-10, iter_lim=500)[0]
    delta[:, 1] += correction


def neck_albedo(
    g, body, vertices, faces, collar, source, source_faces, source_uv, source_image,
    *, source_shell=None, moved=None
):
    """Rebake the translated body port from observed neck skin at its final position."""
    import io

    from common import attr
    from PIL import Image
    from scipy.ndimage import map_coordinates

    mat = g.doc["materials"][body["material"]]["pbrMetallicRoughness"]
    tex = g.doc["textures"][mat["baseColorTexture"]["index"]]
    image = g.doc["images"][tex["source"]]
    view = g.doc["bufferViews"][image["bufferView"]]
    off = view.get("byteOffset", 0)
    atlas = np.array(
        Image.open(io.BytesIO(g.blob[off : off + view["byteLength"]])).convert("RGB")
    )
    height, width = atlas.shape[:2]
    uv = attr(g, body, "TEXCOORD_0") * [width, height] - 0.5
    ring = vertices[collar]
    breadth = np.ptp(ring[:, 0])
    floor = ring[:, 1].min() - breadth * 0.38
    collar_band = ((vertices[faces, 1].max(1) > floor)
                   & (np.linalg.norm(vertices[faces].mean(1)[:, [0, 2]] - ring.mean(0)[[0, 2]], axis=1) < breadth))
    selected = faces[collar_band | (moved[faces].any(1) if moved is not None else False)]
    texels, locations = [], []
    for tri in selected:
        points = uv[tri]
        low = np.maximum(np.floor(points.min(0)).astype(int), 0)
        high = np.minimum(np.ceil(points.max(0)).astype(int), [width - 1, height - 1])
        x, y = np.meshgrid(
            np.arange(low[0], high[0] + 1), np.arange(low[1], high[1] + 1)
        )
        pixels = np.c_[x.ravel(), y.ravel()]
        basis = (points[1:] - points[0]).T
        if abs(np.linalg.det(basis)) < 1.0e-10:
            continue
        bary = (pixels - points[0]) @ np.linalg.inv(basis).T
        bary = np.c_[1 - bary.sum(1), bary]
        inside = bary.min(1) >= -1.0e-7
        texels.append(pixels[inside])
        locations.append(bary[inside] @ vertices[tri])
    if not locations:
        return
    px, py = np.concatenate(texels).T
    where = np.concatenate(locations)
    distance, hits, closest = igl.point_mesh_squared_distance(
        where, source, source_faces
    )
    tri = source_faces[hits]
    bc = igl.barycentric_coordinates(closest, *(source[tri[:, k]] for k in range(3)))
    uv = np.einsum("ij,ijk->ik", bc, source_uv[tri])
    exterior = np.zeros(len(where), bool)
    if source_shell is not None and source_shell[3] is not None:
        tree, mm, triangles, exposed = source_shell
        radial = where - ring.mean(0)
        radial[:, 1] = 0
        radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
        reach = np.linalg.norm(np.ptp(mm, axis=0))
        hit, travel, bary = tree.intersect_ray_first(mm, triangles, where * 1000 + radial * reach, -radial)
        exterior = (hit >= 0) & exposed[hit.clip(min=0)] & (np.abs(travel - reach) < breadth * 500)
        uv[exterior] = np.einsum("ij,ijk->ik",
            np.column_stack([1 - bary[exterior].sum(1), bary[exterior]]), source_uv[triangles[hit[exterior]]])
    sampled = np.column_stack(
        [
            map_coordinates(
                source_image[..., i],
                (
                    uv[:, 1] * source_image.shape[0] - 0.5,
                    uv[:, 0] * source_image.shape[1] - 0.5,
                ),
                order=1,
                mode="nearest",
            )
            for i in range(3)
        ]
    )
    fade = np.clip((where[:, 1] - floor) / (breadth * 0.24), 0, 1)
    fade *= np.clip(1 - np.sqrt(distance) / (breadth * 0.18), 0, 1)
    fade[exterior] = 1.0
    atlas[py, px] = np.rint(
        atlas[py, px] + fade[:, None] * (sampled.astype(float) - atlas[py, px])
    )
    buf = io.BytesIO()
    Image.fromarray(atlas).save(buf, format="PNG")
    data = buf.getvalue()
    g.blob.extend(bytes(-len(g.blob) % 4))
    image["bufferView"] = len(g.doc["bufferViews"])
    image["mimeType"] = "image/png"
    g.doc["bufferViews"].append(
        {"buffer": 0, "byteOffset": len(g.blob), "byteLength": len(data)}
    )
    g.blob.extend(data)
