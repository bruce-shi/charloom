"""Assemble reconstructed animal and fantasy heads on the retained humanoid rigs."""

import argparse, copy, io, json, os, sys
from pathlib import Path
import igl
import numpy as np
import triangle
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree
from PIL import Image
from common import (
    G,
    add,
    attr,
    batch,
    dst,
    finish,
    idx,
    normals,
    material,
    prim,
    place_eye_joint,
    root,
    zipper,
    texture,
)
import manifold3d as manifold
from face_atlas import (inside_source_parts, map_corners, fit_painted_globe, observed_globe, painted_eye, seat_dental_arches,
                        support_chin, lid_contacts, insert_lid_controls, close_authored_lids, neck_albedo)
from facial_expressions import add_expression_targets
import observed_binding
import volume_body
from muzzle import bite_plane, fit_oral_mesh
from observed_volume import collar_cut, fair_collar_band, follow_neck_surface, head_solid, observed_neck, stitch_neck


def solid(v, f):
    m = manifold.Manifold(
        manifold.Mesh(
            np.ascontiguousarray(v, np.float32), np.ascontiguousarray(f, np.uint32)
        )
    )
    if m.status() != manifold.Error.NoError:
        raise ValueError(str(m.status()))
    return m


def cutter(x, y0, y1, z, back, front, muzzle=None):
    # A narrow opening at the surface, with room for both arches farther back.
    mid = (y0 + y1) * .5
    width = np.ptp(x)
    stations = [back, -front] if muzzle is None else [-front, 0., .04*width, .11*width, .3*width, back]
    rings = []
    for inset in stations:
        pitch = muzzle[0] if muzzle is not None else 0.
        roof, floor = y0 + pitch * inset, y1 + pitch * inset
        if muzzle is not None and inset > 0:
            origin = np.c_[x, mid + pitch * inset, z - inset]
            taper = np.sin(np.linspace(0, np.pi, len(x)))
            room = min(.16 * width, inset * .85) * taper
            for rim, sign in ((roof, 1.), (floor, -1.)):
                directions = np.zeros_like(origin)
                directions[:, 1] = sign
                distance = muzzle[1](origin, directions)
                rim += sign * np.minimum(room, .62 * distance)
        rings.append(np.vstack([np.c_[x, roof, z - inset],
                                np.c_[x, floor, z - inset][-2:0:-1]]))
    n = len(rings[0])
    v = np.vstack(rings)
    side = []
    for i in range(len(rings) - 1):
        side.extend(zipper(np.arange(n) + i*n, np.arange(n) + (i+1)*n))
    if muzzle is not None:
        # A cap across successive sections follows the curve of a long muzzle.
        caps = []
        for i in range(len(x) - 1):
            left, right = (-i) % n, (-(i + 1)) % n
            if i != left:
                caps.append([i, left, i + 1])
            if i + 1 != right:
                caps.append([left, right, i + 1])
    else:
        caps = [[0, i, i + 1] for i in range(1, n - 1)]
    side.extend(caps)
    side.extend([[len(v) - n + k for k in reversed(face)] for face in caps])
    f, _ = igl.bfs_orient(np.array(side))
    s = solid(v, f)
    return solid(v, f[:, ::-1]) if s.volume() < 0 else s


def trim_replaced_grafts(g, count, source_uv, source_faces, owned, extra_names=()):
    """Drop scaffold patches already represented by the assembled source head."""
    if not count and not extra_names:
        return {}
    graft_names = {"graft" if k == 0 else f"graft_{k}" for k in range(count)} | set(extra_names)
    nodes = [node for node in g.doc["nodes"] if node.get("name") in graft_names and "mesh" in node]
    if not nodes:
        return {}
    # The staged exporter preserves the source corner order and UVs even when
    # it pushes graft positions away from the body. Some atlases reuse UVs;
    # keep a triangle if any matching source face belongs to another surface.
    uv_faces = np.ascontiguousarray(source_uv[source_faces], dtype="<f4")
    def keys(rows):
        # One sortable byte string per triangle, three pairs of float32 UVs.
        return rows.reshape(-1, 6).view(np.dtype((np.void, 6 * rows.itemsize))).ravel()
    source_keys = keys(uv_faces)
    replaced = np.setdiff1d(source_keys[owned], source_keys[~owned])
    removed = {}
    for node in nodes:
        mesh = g.doc["meshes"][node["mesh"]]
        retained = []
        for pr in mesh["primitives"]:
            faces = g.array(pr["indices"]).reshape(-1, 3)
            uv = np.ascontiguousarray(attr(g, pr, "TEXCOORD_0")[faces], dtype="<f4")
            drop = np.isin(keys(uv), replaced)
            if drop.any():
                removed[node["name"]] = removed.get(node["name"], 0) + int(drop.sum())
            if not drop.all():
                if drop.any():
                    pr["indices"] = idx(g, faces[~drop])
                retained.append(pr)
        if retained:
            mesh["primitives"] = retained
        else:
            node.pop("mesh")
            node.pop("skin", None)
            node.pop("weights", None)
    return removed


def run(n, character_config=None):
    out = dst / n
    r = batch / n
    out.mkdir(parents=True, exist_ok=True)
    character_config = character_config or {}
    surface_binding = character_config.get("face_binding") == "scan_surface"
    volume = character_config.get("raw_head")
    if surface_binding and volume is not None:
        raise ValueError("Scan surface binding requires the original scan")
    cfg = json.loads((r / "report.json").read_text())
    reference = None
    archive = os.getenv("CHARLOOM_VOLUME_DIR")
    if archive and volume is None and (root / "scripts/roster/inputs/roster.json").is_file():
        roster = json.loads((root / "scripts/roster/inputs/roster.json").read_text())
        reference = next((entry.get("prepared_volume") for entry in roster["characters"]
                          if entry["name"] == n and (root / entry["source"]).resolve()
                          == (root / cfg["source"]).resolve()), None)
        if reference:
            volume = Path(archive) / reference["directory"] / "raw-head.npz"
    old = Path(os.getenv("CHARLOOM_NATIVE_DIR", batch / "final")) / n
    selected_body = None
    if volume is None:
        old = r
        selection = r / "body-selection.json"
        if selection.is_file():
            selected_body = Path(json.loads(selection.read_text())["candidate"])

    def base_stage(rel):
        g = G(old / rel)
        if selected_body is not None:
            correction = G(selected_body / rel)
            original, corrected = prim(g, "body"), prim(correction, "body")
            if (not np.array_equal(attr(g, original, "TEXCOORD_0"), attr(correction, corrected, "TEXCOORD_0"))
                    or not np.array_equal(g.array(original["indices"]), correction.array(corrected["indices"]))):
                raise ValueError("Selected body topology differs from transfer")
            for key in ("POSITION", "NORMAL"):
                original["attributes"][key] = add(g, attr(correction, corrected, key).copy(), "VEC3")
        return g

    resolved = json.load(open(r / "0-guide/resolved.json"))
    marks = resolved["landmarks"]
    src = G(cfg["source"])
    mesh_nodes = [i for i in src.doc["nodes"] if "mesh" in i]
    node = mesh_nodes[0]
    primitives = src.doc["meshes"][node["mesh"]]["primitives"]
    if volume is None and (len(mesh_nodes) != 1 or len(primitives) != 1):
        raise ValueError("Scan volume assembly requires the static single-atlas scan used by the transfer")
    sp = primitives[0]
    from transfer.scan import transforms
    source_world, _ = transforms(src.doc)
    M = np.array(cfg["working_frame"]["source_to_working"]).reshape(4,4).T @ source_world[src.doc["nodes"].index(node)]
    sv = attr(src, sp, "POSITION").astype(float) @ M[:3, :3].T + M[:3, 3]
    sf = src.array(sp["indices"]).reshape(-1, 3).astype(int)
    uv0 = attr(src, sp, "TEXCOORD_0")
    # Cut on the anatomical prior instead of removing shoulders that rise above a low creature neck.
    bare = base_stage("3-skin/body.glb")
    body = prim(bare, "body")
    prior = G(cfg["prior"])
    pp = prior.body()
    j0 = attr(bare, body, "JOINTS_0")
    w0 = attr(bare, body, "WEIGHTS_0")
    bp = attr(bare, body, "POSITION")
    bf = bare.array(body["indices"]).reshape(-1, 3)
    du, ii = cKDTree(attr(prior, pp, "TEXCOORD_0")).query(
        attr(bare, body, "TEXCOORD_0")
    )
    rest = attr(prior, pp, "POSITION")[ii]
    body_config = character_config.get("body_volume")
    if body_config and volume is not None:
        raise ValueError("A volume body requires the observed scan-head route")
    if not body_config:
        tt, li, weld = observed_binding.body_collar(rest, bf)

    # New assemblies retain the observed texture, including painted eyes.
    # An archived resampled volume has no original triangle ancestry.
    face_atlas = character_config.get("face_atlas", {"source_hash": cfg["source_hash"]} if volume is None else None)
    parts = {}
    if face_atlas is not None:
        if volume is not None:
            raise ValueError("The textured observed face requires the original scan triangles")
        parts = inside_source_parts(face_atlas, cfg["source_hash"], len(sf))
    observed = None
    if volume is None:
        # An authored upper body can end under clothing in open, branching
        # boundaries. Only the connected surface around the head and neck
        # participates in the head solid.
        floor = observed_neck(resolved)[1] if "skin" in parts and not body_config else None
        if floor is not None and "plane_y_m" in character_config:
            # A shirt can cover the back of the neck higher than its exposed
            # front. Keep the observed skin below the chosen cut for capping.
            floor = min(floor, character_config["plane_y_m"] - .018)
        m, observed, preparation = head_solid(sv, sf, marks, parts, floor)
    else:
        with np.load(volume) as data:
            m = max(solid(data["positions"], data["triangles"]).decompose(), key=lambda a: a.volume())
        if m.volume() <= 0.001:
            raise ValueError(f"Prepared head solid is too small: {m.volume()} cubic metres")
        preparation = {"method": "prepared_volume", "file": str(volume)}
    plane = character_config.get("plane_y_m", (reference or {}).get("plane_y_m"))
    measured_collar = not body_config and (plane is None or (volume is None and "skin" in parts))
    if measured_collar and plane is None:
        plane = collar_cut(bp[li], resolved)
        # On welded scans a high shirt collar belongs to the same component
        # as the face. Start the head on exposed neck so the shirt comes off
        # with the dressed stage. Measure the actual segmented piece in place.
        if face_atlas and preparation.get("components") == 1 and "skin" not in parts:
            neck = np.asarray(observed_neck(resolved))
            radius = np.max(np.linalg.norm((bp[li] - neck)[:, [0, 2]], axis=1))
            margin = np.median(np.linalg.norm(bp[li] - np.roll(bp[li], 1, axis=0), axis=1))
            mouth_floor = min(marks["mouth_l"][1], marks["mouth_r"][1])
            for path in (r / "1-segment/pieces").glob("*.glb"):
                piece = G(path)
                surface = attr(piece, piece.doc["meshes"][0]["primitives"][0], "POSITION")
                near = np.linalg.norm((surface - neck)[:, [0, 2]], axis=1) < radius
                heights = surface[near, 1]
                if len(heights) and heights.min() < plane < heights.max() < mouth_floor - margin:
                    plane = float(heights.max() + margin)
    cut_normal = np.array([0., 1., 0.])
    cut_offset = plane
    if "neck_plane_points_m" in character_config:
        # Three reviewed points on the exposed neck mark the actual sloping
        # collar boundary in working space (left, right and posterior).
        anchors = np.asarray(character_config["neck_plane_points_m"], dtype=float)
        if body_config or not measured_collar or anchors.shape != (3, 3) or not np.isfinite(anchors).all():
            raise ValueError("Expected three measured neck-plane points on the fitted body route")
        direction = np.cross(anchors[1]-anchors[0], anchors[2]-anchors[0])
        if abs(direction[1]) < 1.e-10:
            raise ValueError("Neck plane must span the collar horizontally")
        cut_normal = direction * np.sign(direction[1]) / np.linalg.norm(direction)
        cut_offset = float(anchors[0] @ cut_normal)
    def cut_height(points):
        return points @ cut_normal - cut_offset

    if measured_collar and (cut_height(np.array([marks["mouth_l"], marks["mouth_r"]])).min() <= 0
                            or cut_height(np.asarray(m.to_mesh().vert_properties)[:, :3]).min() >= 0):
        raise ValueError(f"Body collar and mouth leave no anatomical neck to bridge at {plane:.4f} m")
    if m.volume() <= 0:
        raise ValueError("Head volume must enclose a positive volume")
    tree = igl.AABB()
    tree.init(sv, sf)
    raytree = igl.AABB()
    sv_mm = sv * 1000
    raytree.init(sv_mm, sf)

    def ray_distance(origins, directions):
        hits, distances, _ = raytree.intersect_ray_first(sv_mm, sf, origins * 1000, directions)
        if np.any(hits < 0) or not np.isfinite(distances).all():
            raise ValueError("Bite section extends outside the observed muzzle")
        return distances / 1000

    def front(x, y):
        origins = np.c_[x, y, np.ones(len(x))]
        dr = np.tile([0.0, 0.0, -1.0], (len(x), 1))
        i, t, uv = raytree.intersect_ray_first(sv_mm, sf, origins * 1000, dr)
        ok = (i >= 0) & np.isfinite(t) & (t > 0)
        z = 1 - t / 1000
        if np.any(~ok):
            _, idx = cKDTree(sv[:, :2]).query(origins[~ok, :2], k=16)
            z[~ok] = sv[idx, 2].max(1)
        return z

    skin_faces = sf[observed["head_faces"]] if observed else sf
    skin_tree = igl.AABB()
    skin_tree.init(sv, skin_faces)

    def getcolor(q, skin=False):
        faces = skin_faces if skin else sf
        query = skin_tree if skin else tree
        dd, ii, closest = query.squared_distance(sv, faces, q.astype(float))
        bc = igl.barycentric_coordinates(
            closest, sv[faces[ii, 0]], sv[faces[ii, 1]], sv[faces[ii, 2]]
        )
        return np.clip((uv0[faces[ii]] * bc[:, :, None]).sum(1), 0, 1)

    mat0 = src.doc["materials"][sp["material"]]
    tex = mat0["pbrMetallicRoughness"]["baseColorTexture"]["index"]
    im = src.doc["images"][src.doc["textures"][tex]["source"]]
    bv = src.doc["bufferViews"][im["bufferView"]]
    off = bv.get("byteOffset", 0)
    image = np.asarray(
        Image.open(io.BytesIO(src.blob[off : off + bv["byteLength"]])).convert("RGB")
    )
    ht, wd = image.shape[:2]
    curves = {}
    eyes = {}
    eye_cuts = []
    for side in ("l", "r"):
        pre = "eye_" + side + "_"
        ed = np.array(
            sorted([marks[pre + "inner"], marks[pre + "outer"]], key=lambda p: p[0])
        )

        def arc(half):
            names = [pre + half + "_" + str(i) for i in range(1, 4)]
            pts = [marks[k] for k in names if k in marks] or [
                marks[pre + ("top" if half == "upper" else "bottom")]
            ]
            pts = np.array(sorted([*ed, *pts], key=lambda p: p[0]))
            return PchipInterpolator(pts[:, 0], pts[:, 1])

        up = arc("upper")
        lo = arc("lower")
        xs = np.linspace(*ed[:, 0], 49)
        a, b = up(xs), lo(xs)
        cy = (a.max() + b.min()) / 2
        z = np.maximum(front(xs, a), front(xs, b))
        source_eye = observed["eye_parts"].get(side) if observed else None
        if source_eye is not None:
            points, inverse = np.unique(sv[sf[source_eye]].reshape(-1, 3), axis=0, return_inverse=True)
            # An authored socket already has an eyelid rim. Removing its
            # temporary cap restores it; subtracting the eyeball again would
            # punch extra holes through the existing lid folds.
            if not surface_binding and side not in observed["eye_caps"] and "eye_" + side not in parts:
                tri, _ = igl.bfs_orient(inverse.reshape(-1, 3))
                eye_volume = solid(points, tri)
                if eye_volume.volume() < 0:
                    eye_volume = solid(points, tri[:, ::-1])
                eye_volume = eye_volume.as_original()
                if eye_volume.volume() <= 0:
                    raise ValueError("Observed eye does not enclose a positive volume")
                eye_cuts.append(eye_volume.original_id())
                m = m - eye_volume
        elif not surface_binding:
            m = m - cutter(xs, a, b, z, 0.035, 0.12)
        curves[side] = (xs, a, b, z)
        cxy = np.array([xs.mean(), cy])
        q = np.concatenate([np.c_[xs, a], np.c_[xs, b]])
        rad = np.linalg.norm(q - cxy, axis=1).max() * 1.25
        if source_eye is not None:
            # Authored cartoon eyes can be flattened. Keep the pivot at the
            # centre of the original shell, including its front-to-back extent.
            ctr = (points.min(0) + points.max(0)) * 0.5
            eyes[side] = (ctr, float(np.ptp(points, axis=0).max() * 0.5))
        elif face_atlas and not surface_binding:
            cxy, rad = observed_globe(xs, a, b, front, cxy, rad)
            surface_z = front(cxy[0:1], cxy[1:2])[0]
            eyes[side] = (fit_painted_globe(cxy, surface_z, rad, sv, skin_faces, skin_tree), rad)
        else:
            cz = (
                np.median(
                    front(q[:, 0], q[:, 1])
                    - np.sqrt(np.maximum(0, rad**2 - ((q - cxy) ** 2).sum(1)))
                )
                - 0.0015
            )
            eyes[side] = (np.r_[cxy, cz], rad)
    def trace_lips(path=None):
        return observed_binding.project_trace(r, cfg, sv_mm, sf, raytree, path)

    mouth = trace_lips()
    xs = np.linspace(mouth[0, 0], mouth[-1, 0], 68)
    y = np.interp(xs, mouth[:, 0], mouth[:, 1])
    z = np.interp(xs, mouth[:, 0], mouth[:, 2])
    muzzle = bite_plane(xs, y, z, front) if face_atlas and not surface_binding and "upper_lip" not in face_atlas else None
    pitch = 0.
    if muzzle is not None:
        y, z, pitch = muzzle
    h = (0.0004 if face_atlas else 0.00065) * np.sin(np.linspace(0, np.pi, len(xs)))
    upper, lower = y + h, y - h
    painted_smile = bool(face_atlas and not surface_binding and "upper_lip" in face_atlas)
    fitted_teeth = bool(face_atlas) and not surface_binding and not {"upper_dental", "lower_dental"}.intersection(parts)
    if painted_smile:
        # The teeth in this scan are painted onto a single skin surface.
        # Remove the observed tooth band before fitting the two mobile lips.
        traces = [trace_lips(face_atlas[key]) for key in ("upper_lip", "lower_lip")]
        upper, lower = [np.interp(xs, trace[:, 0], trace[:, 1]) for trace in traces]
        upper = np.maximum(upper, lower)
        upper[[0, -1]] = lower[[0, -1]]
        z = front(xs, (upper + lower) * .5)
    pocket_depth = np.ptp(z) if muzzle is not None else .030 if face_atlas else .018
    if not surface_binding:
        m = m - cutter(xs, upper, lower, z, pocket_depth, .13, (pitch, ray_distance) if muzzle is not None else None)
    curves["mouth"] = (xs, upper, lower, z)
    m = max(m.decompose(), key=lambda q: q.volume())
    if body_config:
        m, body_id, body_faces, body_report = volume_body.prepare(
            body_config, cfg["source_hash"], sv, sf, resolved, m)
    else:
        m = m.trim_by_plane(cut_normal, cut_offset)
    mesh = m.to_mesh()
    v = np.array(mesh.vert_properties[:, :3], float)
    f = np.asarray(mesh.tri_verts, int)
    keep = np.ones(len(f), bool) if body_config else ~(np.abs(cut_height(v[f])) < 1e-6).all(1)
    if observed and observed["eye_parts"]:
        owner = np.repeat(mesh.run_original_id, np.diff(mesh.run_index) // 3)
        keep &= ~np.isin(owner, eye_cuts)
        for caps in observed["eye_caps"].values():
            keep &= ~((owner == observed["source_id"]) & np.isin(mesh.face_id, caps))
    if face_atlas:
        owner = np.repeat(mesh.run_original_id, np.diff(mesh.run_index) // 3)
        face_id = np.asarray(mesh.face_id)
        original = observed["head_faces"]
        mapped = (owner == observed["source_id"]) & (face_id < len(original))
        ancestry = np.full(len(f), -1)
        ancestry[mapped] = original[face_id[mapped]]
        head_mask = keep & (owner != body_id) if body_config else keep
        ancestry = ancestry[head_mask]
    if body_config:
        donor = bp
        v, f, ringids, bp, bf, li = volume_body.partition(mesh, body_id, keep)
        neck_fit = {"method": "shared_boolean_boundary", "vertices": len(li)}
    else:
        f = f[keep]
        used, rev = np.unique(f, return_inverse=True)
        v = v[used]
        f = rev.reshape(-1, 3)
        loops = igl.boundary_loop_all(f)
        print(
            n, "cut loops", [(len(q), round(v[q, 1].mean(), 3)) for q in loops], flush=True
        )
        # Inner oral recesses can have more vertices than the outer neck.
        def projected_area(loop):
            x, z = v[loop][:, [0, 2]].T
            return abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1)))

        cut_loops = [q for q in loops if np.max(abs(cut_height(v[q]))) < 1e-5]
        neck = max(cut_loops, key=projected_area if measured_collar else len)
        # Blind recesses and hair overhangs may intersect the cut plane in extra cycles.
        # Cap each auxiliary loop independently; only the main neck joins the torso.
        if len(cut_loops) > 1:
            for loop in cut_loops:
                if list(loop) == list(neck):
                    continue
                ring = np.asarray(loop)
                poly = v[ring][:, [0, 2]]
                ts = triangle.triangulate(
                    dict(
                        vertices=poly,
                        segments=np.c_[
                            np.arange(len(ring)), np.roll(np.arange(len(ring)), -1)
                        ],
                    ),
                    "pYQ",
                )
                if "triangles" in ts:
                    f = np.vstack([f, ring[ts["triangles"]]])
        assert np.max(abs(cut_height(v[neck]))) < 1e-5
        def order(points, ids):
            q = points[ids]
            mid = q[:, [0, 2]].mean(0)
            ang = np.arctan2(q[:, 2] - mid[1], q[:, 0] - mid[0])
            step = np.angle(np.exp(1j * (np.roll(ang, -1) - ang))).sum()
            if step < 0:
                ids = ids[::-1]
                ang = ang[::-1]
            return np.roll(ids, -np.argmin(abs(ang)))

        neck = order(v, np.array(neck))
        li = order(bp, li)
        neck_fit = {}
        if measured_collar:
            original_body = bp
            neck_surface = sf[parts["skin"]] if "skin" in parts else skin_faces if observed else None
            bp, neck_fit = follow_neck_surface(bp, bf[tt], li, v[neck], weld, sv, neck_surface, resolved)
            if "neck_plane_points_m" in character_config:
                radius = float(np.ptp(bp[li, 0]) * .8)
                bp = fair_collar_band(bp, bf[tt], li, weld, radius)
                neck_fit.update(neck_plane_points_m=anchors.tolist(), fairing_radius_m=radius,
                                lower_bridge_texture="fitted_body_skin")
            body_changed = np.any(bp != original_body, axis=1)
            body_normals = normals(bp, bf[tt])[0]
        # Join the head to the exact skinned port of the exported body.
        nh = len(v)
        ringids = np.arange(len(li)) + nh
        v = np.vstack([v, bp[li]])
        bridge = stitch_neck(v, neck, ringids) if measured_collar else zipper(neck, ringids)
        f = np.vstack([f, bridge])
        f, _ = igl.bfs_orient(f)
        if np.sum(v[f[:, 0]] * np.cross(v[f[:, 1]], v[f[:, 2]] - v[f[:, 0]])) < 0:
            f = f[:, ::-1]
    names = [bare.doc["nodes"][i]["name"] for i in bare.doc["skins"][0]["joints"]]
    contacts = {}
    if face_atlas and not surface_binding:
        ancestry = np.pad(ancestry, (0, len(f) - len(ancestry)), constant_values=-1)
        for side in ("l", "r"):
            ids = parts.get("eye_" + side)
            if ids is None:
                continue
            contact = lid_contacts(v, f, sv, sf[ids], curves[side])
            v, f, ancestry, centers, sample = insert_lid_controls(v, f, ancestry, contact)
            contacts[side] = (contact, centers, sample)
    if measured_collar and "skin" in parts:
        # The old body port included jaw-weighted tissue. It now sits at
        # the base of the observed neck, beneath the moving mandible.
        w0 = w0.copy()
        retained = np.unique(bf[tt])
        band = retained[bp[retained, 1] >= min(bp[li, 1]) - .5 * np.ptp(bp[li, 0])]
        weight = np.where(j0[band] == names.index("jaw"), 0., w0[band])
        totals = weight.sum(1, keepdims=True)
        w0[band] = np.where(totals > 0, weight / np.maximum(totals, 1.e-12), w0[band])
    nr = normals(v, f)[0]
    j, w, delta = observed_binding.bind_face(v, names, curves, eyes, bite_pitch=pitch)
    if surface_binding:
        j[:] = 0
        j[:, 0] = names.index("head")
        w[:] = 0
        w[:, 0] = 1
        delta[:] = 0
        ancestry = np.pad(ancestry, (0, len(f) - len(ancestry)), constant_values=-1)
    for side, (contact, centers, sample) in contacts.items():
        close_authored_lids(v, delta, curves[side], contact)
        delta[centers] = (contact["weights"][sample, :, :, None]
                          * delta[contact["corners"][sample]]).sum(axis=1)
    # Separate authored eyebrows do not belong to the closed head solid.
    # Recover only their source-bound, reviewed triangles for both exports.
    brows = None
    owned = np.zeros(len(sf), dtype=bool)
    if observed:
        owned[ancestry[ancestry >= 0] if face_atlas else observed["head_faces"]] = True
        for ids in observed["eye_parts"].values():
            owned[ids] = True
    separate_parts = {name: parts[name] for name in ("lashes", "upper_dental", "lower_dental", "tongue") if name in parts}
    for ids in separate_parts.values():
        owned[ids] = True
    if surface_binding and "face_overlay" in parts:
        owned[parts["face_overlay"]] = True
    review_file = r / "1-segment/review.json"
    if review_file.is_file():
        review = json.loads(review_file.read_text())
        if review["source_hash"] != cfg["source_hash"]:
            raise ValueError("Segmentation review and transfer have different sources")
        for item in review["candidates"]:
            if item["status"] in ("brow", "internal"):
                for start, end in item["triangle_ranges"]:
                    owned[start:end] = True
        ranges = [interval for item in review["candidates"] if item["status"] == "brow"
                  for interval in item["triangle_ranges"]]
        if ranges:
            if len(mesh_nodes) != 1 or len(primitives) != 1:
                raise ValueError("Reviewed brow reconstruction requires the transfer's single-atlas scan")
            brow_faces = np.concatenate([sf[start:end] for start, end in ranges])
            bi, bf_local = np.unique(brow_faces, return_inverse=True)
            bt = bf_local.reshape(-1, 3)
            source_normals = attr(src, sp, "NORMAL")[bi] @ np.linalg.inv(M[:3, :3])
            source_normals /= np.maximum(np.linalg.norm(source_normals, axis=1, keepdims=True), 1.e-20)
            nearest = cKDTree(v).query(sv[bi])[1]
            brows = (sv[bi], bt, source_normals, j[nearest], w[nearest], uv0[bi])
    ctrl = {}
    for key, (xs, a, b, z) in curves.items():
        xx = np.clip(v[:, 0], *xs[[0, -1]])
        A = np.interp(xx, xs, a)
        B = np.interp(xx, xs, b)
        Z = np.interp(xx, xs, z)
        width = np.ptp(xs)
        inn = v[:, 1] - (A + B) / 2
        if key == "mouth":
            inn += pitch * (v[:, 2] - Z)
        on = (v[:, 2] > Z - (0.042 if key != "mouth" else 0.024)) & (
            np.abs(xx - v[:, 0]) < width * 0.19
        )
        xx = xs[len(xs) // 2]
        targets = [
            np.array(
                [
                    xx,
                    k[len(xs) // 2],
                    front(np.array([xx]), np.array([k[len(xs) // 2]]))[0],
                ]
            )
            for k in (a, b)
        ]
        if key in contacts:
            ctrl[key] = dict(zip(("upper", "lower"), ([i] for i in contacts[key][1])))
            continue
        ctrl[key] = {}
        for label, sgn, q in zip(("upper", "lower"), (1, -1), targets):
            sel = on & (sgn * inn > 0) & (np.abs(v[:, 0] - xx) < width * 0.12)
            ids = np.where(sel)[0]
            ctrid = ids[np.linalg.norm(v[ids] - q, axis=1).argmin()]
            ctrl[key][label] = [int(ctrid)]
    if body_config:
        hands = {}
        for side in ("l", "r"):
            node = next(
                (
                    n
                    for n in bare.doc["nodes"]
                    if n.get("name") == "scan_hand_" + side and "mesh" in n
                ),
                None,
            )
            recipe = next(
                (
                    h
                    for h in (cfg.get("stage2_hands", {}).get("hands") or [])
                    if h.get("side") == side
                ),
                {},
            )
            if node and not recipe.get("rigid", True):
                hand = bare.doc["meshes"][node["mesh"]]["primitives"][0]
                hands["arm_" + side] = (
                    *[
                        attr(bare, hand, key)
                        for key in ("POSITION", "JOINTS_0", "WEIGHTS_0")
                    ],
                    recipe["skinned_digits"],
                )
        j0, w0 = volume_body.body_weights(
            bp, bf, sv, sf, body_faces, resolved, donor, j0, w0, names, hands
        )
        source_body_faces = sf[np.concatenate(list(body_faces.values()))]
        _, tri, nearest = igl.point_mesh_squared_distance(bp, sv, source_body_faces)
        source_tri = source_body_faces[tri]
        bc = igl.barycentric_coordinates(
            nearest, *(sv[source_tri[:, k]] for k in range(3))
        )
        body_uv = (uv0[source_tri] * bc[:, :, None]).sum(1).clip(0, 1)
        body_rgb = (
            image[
                (body_uv[:, 1] * ht).astype(int).clip(0, ht - 1),
                (body_uv[:, 0] * wd).astype(int).clip(0, wd - 1),
            ]
            / 255.0
        )
        body_color = np.where(
            body_rgb <= 0.04045, body_rgb / 12.92, ((body_rgb + 0.055) / 1.055) ** 2.4
        )
        for indices in body_faces.values():
            owned[indices] = True
    ctrl["collar"] = {"head": ringids.tolist(), "body": li.tolist()}
    # Carry the original albedo per corner by closest-point projection. Split UV seams only.
    # Vertex colors avoid texture seams between non-corresponding source triangles.
    uv = getcolor(v, skin=observed is not None)
    rgb = (
        image[
            np.clip((uv[:, 1] * ht).astype(int), 0, ht - 1),
            np.clip((uv[:, 0] * wd).astype(int), 0, wd - 1),
        ]
        / 255
    )
    for k, (xs, a, b, z) in curves.items():
        xx = np.clip(v[:, 0], *xs[[0, -1]])
        A = np.interp(xx, xs, a)
        B = np.interp(xx, xs, b)
        Z = np.interp(xx, xs, z)
        inside = (
            (np.abs(xx - v[:, 0]) < 1e-5)
            & (v[:, 1] >= B - 1e-5)
            & (v[:, 1] <= A + 1e-5)
            & (v[:, 2] > Z - 0.06)
        )
        if k == "mouth":
            if muzzle is not None:
                height = v[:, 1] - (A + B) * .5 + pitch * (v[:, 2] - Z)
                inside |= (abs(height) < .18 * np.ptp(xs)) & (v[:, 2] >= Z - pocket_depth - .001)
            rgb[inside] = [0.22, 0.07, 0.09]
        else:
            shade = getcolor(np.c_[xx, A + 0.004, front(xx, A + 0.004)], skin=observed is not None)
            rgb[inside] = (
                image[
                    np.clip((shade[:, 1] * ht).astype(int), 0, ht - 1),
                    np.clip((shade[:, 0] * wd).astype(int), 0, wd - 1),
                ][inside]
                / 255
            )
    if measured_collar and "neck_plane_points_m" in character_config:
        # The measured boundary is the limit of visible skin. The lower
        # bridge meets the fitted body's baked skin rather than the shirt.
        mat = bare.doc["materials"][body["material"]]["pbrMetallicRoughness"]
        tex = bare.doc["textures"][mat["baseColorTexture"]["index"]]
        view = bare.doc["bufferViews"][bare.doc["images"][tex["source"]]["bufferView"]]
        offset = view.get("byteOffset", 0)
        bitmap = np.asarray(Image.open(io.BytesIO(bare.blob[offset:offset + view["byteLength"]])).convert("RGB"))
        body_uv = attr(bare, body, "TEXCOORD_0")[li]
        pixels = (body_uv * bitmap.shape[1::-1]).astype(int).clip(0, np.array(bitmap.shape[1::-1]) - 1)
        rgb[ringids] = bitmap[pixels[:, 1], pixels[:, 0]] / 255.
    lin = np.where(rgb < 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    if face_atlas:
        source_normals = attr(src, sp, "NORMAL") @ np.linalg.inv(M[:3, :3])
        source_normals /= np.maximum(np.linalg.norm(source_normals, axis=1, keepdims=True), 1.e-20)
        collar_uv = None
        if measured_collar and "neck_plane_points_m" not in character_config:
            bridge = (ancestry < 0) & (cut_height(v[f]).max(1) < 1.e-6)
            # Use vertex colour only where a triangle crosses an atlas seam.
            bridge &= np.ptp(uv[f], axis=1).max(1) < .08
            collar_uv = bridge, uv
        tex_ids, tex_faces, tex_uv, tex_normals, tex_tint = map_corners(
            v, f, ancestry, sv, sf, uv0, source_normals, nr, lin, wd, collar_uv)
        # Use a white strip for new cavity and collar triangles. Original
        # corners retain their atlas and authored shading, including UV seams.
        packed = np.pad(image, ((0, 0), (0, 16), (0, 0)), constant_values=255)
        if painted_smile:
            xx = np.clip(v[:, 0], xs[0], xs[-1])
            top, bottom, depth = [np.interp(xx, xs, q) for q in (upper, lower, z)]
            span = top - bottom
            d = v[:, 1] - (top + bottom) * .5
            outside = np.maximum(np.abs(d) - span * .5, 0)
            fade = np.clip(1 - outside / (.32 * np.ptp(xs)), 0, 1)
            fade *= np.clip((v[:, 2] - depth + .034) / .009, 0, 1)
            amount = np.sign(d) * np.minimum(np.abs(d), np.maximum(span - 0.0008, 0) * .5)
            v[:, 1] -= fade * amount
            middle = (upper + lower) * .5
            curves["mouth"] = (xs, middle + h, middle - h, z)
            j, w, _ = observed_binding.bind_face(v, names, curves, eyes)
        # The anatomical binder reserves slots for the jaw and lips. The
        # scan-surface route keeps a rigid skull and drives speech with morphs.
        if not surface_binding:
            support_chin(v, w, curves["mouth"], pitch)
        if not surface_binding and "jawline_zy_m" in face_atlas:
            profile = np.asarray(face_atlas["jawline_zy_m"], dtype=float)
            if profile.ndim != 2 or profile.shape[1] != 2 or len(profile) < 2 or not np.all(np.diff(profile[:, 0]) > 0):
                raise ValueError("Expected a jawline ordered from posterior to anterior in working metres")
            underside = np.interp(v[:, 2], profile[:, 0], profile[:, 1])
            width = np.ptp(curves["mouth"][0])
            chin = np.clip((v[:, 2] - profile[-1, 0] + width * .75) / (width * .55), 0, 1)
            blend = np.clip((v[:, 1] - underside + chin * width * .40) / (width * .25), 0, 1)
            w[:, 1:] *= (blend**2 * (3 - 2 * blend))[:, None]
            w[:, 0] = 1 - w[:, 1:].sum(1)
    if body_config:
        volume_body.bind_neck(v, bp, ringids, li, names, j, w, j0, w0, delta, resolved)
    j[ringids], w[ringids] = j0[li], w0[li]
    delta[ringids] = 0
    def eye_surface(x, y):
        origins = np.column_stack((x, y, np.ones(len(x))))
        hit, distance, bary = raytree.intersect_ray_first(
            sv_mm, sf, origins * 1000, np.tile([0., 0., -1.], (len(x), 1)))
        if np.any(hit < 0):
            raise ValueError("Painted eye extends beyond the observed surface")
        uv = np.einsum("ij,ijk->ik", np.c_[1. - bary.sum(1), bary], uv0[sf[hit]])
        return 1 - distance / 1000, uv

    exposed_source = None
    neck_skin = sf[parts["skin"]] if "skin" in parts else None
    if neck_skin is not None:
        exposed_source = np.zeros(len(sf), bool)
        exposed_source[parts["skin"]] = True
    if measured_collar and face_atlas and neck_skin is None:
        # The static single-atlas transfer exports its segmentation in source
        # triangle order, with remapped vertex indices. Avoid baking a shirt
        # or scarf onto the exposed body port.
        sg = G(r / "1-segment/segmentation.glb")
        sp_seg = sg.doc["meshes"][0]["primitives"][0]
        seg_faces = sg.array(sp_seg["indices"]).reshape(-1, 3)
        band = observed["head_faces"]
        heights = cut_height(sv[sf[band]])
        band = band[(heights.max(1) > -np.ptp(bp[li, 0])) & (heights.min(1) < .01)]
        if len(seg_faces) == len(sf) and np.array_equal(attr(sg, sp_seg, "TEXCOORD_0")[seg_faces[band]], uv0[sf[band]]):
            skin = np.all(np.isclose(attr(sg, sp_seg, "COLOR_0")[:, :3], [.95, .55, .35]), axis=1)
            exposed_source = skin[seg_faces].all(1)
            selected = sf[band[exposed_source[band]]]
            if len(selected):
                neck_skin = selected
    graft_cleanup = {}
    for rel in ["3-skin/body.glb", "4-garments/character.glb"]:
        original_neck = rel.startswith("4-") and character_config.get("dressed_neck_from_scan", False)
        if original_neck and (not surface_binding or not measured_collar or "neck_plane_points_m" not in character_config or "skin" not in parts):
            raise ValueError("Dressed source neck needs a reviewed skin surface and oblique collar plane")
        g = base_stage(rel)
        if observed:
            grafts = cfg.get("stage4_garments", {}).get("grafts") or {}
            graft_cleanup[rel] = trim_replaced_grafts(g, grafts.get("pieces", 0), uv0, sf, owned)
        if face_atlas:
            # A welded scan can include the fringe and clothing in its head
            # component. Their staged copies would cover the source face.
            covered = np.zeros(len(sf), bool)
            if body_config:
                covered[ancestry[ancestry >= 0]] = True
            else:
                covered[observed["head_faces"]] = cut_height(sv[sf[observed["head_faces"]]]).min(1) > 0
            for ids in separate_parts.values():
                covered[ids] = True
            if surface_binding and "face_overlay" in parts:
                covered[parts["face_overlay"]] = True
            garments = [piece["name"] for piece in cfg["stage4_garments"]["garments"]]
            graft_cleanup[rel].update(trim_replaced_grafts(g, 0, uv0, sf, covered, garments))
        body = prim(g, "body")
        if measured_collar:
            if not np.array_equal(attr(g, body, "POSITION"), original_body):
                raise ValueError("Bare and dressed body vertices differ at the neck fitting stage")
            body["attributes"]["POSITION"] = add(g, bp, "VEC3")
            attr(g, body, "NORMAL")[body_changed] = body_normals[body_changed]
            if "skin" in parts:
                body["attributes"]["WEIGHTS_0"] = add(g, w0, "VEC4")
            if neck_skin is not None and "neck_plane_points_m" not in character_config:
                neck_albedo(g, body, bp, bf[tt], li, sv, neck_skin, uv0, image,
                            source_shell=(raytree, sv_mm, sf, exposed_source), moved=body_changed)
        # The dressed stage already omits body triangles covered by clothing.
        # Preserve its index subset while removing the old head.
        if not body_config:
            stage_faces = g.array(body["indices"]).reshape(-1, 3)
            keep_keys = {tuple(t) for t in bf[tt]}
            kept = stage_faces[[tuple(t) in keep_keys for t in stage_faces]]
            if original_neck:
                # Under a reviewed closed collar the visible lower neck comes
                # from the original skin. Covered scaffold faces can protrude
                # after a fitted body's port has been moved.
                keep_neck = ~body_changed[kept].any(axis=1)
                neck_fit["dressed_hidden_collar_triangles"] = int((~keep_neck).sum())
                kept = kept[keep_neck]
            elif measured_collar and rel.startswith("4-") and exposed_source is not None:
                # Clothing was masked on the scaffold. Recheck the moved collar against
                # the outside of the source scan, where an open neckline exposes skin.
                nearby = bf[tt][body_changed[bf[tt]].any(1)]
                retained = set(map(tuple, kept))
                missing = nearby[[tuple(face) not in retained for face in nearby]]
                if len(missing):
                    centers = bp[missing].mean(1)
                    radial = centers - bp[li].mean(0)
                    radial[:, 1] = 0
                    radial /= np.maximum(np.linalg.norm(radial, axis=1)[:, None], 1e-12)
                    reach = np.linalg.norm(np.ptp(sv_mm, axis=0))
                    hit, _, _ = raytree.intersect_ray_first(sv_mm, sf, centers * 1000 + radial * reach, -radial)
                    restore = (hit >= 0) & exposed_source[hit.clip(min=0)]
                    # Retained garment meshes can already contain the observed
                    # exposed skin at an open neckline. Keep that original surface.
                    eligible = np.flatnonzero(restore)
                    if len(eligible):
                        probes = bp[missing[eligible]] * 0.8 + centers[eligible, None] * 0.2
                        probes = np.column_stack([probes.reshape(-1, 9), centers[eligible]]).reshape(-1, 3)
                        directions = np.repeat(radial[eligible], 4, axis=0)
                        radius = np.ptp(bp[li, 0])
                        origin = probes + directions * radius
                        present = np.zeros(len(probes), bool)
                        garment_names = {piece["name"] for piece in cfg["stage4_garments"]["garments"]}
                        for gn in g.doc["nodes"]:
                            if "mesh" not in gn or gn.get("name") not in garment_names:
                                continue
                            for pr in g.doc["meshes"][gn["mesh"]]["primitives"]:
                                gp = attr(g, pr, "POSITION").astype(float)
                                gf = g.array(pr["indices"]).reshape(-1, 3).astype(np.int64)
                                shell = igl.AABB()
                                shell.init(gp, gf)
                                hits, distance, _ = shell.intersect_ray_first(gp, gf, origin, -directions)
                                present |= (hits >= 0) & (distance < radius * 1.35)
                        restore[eligible[present.reshape(-1, 4).all(1)]] = False
                    kept = np.vstack([kept, missing[restore]])
                    neck_fit["restored_exposed_triangles"] = int(restore.sum())
            if len(kept):
                body["indices"] = idx(g, kept)
            elif not original_neck:
                raise ValueError("No visible fitted body triangles remain")
        bn = next(i for i in g.doc["nodes"] if i.get("name") == "body" and "mesh" in i)
        pi = g.doc["nodes"].index(bn)
        parent = next((q for q in g.doc["nodes"] if pi in q.get("children", [])), None)
        children = parent["children"] if parent else g.doc["scenes"][0]["nodes"]

        def put(name, v, f, n, ji, wt, mat, color=None, blink=None, uv=None):
            at = {
                k: add(g, np.asarray(q, dtype=d), typ, ct)
                for k, q, d, typ, ct in [
                    ("POSITION", v, "<f4", "VEC3", 5126),
                    ("NORMAL", n, "<f4", "VEC3", 5126),
                    ("JOINTS_0", ji, "<u2", "VEC4", 5123),
                    ("WEIGHTS_0", wt, "<f4", "VEC4", 5126),
                ]
            }
            if uv is not None:
                at["TEXCOORD_0"] = add(g, uv.astype("<f4"), "VEC2")
            if color is not None:
                at["COLOR_0"] = add(g, color.astype("<f4"), "VEC3")
            pr = dict(attributes=at, indices=idx(g, f), material=mat)
            mm = dict(name=name, primitives=[pr])
            if blink is not None:
                pr["targets"] = [dict(POSITION=add(g, blink.astype("<f4"), "VEC3"))]
                mm.update(weights=[0], extras=dict(targetNames=["eyeBlink"]))
            mi = len(g.doc["meshes"])
            g.doc["meshes"].append(mm)
            node = copy.deepcopy(bn)
            node.update(mesh=mi, name=name)
            node.pop("weights", None)
            children.append(len(g.doc["nodes"]))
            g.doc["nodes"].append(node)

        mat = len(g.doc["materials"])
        g.doc["materials"].append(
            dict(
                name="reconstructed surface",
                doubleSided=True,
                pbrMetallicRoughness=dict(metallicFactor=0, roughnessFactor=0.9),
            )
        )
        if body_config:
            volume_body.align_pivots(g, resolved)
            norm = normals(bp, bf)[0]
            shared = norm[li] + nr[ringids]
            shared /= np.maximum(np.linalg.norm(shared, axis=1, keepdims=True), 1.0e-12)
            norm[li], nr[ringids] = shared, shared
            body.clear()
            body.update(
                attributes={
                    k: add(g, np.asarray(q, dtype=d), typ, ct)
                    for k, q, d, typ, ct in [
                        ("POSITION", bp, "<f4", "VEC3", 5126),
                        ("NORMAL", norm, "<f4", "VEC3", 5126),
                        ("COLOR_0", body_color, "<f4", "VEC3", 5126),
                        ("JOINTS_0", j0, "<u2", "VEC4", 5123),
                        ("WEIGHTS_0", w0, "<f4", "VEC4", 5126),
                    ]
                },
                indices=idx(g, bf),
                material=mat,
            )
            bn.pop("weights", None)
            body_mesh = g.doc["meshes"][bn["mesh"]]
            body_mesh.pop("weights", None)
            body_mesh.get("extras", {}).pop("targetNames", None)
            for node in g.doc["nodes"]:
                if node.get("name", "").startswith(("scan_hand_", "wrist_bridge_")):
                    node.pop("mesh", None)
                    node.pop("skin", None)
        if fitted_teeth:
            teeth = prim(g, "teeth")
            positions = attr(g, teeth, "POSITION").copy()
            shifts = seat_dental_arches(positions, attr(g, teeth, "JOINTS_0"), attr(g, teeth, "WEIGHTS_0"),
                                        names.index("head"), curves["mouth"])
            forward = 0.
            if not painted_smile:
                xs, _, _, depth = curves["mouth"]
                center = (xs[0] + xs[-1]) * .5
                central = np.abs(positions[:, 0] - center) < .10 * np.ptp(xs)
                forward = np.interp(center, xs, depth) - .07 * np.ptp(xs) - positions[central, 2].max()
                positions[:, 2] += forward
                if "jawline_zy_m" in face_atlas:
                    # Fit the gum and root depth inside the observed chin.
                    # Keep the lower incisal edge at its registered height.
                    tooth_j = attr(g, teeth, "JOINTS_0")
                    tooth_w = attr(g, teeth, "WEIGHTS_0")
                    lower_arch = tooth_j[np.arange(len(positions)), tooth_w.argmax(1)] == names.index("jaw")
                    crown = positions[lower_arch, 1].max()
                    floor = np.interp(positions[lower_arch, 2], profile[:, 0], profile[:, 1])
                    room = np.maximum(crown - floor - width * .16, width * .14)
                    height = crown - positions[lower_arch, 1]
                    positions[lower_arch, 1] = crown - np.minimum(room / np.maximum(height.max(), 1.e-8), 1) * height
                    # Incisors stay forward while the lower gum sits behind
                    # the lip and follows the jaw around its rotation centre.
                    positions[lower_arch, 2] -= width * .12 * (height / height.max())**1.5
            tongue = prim(g, "tongue")
            tongue_positions = attr(g, tongue, "POSITION").copy()
            tongue_positions[:, 1] += shifts[1]
            tongue_positions[:, 2] += forward
            if muzzle is not None:
                positions, tongue_positions = fit_oral_mesh(positions, tongue_positions, curves["mouth"], pitch, ray_distance)
            teeth["attributes"]["POSITION"] = add(g, positions, "VEC3")
            tongue["attributes"]["POSITION"] = add(g, tongue_positions, "VEC3")
        if face_atlas:
            face_material = texture(g, packed, "observed facial surface")
            patches = [(v[tex_ids], tex_faces, tex_normals, j[tex_ids], w[tex_ids], tex_uv, tex_tint)]
            if surface_binding and "face_overlay" in parts:
                ids, local = np.unique(sf[parts["face_overlay"]], return_inverse=True)
                near = cKDTree(v).query(sv[ids])[1]
                overlay_uv = uv0[ids].copy()
                overlay_uv[:, 0] *= wd / (wd + 16)
                patches.append((sv[ids], local.reshape(-1, 3), source_normals[ids], j[near], w[near],
                                overlay_uv, np.ones((len(ids), 3))))
            if surface_binding and brows is not None:
                bv, bt, bnrm, bj, bw, buv = brows
                buv = buv.copy()
                buv[:, 0] *= wd / (wd + 16)
                patches.append((bv, bt, bnrm, bj, bw, buv, np.ones((len(bv), 3))))
            offset = 0
            topology = []
            for patch in patches:
                topology.append(patch[1] + offset)
                offset += len(patch[0])
            head_data = [np.concatenate([patch[k] for patch in patches]) for k in (0, 2, 3, 4, 5, 6)]
            hp, hn, hj, hw, huv, hc = head_data
            put("fitted_head", hp, np.concatenate(topology), hn, hj, hw, face_material, hc,
                None if surface_binding else delta[tex_ids], huv)
        else:
            put("fitted_head", v, f, nr, j, w, mat, lin, delta)
        atlas = material(g, src, sp) if brows is not None or parts or (observed and observed["eye_parts"]) else None
        if original_neck:
            from dressed_neck import bind_shared_collar, lower_surface
            nv, nf, nn, nuv = lower_surface(sv, sf[parts["skin"]], cut_normal, cut_offset, source_normals, uv0)
            nv, nj, nw, matches = bind_shared_collar(
                g, nv, sv, sf, parts["skin"], uv0, cut_normal, cut_offset, garments, names.index("head"))
            neck_fit["dressed_shared_collar_vertices"] = matches
            put("source_neck", nv, nf, nn, nj, nw, atlas, uv=nuv)
            neck_fit["dressed_observed_neck_triangles"] = len(nf)
        for name, source_faces in separate_parts.items():
            ids, local = np.unique(sf[source_faces], return_inverse=True)
            positions = sv[ids]
            if name == "lashes":
                nearest = cKDTree(v).query(positions)[1]
                joints, weights, movement = j[nearest], w[nearest], delta[nearest]
            else:
                joints = np.zeros((len(ids), 4), np.uint16)
                weights = np.zeros((len(ids), 4))
                joints[:, 0] = names.index("head" if name == "upper_dental" else "jaw")
                weights[:, 0] = 1
                movement = None
            put("source_" + name, positions, local.reshape(-1, 3), source_normals[ids],
                joints, weights, atlas, blink=movement, uv=uv0[ids])
        if brows is not None and not surface_binding:
            bv, bt, bnrm, bj, bw, buv = brows
            put("source_eyebrows", bv, bt, bnrm, bj, bw, atlas, uv=buv)
        for node in g.doc["nodes"]:
            if node.get("name") == "head" and face_atlas and face_atlas.get("speech_head_motion") is False:
                node.setdefault("extras", {})["charloomSpeechHeadMotion"] = False
            if node.get("name") == "jaw":
                clearance = 0.0008
                if muzzle is not None:
                    mouth_ctrl = ctrl["mouth"]
                    gap = v[mouth_ctrl["upper"][0], 1] - v[mouth_ctrl["lower"][0], 1]
                    clearance = max(0., float(gap))
                node.setdefault("extras", {})["charloomLipClearanceM"] = clearance
                if face_atlas and "labiodental_offset_m" in face_atlas:
                    offset = np.asarray(face_atlas["labiodental_offset_m"], dtype=float)
                    if offset.shape != (2,) or not np.isfinite(offset).all():
                        raise ValueError("Expected vertical and forward F/V offsets in metres")
                    node["extras"]["charloomLabiodentalOffsetM"] = offset.tolist()
            if fitted_teeth and (node.get("name") == "teeth" or (node.get("name") == "tongue" and "tongue" not in parts)):
                continue
            if node.get("name") in ["eyes", "tongue", "teeth", "ears"]:
                node.pop("mesh", None)
                node.pop("skin", None)
        if not body_config:
            for k, i in enumerate(li):
                attr(g, body, "NORMAL")[weld == weld[i]] = nr[ringids[k]]
        for side, (ctr, rad) in eyes.items():
            if surface_binding and side not in observed["eye_parts"]:
                continue
            xs, a, b, z = curves[side]
            bi = place_eye_joint(g, names, side, ctr)
            if observed and side in observed["eye_parts"]:
                ids, local = np.unique(sf[observed["eye_parts"][side]], return_inverse=True)
                nn = attr(src, sp, "NORMAL")[ids] @ np.linalg.inv(M[:3, :3])
                nn /= np.maximum(np.linalg.norm(nn, axis=1, keepdims=True), 1.e-20)
                ji = np.zeros((len(ids), 4), np.uint16)
                wt = np.zeros((len(ids), 4))
                ji[:, 0], wt[:, 0] = bi, 1
                put("eye_" + side, sv[ids], local.reshape(-1, 3), nn, ji, wt, atlas, uv=uv0[ids])
                continue
            ball = manifold.Manifold.sphere(1, 96).to_mesh()
            nn = np.asarray(ball.vert_properties[:, :3], float)
            ep = ctr + nn * rad
            ef = ball.tri_verts
            if face_atlas:
                euv, eye_image = painted_eye(ctr, rad, curves[side], eye_surface, image, nn)
                joints = np.zeros((len(ep), 4), np.uint16)
                weights = np.zeros((len(ep), 4))
                joints[:, 0], weights[:, 0] = bi, 1
                put("eye_" + side, ep, ef, nn, joints, weights,
                    texture(g, eye_image, "observed eye " + side), uv=euv)
                continue
            xc = np.clip(ep[:, 0], *xs[[0, -1]])
            up = np.interp(xc, xs, a)
            low = np.interp(xc, xs, b)
            visible = (
                (xc == ep[:, 0]) & (ep[:, 1] < up) & (ep[:, 1] > low) & (nn[:, 2] > 0)
            )
            col = np.tile([0.84, 0.80, 0.69], (len(ep), 1))
            if visible.any():
                pxy = ep[visible, :2]
                uv = getcolor(np.c_[pxy, front(*pxy.T)])
                rgb = (
                    image[
                        np.clip((uv[:, 1] * ht).astype(int), 0, ht - 1),
                        np.clip((uv[:, 0] * wd).astype(int), 0, wd - 1),
                    ]
                    / 255
                )
                col[visible] = np.where(
                    rgb < 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4
                )
            ji = np.zeros((len(ep), 4), np.uint16)
            wt = np.zeros_like(ji, float)
            ji[:, 0] = bi
            wt[:, 0] = 1
            put("eye_" + side, ep, ef, nn, ji, wt, mat, col)
        if surface_binding:
            from scan_surface_face import bind_face
            # The atlas preserves face order through UV splits. Optional overlay
            # and brow patches appended after it come directly from the scan.
            observed_faces = np.concatenate([ancestry >= 0, *[
                np.ones(len(patch[1]), dtype=bool) for patch in patches[1:]
            ]])
            ctrl.update(bind_face(g, curves, sv, sf, observed_faces=observed_faces,
                                  neck_plane=(cut_normal, cut_offset) if "neck_plane_points_m" in character_config else None))
            if original_neck:
                head = prim(g, "fitted_head")
                retained = g.array(head["indices"]).reshape(-1, 3)
                # Atlas splitting preserves the original fitted-head vertex
                # IDs. Drop the synthetic bridge only from the dressed layer.
                generated_bridge = (ancestry < 0) & (cut_height(v[f]).max(1) < 1.e-6)
                keys = {tuple(row) for row in tex_faces[generated_bridge]}
                head["indices"] = idx(g, retained[[tuple(row) not in keys for row in retained]])
                from dressed_neck import seam_controls
                ctrl["dressedNeck"] = seam_controls(g, nv, cut_normal, cut_offset, garments)
                if not len(kept):
                    # A closed outfit can hide the entire scaffold. Keep the
                    # skeleton and body metadata without an empty index accessor.
                    bn.pop("mesh")
                    bn.pop("skin", None)
        elif face_atlas:
            add_expression_targets(g, curves)
        finish(g, out / rel)
    if face_atlas and not surface_binding:
        ctrl["paintedEyes"] = {
            side: {"radius": radius, "sourceMaxU": wd / (wd + 16)}
            for side, (_, radius) in eyes.items() if side not in observed["eye_parts"]
        }
    (out / "controls.json").write_text(json.dumps(ctrl))
    report = dict(
        route="scan_surface_binding" if surface_binding else "free_topology",
        working_frame=cfg["working_frame"],
        source=cfg["source"],
        source_hash=cfg["source_hash"],
        mouth_bite_pitch=pitch if muzzle is not None else None,
        input_transfer=str(r),
        vertices=len(v),
        triangles=len(f),
        neck_vertices=len(li),
        neck_skin_fit=neck_fit,
        replaced_graft_triangles=graft_cleanup,
        volume_input=str(volume) if volume is not None else None,
        volume_preparation=preparation,
        observed_eye_triangles={
            side: len(tris) for side, tris in observed["eye_parts"].items()
        }
        if observed
        else {},
        reviewed_brow_triangles=len(brows[1]) if brows is not None else 0,
        selected_body=str(selected_body) if selected_body else None,
        plane_y_m=plane,
        body_volume=(
            {
                "surfaces": {key: len(indices) for key, indices in body_faces.items()},
                **body_report,
                "body_triangles": len(bf),
            }
            if body_config
            else None
        ),
        character_config=character_config,
        **({"source_textured_faces": int((ancestry >= 0).sum()),
            "reconstructed_faces": int((ancestry < 0).sum())} if face_atlas else {}),
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(n, "complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--config", type=Path, help="Per-character JSON with an optional prepared head, collar plane or reviewed body_volume inputs")
    parser.add_argument("--python-binding", action="store_true", help="Compute lip rays, the collar and face binding directly in Python")
    args = parser.parse_args()
    config = json.loads(args.config.read_text()) if args.config else {}
    for name in args.names:
        run(name, config.get(name))
