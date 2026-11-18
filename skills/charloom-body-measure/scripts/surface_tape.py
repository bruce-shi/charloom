"""Trace bilateral arm/leg tape paths on visible scan skin and registered references.

    PY surface_tape.py SPEC.json --out DIRECTORY
    PY surface_tape.py SPEC.json --out DIRECTORY --model-spec EXPORT.json

SPEC supplies the calibrated guide, source-to-working frame, source mesh,
reviewed skin triangles, and one path definition per limb. Reference joint
centres must have observations in nonparallel views. Export centres use saved
skin bindings, not unchanged skeleton pivots. No dimensions are filled from a
human-proportion table. The saved raw samples are distinct from the lightly
smoothed tape curve that bridges scan facets.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter1d

from measurement_common import (
    bound_point, checked_file, digest, face_selection, load_mesh, matrix,
    project_working, read, reference_context, transform_points, unit, write,
)


def ray_hit(origin, direction, vertices, faces, max_distance):
    """Nearest positive Moller-Trumbore intersection, preserving source face IDs."""
    triangles = vertices[faces]
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    h = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, h)
    use = np.abs(det) > 1e-12
    inverse = 1 / np.where(use, det, 1)
    delta = origin - triangles[:, 0]
    u = np.einsum("ij,ij->i", delta, h) * inverse
    cross = np.cross(delta, e1)
    v = (cross @ direction) * inverse
    distance = np.einsum("ij,ij->i", e2, cross) * inverse
    hits = np.flatnonzero(use & (u >= -1e-8) & (v >= -1e-8)
                         & (u + v <= 1 + 1e-8) & (distance > 0) & (distance < max_distance))
    if not len(hits):
        return None
    face = int(hits[np.argmin(distance[hits])])
    bary = np.array([1 - u[face] - v[face], u[face], v[face]])
    return origin + distance[face] * direction, face, bary


def guide_centres(guide, cameras, frame, names):
    result, evidence = {}, {}
    for name in names:
        observations = guide["landmarks"].get(name, [])
        if not observations:
            raise ValueError(f"Missing joint centre: {name}")
        projectors, origins, weights = [], [], []
        for mark in observations:
            cam = cameras[mark["view"]]
            x, y = mark["px"]
            if not (0 <= x < cam["width"] and 0 <= y < cam["height"]):
                raise ValueError(f"Joint outside camera: {name}")
            o = (np.asarray(cam["center"])
                 + np.asarray(cam["right"]) * (x - cam["width"] / 2) * cam["meters_per_pixel"]
                 - np.asarray(cam["up"]) * (y - cam["height"] / 2) * cam["meters_per_pixel"])
            d = unit(cam["forward"])
            projectors.append(np.eye(3) - np.outer(d, d))
            origins.append(o)
            weights.append(mark.get("confidence", 1) * {"occluded": .5, "partial": .8}.get(mark.get("visibility"), 1))
        weights = np.asarray(weights)
        if not np.isfinite(weights).all() or (weights < 0).any():
            raise ValueError(f"Invalid observation confidence: {name}")
        total = np.einsum("i,ijk->jk", weights, projectors)
        if np.linalg.cond(total) >= 100:
            raise ValueError(f"{name} needs observations in nonparallel views; cannot invent depth")
        point = np.linalg.solve(total, np.einsum("i,ijk,ik->j", weights, projectors, origins))
        result[name] = transform_points(point, frame)
        residuals = [float(np.linalg.norm(p @ (point - o)) * np.linalg.norm(frame[:3, 0]) * 1000)
                     for p, o in zip(projectors, origins)]
        evidence[name] = {"observations": observations, "xyz_m": result[name].tolist(),
                          "ray_residuals_mm": residuals,
                          "evidence": "marked_anatomical_centre; occluded centres are estimates"}
    return result, evidence


def silhouette_mask(base, record):
    path = checked_file(base, record)
    im = np.asarray(Image.open(path).convert("RGB"))
    if record["kind"] == "binary":
        mask = im[:, :, 0] >= 128
    elif record["kind"] == "figure":
        # The saved calibrated background and threshold are explicit inputs.
        background = np.asarray(record["background_rgb"], float)
        mask = np.linalg.norm(im.astype(float) - background, axis=2) > record["threshold"]
    else:
        raise ValueError("mask.kind must be binary or figure")
    return mask, path


def silhouette_hit(origin, direction, cam, frame, mask, step, max_distance):
    """Intersect a projected 3D ray with the contiguous reference silhouette."""
    distances = np.arange(0, max_distance + step / 2, step)
    xy = project_working(cam, frame, origin + distances[:, None] * direction)
    px = np.rint(xy).astype(int)
    height, width = mask.shape
    valid = ((px[:, 0] >= 0) & (px[:, 0] < width)
             & (px[:, 1] >= 0) & (px[:, 1] < height))
    if not valid[0] or not mask[px[0, 1], px[0, 0]]:
        raise ValueError("Tape ray origin is outside the reviewed limb silhouette")
    inside = np.zeros(len(px), bool)
    inside[valid] = mask[px[valid, 1], px[valid, 0]]
    exits = np.flatnonzero(~inside)
    if not len(exits) or exits[0] == 0 or not valid[exits[0]]:
        raise ValueError("No reference boundary before max_radius_mm/image edge; review region")
    distance = (exits[0] - .5) * step
    point = origin + distance * direction
    return point, project_working(cam, frame, point)[0]


def recipe_rays(joints, kind, side, count, recipe):
    """The measured tape route: lateral limb, around outer/posterior elbow."""
    a, elbow, end = joints
    upper, lower = unit(elbow - a), unit(end - elbow)
    sign = 1 if side == "l" else -1
    for t in np.linspace(0, 2, 2 * count + 1):
        origin = a + (elbow-a)*t if t <= 1 else elbow+(end-elbow)*(t-1)
        k = np.clip((t - recipe["bend_start"]) / recipe["bend_span"], 0, 1)
        blend = k*k*(3-2*k)
        axis = unit(upper*(1-blend) + lower*blend)
        lateral = unit(np.array([sign, 0., 0.]) - axis*(sign*axis[0]))
        back = unit(np.cross(axis, lateral))*sign
        if kind == "arm":
            bend = np.exp(-((t-1)/recipe["elbow_width"])**2)
            tip = np.exp(-(t/recipe["shoulder_width"])**2)
            direction = unit(lateral - recipe["elbow_back"]*back*bend - recipe["shoulder_up"]*axis*tip)
        else:
            direction = lateral
        yield float(t), origin, direction


def smooth_tape(points, count):
    # Preserve shoulder/hip, elbow/knee and wrist/ankle while bridging tiny facets.
    parts = []
    for path in (points[:count+1], points[count:]):
        curve = gaussian_filter1d(path, max(.65, count/30), axis=0, mode="nearest")
        curve[[0, -1]] = path[[0, -1]]
        parts.append(curve)
    return np.concatenate((parts[0], parts[1][1:]))


def inner_upperarm_rays(joints, endpoints, side, count):
    """Medial upper-arm route, from marked axillary fold to marked medial elbow."""
    start, end = joints
    axis = unit(end-start)
    span = np.linalg.norm(end-start)
    stations = (endpoints-start) @ axis
    if not (0 <= stations[0] < stations[1] <= span*1.05):
        raise ValueError('Axilla/inner-elbow endpoints must be ordered along the upper arm')
    inward = np.array([-1. if side == 'l' else 1., 0., 0.])
    medial = unit(inward-axis*np.dot(inward,axis))
    # Axillary fold and medial elbow can be anterior/posterior to the joint
    # axis. Follow their actual radial directions instead of forcing both
    # onto the joint-centre depth plane.
    radial=endpoints-(start+stations[:,None]*axis)
    directions=np.array([unit(v) for v in radial])
    if np.any(directions @ medial <= 0):
        raise ValueError('Inner upper-arm endpoints must lie on the medial side of the limb axis')
    for t, distance in zip(np.linspace(0,1,count+1), np.linspace(*stations,count+1)):
        direction=unit((1-t)*directions[0]+t*directions[1])
        yield float(t), start+axis*distance, direction


def validate_inner_endpoints(raw, endpoints, tolerance_mm):
    error = np.linalg.norm(raw[[0,-1]]-endpoints,axis=1)*1000
    if not np.isfinite(tolerance_mm) or tolerance_mm <= 0:
        raise ValueError('Inner upper arm needs a positive endpoint_tolerance_mm')
    if np.any(error > tolerance_mm):
        raise ValueError(f'Inner-arm ray does not reach the marked axilla/medial elbow: {error.tolist()} mm; review surface points and arm isolation')
    return error.tolist()


def length(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum() * 1000)


DEFAULT_RECIPE = {"bend_start": .86, "bend_span": .28, "elbow_width": .44,
                  "shoulder_width": .22, "elbow_back": .55, "shoulder_up": .68}


def run(spec_path, out, model_path=None):
    base, spec = spec_path.parent, read(spec_path)
    expected={(kind,side) for kind in ('arm','leg','upperarm_inner') for side in ('l','r')}
    present={(p['kind'],p['side']) for p in spec['paths']}
    missing=sorted(expected-present)
    if len({p['name'] for p in spec['paths']}) != len(spec['paths']):
        raise ValueError('Duplicate path name')
    scope=spec.get('scope','selected_paths')
    if scope not in ('full_body','selected_paths'):
        raise ValueError('scope must be full_body or selected_paths')
    if scope=='full_body' and missing:
        raise ValueError(f'Full body path coverage is missing {missing}')
    guide_path, guide, cameras, frame = reference_context(base, spec)
    source, part, vertices, topology = load_mesh(base, spec["source_mesh"])
    skin = set(face_selection(part, source, {"triangle_ranges": spec["reviewed_skin_triangle_ranges"]}).tolist())
    names = sorted({name for p in spec["paths"] for name in p["joints"]+p.get('endpoints',[])})
    centres, centre_evidence = guide_centres(guide, cameras, frame, names)
    model = read(model_path) if model_path else None
    if model:
        surf, model_part, model_vertices, model_topology = load_mesh(model_path.parent, model["mesh"])
        model_centres = {name: bound_point(model_vertices, model_part.faces, model["landmark_bindings"][name]) for name in names}
    rows, failures, pictures = {}, [], {}
    for definition in spec["paths"]:
        name, kind, side = definition["name"], definition["kind"], definition["side"]
        inner = kind == 'upperarm_inner'
        if kind not in ("arm", "leg", "upperarm_inner") or side not in ("l", "r") or len(definition["joints"]) != (2 if inner else 3):
            raise ValueError(f"Invalid path definition: {name}")
        if inner and len(definition.get('endpoints',[])) != 2:
            raise ValueError('Inner upper arm requires independent axilla and medial-elbow surface endpoints')
        count = definition.get("samples_per_segment", 45)
        if not isinstance(count, int) or count < 2:
            raise ValueError("samples_per_segment must be an integer >= 2")
        recipe = DEFAULT_RECIPE | definition.get("recipe", {})
        if recipe["bend_span"] <= 0 or recipe["elbow_width"] <= 0 or recipe["shoulder_width"] <= 0:
            raise ValueError("Invalid tape recipe width")
        radius, step = definition["max_radius_mm"] / 1000, definition.get("step_mm", 1) / 1000
        if not 0 < step < radius:
            raise ValueError("Expected 0 < step_mm < max_radius_mm")
        mask, image_path = silhouette_mask(base, definition["reference_mask"])
        cam = cameras[definition["view"]]
        if mask.shape != (cam["height"], cam["width"]):
            raise ValueError("Reference mask must match its calibrated camera")
        if model:
            region = face_selection(model_part, surf, model["regions"][name])
            if not len(region):
                raise ValueError(f"Empty exported limb region: {name}")
        results = {}
        for label, anchors in (("reference", centres), ("export", model_centres if model else None)):
            if anchors is None:
                continue
            samples, evidence = [], []
            chain = np.array([anchors[n] for n in definition["joints"]])
            endpoints = np.array([anchors[n] for n in definition['endpoints']]) if inner else None
            rays = inner_upperarm_rays(chain,endpoints,side,count) if inner else recipe_rays(chain, kind, side, count, recipe)
            for t, origin, direction in rays:
                if label == "reference":
                    hit = ray_hit(origin, direction, vertices, part.faces, radius)
                    if hit is not None and hit[1] in skin:
                        point, face, bary = hit
                        provenance = {"kind": "visible_scan_skin", "triangle": face, "barycentric": bary.tolist()}
                    else:
                        point, px = silhouette_hit(origin, direction, cam, frame, mask, step, radius)
                        provenance = {"kind": "registered_silhouette_with_marked_profile_axis_estimate", "pixel": px.tolist(), "view": cam["id"]}
                else:
                    hit = ray_hit(origin, direction, model_vertices, model_part.faces[region], radius)
                    if hit is None:
                        raise ValueError(f"Exported skin ray missed: {name} at {t}")
                    point, local_face, bary = hit
                    provenance = {"kind": "exported_skin", "triangle": int(region[local_face]), "barycentric": bary.tolist()}
                samples.append(point)
                evidence.append({"station": t, "origin_m": origin.tolist(), "direction": direction.tolist(), **provenance})
            raw = np.asarray(samples)
            if inner:
                endpoint_errors = validate_inner_endpoints(raw,endpoints,definition['endpoint_tolerance_mm'])
                tape = gaussian_filter1d(raw,max(.65,count/30),axis=0,mode='nearest')
                tape[[0,-1]]=raw[[0,-1]]
                lengths = [length(tape)]
            else:
                tape = smooth_tape(raw, count)
                lengths = [length(tape[:count+1]),length(tape[count:])]
            results[label] = {
                "raw_surface_samples_m": raw.tolist(), "tape_xyz_m": tape.tolist(),
                "sample_evidence": evidence, "segments_mm": lengths,
                "total_mm": length(tape), "joint_chain_mm": length(chain),
                "vertical_drop_mm": float((chain[0, 1]-chain[-1, 1])*1000),
            }
            if inner:
                results[label].update(endpoint_mark_errors_mm=endpoint_errors,
                                      endpoint_names=definition['endpoints'],
                                      endpoint_chord_mm=float(np.linalg.norm(endpoints[1]-endpoints[0])*1000))
        rows[name] = {"definition": definition, "recipe": recipe, **results}
        if model:
            tolerance = model["tolerance_mm"]
            errors = np.array(results["export"]["segments_mm"]) - results["reference"]["segments_mm"]
            total_error = results["export"]["total_mm"] - results["reference"]["total_mm"]
            rows[name].update(segment_errors_mm=errors.tolist(), total_error_mm=total_error)
            if not np.isfinite(tolerance) or tolerance <= 0:
                raise ValueError("Expected positive tolerance_mm selected before fitting")
            if max(np.abs(errors).max(), abs(total_error)) > tolerance:
                failures.append(name)
        # Put each reference and actual curve on the same original camera.
        key = str(image_path)
        if key not in pictures:
            pictures[key] = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(pictures[key])
        for label, color in (("reference", "#f6dd68"), ("export", "#59e6e4")):
            if label in results:
                px = project_working(cam, frame, results[label]["tape_xyz_m"])
                draw.line([tuple(p) for p in px], fill=color, width=2)
                for index in ((0,count) if inner else (0,count,2*count)):
                    x, y = px[index]
                    draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
    out.mkdir(parents=True, exist_ok=False)
    for index, picture in enumerate(pictures.values()):
        picture.save(out / f"paths-{index}.png")
    report = {"spec_sha256": digest(spec_path), "source_hash": spec["source_hash"],
              "guide_sha256": digest(guide_path / "guidance.json"), "source_triangle_sha256": topology,
              "source_to_working": frame.tolist(), "joint_centres": centre_evidence,
              "units": "working-model millimetres", "paths": rows, "failed": failures,
              "scope": scope, "missing_full_body_paths": missing,
              "status": "comparison_failed" if failures else "comparison_passed" if model else "targets_for_visual_review"}
    if model:
        report.update(model_spec_sha256=digest(model_path), model=model["mesh"], model_triangle_sha256=model_topology)
    write(out / "tape.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-spec", type=Path)
    args = parser.parse_args()
    report = run(args.spec.resolve(), args.out, args.model_spec.resolve() if args.model_spec else None)
    print(f"{len(report['paths'])} paths; {report['status']}")
    raise SystemExit(bool(report["failed"]))
