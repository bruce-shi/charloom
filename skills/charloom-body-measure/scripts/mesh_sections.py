"""Cut actual mesh triangles perpendicular to each limb axis and compare registered rays.

    PY mesh_sections.py SPEC.json --model-spec MODEL.json --out DIRECTORY

The spec contains guide-bound anatomical axes, station fractions, and reviewed
reference masks for each visible width/depth direction. MODEL selects the mesh,
its skin-bound centres and anatomical face regions. Hidden reference directions
are explicitly unavailable, never filled by symmetry or an ellipse formula.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from measurement_common import (
    bound_point, digest, face_selection, load_mesh, project_working, read,
    reference_context, unit, write,
)
from surface_tape import guide_centres, ray_hit, silhouette_mask


def section_basis(start, end, width_hint, frontal_normal=(0., 0., 1.)):
    axis = unit(end - start)
    hint = np.asarray(width_hint, float)
    # Retain the measured frontal width direction used by the limb-section
    # workflow: perpendicular to the limb, in the frontal plane.
    width = unit(np.cross(axis, unit(frontal_normal)))
    if np.dot(width, hint) < 0:
        width = -width
    depth = unit(np.cross(axis, width))
    return np.column_stack((axis, width, depth))


def cut_triangles(vertices, faces, center, basis):
    local = (vertices - center) @ basis
    triangles = local[faces]
    distances = triangles[:, :, 0]
    keep = (distances.min(1) < 0) & (distances.max(1) >= 0)
    selected = np.flatnonzero(keep)
    triangles, distances = triangles[keep], distances[keep]
    ends = np.zeros((len(triangles), 2, 2))
    count = np.zeros(len(triangles), int)
    for i, j in ((0, 1), (1, 2), (2, 0)):
        rows = np.flatnonzero((distances[:, i] < 0) != (distances[:, j] < 0))
        t = distances[rows, i] / (distances[rows, i] - distances[rows, j])
        ends[rows, count[rows]] = triangles[rows, i, 1:] + t[:, None] * (triangles[rows, j, 1:] - triangles[rows, i, 1:])
        count[rows] += 1
    if not len(ends) or not np.all(count == 2):
        raise ValueError("Plane did not yield triangle intersection segments")
    return ends, selected


def section_loop(segments, seed=(0., 0.)):
    """Weld coincident cut endpoints, requiring an unambiguous closed component."""
    vertices, inverse = np.unique(segments.reshape(-1, 2).round(6), axis=0, return_inverse=True)
    edges = np.unique(np.sort(inverse.reshape(-1, 2), axis=1), axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    adjacent = {int(i): [] for i in np.unique(edges)}
    for a, b in edges:
        adjacent[int(a)].append(int(b))
        adjacent[int(b)].append(int(a))
    remaining, candidates = set(adjacent), []
    while remaining:
        first = min(remaining)
        todo, component = [first], set()
        while todo:
            node = todo.pop()
            if node in component:
                continue
            component.add(node)
            todo.extend(adjacent[node])
        remaining -= component
        if any(len(adjacent[i]) != 2 for i in component):
            continue
        path, previous, current = [], None, first
        while current not in path:
            path.append(current)
            nxt = [n for n in adjacent[current] if n != previous][0]
            previous, current = current, nxt
        if current != first or len(path) < 3:
            continue
        polygon = vertices[path]
        x, y = seed
        a, b = polygon, np.roll(polygon, -1, axis=0)
        crossing = (a[:, 1] > y) != (b[:, 1] > y)
        denom = np.where(crossing, b[:, 1]-a[:, 1], 1)
        ray_x = a[:, 0] + (y-a[:, 1]) * (b[:, 0]-a[:, 0]) / denom
        if np.count_nonzero(crossing & (x < ray_x)) % 2:
            candidates.append(polygon)
    if len(candidates) != 1:
        raise ValueError(f"Expected one closed section around seed, got {len(candidates)}; isolate the anatomical faces or review the seed")
    return candidates[0]


def reference_span(center, axis, camera, frame, mask, radius, step):
    """Measure along the projection of the SAME 3D section axis, in world mm."""
    distances = np.arange(-radius, radius + step/2, step)
    world = center + distances[:, None] * axis
    xy = project_working(camera, frame, world)
    if np.linalg.norm(xy[-1]-xy[0]) < 2:
        raise ValueError("Reference camera cannot observe this section axis")
    pixel = np.rint(xy).astype(int)
    h, w = mask.shape
    valid = (pixel[:, 0] >= 0) & (pixel[:, 0] < w) & (pixel[:, 1] >= 0) & (pixel[:, 1] < h)
    inside = np.zeros(len(pixel), bool)
    inside[valid] = mask[pixel[valid, 1], pixel[valid, 0]]
    midpoint = int(np.argmin(np.abs(distances)))
    if not inside[midpoint]:
        raise ValueError("Anatomical section centre is outside reference region")
    low = high = midpoint
    while low > 0 and inside[low-1]:
        low -= 1
    while high+1 < len(inside) and inside[high+1]:
        high += 1
    if low == 0 or high == len(inside)-1 or not (valid[low-1] and valid[high+1]):
        raise ValueError("Section reference reaches search/image boundary; cannot infer missing edge")
    return distances[[low, high]], xy[[low, high]]


def reference_targets(spec_path, out):
    """Save measured target spans before a fitted mesh exists."""
    spec=read(spec_path)
    guide_dir,guide,cameras,frame=reference_context(spec_path.parent,spec)
    names=sorted({n for item in spec['sections'] for n in item['axis']})
    centres,evidence=guide_centres(guide,cameras,frame,names)
    rows=[];unavailable=[];pictures={}
    for item in spec['sections']:
        a,b=item['axis'];basis=section_basis(centres[a],centres[b],item['width_hint'],item.get('frontal_normal',[0,0,1]))
        for fraction in item['fractions']:
            if not np.isfinite(fraction) or not 0<=fraction<=1:raise ValueError('Section fraction must be in [0,1]')
            center=centres[a]+fraction*(centres[b]-centres[a]);refs={}
            for index,kind in enumerate(('width','depth')):
                ref=item['references'][kind]
                if 'unavailable' in ref:
                    refs[kind]={'status':'unavailable','reason':ref['unavailable']}
                    unavailable.append(f"{item['name']}:{fraction}:{kind}");continue
                mask,path=silhouette_mask(spec_path.parent,ref['mask']);cam=cameras[ref['view']]
                if mask.shape!=(cam['height'],cam['width']):raise ValueError('Reference mask dimensions disagree with camera')
                radius,step=ref['max_radius_mm']/1000,ref.get('step_mm',.5)/1000
                if not 0<step<radius:raise ValueError('Expected 0 < step_mm < max_radius_mm')
                span,px=reference_span(center,basis[:,index+1],cam,frame,mask,radius,step)
                refs[kind]={'status':'measured_projection_with_marked_axis_estimate','reference_mm':float(np.diff(span)[0]*1000),
                            'local_bounds_mm':(span*1000).tolist(),'view':cam['id'],'endpoints_px':px.tolist(),'image_sha256':digest(path)}
                if str(path) not in pictures:pictures[str(path)]=Image.open(path).convert('RGB')
                ImageDraw.Draw(pictures[str(path)]).line([tuple(p) for p in px],fill='#f4d95d',width=2)
            rows.append({'name':item['name'],'fraction':fraction,'center_m':center.tolist(),'basis':basis.tolist(),'references':refs})
    out.mkdir(parents=True,exist_ok=False)
    for i,picture in enumerate(pictures.values()):picture.save(out/f'targets-{i}.png')
    report={'spec_sha256':digest(spec_path),'source_hash':spec['source_hash'],'guide_sha256':digest(guide_dir/'guidance.json'),
            'source_to_working':frame.tolist(),'joint_centres':evidence,'units':'working-model millimetres',
            'sections':rows,'unavailable':unavailable,'failed':[],
            'status':'incomplete' if unavailable else 'targets_for_visual_review'}
    write(out/'sections.json',report)
    return report


def run(spec_path, model_path, out):
    if model_path is None:
        return reference_targets(spec_path,out)
    spec, model = read(spec_path), read(model_path)
    guide_dir, guide, cameras, frame = reference_context(spec_path.parent, spec)
    names = sorted({n for item in spec["sections"] for n in item["axis"]})
    centres, centre_evidence = guide_centres(guide, cameras, frame, names)
    surface, part, vertices, topology = load_mesh(model_path.parent, model["mesh"])
    observed = {n: bound_point(vertices, part.faces, model["landmark_bindings"][n]) for n in names}
    tolerance = model["tolerance_mm"]
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Expected a positive tolerance_mm selected before fitting")
    rows, failed, unavailable, pictures = [], [], [], {}
    for item in spec["sections"]:
        a, b = item["axis"]
        normal = item.get("frontal_normal", [0, 0, 1])
        target_basis = section_basis(centres[a], centres[b], item["width_hint"], normal)
        basis = section_basis(observed[a], observed[b], item["width_hint"], normal)
        ids = face_selection(part, surface, model["regions"][item["name"]])
        if not len(ids):
            raise ValueError(f"Empty region {item['name']}")
        for fraction in item["fractions"]:
            if not np.isfinite(fraction) or not 0 <= fraction <= 1:
                raise ValueError("Section fraction must be in [0,1]")
            center = observed[a] + fraction*(observed[b]-observed[a])
            target = centres[a] + fraction*(centres[b]-centres[a])
            row = {"name": item["name"], "fraction": fraction, "center_m": center.tolist(),
                   "axis": basis[:, 0].tolist(), "width_axis": basis[:, 1].tolist(), "depth_axis": basis[:, 2].tolist(),
                   "center_offset_mm": ((center-target)*1000).tolist(), "references": {}}
            try:
                segments, hit_ids = cut_triangles(vertices, part.faces[ids], center, basis)
                row["intersection_triangles"] = ids[hit_ids].tolist()
                polygon = section_loop(segments, np.asarray(item.get("loop_seed_mm", [0, 0]))/1000)
            except ValueError as exc:
                row.update(section_status="no_unique_closed_section", reason=str(exc))
                unavailable.append(f"{item['name']}:{fraction}:closed_section")
                polygon = None
            if polygon is not None:
                low, high = polygon.min(0), polygon.max(0)
                world = center + polygon @ basis[:, 1:].T
                row.update(section_status="closed_section", loop_xyz_m=world.tolist(),
                           model_width_mm=float((high[0]-low[0])*1000),
                           model_depth_mm=float((high[1]-low[1])*1000))
            for index, kind in enumerate(("width", "depth")):
                reference = item["references"][kind]
                key = f"{item['name']}:{fraction}:{kind}"
                if "unavailable" in reference:
                    row["references"][kind] = {"status": "unavailable", "reason": reference["unavailable"]}
                    unavailable.append(key)
                    continue
                mask, image = silhouette_mask(spec_path.parent, reference["mask"])
                cam = cameras[reference["view"]]
                if mask.shape != (cam["height"], cam["width"]):
                    raise ValueError("Reference mask dimensions disagree with camera")
                span, px = reference_span(target, target_basis[:, index+1], cam, frame, mask,
                                          reference["max_radius_mm"]/1000, reference.get("step_mm", .5)/1000)
                width = float(np.diff(span)[0]*1000)
                if polygon is None:
                    # A station can merge into the torso. Preserve any actual
                    # directed surface hits without inventing a closed span.
                    endpoints = []
                    for sign, target_dist in zip((-1, 1), span):
                        hit = ray_hit(center, sign*basis[:, index+1], vertices, part.faces[ids],
                                      reference["max_radius_mm"]/1000)
                        if hit is None:
                            endpoints.append({"status": "unavailable"})
                            continue
                        point, face, bary = hit
                        distance = float((point - center) @ basis[:, index+1])
                        error_mm = (distance-target_dist)*1000
                        endpoints.append({"status": "surface_hit", "point_m": point.tolist(),
                                          "triangle": int(ids[face]), "barycentric": bary.tolist(),
                                          "local_mm": distance*1000, "error_mm": error_mm})
                        if abs(error_mm) > tolerance:
                            failed.append(f"{key}:{sign}")
                    row["references"][kind] = {"status": "directed_endpoints_only", "view": cam["id"],
                                               "image_sha256": digest(image), "endpoints_px": px.tolist(),
                                               "target_bounds_mm": (span*1000).tolist(), "endpoints": endpoints}
                    continue
                error = row[f"model_{kind}_mm"]-width
                endpoint_errors = (np.array([low[index], high[index]])-span)*1000
                row["references"][kind] = {
                    "status": "measured_projection_with_marked_axis_estimate", "view": cam["id"],
                    "image_sha256": digest(image), "endpoints_px": px.tolist(),
                    "reference_mm": width, "error_mm": error,
                    "local_endpoint_errors_mm": endpoint_errors.tolist(),
                    "note": "Reference boundary constraint along projected 3D section axis; not a directly scanned hidden-body section",
                }
                if max(abs(error), np.max(np.abs(endpoint_errors))) > tolerance:
                    failed.append(key)
                if str(image) not in pictures:
                    pictures[str(image)] = Image.open(image).convert("RGB")
                draw = ImageDraw.Draw(pictures[str(image)])
                draw.line([tuple(p) for p in px], fill="#f4d95d", width=2)
                actual = project_working(cam, frame, world)
                draw.line([tuple(p) for p in np.vstack((actual, actual[:1]))], fill="#44e8ed")
            rows.append(row)
    out.mkdir(parents=True, exist_ok=False)
    for i, picture in enumerate(pictures.values()):
        picture.save(out / f"sections-{i}.png")
    report = {"spec_sha256": digest(spec_path), "model_spec_sha256": digest(model_path),
              "model": model["mesh"], "triangle_sha256": topology, "source_hash": spec["source_hash"],
              "guide_sha256": digest(guide_dir / "guidance.json"), "joint_centres": centre_evidence,
              "units": "working-model millimetres", "sections": rows, "failed": failed,
              "unavailable": unavailable,
              "status": "incomplete" if unavailable else "comparison_failed" if failed else "comparison_passed"}
    write(out / "sections.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--model-spec", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.spec.resolve(), args.model_spec.resolve() if args.model_spec else None, args.out)
    print(f"{len(report['sections'])} section records; {report['status']}")
    raise SystemExit(bool(report["failed"] or report["unavailable"]))
