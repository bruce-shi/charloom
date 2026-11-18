"""Recover native MakeHuman quad identity without deforming the body.

Run with the existing CV Python environment. The native NPZ is extracted from
the matching editable.blend (helpers hidden, Delete masks disabled), in glTF
coordinates. Rejects mismatched priors, ambiguous triangle ownership and UV
discontinuities inside a quad. This is correspondence data, not a fitted rig.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from roster import mesh as lab

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('native', type=Path)
    parser.add_argument('prior', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Use a new output directory; existing artifacts are immutable')
    native = np.load(args.native)
    positions = native['positions']
    quads = native['quads'].astype(np.int32)
    if quads.ndim != 2 or quads.shape[1] != 4:
        raise ValueError('Expected native quads')
    prior = lab.Glb(args.prior)
    body = prior.body()
    raw = prior.array(body['attributes']['POSITION'])
    triangles = prior.array(body['indices']).reshape(-1, 3).astype(np.int32)
    uv = prior.array(body['attributes']['TEXCOORD_0'])
    distance, raw_to_native = cKDTree(positions).query(raw)
    if distance.max() > 1e-6 or len(np.unique(raw_to_native)) != len(positions):
        raise ValueError('Prior does not cover the native template within 1 micrometre')
    incident = [set() for _ in positions]
    for face_id, quad in enumerate(quads):
        for vertex in quad:
            incident[vertex].add(face_id)
    triangle_to_quad = np.empty(len(triangles), np.int32)
    quad_uv = np.full((len(quads), 4, 2), np.nan)
    for index, triangle in enumerate(triangles):
        mapped = raw_to_native[triangle]
        owners = set.intersection(*(incident[v] for v in mapped))
        if len(owners) != 1:
            raise ValueError(f'Triangle {index} has ambiguous native quad ownership')
        owner = owners.pop()
        triangle_to_quad[index] = owner
        for raw_id, vertex in zip(triangle, mapped):
            corner = int(np.flatnonzero(quads[owner] == vertex)[0])
            old = quad_uv[owner, corner]
            if np.isfinite(old).all() and np.max(abs(old - uv[raw_id])) > 1e-7:
                raise ValueError(f'UV discontinuity inside native quad {owner}')
            quad_uv[owner, corner] = uv[raw_id]
    if not (np.bincount(triangle_to_quad, minlength=len(quads)) == 2).all():
        raise ValueError('Expected exactly two exported triangles per native quad')
    if not np.isfinite(quad_uv).all():
        raise ValueError('Missing quad corner UVs')
    args.out.mkdir(parents=True)
    np.savez_compressed(args.out / 'correspondence.npz',
                        positions=positions, quads=quads,
                        original_ids=native['original_ids'],
                        raw_to_native=raw_to_native.astype(np.int32),
                        triangle_to_quad=triangle_to_quad, quad_corner_uv=quad_uv)
    result = {
        'status': 'validated_correspondence_only',
        'native': str(args.native.resolve()), 'prior': str(args.prior.resolve()),
        'native_sha256': hashlib.sha256(args.native.read_bytes()).hexdigest(),
        'prior_sha256': hashlib.sha256(args.prior.read_bytes()).hexdigest(),
        'native_vertices': len(positions), 'native_quads': len(quads),
        'exported_vertices': len(raw), 'exported_triangles': len(triangles),
        'max_mapping_error_m': float(distance.max()),
        'all_quads_have_two_triangles': True, 'quad_corner_uvs_complete': True,
        'geometry_modified': False, 'rig_modified': False,
        'scope': 'Native topology and UV identity only; no shape or animation approval',
    }
    (args.out / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
