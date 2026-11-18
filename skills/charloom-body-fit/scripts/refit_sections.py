"""Map an intact native body to reviewed anatomical stations without changing topology.

PY refit_sections.py CONFIG.json --out NEW_DIRECTORY

Each operation supplies native/target axes, paired source/target stations and
explicit per-vertex blend weights. Target bounds come from reviewed measurements;
this command does not infer unobserved depth or choose anatomical regions.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'skills/charloom-body-measure/scripts'))
from measurement_common import read, write, checked_file, load_mesh, face_selection, digest
from mesh_sections import section_basis, cut_triangles, section_loop
from roster.mesh import adjacency
from roster.mesh_output import add, finish, normals


def checked_basis(value):
    b = np.asarray(value, float)
    if b.shape != (3, 3) or not np.isfinite(b).all() or not np.allclose(b.T @ b, np.eye(3), atol=1e-6) or np.linalg.det(b) < .999:
        raise ValueError('basis must be a right-handed orthonormal 3x3, columns axis/width/depth')
    return b


def map_sections(points, origin, basis, target_origin, target_basis, stations):
    """Preserve the native cross-section shape while mapping its paired bounds."""
    origin, target_origin = np.asarray(origin), np.asarray(target_origin)
    basis, target_basis = checked_basis(basis), checked_basis(target_basis)
    s = np.asarray([v['native_axis_m'] for v in stations], float)
    t = np.asarray([v['target_axis_m'] for v in stations], float)
    bounds = np.asarray([v['native_bounds_m'] for v in stations], float)
    goals = np.asarray([v['target_bounds_m'] for v in stations], float)
    if (len(s) < 2 or np.any(np.diff(s) <= 0) or np.any(np.diff(t) <= 0)
            or bounds.shape != (len(s), 2, 2) or goals.shape != bounds.shape
            or not all(np.isfinite(v).all() for v in (s,t,bounds,goals))
            or np.any(bounds[:,1] <= bounds[:,0]) or np.any(goals[:,1] <= goals[:,0])):
        raise ValueError('Ordered anatomical stations and positive width/depth bounds required')
    local = (points-origin) @ basis
    at = np.clip(local[:,0], s[0], s[-1])
    # PCHIP retains anatomical height order; transverse interpolation cannot
    # overshoot measured section bounds as a free cubic spline can.
    mapped = local.copy()
    mapped[:,0] = PchipInterpolator(s,t)(at)
    for k in range(2):
        lo = np.interp(at,s,bounds[:,0,k]); hi = np.interp(at,s,bounds[:,1,k])
        dest_lo = np.interp(at,s,goals[:,0,k]); dest_hi = np.interp(at,s,goals[:,1,k])
        mapped[:,k+1] = dest_lo + (local[:,k+1]-lo)*(dest_hi-dest_lo)/(hi-lo)
    return target_origin + mapped @ target_basis.T


def world_to_local(surface, part, points):
    world = surface.scene.world(surface.scene.rest)
    if part.weights is None:
        m = world[part.node]
        return np.linalg.solve(m[:3,:3], (points-m[:3,3]).T).T
    m = ((world[part.bones] @ part.inverse)[part.palette] * part.weights[:,:,None,None]).sum(1)
    if np.any(np.abs(np.linalg.det(m[:,:3,:3])) < 1e-10):
        raise ValueError('Singular skin transform')
    return np.linalg.solve(m[:,:3,:3], (points-m[:,:3,3])[:,:,None])[:,:,0]


def run(config_path, out):
    cfg = read(config_path); base = config_path.parent
    surface, part, old, topology = load_mesh(base,cfg['mesh'])
    native_surface, native_part, native, native_topology = load_mesh(base,cfg['native_mesh'])
    native_faces = {tuple(row) for row in native_part.faces}
    if len(native) != len(old) or any(tuple(row) not in native_faces for row in part.faces):
        raise ValueError('Fitted body must retain native vertex order and a subset of native triangles')
    if any(k in cfg['mesh'] for k in ('clip','world_from_model')):
        raise ValueError('Edit the untransformed rest export; mapping acts in its world frame')
    if part.deltas:
        raise ValueError('Refit the body before adding body morph targets')
    if out.exists():
        raise FileExistsError(out)
    total = np.zeros(len(old)); displacement = np.zeros_like(old); records=[]
    for op in cfg['operations']:
        blend = np.load(checked_file(base, op['blend']), allow_pickle=False)
        if blend.shape != (len(old),) or not np.isfinite(blend).all() or np.any((blend<0)|(blend>1)):
            raise ValueError('blend requires one reviewed weight in [0,1] per native vertex')
        stations = op['stations']
        for row in stations:
            if not row.get('evidence'):
                raise ValueError('Every station requires measured target evidence or an explicit estimate source')
        basis = checked_basis(op['native_basis']); origin=np.asarray(op['native_origin_m'],float)
        coord = (native-origin) @ basis
        active = blend > 1e-8
        if np.any(active & ((coord[:,0] < stations[0]['native_axis_m']) | (coord[:,0] > stations[-1]['native_axis_m']))):
            raise ValueError('Active blend extends outside measured native stations; extend observations or taper the region')
        region = face_selection(native_part,native_surface,op['native_region'])
        for row in stations:
            segments,_ = cut_triangles(native,native_part.faces[region],origin+basis[:,0]*row['native_axis_m'],basis)
            loop = section_loop(segments,np.asarray(row.get('native_seed_m',[0,0])))
            actual=np.array([loop.min(0),loop.max(0)])
            if 'native_bounds_m' in row and not np.allclose(actual,row['native_bounds_m'],atol=1e-6):
                raise ValueError('Saved native bounds differ from actual native triangle sections')
            row['native_bounds_m']=actual.tolist()
        proposal=map_sections(native,origin,basis,op['target_origin_m'],op['target_basis'],stations)
        displacement+=(proposal-old)*blend[:,None];total+=blend
        records.append({'name':op['name'],'stations':stations,'blend_sha256':op['blend']['sha256']})
    if np.any(total>1+1e-6):
        raise ValueError('Overlapping region blends exceed 1; blend chest/arm transition explicitly')
    strength=float(cfg.get('displacement_fairing',0))
    if not np.isfinite(strength) or strength<0:
        raise ValueError('displacement_fairing must be nonnegative')
    if strength:
        _,first,weld=np.unique(native.round(6),axis=0,return_index=True,return_inverse=True)
        _,lap=adjacency(weld[part.faces],len(first))
        displacement=spsolve((eye(len(first))+strength*(lap.T@lap)).tocsc(),displacement[first])[weld]*total[:,None]
    changed=old+displacement
    local=world_to_local(surface,part,changed)
    g=surface.scene.glb; attrs=part.prim['attributes']
    _,first,weld=np.unique(local.round(6),axis=0,return_index=True,return_inverse=True)
    n=normals(local[first],weld[part.faces])[0][weld]
    moved=np.linalg.norm(displacement,axis=1)>1e-10
    attrs['POSITION']=add(g,local.astype('<f4'),'VEC3')
    previous=g.array(attrs['NORMAL']) if 'NORMAL' in attrs else n
    attrs['NORMAL']=add(g,np.where(moved[:,None],n,previous).astype('<f4'),'VEC3')
    out.mkdir(parents=True)
    finish(g,out/'body.glb')
    report={'status':'requires_measurement_and_motion_review','input':cfg['mesh'],
            'native':cfg['native_mesh'],'config_sha256':digest(config_path),'triangle_sha256':topology,
            'operations':records,'changed_vertices':int(moved.sum()),
            'max_displacement_mm':float(np.linalg.norm(displacement,axis=1).max()*1000),
            'output_sha256':digest(out/'body.glb')}
    write(out/'refit.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('config',type=Path);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();result=run(args.config.resolve(),args.out)
    print(f"{result['changed_vertices']} vertices mapped; {result['status']}")
