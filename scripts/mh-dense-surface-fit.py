"""Diagnostic image-space first pass for the Brute upper-body surface fit.

    out/tools/cv/.venv/bin/python scripts/mh-dense-surface-fit.py V2 GUIDE FRAME_RUN OUT

The displacement is a smooth field in each calibrated view. The front drives X/Y,
the side drives Z, so the two fits cannot move each other's horizontal axis.
Every pass reprojects the current surface. All neutral meshes share the warp.
The output is an intermediate. Visible scan skin, face preservation and
actual-export collision checks are separate steps for the reviewed v3 variant.
"""
import argparse
import hashlib
import json
import shutil
from importlib import import_module
from pathlib import Path

import cv2
import numpy as np
from scipy import sparse
from scipy.ndimage import map_coordinates
from scipy.sparse.linalg import spsolve

import mh_sections

match = import_module("mh-match")
from roster import mesh as lab
contours = import_module("mh-suit-contours")


def stationary_field(samples, shape, spacing, smoothness):
    """Least-squares displacement grid with bending regularization and weak rest bias."""
    ny, nx = (int(np.ceil(s / spacing)) + 2 for s in shape)
    points = np.array([sample[0] for sample in samples]) / spacing
    target = np.array([sample[1] for sample in samples])
    weights = np.sqrt([sample[2] for sample in samples])
    x, y = points.T
    ix, iy = x.astype(int), y.astype(int)
    tx, ty = x - ix, y - iy
    index = iy * nx + ix
    ids = np.stack([index, index+1, index+nx, index+nx+1], axis=1)
    mult = np.stack([(1-tx)*(1-ty),tx*(1-ty),(1-tx)*ty,tx*ty], axis=1)
    design = sparse.csr_matrix(((mult * weights[:, None]).ravel(),
                               (np.repeat(np.arange(len(points)), 4), ids.ravel())),
                              shape=(len(samples), nx*ny))
    lx = sparse.diags([1.,-2.,1.], [0,1,2], shape=(nx-2,nx))
    ly = sparse.diags([1.,-2.,1.], [0,1,2], shape=(ny-2,ny))
    penalty = sparse.vstack([sparse.kron(sparse.eye(ny),lx), sparse.kron(ly,sparse.eye(nx))])
    lhs = design.T @ design + smoothness * (penalty.T @ penalty) + .032 * sparse.eye(nx*ny)
    values = spsolve(lhs.tocsc(), design.T @ (target * weights[:, None]))
    return values.reshape(ny,nx,-1)


def sample_field(field, coords, step):
    loc = (coords[:,::-1] / step).T
    return np.column_stack([map_coordinates(field[...,i],loc,order=3,mode="nearest")
                            for i in range(field.shape[2])])


def row(mask, y, cx):
    return mh_sections.edges(mask, [cx-1,y],[cx+1,y])


def constraints(camera, mask, ref, stations, front, landmarks, neck_rows):
    out = []
    cy = landmarks["neck"][1]
    cx = landmarks["neck"][0]
    for name, endpoints in stations.items():
        if not name.startswith(("neck", "arm_", "trunk")):
            continue
        model = mh_sections.edges(mask, *endpoints)
        if model is None:
            continue
        delta = endpoints - model
        # A ray crossing a merged arm/chest or jaw/torso is not an anatomical
        # diameter. Neighbouring visible silhouette edges supply its motion.
        if np.max(np.linalg.norm(delta,axis=1)) > 38:
            continue
        for idx,(point, move) in enumerate(zip(model, delta)):
            if name.startswith("neck") and idx==0:
                # A side ray through the bare jaw includes beard in the reference.
                # Keep the bare jaw inside its observed envelope.
                if point[1] < np.mean([p[1] for key,pair in stations.items() if key.startswith("neck") for p in pair])+15 and move[0] < 0:
                    continue
            out.append((point, move if front else [move[0]], 1.3))
    if front:
        # Measure vertical columns across the upper outline. This captures the
        # shoulder shelf at almost constant height without flattening the deltoid.
        lateral = max(abs(landmarks["shoulder_l"][0]-cx),abs(landmarks["shoulder_r"][0]-cx))
        for u in np.arange(round(cx-1.45*lateral), round(cx+1.45*lateral), 4):
            a, b = np.flatnonzero(ref[:,u]), np.flatnonzero(mask[:,u])
            if not len(a) or not len(b) or abs(u-cx)<.245*lateral:
                continue
            shift = a[0]-b[0]
            out.append(([u,b[0]], [0,shift], 4))
        for name, endpoints in neck_rows.items():
            y = int(endpoints[0,1])
            model = row(mask,y,cx)
            if model is None:
                continue
            # Near horizontal shoulder caps the vertical distance above is the
            # stable observation. Across steeper neck rows use the normal residual.
            for side,(dst,src) in enumerate(zip(endpoints,model)):
                adjacent = row(ref,y+1,cx)
                tangent = (adjacent[side,0]-dst[0]) if adjacent is not None else 1
                normal = np.array([1.,-tangent])
                normal /= np.linalg.norm(normal)
                move = normal * np.dot(dst-src,normal)
                out.append((src,move,1))
        for y in range(round(cy)-100,round(cy)-46,8):
            for u in (cx-27,cx,cx+27):
                out.append(([u,y],[0,0],9))
    else:
        # Sample the side envelope through the throat and chest, fixing the
        # face above the collar. The seven oblique cuts are rechecked separately.
        end = max(int(max(p[1] for p in pair)) for key,pair in stations.items() if key.startswith("trunk"))
        cuts = [p for key,pair in stations.items() if key.startswith("neck") for p in pair]
        if not cuts:
            return out
        center = float(np.mean([p[0] for p in cuts]))
        top = min(p[1] for p in cuts)
        for y in range(int(top)-30,end,4):
            model, target = row(mask,y,center), row(ref,y,center)
            if model is None or target is None:
                continue
            for side,(src,dst) in enumerate(zip(model,target)):
                # The anterior silhouette above the collar contains the beard
                # in the image. Never expand the bare jaw to include it.
                if side==0 and y < max(p[1] for p in cuts)+3:
                    shift = max(0.,dst[0]-src[0]) if y>top+24 else 0.
                    if y>top+44:
                        shift = dst[0]-src[0]
                else:
                    shift=dst[0]-src[0]
                w = np.clip((y-top+24)/24,0,1)
                if side==0 and y<max(p[1] for p in cuts):
                    shift=max(0.,shift)
                out.append((src,[w*shift],2))
        for y in range(round(top)-54,round(top)-5,8):
            for u in (center-40,center,center+25):
                out.append(([u,y],[0],6))
    return out


def main(run, guide, frame, out):
    out.mkdir(parents=True,exist_ok=False)
    base = json.loads((frame/'report.json').read_text())
    guidance = json.loads((guide/'guidance.json').read_text())
    views = json.loads((guide/'views.json').read_text())
    assert base["source_hash"] == guidance["source_hash"] == views["source_hash"]
    M = np.array(base["working_frame"]["source_to_working"]).reshape(4,4).T
    inverse = np.linalg.inv(M)
    front = guidance["front_view"]
    side = f"body_y{(int(front.split('_y')[1])+90)%360:03d}"
    cameras = {c["id"]: c for c in views["cameras"] if c["id"] in (front,side)}
    marks = json.loads((guide/'views/bodysuit.json').read_text())["marks"]
    stations = {vid:{k[8:-1]:np.array([p,marks[k[:-1]+"2"][1]]) for k,(v,p) in marks.items()
                     if v==vid and k.startswith('outline_') and k.endswith('1')}for vid in cameras}
    fm = {k:o["px"] for k,obs in guidance["landmarks"].items() for o in obs if o["view"]==front}
    refs = {vid:contours.figure_mask(guide/'views'/f'bodysuit_{vid}-reg.png') for vid in cameras}
    neck_rows = mh_sections.neck_transition(refs[front],fm)
    neck_cut_points = np.array([p for name,pair in stations[side].items() if name.startswith("neck") for p in pair])
    if not len(neck_cut_points):
        raise ValueError("No saved side neck measurements for this guide")
    neck_center=neck_cut_points.mean(0)
    neck_width=np.ptp(neck_cut_points[:,0])
    fields = []
    meshes = match.glb_meshes(run/'3-skin/body.glb')
    parts = [(pos,tri) for name,pos,tri in meshes if name=='body' or name.startswith(('scan_hand_','wrist_bridge_'))]
    p = np.concatenate([v for v,t in parts])
    off = np.cumsum([0]+[len(v) for v,t in parts])
    triangles = np.concatenate([t+offset for (v,t),offset in zip(parts,off)])
    def project(vid,p):
        return match.project(cameras[vid],p @ inverse[:3,:3].T + inverse[:3,3])
    def apply(p,vid,field,spacing):
        cam = cameras[vid]
        uv = project(vid,p)
        move = sample_field(field,uv,spacing)
        if vid == side:
            # Ahead of the chest, an unobserved anterior neck must not expand
            # into the beard silhouette. A smooth cap avoids a kink at zero.
            front_px = project(front,p)
            # The profile camera projects anterior towards decreasing image X.
            mid_x,mid_y=neck_center
            center_weight=np.clip((neck_width*.72-abs(front_px[:,0]-fm['neck'][0]))/(neck_width*.34),0,1)
            under_jaw = np.clip((uv[:,1]-mid_y+neck_width*.20)/(neck_width*.27),0,1)*np.clip((mid_y+neck_width*.37-uv[:,1])/(neck_width*.18),0,1)
            anterior = np.clip((mid_x+neck_width*.14-uv[:,0])/(neck_width*.54),0,1)
            move[:,0] -= np.minimum(0.,move[:,0])*center_weight*under_jaw*anterior

        d = move[:,:1] * np.array(cam['right'])
        if vid==front:
            d -= move[:,1:] * cam['up']
        return p + cam['meters_per_pixel']*(d @ M[:3,:3].T)
    for _ in range(10):
        for vid in (front,side):
            cam = cameras[vid]
            mask = match.silhouette(cam,p @ inverse[:3,:3].T + inverse[:3,3],triangles)
            samples = constraints(cam,mask,refs[vid],stations[vid],vid==front,fm,neck_rows)
            spacing = 18
            field = stationary_field(samples,mask.shape,spacing,.7)
            field *= .7
            grid_y, grid_x = np.mgrid[0:mask.shape[0]:3, 0:mask.shape[1]:3]
            grid = np.c_[grid_x.ravel(), grid_y.ravel()]
            move = sample_field(field, grid, spacing).reshape(*grid_x.shape,-1)
            # Keep each update locally invertible, even at garment positions
            # between the measured surface points. The final mesh is checked again.
            derivatives = [np.gradient(move[...,i],3.,axis=axis) for i in range(move.shape[-1]) for axis in (1,0)]
            lipschitz = np.sqrt(sum(d*d for d in derivatives)).max()
            field *= min(1., .58 / max(lipschitz, 1e-9))
            fields.append((vid,field,spacing))
            p = apply(p,vid,field,spacing)
    def warp(p):
        for vid,field,spacing in fields:
            p = apply(p,vid,field,spacing)
        return p
    records = []
    for rel in ('3-skin/body.glb','4-garments/character.glb'):
        g = lab.Glb(run/rel)
        seen = set()
        for mesh in g.doc['meshes']:
            for pr in mesh['primitives']:
                attr = pr['attributes']
                if attr['POSITION'] in seen:
                    continue
                seen.add(attr['POSITION'])
                pos = g.array(attr['POSITION'])
                normals = g.array(attr['NORMAL'])
                for i in range(0,len(pos),24000):
                    sl = slice(i,i+24000)
                    old = pos[sl].astype(float)
                    new = warp(old)
                    h = .00006
                    jac = np.stack([(warp(old+h*axis)-new)/h for axis in np.eye(3)],axis=2)
                    det = np.linalg.det(jac)
                    if det.min()<=0:
                        raise ValueError(f"Folded deformation in {mesh['name']}: {det.min()}")
                    n = np.linalg.solve(jac.transpose(0,2,1),normals[sl,:,None]).squeeze(-1)
                    normals[sl] = n / np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-12)
                    pos[sl] = new
                access = g.doc['accessors'][attr['POSITION']]
                access.update(min=pos.min(0).tolist(),max=pos.max(0).tolist())
        path=out/rel
        path.parent.mkdir(exist_ok=True)
        g.write(path)
        records.append({"file":rel,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    shutil.copy(run/'candidate-report.json',out/'candidate-report.json')
    metadata = json.loads((run/'report.json').read_text())
    metadata.update(parent=str(run.resolve()),exports=records,status='pending_static_review',
                    static_review={'native_pipeline_adopted':False,'user_approved':False,'animation_validated':False})
    metadata.pop('torso_profile',None)
    metadata['dense_surface_fit']={'frame_report':str((frame/'report.json').resolve()),
                                   'station_file':str((guide/'views/bodysuit.json').resolve()),
                                   'scope':'Neutral surface; front and side silhouette observations, static rig provisional.'}
    (out/'report.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for key in ('run','guide','frame','out'):
        ap.add_argument(key,type=Path)
    main(**vars(ap.parse_args()))
