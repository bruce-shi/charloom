"""Offline same-sheet profile calibration and back-head observations.
Only creates files below suit-back/registration; e3 never enters training.
"""
import json
from importlib import import_module
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

MIN_HEAD_IOU = .80

def boundary(mask, box):
    edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    y, x = np.nonzero(edge)
    p = np.column_stack((x + .5, y + .5))
    lo, hi = box
    return p[np.all(p >= lo, axis=1) & np.all(p < hi, axis=1)]

def profile_similarity(points, target, center, hr):
    tree = cKDTree(target)
    def residual(p):
        q = hr.transform(points, hr.affine(p, center))
        return np.r_[tree.query(q)[0], cKDTree(q).query(target)[0], (p[0]-1)*2, p[1]*2]
    result = least_squares(residual, [1., 0., 0., 0.],
        bounds=([.8, -np.deg2rad(5), -35, -35], [1.25, np.deg2rad(5), 35, 35]),
        loss='soft_l1', f_scale=1.5, max_nfev=180, diff_step=1e-4)
    mat = hr.affine(result.x, center)
    q = hr.transform(points, mat)
    distances = np.r_[tree.query(q)[0], cKDTree(q).query(target)[0]]
    return mat, float(np.percentile(distances, 95)), result.x

def outline_matches(points, ep, matrix, hr):
    if not len(points) or not len(ep):
        return []
    raw = hr.transform(points, cv2.invertAffineTransform(matrix))
    distance, ids = cKDTree(ep).query(raw)
    # Only already z-buffered exterior scan skin is eligible. Never use a
    # skin/hair occlusion boundary or a helmet outline as an equality target.
    return [[q.tolist(), ep[k].tolist(), 1.] for q, k, d in zip(points, ids, distance) if d <= 24.]


def run(folder, source_run, ensemble):
    hr = import_module('mh-head-registration')
    directory = folder/'views/suit-back'
    out = directory/'registration'
    out.mkdir(exist_ok=False)
    side, back = json.loads((directory/'back.json').read_text())['views']
    original = json.loads((folder/'views/suit_contours.ensemble-all.json').read_text())
    context = hr.source_context(folder, source_run)
    labels, box, ramp = hr.skin_raster(context, side)
    # Profile target already has x116 registration; no rescaling from the back.
    target_mask = np.zeros((1024, 768), np.uint8)
    cv2.fillPoly(target_mask, [np.rint(c).astype(np.int32) for c in original['head_views'][side]], 1)
    profile_box = (box[0], np.array([box[1][0], ramp[0]]))
    target_points = boundary(target_mask.astype(bool), profile_box)
    back_labels, back_box, back_ramp = hr.skin_raster(context, back)
    old_back = cv2.imread(str(folder/f'views/{back}.png'))
    train, infos, held, panels = {}, {}, {}, []
    for member in ['e1', 'e2', 'e3']:
        profile = cv2.imread(str(directory/f'{member}_{side}.png'))
        image = cv2.imread(str(directory/f'{member}_{back}.png'))
        assert profile.shape == image.shape == (1024, 768, 3)
        mat, residual, params = profile_similarity(boundary(ensemble.figure(profile), profile_box), target_points, np.array([384., 512.]), hr)
        scale, angle, tx, ty = params
        # Transfer scale and vertical placement through the calibrated camera.
        # Profile horizontal translation/rotation acts in depth, so it cannot
        # be copied as lateral movement/roll in the back camera.
        profile_points = boundary(ensemble.figure(profile), profile_box)
        vertical_shift = float(np.mean(hr.transform(profile_points, mat)[:, 1] - scale*profile_points[:, 1]))
        bm = np.array([[scale, 0., (1.-scale)*384.], [0., scale, vertical_shift]])
        points, templates = hr.skin_anchors(old_back, image, back_labels, back_ramp)
        ep = boundary(ensemble.figure(image), back_box)
        tree = cKDTree(ep)
        # Full 3D-head placement and skin-constrained refinement happen inside
        # the native <=3-round solver, where depth and occlusion are available.
        registered = cv2.warpAffine(image, bm, (768, 1024), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        pm = cv2.warpAffine(profile, mat, (768, 1024), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        cv2.imwrite(str(out/f'{member}_{side}-reg.png'), pm)
        cv2.imwrite(str(out/f'{member}_{back}-reg.png'), registered)
        mask = ensemble.figure(registered)
        inv = cv2.invertAffineTransform(bm)
        d = tree.query(hr.transform(points, inv))[0]*scale if len(points) else np.array([])
        info = {'accepted': member != 'e3', 'reasons': ['held_out'] if member == 'e3' else [],
            'same_sheet_profile': side, 'profile_p95_px': residual, 'profile_matrix': mat.tolist(),
            'similarity': {'scale': float(scale), 'rotation_deg': float(np.rad2deg(np.arctan2(bm[1, 0], bm[0, 0]))), 'matrix': bm.tolist()},
            'skin_outline_points': len(points), 'skin_outline_p95_px': float(np.percentile(d, 95)) if len(d) else None,
            'texture_matches_detail': templates, 'outline_matches_detail': outline_matches(points, ep, bm, hr), 'visible_scan_skin_pixels': int((back_labels == 1).sum()), 'scale_source': 'same_sheet_profile_to_existing_registered_profile',
            'ramp_y_px': list(back_ramp), 'crop': [*back_box[0].tolist(), *back_box[1].tolist()]}
        infos[member] = {back: info}
        if member != 'e3': train[member] = mask
        else: held[back] = hr.curves_for(mask, ensemble.contours_mod)
        panels.append(registered)
    # Simultaneous leave-one-training-member-out screening. E3 is never consulted.
    crop = np.s_[back_box[0][1]:int(back_ramp[0]), back_box[0][0]:back_box[1][0]]
    rejected = []
    for member, mask in train.items():
        other = np.mean([ensemble.sdf(m) for n, m in train.items() if n != member], axis=0) > 0
        score = ensemble.iou(mask[crop], other[crop])
        infos[member][back]['head_iou_other_training_members'] = score
        if score < MIN_HEAD_IOU: rejected.append(member)
    for member in rejected:
        del train[member];infos[member][back]['accepted'] = False
        infos[member][back]['reasons'].append('head_iou_below_0.80')
    if not train: raise RuntimeError('No accepted back training member; no fit channel written')
    fields = np.stack([ensemble.sdf(m) for m in train.values()])
    curves = hr.curves_for(fields.mean(0) > 0, ensemble.contours_mod)
    spread = fields.std(0)
    weights = [np.round(1/(1+(ensemble.sample(spread, np.array(c))/ensemble.SPREAD_PX)**2), 3).tolist() for c in curves]
    data = {'head_views': {back: curves}, 'head_weights': {back: weights}, 'head_scale_ties': {back: side},
        'held_out_head_views': held, 'head_registration': {'members': infos,
        'views': {back: {'accepted_training_members': list(train), 'held_out_member': 'e3', 'missing_consensus': False}},
        'minimum_head_iou': MIN_HEAD_IOU, 'back_view': back, 'profile_view': side,
        'source_run': str(source_run), 'trust_weight': .8, 'held_out_member': 'e3'}}
    # Independent pre-fit intermember silhouette spread in calibrated mm.
    edges = [boundary(m, (back_box[0], np.array([back_box[1][0], back_ramp[0]]))) for m in train.values()]
    spread_mm = 0.
    if len(edges) > 1:
        spread_mm = float(np.percentile(np.r_[cKDTree(edges[0]).query(edges[1])[0], cKDTree(edges[1]).query(edges[0])[0]], 95)*context[0][back]['meters_per_pixel']*1000)
    data['head_registration']['member_silhouette_spread_mm'] = spread_mm
    data['head_registration']['back_gate_mm'] = max(3., spread_mm/2)
    (out/'head-contours.json').write_text(json.dumps(data, indent=2)+'\n')
    cv2.imwrite(str(out/'members.png'), np.hstack([p[max(0,back_box[0][1]):int(back_ramp[1])+5, back_box[0][0]:back_box[1][0]] for p in panels]))
    return {'back': back, 'profile': side, 'members': infos, 'spread_mm': spread_mm, 'output': str(out)}
