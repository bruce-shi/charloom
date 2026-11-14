 use crate::geom::{V, a, proper_rotation, v};
use crate::scan::{Scan, build_query};
use crate::segment::{Label, Segmentation};
use nalgebra::Matrix3;
use rayon::prelude::*;
use serde_json::{Value, json};

/// Rounds of closest points and solves, and the data weight's rise over
/// them (against the edges' unit weight).
const ROUNDS: usize = 20;
/// (Ending at 200, every vertex pinned on its target, the edges could not
/// smooth what the closest points left: creases.)
const DATA_WEIGHT: (f64, f64) = (1.0, 50.0);
 const REACH: (f64, f64) = (0.05, 0.012);
const REVERSE_REACH: f64 = 0.06;
 const SCALE: (f64, f64) = (0.5, 3.0);
 const BODY_HOLD: f64 = 1e4;
/// Tangential relaxation on the scan after the solve: passes, and how
/// close to the scan's skin a vertex must lie to take part, metres.
const RELAX_PASSES: usize = 8;
const RELAX_ON: f64 = 0.002;
const ARMATURE_HOLD: f64 = 20.0;
const SKULL_HOLD: f64 = 2.0;
/// Vertices without data of their own stay this far inside the scan's skin
/// (within CAVITY_REACH of it, share of the height), held there this hard.
const INSIDE_M: f64 = 0.001;
const CAVITY_REACH: f64 = 0.01;
const INSIDE_HOLD: f64 = 20.0;

pub struct Head<'a> {
    pub neighbours: &'a [Vec<u32>],
    pub triangles: &'a [[u32; 3]],
    /// Vertices the sculpt may move (the head); all others are held.
    pub free: &'a [bool],
    /// Free vertices that may take skin data (not cavities, lids, eye
    /// sockets).
    pub data: &'a [bool],
    /// Free vertices held where the fit put them (lids, landmarks).
    pub armature: &'a [bool],
    pub scalp_hold: &'a super::suit_head::Hold,
    pub skin_only_query: &'a [bool],
         pub eyes: &'a [(V, f64)],
    pub openings: &'a [crate::MarkedOpening],
    pub guard_surface: bool,
    pub height: f64,
}

fn in_eye(eyes: &[(V, f64)], openings: &[crate::MarkedOpening], p: V) -> bool {
    let near=eyes.iter().any(|(c, r)| (p - c).norm() < *r);
    near && (openings.is_empty() || openings.iter().any(|o| {
        let (s,t)=o.st(p);o.opening.depth(s,t)>=0.0
    }))
}

fn normals(x: &[V], triangles: &[[u32; 3]]) -> Vec<V> {
    let mut n = vec![V::zeros(); x.len()];
    for t in triangles {
        let [i, j, k] = t.map(|i| i as usize);
        let f = (x[j] - x[i]).cross(&(x[k] - x[i]));
        n[i] += f;
        n[j] += f;
        n[k] += f;
    }
    n.iter()
        .map(|m| if m.norm() > 1e-12 { m.normalize() } else { *m })
        .collect()
}

pub fn sculpt(head: &Head, x: &mut [V], scan: &Scan, seg: &Segmentation) -> Value {
    let n = x.len();
    let x0: Vec<V> = x.to_vec();
    // Prevent new external folds while allowing the interior mouth/eye
    // lining to keep its existing containment treatment. In particular a
    // visible ear must not gain inverted triangles while following skin.
    let watched: Vec<_> = head.triangles.iter().filter(|t|t.iter().all(|&i|head.data[i as usize])).copied().collect();
    let rest_normals: Vec<_> = watched.iter().map(|t| {
        let [a,b,c]=t.map(|i|x0[i as usize]);(b-a).cross(&(c-a))
    }).collect();
    let surface_guard=head.guard_surface.then(||super::guard::Guard::new(
        &x0,&watched,&rest_normals,head.neighbours,head.data,0.008));
    let mut guard_reports=vec![];
    let skin_tri = |t: usize| seg.triangle_label[t] == Label::Skin && !seg.is_prop(t);
    let visible_triangles:Vec<_>=scan.triangles.iter().enumerate().filter(|(i,t)|
        skin_tri(*i) && t.iter().any(|&s|head.scalp_hold.scan_seen[s as usize])).map(|(i,_)|i).collect();
    let skin_query=build_query(&scan.positions,&visible_triangles.iter().map(|&i|scan.triangles[i]).collect::<Vec<_>>());
    let skin_point=|i:usize,point,reach,normal|{
        if head.guard_surface && head.skin_only_query[i] {
            let mut hit=skin_query.closest_point(point,reach,normal)?;
            hit.source[1]=visible_triangles[hit.source[1]];
            Some(hit)
        }else{scan.query.closest_point(point,reach,normal)}
    };
    // Scan skin samples near the head (its box, padded).
    let (mut lo, mut hi) = (V::repeat(f64::INFINITY), V::repeat(f64::NEG_INFINITY));
    for i in (0..n).filter(|&i| head.data[i]) {
        lo = lo.inf(&x[i]);
        hi = hi.sup(&x[i]);
    }
    let pad = V::repeat(REACH.0 * head.height);
    let samples: Vec<usize> = (0..scan.weld.count)
        .map(|w| scan.weld.representative[w])
        .filter(|&r| {
            let p = v(scan.positions[r]);
            seg.vertex_label[r] == Label::Skin
                && (0..3).all(|k| p[k] >= lo[k] - pad[k] && p[k] <= hi[k] + pad[k])
                && !in_eye(head.eyes, head.openings, p)
        })
        .collect();
    let stride = (samples.len() / 40000).max(1);
    let samples: Vec<usize> = samples.into_iter().step_by(stride).collect();
    let distance_stats = |x: &[V]| -> (usize, f64, f64) {
        let mut d: Vec<f64> = (0..n)
            .into_par_iter()
            .filter(|&i| head.data[i] && !in_eye(head.eyes, head.openings, x[i]))
            .filter_map(|i| {
                let h = scan
                    .query
                    .closest_point(a(x[i]), 0.01 * head.height, None)?;
                skin_tri(h.source[1]).then_some(h.distance_m)
            })
            .collect();
        d.sort_by(f64::total_cmp);
        let pct = |f: f64| {
            d.get(((d.len().max(1) - 1) as f64 * f) as usize)
                .copied()
                .unwrap_or(0.0)
        };
        (d.len(), pct(0.5), pct(0.95))
    };
    let before = distance_stats(x);
    let adj: Vec<Vec<(u32, f64)>> = head
        .neighbours
        .iter()
        .map(|r| r.iter().map(|&j| (j, 1.0)).collect())
        .collect();
    let mut frames = vec![Matrix3::<f64>::identity(); n];
    let mut residual = 0.0;
    let mut on_skin = 0;
    let mut pushed_in = 0usize;
    for round in 0..ROUNDS {
        let before_step=x.to_vec();
        let s = round as f64 / (ROUNDS - 1) as f64;
        let weight = DATA_WEIGHT.0 * (DATA_WEIGHT.1 / DATA_WEIGHT.0).powf(s);
        let reach = (REACH.0 + (REACH.1 - REACH.0) * s) * head.height;
        let nrm = normals(x, head.triangles);
        // Forward: each data vertex to the scan's skin (orientation-gated).
        let forward: Vec<Option<V>> = (0..n)
            .into_par_iter()
            .map(|i| {
                if !head.data[i] || in_eye(head.eyes, head.openings, x[i]) {
                    return None;
                }
                let h = skin_point(i,a(x[i]), reach, Some((a(nrm[i]), 0.3)))?;
                 
                let (q, nq) = (v(h.point), v(h.normal));
                (skin_tri(h.source[1]) && !in_eye(head.eyes, head.openings, q)
                    && (!head.guard_surface || scan.triangles[h.source[1]].iter()
                        .any(|&s|head.scalp_hold.scan_seen[s as usize])))
                    .then(|| x[i] + nq * (q - x[i]).dot(&nq))
            })
            .collect();
        // Reverse: each skin sample to the nearest data vertex, so the
        // scan's far parts (a muzzle tip, an ear's rim) are reached too.
        let body_query = build_query(&x.iter().map(|p| a(*p)).collect::<Vec<_>>(), head.triangles);
        let reverse: Vec<(usize, V)> = samples
            .par_iter()
            .filter_map(|&r| {
                let h = body_query.closest_point(
                    scan.positions[r],
                    REVERSE_REACH * head.height,
                    Some((scan.normals[r], 0.3)),
                )?;
                let t = head.triangles[h.source[1]];
                let k = (0..3).max_by(|&p, &q| h.barycentric[p].total_cmp(&h.barycentric[q]))?;
                let i = t[k] as usize;
                (head.data[i] && !in_eye(head.eyes, head.openings, x[i])
                    && (!head.guard_surface || head.scalp_hold.scan_seen[r])).then(|| (i, v(scan.positions[r])))
            })
            .collect();
        let mut rsum = vec![V::zeros(); n];
        let mut rcount = vec![0.0f64; n];
        // Along the body's normal (point to plane), as the forward pull.
        for (i, q) in &reverse {
            rsum[*i] += x[*i] + nrm[*i] * (q - x[*i]).dot(&nrm[*i]);
            rcount[*i] += 1.0;
        }
        on_skin = 0;
        pushed_in = 0;
        let mut diag = vec![0.0f64; n];
        let mut target = vec![V::zeros(); n];
        for i in 0..n {
            if !head.free[i] || head.scalp_hold.hard(i) {
                diag[i] = BODY_HOLD;
                target[i] = x0[i];
                continue;
            }
            if head.armature[i] {
                diag[i] = ARMATURE_HOLD;
                target[i] = x0[i];
                continue;
            }
            let mut sum = V::zeros();
            let mut w = 0.0;
            if let Some(q) = forward[i] {
                sum += q;
                w += 1.0;
            }
            if rcount[i] > 0.0 {
                sum += rsum[i] / rcount[i];
                w += 1.0;
            }
            if w > 0.0 {
                on_skin += 1;
                diag[i] = weight;
                target[i] = sum / w;
            } else if head.data[i] && !in_eye(head.eyes, head.openings, x0[i]) {
                // No skin in reach: under hair, the measured skull holds.
                diag[i] = SKULL_HOLD;
                target[i] = x0[i];
            } else if !head.scalp_hold.gpt_scalp[i]
                && (!head.guard_surface || head.scalp_hold.classes[i]==0)
                && let Some(h) = scan
                    .query
                    .closest_point(a(x[i]), CAVITY_REACH * head.height, None)
                && skin_tri(h.source[1])
            {
                 
                let (q, nq) = (v(h.point), v(h.normal));
                let out = (x[i] - q).dot(&nq);
                if out > -INSIDE_M {
                    diag[i] = INSIDE_HOLD;
                    target[i] = x[i] - nq * (out + 2.0 * INSIDE_M);
                    pushed_in += 1;
                }
            }
        }
        let mut sums: Vec<V> = target.iter().zip(&diag).map(|(t, w)| t * *w).collect();
        head.scalp_hold.add_targets(&mut sums, &mut diag);
        for i in 0..n {
            if diag[i] > 0.0 {
                target[i] = sums[i] / diag[i];
            }
        }
        // As-rigid-as-possible: edges of the fitted head, turned by the
        // local rotations of the last solve.
        let rhs: Vec<V> = (0..n)
            .into_par_iter()
            .map(|i| {
                let e = head.neighbours[i].iter().fold(V::zeros(), |s, &j| {
                    let j = j as usize;
                    s + (frames[i] + frames[j]) * (x0[i] - x0[j]) * 0.5
                });
                e + target[i] * diag[i]
            })
            .collect();
        residual = head.scalp_hold.solve(&adj, &diag, &rhs, x, 400);
        if let Some(guard)=&surface_guard {
            guard_reports.push(guard.limit(&before_step,x,&frames).report);
        }
        head.scalp_hold.enforce(x);
        frames = (0..n)
            .into_par_iter()
            .map(|i| {
                if !head.free[i] {
                    return Matrix3::identity();
                }
                let (mut m, mut now, mut was) = (Matrix3::zeros(), 0.0, 0.0);
                for &j in &head.neighbours[i] {
                    let (e, e0) = (x[i] - x[j as usize], x0[i] - x0[j as usize]);
                    m += e * e0.transpose();
                    now += e.norm_squared();
                    was += e0.norm_squared();
                }
                let scale = if was > 1e-18 {
                    (now / was).sqrt().clamp(SCALE.0, SCALE.1)
                } else {
                    1.0
                };
                proper_rotation(m) * scale
            })
            .collect();
    }
     
    let mut relaxed = 0usize;
    for _ in 0..RELAX_PASSES {
        let before_step=x.to_vec();
        let nrm = normals(x, head.triangles);
        let moved: Vec<Option<V>> = (0..n)
            .into_par_iter()
            .map(|i| {
                if !head.data[i] || head.neighbours[i].is_empty() || in_eye(head.eyes, head.openings, x[i]) {
                    return None;
                }
                let h = skin_point(i,a(x[i]), RELAX_ON, None)?;
                if !skin_tri(h.source[1]) || head.guard_surface && !scan.triangles[h.source[1]].iter().any(|&s|head.scalp_hold.scan_seen[s as usize]) {
                    return None;
                }
                let m = head.neighbours[i]
                    .iter()
                    .fold(V::zeros(), |s, &j| s + x[j as usize])
                    / head.neighbours[i].len() as f64;
                let d = m - x[i];
                let along = d - nrm[i] * d.dot(&nrm[i]);
                let p = x[i] + along * 0.5;
                let h = skin_point(i,a(p), RELAX_ON * 2.0, None)?;
                (skin_tri(h.source[1]) && (!head.guard_surface || scan.triangles[h.source[1]].iter().any(|&s|head.scalp_hold.scan_seen[s as usize]))).then(|| v(h.point))
            })
            .collect();
        for (i, m) in moved.into_iter().enumerate() {
            if let Some(p) = m {
                x[i] = p;
                relaxed += 1;
            }
        }
        if let Some(guard)=&surface_guard {
            guard_reports.push(guard.limit(&before_step,x,&frames).report);
        }
        head.scalp_hold.enforce(x);
    }
    let after = distance_stats(x);
    let coverage = |x: &[V]| -> (f64, f64) {
        let q = build_query(&x.iter().map(|p| a(*p)).collect::<Vec<_>>(), head.triangles);
        let mut d: Vec<f64> = samples
            .par_iter()
            .map(|&r| {
                q.closest_point(scan.positions[r], 0.1 * head.height, None)
                    .map_or(0.1 * head.height, |h| h.distance_m)
            })
            .collect();
        d.sort_by(f64::total_cmp);
        let pct = |f: f64| {
            d.get(((d.len().max(1) - 1) as f64 * f) as usize)
                .copied()
                .unwrap_or(0.0)
        };
        (pct(0.5), pct(0.95))
    };
    let (cov_after, cov_before) = (coverage(x), coverage(&x0));
    let moved: Vec<f64> = (0..n)
        .filter(|&i| head.free[i])
        .map(|i| (x[i] - x0[i]).norm())
        .collect();
    let mm = |m: f64| (m * 1e4).round() / 10.0;
    json!({
        "applied": true,
        "free_vertices": head.free.iter().filter(|f| **f).count(),
        "on_skin_vertices": on_skin,
        "gpt_core_max_error_mm": head.scalp_hold.error_mm(x),
        "surface_guard":guard_reports,
        "relaxed_moves": relaxed,
        "kept_inside_vertices": pushed_in,
        "skin_samples": samples.len(),
        "visible_skin_mm": {"vertices": after.0, "p50_before": mm(before.1), "p95_before": mm(before.2), "p50_after": mm(after.1), "p95_after": mm(after.2)},
        "scan_skin_to_body_mm": {"p50_before": mm(cov_before.0), "p95_before": mm(cov_before.1), "p50_after": mm(cov_after.0), "p95_after": mm(cov_after.1)},
        "moved_max_mm": mm(moved.iter().copied().fold(0.0, f64::max)),
        "cg_residual": residual,
    })
}
