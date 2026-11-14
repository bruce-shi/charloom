 use crate::geom::{V, a, proper_rotation, v};
use crate::guide::{Camera, rasterize};
use crate::scan::Scan;
use crate::segment::{Label, Segmentation};
use nalgebra::Matrix3;
use rayon::prelude::*;
use serde_json::{Value, json};

/// A vertex this close to the scan's visible skin (share of the height)
/// shows.
const VISIBLE: f64 = 0.002;
 const LOOKS: usize = 40;
const LOOK_PX: usize = 384;
/// Skin seen from fewer than this share of the directions it faces is
/// hidden: glimpsed through gaps between braids or strands, it is no shape
/// to fit to.
const SEEN_SHARE: f64 = 0.35;
/// A neck vertex is hidden under hair when the scan's nearest surface
/// within this (share of the height) is hair: the neck under a collar is
/// the clothed body's (`fit::hidden`).
const HAIR_REACH: f64 = 0.03;
 const BAND: f64 = 0.015;
const DATA_WEIGHT: f64 = 40.0;
 const OUT_MAX: f64 = 0.008;
/// Hidden vertices that would leave the hair are held this far inside it
/// (share of the height), this hard.
const INSIDE_HAIR: f64 = 0.0012;
const HOLD: f64 = 200.0;
/// The head's scale against MakeHuman's within which it is restored.
const SCALE: (f64, f64) = (0.8, 1.25);

pub struct Input<'a> {
    pub neighbours: &'a [Vec<u32>],
    /// MakeHuman's measured rest shape, welded.
    pub shape: &'a [V],
    /// The head (scalp, face, jaw): hidden where it shows no skin.
    pub head: &'a [bool],
    /// The neck: hidden where it shows no skin and hair covers it.
    pub neck: &'a [bool],
    /// Continuous anatomical weights: several nape rows split their neck
    /// influence between neck01/02/03, so no individual neck bone wins.
    pub neck_weight: &'a [f64],
    pub skull_weight: &'a [f64],
    /// Never moved: ears and the ring round them, lids, eye sockets,
    /// cavities, marked landmarks.
    pub fixed: &'a [bool],
    pub height: f64,
    pub neck_limit: f64,
}

/// Scan vertices seen from outside, from LOOKS directions round a sphere
/// (centre, radius) holding the head: a vertex counts where it is no deeper
/// than the nearest surface at its pixel (depth buffers, so the scan's
/// sub-pixel triangles count too), from at least SEEN_SHARE of the
/// directions it faces.
pub(crate) fn seen_from_outside(scan: &Scan, centre: V, radius: f64) -> Vec<bool> {
    let golden = std::f64::consts::PI * (3.0 - 5f64.sqrt());
    let mpp = 2.0 * radius / LOOK_PX as f64;
    let cams: Vec<Camera> = (0..LOOKS)
        .map(|k| {
            let y = 1.0 - 2.0 * (k as f64 + 0.5) / LOOKS as f64;
            let r = (1.0 - y * y).sqrt();
            let th = golden * k as f64;
            let forward = -V::new(r * th.cos(), y, r * th.sin());
            let helper = if forward.y.abs() > 0.9 {
                V::x()
            } else {
                V::y()
            };
            let right = forward.cross(&helper).normalize();
            let up = right.cross(&forward).normalize();
            Camera {
                id: format!("look{k}"),
                width: LOOK_PX,
                height: LOOK_PX,
                center: a(centre),
                right: a(right),
                up: a(up),
                forward: a(forward),
                meters_per_pixel: mpp,
            }
        })
        .collect();
    let depths: Vec<Vec<f32>> = cams
        .par_iter()
        .map(|c| rasterize(c, &scan.positions, &scan.triangles).depth)
        .collect();
    let tol = 2.0 * mpp;
    (0..scan.positions.len())
        .into_par_iter()
        .map(|i| {
            let p = v(scan.positions[i]);
            if (p - centre).norm() > radius {
                return true;
            }
            let nrm = v(scan.normals[i]);
            let (mut facing, mut hits) = (0usize, 0usize);
            for (cam, depth) in cams.iter().zip(&depths) {
                if nrm.dot(&-v(cam.forward)) <= 0.2 {
                    continue;
                }
                facing += 1;
                let (px, py, z) = cam.project(scan.positions[i]);
                let (x, y) = (px.floor() as i64, py.floor() as i64);
                if x < 0 || y < 0 || x >= LOOK_PX as i64 || y >= LOOK_PX as i64 {
                    continue;
                }
                if z <= depth[y as usize * LOOK_PX + x as usize] as f64 + tol {
                    hits += 1;
                }
            }
            hits > 0 && hits as f64 >= SEEN_SHARE * facing.max(1) as f64
        })
        .collect()
}

/// A hair shell bounds a point along the ray from inside the skull.
/// Only outward crossings count: the inward side of a braid is not a
/// surface to grow the skull toward. No crossing means no hair constraint.
fn hair_ceiling(
    hair: &charloom_engine::surface_query::SurfaceQuery,
    centre: V,
    point: V,
    reach: f64,
    margin: f64,
) -> Option<V> {
    let delta = point - centre;
    let length = delta.norm();
    let direction = delta.try_normalize(1e-12)?;
    let boundary = hair.outward_ray_distance(a(centre), a(direction), reach)?;
    let limit = (boundary - margin).max(0.0);
    (length > limit).then(|| centre + direction * limit)
}

pub fn restore(inp: &Input, x: &mut [V], scan: &Scan, seg: &Segmentation) -> Value {
    let n = x.len();
    let h = inp.height;
    let head_ids: Vec<usize> = (0..n).filter(|&i| inp.head[i] || inp.neck[i]).collect();
    let centre = head_ids.iter().fold(V::zeros(), |s, &i| s + x[i]) / head_ids.len().max(1) as f64;
    let radius = head_ids
        .iter()
        .map(|&i| (x[i] - centre).norm())
        .fold(0.0, f64::max)
        * 1.3;
    let seen = seen_from_outside(scan, centre, radius);
    // Skin counts where it is seen; unseen skin-labelled surface is no skin.
    let label_near = |p: V, reach: f64| {
        scan.query.closest_point(a(p), reach * h, None).map(|hit| {
            let t = hit.source[1];
            let shown = scan.triangles[t].iter().any(|&k| seen[k as usize]);
            match seg.triangle_label[t] {
                Label::Skin if !shown => Label::Hair,
                l => l,
            }
        })
    };
    let hidden: Vec<bool> = (0..n)
        .into_par_iter()
        .map(|i| {
            if !(inp.head[i] || inp.neck[i]) || inp.fixed[i] {
                return false;
            }
            if label_near(x[i], VISIBLE) == Some(Label::Skin) {
                return false;
            }
            !inp.neck[i] || label_near(x[i], HAIR_REACH) == Some(Label::Hair)
        })
        .collect();
    let count = hidden.iter().filter(|h| **h).count();
    let candidates = (0..n).filter(|&i| inp.head[i] || inp.neck[i]).count();
    let fixed = (0..n)
        .filter(|&i| (inp.head[i] || inp.neck[i]) && inp.fixed[i])
        .count();
    let shown = (0..n)
        .filter(|&i| {
            (inp.head[i] || inp.neck[i])
                && !inp.fixed[i]
                && label_near(x[i], VISIBLE) == Some(Label::Skin)
        })
        .count();
    if count == 0 {
        return json!({"applied": false, "reason": "no hidden head"});
    }
    // The head's turn and scale from MakeHuman's rest (all of it, the face
    // included: the scalp alone may be all hidden).
    let ids: Vec<usize> = (0..n).filter(|&i| inp.head[i]).collect();
    let mean =
        |f: &dyn Fn(usize) -> V| ids.iter().fold(V::zeros(), |s, &i| s + f(i)) / ids.len() as f64;
    let (cx, cs) = (mean(&|i| x[i]), mean(&|i| inp.shape[i]));
    let (mut m, mut sx, mut ss) = (Matrix3::zeros(), 0.0, 0.0);
    for &i in &ids {
        let (p, q) = (x[i] - cx, inp.shape[i] - cs);
        m += p * q.transpose();
        sx += p.norm_squared();
        ss += q.norm_squared();
    }
    let rot = proper_rotation(m);
    let raw_scale = (sx / ss.max(1e-18)).sqrt();
     
    if !(SCALE.0..=SCALE.1).contains(&raw_scale) {
        return json!({"applied": false, "reason": "head scale outside MakeHuman's range (a cartoon head)", "scale": raw_scale, "hidden_vertices": count});
    }
    let scale = raw_scale;
     
    let t = cx - rot * cs * scale;
    let hair_triangles: Vec<[u32; 3]> = scan
        .triangles
        .iter()
        .enumerate()
        .filter(|(i, _)| seg.triangle_label[*i] == Label::Hair)
        .map(|(_, t)| *t)
        .collect();
    let hair = crate::scan::build_query(&scan.positions, &hair_triangles);
    let ceiling = |i: usize, point: V| {
        (hidden[i] && inp.head[i])
            .then(|| hair_ceiling(&hair, cx, point, radius * 2.0, INSIDE_HAIR * h))
            .flatten()
    };

     
    let base: Vec<V> = (0..n).map(|i| rot * inp.shape[i] * scale + t).collect();
    let lap = |y: &[V]| -> Vec<V> {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let nb = &inp.neighbours[i];
                nb.iter()
                    .fold(y[i] * nb.len() as f64, |s, &j| s - y[j as usize])
            })
            .collect()
    };
     
    let band = super::guard::surface_distance(inp.shape, inp.neighbours, &hidden, BAND * h);
    let in_band = |i: usize| {
        !hidden[i] && inp.neck[i] && !inp.fixed[i] && band[i].is_finite() && band[i] < BAND * h
    };
    let mut ease: Vec<f64> = (0..n)
        .map(|i| {
            if hidden[i] {
                0.0
            } else if in_band(i) {
                DATA_WEIGHT * (band[i] / (BAND * h)).powi(2)
            } else {
                f64::INFINITY
            }
        })
        .collect();
    let hid: Vec<usize> = (0..n).filter(|&i| ease[i].is_finite()).collect();
     
    let mut target: Vec<V> = (0..n).map(|i| x[i] - base[i]).collect();
    let known: Vec<V> = (0..n)
        .map(|i| {
            if ease[i].is_finite() {
                V::zeros()
            } else {
                x[i] - base[i]
            }
        })
        .collect();
    let ll_known = lap(&lap(&known));
    let dot = |a: &[V], b: &[V]| a.iter().zip(b).map(|(p, q)| p.dot(q)).sum::<f64>();
    let before: Vec<V> = hid.iter().map(|&i| x[i]).collect();
    let mut d: Vec<V> = hid.iter().map(|&i| x[i] - base[i]).collect();
    let mut residual = 0.0;
    let mut held = 0usize;
    for round in 0..4 {
        let rhs: Vec<V> = hid
            .iter()
            .map(|&i| -ll_known[i] + target[i] * ease[i])
            .collect();
        let apply = |p: &[V]| -> Vec<V> {
            let mut full = vec![V::zeros(); n];
            for (k, &i) in hid.iter().enumerate() {
                full[i] = p[k];
            }
            let ll = lap(&lap(&full));
            hid.iter()
                .enumerate()
                .map(|(k, &i)| ll[i] + p[k] * ease[i])
                .collect()
        };
        let ad = apply(&d);
        let mut r: Vec<V> = rhs.iter().zip(&ad).map(|(b, a)| b - a).collect();
        let mut q = r.clone();
        let mut rr = dot(&r, &r);
        let r0 = rr.max(1e-30);
        for _ in 0..4000 {
            if rr <= 1e-20 * r0 {
                break;
            }
            let aq = apply(&q);
            let alpha = rr / dot(&q, &aq).max(1e-300);
            for k in 0..d.len() {
                d[k] += q[k] * alpha;
                r[k] -= aq[k] * alpha;
            }
            let rr_new = dot(&r, &r);
            let beta = rr_new / rr;
            rr = rr_new;
            for k in 0..q.len() {
                q[k] = r[k] + q[k] * beta;
            }
        }
        residual = (rr / r0).sqrt();
        if round == 3 {
            break;
        }
        // Hidden vertices outside (or within INSIDE_HAIR of) the scan's
        // hair or garment surface over them.
        let out: Vec<Option<(usize, V)>> = hid
            .par_iter()
            .enumerate()
            .map(|(k, &i)| {
                let p = base[i] + d[k];
                if in_band(i) {
                    let m = (p - x[i]).norm();
                    return (m > inp.neck_limit)
                        .then(|| (i, x[i] + (p - x[i]) * (inp.neck_limit / m)));
                }
                if let Some(limit) = ceiling(i, p) {
                    return Some((i, limit));
                }
                let out_dir = (x[i] - cx).try_normalize(1e-12)?;
                let out = (p - x[i]).dot(&out_dir);
                if out > OUT_MAX * h {
                    return Some((i, p - out_dir * (out - OUT_MAX * h)));
                }
                let hit = scan.query.closest_point(a(p), HAIR_REACH * h, None)?;
                // Hair, a hat, or skin that shows (a grafted ear): never
                // the skin-labelled surface no one sees under the hair.
                let t = hit.source[1];
                let skin = seg.triangle_label[t] == Label::Skin;
                if skin && !scan.triangles[t].iter().any(|&k| seen[k as usize]) {
                    return None;
                }
                // Under hair a gap; shown skin only bounds (the eased nape
                // stays on its neck, not dented under it).
                let gap = if skin { 0.0 } else { INSIDE_HAIR * h };
                let (qp, nq) = (v(hit.point), v(hit.normal));
                 
                if nq.dot(&out_dir) < 0.5 {
                    return None;
                }
                let above = (p - qp).dot(&nq);
                (above > -gap).then(|| (i, p - nq * (above + gap)))
            })
            .collect();
        let new: Vec<(usize, V)> = out
            .into_iter()
            .flatten()
            .filter(|(i, _)| ease[*i] < HOLD)
            .collect();
        if new.is_empty() {
            break;
        }
        for (i, p) in new {
            ease[i] = HOLD;
            target[i] = p - base[i];
            held += 1;
        }
    }
    let mut y: Vec<V> = hid.iter().zip(&d).map(|(&i, d)| base[i] + d).collect();
    let mut hair_projected = 0usize;
    let mut hair_projection_max = 0.0f64;
    for (k, &i) in hid.iter().enumerate() {
        // A penalty is not a travel bound. The visible neck must not move
        // farther than the graft threshold even after the last solve.
        if inp.neck[i] {
            let delta = y[k] - x[i];
            let limit = inp.neck_limit - 1e-6;
            if delta.norm() > limit {
                y[k] = x[i] + delta.normalize() * limit;
            }
        }
        if let Some(limit) = ceiling(i, y[k]) {
            hair_projected += 1;
            hair_projection_max = hair_projection_max.max((y[k] - limit).norm());
            y[k] = limit;
        }
    }
    let original = x.to_vec();
    for (k, &i) in hid.iter().enumerate() {
        x[i] = y[k];
    }
    let nape = fair_nape(inp, x, &original, &hidden, &ceiling);
    let mut moved: Vec<f64> = hid
        .iter()
        .zip(&before)
        .map(|(&i, q)| (x[i] - q).norm())
        .collect();
    moved.sort_by(f64::total_cmp);
    let pct = |f: f64| {
        moved
            .get(((moved.len() - 1) as f64 * f) as usize)
            .copied()
            .unwrap_or(0.0)
    };
    let mm = |m: f64| (m * 1e4).round() / 10.0;
    json!({
        "applied": true,
        "hidden_vertices": count,
        "candidates": candidates,
        "fixed": fixed,
        "shown": shown,
        "hidden_neck_vertices": (0..n).filter(|&i| hidden[i] && inp.neck[i]).count(),
        "held_inside_hair": held,
        "hair_radial_projected_vertices": hair_projected,
        "hair_radial_projection_max_mm": mm(hair_projection_max),
        "band_vertices": (0..n).filter(|&i| in_band(i)).count(),
        "scale": scale,
        "moved_mm": {"p50": mm(pct(0.5)), "p95": mm(pct(0.95)), "max": mm(pct(1.0))},
        "cg_residual": residual,
        "nape_fairing": nape,
        "skull_frame": {"source": "whole_head"},
    })
}

 fn fair_nape(
    inp: &Input,
    x: &mut [V],
    fitted: &[V],
    hidden: &[bool],
    ceiling: &dyn Fn(usize, V) -> Option<V>,
) -> Value {
    let n = x.len();
    let skull: Vec<usize> = (0..n).filter(|&i| inp.skull_weight[i] > 0.9).collect();
    if skull.is_empty() {
        return json!({"vertices": 0});
    }
    let back = skull.iter().map(|&i| inp.shape[i].z).sum::<f64>() / skull.len() as f64;
    let seed: Vec<bool> = (0..n)
        .map(|i| {
            inp.neck_weight[i] > 0.05
                && inp.skull_weight[i] > 0.02
                && inp.shape[i].z < back
                && !inp.fixed[i]
        })
        .collect();
    // Three and a half percent of stature reaches the lower occiput from
    // the mixed neck rows; it does not alter the crown or front of the head.
    let reach = 0.035 * inp.height;
    let distance = super::guard::surface_distance(inp.shape, inp.neighbours, &seed, reach);
    let free: Vec<usize> = (0..n)
        .filter(|&i| {
            distance[i] < reach
                && inp.skull_weight[i] + inp.neck_weight[i] > 0.8
                && inp.shape[i].z < back
                && !inp.fixed[i]
                && (hidden[i] || seed[i])
        })
        .collect();
    if free.is_empty() {
        return json!({"vertices": 0});
    }
    let lap = |y: &[V]| -> Vec<V> {
        (0..n)
            .into_par_iter()
            .map(|i| {
                let nb = &inp.neighbours[i];
                nb.iter()
                    .fold(y[i] * nb.len() as f64, |s, &j| s - y[j as usize])
            })
            .collect()
    };
    let original = x.to_vec();
    let mut constrained = vec![false; n];
    let dot = |p: &[V], q: &[V]| p.iter().zip(q).map(|(p, q)| p.dot(q)).sum::<f64>();
    let (mut capped, mut hair_held, mut conflicts, mut rounds) = (0usize, 0usize, 0usize, 0usize);
    let mut residual = 0.0;
     
    for _ in 0..=free.len() {
        let active: Vec<usize> = free.iter().copied().filter(|&i| !constrained[i]).collect();
        if active.is_empty() {
            break;
        }
        rounds += 1;
        let weight: Vec<f64> = active
            .iter()
            .map(|&i| if hidden[i] { 0.0 } else { 1.0 })
            .collect();
        let mut known = x.to_vec();
        for &i in &active {
            known[i] = V::zeros();
        }
        let ll = lap(&lap(&known));
        let rhs: Vec<V> = active
            .iter()
            .zip(&weight)
            .map(|(&i, &w)| -ll[i] + fitted[i] * w)
            .collect();
        let apply = |p: &[V]| -> Vec<V> {
            let mut full = vec![V::zeros(); n];
            for (k, &i) in active.iter().enumerate() {
                full[i] = p[k];
            }
            let ll = lap(&lap(&full));
            active
                .iter()
                .enumerate()
                .map(|(k, &i)| ll[i] + p[k] * weight[k])
                .collect()
        };
        let mut y: Vec<V> = active.iter().map(|&i| x[i]).collect();
        let ay = apply(&y);
        let mut r: Vec<V> = rhs.iter().zip(&ay).map(|(b, a)| b - a).collect();
        let mut q = r.clone();
        let mut rr = dot(&r, &r);
        let r0 = rr.max(1e-30);
        for _ in 0..4000 {
            if rr <= 1e-20 * r0 {
                break;
            }
            let aq = apply(&q);
            let alpha = rr / dot(&q, &aq).max(1e-300);
            for k in 0..y.len() {
                y[k] += q[k] * alpha;
                r[k] -= aq[k] * alpha;
            }
            let next = dot(&r, &r);
            let beta = next / rr;
            rr = next;
            for k in 0..q.len() {
                q[k] = r[k] + q[k] * beta;
            }
        }
        residual = (rr / r0).sqrt();
        let mut added = 0;
        for (k, &i) in active.iter().enumerate() {
            let candidate = y[k];
            if let Some(limit) = ceiling(i, y[k]) {
                y[k] = limit;
                hair_held += 1;
            }
            if !hidden[i] || inp.neck[i] {
                let delta = y[k] - fitted[i];
                let limit = inp.neck_limit - 1e-6;
                if delta.norm() > limit {
                    y[k] = fitted[i] + delta.normalize() * limit;
                    capped += 1;
                     
                    if ceiling(i, y[k]).is_some() {
                        y[k] = fitted[i];
                        conflicts += 1;
                    }
                }
            }
            if (y[k] - candidate).norm() > 1e-8 {
                constrained[i] = true;
                added += 1;
            }
            x[i] = y[k];
        }
        if added == 0 {
            break;
        }
    }
    let max_move = free
        .iter()
        .map(|&i| (x[i] - original[i]).norm())
        .fold(0.0, f64::max);
    let max_visible_move = free
        .iter()
        .filter(|&&i| !hidden[i] || inp.neck[i])
        .map(|&i| (x[i] - fitted[i]).norm())
        .fold(0.0, f64::max);
    json!({"vertices":free.len(), "capped_visible_vertices":capped,
        "hair_constraints":hair_held, "constraint_conflicts":conflicts, "rounds":rounds,
        "max_move_mm":max_move*1000.0, "max_visible_move_mm":max_visible_move*1000.0,
        "visible_move_limit_mm":inp.neck_limit*1000.0, "cg_residual":residual})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nape_fairing_resolves_around_bounds_and_preserves_fixed_skin() {
        let side = 9usize;
        let n = side * side;
        let mut neighbours = vec![vec![]; n];
        let mut shape = vec![V::zeros(); n];
        let mut fixed = vec![false; n];
        let mut hidden = vec![false; n];
        let mut neck = vec![false; n];
        let mut neck_weight = vec![0.0; n];
        let mut skull_weight = vec![1.0; n];
        for row in 0..side {
            for col in 0..side {
                let i = row * side + col;
                shape[i] = V::new(
                    (col as f64 - 4.0) * 0.01,
                    1.0 + row as f64 * 0.01,
                    -0.05 + (row as f64 - 4.0).powi(2) * 0.001,
                );
                fixed[i] = row == 0 || row == side - 1 || col == 0 || col == side - 1;
                hidden[i] = !fixed[i] && row != 4;
                if (3..=5).contains(&row) {
                    neck_weight[i] = 0.2;
                    skull_weight[i] = 0.8;
                }
                neck[i] = row == 4;
                for (dr, dc) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                    let (r, c) = (row as isize + dr, col as isize + dc);
                    if r >= 0 && c >= 0 && r < side as isize && c < side as isize {
                        neighbours[i].push((r as usize * side + c as usize) as u32);
                    }
                }
            }
        }
        let head = vec![true; n];
        let mut fitted = shape.clone();
        for i in 0..n {
            if !fixed[i] && i / side >= 4 {
                fitted[i].z += 0.02;
            }
        }
        let mut x = fitted.clone();
        let input = Input {
            neighbours: &neighbours,
            shape: &shape,
            head: &head,
            neck: &neck,
            neck_weight: &neck_weight,
            skull_weight: &skull_weight,
            fixed: &fixed,
            height: 2.0,
            neck_limit: 0.002,
        };
        let report = fair_nape(&input, &mut x, &fitted, &hidden, &|_, _| None);
        assert!(report["rounds"].as_u64().unwrap() >= 2);
        assert!(report["capped_visible_vertices"].as_u64().unwrap() > 0);
        for i in 0..n {
            assert!(x[i].iter().all(|v| v.is_finite()));
            if fixed[i] {
                assert_eq!(x[i], fitted[i]);
            }
            if !hidden[i] {
                assert!((x[i] - fitted[i]).norm() <= input.neck_limit);
            }
        }
        assert!(x.iter().zip(&fitted).any(|(a, b)| (a - b).norm() > 0.003));
    }

    #[test]
    fn radial_hair_ceiling_keeps_clearance_without_growing_a_skull() {
        let hair = crate::scan::build_query(
            &[
                [-1.0, -1.0, 0.1],
                [1.0, -1.0, 0.1],
                [1.0, 1.0, 0.1],
                [-1.0, 1.0, 0.1],
            ],
            &[[0, 1, 2], [0, 2, 3]],
        );
        let centre = V::zeros();
        let limit = hair_ceiling(&hair, centre, V::new(0.0, 0.0, 0.12), 1.0, 0.002).unwrap();
        assert!((limit.z - 0.098).abs() < 1e-12);
        assert!(hair_ceiling(&hair, centre, V::new(0.0, 0.0, 0.08), 1.0, 0.002).is_none());
        assert!(hair_ceiling(&hair, centre, V::new(0.0, 0.0, -0.12), 1.0, 0.002).is_none());
    }
}
