 use super::silhouette;
use crate::geom::V;
use crate::prior::Region;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

pub struct Settings {
    /// RBF support radius along the surface (m).
    pub support: f64,
    /// Spacing of the RBF centres (m).
    pub spacing: f64,
    /// Geodesic falloff from pinned vertices to full freedom (m).
    pub falloff: f64,
    /// Correspondence reach in the image (m).
    pub reach: f64,
    pub iterations: usize,
    /// Smoothness between neighbouring centres and plain damping.
    pub smooth: f64,
    pub damping: f64,
    /// Largest displacement the field may make (m).
    pub cap: f64,
    /// Arms pinned beyond this geodesic distance from the trunk (m); the
    /// shoulder band within it stays free and constrained in every view.
    /// Pinning the whole arm (y3) left the deltoids and the upper chest
    /// they overhang standing 2-3 cm outside the suit's side outline.
    /// None frees the arms entirely.
    pub shoulder: Option<f64>,
}

#[derive(PartialEq)]
struct Item(f64, usize);
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0)
    }
}

/// Geodesic (edge-path) distances from `sources`, up to `limit`.
fn dijkstra(x: &[V], neighbours: &[Vec<u32>], sources: &[usize], limit: f64) -> HashMap<usize, f64> {
    let mut dist: HashMap<usize, f64> = HashMap::new();
    let mut heap = BinaryHeap::new();
    for &s in sources {
        dist.insert(s, 0.0);
        heap.push(Item(0.0, s));
    }
    while let Some(Item(d, i)) = heap.pop() {
        if d > dist.get(&i).copied().unwrap_or(f64::INFINITY) {
            continue;
        }
        for &j in &neighbours[i] {
            let j = j as usize;
            let nd = d + (x[i] - x[j]).norm();
            if nd <= limit && nd < dist.get(&j).copied().unwrap_or(f64::INFINITY) {
                dist.insert(j, nd);
                heap.push(Item(nd, j));
            }
        }
    }
    dist
}

fn wendland(d: f64, r: f64) -> f64 {
    let t = (d / r).clamp(0.0, 1.0);
    (1.0 - t).powi(4) * (4.0 * t + 1.0)
}

fn smoothstep(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[allow(clippy::too_many_arguments)]
pub fn fit(
    views: &[silhouette::View],
    x: &mut [V],
    triangles: &[[u32; 3]],
    neighbours: &[Vec<u32>],
    region: &[Region],
    pinned: &[bool],
    s: &Settings,
) -> Value {
    let n = x.len();
    if views.is_empty() {
        return json!({"applied": false, "reason": "no suit contours"});
    }
    let base = x.to_vec();
    // The shoulder band: arm vertices near the trunk along the surface.
    let shoulder: Vec<bool> = match s.shoulder {
        Some(r) => {
            let trunk: Vec<usize> = (0..n).filter(|&i| region[i] == Region::Torso).collect();
            let near = dijkstra(&base, neighbours, &trunk, r);
            (0..n).map(|i| region[i] == Region::Arm && near.contains_key(&i)).collect()
        }
        None => (0..n).map(|i| region[i] == Region::Arm).collect(),
    };
    let pinned: Vec<bool> = (0..n).map(|i| pinned[i] || (region[i] == Region::Arm && !shoulder[i])).collect();
    // Freedom: 0 on pinned vertices, rising to 1 over the falloff.
    let seeds: Vec<usize> = (0..n).filter(|&i| pinned[i]).collect();
    let near = dijkstra(&base, neighbours, &seeds, s.falloff);
    let free: Vec<f64> = (0..n)
        .map(|i| near.get(&i).map_or(1.0, |&d| smoothstep(d / s.falloff)))
        .collect();
    // Centres: farthest-point sampling over the free surface.
    let candidates: Vec<usize> = (0..n).filter(|&i| free[i] > 0.05).collect();
    if candidates.is_empty() {
        return json!({"applied": false, "reason": "nothing free"});
    }
    let mut centres = vec![candidates[0]];
    let mut gap: Vec<f64> = candidates.iter().map(|&i| (base[i] - base[candidates[0]]).norm()).collect();
    loop {
        let (k, &g) = gap.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap();
        if g < s.spacing {
            break;
        }
        let c = candidates[k];
        centres.push(c);
        for (j, &i) in candidates.iter().enumerate() {
            gap[j] = gap[j].min((base[i] - base[c]).norm());
        }
    }
    let k = centres.len();
    // Basis: per vertex, the centres reaching it along the surface.
    let reach: Vec<HashMap<usize, f64>> = centres
        .par_iter()
        .map(|&c| dijkstra(&base, neighbours, &[c], s.support))
        .collect();
    let mut phi: Vec<Vec<(usize, f64)>> = vec![vec![]; n];
    for (ci, m) in reach.iter().enumerate() {
        for (&i, &d) in m {
            let w = wendland(d, s.support) * free[i];
            if w > 1e-6 {
                phi[i].push((ci, w));
            }
        }
    }
    // Neighbouring centres for the smoothness term.
    let mut pairs = vec![];
    for (a, m) in reach.iter().enumerate() {
        for (b, &cb) in centres.iter().enumerate().skip(a + 1) {
            if let Some(&d) = m.get(&cb) {
                if d < 2.0 * s.spacing {
                    pairs.push((a, b));
                }
            }
        }
    }
    let movable: Vec<bool> = (0..n)
        .map(|i| free[i] > 0.05 && !matches!(region[i], Region::Head | Region::Hand | Region::Foot))
        .collect();
     
    let hands: Vec<V> = (0..n).filter(|&i| region[i] == Region::Hand).map(|i| base[i]).collect();
    let lateral = if hands.len() > 2 {
        let mean = hands.iter().fold(V::zeros(), |s, p| s + p) / hands.len() as f64;
        let cov = hands.iter().fold(nalgebra::Matrix3::zeros(), |m, p| m + (p - mean) * (p - mean).transpose());
        let e = cov.symmetric_eigen();
        let k = e.eigenvalues.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        e.eigenvectors.column(k).into_owned()
    } else {
        V::x()
    };
    let masks: Vec<Vec<bool>> = views
        .iter()
        .map(|view| {
            let side = crate::geom::v(view.camera().right).dot(&lateral).abs() < 0.5;
            (0..n).map(|i| movable[i] && !(side && region[i] == Region::Arm && !shoulder[i])).collect()
        })
        .collect();
    let constraints = |x: &[V]| {
        let mut rows = vec![];
        let mut stats = vec![];
        for (j, m) in masks.iter().enumerate() {
            let (r, _, st) = silhouette::rendered_constraints(&views[j..j + 1], x, triangles, m, s.reach);
            rows.extend(r);
            stats.extend(st);
        }
        (rows, stats)
    };
    let mut c = vec![V::zeros(); k];
    let apply = |c: &[V], out: &mut [V]| {
        for i in 0..n {
            let d = phi[i].iter().fold(V::zeros(), |acc, &(ci, w)| acc + c[ci] * w);
            out[i] = base[i] + d;
        }
    };
    let mut history = vec![];
    let dim = 3 * k;
    for it in 0..s.iterations {
        let (rows, stats) = constraints(x);
        let total: f64 = rows.iter().map(|r| r.weight.max(0.0)).sum::<f64>().max(1.0);
        let mut m = DMatrix::<f64>::zeros(dim, dim);
        let mut rhs = DVector::<f64>::zeros(dim);
        let mut used = 0usize;
        for r in &rows {
            if r.weight <= 0.0 {
                continue;
            }
            used += 1;
            let w = r.weight / total;
            for axis in r.axes {
                // Jacobian of (point · axis) in the centre coefficients.
                let mut jac: HashMap<usize, f64> = HashMap::new();
                for (&vi, &b) in r.vertices.iter().zip(&r.bary) {
                    for &(ci, f) in &phi[vi as usize] {
                        for comp in 0..3 {
                            *jac.entry(3 * ci + comp).or_insert(0.0) += b * f * axis[comp];
                        }
                    }
                }
                let target = r.delta.dot(&axis);
                let entries: Vec<(usize, f64)> = jac.into_iter().filter(|e| e.1.abs() > 1e-12).collect();
                for &(a, ja) in &entries {
                    rhs[a] += w * ja * target;
                    for &(b, jb) in &entries {
                        m[(a, b)] += w * ja * jb;
                    }
                }
            }
        }
        // Regularisers on the total coefficients (increment solved).
        let scale = m.diagonal().iter().cloned().fold(0.0, f64::max).max(1e-12);
        let (ls, ld) = (s.smooth * scale, s.damping * scale);
        for &(a, b) in &pairs {
            for comp in 0..3 {
                let (ia, ib) = (3 * a + comp, 3 * b + comp);
                m[(ia, ia)] += ls;
                m[(ib, ib)] += ls;
                m[(ia, ib)] -= ls;
                m[(ib, ia)] -= ls;
                let diff = c[a][comp] - c[b][comp];
                rhs[ia] -= ls * diff;
                rhs[ib] += ls * diff;
            }
        }
        for a in 0..k {
            for comp in 0..3 {
                m[(3 * a + comp, 3 * a + comp)] += ld;
                rhs[3 * a + comp] -= ld * c[a][comp];
            }
        }
        let step = match m.clone().cholesky() {
            Some(ch) => ch.solve(&rhs),
            None => break,
        };
        for a in 0..k {
            c[a] += V::new(step[3 * a], step[3 * a + 1], step[3 * a + 2]);
        }
        apply(&c, x);
        // Global cap, preserving the field's shape.
        let max = (0..n).map(|i| (x[i] - base[i]).norm()).fold(0.0, f64::max);
        if max > s.cap {
            for ci in &mut c {
                *ci *= s.cap / max;
            }
            apply(&c, x);
        }
        history.push(json!({"iteration": it, "constraints": used,
            "views": stats.iter().map(|s| json!({"pairs": s.0, "p50_mm": s.1 * 1000.0, "p95_mm": s.2 * 1000.0})).collect::<Vec<_>>(),
            "max_move_mm": max.min(s.cap) * 1000.0}));
    }
    let (_, stats) = constraints(x);
    let side_views: Vec<bool> = views.iter().map(|view| crate::geom::v(view.camera().right).dot(&lateral).abs() < 0.5).collect();
    let moves: Vec<f64> = (0..n).map(|i| (x[i] - base[i]).norm()).collect();
    let mut sorted = moves.clone();
    sorted.sort_by(f64::total_cmp);
    json!({"applied": true, "centres": k, "support_m": s.support, "spacing_m": s.spacing,
        "falloff_m": s.falloff, "shoulder_m": s.shoulder, "shoulder_vertices": shoulder.iter().filter(|&&b| b).count(), "side_views_without_arms": side_views, "pinned_vertices": seeds.len(), "history": history,
        "final_views": stats.iter().map(|s| json!({"pairs": s.0, "p50_mm": s.1 * 1000.0, "p95_mm": s.2 * 1000.0})).collect::<Vec<_>>(),
        "move_max_mm": sorted.last().copied().unwrap_or(0.0) * 1000.0,
        "move_p95_mm": sorted[(sorted.len() as f64 * 0.95) as usize] * 1000.0})
}

 pub fn crotch_patch(x: &[V], neighbours: &[Vec<u32>], region: &[Region], radius: f64) -> Option<(usize, Vec<bool>)> {
    let n = x.len();
    let hands: Vec<V> = (0..n).filter(|&i| region[i] == Region::Hand).map(|i| x[i]).collect();
    if hands.len() < 3 {
        return None;
    }
    let mean = hands.iter().fold(V::zeros(), |s, p| s + p) / hands.len() as f64;
    let cov = hands.iter().fold(nalgebra::Matrix3::zeros(), |m, p| m + (p - mean) * (p - mean).transpose());
    let e = cov.symmetric_eigen();
    let k = e.eigenvalues.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let lateral: V = e.eigenvectors.column(k).into_owned();
    let trunk: Vec<f64> = (0..n).filter(|&i| region[i] == Region::Torso).map(|i| x[i].dot(&lateral)).collect();
    if trunk.is_empty() {
        return None;
    }
    let mid = trunk.iter().sum::<f64>() / trunk.len() as f64;
    let apex = (0..n)
        .filter(|&i| matches!(region[i], Region::Leg | Region::Torso) && (x[i].dot(&lateral) - mid).abs() < 0.01)
        .min_by(|&a, &b| x[a].y.total_cmp(&x[b].y))?;
    let near = dijkstra(x, neighbours, &[apex], radius);
    Some((apex, (0..n).map(|i| near.contains_key(&i)).collect()))
}

/// Crotch height from the front suit edit. Silhouette matching cannot lift
/// a crotch that sits too low: the arch is a narrow notch and the inner
/// thighs' outline beside it is always the nearer match (z10/z11 left the
/// thighs meeting 1.5-3 cm under the edit's arch). Here the edit's outline
/// is crossed with the body's midline (the arch top) and compared with the
/// lowest point of our crotch on the midline; the pelvis and upper thighs
/// are raised by the difference, fading out over `radius` (Wendland).
pub fn crotch_lift(views: &[silhouette::View], x: &mut [V], triangles: &[[u32; 3]], region: &[Region], pinned: &[bool], radius: f64, cap: f64) -> Value {
    let n = x.len();
    let hands: Vec<V> = (0..n).filter(|&i| region[i] == Region::Hand).map(|i| x[i]).collect();
    if hands.len() < 3 {
        return json!({"applied": false, "reason": "no hands"});
    }
    let mean = hands.iter().fold(V::zeros(), |s, p| s + p) / hands.len() as f64;
    let cov = hands.iter().fold(nalgebra::Matrix3::zeros(), |m, p| m + (p - mean) * (p - mean).transpose());
    let e = cov.symmetric_eigen();
    let k = e.eigenvalues.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let lateral: V = e.eigenvectors.column(k).into_owned();
    let Some(view) = views.iter().max_by(|a, b| {
        crate::geom::v(a.camera().right).dot(&lateral).abs().total_cmp(&crate::geom::v(b.camera().right).dot(&lateral).abs())
    }) else {
        return json!({"applied": false, "reason": "no views"});
    };
    let cam = view.camera();
    let mpp = cam.meters_per_pixel;
    let trunk: Vec<usize> = (0..n).filter(|&i| region[i] == Region::Torso).collect();
    let pelvis: Vec<usize> = (0..n).filter(|&i| region[i] == Region::Leg).collect();
    if trunk.is_empty() || pelvis.is_empty() {
        return json!({"applied": false, "reason": "no trunk"});
    }
    let project = |p: &V| cam.project(crate::geom::a(*p));
    // Our body in the front camera (coverage only).
    let (w, h) = (cam.width, cam.height);
    let mut cover = vec![false; w * h];
    let pr: Vec<(f64, f64, f64)> = x.iter().map(project).collect();
    for t in triangles {
        let p = t.map(|i| pr[i as usize]);
        let det = (p[1].1 - p[2].1) * (p[0].0 - p[2].0) + (p[2].0 - p[1].0) * (p[0].1 - p[2].1);
        if det.abs() < 1e-12 { continue; }
        let x0 = p.iter().map(|q| q.0).fold(f64::INFINITY, f64::min).floor().max(0.0) as usize;
        let x1 = (p.iter().map(|q| q.0).fold(f64::NEG_INFINITY, f64::max).ceil() as usize).min(w.saturating_sub(1));
        let y0 = p.iter().map(|q| q.1).fold(f64::INFINITY, f64::min).floor().max(0.0) as usize;
        let y1 = (p.iter().map(|q| q.1).fold(f64::NEG_INFINITY, f64::max).ceil() as usize).min(h.saturating_sub(1));
        for py in y0..=y1 { for px in x0..=x1 {
            let (u, v) = (px as f64 + 0.5, py as f64 + 0.5);
            let a = ((p[1].1 - p[2].1) * (u - p[2].0) + (p[2].0 - p[1].0) * (v - p[2].1)) / det;
            let b = ((p[2].1 - p[0].1) * (u - p[2].0) + (p[0].0 - p[2].0) * (v - p[2].1)) / det;
            if a >= -1e-9 && b >= -1e-9 && a + b <= 1.0 + 1e-9 { cover[py * w + px] = true; }
        }}
    }
    // Thigh gap: at a row a third of the way down the legs, the widest
    // uncovered run between the leftmost and rightmost leg pixels; from its
    // centre, up to the first covered pixel: our crotch.
    let leg_v: Vec<f64> = pelvis.iter().map(|&i| pr[i].1).collect();
    let (leg_top, leg_bottom) = leg_v.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v)));
    let row = ((leg_top + (leg_bottom - leg_top) * 0.45) as usize).min(h - 1);
    let legs: Vec<usize> = pelvis.iter().filter(|&&i| (pr[i].1 - row as f64).abs() < 3.0).map(|&i| pr[i].0.clamp(0.0, (w - 1) as f64) as usize).collect();
    let (Some(&lo), Some(&hi)) = (legs.iter().min(), legs.iter().max()) else {
        return json!({"applied": false, "reason": "no legs at the gap row"});
    };
    let mut best = (0usize, 0usize);
    let mut runstart = None;
    for px in lo..=hi {
        let free = !cover[row * w + px];
        match (free, runstart) {
            (true, None) => runstart = Some(px),
            (false, Some(s0)) => { if px - s0 > best.1 - best.0 { best = (s0, px); } runstart = None; }
            _ => {}
        }
    }
    if best.1 <= best.0 {
        return json!({"applied": false, "reason": "thighs closed at the gap row", "row": row});
    }
    let gu = (best.0 + best.1) / 2;
    let mut vy = row;
    while vy > 0 && !cover[vy * w + gu] { vy -= 1; }
    let ours_v = vy as f64 + 0.5;
    let mid_u = gu as f64 + 0.5;
    let tol = 0.012 / mpp;
    // The edit's arch: its outline over the same gap within 15 cm of ours.
    let window = 0.15 / mpp;
    let arch: Vec<f64> = view.samples().iter()
        .filter(|q| (q[0] - mid_u).abs() < tol && q[1] < ours_v + 0.03 / mpp && q[1] > ours_v - window)
        .map(|q| q[1])
        .collect();
    let Some(gpt_v) = arch.iter().cloned().reduce(f64::min) else {
        return json!({"applied": false, "reason": "no arch over the gap", "ours_px": ours_v});
    };
    // The lift's centre: the body vertex nearest our crotch pixel.
    let Some(ours) = (0..n)
        .filter(|&i| matches!(region[i], Region::Torso | Region::Leg))
        .min_by(|&a, &b| {
            let da = (pr[a].0 - mid_u).hypot(pr[a].1 - ours_v) + pr[a].2.max(0.0) * 0.0;
            let db = (pr[b].0 - mid_u).hypot(pr[b].1 - ours_v);
            da.total_cmp(&db)
        })
    else {
        return json!({"applied": false, "reason": "no crotch vertex"});
    };
    let lift = ((ours_v - gpt_v) * mpp).clamp(0.0, cap);
    let up = crate::geom::v(cam.up);
    let apex = x[ours];
    let mut moved = 0;
    for i in 0..n {
        if pinned[i] || !matches!(region[i], Region::Torso | Region::Leg) {
            continue;
        }
        let w = wendland((x[i] - apex).norm(), radius);
        if w > 0.0 {
            x[i] += up * (lift * w);
            moved += 1;
        }
    }
    json!({"applied": true, "view": cam.id, "gap_column_px": mid_u, "ours_px": ours_v, "arch_px": gpt_v, "arch_samples": arch.len(),
        "lift_mm": lift * 1000.0, "raw_mm": (ours_v - gpt_v) * mpp * 1000.0, "radius_m": radius, "vertices": moved})
}
