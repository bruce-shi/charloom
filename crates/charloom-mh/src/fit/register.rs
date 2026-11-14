//! Registration: MakeHuman's clean rest body onto the scan in one
//! coarse-to-fine solve (optimal-step non-rigid ICP, one affine map per
//! vertex; Amberg, Romdhani, Vetter 2007).
//!
//! The pose and morph stages put proportions into the skeleton (limb
//! scaling, a neck squeezed to half its length) and into extrapolated
//! morphs; linear-blend skinning then folds the shoulder over the chest
//! and every later stage inherits that body. Here the rest mesh is reposed
//! by bone *rotations* only, and its shape comes from one energy solved at
//! falling stiffness:
//!
//! - scan skin, both directions, onto the tangent plane: exact where seen;
//! - the suit edits' outlines on the rendered exterior rim: the body under
//!   the clothes and the skull under the hair;
//! - garment and hair surfaces as a one-sided bound;
//! - landmarks (face sets, joint cross-sections).
//!
//! The stiffness term penalises the difference between neighbouring
//! vertices' affine maps, so where nothing is measured the surface follows
//! a smooth extension of its surroundings, and the rest mesh (not a posed,
//! folded one) is the reference throughout.
use super::silhouette;
use crate::geom::{V, proper_rotation};
use charloom_core::Influence;
use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use nalgebra::Matrix3;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;

pub struct Landmark {
    pub vertices: Vec<usize>,
    pub target: V,
    pub weight: f64,
         pub joint: bool,
}

pub struct Input<'a> {
    /// MakeHuman's rest surface, welded.
    pub rest: &'a [V],
    /// The pose stage's body, welded: bone rotations and placement only.
    pub posed: &'a [V],
    pub triangles: &'a [[u32; 3]],
    pub neighbours: &'a [Vec<u32>],
    pub influences: &'a [&'a [Influence]],
    pub parents: &'a [Option<usize>],
    /// Nodes that turn with their parent (face bones with the head, toes
    /// with the foot): turned on their own few vertices, the jaw opened.
    pub follow: &'a [bool],
    /// Rest world origin of every skeleton node.
    pub joint_rest: &'a [V],
    /// Per vertex stiffness multiplier (hands, feet, cavities).
    pub stiffness: &'a [f64],
    /// Vertices that take no surface data (head cavities, lip contact).
    pub no_data: &'a [bool],
    /// Weight of scan skin per vertex (lips and hands: weak relief or fused
    /// fingers on a generated scan).
    pub trust: &'a [f64],
    /// Vertices whose shape the template takes from the posed body: the
    /// hands, which the pose stage has fitted digit by digit (`fingers.rs`).
    /// Their place still comes from the solve.
    pub posed_shape: &'a [bool],
    /// The lips: where the scan shows none of them (a beard, a scarf),
    /// they keep MakeHuman's proportions at the size the marks give.
    pub lips: &'a [bool],
    /// Trunk vertices: the reposed body is placed on the posed one by them.
    pub core: &'a [bool],
    pub scan_points: &'a [V],
    pub scan_normals: &'a [V],
    pub scan_skin: &'a [bool],
    /// Whether a point with this outward normal is outside the whole scan
    /// (a ray along the normal meets no scan surface).
    pub outside: &'a (dyn Fn(V, V) -> bool + Sync),
    pub views: &'a [silhouette::View],
    /// Per view, the suit's filled silhouette (`suit_mask`), row-major.
    pub masks: &'a [Vec<bool>],
    /// Further views of the suit edit (three-quarter), used on the head
    /// only: with front and side alone the skull between them is a box.
    pub head_views: &'a [silhouette::View],
    pub head: &'a [bool],
    /// The scalp (MakeHuman's `head` bone, not the face's own bones): what
    /// the skull hold applies to.
    pub scalp: &'a [bool],
    /// Face landmark sets on the template, with targets.
    pub landmarks: Vec<Landmark>,
    /// Joint landmarks: skeleton node and target.
    pub joints: Vec<(usize, V)>,
    /// Joints marked in one view only: skeleton node and the viewing ray
    /// (origin, direction) the joint lies on.
    pub joint_rays: Vec<(usize, V, V)>,
}

fn env(name: &str, default: f64) -> f64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

 struct Grid {
    cell: f64,
    cells: HashMap<(i32, i32, i32), Vec<u32>>,
}

impl Grid {
    fn key(&self, p: &V) -> (i32, i32, i32) {
        ((p.x / self.cell).floor() as i32, (p.y / self.cell).floor() as i32, (p.z / self.cell).floor() as i32)
    }
    fn new(points: &[V], keep: impl Fn(usize) -> bool, cell: f64) -> Self {
        let mut g = Grid { cell, cells: HashMap::new() };
        for (i, p) in points.iter().enumerate() {
            if keep(i) {
                let k = g.key(p);
                g.cells.entry(k).or_default().push(i as u32);
            }
        }
        g
    }
    /// Nearest kept point within `reach` (ties: the lower index).
    fn nearest(&self, points: &[V], p: &V, reach: f64) -> Option<(usize, f64)> {
        let c = self.key(p);
        let rings = (reach / self.cell).ceil() as i32;
        let mut best: Option<(usize, f64)> = None;
        for r in 0..=rings {
            // Everything nearer than (r - 1) cells has been visited.
            if let Some((_, d)) = best {
                if d <= (r - 1).max(0) as f64 * self.cell {
                    break;
                }
            }
            for dx in -r..=r {
                for dy in -r..=r {
                    for dz in -r..=r {
                        if dx.abs().max(dy.abs()).max(dz.abs()) != r {
                            continue;
                        }
                        let Some(list) = self.cells.get(&(c.0 + dx, c.1 + dy, c.2 + dz)) else { continue };
                        for &j in list {
                            let d = (points[j as usize] - p).norm();
                            if d <= reach && best.is_none_or(|(bj, bd)| d < bd || (d == bd && (j as usize) < bj)) {
                                best = Some((j as usize, d));
                            }
                        }
                    }
                }
            }
        }
        best
    }
}

/// The filled silhouette of a view's closed outline curves (even-odd).
pub fn suit_mask(width: usize, height: usize, curves: &[Vec<[f64; 2]>]) -> Vec<bool> {
    let mut mask = vec![false; width * height];
    for row in 0..height {
        let y = row as f64 + 0.5;
        let mut crossings: Vec<f64> = vec![];
        for curve in curves {
            for (k, p) in curve.iter().enumerate() {
                let q = curve[(k + 1) % curve.len()];
                if (p[1] <= y) != (q[1] <= y) {
                    crossings.push(p[0] + (y - p[1]) / (q[1] - p[1]) * (q[0] - p[0]));
                }
            }
        }
        crossings.sort_by(f64::total_cmp);
        for pair in crossings.chunks_exact(2) {
            let (from, to) = (pair[0].max(0.0).ceil() as usize, (pair[1].min(width as f64 - 1.0)).floor() as usize);
            for column in from..=to.min(width - 1) {
                if (column as f64 + 0.5) >= pair[0] && (column as f64 + 0.5) <= pair[1] {
                    mask[row * width + column] = true;
                }
            }
        }
    }
    mask
}

fn vertex_normals(x: &[V], triangles: &[[u32; 3]]) -> Vec<V> {
    let mut n = vec![V::zeros(); x.len()];
    for t in triangles {
        let [a, b, c] = t.map(|i| i as usize);
        let f = (x[b] - x[a]).cross(&(x[c] - x[a]));
        n[a] += f;
        n[b] += f;
        n[c] += f;
    }
    n.iter().map(|q| q / q.norm().max(1e-12)).collect()
}

fn centroid(points: &[V], ids: &[usize]) -> V {
    ids.iter().fold(V::zeros(), |s, &i| s + points[i]) / ids.len().max(1) as f64
}

/// The rest mesh in the scan's pose: a similarity onto the posed body, then
/// every bone turned (never scaled or shifted) as the pose stage turned it.
fn repose(input: &Input) -> (Vec<V>, Vec<V>, f64) {
    let (rest, posed) = (input.rest, input.posed);
    let n = rest.len();
    let all: Vec<usize> = (0..n).collect();
    let (ma, mb) = (centroid(rest, &all), centroid(posed, &all));
    let h = (0..n).fold(Matrix3::zeros(), |s, i| s + (posed[i] - mb) * (rest[i] - ma).transpose());
    let r0 = proper_rotation(h);
    let scale = (0..n).map(|i| (posed[i] - mb).dot(&(r0 * (rest[i] - ma)))).sum::<f64>()
        / (0..n).map(|i| (rest[i] - ma).norm_squared()).sum::<f64>();
    let map = |p: &V| r0 * (p - ma) * scale + mb;
    let v: Vec<V> = rest.iter().map(map).collect();
    let joint: Vec<V> = input.joint_rest.iter().map(map).collect();
    let nodes = joint.len();
    let mut owned: Vec<Vec<usize>> = vec![vec![]; nodes];
    for (i, row) in input.influences.iter().enumerate() {
        if let Some(w) = row.iter().max_by(|a, b| a.weight.total_cmp(&b.weight)) {
            owned[w.joint].push(i);
        }
    }
    let depth = |mut q: usize| {
        let mut d = 0;
        while let Some(p) = input.parents[q] {
            q = p;
            d += 1;
        }
        d
    };
    let mut order: Vec<usize> = (0..nodes).collect();
    order.sort_by_key(|&q| (depth(q), q));
    let mut turn = vec![Matrix3::identity(); nodes];
    let mut at = joint.clone();
    for &q in &order {
        let ids = &owned[q];
        let parent = input.parents[q];
        turn[q] = if ids.len() >= 25 && !input.follow[q] {
            let (ca, cb) = (centroid(&v, ids), centroid(posed, ids));
            proper_rotation(ids.iter().fold(Matrix3::zeros(), |s, &i| s + (posed[i] - cb) * (v[i] - ca).transpose()))
        } else {
            parent.map_or(Matrix3::identity(), |p| turn[p])
        };
        if let Some(p) = parent {
            at[q] = at[p] + turn[p] * (joint[q] - joint[p]);
        }
    }
    let mut out: Vec<V> = (0..n)
        .map(|i| {
            let total: f64 = input.influences[i].iter().map(|w| w.weight).sum();
            input.influences[i].iter().fold(V::zeros(), |s, w| {
                s + (turn[w.joint] * (v[i] - joint[w.joint]) + at[w.joint]) * (w.weight / total.max(1e-12))
            })
        })
        .collect();
    let core: Vec<usize> = (0..n).filter(|&i| input.core[i]).collect();
    let shift = centroid(posed, &core) - centroid(&out, &core);
    for p in &mut out {
        *p += shift;
    }
    for p in &mut at {
        *p += shift;
    }
    (out, at, scale)
}

struct Rows {
    /// (vertex, target, weight)
    single: Vec<(usize, V, f64)>,
    /// (vertices, target of their mean, weight)
    sets: Vec<(Vec<usize>, V, f64)>,
    /// Weighted vertex combinations, used for local surface curvature.
    linear: Vec<(Vec<(usize, f64)>, V, f64)>,
}

/// One optimal step: the affine maps minimising stiffness + data, applied.
fn solve(template: &[V], edges: &[(usize, usize, f64)], alpha: f64, rows: &Rows) -> Vec<V> {
    let n = template.len();
    let h = |i: usize| [template[i].x, template[i].y, template[i].z, 1.0];
    let mut diag = vec![[0.0f64; 16]; n];
    let mut rhs = vec![[0.0f64; 12]; n];
    let mut triplets: Vec<Triplet<usize, usize, f64>> = Vec::with_capacity(edges.len() * 8 + n * 16);
    for &(i, j, m) in edges {
        let w = alpha * alpha * m;
        for k in 0..4 {
            diag[i][k * 5] += w;
            diag[j][k * 5] += w;
            triplets.push(Triplet::new(i * 4 + k, j * 4 + k, -w));
            triplets.push(Triplet::new(j * 4 + k, i * 4 + k, -w));
        }
    }
    for &(i, u, w) in &rows.single {
        let (vi, w2) = (h(i), w * w);
        for k in 0..4 {
            for l in 0..4 {
                diag[i][k * 4 + l] += w2 * vi[k] * vi[l];
            }
            for c in 0..3 {
                rhs[i][k * 3 + c] += w2 * vi[k] * u[c];
            }
        }
    }
    for (ids, u, w) in &rows.sets {
        let (m, w2) = (ids.len() as f64, w * w);
        for &i in ids {
            let vi = h(i);
            for k in 0..4 {
                for c in 0..3 {
                    rhs[i][k * 3 + c] += w2 / m * vi[k] * u[c];
                }
            }
            for &j in ids {
                let vj = h(j);
                for k in 0..4 {
                    for l in 0..4 {
                        let value = w2 / (m * m) * vi[k] * vj[l];
                        if i == j {
                            diag[i][k * 4 + l] += value;
                        } else {
                            triplets.push(Triplet::new(i * 4 + k, j * 4 + l, value));
                        }
                    }
                }
            }
        }
    }
    for (terms, u, w) in &rows.linear {
        let w2 = w * w;
        for &(i, ci) in terms {
            let vi = h(i);
            for k in 0..4 {
                for c in 0..3 {
                    rhs[i][k * 3 + c] += w2 * ci * vi[k] * u[c];
                }
            }
            for &(j, cj) in terms {
                let vj = h(j);
                for k in 0..4 {
                    for l in 0..4 {
                        let value = w2 * ci * cj * vi[k] * vj[l];
                        if i == j {
                            diag[i][k * 4 + l] += value;
                        } else {
                            triplets.push(Triplet::new(i * 4 + k, j * 4 + l, value));
                        }
                    }
                }
            }
        }
    }
    for (i, d) in diag.iter().enumerate() {
        for k in 0..4 {
            for l in 0..4 {
                // A vertex with no data still has a full block (pattern kept).
                triplets.push(Triplet::new(i * 4 + k, i * 4 + l, d[k * 4 + l] + if k == l { 1e-9 } else { 0.0 }));
            }
        }
    }
    let matrix = SparseColMat::<usize, f64>::try_new_from_triplets(n * 4, n * 4, &triplets).expect("registration matrix");
    let factor = matrix.sp_cholesky(faer::Side::Lower).expect("registration system is positive definite");
    let b = faer::Mat::<f64>::from_fn(n * 4, 3, |r, c| rhs[r / 4][(r % 4) * 3 + c]);
    let x = factor.solve(&b);
    (0..n)
        .map(|i| {
            let vi = h(i);
            V::from_fn(|c, _| (0..4).map(|k| vi[k] * x[(i * 4 + k, c)]).sum())
        })
        .collect()
}

pub fn fit(input: &Input) -> (Vec<V>, Value) {
    let n = input.rest.len();
    let (mut template, joint_at, scale) = repose(input);
    // Each posed hand, shifted onto the reposed one and faded in over four
    // rings from the wrist, replaces it in the template.
    {
        let mut ring = vec![usize::MAX; n];
        let mut queue = std::collections::VecDeque::new();
        for i in 0..n {
            if input.posed_shape[i] && input.neighbours[i].iter().any(|&j| !input.posed_shape[j as usize]) {
                ring[i] = 0;
                queue.push_back(i);
            }
        }
        while let Some(i) = queue.pop_front() {
            for &j in &input.neighbours[i] {
                let j = j as usize;
                if input.posed_shape[j] && ring[j] == usize::MAX {
                    ring[j] = ring[i] + 1;
                    queue.push_back(j);
                }
            }
        }
        let mut part = vec![usize::MAX; n];
        let mut parts = 0;
        for seed in 0..n {
            if !input.posed_shape[seed] || part[seed] != usize::MAX {
                continue;
            }
            let mut members = vec![seed];
            part[seed] = parts;
            let mut k = 0;
            while k < members.len() {
                for &j in &input.neighbours[members[k]] {
                    let j = j as usize;
                    if input.posed_shape[j] && part[j] == usize::MAX {
                        part[j] = parts;
                        members.push(j);
                    }
                }
                k += 1;
            }
            members.sort_unstable();
            let shift = centroid(&template, &members) - centroid(input.posed, &members);
            for &i in &members {
                let w = (ring[i].min(4) as f64 / 4.0).min(1.0);
                template[i] = template[i] * (1.0 - w) + (input.posed[i] + shift) * w;
            }
            parts += 1;
        }
    }
    let mut edges: Vec<(usize, usize, f64)> = vec![];
    for (i, row) in input.neighbours.iter().enumerate() {
        for &j in row {
            let j = j as usize;
            if i < j {
                edges.push((i, j, input.stiffness[i] * input.stiffness[j]));
            }
        }
    }
    // Landmarks: sets as given; joints as the cross-section ring around the
    // reposed joint, with the joint's offset from the ring's centre.
    let mut sets: Vec<(Vec<usize>, V, f64, bool)> = input
        .landmarks
        .iter()
        .map(|l| {
            // A single vertex is a point load (a spike at a mouth corner as
            // the stiffness falls): its ring carries the mark with it.
            let ids: Vec<usize> = if l.vertices.len() == 1 {
                let v = l.vertices[0];
                std::iter::once(v).chain(input.neighbours[v].iter().map(|&j| j as usize)).collect()
            } else {
                let step = l.vertices.len().div_ceil(24).max(1);
                l.vertices.iter().copied().step_by(step).collect()
            };
            // The set used stands for the marked one.
            let target = l.target - (centroid(&template, &l.vertices) - centroid(&template, &ids));
            (ids, target, l.weight, l.joint)
        })
        .collect();
    // The cross-section ring around a reposed joint and the joint's offset
    // from the ring's centre.
    let ring_of = |node: usize| -> Option<(Vec<usize>, V)> {
        let c = joint_at[node];
        let nearest = template.iter().map(|p| (p - c).norm()).fold(f64::INFINITY, f64::min);
        let radius = (nearest * 1.6).max(0.03);
        let ring: Vec<usize> = (0..n).filter(|&i| (template[i] - c).norm() <= radius).collect();
        if ring.len() < 6 {
            return None;
        }
        let step = ring.len().div_ceil(24).max(1);
        let ids: Vec<usize> = ring.into_iter().step_by(step).collect();
        let offset = c - centroid(&template, &ids);
        Some((ids, offset))
    };
    for &(node, target) in &input.joints {
        if let Some((ids, offset)) = ring_of(node) {
            sets.push((ids, target - offset, 1.0, true));
        }
    }
    // A joint seen in one view lies on its viewing ray: each iteration its
    // ring is pulled to the nearest point of that ray (depth stays free).
    let ray_sets: Vec<(Vec<usize>, V, V, V)> = input.joint_rays.iter().filter_map(|&(node, origin, direction)| {
        let (ids, offset) = ring_of(node)?;
        Some((ids, offset, origin, direction.normalize()))
    }).collect();
    let skin_grid = Grid::new(input.scan_points, |i| input.scan_skin[i], 0.012);
    let all_grid = Grid::new(input.scan_points, |_| true, 0.012);
    let skin_ids: Vec<usize> = (0..input.scan_points.len()).filter(|&i| input.scan_skin[i]).collect();
    let stride = (skin_ids.len() / 40_000).max(1);
    let reverse: Vec<usize> = skin_ids.into_iter().step_by(stride).collect();
    let everywhere = vec![true; n];
    let marked: Vec<usize> = sets.iter().filter(|s| !s.3).flat_map(|s| s.0.iter().copied()).collect();
    let lips: Vec<usize> = (0..n).filter(|&i| input.lips[i]).collect();
     
    let mut near_mark = vec![false; n];
    for &i in &marked {
        near_mark[i] = true;
    }
    for _ in 0..2 {
        let grown: Vec<usize> = (0..n).filter(|&i| !near_mark[i] && input.neighbours[i].iter().any(|&j| near_mark[j as usize])).collect();
        for i in grown {
            near_mark[i] = true;
        }
    }
    let w_lips = env("MH_REG_WLIPS", 1.5);
    let w_skull = env("MH_REG_WSKULL", 0.3);

    let a0 = env("MH_REG_A0", 60.0);
    let steps = env("MH_REG_STEPS", 16.0) as usize;
    let (reach0, reach1) = (env("MH_REG_REACH0", 0.14), env("MH_REG_REACH1", 0.05));
    let (tau0, tau1) = (env("MH_REG_TAU0", 0.07), env("MH_REG_TAU1", 0.03));
     
    let start_gap = {
        let grid = Grid::new(&template, |i| !input.no_data[i], 0.03);
        let mut d: Vec<f64> = reverse.par_iter().map(|&k| grid.nearest(&template, &input.scan_points[k], 0.3).map_or(0.3, |h| h.1)).collect();
        d.sort_by(f64::total_cmp);
        d.get(d.len() * 9 / 10).copied().unwrap_or(0.0)
    };
    let tau0 = tau0.max((1.3 * start_gap).min(env("MH_REG_TAU_MAX", 0.25)));
    let reach0 = reach0.max((2.0 * tau0).min(0.30));
     
    let skin_from = env("MH_REG_SKIN_FROM", 0.0);
     
    let a1 = env("MH_REG_A1", if start_gap > 0.09 { 0.25 } else { 0.7 });
    let clear = env("MH_REG_CLEAR", 0.003);
    let max_inner = env("MH_REG_MAX_INNER", 10.0) as usize;
    let (w_sil, w_bound, w_prior0) = (env("MH_REG_WSIL", 1.0), env("MH_REG_WBOUND", 1.0), env("MH_REG_WPRIOR", 0.1));
    let mut y = template.clone();
    let w_curve = env("MH_REG_HEAD_CURVATURE", if start_gap > 0.09 { 6.0 } else { 2.0 });
    let mut history = vec![];
    let mut seconds = [0.0f64; 3];
    // The outlines' largest remaining error (95th percentile over views).
    let mut outline_gap = 0.0f64;
    let mut head_views_dropped = 0usize;
    let head_gate = env("MH_REG_HEAD_GATE", 1.0);
    let head_own_views = env("MH_REG_HEAD_OWN_VIEWS", 1.0) > 0.0;
    let cover_guard = env("MH_REG_COVER_GUARD", 1.0) > 0.0;
    let hand_lo = env("MH_REG_HAND_SEEN", 0.0);
    let mut hand_seen = [0.0f64; 2];
    let cover_depth = env("MH_REG_COVER_DEPTH", 0.008);
    let no_crossing = env("MH_REG_NO_CROSS", 1.0) > 0.0;
    let back_pull = env("MH_REG_BACK_PULL", 1.0) > 0.0;
     
    let settle = env("MH_REG_SETTLE", 12.0) as usize;
    let mut last_seen = vec![false; n];
    for s in 0..steps {
        let f = s as f64 / (steps - 1).max(1) as f64;
        let alpha = a0 * (a1 / a0).powf(f);
        let tau = tau0 * (1.0 - f) + tau1 * f;
         
        let reach = (reach0 * (1.0 - f) + reach1 * f).max((1.5 * outline_gap).min(reach0));
        let w_prior = w_prior0 * (1.0 - 3.0 * f).max(0.0);
        let w_joint = 2.0 * (1.0 - 1.5 * f).max(0.0);
        let mut last = json!(null);
         
        let base = if f < 0.7 { 2 } else { 3 };
        let cap = if f < 0.34 { max_inner } else if s + 1 == steps { settle.max(base) } else { base + 2 };
        let (mut inner, mut previous) = (0, f64::INFINITY);
        loop {
            let clock = std::time::Instant::now();
            let normals = vertex_normals(&y, input.triangles);
            let mut rows = Rows { single: vec![], sets: vec![], linear: vec![] };
            // Template vertex -> scan skin, onto its tangent plane.
            let forward: Vec<(Option<V>, Option<(usize, f64)>, u8)> = (0..n)
                .into_par_iter()
                .map(|i| {
                    let any = all_grid.nearest(input.scan_points, &y[i], 0.08);
                    let skin = (!input.no_data[i] && f >= skin_from)
                        .then(|| skin_grid.nearest(input.scan_points, &y[i], tau))
                        .flatten()
                        .filter(|&(k, d)| {
                            normals[i].dot(&input.scan_normals[k]) > 0.35 && any.is_none_or(|(_, da)| d < da + 0.004)
                        })
                        .map(|(k, _)| {
                            let (q, nq) = (input.scan_points[k], input.scan_normals[k]);
                            (y[i] + nq * (q - y[i]).dot(&nq)) * 0.8 + q * 0.2
                        });
                     
                    let outside = skin.is_none() && !input.no_data[i] && f >= skin_from && any.is_some() && (input.outside)(y[i], normals[i]);
                    let out = outside && !near_mark[i] && any.is_some_and(|(k, _)| !input.scan_skin[k]);
                     
                    let shallow = skin.is_none() && !near_mark[i] && !input.lips[i] && !input.no_data[i] && f >= skin_from && cover_depth > 0.0
                        && any.is_some_and(|(k, d)| {
                            let nk = input.scan_normals[k];
                            !input.scan_skin[k] && d < 0.02 && normals[i].dot(&nk) > 0.6 && (y[i] - input.scan_points[k]).dot(&nk) > -cover_depth
                        });
                    let out = if shallow && !out { 2u8 } else { out as u8 };
                     
                    let skin = skin.or_else(|| {
                        let (k, _) = any.filter(|&(k, _)| outside && back_pull && input.scan_skin[k] && normals[i].dot(&input.scan_normals[k]) > 0.0)?;
                        let (q, nq) = (input.scan_points[k], input.scan_normals[k]);
                        Some((y[i] + nq * (q - y[i]).dot(&nq)) * 0.5 + q * 0.5)
                    });
                    (skin, any, out)
                })
                .collect();
             
            let mid = centroid(&y, &(0..n).filter(|&i| input.core[i]).collect::<Vec<_>>()).x;
            let mut hand_count = [[0usize; 2]; 2];
            for i in (0..n).filter(|&i| input.posed_shape[i]) {
                let side = (y[i].x > mid) as usize;
                hand_count[side][0] += 1;
                hand_count[side][1] += forward[i].0.is_some() as usize;
            }
            for side in 0..2 {
                hand_seen[side] = hand_count[side][1] as f64 / hand_count[side][0].max(1) as f64;
            }
            let hand_scale = |i: usize| if input.posed_shape[i] && hand_lo > 0.0 {
                ((hand_seen[(y[i].x > mid) as usize] - hand_lo) / hand_lo).clamp(0.0, 1.0)
            } else { 1.0 };
            let mut seen = vec![false; n];
            for (i, (skin, _, _)) in forward.iter().enumerate() {
                if let Some(u) = skin {
                    rows.single.push((i, *u, input.trust[i] * hand_scale(i)));
                    seen[i] = true;
                }
            }
            let forward_count = rows.single.len();
            // Scan skin sample -> nearest open template vertex, onto the sample's plane.
            let body_grid = Grid::new(&y, |i| !input.no_data[i], 0.012);
            let hits: Vec<Option<(usize, V)>> = reverse
                .par_iter()
                .filter(|_| f >= skin_from)
                .map(|&k| {
                    let (q, nq) = (input.scan_points[k], input.scan_normals[k]);
                    body_grid
                        .nearest(&y, &q, tau)
                         
                        .filter(|&(i, d)| {
                            normals[i].dot(&nq) > 0.35
                                && (!cover_guard || forward[i].1.is_none_or(|(k, covered)| input.scan_skin[k] || d < covered + 0.004))
                        })
                        .map(|(i, _)| (i, nq * (q - y[i]).dot(&nq)))
                })
                .collect();
            let mut sum = vec![V::zeros(); n];
            let mut count = vec![0u32; n];
            for (i, d) in hits.into_iter().flatten() {
                sum[i] += d;
                count[i] += 1;
            }
            let mut reverse_count = 0;
            for i in 0..n {
                if count[i] > 0 {
                    rows.single.push((i, y[i] + sum[i] / count[i] as f64, 0.7 * input.trust[i] * hand_scale(i)));
                    seen[i] = true;
                    reverse_count += count[i];
                }
            }
            last_seen.clone_from(&seen);
            // One-sided bound: a hidden vertex may not leave the cloth or hair above it.
            let mut bound = 0;
             
            let mut by_skin = seen.clone();
            for _ in 0..3 {
                let grown: Vec<usize> = (0..n).filter(|&i| !by_skin[i] && input.neighbours[i].iter().any(|&j| by_skin[j as usize])).collect();
                for i in grown {
                    by_skin[i] = true;
                }
            }
            for (i, (_, any, out)) in forward.iter().enumerate() {
                let Some((k, _)) = *any else { continue };
                if *out == 0 || seen[i] || (*out == 2 && by_skin[i]) {
                    continue;
                }
                rows.single.push((i, input.scan_points[k] - normals[i] * if by_skin[i] { clear } else { clear.max(cover_depth) }, w_bound));
                bound += 1;
            }
             
            let face: Vec<usize> = (0..n).filter(|&i| input.head[i] && seen[i]).collect();
            if face.len() >= 50 && w_skull > 0.0 {
                let (ma, mb) = (centroid(&template, &face), centroid(&y, &face));
                let h = face.iter().fold(Matrix3::zeros(), |s, &i| s + (y[i] - mb) * (template[i] - ma).transpose());
                let r = proper_rotation(h);
                let scale = face.iter().map(|&i| (y[i] - mb).dot(&(r * (template[i] - ma)))).sum::<f64>()
                    / face.iter().map(|&i| (template[i] - ma).norm_squared()).sum::<f64>();
                 
                let mut ring = vec![u32::MAX; n];
                let mut queue = std::collections::VecDeque::new();
                for i in (0..n).filter(|&i| seen[i]).chain(marked.iter().copied()) {
                    if ring[i] == u32::MAX {
                        ring[i] = 0;
                        queue.push_back(i);
                    }
                }
                while let Some(i) = queue.pop_front() {
                    if ring[i] >= 4 {
                        continue;
                    }
                    for &j in &input.neighbours[i] {
                        let j = j as usize;
                        if ring[j] == u32::MAX {
                            ring[j] = ring[i] + 1;
                            queue.push_back(j);
                        }
                    }
                }
                for i in 0..n {
                    if input.scalp[i] && !seen[i] {
                        let fade = ((ring[i].min(4) as f64 - 1.0) / 3.0).clamp(0.0, 1.0);
                        if fade > 0.0 {
                            rows.single.push((i, r * (template[i] - ma) * scale + mb, w_skull * fade));
                        }
                    }
                }
            }
             
            let hidden = 1.0 - lips.iter().filter(|&&i| seen[i]).count() as f64 / lips.len().max(1) as f64;
            let w_lips = w_lips * hidden * hidden;
            if lips.len() >= 12 && w_lips > 0.05 {
                let (ma, mb) = (centroid(&template, &lips), centroid(&y, &lips));
                let h = lips.iter().fold(Matrix3::zeros(), |s, &i| s + (y[i] - mb) * (template[i] - ma).transpose());
                let r = proper_rotation(h);
                let scale = lips.iter().map(|&i| (y[i] - mb).dot(&(r * (template[i] - ma)))).sum::<f64>()
                    / lips.iter().map(|&i| (template[i] - ma).norm_squared()).sum::<f64>();
                for &i in &lips {
                    rows.single.push((i, r * (template[i] - ma) * scale + mb, w_lips));
                }
            }
            seconds[0] += clock.elapsed().as_secs_f64();
            let clock = std::time::Instant::now();
            // Suit outlines on the rendered exterior rim, both directions.
            let (all, _, stats) = silhouette::rendered_constraints_with(input.views, &y, input.triangles, &everywhere, reach, no_crossing);
             
            let mut constraints: Vec<(silhouette::SurfaceConstraint, usize)> = vec![];
            for c in all {
                let ids = c.vertices.map(|i| i as usize);
                let point = (0..3).fold(V::zeros(), |s, k| s + y[ids[k]] * c.bary[k]);
                let normal = (0..3).fold(V::zeros(), |s, k| s + normals[ids[k]] * c.bary[k]);
                let view = input.views.iter().position(|v| {
                    let cam = v.camera();
                    (crate::geom::v(cam.right) - c.axes[0]).norm() < 1e-9 && (crate::geom::v(cam.up) - c.axes[1]).norm() < 1e-9
                });
                let keep = view.is_none_or(|v| {
                    let cam = input.views[v].camera();
                    let (px, py, _) = cam.project(crate::geom::a(point));
                    let (w, h) = (cam.width, cam.height);
                    if px < 0.0 || py < 0.0 || px >= w as f64 || py >= h as f64 || input.masks.get(v).is_none_or(|m| m.len() != w * h) {
                        return true;
                    }
                    let inside = input.masks[v][py as usize * w + px as usize];
                    let along = c.delta.dot(&(c.axes[0] * normal.dot(&c.axes[0]) + c.axes[1] * normal.dot(&c.axes[1])));
                    c.delta.norm() < 0.004 || if inside { along > 0.0 } else { along < 0.0 }
                });
                if keep {
                    constraints.push((c, view.unwrap_or(0)));
                }
            }
            for c in silhouette::rendered_constraints(input.head_views, &y, input.triangles, input.head, reach.min(0.08)).0 {
                let view = input.head_views.iter().position(|v| {
                    let cam = v.camera();
                    (crate::geom::v(cam.right) - c.axes[0]).norm() < 1e-9 && (crate::geom::v(cam.up) - c.axes[1]).norm() < 1e-9
                });
                constraints.push((c, input.views.len() + view.unwrap_or(0)));
            }
             
            let view_count = input.views.len() + input.head_views.len();
            let mut on_skin: Vec<Vec<f64>> = vec![vec![]; view_count];
            for (c, view) in &constraints {
                if c.vertices.iter().any(|&i| input.head[i as usize] && seen[i as usize]) {
                    on_skin[*view].push(c.delta.norm());
                }
            }
            let misdrawn: Vec<bool> = on_skin.iter_mut().map(|d| {
                d.sort_by(f64::total_cmp);
                d.len() >= 20 && d[d.len() / 2] > head_gate
            }).collect();
            head_views_dropped = misdrawn.iter().filter(|m| **m).count();
            let mut pull = vec![V::zeros(); n];
            let mut mass = vec![0.0f64; n];
            for (c, view) in &constraints {
                for (&i, &b) in c.vertices.iter().zip(&c.bary) {
                    // The head goes by its own views (the suit's head outline
                    // registered to the scan's head in each view); the
                    // full-body views draw the head where the edit posed it.
                    let body_view_on_head = head_own_views && *view < input.views.len() && !input.head_views.is_empty() && input.head[i as usize];
                    if b > 0.05 && !body_view_on_head && !(misdrawn[*view] && input.head[i as usize]) {
                        pull[i as usize] += c.delta * (b * c.weight);
                        mass[i as usize] += b * c.weight;
                    }
                }
            }
            for i in 0..n {
                if mass[i] > 0.0 {
                    let w = w_sil * (mass[i] / 1.5).clamp(0.2, 1.0) * if seen[i] { 0.15 } else { 1.0 } * hand_scale(i);
                    rows.single.push((i, y[i] + pull[i] / mass[i], w));
                }
            }
            if w_prior > 0.0 {
                rows.single.extend((0..n).map(|i| (i, input.posed[i], w_prior)));
            }
            if w_joint > 0.0 {
                for (ids, offset, origin, direction) in &ray_sets {
                    let joint = centroid(&y, ids) + offset;
                    let on_ray = origin + direction * (joint - origin).dot(direction);
                    rows.sets.push((ids.clone(), on_ray - offset, w_joint));
                }
            }
            for (ids, target, weight, joint) in &sets {
                let w = weight * if *joint { w_joint } else { 3.0 };
                if w > 0.0 {
                    rows.sets.push((ids.clone(), *target, w));
                }
            }
             
            if f >= 0.35 && w_curve > 0.0 {
                for i in (0..n).filter(|&i| input.head[i] && !input.neighbours[i].is_empty()) {
                    let neighbours = &input.neighbours[i];
                    let k = neighbours.len() as f64;
                    let mean = neighbours.iter().fold(V::zeros(), |s, &j| s + template[j as usize]) / k;
                    let mut covariance = Matrix3::zeros();
                    let (mut old_length, mut new_length) = (0.0, 0.0);
                    for &j in neighbours {
                        let j = j as usize;
                        let (old, new) = (template[j] - template[i], y[j] - y[i]);
                        covariance += new * old.transpose();
                        old_length += old.norm();
                        new_length += new.norm();
                    }
                    let target = proper_rotation(covariance) * (template[i] - mean)
                        * (new_length / old_length.max(1e-12)).clamp(0.25, 6.0);
                    let terms = std::iter::once((i, 1.0))
                        .chain(neighbours.iter().map(|&j| (j as usize, -1.0 / k))).collect();
                    rows.linear.push((terms, target, w_curve));
                }
            }
            seconds[1] += clock.elapsed().as_secs_f64();
            let clock = std::time::Instant::now();
            y = solve(&template, &edges, alpha, &rows);
            seconds[2] += clock.elapsed().as_secs_f64();
            // Extent of the trunk across, up and front to back: a collapse shows here.
            let extent: Vec<f64> = (0..3).map(|c| {
                let values = (0..n).filter(|&i| input.core[i]).map(|i| y[i][c]);
                let (lo, hi) = values.fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), q| (l.min(q), h.max(q)));
                hi - lo
            }).collect();
            let measure: f64 = stats.iter().map(|s| s.1).sum();
            outline_gap = stats.iter().map(|s| s.2).fold(0.0, f64::max);
            inner += 1;
            last = json!({"alpha": alpha, "iterations": inner, "head_views_dropped": head_views_dropped, "trunk_extent_m": extent, "skin_forward": forward_count, "skin_reverse": reverse_count, "bound": bound,
                "outline_m": stats.iter().map(|s| json!({"pairs": s.0, "p50": s.1, "p95": s.2})).collect::<Vec<_>>()});
            if inner >= cap || (inner >= base && !(measure < 0.95 * previous)) {
                break;
            }
            previous = measure;
        }
        history.push(last);
    }
     
    let mut unfolded = [0usize; 2];
    if env("MH_REG_UNFOLD", 1.0) > 0.0 {
        let mut edge_faces: std::collections::HashMap<(u32, u32), Vec<usize>> = Default::default();
        for (t, tri) in input.triangles.iter().enumerate() {
            if tri.iter().all(|&i| input.head[i as usize]) {
                for k in 0..3 {
                    let (p, q) = (tri[k], tri[(k + 1) % 3]);
                    edge_faces.entry((p.min(q), p.max(q))).or_default().push(t);
                }
            }
        }
        let pairs: Vec<(usize, usize)> = edge_faces.values().filter(|f| f.len() == 2).map(|f| (f[0], f[1])).collect();
        let face_normal = |x: &[V], t: usize| {
            let [p, q, r] = input.triangles[t].map(|i| x[i as usize]);
            (q - p).cross(&(r - p)).try_normalize(1e-15).unwrap_or_else(V::zeros)
        };
        let flat: Vec<bool> = pairs.iter().map(|&(s, t)| face_normal(&template, s).dot(&face_normal(&template, t)) > 0.3).collect();
        let is_marked = { let mut m = vec![false; n]; for &i in &marked { m[i] = true; } m };
        let folded = |x: &[V]| -> Vec<bool> {
            let mut on = vec![false; n];
            for (&(s, t), _) in pairs.iter().zip(&flat).filter(|(_, f)| **f) {
                if face_normal(x, s).dot(&face_normal(x, t)) < 0.0 {
                    for &i in input.triangles[s].iter().chain(&input.triangles[t]) {
                        on[i as usize] = true;
                    }
                }
            }
            on
        };
        for round in 0..15 {
            let mut region = folded(&y);
            let count = region.iter().filter(|r| **r).count();
            if round == 0 {
                unfolded[0] = count;
            }
            unfolded[1] = count;
            if count == 0 {
                break;
            }
            for _ in 0..2 {
                let grown: Vec<usize> = (0..n).filter(|&i| !region[i] && input.neighbours[i].iter().any(|&j| region[j as usize])).collect();
                for i in grown {
                    region[i] = true;
                }
            }
            let ids: Vec<usize> = (0..n).filter(|&i| region[i] && input.head[i] && !is_marked[i] && !input.neighbours[i].is_empty()).collect();
            for _ in 0..4 {
                let next: Vec<V> = ids.iter().map(|&i| {
                    let k = input.neighbours[i].len() as f64;
                    let mean = input.neighbours[i].iter().fold(V::zeros(), |s, &j| s + y[j as usize]) / k;
                    let rest_mean = input.neighbours[i].iter().fold(V::zeros(), |s, &j| s + template[j as usize]) / k;
                    let size = input.neighbours[i].iter().map(|&j| (y[j as usize] - mean).norm()).sum::<f64>()
                        / input.neighbours[i].iter().map(|&j| (template[j as usize] - rest_mean).norm()).sum::<f64>().max(1e-12);
                    mean + (template[i] - rest_mean) * size.min(3.0)
                }).collect();
                for (&i, q) in ids.iter().zip(next) {
                    y[i] = (y[i] + q) * 0.5;
                }
            }
        }
    }
     
    let tuck = env("MH_REG_TUCK", 0.005);
    let fair = env("MH_REG_FAIR", 6.0) as usize;
    let (mut tucked, mut tuck_max) = (0usize, 0.0f64);
    if tuck > 0.0 {
        let hidden: Vec<bool> = (0..n).map(|i| input.head[i] && !last_seen[i] && !near_mark[i] && !input.no_data[i] && !input.lips[i]).collect();
        let mut by_skin = last_seen.clone();
        for _ in 0..3 {
            let grown: Vec<usize> = (0..n).filter(|&i| !by_skin[i] && input.neighbours[i].iter().any(|&j| by_skin[j as usize])).collect();
            for i in grown {
                by_skin[i] = true;
            }
        }
        let smooth = |y: &mut Vec<V>, factor: f64| {
            let moved: Vec<(usize, V)> = (0..n).filter(|&i| hidden[i] && !by_skin[i] && !input.neighbours[i].is_empty()).map(|i| {
                let mean = input.neighbours[i].iter().fold(V::zeros(), |s, &j| s + y[j as usize]) / input.neighbours[i].len() as f64;
                (i, y[i] + (mean - y[i]) * factor)
            }).collect();
            for (i, p) in moved {
                y[i] = p;
            }
        };
        for _ in 0..fair {
            smooth(&mut y, 0.5);
            smooth(&mut y, -0.53);
        }
        let mut total = vec![0.0f64; n];
        for _ in 0..6 {
            let normals = vertex_normals(&y, input.triangles);
            let mut push: Vec<V> = (0..n).into_par_iter().map(|i| {
                if !hidden[i] {
                    return V::zeros();
                }
                let Some((k, _)) = all_grid.nearest(input.scan_points, &y[i], 0.03) else { return V::zeros() };
                let nk = input.scan_normals[k];
                if input.scan_skin[k] || normals[i].dot(&nk) < 0.3 {
                    return V::zeros();
                }
                let out = (y[i] - input.scan_points[k]).dot(&nk);
                let (from, depth) = if by_skin[i] { (-0.001, 0.003) } else { (-tuck, tuck) };
                if out > from { -nk * (out + depth) } else { V::zeros() }
            }).collect();
            if push.iter().all(|d| d.norm() == 0.0) {
                break;
            }
            for _ in 0..3 {
                let spread: Vec<V> = (0..n).map(|i| {
                    if !hidden[i] || input.neighbours[i].is_empty() {
                        return push[i];
                    }
                    let mean = input.neighbours[i].iter().fold(V::zeros(), |s, &j| s + push[j as usize]) / input.neighbours[i].len() as f64 * 0.7;
                    if mean.norm() > push[i].norm() { mean } else { push[i] }
                }).collect();
                push = spread;
            }
            for i in 0..n {
                y[i] += push[i];
                total[i] += push[i].norm();
            }
        }
        tucked = total.iter().filter(|d| **d > 1e-4).count();
        tuck_max = total.iter().copied().fold(0.0, f64::max);
    }
    // Scan skin -> fitted body (plane distance at the nearest vertex).
    let normals = vertex_normals(&y, input.triangles);
    let body_grid = Grid::new(&y, |i| !input.no_data[i], 0.012);
    let mut residual: Vec<f64> = reverse
        .par_iter()
        .filter_map(|&k| {
            let q = input.scan_points[k];
            body_grid.nearest(&y, &q, 0.05).map(|(i, _)| (q - y[i]).dot(&normals[i]).abs())
        })
        .collect();
    residual.sort_by(f64::total_cmp);
    let pct = |f: f64| residual.get(((residual.len().saturating_sub(1)) as f64 * f) as usize).copied().unwrap_or(0.0);
    let report = json!({
        "applied": true, "seconds": {"skin_and_bound": seconds[0], "outlines": seconds[1], "solve": seconds[2]}, "template_scale": scale, "start_gap_p90_m": start_gap, "hand_seen": hand_seen, "tucked": {"vertices": tucked, "max_m": tuck_max}, "fold_vertices": {"before": unfolded[0], "after": unfolded[1]}, "skin_from": skin_from, "skin_reach_start_m": tau0, "outline_reach_start_m": reach0, "steps": steps, "landmark_sets": sets.len(), "joint_rays": ray_sets.len(),
        "skin_to_body_m": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99), "samples": residual.len()},
        "history": history,
        "head_curvature": {"weight": w_curve, "schedule_from": 0.35},
    });
    (y, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curvature_suppresses_a_surface_spike_without_pinning_landmarks_to_the_template() {
        let template = vec![V::new(-1., -1., 0.), V::new(1., -1., 0.),
            V::new(1., 1., 0.), V::new(-1., 1., 0.), V::zeros()];
        let shift = V::new(4., -2., 3.);
        let edges = vec![(0, 1, 1.), (1, 2, 1.), (2, 3, 1.), (0, 3, 1.),
            (0, 4, 1.), (1, 4, 1.), (2, 4, 1.), (3, 4, 1.)];
        let mut rows = Rows { single: (0..4).map(|i| (i, template[i] + shift, 10.)).collect(),
            sets: vec![], linear: vec![] };
        rows.single.push((4, shift + V::z(), 1.));
        let noisy = solve(&template, &edges, 0.1, &rows);
        rows.linear.push((vec![(4, 1.), (0, -0.25), (1, -0.25),
            (2, -0.25), (3, -0.25)], V::zeros(), 4.));
        let smooth = solve(&template, &edges, 0.1, &rows);
        assert!((noisy[4] - shift).norm() > 0.5);
        assert!((smooth[4] - shift).norm() < 0.15);
        for i in 0..4 {
            assert!((smooth[i] - template[i] - shift).norm() < 0.02);
        }
    }
    /// A 6 x 6 sheet in the xy plane, bent a little so it is not degenerate.
    fn sheet() -> (Vec<V>, Vec<(usize, usize, f64)>) {
        let mut points = vec![];
        let mut edges = vec![];
        for j in 0..6 {
            for i in 0..6 {
                points.push(V::new(i as f64 * 0.1, j as f64 * 0.1, 0.02 * ((i * j) as f64).sin()));
                let k = j * 6 + i;
                if i > 0 {
                    edges.push((k - 1, k, 1.0));
                }
                if j > 0 {
                    edges.push((k - 6, k, 1.0));
                }
            }
        }
        (points, edges)
    }

    #[test]
    fn grid_finds_the_nearest_point_within_reach() {
        let (points, _) = sheet();
        let grid = Grid::new(&points, |i| i % 2 == 0, 0.07);
        for q in [V::new(0.23, 0.31, 0.0), V::new(-0.2, 0.1, 0.05), V::new(0.5, 0.5, 0.3)] {
            let brute = (0..points.len())
                .filter(|i| i % 2 == 0)
                .map(|i| (i, (points[i] - q).norm()))
                .filter(|&(_, d)| d <= 0.25)
                .min_by(|a, b| a.1.total_cmp(&b.1));
            assert_eq!(grid.nearest(&points, &q, 0.25).map(|h| h.0), brute.map(|h| h.0));
        }
    }

    #[test]
    fn a_rigid_shift_is_reproduced_at_any_stiffness() {
        let (points, edges) = sheet();
        let shift = V::new(0.03, -0.02, 0.05);
        let rows = Rows { single: (0..points.len()).map(|i| (i, points[i] + shift, 1.0)).collect(), sets: vec![], linear: vec![] };
        for alpha in [50.0, 1.0] {
            let y = solve(&points, &edges, alpha, &rows);
            assert!(y.iter().zip(&points).all(|(a, b)| (a - b - shift).norm() < 1e-6));
        }
    }

    #[test]
    fn unmeasured_vertices_follow_the_measured_ones_scale() {
        // Data on the left half only: a 1.3x scaling about the origin. The
        // right half has no data and must continue it, not stay at rest.
        let (points, edges) = sheet();
        let rows = Rows {
            single: (0..points.len()).filter(|i| i % 6 < 3).map(|i| (i, points[i] * 1.3, 1.0)).collect(),
            sets: vec![],
            linear: vec![],
        };
        let y = solve(&points, &edges, 20.0, &rows);
        let worst = y.iter().zip(&points).map(|(a, b)| (a - b * 1.3).norm()).fold(0.0, f64::max);
        assert!(worst < 2e-3, "{worst}");
    }

    #[test]
    fn a_set_row_of_one_vertex_equals_a_single_row() {
        let (points, edges) = sheet();
        let hold: Vec<(usize, V, f64)> = (0..6).map(|i| (i, points[i], 1.0)).collect();
        let target = points[20] + V::new(0.01, 0.0, 0.04);
        let mut single = hold.clone();
        single.push((20, target, 3.0));
        let a = solve(&points, &edges, 2.0, &Rows { single, sets: vec![], linear: vec![] });
        let b = solve(&points, &edges, 2.0, &Rows { single: hold, sets: vec![(vec![20], target, 3.0)], linear: vec![] });
        assert!(a.iter().zip(&b).all(|(p, q)| (p - q).norm() < 1e-9));
    }

    #[test]
    fn an_unopposed_set_row_puts_the_mean_on_its_target() {
        let (points, edges) = sheet();
        let ids = vec![14usize, 15, 20, 21];
        let target = centroid(&points, &ids) + V::new(0.02, 0.0, 0.04);
        let y = solve(&points, &edges, 2.0, &Rows { single: vec![], sets: vec![(ids.clone(), target, 5.0)], linear: vec![] });
        assert!((centroid(&y, &ids) - target).norm() < 1e-5);
    }
}
