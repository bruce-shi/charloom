 use crate::geom::{V, Weld, a, percentiles, v, vertex_normals};
use crate::scan::build_query;
use charloom_core::Vec3;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::HashMap;

/// Plane barycentric coordinates of `p` over triangle `x` (they may lie
/// outside `[0, 1]`) and its signed gap along the triangle's unit normal.
pub fn frame_of(p: V, x: [V; 3]) -> Option<([f64; 3], f64)> {
    let (e1, e2) = (x[1] - x[0], x[2] - x[0]);
    let n = e1.cross(&e2).try_normalize(1e-14)?;
    let h = (p - x[0]).dot(&n);
    let q = p - x[0] - n * h;
    let (a11, a12, a22) = (e1.dot(&e1), e1.dot(&e2), e2.dot(&e2));
    let (b1, b2) = (q.dot(&e1), q.dot(&e2));
    let det = a11 * a22 - a12 * a12;
    if det.abs() < 1e-24 {
        return None;
    }
    let s = (a22 * b1 - a12 * b2) / det;
    let t = (a11 * b2 - a12 * b1) / det;
    Some(([1.0 - s - t, s, t], h))
}

/// The point with local frame `(bary, gap)` over (deformed) triangle `x`.
pub fn rebuild(bary: [f64; 3], gap: f64, x: [V; 3]) -> V {
    let n = (x[1] - x[0])
        .cross(&(x[2] - x[0]))
        .try_normalize(1e-14)
        .unwrap_or_else(V::zeros);
    x[0] * bary[0] + x[1] * bary[1] + x[2] * bary[2] + n * gap
}

fn smoothstep(lo: f64, hi: f64, x: f64) -> f64 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The body the cloth is bound to (rest positions).
pub struct Body<'a> {
    pub positions: &'a [Vec3],
    pub triangles: &'a [[u32; 3]],
    /// Hand triangles (wrist and fingers). Cloth pieces that are mostly not
    /// on the hands bind to, and clear against, the body without them: a
    /// hand resting on a robe must not push the robe around.
    pub hands: Option<&'a [bool]>,
}

/// Closest-point queries over a body, with and without its hands; the
/// second maps its triangles back to the body's.
struct Surfaces {
    full: charloom_engine::surface_query::SurfaceQuery,
    no_hands: Option<(charloom_engine::surface_query::SurfaceQuery, Vec<usize>)>,
}

impl Surfaces {
    fn new(positions: &[Vec3], body: &Body) -> Self {
        let no_hands = body.hands.filter(|h| h.iter().any(|x| *x)).map(|h| {
            let kept: Vec<usize> = (0..body.triangles.len()).filter(|&t| !h[t]).collect();
            let tris: Vec<[u32; 3]> = kept.iter().map(|&t| body.triangles[t]).collect();
            (build_query(positions, &tris), kept)
        });
        Self {
            full: build_query(positions, body.triangles),
            no_hands,
        }
    }

    /// Closest point (optionally normal-filtered) with the body triangle index.
    fn closest(
        &self,
        p: Vec3,
        reach: f64,
        normal: Option<(Vec3, f64)>,
        no_hands: bool,
    ) -> Option<(charloom_engine::surface_query::SurfaceHit, usize)> {
        match (&self.no_hands, no_hands) {
            (Some((q, map)), true) => q.closest_point(p, reach, normal).map(|h| {
                let t = map[h.source[1]];
                (h, t)
            }),
            _ => self.full.closest_point(p, reach, normal).map(|h| {
                let t = h.source[1];
                (h, t)
            }),
        }
    }
}

/// One mesh to transport.
pub struct ClothPart<'a> {
    pub positions: &'a [Vec3],
    pub normals: &'a [Vec3],
    pub triangles: &'a [[u32; 3]],
    pub hair: bool,
    pub rigid: bool,
}

/// All transported meshes welded by position (pieces cut from one fused scan
/// share their cut vertices, so a seam moves as one).
pub struct Cloth {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub adjacency: Vec<Vec<u32>>,
    pub hair: Vec<bool>,
    /// Rigid connected component of the vertex, if it belongs to a rigid part.
    pub rigid: Vec<Option<u32>>,
    /// Welded id of every vertex of every part.
    pub ids: Vec<Vec<u32>>,
    /// First part of every welded vertex.
    pub part: Vec<u32>,
    /// Welded triangles and their part.
    pub triangles: Vec<([u32; 3], u32)>,
}

impl Cloth {
    pub fn new(parts: &[ClothPart]) -> Self {
        let all: Vec<Vec3> = parts
            .iter()
            .flat_map(|p| p.positions.iter().copied())
            .collect();
        let weld = Weld::new(&all, 1e-7);
        let n = weld.count;
        let mut ids = vec![];
        let mut positions = vec![[0.0; 3]; n];
        let mut normal_sum = vec![V::zeros(); n];
        let mut hair = vec![false; n];
        let mut rigid_vertex = vec![false; n];
        let mut part = vec![0u32; n];
        let mut seen = vec![false; n];
        let mut o = 0;
        for (pi, p) in parts.iter().enumerate() {
            let row: Vec<u32> = (0..p.positions.len())
                .map(|i| weld.id[o + i] as u32)
                .collect();
            for (i, &w) in row.iter().enumerate() {
                if !seen[w as usize] {
                    seen[w as usize] = true;
                    part[w as usize] = pi as u32;
                }
                positions[w as usize] = p.positions[i];
                normal_sum[w as usize] += v(p.normals[i]);
                hair[w as usize] |= p.hair;
                rigid_vertex[w as usize] |= p.rigid;
            }
            o += p.positions.len();
            ids.push(row);
        }
        let mut adjacency: Vec<Vec<u32>> = vec![vec![]; n];
        let mut parent: Vec<u32> = (0..n as u32).collect();
        fn find(parent: &mut [u32], mut k: u32) -> u32 {
            while parent[k as usize] != k {
                let up = parent[parent[k as usize] as usize];
                parent[k as usize] = up;
                k = up;
            }
            k
        }
        let mut triangles = vec![];
        for (pi, (p, row)) in parts.iter().zip(&ids).enumerate() {
            for t in p.triangles {
                let w = t.map(|i| row[i as usize]);
                triangles.push((w, pi as u32));
                for (x, y) in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
                    if x != y {
                        adjacency[x as usize].push(y);
                        adjacency[y as usize].push(x);
                        if p.rigid {
                            let (rx, ry) = (find(&mut parent, x), find(&mut parent, y));
                            if rx != ry {
                                parent[rx as usize] = ry;
                            }
                        }
                    }
                }
            }
        }
        adjacency.par_iter_mut().for_each(|r| {
            r.sort_unstable();
            r.dedup();
        });
        let rigid = (0..n as u32)
            .map(|k| rigid_vertex[k as usize].then(|| find(&mut parent, k)))
            .collect();
        Self {
            positions,
            normals: normal_sum
                .iter()
                .map(|s| a(s.try_normalize(1e-20).unwrap_or(V::y())))
                .collect(),
            adjacency,
            hair,
            rigid,
            ids,
            part,
            triangles,
        }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Per-part vertex values from welded ones.
    pub fn split<T: Clone>(&self, welded: &[T]) -> Vec<Vec<T>> {
        self.ids
            .iter()
            .map(|row| row.iter().map(|&w| welded[w as usize].clone()).collect())
            .collect()
    }
}

/// Inner-garment point under a cloth vertex: welded triangle, barycentrics,
/// rest gap.
pub type LayerAnchor = ([u32; 3], [f64; 3], f64);

/// Binding of every welded cloth vertex to the rest body.
pub struct Binding {
    /// Body triangle, plane barycentrics and normal gap (local frame).
    pub frame: Vec<Option<(u32, [f64; 3], f64)>>,
    /// Barycentrics of the closest point (inside the triangle).
    pub anchor: Vec<[f64; 3]>,
    pub distance: Vec<f64>,
    /// Signed distance to the rest body (closest point, any orientation)
    /// along its interpolated normal: the clearance pass's reference.
    pub signed: Vec<f64>,
    /// 0: local frame, 1: anchor translation.
    pub near: Vec<f64>,
    /// Share taken from the harmonic extension (loose cloth).
    pub far: Vec<f64>,
    /// Bound to (and cleared against) the body without its hands.
    pub no_hands: Vec<bool>,
    /// Layering: the nearest point of an inner garment (closer to the body)
    /// under this vertex — welded triangle, barycentrics, rest gap.
    pub layer: Vec<Option<LayerAnchor>>,
}

/// Thresholds as fractions of the body height.
const TIGHT: f64 = 0.01;
const MEDIUM: f64 = 0.03;
const LOOSE: f64 = 0.06;
/// Distance from the body within which the clearance pass judges inside /
/// outside (fraction of the height).
pub const CONTACT: f64 = 0.02;

/// Interpolated (welded) body normal at a closest-point hit.
fn interpolated(normals: &[Vec3], tri: [u32; 3], bary: [f64; 3]) -> V {
    (0..3)
        .fold(V::zeros(), |s, j| s + v(normals[tri[j] as usize]) * bary[j])
        .try_normalize(1e-12)
        .unwrap_or_else(V::y)
}

pub fn bind(body: &Body, cloth: &Cloth, height: f64) -> (Binding, Value) {
    let surfaces = Surfaces::new(body.positions, body);
    let weld = Weld::new(body.positions, 1e-7);
    let normals = vertex_normals(body.positions, body.triangles, &weld);
    // (frame, anchor barycentrics, distance, signed, frame and closest triangles)
    type Row = (Option<(u32, [f64; 3], f64)>, [f64; 3], f64, f64, [usize; 2]);
    let row = |k: usize, no_hands: bool| -> Row {
        let p = cloth.positions[k];
        let reach = height * 0.3;
        let Some((hit, t)) = surfaces
            .closest(p, reach, Some((cloth.normals[k], 0.0)), no_hands)
            .or_else(|| surfaces.closest(p, reach, None, no_hands))
        else {
            return (None, [0.0; 3], f64::INFINITY, 0.0, [usize::MAX; 2]);
        };
        let tri = body.triangles[t];
        let x = tri.map(|i| v(body.positions[i as usize]));
        let frame = frame_of(v(p), x).map(|(b, h)| (t as u32, b, h));
        // The clearance pass measures against the closest point of any
        // orientation; its rest reference must be measured the same way.
        let any = surfaces.closest(p, reach, None, no_hands);
        let (point, ta, bary) = any
            .as_ref()
            .map_or((hit.point, t, hit.barycentric), |(h, ta)| {
                (h.point, *ta, h.barycentric)
            });
        let n = interpolated(&normals, body.triangles[ta], bary);
        let signed = (v(p) - v(point)).dot(&n);
        (frame, hit.barycentric, hit.distance_m, signed, [t, ta])
    };
    let mut rows: Vec<Row> = (0..cloth.len())
        .into_par_iter()
        .map(|k| row(k, false))
        .collect();
    // Hand domain per piece (as in the weight transfer).
    let mut no_hands = vec![false; cloth.len()];
    let mut rebound = 0usize;
    if let (Some(h), Some(_)) = (body.hands, &surfaces.no_hands) {
        let on_hand = |r: &Row| r.4[0] != usize::MAX && h[r.4[0]];
        let parts = cloth
            .part
            .iter()
            .map(|&p| p as usize + 1)
            .max()
            .unwrap_or(0);
        let mut share = vec![(0usize, 0usize); parts];
        for (k, r) in rows.iter().enumerate() {
            let p = cloth.part[k] as usize;
            share[p].1 += 1;
            share[p].0 += usize::from(on_hand(r));
        }
        for k in 0..cloth.len() {
            let (on, all) = share[cloth.part[k] as usize];
            no_hands[k] = (on as f64) < 0.2 * all as f64;
        }
        let redo: Vec<usize> = (0..cloth.len())
            .filter(|&k| no_hands[k] && rows[k].4.iter().any(|&t| t != usize::MAX && h[t]))
            .collect();
        let again: Vec<(usize, Row)> = redo.par_iter().map(|&k| (k, row(k, true))).collect();
        rebound = again.len();
        for (k, r) in again {
            rows[k] = r;
        }
    }
    let frame: Vec<_> = rows.iter().map(|r| r.0).collect();
    let distance: Vec<f64> = rows.iter().map(|r| r.2).collect();
    let layer = layers(cloth, &distance, height);
    let near: Vec<f64> = distance
        .iter()
        .map(|&d| smoothstep(height * TIGHT, height * MEDIUM, d))
        .collect();
    let far: Vec<f64> = distance
        .iter()
        .map(|&d| smoothstep(height * MEDIUM, height * LOOSE, d))
        .collect();
    let count = |f: &dyn Fn(usize) -> bool| (0..cloth.len()).filter(|&k| f(k)).count();
    let report = json!({
        "vertices": cloth.len(),
        "unbound": count(&|k| frame[k].is_none()),
        "local_frame": count(&|k| frame[k].is_some() && near[k] < 0.5),
        "anchor_translation": count(&|k| near[k] >= 0.5 && far[k] < 0.5),
        "harmonic_loose": count(&|k| far[k] >= 0.5),
        "rigid": count(&|k| cloth.rigid[k].is_some()),
        "hair": count(&|k| cloth.hair[k]),
        "hand_domain_rebound": rebound,
        "layered_over_inner_garment": layer.iter().filter(|l| l.is_some()).count(),
        "outside_triangle": count(&|k| frame[k].is_some_and(|(_, b, _)| b.iter().any(|x| *x < -1e-6))),
        "distance_m": percentiles(distance.clone()),
        "thresholds_m": {"local_frame_below": height * TIGHT, "translation_from": height * MEDIUM, "harmonic_from": height * LOOSE},
    });
    (
        Binding {
            frame,
            anchor: rows.iter().map(|r| r.1).collect(),
            distance,
            signed: rows.iter().map(|r| r.3).collect(),
            near,
            far,
            no_hands,
            layer,
        },
        report,
    )
}

/// For every (non-hair, non-rigid) cloth vertex, the nearest surface of
/// another piece within reach that lies under it (the vertex is outside that
/// surface and further from the body): a tunic over trousers, armour over a
/// shirt. Pieces are bound to the body independently, so without this an
/// outer layer that follows the body more loosely could sink through.
fn layers(cloth: &Cloth, distance: &[f64], height: f64) -> Vec<Option<LayerAnchor>> {
    let n = cloth.len();
    let mut out = vec![None; n];
    let parts = cloth
        .part
        .iter()
        .map(|&p| p as usize + 1)
        .max()
        .unwrap_or(0);
    if parts < 2 {
        return out;
    }
    let layerable = |k: usize| !cloth.hair[k] && cloth.rigid[k].is_none();
    let reach = height * 0.015;
    for p in 0..parts as u32 {
        let tris: Vec<[u32; 3]> = cloth
            .triangles
            .iter()
            .filter(|(t, q)| *q != p && t.iter().all(|&i| layerable(i as usize)))
            .map(|(t, _)| *t)
            .collect();
        if tris.is_empty() {
            continue;
        }
        let query = build_query(&cloth.positions, &tris);
        let found: Vec<(usize, LayerAnchor)> = (0..n)
            .into_par_iter()
            .filter(|&k| cloth.part[k] == p && layerable(k) && distance[k].is_finite())
            .filter_map(|k| {
                let hit = query.closest_point(cloth.positions[k], reach, None)?;
                let t = tris[hit.source[1]];
                let inner = (0..3)
                    .map(|j| distance[t[j] as usize] * hit.barycentric[j])
                    .sum::<f64>();
                let nrm = interpolated(&cloth.normals, t, hit.barycentric);
                let gap = (v(cloth.positions[k]) - v(hit.point)).dot(&nrm);
                (gap > 0.0 && inner < distance[k] - 1e-3).then_some((k, (t, hit.barycentric, gap)))
            })
            .collect();
        for (k, l) in found {
            out[k] = Some(l);
        }
    }
    out
}

/// Keeps every layered vertex at least `min(rest gap, clearance)` outside
/// the moved inner garment (three sweeps, so armour over a tunic over a
/// shirt settles), then spreads the pushes like the body clearance.
fn layer_clearance(binding: &Binding, cloth: &Cloth, d: &mut [V], clearance: f64) -> Value {
    let n = cloth.len();
    let mut total = vec![V::zeros(); n];
    for _ in 0..3 {
        let push: Vec<V> = (0..n)
            .into_par_iter()
            .map(|k| {
                let Some((t, b, gap)) = binding.layer[k] else {
                    return V::zeros();
                };
                let c = (0..3).fold(V::zeros(), |s, j| {
                    let i = t[j] as usize;
                    s + (v(cloth.positions[i]) + d[i]) * b[j]
                });
                let nrm = interpolated(&cloth.normals, t, b);
                let s1 = (v(cloth.positions[k]) + d[k] - c).dot(&nrm);
                let want = gap.min(clearance);
                let inner = (0..3).fold(0.0, |s, j| s + d[t[j] as usize].norm() * b[j]);
                if s1 < want - 1e-5 {
                    nrm * (want - s1).min(inner + d[k].norm() + clearance)
                } else {
                    V::zeros()
                }
            })
            .collect();
        for k in 0..n {
            d[k] += push[k];
            total[k] += push[k];
        }
    }
    let pushed = total.iter().filter(|p| p.norm() > 0.0).count();
    let max_push = total.iter().map(|p| p.norm()).fold(0.0, f64::max);
    let spread = spread_push(cloth, &total);
    for k in 0..n {
        d[k] += spread[k] - total[k];
    }
    json!({"pushed_vertices": pushed, "max_push_m": max_push})
}

 fn harmonic_fill(values: &mut [f32], k: usize, fixed: &[bool], cloth: &Cloth, cell: f64) {
    let n = fixed.len();
    if n == 0 || fixed.iter().all(|f| *f) || !fixed.iter().any(|f| *f) {
        return;
    }
    let mut keys: HashMap<(i64, i64, i64), usize> = HashMap::new();
    let cluster_of: Vec<usize> = cloth.positions[..n]
        .iter()
        .map(|p| {
            let key = (
                (p[0] / cell).floor() as i64,
                (p[1] / cell).floor() as i64,
                (p[2] / cell).floor() as i64,
            );
            let next = keys.len();
            *keys.entry(key).or_insert(next)
        })
        .collect();
    let m = keys.len();
     
    let mut coarse = vec![0f32; m * k];
    let mut fixed_members = vec![0f32; m];
    for i in (0..n).filter(|&i| fixed[i]) {
        let c = cluster_of[i];
        fixed_members[c] += 1.0;
        for j in 0..k {
            coarse[c * k + j] += values[i * k + j];
        }
    }
    let cfixed: Vec<bool> = fixed_members.iter().map(|x| *x > 0.0).collect();
    for c in 0..m {
        if cfixed[c] {
            for j in 0..k {
                coarse[c * k + j] /= fixed_members[c];
            }
        }
    }
    let mut cadj: Vec<Vec<u32>> = vec![vec![]; m];
    for i in 0..n {
        for &j in &cloth.adjacency[i] {
            let (x, y) = (cluster_of[i], cluster_of[j as usize]);
            if x != y {
                cadj[x].push(y as u32);
            }
        }
    }
    cadj.par_iter_mut().for_each(|r| {
        r.sort_unstable();
        r.dedup();
    });
    // Breadth-first start values, then Jacobi relaxation.
    let mut reached = cfixed.clone();
    let mut frontier: Vec<usize> = (0..m).filter(|&c| cfixed[c]).collect();
    while !frontier.is_empty() {
        let mut next = vec![];
        for &c in &frontier {
            for &d in &cadj[c] {
                let d = d as usize;
                if !reached[d] {
                    reached[d] = true;
                    for j in 0..k {
                        coarse[d * k + j] = coarse[c * k + j];
                    }
                    next.push(d);
                }
            }
        }
        frontier = next;
    }
    for _ in 0..800 {
        let next: Vec<f32> = (0..m)
            .into_par_iter()
            .flat_map_iter(|c| {
                let row: Vec<f32> = if cfixed[c] || !reached[c] || cadj[c].is_empty() {
                    coarse[c * k..(c + 1) * k].to_vec()
                } else {
                    let mut acc = vec![0f32; k];
                    for &d in &cadj[c] {
                        for j in 0..k {
                            acc[j] += coarse[d as usize * k + j];
                        }
                    }
                    let inv = 1.0 / cadj[c].len() as f32;
                    acc.iter().map(|x| x * inv).collect()
                };
                row
            })
            .collect();
        let change = next
            .iter()
            .zip(&coarse)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f32::max);
        coarse = next;
        if change < 1e-6 {
            break;
        }
    }
    let free: Vec<bool> = (0..n)
        .map(|i| !fixed[i] && reached[cluster_of[i]])
        .collect();
    for i in (0..n).filter(|&i| free[i]) {
        let c = cluster_of[i];
        values[i * k..(i + 1) * k].copy_from_slice(&coarse[c * k..(c + 1) * k]);
    }
    for _ in 0..8 {
        let next: Vec<f32> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let row: Vec<f32> = if !free[i] || cloth.adjacency[i].is_empty() {
                    values[i * k..(i + 1) * k].to_vec()
                } else {
                    let mut acc = values[i * k..(i + 1) * k].to_vec();
                    for &d in &cloth.adjacency[i] {
                        for j in 0..k {
                            acc[j] += values[d as usize * k + j];
                        }
                    }
                    let inv = 1.0 / (1 + cloth.adjacency[i].len()) as f32;
                    acc.iter().map(|x| x * inv).collect()
                };
                row
            })
            .collect();
        values.copy_from_slice(&next);
    }
}

/// Settings of [`transport`].
pub struct Settings {
    pub height: f64,
    /// Gap the clearance pass keeps between body and cloth, metres.
    pub clearance: f64,
}

/// Displacement of every welded cloth vertex for each new body shape
/// (`targets[m]`: new positions of the rest body's vertices), with the
/// clearance pass applied per target. Returns the displacements and a report
/// per target.
pub fn transport(
    binding: &Binding,
    body: &Body,
    cloth: &Cloth,
    targets: &[Vec<Vec3>],
    settings: &Settings,
) -> (Vec<Vec<V>>, Vec<Value>) {
    let n = cloth.len();
    let mt = targets.len();
    let k = mt * 3;
    if n == 0 || mt == 0 {
        return (vec![vec![]; mt], vec![json!(null); mt]);
    }
    let height = settings.height;
     
    let weld = Weld::new(body.positions, 1e-7);
    let rest_normals = vertex_normals(body.positions, body.triangles, &weld);
    let target_normals: Vec<Vec<Vec3>> = targets
        .iter()
        .map(|t| vertex_normals(t, body.triangles, &weld))
        .collect();
    let mut values = vec![0f32; n * k];
    values.par_chunks_mut(k).enumerate().for_each(|(i, row)| {
        let Some((t, bary, gap)) = binding.frame[i] else {
            return;
        };
        let tri = body.triangles[t as usize];
        let rest = tri.map(|j| v(body.positions[j as usize]));
        let c = binding.anchor[i];
        let overhang = -bary.iter().copied().fold(f64::INFINITY, f64::min);
        let frame_share = (1.0 - binding.near[i]) * (1.0 - smoothstep(0.5, 2.0, overhang));
        for (m, target) in targets.iter().enumerate() {
            let x = tri.map(|j| v(target[j as usize]));
            let turn =
                interpolated(&target_normals[m], tri, c) - interpolated(&rest_normals, tri, c);
            let framed =
                (0..3).fold(V::zeros(), |s, j| s + (x[j] - rest[j]) * bary[j]) + turn * gap;
            let anchor = (0..3).fold(V::zeros(), |s, j| s + (x[j] - rest[j]) * c[j]);
            let d = framed * frame_share + anchor * (1.0 - frame_share);
            for c in 0..3 {
                row[m * 3 + c] = d[c] as f32;
            }
        }
    });
    let near_values = values.clone();
    // 2. Loose cloth: harmonic extension from the cloth near the body.
    let fixed: Vec<bool> = (0..n)
        .map(|i| binding.frame[i].is_some() && binding.far[i] <= 0.0)
        .collect();
    harmonic_fill(&mut values, k, &fixed, cloth, height * 0.012);
    for i in 0..n {
        let f = binding.far[i] as f32;
        for c in 0..k {
            values[i * k + c] = near_values[i * k + c] * (1.0 - f) + values[i * k + c] * f;
        }
    }
    // 3. Rigid parts move as a whole with their attachment.
    let mut groups: HashMap<u32, Vec<usize>> = HashMap::new();
    for i in 0..n {
        if let Some(g) = cloth.rigid[i] {
            groups.entry(g).or_default().push(i);
        }
    }
    for members in groups.values() {
        let nearest = members
            .iter()
            .map(|&i| binding.distance[i])
            .fold(f64::INFINITY, f64::min);
        let band = height * 0.01;
        let mut acc = vec![0f64; k];
        let mut total = 0.0;
        for &i in members {
            let w = 1.0 - (binding.distance[i] - nearest) / band;
            if w > 0.0 {
                total += w;
                for c in 0..k {
                    acc[c] += values[i * k + c] as f64 * w;
                }
            }
        }
        if total > 0.0 {
            for &i in members {
                for c in 0..k {
                    values[i * k + c] = (acc[c] / total) as f32;
                }
            }
        }
    }
    // 4. Light smoothing: removes spikes from isolated bad bindings.
    for _ in 0..3 {
        let next: Vec<f32> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                let nb = &cloth.adjacency[i];
                let row: Vec<f32> = if cloth.rigid[i].is_some() || nb.is_empty() {
                    values[i * k..(i + 1) * k].to_vec()
                } else {
                    let inv = 0.5 / nb.len() as f32;
                    (0..k)
                        .map(|c| {
                            values[i * k + c] * 0.5
                                + nb.iter().map(|&d| values[d as usize * k + c]).sum::<f32>() * inv
                        })
                        .collect()
                };
                row
            })
            .collect();
        values = next;
    }
    // 5. Clearance per target.
    let mut out = vec![];
    let mut reports = vec![];
    for (m, target) in targets.iter().enumerate() {
        let mut d: Vec<V> = (0..n)
            .map(|i| {
                V::new(
                    values[i * k + m * 3] as f64,
                    values[i * k + m * 3 + 1] as f64,
                    values[i * k + m * 3 + 2] as f64,
                )
            })
            .collect();
        let mut report = clearance(binding, body, cloth, target, &weld, &mut d, settings);
        report["layers"] = layer_clearance(binding, cloth, &mut d, settings.clearance);
        out.push(d);
        reports.push(report);
    }
    (out, reports)
}

 fn clearance(
    binding: &Binding,
    body: &Body,
    cloth: &Cloth,
    target: &[Vec3],
    weld: &Weld,
    d: &mut [V],
    settings: &Settings,
) -> Value {
    let height = settings.height;
    let surfaces = Surfaces::new(target, body);
    let normals = vertex_normals(target, body.triangles, weld);
    let reach = height * 0.05;
    let push: Vec<V> = (0..cloth.len())
        .into_par_iter()
        .map(|i| {
            if cloth.hair[i]
                || cloth.rigid[i].is_some()
                || binding.frame[i].is_none()
                || binding.signed[i] < -height * 0.004
                || binding.distance[i] > reach
            {
                return V::zeros();
            }
            let q = v(cloth.positions[i]) + d[i];
            // Only near the surface: at several centimetres the sign of a
            // closest point in a concave region (hip under a hanging hand,
            // the armpit) is not a reliable inside test.
            let Some((hit, t)) =
                surfaces.closest(a(q), height * CONTACT, None, binding.no_hands[i])
            else {
                return V::zeros();
            };
            // Same body region as the binding: the new closest point lies
            // near the moved anchor. A cloth vertex whose closest point jumped
            // to another part (a robe between hip and forearm) is left alone.
            let (ft, _, _) = binding.frame[i].unwrap();
            let ftri = body.triangles[ft as usize];
            let anchor = (0..3).fold(V::zeros(), |acc, j| {
                acc + v(target[ftri[j] as usize]) * binding.anchor[i][j]
            });
            if (v(hit.point) - anchor).norm() > height * CONTACT {
                return V::zeros();
            }
            let tri = body.triangles[t];
            let n = interpolated(&normals, tri, hit.barycentric);
            let s = (q - v(hit.point)).dot(&n);
            let want = binding.signed[i].min(settings.clearance);
            let moved = (0..3)
                .fold(V::zeros(), |acc, j| {
                    let k = tri[j] as usize;
                    acc + (v(target[k]) - v(body.positions[k])) * hit.barycentric[j]
                })
                .norm();
            if s < want - 1e-5 {
                n * (want - s).min(moved + d[i].norm() + settings.clearance)
            } else {
                V::zeros()
            }
        })
        .collect();
    let pushed = push.iter().filter(|p| p.norm() > 0.0).count();
    let max_push = push.iter().map(|p| p.norm()).fold(0.0, f64::max);
    let spread = spread_push(cloth, &push);
    for (x, p) in d.iter_mut().zip(&spread) {
        *x += p;
    }
    json!({"pushed_vertices": pushed, "max_push_m": max_push, "spread_vertices": spread.iter().filter(|p| p.norm() > 0.0).count()})
}

 fn spread_push(cloth: &Cloth, raw: &[V]) -> Vec<V> {
    let fixed =
        |i: usize| cloth.hair[i] || cloth.rigid[i].is_some() || cloth.adjacency[i].is_empty();
    let mean = |f: &[V], i: usize, with_self: bool| {
        let nb = &cloth.adjacency[i];
        let s = nb
            .iter()
            .fold(if with_self { f[i] } else { V::zeros() }, |s, &j| {
                s + f[j as usize]
            });
        s / (nb.len() + usize::from(with_self)) as f64
    };
    let mut f = raw.to_vec();
    for _ in 0..6 {
        f = (0..cloth.len())
            .into_par_iter()
            .map(|i| {
                if fixed(i) {
                    return f[i];
                }
                let m = mean(&f, i, false);
                if m.norm() > f[i].norm() { m } else { f[i] }
            })
            .collect();
    }
    for _ in 0..8 {
        f = (0..cloth.len())
            .into_par_iter()
            .map(|i| {
                if fixed(i) {
                    return f[i];
                }
                let m = mean(&f, i, true);
                if raw[i].norm() > m.norm() { raw[i] } else { m }
            })
            .collect();
    }
    f
}

/// Laplacian magnitude of a displacement field over the cloth (how far a
/// vertex moves differently from its neighbours): spikes from bad bindings
/// show up as outliers.
pub fn roughness(cloth: &Cloth, d: &[V]) -> Value {
    let lap: Vec<f64> = (0..cloth.len())
        .into_par_iter()
        .filter(|&i| !cloth.adjacency[i].is_empty())
        .map(|i| {
            let nb = &cloth.adjacency[i];
            let mean = nb.iter().fold(V::zeros(), |s, &j| s + d[j as usize]) / nb.len() as f64;
            (d[i] - mean).norm()
        })
        .collect();
    let over_2mm = lap.iter().filter(|x| **x > 0.002).count();
    json!({"laplacian_m": percentiles(lap), "vertices_over_2mm": over_2mm})
}

/// Values on a body surface interpolated at points lying on it (an exported
/// skin whose topology differs from the morphed body: cut digits, kept head).
pub fn interpolate_on_body(body: &Body, points: &[Vec3], values: &[Vec<Vec3>]) -> Vec<Vec<Vec3>> {
    let query = build_query(body.positions, body.triangles);
    let hits: Vec<Option<([u32; 3], [f64; 3])>> = points
        .par_iter()
        .map(|p| {
            query
                .closest_point(*p, 0.05, None)
                .map(|h| (body.triangles[h.source[1]], h.barycentric))
        })
        .collect();
    values
        .iter()
        .map(|field| {
            hits.iter()
                .map(|h| match h {
                    Some((t, b)) => {
                        a((0..3).fold(V::zeros(), |s, j| s + v(field[t[j] as usize]) * b[j]))
                    }
                    None => [0.0; 3],
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Matrix4;

    fn load_fixture() -> Value {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/garment-retarget/local-frame-v1.json"
        );
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    fn points(x: &Value) -> Vec<V> {
        x.as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let p: Vec<f64> = p
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| c.as_f64().unwrap())
                    .collect();
                V::new(p[0], p[1], p[2])
            })
            .collect()
    }

         #[test]
    fn local_frame_fixture_rebuilds_expected_points_and_delta_only_fails() {
        let f = load_fixture();
        let body = points(&f["source_body"]);
        let tris: Vec<[usize; 3]> = f["body_triangles"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| {
                let t: Vec<usize> = t
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|i| i.as_u64().unwrap() as usize)
                    .collect();
                [t[0], t[1], t[2]]
            })
            .collect();
        let rows: Vec<f64> = f["target_affine"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|r| r.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
            .collect();
        let affine = Matrix4::from_row_slice(&rows);
        let map = |p: V| (affine * p.push(1.0)).xyz();
        let target: Vec<V> = body.iter().map(|p| map(*p)).collect();
        let garment = points(&f["garment_points"]);
        let expected = points(&f["expected_garment_points"]);
        let mut legacy_worst: f64 = 0.0;
        for (p, e) in garment.iter().zip(&expected) {
            // Nearest triangle by plane-projected distance to its centroid.
            let t = tris
                .iter()
                .min_by(|x, y| {
                    let c = |t: &[usize; 3]| (body[t[0]] + body[t[1]] + body[t[2]]) / 3.0;
                    let dx = (p.xy() - c(x).xy()).norm();
                    let dy = (p.xy() - c(y).xy()).norm();
                    dx.total_cmp(&dy)
                })
                .unwrap();
            let rest = t.map(|i| body[i]);
            let new = t.map(|i| target[i]);
            let (bary, gap) = frame_of(*p, rest).unwrap();
            assert!((rebuild(bary, gap, rest) - p).norm() < 1e-12);
            let got = rebuild(bary, gap, new);
            assert!((got - e).norm() < 1e-9, "{got:?} vs {e:?}");
            // Legacy: body displacement at the anchor added to the point.
            let anchor = (0..3).fold(V::zeros(), |s, j| s + (new[j] - rest[j]) * bary[j]);
            legacy_worst = legacy_worst.max((p + anchor - e).norm());
        }
        // Outside-edge points keep their tangential offset.
        assert!(
            garment.iter().any(|p| p.x > 1.0),
            "fixture covers outside-edge cases"
        );
        assert!(
            legacy_worst > 0.01,
            "delta-only transport must fail: {legacy_worst}"
        );
    }

    /// A cylinder "torso" inside a flared "shirt" whose hem hangs far from
    /// it; the torso grows a 5 cm belly near the tight top. Nothing pokes
    /// through, the gap is kept, the field stays smooth and the loose hem
    /// does not overshoot the motion of the cloth near the body.
    #[test]
    fn growing_body_keeps_cloth_outside_and_smooth() {
        let ring = |r: f64, y: f64, segments: usize| -> Vec<V> {
            (0..segments)
                .map(|s| {
                    let t = s as f64 / segments as f64 * std::f64::consts::TAU;
                    V::new(r * t.cos(), y, r * t.sin())
                })
                .collect()
        };
        let tube = |radius: &dyn Fn(f64) -> f64, ys: &[f64], segments: usize| {
            let mut p = vec![];
            for &y in ys {
                p.extend(ring(radius(y), y, segments));
            }
            let mut t = vec![];
            for r in 0..ys.len() - 1 {
                for s in 0..segments {
                    let i = (r * segments + s) as u32;
                    let j = (r * segments + (s + 1) % segments) as u32;
                    let seg = segments as u32;
                    // Outward-facing winding.
                    t.push([i, i + seg, j]);
                    t.push([j, i + seg, j + seg]);
                }
            }
            (p, t)
        };
        let ys: Vec<f64> = (0..=40).map(|i| i as f64 * 0.025).collect();
        let (body_v, body_t) = tube(&|_| 0.15, &ys, 48);
        let body_p: Vec<Vec3> = body_v.iter().map(|p| a(*p)).collect();
        let grown: Vec<Vec3> = body_v
            .iter()
            .map(|p| {
                let r = p.xz().norm();
                let bump = 0.05 * (-(p.y - 0.35).powi(2) / 0.01).exp();
                a(V::new(p.x * (r + bump) / r, p.y, p.z * (r + bump) / r))
            })
            .collect();
        let shirt_ys: Vec<f64> = (0..=24).map(|i| 0.2 + i as f64 * 0.025).collect();
        let (shirt_v, shirt_t) = tube(&|y| 0.157 + (y - 0.2) * 0.3, &shirt_ys, 64);
        let shirt_p: Vec<Vec3> = shirt_v.iter().map(|p| a(*p)).collect();
        let shirt_n: Vec<Vec3> = shirt_v
            .iter()
            .map(|p| a(V::new(p.x, 0.0, p.z).normalize()))
            .collect();
        let cloth = Cloth::new(&[ClothPart {
            positions: &shirt_p,
            normals: &shirt_n,
            triangles: &shirt_t,
            hair: false,
            rigid: false,
        }]);
        let body = Body {
            positions: &body_p,
            triangles: &body_t,
            hands: None,
        };
        let height = 1.7;
        let (binding, report) = bind(&body, &cloth, height);
        assert_eq!(report["unbound"], json!(0));
        assert!(report["harmonic_loose"].as_u64().unwrap() > 0);
        let (d, reports) = transport(
            &binding,
            &body,
            &cloth,
            std::slice::from_ref(&grown),
            &Settings {
                height,
                clearance: 0.004,
            },
        );
        let query = build_query(&grown, &body_t);
        for (i, p) in cloth.positions.iter().enumerate() {
            let q = v(*p) + d[0][i];
            let r = q.xz().norm();
            let hit = query.closest_point(a(q), 1.0, None).unwrap();
            let inside = r < v(hit.point).xz().norm();
            assert!(!inside, "vertex {i} inside the grown body");
            let gap = r - v(hit.point).xz().norm();
            assert!(
                gap > binding.signed[i].min(0.004) - 1e-3,
                "gap {gap} at {i}"
            );
        }
        let rough = roughness(&cloth, &d[0]);
        assert!(
            rough["laplacian_m"]["max"].as_f64().unwrap() < 0.004,
            "{rough}"
        );
        assert!(reports[0]["max_push_m"].as_f64().unwrap() < 0.01);
        let moved = |keep: &dyn Fn(usize) -> bool| {
            (0..cloth.len())
                .filter(|&i| keep(i))
                .map(|i| d[0][i].norm())
                .fold(0.0, f64::max)
        };
        let near = moved(&|i| binding.far[i] == 0.0);
        let hem = moved(&|i| cloth.positions[i][1] > 0.79);
        assert!(binding.far.iter().any(|f| *f >= 1.0));
        assert!(hem <= near + 1e-3, "hem {hem} overshoots near {near}");
    }
}
