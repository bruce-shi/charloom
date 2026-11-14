 use crate::check::RigView;
use crate::fit::Rig;
use crate::garment::Garment;
use crate::geom::{V, Weld, a, v, vertex_normals};
use crate::prior::{PartKind, Prior, Region};
use crate::scan::{Scan, build_query};
use crate::segment::{Label, Segmentation};
use crate::source;
use anyhow::{Context, Result, bail, ensure};
use charloom_core::{Influence, MotionClip, Vec3};
use nalgebra::Matrix4;
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HeadRoute {
    /// MakeHuman's head and face rig fitted to the scan; the scan's colour is
    /// baked into MakeHuman's UVs and the eyes are MakeHuman's eyeballs.
    #[default]
    Fit,
    /// Keep the scan's own head (mesh, UVs, texture, painted eyes), rigid to
    /// MakeHuman's `head` bone and lofted onto the MakeHuman neck. Exact face,
    /// no facial rig.
    KeepScan,
}

/// MakeHuman parts removed together with its head.
pub fn removes_part(kind: PartKind) -> bool {
    matches!(
        kind,
        PartKind::Eyes
            | PartKind::Eyebrows
            | PartKind::Eyelashes
            | PartKind::Teeth
            | PartKind::Tongue
    )
}

pub struct Input<'a> {
    pub prior: &'a Prior,
    pub rig: &'a Rig,
    pub scan: &'a Scan,
    pub seg: &'a Segmentation,
    pub height: f64,
}

/// The scan head, one part per scan material.
pub struct HeadPart {
    pub name: String,
    pub material: Option<usize>,
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub triangles: Vec<[u32; 3]>,
    /// Prior joint indices (as the prior's influences).
    pub influences: Vec<Vec<Influence>>,
}

pub struct Bridge {
    /// `neck_bridge`, `wrist_bridge_l`, `wrist_bridge_r`.
    pub name: String,
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub triangles: Vec<[u32; 3]>,
    pub influences: Vec<Vec<Influence>>,
    /// Strip texture (PNG), filled by [`KeptHead::bake_bridge`].
    pub png: Vec<u8>,
    /// Ring parameter and MakeHuman UV of every body-ring vertex.
    pub(crate) bottom: Vec<(f64, [f32; 2])>,
    /// Ring parameter and scan colour (sRGB) of every scan-ring vertex.
    pub(crate) top: Vec<(f64, Option<[f64; 3]>)>,
}

pub struct KeptHead {
    pub body_positions: Vec<Vec3>,
    pub body_normals: Vec<Vec3>,
    pub body_uvs: Vec<[f32; 2]>,
    pub body_triangles: Vec<[u32; 3]>,
    /// Prior joint indices.
    pub body_influences: Vec<Vec<Influence>>,
    /// Original body triangle of every kept triangle.
    pub(crate) body_triangle_origin: Vec<usize>,
    /// Kept triangle touches a seam ring (never hidden under clothing).
    pub(crate) body_near_seam: Vec<bool>,
    /// The kept scan parts (head, hands), one per scan material each.
    pub heads: Vec<HeadPart>,
    /// One bridge per kept part.
    pub bridges: Vec<Bridge>,
         pub head: bool,
    pub report: Value,
}

#[derive(Clone, Copy)]
pub(crate) struct Frame {
    pub(crate) origin: V,
    pub(crate) axis: V,
    pub(crate) e1: V,
    pub(crate) e2: V,
}

impl Frame {
    pub(crate) fn axial(&self, p: V) -> f64 {
        (p - self.origin).dot(&self.axis)
    }
    pub(crate) fn radial(&self, p: V) -> V {
        let d = p - self.origin;
        d - self.axis * d.dot(&self.axis)
    }
    pub(crate) fn angle(&self, p: V) -> f64 {
        let d = p - self.origin;
        d.dot(&self.e2).atan2(d.dot(&self.e1))
    }
}

pub(crate) fn smoothstep(x: f64) -> f64 {
    let t = x.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

 pub(crate) fn finish(acc: BTreeMap<usize, f64>, limit: usize) -> Vec<Influence> {
    let mut out: Vec<Influence> = acc
        .into_iter()
        .filter(|(_, w)| *w > 1e-6)
        .map(|(joint, weight)| Influence { joint, weight })
        .collect();
    out.sort_by(|x, y| y.weight.total_cmp(&x.weight).then(x.joint.cmp(&y.joint)));
    out.truncate(limit);
    let sum: f64 = out.iter().map(|x| x.weight).sum();
    for x in &mut out {
        x.weight /= sum.max(1e-12);
    }
    out
}

pub(crate) fn mix(rows: &[(&[Influence], f64)], joint_map: &[usize], limit: usize) -> Vec<Influence> {
    let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
    for (row, w) in rows {
        for i in row.iter() {
            *acc.entry(joint_map[i.joint]).or_default() += i.weight * w;
        }
    }
    finish(acc, limit)
}

/// Undirected boundary edges of a triangle subset on welded ids.
pub(crate) fn boundary_edges(triangles: &[[u32; 3]], keep: &[bool], weld: &Weld) -> Vec<(usize, usize)> {
    let mut count: HashMap<(usize, usize), u32> = HashMap::new();
    for (tri, _) in triangles.iter().zip(keep).filter(|(_, k)| **k) {
        let w = tri.map(|i| weld.id[i as usize]);
        for (x, y) in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
            if x != y {
                *count.entry((x.min(y), x.max(y))).or_default() += 1;
            }
        }
    }
    let mut edges: Vec<(usize, usize)> = count
        .into_iter()
        .filter(|(_, c)| *c == 1)
        .map(|(e, _)| e)
        .collect();
    edges.sort_unstable();
    edges
}

/// Drops kept triangles at boundary vertices touched by more than two
/// boundary edges (bow-ties), until every boundary is a set of simple loops.
pub(crate) fn erode_bowties(triangles: &[[u32; 3]], keep: &mut [bool], weld: &Weld) -> usize {
    let mut dropped = 0;
    for _ in 0..12 {
        let mut degree: HashMap<usize, u32> = HashMap::new();
        for (x, y) in boundary_edges(triangles, keep, weld) {
            *degree.entry(x).or_default() += 1;
            *degree.entry(y).or_default() += 1;
        }
        let bad: HashSet<usize> = degree
            .into_iter()
            .filter(|(_, d)| *d > 2)
            .map(|(w, _)| w)
            .collect();
        if bad.is_empty() {
            break;
        }
        for (tri, k) in triangles.iter().zip(keep.iter_mut()) {
            if *k && tri.iter().any(|&i| bad.contains(&weld.id[i as usize])) {
                *k = false;
                dropped += 1;
            }
        }
    }
    dropped
}

/// Drops connected components (welded) smaller than `fraction` of the
/// largest one. Returns the dropped triangle count.
pub(crate) fn keep_large_components(
    triangles: &[[u32; 3]],
    keep: &mut [bool],
    weld: &Weld,
    fraction: f64,
) -> usize {
    let mut parent: Vec<usize> = (0..weld.count).collect();
    fn find(p: &mut [usize], mut x: usize) -> usize {
        while p[x] != x {
            p[x] = p[p[x]];
            x = p[x];
        }
        x
    }
    for (tri, _) in triangles.iter().zip(keep.iter()).filter(|(_, k)| **k) {
        let w = tri.map(|i| weld.id[i as usize]);
        for k in 1..3 {
            let (ra, rb) = (find(&mut parent, w[0]), find(&mut parent, w[k]));
            if ra != rb {
                parent[ra.max(rb)] = ra.min(rb);
            }
        }
    }
    let mut size: HashMap<usize, usize> = HashMap::new();
    let roots: Vec<Option<usize>> = triangles
        .iter()
        .zip(keep.iter())
        .map(|(tri, k)| {
            k.then(|| {
                let r = find(&mut parent, weld.id[tri[0] as usize]);
                *size.entry(r).or_default() += 1;
                r
            })
        })
        .collect();
    let largest = size.values().copied().max().unwrap_or(0) as f64;
    let mut dropped = 0;
    for (k, r) in keep.iter_mut().zip(roots) {
        if let Some(r) = r
            && (size[&r] as f64) < largest * fraction
        {
            *k = false;
            dropped += 1;
        }
    }
    dropped
}

/// Closed boundary loops (welded ids in order) of a triangle subset.
pub(crate) fn boundary_loops(triangles: &[[u32; 3]], keep: &[bool], weld: &Weld) -> Vec<Vec<usize>> {
    let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (x, y) in boundary_edges(triangles, keep, weld) {
        adj.entry(x).or_default().push(y);
        adj.entry(y).or_default().push(x);
    }
    let mut seen: HashSet<usize> = HashSet::new();
    let mut loops = vec![];
    for &start in adj.keys() {
        if seen.contains(&start) {
            continue;
        }
        let mut component = vec![];
        let mut stack = vec![start];
        while let Some(x) = stack.pop() {
            if !seen.insert(x) {
                continue;
            }
            component.push(x);
            stack.extend(adj[&x].iter().copied());
        }
        if component.iter().any(|x| adj[x].len() != 2) {
            continue;
        }
        let mut ring = vec![start];
        let (mut prev, mut cur) = (start, adj[&start][0]);
        while cur != start && ring.len() <= component.len() {
            ring.push(cur);
            let next = if adj[&cur][0] == prev {
                adj[&cur][1]
            } else {
                adj[&cur][0]
            };
            prev = cur;
            cur = next;
        }
        if ring.len() == component.len() && ring.len() >= 3 {
            loops.push(ring);
        }
    }
    loops
}

pub(crate) fn winding(frame: &Frame, points: &[V]) -> f64 {
    let n = points.len();
    let mut total = 0.0;
    for k in 0..n {
        let (a0, a1) = (frame.angle(points[k]), frame.angle(points[(k + 1) % n]));
        let mut d = a1 - a0;
        while d > std::f64::consts::PI {
            d -= std::f64::consts::TAU;
        }
        while d < -std::f64::consts::PI {
            d += std::f64::consts::TAU;
        }
        total += d;
    }
    total / std::f64::consts::TAU
}

/// The loop around the neck axis whose mean axial height is nearest `offset`.
pub(crate) fn neck_loop(
    frame: &Frame,
    loops: Vec<Vec<usize>>,
    point: impl Fn(usize) -> V,
    offset: f64,
) -> Option<Vec<usize>> {
    loops
        .into_iter()
        .filter_map(|l| {
            let pts: Vec<V> = l.iter().map(|&w| point(w)).collect();
            let wn = winding(frame, &pts);
            (wn.abs() > 0.5).then(|| {
                let mean = pts.iter().map(|p| frame.axial(*p)).sum::<f64>() / pts.len() as f64;
                ((mean - offset).abs(), l)
            })
        })
        .min_by(|x, y| x.0.total_cmp(&y.0))
        .map(|x| x.1)
}

/// The boundary loop nearest the plane at `offset` that winds about its own
/// centre (within `cylinder` of the axis), and that centre.
pub(crate) fn own_centre_loop(
    frame: &Frame,
    loops: Vec<Vec<usize>>,
    point: impl Fn(usize) -> V,
    offset: f64,
    cylinder: f64,
) -> Option<(Vec<usize>, V)> {
    loops
        .into_iter()
        .filter_map(|l| {
            let pts: Vec<V> = l.iter().map(|&w| point(w)).collect();
            let c = pts.iter().sum::<V>() / pts.len().max(1) as f64;
            let own = Frame { origin: c, ..*frame };
            (winding(&own, &pts).abs() > 0.5 && frame.radial(c).norm() < cylinder).then(|| {
                let mean = pts.iter().map(|p| frame.axial(*p)).sum::<f64>() / pts.len() as f64;
                ((mean - offset).abs(), l, c)
            })
        })
        .min_by(|x, y| x.0.total_cmp(&y.0))
        .map(|x| (x.1, x.2))
}

/// A seam ring: welded ids, counter-clockwise about the axis, starting at
/// the front, with arc-length parameters (`params[n] == 1`).
pub(crate) struct Ring {
    pub(crate) ids: Vec<usize>,
    pub(crate) points: Vec<V>,
    pub(crate) normals: Vec<V>,
    pub(crate) params: Vec<f64>,
}

impl Ring {
    pub(crate) fn new(
        frame: &Frame,
        mut ids: Vec<usize>,
        point: impl Fn(usize) -> V,
        normal: impl Fn(usize) -> V,
    ) -> Self {
        let pts: Vec<V> = ids.iter().map(|&w| point(w)).collect();
        if winding(frame, &pts) < 0.0 {
            ids.reverse();
        }
        let start = (0..ids.len())
            .min_by(|&x, &y| {
                frame
                    .angle(point(ids[x]))
                    .abs()
                    .total_cmp(&frame.angle(point(ids[y])).abs())
            })
            .unwrap_or(0);
        ids.rotate_left(start);
        let points: Vec<V> = ids.iter().map(|&w| point(w)).collect();
        let normals: Vec<V> = ids.iter().map(|&w| normal(w)).collect();
        let n = points.len();
        let mut params = vec![0.0];
        for k in 0..n {
            let last = params[k];
            params.push(last + (points[(k + 1) % n] - points[k]).norm());
        }
        let total = params[n].max(1e-12);
        for p in &mut params {
            *p /= total;
        }
        Self {
            ids,
            points,
            normals,
            params,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    /// Segment index and fraction at parameter `u` ∈ [0, 1].
    pub(crate) fn locate(&self, u: f64) -> (usize, usize, f64) {
        let n = self.len();
        let k = self.params.partition_point(|&p| p <= u).clamp(1, n) - 1;
        let span = (self.params[k + 1] - self.params[k]).max(1e-12);
        (
            k,
            (k + 1) % n,
            ((u - self.params[k]) / span).clamp(0.0, 1.0),
        )
    }

    pub(crate) fn sample(&self, u: f64) -> (V, V) {
        let (i, j, f) = self.locate(u);
        let p = self.points[i] * (1.0 - f) + self.points[j] * f;
        let n = self.normals[i] * (1.0 - f) + self.normals[j] * f;
        (p, n.try_normalize(1e-12).unwrap_or(self.normals[i]))
    }

    pub(crate) fn mean_radius(&self, frame: &Frame) -> f64 {
        self.points
            .iter()
            .map(|p| frame.radial(*p).norm())
            .sum::<f64>()
            / self.len() as f64
    }
}

/// Tangent of the loft at one end: the neck axis projected onto the
/// adjacent surface, scaled to the chord's axial length; lateral components
/// are kept only where they point along the chord and at most 3× its length
/// (no overshoot, no bulge).
pub(crate) fn end_tangent(frame: &Frame, normal: V, delta: V) -> V {
    let along = frame.axial(frame.origin + delta);
    let mut dir = frame.axis - normal * normal.dot(&frame.axis);
    dir /= dir.dot(&frame.axis).max(0.15);
    let d = dir * along;
    let mut out = frame.axis * along;
    for e in [frame.e1, frame.e2] {
        let (c, w) = (d.dot(&e), delta.dot(&e));
        if c * w > 0.0 {
            out += e * w.signum() * c.abs().min(w.abs() * 3.0);
        }
    }
    out
}

pub(crate) fn hermite(p0: V, t0: V, p1: V, t1: V, t: f64) -> V {
    let (t2, t3) = (t * t, t * t * t);
    p0 * (2.0 * t3 - 3.0 * t2 + 1.0)
        + t0 * (t3 - 2.0 * t2 + t)
        + p1 * (-2.0 * t3 + 3.0 * t2)
        + t1 * (t3 - t2)
}

/// Triangulates the strip between two open rows (first and last entries of
/// each are the same ring point) by merging their parameters.
pub(crate) fn zipper(low: &[u32], lu: &[f64], high: &[u32], hu: &[f64], out: &mut Vec<[u32; 3]>) {
    let (nl, nh) = (low.len() - 1, high.len() - 1);
    let (mut i, mut j) = (0, 0);
    while i < nl || j < nh {
        if i < nl && (j == nh || lu[i + 1] <= hu[j + 1]) {
            out.push([low[i], low[i + 1], high[j]]);
            i += 1;
        } else {
            out.push([low[i], high[j + 1], high[j]]);
            j += 1;
        }
    }
}

pub(crate) fn segment_crosses(p: V, q: V, t: &[V; 3]) -> bool {
    const EPS: f64 = 1e-7;
    let d = q - p;
    let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
    let h = d.cross(&e2);
    let det = e1.dot(&h);
    if det.abs() < 1e-20 {
        return false;
    }
    let f = 1.0 / det;
    let s = p - t[0];
    let u = f * s.dot(&h);
    if !(EPS..=1.0 - EPS).contains(&u) {
        return false;
    }
    let qv = s.cross(&e1);
    let w = f * d.dot(&qv);
    if w < EPS || u + w > 1.0 - EPS {
        return false;
    }
    let s = f * e2.dot(&qv);
    s > EPS && s < 1.0 - EPS
}

/// Strict crossing of two triangles (an edge of one passes through the
/// interior of the other).
pub(crate) fn triangles_cross(x: &[V; 3], y: &[V; 3]) -> bool {
    (0..3).any(|k| segment_crosses(x[k], x[(k + 1) % 3], y))
        || (0..3).any(|k| segment_crosses(y[k], y[(k + 1) % 3], x))
}

/// Crossing pairs between the bridge and the nearby body, head and bridge
/// triangles. Pairs sharing a (welded) vertex are not crossings.
pub(crate) fn crossing_pairs(bridge: &[[V; 3]], others: &[[V; 3]], cell: f64) -> Vec<(usize, usize)> {
    let mut all: Vec<[V; 3]> = bridge.to_vec();
    all.extend_from_slice(others);
    let mut ids: HashMap<[i64; 3], usize> = HashMap::new();
    let key = |p: &V| [p.x, p.y, p.z].map(|x| (x / 1e-7).round() as i64);
    let vid: Vec<[usize; 3]> = all
        .iter()
        .map(|t| {
            t.map(|p| {
                let n = ids.len();
                *ids.entry(key(&p)).or_insert(n)
            })
        })
        .collect();
    let cell_of = |p: &V| [p.x, p.y, p.z].map(|x| (x / cell).floor() as i64);
    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for (i, t) in all.iter().enumerate() {
        let (lo, hi) = (
            cell_of(&t[0].inf(&t[1]).inf(&t[2])),
            cell_of(&t[0].sup(&t[1]).sup(&t[2])),
        );
        for x in lo[0]..=hi[0] {
            for y in lo[1]..=hi[1] {
                for z in lo[2]..=hi[2] {
                    grid.entry([x, y, z]).or_default().push(i);
                }
            }
        }
    }
    (0..bridge.len())
        .into_par_iter()
        .map(|i| {
            let t = &all[i];
            let (lo, hi) = (
                cell_of(&t[0].inf(&t[1]).inf(&t[2])),
                cell_of(&t[0].sup(&t[1]).sup(&t[2])),
            );
            let mut candidates = vec![];
            for x in lo[0]..=hi[0] {
                for y in lo[1]..=hi[1] {
                    for z in lo[2]..=hi[2] {
                        if let Some(c) = grid.get(&[x, y, z]) {
                            candidates.extend(c.iter().copied());
                        }
                    }
                }
            }
            candidates.sort_unstable();
            candidates.dedup();
            candidates
                .into_iter()
                // Bridge–bridge pairs once.
                .filter(|&j| j != i && (j >= bridge.len() || j > i))
                .filter(|&j| !vid[i].iter().any(|x| vid[j].contains(x)))
                .filter(|&j| triangles_cross(t, &all[j]))
                .map(|j| (i, j))
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect()
}

/// Signed axial offset from a cut plane; values within 1 µm of the plane
/// count as below, so a split never creates a vertex on top of another.
pub(crate) fn plane_offset(frame: &Frame, p: V, cut: f64) -> f64 {
    let s = frame.axial(p) - cut;
    if s.abs() < 1e-6 { -1e-6 } else { s }
}

/// Positions, normals, UVs and triangles of a mesh.
pub(crate) type MeshView<'a> = (&'a [Vec3], &'a [Vec3], &'a [[f32; 2]], &'a [[u32; 3]]);

/// A mesh whose eligible triangles crossing the plane `axial = cut` are
/// split exactly at it (both sides kept, winding kept). Original vertices
/// keep their indices and welded ids; cut vertices are appended. A cut
/// vertex is placed from the welded representatives, so its UV duplicates
/// coincide exactly.
pub(crate) struct Split {
    pub(crate) positions: Vec<Vec3>,
    pub(crate) normals: Vec<Vec3>,
    pub(crate) uvs: Vec<[f32; 2]>,
    pub(crate) triangles: Vec<[u32; 3]>,
    /// Source triangle of every triangle.
    pub(crate) origin: Vec<usize>,
    pub(crate) weld: Weld,
    /// Source vertices and fraction of every appended vertex, from
    /// `first_new` on.
    pub(crate) ends: Vec<(usize, usize, f64)>,
    pub(crate) first_new: usize,
}

impl Split {
    pub(crate) fn new(
        mesh: MeshView,
        weld: &Weld,
        frame: &Frame,
        cut: f64,
        eligible: impl Fn(usize) -> bool,
    ) -> Self {
        let (positions, normals, uvs, triangles) = mesh;
        let rep = |w: usize| v(positions[weld.representative[w]]);
        let mut out = Self {
            positions: positions.to_vec(),
            normals: normals.to_vec(),
            uvs: uvs.to_vec(),
            triangles: Vec::with_capacity(triangles.len()),
            origin: Vec::with_capacity(triangles.len()),
            weld: Weld {
                id: weld.id.clone(),
                count: weld.count,
                representative: weld.representative.clone(),
            },
            ends: vec![],
            first_new: positions.len(),
        };
        let mut by_raw: HashMap<(u32, u32), u32> = HashMap::new();
        let mut by_weld: HashMap<(usize, usize), usize> = HashMap::new();
        let mut cut_vertex = |out: &mut Self, i: u32, j: u32| -> u32 {
            if let Some(&k) = by_raw.get(&(i.min(j), i.max(j))) {
                return k;
            }
            let (wi, wj) = (weld.id[i as usize], weld.id[j as usize]);
            let (r0, r1, w0, w1) = if wi < wj {
                (i as usize, j as usize, wi, wj)
            } else {
                (j as usize, i as usize, wj, wi)
            };
            let (s0, s1) = (
                plane_offset(frame, rep(w0), cut),
                plane_offset(frame, rep(w1), cut),
            );
            let f = s0 / (s0 - s1);
            let p = rep(w0) + (rep(w1) - rep(w0)) * f;
            let n = v(normals[r0]) * (1.0 - f) + v(normals[r1]) * f;
            let uv = [0, 1].map(|k| uvs[r0][k] + (uvs[r1][k] - uvs[r0][k]) * f as f32);
            let k = out.positions.len() as u32;
            out.positions.push(a(p));
            out.normals
                .push(a(n.try_normalize(1e-12).unwrap_or(frame.axis)));
            out.uvs.push(uv);
            out.ends.push((r0, r1, f));
            let id = *by_weld.entry((w0, w1)).or_insert_with(|| {
                out.weld.representative.push(k as usize);
                out.weld.count += 1;
                out.weld.count - 1
            });
            out.weld.id.push(id);
            by_raw.insert((i.min(j), i.max(j)), k);
            k
        };
        for (t, tri) in triangles.iter().enumerate() {
            let up = tri.map(|i| plane_offset(frame, rep(weld.id[i as usize]), cut) > 0.0);
            let alone = (0..3).find(|&k| up[k] != up[(k + 1) % 3] && up[k] != up[(k + 2) % 3]);
            let Some(k) = alone.filter(|_| eligible(t)) else {
                out.triangles.push(*tri);
                out.origin.push(t);
                continue;
            };
            let (p, q, r) = (tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]);
            let pq = cut_vertex(&mut out, p, q);
            let pr = cut_vertex(&mut out, p, r);
            for piece in [[p, pq, pr], [pq, q, r], [pq, r, pr]] {
                out.triangles.push(piece);
                out.origin.push(t);
            }
        }
        out
    }
}

/// Scan colour (sRGB) at a vertex of a triangle of the given material.
pub(crate) fn colour_at_uv(scan: &Scan, material: Option<usize>, uv: [f32; 2]) -> Option<[f64; 3]> {
    let m = scan.materials.get(material?)?;
    let c = match m.base_color {
        Some(i) => source::sample(&scan.images.get(i)?.decoded, [uv[0] as f64, uv[1] as f64]),
        None => [1.0; 3],
    };
    Some(std::array::from_fn(|k| c[k] * m.base_color_factor[k]))
}

pub fn keep_scan_head(inp: &Input, garments: &mut [Garment]) -> Result<KeptHead> {
    let Input {
        prior,
        rig,
        scan,
        seg,
        height,
    } = *inp;
    let joint_pos = |name: &str| -> Result<V> {
        let j = *prior
            .by_name
            .get(name)
            .with_context(|| format!("prior has no {name} joint"))?;
        Ok(rig.worlds[j].fixed_view::<3, 1>(0, 3).into_owned())
    };
    let (n1, n2, hd) = (
        joint_pos("neck01")?,
        joint_pos("neck02")?,
        joint_pos("head")?,
    );
    let head_joint = prior.joint("head");
    let axis = (hd - n1)
        .try_normalize(1e-9)
        .context("degenerate neck chain")?;
    let front = V::z() - axis * axis.z;
    let e1 = front.try_normalize(1e-9).unwrap_or(V::x());
    let e2 = axis.cross(&e1);
    let frame = Frame {
        origin: n1,
        axis,
        e1,
        e2,
    };
    // Face bones fold into `head`: they stay in the skeleton, drive nothing.
    let joint_map: Vec<usize> = (0..prior.skeleton().joints.len())
        .map(|j| {
            if j != head_joint && prior.is_ancestor(head_joint, j) {
                head_joint
            } else {
                j
            }
        })
        .collect();

    let body = prior.body();
    let body_pos = &rig.meshes[prior.body];
    let body_weld = Weld::new(&body.bind, 1e-7);
    let tris = &body.prim.triangles;
    // Head region: MakeHuman's fitted head (dominant joint in the head
    // region), widened. Limits both cuts to a cylinder about the neck axis.
    let head_radius = (0..body_pos.len())
        .filter(|&i| prior.region[prior.dominant_joint(prior.body, i)] == Some(Region::Head))
        .map(|i| frame.radial(v(body_pos[i])).norm())
        .fold(0.0, f64::max)
        .max(height * 0.05);
    let cylinder = head_radius * 1.6;
    // Lowest point of the fitted jaw (chin underside): the scan cut stays
    // below it, so the whole chin is the scan's.
    let jaw_min = prior.by_name.get("jaw").map(|&jaw| {
        (0..body_pos.len())
            .filter(|&i| prior.is_ancestor(jaw, prior.dominant_joint(prior.body, i)))
            .map(|i| frame.axial(v(body_pos[i])))
            .fold(f64::INFINITY, f64::min)
    });
     
    let mut cuts = None;
    for k in [1.0, 0.5, 0.25] {
        let gap = height * 0.012 * k;
        let margin = height * 0.008 * k;
        let mut body_cut = frame.axial(n2);
        let mut scan_cut = body_cut + gap;
        if let Some(j) = jaw_min.filter(|j| j.is_finite())
            && scan_cut > j - margin
        {
            scan_cut = j - margin;
            // Not below neck01 (origin of the frame).
            body_cut = (scan_cut - gap).max(height * 0.004 * k);
        }
        if scan_cut - body_cut > height * 0.004 * k {
            cuts = Some((body_cut, scan_cut));
            break;
        }
    }
    let Some((body_cut, scan_cut)) = cuts else {
        anyhow::bail!(
            "head_keep_no_room: jaw too close to the neck base for a seam (jaw {:.1} mm above neck01, neck plane {:.1} mm, height {:.2} m)",
            jaw_min.unwrap_or(f64::NAN) * 1000.0,
            frame.axial(n2) * 1000.0,
            height
        );
    };

    // ---- Body side: remove MakeHuman's head above the body plane ----
    // Triangles inside the cylinder are split exactly at the plane, so the
    // ring is planar and its texture unstretched.
    let full_normals = vertex_normals(body_pos, tris, &body_weld);
    let bs = Split::new(
        (body_pos, &full_normals, &body.prim.uvs, tris),
        &body_weld,
        &frame,
        body_cut,
        |t| {
            tris[t]
                .iter()
                .all(|&i| frame.radial(v(body_pos[i as usize])).norm() < cylinder)
        },
    );
    let identity: Vec<usize> = (0..prior.skeleton().joints.len()).collect();
    let body_infl: Vec<Vec<Influence>> = body
        .influences
        .iter()
        .cloned()
        .chain(bs.ends.iter().map(|&(i, j, f)| {
            mix(
                &[(&body.influences[i], 1.0 - f), (&body.influences[j], f)],
                &identity,
                4,
            )
        }))
        .collect();
    let (body_weld, btris) = (&bs.weld, &bs.triangles);
    let removed_vertex: Vec<bool> = (0..body_weld.count)
        .map(|w| {
            let i = body_weld.representative[w];
            let p = v(bs.positions[i]);
            i < bs.first_new
                && plane_offset(&frame, p, body_cut) > 0.0
                && frame.radial(p).norm() < cylinder
        })
        .collect();
    let mut body_keep: Vec<bool> = btris
        .iter()
        .map(|t| !t.iter().any(|&i| removed_vertex[body_weld.id[i as usize]]))
        .collect();
    let body_eroded = erode_bowties(btris, &mut body_keep, body_weld);
    keep_large_components(btris, &mut body_keep, body_weld, 0.05);
    let body_loops = boundary_loops(btris, &body_keep, body_weld);
    let body_loop_count = body_loops.len();
    let bpoint = |w: usize| v(bs.positions[body_weld.representative[w]]);
     
    let (body_ring_ids, recentre) = match neck_loop(&frame, body_loops.clone(), bpoint, body_cut) {
        Some(ids) => (ids, None),
        None => {
            let (ids, c) = own_centre_loop(&frame, body_loops, bpoint, body_cut, cylinder)
                .context("head_keep_no_body_ring: no closed MakeHuman neck boundary")?;
            (ids, Some(c))
        }
    };
    let frame = match recentre {
        Some(c) => Frame {
            origin: c - frame.axis * (c - frame.origin).dot(&frame.axis),
            ..frame
        },
        None => frame,
    };
    let body_ring_set: HashSet<usize> = body_ring_ids.iter().copied().collect();
    // Level what remains off the plane (ring vertices of triangles outside
    // the cylinder, bow-tie erosion) onto it, up along the axis: steps
    // taller than a loft band fold the first bands.
    let mut body_lv: Vec<Vec3> = bs.positions.clone();
    let mut body_levelled: f64 = 0.0;
    for (i, p) in body_lv.iter_mut().enumerate() {
        if body_ring_set.contains(&body_weld.id[i]) {
            let d = body_cut - frame.axial(v(*p));
            if d > 0.0 && d < height * 0.02 {
                *p = a(v(*p) + frame.axis * d);
                body_levelled = body_levelled.max(d);
            }
        }
    }
    let bpoint = |w: usize| v(body_lv[body_weld.representative[w]]);

    // Kept body, compacted.
    let mut new_index = vec![u32::MAX; bs.positions.len()];
    let mut kept = KeptHead {
        body_positions: vec![],
        body_normals: vec![],
        body_uvs: vec![],
        body_triangles: vec![],
        body_influences: vec![],
        body_triangle_origin: vec![],
        body_near_seam: vec![],
        heads: vec![],
        bridges: vec![],
        head: true,
        report: Value::Null,
    };
    let mut bridge = Bridge {
            name: "neck_bridge".into(),
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            triangles: vec![],
            influences: vec![],
            png: vec![],
            bottom: vec![],
            top: vec![],
    };
    let mut kept_split_triangles = vec![];
    for (t, tri) in btris.iter().enumerate() {
        if !body_keep[t] {
            continue;
        }
        let nt = tri.map(|i| {
            let i = i as usize;
            if new_index[i] == u32::MAX {
                new_index[i] = kept.body_positions.len() as u32;
                kept.body_positions.push(body_lv[i]);
                kept.body_normals.push(bs.normals[i]);
                kept.body_uvs.push(bs.uvs[i]);
                kept.body_influences.push(body_infl[i].clone());
            }
            new_index[i]
        });
        kept.body_triangles.push(nt);
        kept.body_triangle_origin.push(bs.origin[t]);
        kept.body_near_seam.push(
            tri.iter()
                .any(|&i| body_ring_set.contains(&body_weld.id[i as usize])),
        );
        kept_split_triangles.push(t);
    }

    // ---- Scan side: the scan head above the scan plane ----
    // Skin triangles inside the cylinder are split exactly at the plane
    // (hair and garments stay whole: they are exported as they are).
    let ss = Split::new(
        (&scan.positions, &scan.normals, &scan.uvs, &scan.triangles),
        &scan.weld,
        &frame,
        scan_cut,
        |t| {
            seg.triangle_label[t] == Label::Skin
                && scan.triangles[t]
                    .iter()
                    .all(|&i| frame.radial(v(scan.positions[i as usize])).norm() < cylinder)
        },
    );
    let slabel: Vec<Label> = ss.origin.iter().map(|&t| seg.triangle_label[t]).collect();
    let smaterial: Vec<Option<usize>> = ss
        .origin
        .iter()
        .map(|&t| scan.triangle_material[t])
        .collect();
    let sweld = &ss.weld;
    let above: Vec<bool> = (0..sweld.count)
        .map(|w| {
            let i = sweld.representative[w];
            let p = v(ss.positions[i]);
            (i >= ss.first_new || plane_offset(&frame, p, scan_cut) > 0.0)
                && frame.radial(p).norm() < cylinder
        })
        .collect();
    let all_above: Vec<bool> = ss
        .triangles
        .iter()
        .map(|t| t.iter().all(|&i| above[sweld.id[i as usize]]))
        .collect();
    // Skin alone first; when hair or a garment interrupts the neck ring,
    // skin and hair together (the hair keeps its own piece and mesh).
    let attempt = |with_hair: bool| {
        let mut keep: Vec<bool> = all_above
            .iter()
            .zip(&slabel)
            .map(|(a0, l)| *a0 && (*l == Label::Skin || (with_hair && *l == Label::Hair)))
            .collect();
        let islands = keep_large_components(&ss.triangles, &mut keep, sweld, 0.02);
        let eroded = erode_bowties(&ss.triangles, &mut keep, sweld);
        let islands2 = keep_large_components(&ss.triangles, &mut keep, sweld, 0.02);
        let loops = boundary_loops(&ss.triangles, &keep, sweld);
        let spoint = |w: usize| v(ss.positions[sweld.representative[w]]);
        neck_loop(&frame, loops, spoint, scan_cut)
            .map(|ring| (keep, ring, islands + islands2, eroded))
    };
    let (ring_mode, (mut region, scan_ring_ids, islands, scan_eroded)) = match attempt(false) {
        Some(r) => ("skin", r),
        None => (
            "skin_and_hair",
            attempt(true).context("head_keep_no_scan_ring: no closed scan neck boundary")?,
        ),
    };
    // Head mesh: the skin triangles of the region.
    let ring_region = region.clone();
    for (k, l) in region.iter_mut().zip(&slabel) {
        *k &= *l == Label::Skin;
    }
    let head_triangles: Vec<usize> = (0..ss.triangles.len()).filter(|&t| region[t]).collect();
    ensure!(
        head_triangles.len() > 100,
        "head_keep_empty_head: only {} scan skin triangles above the cut",
        head_triangles.len()
    );
    let scan_ring_set: HashSet<usize> = scan_ring_ids.iter().copied().collect();
    // Level what remains off the plane (hair edges, garment-free ring
    // vertices of whole triangles) onto it likewise (down), except vertices
    // shared with hair or garments (junctions stay exact).
    let garment_ids: HashSet<usize> = garments
        .iter()
        .flat_map(|g| g.weld_id.iter().copied())
        .collect();
    let mut spos: Vec<Vec3> = ss.positions.clone();
    let mut scan_levelled: f64 = 0.0;
    for (i, p) in spos.iter_mut().enumerate() {
        let w = sweld.id[i];
        if scan_ring_set.contains(&w) && !garment_ids.contains(&w) {
            let d = frame.axial(v(*p)) - scan_cut;
            if d > 0.0 && d < height * 0.01 {
                *p = a(v(*p) - frame.axis * d);
                scan_levelled = scan_levelled.max(d);
            }
        }
    }

    // ---- Weights: MakeHuman's at the seam, rigid head above ----
    let body_query = build_query(body_pos, tris);
    // (closest points on the whole fitted body, original triangles)
    let feather = height * 0.025;
    let mut weld_ids: Vec<usize> = head_triangles
        .iter()
        .flat_map(|&t| ss.triangles[t].map(|i| sweld.id[i as usize]))
        .chain(scan_ring_ids.iter().copied())
        .collect();
    weld_ids.sort_unstable();
    weld_ids.dedup();
    let rows: Vec<Vec<Influence>> = weld_ids
        .par_iter()
        .map(|&w| {
            let p = spos[sweld.representative[w]];
            let f = smoothstep((frame.axial(v(p)) - scan_cut) / feather);
            let head_row = [Influence {
                joint: head_joint,
                weight: 1.0,
            }];
            let Some(hit) = body_query.closest_point(p, height * 0.3, None) else {
                return head_row.to_vec();
            };
            let tri = tris[hit.source[1]];
            let mut parts: Vec<(&[Influence], f64)> = (0..3)
                .map(|k| {
                    (
                        body.influences[tri[k] as usize].as_slice(),
                        hit.barycentric[k] * (1.0 - f),
                    )
                })
                .collect();
            parts.push((&head_row, f));
            mix(&parts, &joint_map, 4)
        })
        .collect();
    let weight_of: HashMap<usize, usize> =
        weld_ids.iter().enumerate().map(|(k, &w)| (w, k)).collect();

    // ---- Seam normals: averaged over both sides of each ring ----
    // (Filled after the bridge exists; start with one-sided sums.)
    let mut body_seam_normal: HashMap<usize, V> = HashMap::new();
    for t in kept_split_triangles.iter().map(|&t| btris[t]) {
        let n = crate::geom::face_normal(&body_lv, &t);
        for &i in &t {
            let w = body_weld.id[i as usize];
            if body_ring_set.contains(&w) {
                *body_seam_normal.entry(w).or_insert(V::zeros()) += n;
            }
        }
    }
    let mut scan_seam_normal: HashMap<usize, V> = HashMap::new();
    for t in (0..ss.triangles.len()).filter(|&t| ring_region[t]) {
        let tri = ss.triangles[t];
        let n = crate::geom::face_normal(&spos, &tri);
        for &i in &tri {
            let w = sweld.id[i as usize];
            if scan_ring_set.contains(&w) {
                *scan_seam_normal.entry(w).or_insert(V::zeros()) += n;
            }
        }
    }

    // ---- Loft ----
    let bring = Ring::new(&frame, body_ring_ids, bpoint, |w| {
        v(bs.normals[body_weld.representative[w]])
    });
    let sring = Ring::new(
        &frame,
        scan_ring_ids,
        |w| v(spos[sweld.representative[w]]),
        |w| v(ss.normals[sweld.representative[w]]),
    );
    let body_row = |k: usize| body_infl[body_weld.representative[bring.ids[k]]].clone();
    let scan_row = |k: usize| rows[weight_of[&sring.ids[k]]].clone();
    let rings = 8usize;
    let samples = 128usize;
    let br = &mut bridge;
    let push = |br: &mut Bridge, p: V, uv: [f64; 2], infl: Vec<Influence>| -> u32 {
        br.positions.push(a(p));
        br.uvs.push([uv[0] as f32, uv[1] as f32]);
        br.influences.push(infl);
        (br.positions.len() - 1) as u32
    };
    let mut row_index: Vec<Vec<u32>> = vec![];
    let mut row_params: Vec<Vec<f64>> = vec![];
    // Bottom row: the body ring (first vertex repeated at u = 1).
    {
        let n = bring.len();
        let mut idx = vec![];
        for k in 0..=n {
            let infl = body_row(k % n);
            idx.push(push(br, bring.points[k % n], [bring.params[k], 0.0], infl));
            if k < n {
                br.bottom.push((
                    bring.params[k],
                    bs.uvs[body_weld.representative[bring.ids[k]]],
                ));
            }
        }
        row_index.push(idx);
        row_params.push(bring.params.clone());
    }
    let blend_rows = |ring: &Ring, row: &dyn Fn(usize) -> Vec<Influence>, u: f64| {
        let (i, j, f) = ring.locate(u);
        let (ri, rj) = (row(i), row(j));
        mix(&[(&ri, 1.0 - f), (&rj, f)], &joint_map, 8)
    };
    let ends: Vec<(V, V, V, V)> = (0..samples)
        .map(|j| {
            let u = j as f64 / samples as f64;
            let (lo, ln) = bring.sample(u);
            let (hi, hn) = sring.sample(u);
            let delta = hi - lo;
            (
                lo,
                end_tangent(&frame, ln, delta),
                hi,
                end_tangent(&frame, hn, delta),
            )
        })
        .collect();
    for r in 1..rings {
        let t = r as f64 / rings as f64;
        let mut pts: Vec<V> = ends
            .iter()
            .map(|(lo, t0, hi, t1)| hermite(*lo, *t0, *hi, *t1, t))
            .collect();
        // Fair the interior rings around the neck (the ends are jagged
        // mesh boundaries); strongest mid-loft.
        let strength = (std::f64::consts::PI * t).sin();
        for _ in 0..2 {
            let prev = pts.clone();
            for j in 0..samples {
                let mut acc = V::zeros();
                for d in -3i64..=3 {
                    acc += prev[(j as i64 + d).rem_euclid(samples as i64) as usize];
                }
                pts[j] = prev[j] * (1.0 - strength) + acc / 7.0 * strength;
            }
        }
        let s = smoothstep(t);
        let mut idx = vec![];
        let mut params = vec![];
        for j in 0..=samples {
            let u = j as f64 / samples as f64;
            let lw = blend_rows(&bring, &body_row, u);
            let hw = blend_rows(&sring, &scan_row, u);
            let infl = mix(&[(&lw, 1.0 - s), (&hw, s)], &joint_map, 4);
            idx.push(push(br, pts[j % samples], [u, t], infl));
            params.push(u);
        }
        row_index.push(idx);
        row_params.push(params);
    }
    {
        let n = sring.len();
        let mut idx = vec![];
        for k in 0..=n {
            let infl = scan_row(k % n);
            idx.push(push(br, sring.points[k % n], [sring.params[k], 1.0], infl));
            if k < n {
                br.top.push((sring.params[k], None));
            }
        }
        row_index.push(idx);
        row_params.push(sring.params.clone());
    }
    // Scan colours at the ring: the mean, over the region triangles at each
    // ring vertex (skin or hair), of the texture at that vertex. Where hair
    // closes the ring, the bridge fades into the hair colour and reads as
    // a nape hairline when it shows below the hair.
    {
        let mut colour: HashMap<usize, ([f64; 3], f64)> = HashMap::new();
        for (t, tri) in ss.triangles.iter().enumerate() {
            if !ring_region[t] {
                continue;
            }
            for &i in tri {
                let w = sweld.id[i as usize];
                if scan_ring_set.contains(&w)
                    && let Some(col) = colour_at_uv(scan, smaterial[t], ss.uvs[i as usize])
                {
                    let e = colour.entry(w).or_insert(([0.0; 3], 0.0));
                    for (sum, c) in e.0.iter_mut().zip(col) {
                        *sum += source::srgb_to_linear(c);
                    }
                    e.1 += 1.0;
                }
            }
        }
        for (k, entry) in br.top.iter_mut().enumerate() {
            entry.1 = colour
                .get(&sring.ids[k])
                .map(|(sum, n)| sum.map(|c| source::linear_to_srgb(c / n)));
        }
    }
    // Loft band (0 = at the body ring) of every bridge triangle.
    let mut band = vec![];
    for r in 0..rings {
        zipper(
            &row_index[r],
            &row_params[r],
            &row_index[r + 1],
            &row_params[r + 1],
            &mut br.triangles,
        );
        band.resize(br.triangles.len(), r);
    }
    // Outward orientation (by construction; verified).
    let inward = br
        .triangles
        .iter()
        .filter(|t| {
            let p = t.map(|i| v(br.positions[i as usize]));
            let n = (p[1] - p[0]).cross(&(p[2] - p[0]));
            n.dot(&frame.radial((p[0] + p[1] + p[2]) / 3.0)) < 0.0
        })
        .count();
    if inward * 2 > br.triangles.len() {
        for t in &mut br.triangles {
            t.swap(1, 2);
        }
    }
    let degenerate = br
        .triangles
        .iter()
        .filter(|t| crate::geom::face_normal(&br.positions, t).norm() < 1e-14)
        .count();
    // Seam normals: both sides of each ring.
    let bottom_n = bring.len() + 1;
    let top_start = br.positions.len() - (sring.len() + 1);
    for t in &br.triangles {
        let n = crate::geom::face_normal(&br.positions, t);
        for &i in t {
            let i = i as usize;
            if i < bottom_n {
                *body_seam_normal
                    .entry(bring.ids[i % bring.len()])
                    .or_insert(V::zeros()) += n;
            } else if i >= top_start {
                *scan_seam_normal
                    .entry(sring.ids[(i - top_start) % sring.len()])
                    .or_insert(V::zeros()) += n;
            }
        }
    }
    let bw = Weld::new(&br.positions, 1e-9);
    br.normals = vertex_normals(&br.positions, &br.triangles, &bw);
    let unit = |n: V| a(n.try_normalize(1e-20).unwrap_or(frame.axis));
    for k in 0..bottom_n {
        br.normals[k] = unit(body_seam_normal[&bring.ids[k % bring.len()]]);
    }
    for k in top_start..br.positions.len() {
        br.normals[k] = unit(scan_seam_normal[&sring.ids[(k - top_start) % sring.len()]]);
    }
    // Body ring vertices (all UV duplicates) take the two-sided normal.
    for (old, &ni) in new_index.iter().enumerate() {
        if ni != u32::MAX
            && let Some(n) = body_seam_normal.get(&body_weld.id[old])
        {
            kept.body_normals[ni as usize] = unit(*n);
        }
    }

    // ---- Head parts, one per scan material ----
    let mut groups: BTreeMap<Option<usize>, Vec<usize>> = BTreeMap::new();
    for &t in &head_triangles {
        groups.entry(smaterial[t]).or_default().push(t);
    }
    let multi = groups.len() > 1;
    for (gi, (material, list)) in groups.into_iter().enumerate() {
        let mut remap: HashMap<u32, u32> = HashMap::new();
        let mut part = HeadPart {
            name: if multi {
                format!("scan_head_{gi}")
            } else {
                "scan_head".into()
            },
            material,
            positions: vec![],
            normals: vec![],
            uvs: vec![],
            triangles: vec![],
            influences: vec![],
        };
        for t in list {
            let tri = ss.triangles[t].map(|i| {
                *remap.entry(i).or_insert_with(|| {
                    let w = sweld.id[i as usize];
                    part.positions.push(spos[i as usize]);
                    part.normals.push(match scan_seam_normal.get(&w) {
                        Some(n) => unit(*n),
                        None => ss.normals[i as usize],
                    });
                    part.uvs.push(ss.uvs[i as usize]);
                    part.influences.push(rows[weight_of[&w]].clone());
                    (part.positions.len() - 1) as u32
                })
            });
            part.triangles.push(tri);
        }
        kept.heads.push(part);
    }

    // ---- Checks: closed seam, no crossings ----
    let mut all_pos: Vec<Vec3> = kept.body_positions.clone();
    let mut all_tri: Vec<[u32; 3]> = kept.body_triangles.clone();
    fn append(all: &mut (Vec<Vec3>, Vec<[u32; 3]>), p: &[Vec3], t: &[[u32; 3]]) {
        let o = all.0.len() as u32;
        all.0.extend_from_slice(p);
        all.1.extend(t.iter().map(|x| x.map(|i| i + o)));
    }
    let mut all = (std::mem::take(&mut all_pos), std::mem::take(&mut all_tri));
    for h in &kept.heads {
        append(&mut all, &h.positions, &h.triangles);
    }
    // Hair triangles of the ring region (ring mode skin_and_hair). Garments
    // meeting the ring from outside the region (a collar) form a T-junction
    // there and are counted separately.
    let hair_region: Vec<[u32; 3]> = (0..ss.triangles.len())
        .filter(|&t| {
            ring_region[t]
                && !region[t]
                && ss.triangles[t]
                    .iter()
                    .any(|&i| scan_ring_set.contains(&sweld.id[i as usize]))
        })
        .map(|t| ss.triangles[t])
        .collect();
    // What lies below each scan ring edge outside the region.
    let ring_edges: HashSet<(usize, usize)> = (0..sring.len())
        .map(|k| {
            let (x, y) = (sring.ids[k], sring.ids[(k + 1) % sring.len()]);
            (x.min(y), x.max(y))
        })
        .collect();
    let mut below: HashMap<(usize, usize), Label> = HashMap::new();
    for (t, tri) in ss.triangles.iter().enumerate() {
        if ring_region[t] {
            continue;
        }
        let w = tri.map(|i| sweld.id[i as usize]);
        for (x, y) in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
            let e = (x.min(y), x.max(y));
            if ring_edges.contains(&e) {
                let l = slabel[t];
                // A garment junction wins over cut skin on the same edge.
                below
                    .entry(e)
                    .and_modify(|x| {
                        if l == Label::Garment {
                            *x = l
                        }
                    })
                    .or_insert(l);
            }
        }
    }
    let count_below = |l: Option<Label>| {
        ring_edges
            .iter()
            .filter(|e| below.get(e).copied() == l)
            .count()
    };
    let composition = json!({
        "cut_skin": count_below(Some(Label::Skin)),
        "hair": count_below(Some(Label::Hair)),
        "garment_junction": count_below(Some(Label::Garment)),
        "scan_open": count_below(None),
    });
    append(&mut all, &spos, &hair_region);
    append(&mut all, &br.positions, &br.triangles);
    let (all_pos, all_tri) = all;
    let all_weld = Weld::new(&all_pos, 1e-7);
    let mut edge_faces: HashMap<(usize, usize), u32> = HashMap::new();
    for t in &all_tri {
        let w = t.map(|i| all_weld.id[i as usize]);
        for (x, y) in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
            if x != y {
                *edge_faces.entry((x.min(y), x.max(y))).or_default() += 1;
            }
        }
    }
    let wid = |p: V| -> Option<usize> {
        // Look up the welded id of a ring point through the bridge copy.
        let k = [p.x, p.y, p.z];
        br.positions
            .iter()
            .position(|q| *q == k)
            .map(|i| all_weld.id[all_pos.len() - br.positions.len() + i])
    };
    let mut open = 0usize;
    let mut nonmanifold = 0usize;
    for ring in [&bring, &sring] {
        let ids: Vec<Option<usize>> = ring.points.iter().map(|p| wid(*p)).collect();
        for k in 0..ring.len() {
            let (Some(x), Some(y)) = (ids[k], ids[(k + 1) % ring.len()]) else {
                open += 1;
                continue;
            };
            match edge_faces.get(&(x.min(y), x.max(y))).copied().unwrap_or(0) {
                2 => {}
                0 | 1 => open += 1,
                _ => nonmanifold += 1,
            }
        }
    }
    let bridge_tris: Vec<[V; 3]> = br
        .triangles
        .iter()
        .map(|t| t.map(|i| v(br.positions[i as usize])))
        .collect();
    let (lo, hi) = bridge_tris.iter().flatten().fold(
        (V::repeat(f64::INFINITY), V::repeat(f64::NEG_INFINITY)),
        |(l, h), p| (l.inf(p), h.sup(p)),
    );
    let pad = V::repeat(height * 0.01);
    let (lo, hi) = (lo - pad, hi + pad);
    let inside = |t: &[V; 3]| {
        t.iter()
            .any(|p| (0..3).all(|k| p[k] >= lo[k] && p[k] <= hi[k]))
    };
    let mut others: Vec<[V; 3]> = kept
        .body_triangles
        .iter()
        .map(|t| t.map(|i| v(kept.body_positions[i as usize])))
        .filter(&inside)
        .collect();
    let body_others = others.len();
    for h in &kept.heads {
        others.extend(
            h.triangles
                .iter()
                .map(|t| t.map(|i| v(h.positions[i as usize])))
                .filter(&inside),
        );
    }
    let pairs = crossing_pairs(&bridge_tris, &others, height * 0.01);
    let crossings = pairs.len();
    let nb = bridge_tris.len();
    let mut crossing_detail =
        json!({"bridge_bridge": 0, "bridge_body": 0, "bridge_head": 0, "bands": []});
    let mut bands: Vec<usize> = vec![];
    for &(i, j) in &pairs {
        let kind = if j < nb {
            "bridge_bridge"
        } else if j - nb < body_others {
            "bridge_body"
        } else {
            "bridge_head"
        };
        crossing_detail[kind] = json!(crossing_detail[kind].as_u64().unwrap_or(0) + 1);
        bands.push(band[i]);
        if j < nb {
            bands.push(band[j]);
        }
    }
    bands.sort_unstable();
    crossing_detail["bands"] = json!(bands);

    // ---- Garments that share scan vertices with the head ----
    // Hair and collars meeting the head take its weights (and the scan's
    // exact position, undoing any clearance push), so the junction stays
    // closed; face-bone weights elsewhere fold into `head`.
    let mut reweighted = 0usize;
    let mut restored = 0usize;
    let mut folded = 0usize;
    for g in garments.iter_mut() {
        for i in 0..g.positions.len() {
            let w = g.weld_id[i];
            if let Some(&k) = weight_of.get(&w) {
                g.influences[i] = rows[k].clone();
                let original = spos[sweld.representative[w]];
                if g.positions[i] != original {
                    g.positions[i] = original;
                    restored += 1;
                }
                reweighted += 1;
            } else if g.influences[i]
                .iter()
                .any(|x| joint_map[x.joint] != x.joint)
            {
                g.influences[i] = mix(&[(&g.influences[i], 1.0)], &joint_map, 4);
                folded += 1;
            }
        }
    }

    let removed_parts: Vec<String> = prior
        .meshes
        .iter()
        .filter(|m| removes_part(m.kind))
        .map(|m| format!("{:?}", m.kind).to_lowercase())
        .collect();
    let removed_body = tris.len() - kept.body_triangles.len();
    let head_vertices: usize = kept.heads.iter().map(|h| h.positions.len()).sum();
    kept.report = json!({
        "route": "keep_scan",
        "cut_plane": {
            "axis": a(frame.axis),
            "origin_neck01": a(frame.origin),
            "body_offset_m": body_cut,
            "scan_offset_m": scan_cut,
            "gap_m": scan_cut - body_cut,
            "neck02_offset_m": frame.axial(n2),
            "head_offset_m": frame.axial(hd),
            "jaw_min_offset_m": jaw_min,
            "cylinder_radius_m": cylinder,
            "body_point": a(frame.origin + frame.axis * body_cut),
            "scan_point": a(frame.origin + frame.axis * scan_cut),
        },
        "seam_vertices": {"body": bring.len(), "scan": sring.len()},
        "ring_levelling_max_m": {"body": body_levelled, "scan": scan_levelled},
        "ring_radius_m": {"body": bring.mean_radius(&frame), "scan": sring.mean_radius(&frame)},
        "scan_ring_mode": ring_mode,
        "scan_ring_edges_below": composition,
        "bridge_rings": rings + 1,
        "bridge_ring_samples": samples,
        "bridge_triangles": br.triangles.len(),
        "bridge_degenerate_triangles": degenerate,
        "open_seam_edges": open,
        "nonmanifold_seam_edges": nonmanifold,
        "bridge_crossing_pairs": crossings,
        "bridge_crossings": crossing_detail,
        "crossing_scope": "bridge vs itself and the kept body and head triangles within 1 % of the height",
        "body": {
            "removed_triangles": removed_body,
            "bowtie_eroded_triangles": body_eroded,
            "boundary_loops": body_loop_count,
            "removed_parts": removed_parts,
        },
        "head": {
            "parts": kept.heads.len(),
            "triangles": head_triangles.len(),
            "vertices": head_vertices,
            "dropped_island_triangles": islands,
            "bowtie_eroded_triangles": scan_eroded,
            "feather_m": feather,
            "skinning": "MakeHuman weights (face bones folded into head) at the seam, smoothstep to rigid head over feather_m",
        },
        "garments": {
            "shared_vertices_reweighted": reweighted,
            "shared_vertices_position_restored": restored,
            "face_bone_weights_folded": folded,
        },
    });
    if open > 0 || nonmanifold > 0 {
        bail!("head_keep_open_seam: {open} open, {nonmanifold} non-manifold seam edges");
    }
    kept.bridges.push(bridge);
    Ok(kept)
}

impl KeptHead {
    /// Hidden flags for the kept body triangles; the seam ring stays visible.
    pub fn hidden(&self, full: &[bool]) -> Vec<bool> {
        self.body_triangle_origin
            .iter()
            .zip(&self.body_near_seam)
            .map(|(&t, near)| full[t] && !near)
            .collect()
    }

    /// Strip texture of the bridge: per ring column, the baked body colour at
    /// the bottom fading (smoothstep, linear light) into the scan's neck
    /// colour at the top. Columns are smoothed around the ring.
    pub fn bake_bridge(&mut self, baked: &image::RgbaImage) -> Result<()> {
        let mut reports = serde_json::Map::new();
        for bridge in &mut self.bridges {
            reports.insert(bridge.name.clone(), bake_one(bridge, baked)?);
        }
        if self.bridges.len() == 1 && self.head {
            self.report["bridge_texture"] = reports.values().next().cloned().unwrap_or(Value::Null);
        } else {
            self.report["bridge_textures"] = Value::Object(reports);
        }
        Ok(())
    }
}

/// One bridge's strip texture (see [`KeptHead::bake_bridge`]).
fn bake_one(bridge: &mut Bridge, baked: &image::RgbaImage) -> Result<Value> {{
        const W: usize = 512;
        const H: usize = 16;
        let lin = |c: [f64; 3]| c.map(source::srgb_to_linear);
        let column = |entries: &[(f64, Option<[f64; 3]>)]| -> Vec<Option<[f64; 3]>> {
            let mut acc = vec![([0.0; 3], 0.0); W];
            for (u, c) in entries {
                if let Some(c) = c {
                    // Columns 0 and W-1 are the same ring point (u = 0, 1).
                    let x = (u * (W - 1) as f64).round() as usize % (W - 1);
                    for (sum, l) in acc[x].0.iter_mut().zip(lin(*c)) {
                        *sum += l;
                    }
                    acc[x].1 += 1.0;
                }
            }
            // Circular Gaussian smoothing (~1.5 % of the ring, a few mm).
            let sigma = W as f64 * 0.015;
            let reach = (sigma * 3.0) as i64;
            (0..W)
                .map(|x| {
                    let (mut s, mut n) = ([0.0; 3], 0.0);
                    for d in -reach..=reach {
                        let j = (x as i64 + d).rem_euclid(W as i64 - 1) as usize;
                        let g = (-0.5 * (d as f64 / sigma).powi(2)).exp();
                        for (sum, c) in s.iter_mut().zip(acc[j].0) {
                            *sum += c * g;
                        }
                        n += acc[j].1 * g;
                    }
                    (n > 1e-9).then(|| s.map(|c| c / n))
                })
                .collect()
        };
        let bottom: Vec<(f64, Option<[f64; 3]>)> = bridge
            .bottom
            .iter()
            .map(|(u, uv)| {
                (
                    *u,
                    Some(source::sample(baked, [uv[0] as f64, uv[1] as f64])),
                )
            })
            .collect();
        let lo = column(&bottom);
        let hi = column(&bridge.top);
        let mean = |c: &[Option<[f64; 3]>]| {
            let xs: Vec<[f64; 3]> = c.iter().flatten().copied().collect();
            let n = xs.len().max(1) as f64;
            [0, 1, 2].map(|k| xs.iter().map(|x| x[k]).sum::<f64>() / n)
        };
        let (lo_mean, hi_mean) = (mean(&lo), mean(&hi));
        let mut img = image::RgbaImage::new(W as u32, H as u32);
        for y in 0..H {
            let t = smoothstep(y as f64 / (H - 1) as f64);
            for x in 0..W {
                let l = lo[x].unwrap_or(lo_mean);
                let h = hi[x].or(lo[x]).unwrap_or(hi_mean);
                let c = [0, 1, 2].map(|k| source::linear_to_srgb(l[k] * (1.0 - t) + h[k] * t));
                img.put_pixel(
                    x as u32,
                    y as u32,
                    image::Rgba([
                        (c[0] * 255.0).round() as u8,
                        (c[1] * 255.0).round() as u8,
                        (c[2] * 255.0).round() as u8,
                        255,
                    ]),
                );
            }
        }
        // Texel centres: u = 0 / 1 on the first / last column, t likewise.
        for uv in &mut bridge.uvs {
            uv[0] = (0.5 + uv[0] * (W - 1) as f32) / W as f32;
            uv[1] = (0.5 + uv[1] * (H - 1) as f32) / H as f32;
        }
        bridge.png.clear();
        image::DynamicImage::ImageRgba8(img).write_to(
            &mut std::io::Cursor::new(&mut bridge.png),
            image::ImageFormat::Png,
        )?;
        let srgb = |c: [f64; 3]| c.map(source::linear_to_srgb);
        Ok(json!({
            "size": [W, H],
            "body_ring_mean_srgb": srgb(lo_mean),
            "scan_ring_mean_srgb": srgb(hi_mean),
            "scan_ring_samples": bridge.top.iter().filter(|x| x.1.is_some()).count(),
        }))
    }
}

/// Largest distance, over the clip, between rest-coincident vertices of the
/// exported skin (body, scan head, bridge): the seam must not open.
pub fn seam_gap(
    rig: &RigView,
    clip: &MotionClip,
    frames: usize,
    positions: &[Vec3],
    influences: &[Vec<Influence>],
) -> Result<Value> {
    let weld = Weld::new(positions, 1e-7);
    let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &w) in weld.id.iter().enumerate() {
        members.entry(w).or_default().push(i);
    }
    let groups: Vec<Vec<usize>> = members.into_values().filter(|m| m.len() > 1).collect();
    let ibm: Vec<Matrix4<f64>> = rig
        .palette
        .iter()
        .map(|&j| {
            rig.rest_world[j]
                .try_inverse()
                .unwrap_or_else(Matrix4::identity)
        })
        .collect();
    let [start, end] = clip.time_range();
    let mut worst: f64 = 0.0;
    for f in 0..frames.max(2) {
        let time = start + (end - start) * f as f64 / (frames.max(2) - 1) as f64;
        let pose = clip.sample(rig.skeleton, time)?;
        let world = rig.skeleton.world_matrices(&pose)?;
        let palette: HashMap<usize, Matrix4<f64>> = rig
            .palette
            .iter()
            .zip(&ibm)
            .map(|(&j, b)| (j, world[j] * b))
            .collect();
        let skin = |i: usize| {
            let h = v(positions[i]).push(1.0);
            // As exported: the first four influences, renormalised.
            let row = &influences[i][..influences[i].len().min(4)];
            let sum: f64 = row.iter().map(|w| w.weight).sum();
            row.iter()
                .fold(nalgebra::Vector4::zeros(), |s, w| {
                    s + palette[&w.joint] * h * (w.weight / sum)
                })
                .xyz()
        };
        let gap = groups
            .par_iter()
            .map(|g| {
                let p0 = skin(g[0]);
                g[1..]
                    .iter()
                    .map(|&i| (skin(i) - p0).norm())
                    .fold(0.0, f64::max)
            })
            .reduce(|| 0.0, f64::max);
        worst = worst.max(gap);
    }
    Ok(json!({"clip":clip.name,"coincident_groups":groups.len(),"max_gap_m":worst}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame() -> Frame {
        Frame {
            origin: V::zeros(),
            axis: V::y(),
            e1: V::z(),
            e2: V::y().cross(&V::z()),
        }
    }

    /// Open cylinder (rings around +Y), quads split into triangles.
    fn cylinder(rings: usize, segments: usize, r: f64, dy: f64) -> (Vec<Vec3>, Vec<[u32; 3]>) {
        let mut p = vec![];
        for i in 0..rings {
            for j in 0..segments {
                let phi = std::f64::consts::TAU * j as f64 / segments as f64;
                p.push([r * phi.sin(), i as f64 * dy, r * phi.cos()]);
            }
        }
        let mut t = vec![];
        for i in 0..rings - 1 {
            for j in 0..segments {
                let a0 = (i * segments + j) as u32;
                let b0 = (i * segments + (j + 1) % segments) as u32;
                let c0 = a0 + segments as u32;
                let d0 = b0 + segments as u32;
                t.push([a0, b0, d0]);
                t.push([a0, d0, c0]);
            }
        }
        (p, t)
    }

    #[test]
    fn cut_cylinder_has_one_neck_ring_and_loft_closes_it() {
        let (p, t) = cylinder(10, 24, 0.05, 0.01);
        let weld = Weld::new(&p, 1e-9);
        let f = frame();
        let mut keep: Vec<bool> = t
            .iter()
            .map(|tri| tri.iter().all(|&i| p[i as usize][1] < 0.045))
            .collect();
        assert_eq!(erode_bowties(&t, &mut keep, &weld), 0);
        let loops = boundary_loops(&t, &keep, &weld);
        // Bottom rim and the cut.
        assert_eq!(loops.len(), 2);
        let ring = neck_loop(&f, loops, |w| v(p[weld.representative[w]]), 0.045).unwrap();
        assert!(
            ring.iter()
                .all(|&w| (p[weld.representative[w]][1] - 0.04).abs() < 1e-9)
        );
        let r = Ring::new(&f, ring, |w| v(p[weld.representative[w]]), |_| V::z());
        assert!(winding(&f, &r.points) > 0.9);
        assert!(r.params[0] == 0.0 && (r.params[r.len()] - 1.0).abs() < 1e-12);
        // Zipper between rows of different density covers both rows once.
        let low: Vec<u32> = (0..=4).collect();
        let high: Vec<u32> = (10..=16).collect();
        let lu: Vec<f64> = (0..=4).map(|k| k as f64 / 4.0).collect();
        let hu: Vec<f64> = (0..=6).map(|k| k as f64 / 6.0).collect();
        let mut out = vec![];
        zipper(&low, &lu, &high, &hu, &mut out);
        assert_eq!(out.len(), 4 + 6);
    }

    #[test]
    fn split_cuts_exactly_at_the_plane_and_stays_welded() {
        let (p, t) = cylinder(10, 24, 0.05, 0.01);
        let weld = Weld::new(&p, 1e-9);
        let f = frame();
        let n = vec![[0.0, 0.0, 1.0]; p.len()];
        let uv: Vec<[f32; 2]> = p.iter().map(|q| [q[0] as f32, q[1] as f32]).collect();
        let cut = 0.0437;
        let s = Split::new((&p, &n, &uv, &t), &weld, &f, cut, |_| true);
        // Each crossing triangle becomes three; one cut vertex per edge.
        assert_eq!(s.triangles.len(), t.len() + 2 * 2 * 24);
        // (24 vertical and 24 diagonal edges cross.)
        assert_eq!(s.positions.len() - s.first_new, 2 * 24);
        let keep: Vec<bool> = s
            .triangles
            .iter()
            .map(|tri| {
                tri.iter().all(|&i| {
                    i as usize >= s.first_new
                        || plane_offset(&f, v(s.positions[i as usize]), cut) < 0.0
                })
            })
            .collect();
        let loops = boundary_loops(&s.triangles, &keep, &s.weld);
        let ring = neck_loop(&f, loops, |w| v(s.positions[s.weld.representative[w]]), cut).unwrap();
        assert_eq!(ring.len(), 24 + 24);
        assert!(
            ring.iter()
                .all(|&w| (s.positions[s.weld.representative[w]][1] - cut).abs() < 1e-12)
        );
        // UVs follow the cut.
        for (k, &(i, j, x)) in s.ends.iter().enumerate() {
            let q = s.uvs[s.first_new + k];
            assert!((q[1] as f64 - cut).abs() < 1e-6, "{i} {j} {x}");
        }
    }

    #[test]
    fn hermite_tangent_never_overshoots_the_chord() {
        let f = frame();
        // Surface leaning outwards: lateral tangent clamped to 3× the chord.
        let n = V::new(0.0, -0.3, 1.0).normalize();
        let delta = V::new(0.0, 0.02, 0.001);
        let t = end_tangent(&f, n, delta);
        assert!((t.y - 0.02).abs() < 1e-12);
        assert!(t.z <= 0.003 + 1e-12 && t.z >= 0.0);
        // Opposite to the chord: dropped.
        let t = end_tangent(&f, V::new(0.0, 0.3, 1.0).normalize(), delta);
        assert!(t.z.abs() < 1e-12);
    }

    #[test]
    fn crossing_test_is_strict() {
        let a0 = [V::new(0., 0., 0.), V::new(1., 0., 0.), V::new(0., 1., 0.)];
        let b0 = [
            V::new(0.2, 0.2, -1.),
            V::new(0.2, 0.2, 1.),
            V::new(0.3, 0.3, 1.),
        ];
        assert!(triangles_cross(&a0, &b0));
        let c0 = [
            V::new(0., 0., 0.),
            V::new(-1., 0., 0.),
            V::new(0., -1., 0.5),
        ];
        assert!(!triangles_cross(&a0, &c0));
    }
}
