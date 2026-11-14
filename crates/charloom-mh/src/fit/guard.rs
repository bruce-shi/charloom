 use super::{carried, inverted};
use crate::geom::{V, a, face_normal};
use nalgebra::Matrix3;
use serde_json::{Value, json};
use std::collections::VecDeque;

/// Bend above MakeHuman's rest bend (radians) that counts as a fold. The
/// reported metric uses 0.8; fairing starts with a margin.
pub const FOLD_MARGIN: f64 = 0.7;

pub struct Guard<'a> {
    rest: &'a [V],
    triangles: &'a [[u32; 3]],
    rest_normals: &'a [V],
    neighbours: &'a [Vec<u32>],
    /// Watched edges `(i, j, t0, t1)`: interior edges whose fold is checked.
    edges: Vec<(u32, u32, usize, usize)>,
    rest_bend: Vec<f64>,
    radius: f64,
}

/// Dihedral bend (radians) between two triangles.
pub fn bend(pos: &[V], triangles: &[[u32; 3]], t0: usize, t1: usize) -> f64 {
    let normal = |t: usize| {
        let [p0, p1, p2] = triangles[t].map(|i| pos[i as usize]);
        (p1 - p0).cross(&(p2 - p0))
    };
    let (n0, n1) = (normal(t0), normal(t1));
    let d = n0.norm() * n1.norm();
    if d < 1e-20 {
        0.0
    } else {
        (n0.dot(&n1) / d).clamp(-1.0, 1.0).acos()
    }
}

/// Interior edges with both ends in `mask`, as `(i, j, t0, t1)`, sorted.
pub fn watched_edges(triangles: &[[u32; 3]], mask: &[bool]) -> Vec<(u32, u32, usize, usize)> {
    let mut edge_tris: std::collections::HashMap<(u32, u32), Vec<usize>> = Default::default();
    for (t, tri) in triangles.iter().enumerate() {
        for k in 0..3 {
            let (p, q) = (tri[k], tri[(k + 1) % 3]);
            edge_tris.entry((p.min(q), p.max(q))).or_default().push(t);
        }
    }
    let mut out: Vec<(u32, u32, usize, usize)> = edge_tris
        .into_iter()
        .filter(|(e, ts)| ts.len() == 2 && mask[e.0 as usize] && mask[e.1 as usize])
        .map(|(e, ts)| (e.0, e.1, ts[0], ts[1]))
        .collect();
    out.sort_unstable();
    out
}

pub fn smoothstep(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

 pub fn surface_distance(
    rest: &[V],
    neighbours: &[Vec<u32>],
    seeds: &[bool],
    radius: f64,
) -> Vec<f64> {
    let mut distance = vec![f64::INFINITY; rest.len()];
    let mut queue = VecDeque::new();
    for (i, s) in seeds.iter().enumerate() {
        if *s {
            distance[i] = 0.0;
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        for &j in &neighbours[i] {
            let j = j as usize;
            let next = distance[i] + (rest[i] - rest[j]).norm();
            if next < radius && next + 1e-12 < distance[j] {
                distance[j] = next;
                queue.push_back(j);
            }
        }
    }
    distance
}

pub struct Outcome {
    pub report: Value,
    /// Vertices the guard had to hold back (fully reverted).
    pub anchors: Vec<bool>,
}

impl<'a> Guard<'a> {
    pub fn new(
        rest: &'a [V],
        triangles: &'a [[u32; 3]],
        rest_normals: &'a [V],
        neighbours: &'a [Vec<u32>],
        fold_mask: &[bool],
        radius: f64,
    ) -> Self {
        let edges = watched_edges(triangles, fold_mask);
        let rest_bend = edges
            .iter()
            .map(|&(_, _, t0, t1)| bend(rest, triangles, t0, t1))
            .collect();
        Self {
            rest,
            triangles,
            rest_normals,
            neighbours,
            edges,
            rest_bend,
            radius,
        }
    }

    /// Per triangle: inverted (against the rest normal carried by the local
    /// frames) or collapsed (under 5 % of its rest area).
    fn bad_triangles(&self, x: &[V], frames: &[Matrix3<f64>]) -> Vec<bool> {
        let pos: Vec<charloom_core::Vec3> = x.iter().map(|q| a(*q)).collect();
        self.triangles
            .iter()
            .enumerate()
            .map(|(k, tri)| {
                let now = face_normal(&pos, tri);
                let rest = self.rest_normals[k];
                inverted(&now, &carried(frames, tri, &rest)) || now.norm() < rest.norm() * 0.05
            })
            .collect()
    }

    fn folded_edges(&self, x: &[V]) -> Vec<bool> {
        self.edges
            .iter()
            .zip(&self.rest_bend)
            .map(|(&(_, _, t0, t1), rb)| bend(x, self.triangles, t0, t1) - rb > FOLD_MARGIN)
            .collect()
    }

    /// Count of currently inverted/collapsed triangles and folded edges.
    pub fn defects(&self, x: &[V], frames: &[Matrix3<f64>]) -> (usize, usize) {
        (
            self.bad_triangles(x, frames).iter().filter(|b| **b).count(),
            self.folded_edges(x).iter().filter(|b| **b).count(),
        )
    }

    /// Vertices of the two triangles of every currently folded edge.
    pub fn fold_vertices(&self, x: &[V]) -> Vec<bool> {
        let mut out = vec![false; x.len()];
        for (k, f) in self.folded_edges(x).iter().enumerate() {
            if *f {
                let (_, _, t0, t1) = self.edges[k];
                for t in [t0, t1] {
                    self.triangles[t]
                        .iter()
                        .for_each(|&q| out[q as usize] = true);
                }
            }
        }
        out
    }

         pub fn limit(&self, before: &[V], x: &mut [V], frames: &[Matrix3<f64>]) -> Outcome {
        let n = x.len();
        let delta: Vec<V> = x.iter().zip(before).map(|(p, b)| p - b).collect();
        let moved: Vec<bool> = delta.iter().map(|d| d.norm() > 1e-10).collect();
        let bad_before = self.bad_triangles(before, frames);
        let mut anchors = vec![false; n];
        let mut history = vec![];
        let mut weight = vec![1.0f64; n];
        let mut converged = false;
        for _ in 0..24 {
            let bad = self.bad_triangles(x, frames);
            let mut new_bad = 0;
            for (k, tri) in self.triangles.iter().enumerate() {
                if bad[k] && !bad_before[k] {
                    new_bad += 1;
                    for &i in tri {
                        if moved[i as usize] {
                            anchors[i as usize] = true;
                        }
                    }
                }
            }
            history.push(new_bad);
            if new_bad == 0 {
                converged = true;
                break;
            }
            // Geodesic distance (on the rest surface) to the anchors.
            let distance = surface_distance(self.rest, self.neighbours, &anchors, self.radius);
            for i in 0..n {
                weight[i] = smoothstep(distance[i] / self.radius);
                x[i] = before[i] + delta[i] * weight[i];
            }
        }
        let held: Vec<f64> = weight.iter().filter(|w| **w < 1.0).copied().collect();
        Outcome {
            report: json!({
                "converged": converged,
                "anchors": anchors.iter().filter(|a| **a).count(),
                "held_back_vertices": held.len(),
                "new_bad_triangles_per_pass": history,
            }),
            anchors,
        }
    }
}
