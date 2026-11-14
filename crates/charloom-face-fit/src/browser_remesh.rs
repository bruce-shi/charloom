//! Browser port of standardize/skin_remesh, using the portable surface BVH.
//! Conservative local remeshing of observed skin. Collapses keep an original
//! endpoint, preserve semantic/UV/crease boundaries, and test the changed faces
//! against the original skin surface and UV interpolation. Hidden body vertices
//! and observed hair/garments are outside the optimization domain.
use super::browser_mesh::MeshData;
use crate::surface::{Surface, V};
use anyhow::{Result, ensure};
use schemars::JsonSchema;
type Vec3 = [f64; 3];
fn v3(v: Vec3) -> V {
    V::from(v)
}
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct SkinRemeshConfig {
    pub target_edge_m: f64,
    pub target_face_fraction: f64,
    pub maximum_sampled_surface_error_m: f64,
    pub maximum_uv_error: f64,
    pub feature_angle_degrees: f64,
    pub maximum_collapses: usize,
}
impl Default for SkinRemeshConfig {
    fn default() -> Self {
        Self {
            target_edge_m: 0.004,
            target_face_fraction: 0.65,
            maximum_sampled_surface_error_m: 0.0006,
            maximum_uv_error: 0.0015,
            feature_angle_degrees: 45.,
            maximum_collapses: 20000,
        }
    }
}
pub struct Remeshed {
    pub faces: Vec<[u32; 3]>,
    pub report: Value,
    pub face_sources: Vec<usize>,
}
#[derive(Clone, Copy)]
struct Edge {
    length: f64,
    a: usize,
    b: usize,
    version_a: u32,
    version_b: u32,
}
impl PartialEq for Edge {
    fn eq(&self, b: &Self) -> bool {
        self.cmp(b) == Ordering::Equal
    }
}
impl Eq for Edge {}
impl PartialOrd for Edge {
    fn partial_cmp(&self, b: &Self) -> Option<Ordering> {
        Some(self.cmp(b))
    }
}
impl Ord for Edge {
    fn cmp(&self, b: &Self) -> Ordering {
        b.length
            .total_cmp(&self.length)
            .then_with(|| b.a.cmp(&self.a))
            .then_with(|| b.b.cmp(&self.b))
    }
}

struct Domain<'a> {
    points: &'a [Vec3],
    uv: Option<&'a [[f32; 2]]>,
    faces: Vec<[u32; 3]>,
    alive: Vec<bool>,
    incidents: Vec<BTreeSet<usize>>,
    query: Surface,
    query_faces: Vec<[u32; 3]>,
    config: &'a SkinRemeshConfig,
}
impl Domain<'_> {
    fn neighbours(&self, v: usize) -> BTreeSet<usize> {
        self.incidents[v]
            .iter()
            .flat_map(|&f| self.faces[f].into_iter().map(|v| v as usize))
            .filter(|&j| j != v)
            .collect()
    }
    fn check(&self, keep: usize, remove: usize) -> Option<f64> {
        let shared: Vec<_> = self.incidents[keep]
            .intersection(&self.incidents[remove])
            .copied()
            .collect();
        if shared.len() != 2 {
            return None;
        }
        let common: BTreeSet<_> = self
            .neighbours(keep)
            .intersection(&self.neighbours(remove))
            .copied()
            .collect();
        let opposites: BTreeSet<_> = shared
            .iter()
            .flat_map(|&f| self.faces[f])
            .map(|v| v as usize)
            .filter(|&v| v != keep && v != remove)
            .collect();
        if common != opposites {
            return None;
        }
        let mut maximum = 0.0f64;
        for &index in &self.incidents[remove] {
            let old = self.faces[index];
            if old.contains(&(keep as u32)) {
                continue;
            }
            let next = old.map(|v| if v as usize == remove { keep as u32 } else { v });
            let [a, b, c] = next.map(|v| v3(self.points[v as usize]));
            let [x, y, z] = old.map(|v| v3(self.points[v as usize]));
            let normal = (b - a).cross(&(c - a));
            let old_normal = (y - x).cross(&(z - x));
            if normal.norm() < old_normal.norm() * 0.15
                || normal.dot(&old_normal) < normal.norm() * old_normal.norm() * 0.75
            {
                return None;
            }
            for weights in [[1. / 3.; 3], [0.5, 0.5, 0.], [0., 0.5, 0.5], [0.5, 0., 0.5]] {
                let p = a * weights[0] + b * weights[1] + c * weights[2];
                let hit = self.query.nearest_detail(
                    p,
                    self.config.maximum_sampled_surface_error_m * 1.00001,
                    Some(normal.normalize()),
                )?;
                maximum = maximum.max(hit.distance);
                if let Some(uv) = self.uv {
                    let reference = self.query_faces[hit.triangle];
                    let point = |v: u32| {
                        nalgebra::Vector2::new(
                            f64::from(uv[v as usize][0]),
                            f64::from(uv[v as usize][1]),
                        )
                    };
                    let expected = reference
                        .iter()
                        .zip(hit.barycentric)
                        .map(|(&v, w)| point(v) * w)
                        .sum::<nalgebra::Vector2<f64>>();
                    let actual = next
                        .iter()
                        .zip(weights)
                        .map(|(&v, w)| point(v) * w)
                        .sum::<nalgebra::Vector2<f64>>();
                    if (expected - actual).abs().max() > self.config.maximum_uv_error {
                        return None;
                    }
                }
            }
        }
        Some(maximum)
    }
    fn collapse(&mut self, keep: usize, remove: usize) -> usize {
        let incident: Vec<_> = self.incidents[remove].iter().copied().collect();
        let mut removed = 0;
        for i in incident {
            if self.faces[i].contains(&(keep as u32)) {
                self.alive[i] = false;
                removed += 1;
                for v in self.faces[i] {
                    self.incidents[v as usize].remove(&i);
                }
            } else {
                self.faces[i] =
                    self.faces[i].map(|v| if v as usize == remove { keep as u32 } else { v });
                self.incidents[keep].insert(i);
            }
        }
        self.incidents[remove].clear();
        removed
    }
}

pub fn remesh(
    mesh: &MeshData,
    faces: &[[u32; 3]],
    points: &[Vec3],
    uv: Option<&[[f32; 2]]>,
    c: &SkinRemeshConfig,
) -> Result<Remeshed> {
    ensure!(
        (0.0005..=0.03).contains(&c.target_edge_m)
            && (0.1..=1.0).contains(&c.target_face_fraction)
            && (0.00001..=0.003).contains(&c.maximum_sampled_surface_error_m)
            && (0.00001..=0.01).contains(&c.maximum_uv_error)
            && (10.0..=80.0).contains(&c.feature_angle_degrees)
            && c.maximum_collapses <= 50000,
        "invalid_skin_remesh_config"
    );
    let selected: BTreeSet<_> = (0..points.len()).collect();
    let query_faces: Vec<_> = faces
        .iter()
        .copied()
        .filter(|f| f.iter().all(|v| selected.contains(&(*v as usize))))
        .collect();
    ensure!(
        !query_faces.is_empty(),
        "skin_remesh_no_observed_skin_faces"
    );
    let query = Surface::new(
        &points.iter().copied().map(V::from).collect::<Vec<_>>(),
        &query_faces
            .iter()
            .map(|f| f.map(|v| v as usize))
            .collect::<Vec<_>>(),
    );
    let mut incidents = vec![BTreeSet::new(); points.len()];
    let mut edges = BTreeMap::<(usize, usize), Vec<usize>>::new();
    let mut pinned = vec![false; points.len()];
    for (i, &face) in faces.iter().enumerate() {
        let skin = face.iter().all(|v| selected.contains(&(*v as usize)));
        for v in face {
            incidents[v as usize].insert(i);
            if !skin {
                pinned[v as usize] = true;
            }
        }
        let [a, b, c] = face.map(|v| v as usize);
        for (a, b) in [(a, b), (b, c), (c, a)] {
            edges.entry((a.min(b), a.max(b))).or_default().push(i);
        }
    }
    let normal = |i: usize| {
        let [a, b, c] = faces[i].map(|v| v3(points[v as usize]));
        (b - a)
            .cross(&(c - a))
            .try_normalize(1e-15)
            .unwrap_or_default()
    };
    let crease = c.feature_angle_degrees.to_radians().cos();
    for (&(a, b), row) in &edges {
        if row.len() != 2
            || normal(row[0]).dot(&normal(row[1])) < crease
            || mesh.materials.get(row[0]) != mesh.materials.get(row[1])
        {
            pinned[a] = true;
            pinned[b] = true;
        }
    }
    let mut heap = BinaryHeap::new();
    let mut version = vec![0u32; points.len()];
    let mut live_vertex = vec![true; points.len()];
    let enqueue = |a: usize, b: usize, heap: &mut BinaryHeap<Edge>, version: &[u32]| {
        if a == b || pinned[a] || pinned[b] || !selected.contains(&a) || !selected.contains(&b) {
            return;
        }
        let length = (v3(points[a]) - v3(points[b])).norm_squared();
        if length <= c.target_edge_m.powi(2) {
            heap.push(Edge {
                length,
                a,
                b,
                version_a: version[a],
                version_b: version[b],
            });
        }
    };
    for &(a, b) in edges.keys() {
        enqueue(a, b, &mut heap, &version);
    }
    let original_skin_faces = query_faces.len();
    let target = (original_skin_faces as f64 * c.target_face_fraction).ceil() as usize;
    let mut domain = Domain {
        points,
        uv,
        faces: faces.to_vec(),
        alive: vec![true; faces.len()],
        incidents,
        query,
        query_faces,
        config: c,
    };
    let mut collapsed = 0;
    let mut removed = 0;
    let mut maximum_error = 0.0f64;
    while let Some(edge) = heap.pop() {
        if collapsed >= c.maximum_collapses || original_skin_faces.saturating_sub(removed) <= target
        {
            break;
        }
        if !live_vertex[edge.a]
            || !live_vertex[edge.b]
            || version[edge.a] != edge.version_a
            || version[edge.b] != edge.version_b
        {
            continue;
        }
        let choice = domain
            .check(edge.a, edge.b)
            .map(|e| (edge.a, edge.b, e))
            .or_else(|| domain.check(edge.b, edge.a).map(|e| (edge.b, edge.a, e)));
        let Some((keep, remove, error)) = choice else {
            continue;
        };
        removed += domain.collapse(keep, remove);
        collapsed += 1;
        maximum_error = maximum_error.max(error);
        live_vertex[remove] = false;
        version[keep] += 1;
        version[remove] += 1;
        for j in domain.neighbours(keep) {
            enqueue(keep.min(j), keep.max(j), &mut heap, &version);
        }
    }
    let face_sources = domain
        .alive
        .iter()
        .enumerate()
        .filter_map(|(i, &alive)| alive.then_some(i))
        .collect();
    let remeshed = domain
        .faces
        .into_iter()
        .zip(domain.alive)
        .filter_map(|(f, alive)| alive.then_some(f))
        .collect::<Vec<_>>();
    let report = json!({"method":"skin_domain_topology_safe_endpoint_remesh_v1","source_skin_faces":original_skin_faces,"output_skin_faces":original_skin_faces-removed,"source_part_faces":faces.len(),"output_part_faces":remeshed.len(),"collapsed_edges":collapsed,"maximum_sampled_surface_error_m":maximum_error,"sampled_error_budget_m":c.maximum_sampled_surface_error_m,"uv_interpolation_error_budget":c.maximum_uv_error,"boundary_and_crease_vertices_pinned":pinned.iter().filter(|v|**v).count(),"surviving_source_vertex_positions_and_uv":"unchanged","face_samples_tested":"centroid_and_three_edge_midpoints","selected_mesh_only":true,"quad_facial_edge_flow_created":false});
    Ok(Remeshed {
        faces: remeshed,
        face_sources,
        report,
    })
}
