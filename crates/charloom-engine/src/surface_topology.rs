//! Recover geometric adjacency across render-only UV/normal seams without
//! changing positions, indices, UVs, normals, materials, or nearby layers.
use anyhow::{Result, ensure};
use charloom_core::{Bounds, Influence, Mesh, v3};
use charloom_io::{load_gltf, write_json};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashMap},
    path::Path,
};

#[derive(Debug, Clone)]
pub struct SurfaceTopology {
    pub representative: Vec<usize>,
    pub groups: Vec<Vec<usize>>,
    pub group_index: Vec<Option<usize>>,
    pub adjacency: Vec<Vec<usize>>,
    pub ambiguous_edges: usize,
}

fn root(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}
fn join(parent: &mut [usize], a: usize, b: usize) {
    let (a, b) = (root(parent, a), root(parent, b));
    parent[a.max(b)] = a.min(b);
}

type EdgeIncidents = HashMap<(usize, usize), Vec<(usize, usize, usize)>>;

impl SurfaceTopology {
    pub fn new(mesh: &Mesh) -> Self {
        Self::infer(mesh, &[])
    }

    pub fn with_reviewed_couplings(mesh: &Mesh, reviewed: &[Vec<usize>]) -> Result<Self> {
        for group in reviewed {
            ensure!(group.len() >= 2, "seam_coupling_requires_multiple_vertices");
            ensure!(
                group.iter().all(|&i| i < mesh.positions.len()),
                "seam_coupling_vertex_out_of_range"
            );
            let point = mesh.positions[group[0]];
            ensure!(
                group.iter().all(|&i| mesh.positions[i] == point),
                "seam_coupling_positions_must_be_exact"
            );
        }
        Ok(Self::infer(mesh, reviewed))
    }

    fn infer(mesh: &Mesh, reviewed: &[Vec<usize>]) -> Self {
        let n = mesh.positions.len();
        let mut point_ids = HashMap::new();
        let mut duplicates = vec![0usize; n];
        let exact: Vec<_> = mesh
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                // Exact source values only; proximity must not weld fingers or layers.
                let key = p.map(|v| if v == 0.0 { 0 } else { v.to_bits() });
                let id = *point_ids.entry(key).or_insert(i);
                duplicates[id] += 1;
                id
            })
            .collect();
        let mut edges: EdgeIncidents = HashMap::new();
        for (t, tri) in mesh.triangles.iter().enumerate() {
            let [a, b, c] = tri.map(|v| v as usize);
            for (a, b) in [(a, b), (b, c), (c, a)] {
                let (x, y) = (exact[a], exact[b]);
                if x != y && (duplicates[x] > 1 || duplicates[y] > 1) {
                    edges
                        .entry((x.min(y), x.max(y)))
                        .or_default()
                        .push((a, b, t));
                }
            }
        }
        let mut parent: Vec<_> = (0..n).collect();
        let mut ambiguous_edges = 0;
        for incidents in edges.values() {
            // One opposite directed face on each side establishes a seam.
            // Coincident complete sheets and non-manifold edges are ambiguous.
            if incidents.len() != 2 {
                if incidents.len() > 2 {
                    ambiguous_edges += 1;
                }
                continue;
            }
            let (a, b, ta) = incidents[0];
            let (c, d, tb) = incidents[1];
            if a == d && b == c {
                continue;
            }
            if exact[a] != exact[d] || exact[b] != exact[c] {
                continue;
            }
            let normal = |t: usize| {
                let [i, j, k] = mesh.triangles[t].map(|i| v3(mesh.positions[i as usize]));
                (j - i).cross(&(k - i))
            };
            let (na, nb) = (normal(ta), normal(tb));
            // Do not join opposing faces of touching inner/outer layers.
            if na.norm_squared() < 1e-24
                || nb.norm_squared() < 1e-24
                || na.dot(&nb) < -1e-10 * na.norm() * nb.norm()
            {
                continue;
            }
            join(&mut parent, a, d);
            join(&mut parent, b, c);
        }
        for group in reviewed {
            for &i in &group[1..] {
                join(&mut parent, group[0], i);
            }
        }
        let representative: Vec<_> = (0..n).map(|i| root(&mut parent, i)).collect();
        let mut members = BTreeMap::<usize, Vec<usize>>::new();
        for (i, &r) in representative.iter().enumerate() {
            members.entry(r).or_default().push(i);
        }
        let groups: Vec<_> = members.values().filter(|v| v.len() > 1).cloned().collect();
        let mut group_index = vec![None; n];
        for (g, vs) in groups.iter().enumerate() {
            for &v in vs {
                group_index[v] = Some(g);
            }
        }
        let mut adjacency = vec![Vec::new(); n];
        for tri in &mesh.triangles {
            let [a, b, c] = tri.map(|v| representative[v as usize]);
            for (a, b) in [(a, b), (b, c), (c, a)] {
                if a != b {
                    adjacency[a].push(b);
                    adjacency[b].push(a);
                }
            }
        }
        for row in &mut adjacency {
            row.sort_unstable();
            row.dedup();
        }
        for vs in &groups {
            let stencil = adjacency[vs[0]].clone();
            for &i in &vs[1..] {
                adjacency[i] = stencil.clone();
            }
        }
        Self {
            representative,
            groups,
            group_index,
            adjacency,
            ambiguous_edges,
        }
    }

    pub fn components(&self) -> Vec<Vec<usize>> {
        let mut parent = self.representative.clone();
        for (i, row) in self.adjacency.iter().enumerate() {
            for &j in row {
                join(&mut parent, i, j);
            }
        }
        let mut groups = BTreeMap::<usize, Vec<usize>>::new();
        for i in 0..parent.len() {
            let r = root(&mut parent, i);
            groups.entry(r).or_default().push(i);
        }
        let mut groups: Vec<_> = groups.into_values().collect();
        groups.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
        groups
    }

    pub fn weight_l1_max(&self, weights: &[Vec<Influence>]) -> f64 {
        self.groups
            .iter()
            .flat_map(|group| {
                group
                    .iter()
                    .skip(1)
                    .map(|&i| weight_l1(&weights[group[0]], &weights[i]))
            })
            .fold(0.0, f64::max)
    }
}

pub fn weight_l1(a: &[Influence], b: &[Influence]) -> f64 {
    let mut differences = BTreeMap::<usize, f64>::new();
    for w in a {
        *differences.entry(w.joint).or_default() += w.weight;
    }
    for w in b {
        *differences.entry(w.joint).or_default() -= w.weight;
    }
    differences.values().map(|v| v.abs()).sum()
}

/// Shared palette selection is required: independently truncating the duplicate
/// rows after smoothing can recreate a crack even when the full weights agree.
pub fn mean_weights<'a>(
    rows: impl Iterator<Item = &'a Vec<Influence>>,
    limit: usize,
) -> Vec<Influence> {
    let mut combined = BTreeMap::<usize, f64>::new();
    for row in rows {
        for w in row {
            *combined.entry(w.joint).or_default() += w.weight;
        }
    }
    let mut row: Vec<_> = combined
        .into_iter()
        .map(|(joint, weight)| Influence { joint, weight })
        .collect();
    row.sort_by(|a, b| b.weight.total_cmp(&a.weight).then(a.joint.cmp(&b.joint)));
    row.truncate(limit);
    let sum: f64 = row.iter().map(|w| w.weight).sum();
    if sum > 0.0 {
        for w in &mut row {
            w.weight /= sum;
        }
    }
    row
}

pub fn analyze_path(input: &Path, out: &Path) -> Result<Value> {
    let source = load_gltf(input)?;
    let positions = source.asset.world_meshes()?;
    let mut summaries = Vec::new();
    let mut seams = Vec::new();
    let mut components = Vec::new();
    for (m, mesh) in source.asset.meshes.iter().enumerate() {
        let topology = SurfaceTopology::new(mesh);
        let groups = topology.components();
        let hash = crate::surface_region::geometry_hash(mesh);
        summaries.push(json!({"mesh":m,"vertices":mesh.positions.len(),"seam_groups":topology.groups.len(),"seam_vertices":topology.groups.iter().map(Vec::len).sum::<usize>(),"max_seam_weight_l1":if mesh.skin.is_some(){topology.weight_l1_max(&mesh.influences)}else{0.0},"ambiguous_edges":topology.ambiguous_edges,"connected_surfaces":groups.len(),"surfaces":groups.iter().enumerate().map(|(i,vs)|json!({"id":i,"vertices":vs.len(),"bounds":Bounds::from_points(vs.iter().map(|&v|positions[m][v]))})).collect::<Vec<_>>()}));
        seams.push(json!({"mesh":m,"geometry_hash":hash,"groups":topology.groups}));
        components.push(json!({"mesh":m,"geometry_hash":hash,"surfaces":groups}));
    }
    let summary = json!({"source_hash":source.asset.source_hash,"meshes":summaries,"method":"exact_opposite_boundary_edges_v1","limitations":["Only exact-position, consistently oriented manifold seams are coupled; ambiguous and nearby surfaces remain separate.","Connected surfaces are geometric components, not semantic clothing layers or collision certification."]});
    crate::pipeline::publish_bundle(out, |dir| {
        write_json(&dir.join("summary.json"), &summary)?;
        write_json(&dir.join("seams.json"), &seams)?;
        write_json(&dir.join("components.json"), &components)
    })?;
    Ok(json!({"out":std::fs::canonicalize(out)?,"summary":summary}))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn mesh(positions: Vec<[f64; 3]>, triangles: Vec<[u32; 3]>) -> Mesh {
        Mesh {
            name: "test".into(),
            node: 0,
            primitive: 0,
            positions,
            triangles,
            skin: None,
            influences: vec![],
            morph_targets: vec![],
            morph_weights: vec![],
        }
    }
    #[test]
    fn joins_split_boundary_stencils_but_not_touching_or_overlapping_layers() {
        let p = vec![
            [0., 0., 0.],
            [1., 0., 0.],
            [0., 1., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
        ];
        let m = mesh(p.clone(), vec![[0, 1, 2], [3, 4, 5]]);
        let t = SurfaceTopology::new(&m);
        assert_eq!(t.groups, vec![vec![1, 3], vec![2, 5]]);
        assert_eq!(t.adjacency[1], t.adjacency[3]);
        assert_eq!(t.components().len(), 1);
        let mut overlap = p;
        overlap.extend(m.positions.clone());
        let t = SurfaceTopology::new(&mesh(
            overlap,
            vec![[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        ));
        assert!(
            t.groups.is_empty(),
            "coincident complete layers are ambiguous"
        );
        let t = SurfaceTopology::new(&mesh(
            vec![
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.],
                [0., -1., 0.],
                [-1., 0., 0.],
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        ));
        assert!(t.groups.is_empty(), "point contact is not a seam");
    }
}
