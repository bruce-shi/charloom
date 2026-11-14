//! Small, versioned surface constraints from the web review editor.
//! Compact geodesic support avoids pulling a nearby arm and torso together.
use crate::{
    Input, Output, arr,
    surface::{Surface, V, closest},
};
use anyhow::{Context, Result, ensure};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, BinaryHeap},
};

#[derive(Clone, Deserialize)]
pub struct HintDocument {
    pub schema_version: u32,
    pub character_id: String,
    pub reference_result_sha256: String,
    pub coordinate_system: String,
    pub points: Vec<Hint>,
}
#[derive(Clone, Deserialize)]
pub struct Hint {
    pub id: String,
    pub anchor_m: [f64; 3],
    pub target_m: [f64; 3],
    pub role: String,
    pub clearance_mm: f64,
    pub influence_radius_m: f64,
}
fn from_gltf([x, y, z]: [f64; 3]) -> V {
    V::new(x, -z, y)
}
fn to_gltf(p: V) -> [f64; 3] {
    [p.x, p.z, -p.y]
}
impl HintDocument {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema_version == 1 && self.coordinate_system == "gltf_y_up_m",
            "Unsupported annotation version or coordinates"
        );
        ensure!(
            !self.points.is_empty() && self.points.len() <= 22,
            "Use one to twenty-two points"
        );
        let mut ids = BTreeSet::new();
        for p in &self.points {
            let valid = [
                "neck",
                "shoulder",
                "upper_arm",
                "elbow",
                "wrist",
                "chest",
                "waist",
                "hip",
                "knee",
                "ankle",
            ]
            .iter()
            .any(|name| p.id == format!("{name}_left") || p.id == format!("{name}_right"))
                || ["neck_front", "neck_back"].contains(&p.id.as_str());
            ensure!(
                valid && ids.insert(&p.id),
                "Unknown or duplicate point: {}",
                p.id
            );
            ensure!(
                p.anchor_m
                    .iter()
                    .chain(&p.target_m)
                    .all(|v| v.is_finite() && v.abs() < 10.),
                "Nonfinite or invalid point"
            );
            ensure!(
                (from_gltf(p.target_m) - from_gltf(p.anchor_m)).norm() <= 0.15,
                "Point movement exceeds 150 mm"
            );
            ensure!(
                ["design_target", "observed_surface"].contains(&p.role.as_str()),
                "Unknown constraint role"
            );
            ensure!(
                p.clearance_mm.is_finite() && (0.0..=50.).contains(&p.clearance_mm),
                "Invalid inward allowance"
            );
            ensure!(
                p.influence_radius_m.is_finite() && (0.06..=0.35).contains(&p.influence_radius_m),
                "Invalid influence radius"
            );
        }
        Ok(())
    }
}
#[derive(Clone, Copy, PartialEq)]
struct Visit {
    distance: f64,
    id: usize,
}
impl Eq for Visit {}
impl Ord for Visit {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .distance
            .total_cmp(&self.distance)
            .then_with(|| self.id.cmp(&other.id))
    }
}
impl PartialOrd for Visit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
fn distances(
    graph: &[Vec<(usize, f64)>],
    seeds: impl IntoIterator<Item = (usize, f64)>,
    limit: f64,
) -> Vec<f64> {
    let mut d = vec![f64::INFINITY; graph.len()];
    let mut heap = BinaryHeap::new();
    for (id, distance) in seeds {
        if distance < d[id] {
            d[id] = distance;
            heap.push(Visit { id, distance });
        }
    }
    while let Some(Visit { id, distance }) = heap.pop() {
        if distance > d[id] || distance > limit {
            continue;
        }
        for &(next, length) in &graph[id] {
            let v = distance + length;
            if v < d[next] && v <= limit {
                d[next] = v;
                heap.push(Visit {
                    id: next,
                    distance: v,
                });
            }
        }
    }
    d
}
struct Handle {
    face: [usize; 3],
    weights: [f64; 3],
    anchor: V,
    normal: V,
    requested: [f64; 3],
    radius: f64,
    // Sparse geodesic support makes each drag update proportional to affected vertices.
    basis: Vec<(usize, f64)>,
}
/// Portable input for both native and browser local edits. Coordinates are Blender Z-up metres.
#[derive(Serialize, Deserialize)]
pub struct DeformationMesh {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[usize; 3]>,
    pub frozen: Vec<usize>,
}
/// An immutable base mesh and reusable control supports. Preview never runs collision checks.
/// Every edit is evaluated against this base, so undo and repeated requests cannot accumulate drift.
pub struct LocalDeformer {
    before: Vec<V>,
    faces: Vec<[usize; 3]>,
    frozen: BTreeSet<usize>,
    graph: Vec<Vec<(usize, f64)>>,
    fixed_distance: Vec<f64>,
    handles: BTreeMap<String, Handle>,
    prior_crossings: Option<BTreeSet<[usize; 2]>>,
    base_normals: Vec<V>,
}
fn barycentric(q: V, [a, b, c]: [V; 3]) -> [f64; 3] {
    let ab = b - a;
    let ac = c - a;
    let v = q - a;
    let d = ab.dot(&ab) * ac.dot(&ac) - ab.dot(&ac).powi(2);
    let y = (ac.dot(&ac) * v.dot(&ab) - ab.dot(&ac) * v.dot(&ac)) / d;
    let z = (ab.dot(&ab) * v.dot(&ac) - ab.dot(&ac) * v.dot(&ab)) / d;
    [1. - y - z, y, z]
}
impl LocalDeformer {
    pub fn new(mesh: DeformationMesh) -> Result<Self> {
        let before: Vec<V> = mesh.vertices.into_iter().map(V::from).collect();
        let faces = mesh.triangles;
        let frozen = mesh.frozen.into_iter().collect::<BTreeSet<_>>();
        ensure!(
            before.len() >= 3 && !faces.is_empty(),
            "Empty deformation mesh"
        );
        ensure!(
            before.iter().flatten().all(|v| v.is_finite()),
            "Nonfinite mesh"
        );
        ensure!(
            faces
                .iter()
                .flatten()
                .chain(frozen.iter())
                .all(|&i| i < before.len()),
            "Invalid topology index"
        );
        let mut graph = vec![Vec::new(); before.len()];
        let mut edge_set = BTreeSet::new();
        for &[a, b, c] in &faces {
            for (a, b) in [(a, b), (b, c), (c, a)] {
                edge_set.insert((a.min(b), a.max(b)));
            }
        }
        for (a, b) in edge_set {
            let d = (before[a] - before[b]).norm();
            graph[a].push((b, d));
            graph[b].push((a, d));
        }
        let fixed_distance = distances(&graph, frozen.iter().map(|&i| (i, 0.)), 0.03);
        let base_normals = crate::normals(&before, &faces);
        Ok(Self {
            before,
            faces,
            frozen,
            graph,
            fixed_distance,
            handles: BTreeMap::new(),
            prior_crossings: None,
            base_normals,
        })
    }
    fn prepare(&mut self, hint: &Hint) -> Result<()> {
        if self
            .handles
            .get(&hint.id)
            .is_some_and(|h| h.requested == hint.anchor_m && h.radius == hint.influence_radius_m)
        {
            return Ok(());
        }
        let requested = from_gltf(hint.anchor_m);
        let (face, anchor) = self
            .faces
            .iter()
            .filter(|f| !f.iter().all(|i| self.frozen.contains(i)))
            .filter(|f| {
                (self.before[f[1]] - self.before[f[0]])
                    .cross(&(self.before[f[2]] - self.before[f[0]]))
                    .norm()
                    > 1e-12
            })
            .map(|&f| (f, closest(requested, f.map(|i| self.before[i]))))
            .min_by(|a, b| {
                (a.1 - requested)
                    .norm_squared()
                    .total_cmp(&(b.1 - requested).norm_squared())
            })
            .context("No editable skin surface")?;
        ensure!(
            (anchor - requested).norm() < 0.015,
            "{} is more than 15 mm from the editable surface",
            hint.id
        );
        let normal = (self.before[face[1]] - self.before[face[0]])
            .cross(&(self.before[face[2]] - self.before[face[0]]))
            .normalize();
        let weights = barycentric(anchor, face.map(|i| self.before[i]));
        let d = distances(
            &self.graph,
            face.map(|i| (i, (self.before[i] - anchor).norm())),
            hint.influence_radius_m,
        );
        let basis = d
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| {
                let r = (d / hint.influence_radius_m).min(1.);
                let t = (self.fixed_distance[i] / 0.03).min(1.);
                let value = (1. - r).powi(4) * (4. * r + 1.) * t * t * (3. - 2. * t);
                (value > 0.).then_some((i, value))
            })
            .collect();
        self.handles.insert(
            hint.id.clone(),
            Handle {
                face,
                weights,
                anchor,
                normal,
                requested: hint.anchor_m,
                radius: hint.influence_radius_m,
                basis,
            },
        );
        Ok(())
    }
    fn target(h: &Handle, p: &Hint) -> V {
        from_gltf(p.target_m)
            - h.normal
                * if p.role == "observed_surface" {
                    p.clearance_mm / 1000.
                } else {
                    0.
                }
    }
    fn displacement(&mut self, hints: &[Hint]) -> Result<Vec<V>> {
        let n = hints.len();
        let mut delta = vec![V::zeros(); self.before.len()];
        if n == 0 {
            return Ok(delta);
        }
        ensure!(n <= 22, "Use at most twenty-two points");
        for hint in hints {
            self.prepare(hint)?;
        }
        let handles = hints
            .iter()
            .map(|p| &self.handles[&p.id])
            .collect::<Vec<_>>();
        let matrix = DMatrix::from_fn(n, n, |j, k| {
            (0..3)
                .map(|v| {
                    let id = handles[j].face[v];
                    let value = handles[k]
                        .basis
                        .binary_search_by_key(&id, |&(i, _)| i)
                        .map(|i| handles[k].basis[i].1)
                        .unwrap_or(0.);
                    handles[j].weights[v] * value
                })
                .sum()
        });
        let desired = DMatrix::from_fn(n, 3, |j, axis| {
            (Self::target(handles[j], &hints[j]) - handles[j].anchor)[axis]
        });
        let coefficients = matrix
            .lu()
            .solve(&desired)
            .context("Overlapping or fixed constraints cannot be resolved; reposition points")?;
        for (k, h) in handles.iter().enumerate() {
            let vector = V::new(
                coefficients[(k, 0)],
                coefficients[(k, 1)],
                coefficients[(k, 2)],
            );
            for &(i, value) in &h.basis {
                delta[i] += vector * value;
            }
        }
        ensure!(
            delta.iter().flatten().all(|v| v.is_finite()),
            "Nonfinite deformation"
        );
        ensure!(
            delta.iter().all(|v| v.norm() <= 0.20),
            "Unstable constraints; reduce movement or separate points"
        );
        Ok(delta)
    }
    /// Fast, unvalidated positions. Caller must label them as a preview until `checked` succeeds.
    pub fn preview(&mut self, hints: &[Hint]) -> Result<Vec<[f64; 3]>> {
        let delta = self.displacement(hints)?;
        Ok(self
            .before
            .iter()
            .zip(delta)
            .map(|(p, d)| arr(p + d))
            .collect())
    }
    pub fn checked(&mut self, hints: &[Hint]) -> Result<(Vec<[f64; 3]>, Value)> {
        let delta = self.displacement(hints)?;
        let prior_crossings = self
            .prior_crossings
            .get_or_insert_with(|| Surface::new(&self.before, &self.faces).crossings());
        // Unchanged triangles cannot introduce a crossing or a fold. Use exact
        // zero displacement here: the reporting threshold is not a safety cutoff.
        let changed_faces: Vec<_> = self
            .faces
            .iter()
            .map(|f| f.iter().any(|&i| delta[i] != V::zeros()))
            .collect();
        let unchanged_crossing = prior_crossings
            .iter()
            .any(|&[a, b]| !changed_faces[a] && !changed_faces[b]);
        let mut scale = 1.;
        let mut attempts = Vec::new();
        loop {
            let candidate = self
                .before
                .iter()
                .zip(&delta)
                .map(|(p, d)| p + d * scale)
                .collect::<Vec<_>>();
            let failures = self
                .faces
                .iter()
                .zip(&changed_faces)
                .filter(|&(&[a, b, c], &changed)| {
                    if !changed {
                        return false;
                    }
                    let old =
                        (self.before[b] - self.before[a]).cross(&(self.before[c] - self.before[a]));
                    let new = (candidate[b] - candidate[a]).cross(&(candidate[c] - candidate[a]));
                    old.norm() > 1e-12 && (old.dot(&new) <= 0. || new.norm() < old.norm() * 0.05)
                })
                .count();
            let crossings =
                Surface::new(&candidate, &self.faces).crossings_involving(&changed_faces);
            let introduced = crossings.difference(prior_crossings).count();
            attempts.push(json!({"fraction":scale,"new_crossings":introduced,"folded_or_collapsed_triangles":failures}));
            if failures == 0 && introduced == 0 {
                let rows=hints.iter().map(|p| {
                    let h=&self.handles[&p.id];
                    let target=Self::target(h,p);
                    let actual=(0..3).map(|k|candidate[h.face[k]]*h.weights[k]).sum::<V>();
                    json!({"id":p.id,"role":p.role,"target_m":to_gltf(target),"actual_m":to_gltf(actual),"residual_mm":(actual-target).norm()*1000.,"anchor_projection_error_mm":(h.anchor-from_gltf(p.anchor_m)).norm()*1000.,"influence_radius_m":p.influence_radius_m})
                }).collect::<Vec<_>>();
                let residual = rows
                    .iter()
                    .map(|r| r["residual_mm"].as_f64().unwrap())
                    .fold(0f64, f64::max);
                return Ok((
                    candidate.into_iter().map(arr).collect(),
                    json!({"method":"compact geodesic radial basis displacement with barycentric point constraints and fixed facial regions","accepted_fraction":scale,"targets_met_within_1mm":residual<=1.,"points":rows,"attempts":attempts,"existing_crossings":prior_crossings.len(),"new_crossings":0,"folded_or_collapsed_triangles":0,"changed_vertices":delta.iter().filter(|d|d.norm()>1e-10).count(),"all_body_contacts_resolved":!unchanged_crossing && crossings.is_empty(),"clothing_refit_required_before_export":true,"user_approved":false,"animation_validated":false,"runtime_adoption":false}),
                ));
            }
            scale *= 0.5;
            ensure!(
                scale >= 0.03125,
                "No acceptable local update; constraints remain saved for revision"
            );
        }
    }
    /// Transferable XYZ displacement and normal-change buffers in glTF Y-up coordinates.
    /// Keeping offsets preserves GLB UV splits and exactly restores original buffers on reset.
    pub fn render_offsets(&self, points: &[[f64; 3]]) -> Vec<f32> {
        let after = points.iter().copied().map(V::from).collect::<Vec<_>>();
        let normals = crate::normals(&after, &self.faces);
        let mut out = Vec::with_capacity(self.before.len() * 6);
        for (p, base) in after.iter().zip(&self.before) {
            out.extend(to_gltf(p - base).map(|v| v as f32));
        }
        for (n, base) in normals.iter().zip(&self.base_normals) {
            out.extend(to_gltf(n - base).map(|v| v as f32));
        }
        out
    }
}
fn deform(
    before: &[V],
    faces: &[[usize; 3]],
    frozen: &BTreeSet<usize>,
    hints: &[Hint],
) -> Result<(Vec<V>, Value)> {
    let mut solver = LocalDeformer::new(DeformationMesh {
        vertices: before.iter().copied().map(arr).collect(),
        triangles: faces.to_vec(),
        frozen: frozen.iter().copied().collect(),
    })?;
    let (points, report) = solver.checked(hints)?;
    Ok((points.into_iter().map(V::from).collect(), report))
}
pub fn apply(input: &Input, result: &mut Output, hints: &HintDocument) -> Result<()> {
    hints.validate()?;
    ensure!(
        input.id == result.id && result.id == hints.character_id,
        "Character identity mismatch"
    );
    ensure!(
        input.template.len() == result.assembled.len()
            && input.body_ids.iter().all(|&i| i < result.assembled.len())
            && input
                .triangles
                .iter()
                .flatten()
                .all(|&i| i < result.assembled.len()),
        "Topology mismatch"
    );
    let before = result
        .assembled
        .iter()
        .copied()
        .map(V::from)
        .collect::<Vec<_>>();
    ensure!(
        before.iter().flatten().all(|v| v.is_finite()),
        "Nonfinite mesh"
    );
    let skin = input.body_ids.iter().copied().collect::<BTreeSet<_>>();
    let faces = input
        .triangles
        .iter()
        .filter(|f| f.iter().all(|i| skin.contains(i)))
        .copied()
        .collect::<Vec<_>>();
    let chin = result
        .landmarks
        .iter()
        .find(|l| l.name == "chin")
        .context("Missing chin landmark")?;
    let frozen = input
        .template
        .iter()
        .enumerate()
        .filter(|(_, p)| p[2] >= chin.original[2] + 0.035)
        .map(|(i, _)| i)
        .chain(result.landmarks.iter().map(|l| l.vertex))
        .chain(input.face_regions.cavities.iter().flatten().copied())
        .chain(input.face_regions.scalp_preserve.iter().copied())
        .collect::<BTreeSet<_>>();
    let (points, report) = deform(&before, &faces, &frozen, &hints.points)?;
    result.assembled = points.into_iter().map(arr).collect();
    result.report["shape_hints"] = report;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_body_editor_points_validate_without_relaxing_movement_limits() {
        let mut points = Vec::new();
        for part in [
            "neck",
            "shoulder",
            "upper_arm",
            "elbow",
            "wrist",
            "chest",
            "waist",
            "hip",
            "knee",
            "ankle",
        ] {
            for side in ["left", "right"] {
                points.push(Hint {
                    id: format!("{part}_{side}"),
                    anchor_m: [0., 1., 0.],
                    target_m: [0., 1., 0.],
                    role: "design_target".into(),
                    clearance_mm: 0.,
                    influence_radius_m: 0.18,
                });
            }
        }
        for id in ["neck_front", "neck_back"] {
            points.push(Hint {
                id: id.into(),
                anchor_m: [0., 1., 0.],
                target_m: [0., 1., 0.],
                role: "design_target".into(),
                clearance_mm: 0.,
                influence_radius_m: 0.12,
            });
        }
        let mut doc = HintDocument {
            schema_version: 1,
            character_id: "editor-test".into(),
            reference_result_sha256: "a".repeat(64),
            coordinate_system: "gltf_y_up_m".into(),
            points,
        };
        assert_eq!(doc.points.len(), 22);
        doc.validate().unwrap();
        doc.points[0].target_m = [0.151, 1., 0.];
        assert!(doc.validate().is_err());
        doc.points[0].target_m = [0., 1., 0.];
        doc.points[0].id = "unsupported".into();
        assert!(doc.validate().is_err());
    }
    #[test]
    fn gltf_coordinates_roundtrip() {
        let p = [0.2, 1.3, -0.4];
        assert_eq!(to_gltf(from_gltf(p)), p);
        assert_eq!(from_gltf(p), V::new(0.2, 0.4, 1.3));
    }
    #[test]
    fn point_targets_and_observation_clearance_keep_distant_skin_fixed() {
        let mut points = Vec::new();
        for y in 0..11 {
            for x in 0..11 {
                points.push(V::new((x as f64 - 5.) * 0.02, (y as f64 - 5.) * 0.02, 0.));
            }
        }
        let mut faces = Vec::new();
        for y in 0..10 {
            for x in 0..10 {
                let i = y * 11 + x;
                faces.extend([[i, i + 1, i + 11], [i + 1, i + 12, i + 11]]);
            }
        }
        let hint = Hint {
            id: "waist_left".into(),
            anchor_m: [0., 0., 0.],
            target_m: [0., 0.012, 0.],
            role: "observed_surface".into(),
            clearance_mm: 2.,
            influence_radius_m: 0.08,
        };
        let (after, report) = deform(&points, &faces, &BTreeSet::new(), &[hint]).unwrap();
        assert!((after[60].z - 0.010).abs() < 1e-10);
        assert_eq!(after[0], points[0]);
        assert_eq!(report["accepted_fraction"], 1.);
        assert_eq!(report["targets_met_within_1mm"], true);
    }
    #[test]
    fn crossing_guard_reports_when_requested_target_cannot_be_reached() {
        let mut points = Vec::new();
        for z in [0., 0.01] {
            for y in 0..11 {
                for x in 0..11 {
                    points.push(V::new((x as f64 - 5.) * 0.02, (y as f64 - 5.) * 0.02, z));
                }
            }
        }
        let mut faces = Vec::new();
        for offset in [0, 121] {
            for y in 0..10 {
                for x in 0..10 {
                    let i = offset + y * 11 + x;
                    faces.extend([[i, i + 1, i + 11], [i + 1, i + 12, i + 11]]);
                }
            }
        }
        let hint = Hint {
            id: "waist_left".into(),
            anchor_m: [0., 0., 0.],
            target_m: [0., 0.02, 0.],
            role: "design_target".into(),
            clearance_mm: 0.,
            influence_radius_m: 0.08,
        };
        let (after, report) = deform(&points, &faces, &BTreeSet::new(), &[hint]).unwrap();
        assert!(report["accepted_fraction"].as_f64().unwrap() < 1.);
        assert_eq!(report["targets_met_within_1mm"], false);
        assert!(report["attempts"][0]["new_crossings"].as_u64().unwrap() > 0);
        assert_eq!(&after[121..], &points[121..]);
    }
    #[test]
    fn disconnected_nearby_surface_is_not_pulled() {
        let p = vec![
            V::new(0., 0., 0.),
            V::new(0.03, 0., 0.),
            V::new(0., 0.03, 0.),
            V::new(0., 0., 0.01),
            V::new(0.03, 0., 0.01),
            V::new(0., 0.03, 0.01),
        ];
        let h = Hint {
            id: "upper_arm_left".into(),
            anchor_m: [0.005, 0., -0.005],
            target_m: [0.005, -0.005, -0.005],
            role: "design_target".into(),
            clearance_mm: 0.,
            influence_radius_m: 0.08,
        };
        let (q, _) = deform(&p, &[[0, 1, 2], [3, 4, 5]], &BTreeSet::new(), &[h]).unwrap();
        assert_eq!(&q[3..], &p[3..]);
    }
}
