//! Explicit, reviewable body fitting. A source garment is an outer bound; hidden
//! anatomy remains the user's selected prior. No species or semantic prediction.
use crate::{
    body_warp::{Landmark, Warp},
    browser_mesh::MeshData,
    surface::{Surface, V},
};
use anyhow::{Result, ensure};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Evidence {
    Unknown,
    Skin,
    Clothing,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Region {
    pub id: String,
    pub axis: [String; 2],
    /// Exact template vertex IDs. Domains never use nearest source object names.
    pub vertices: Vec<usize>,
    #[serde(default)]
    pub source_triangles: Vec<usize>,
    pub evidence: Evidence,
    /// Authored radial build; one preserves the current anatomy prior.
    pub radial_scale: f64,
    pub material_thickness_mm: [f64; 2],
    pub air_gap_mm: [f64; 2],
    /// Explicit surface-capture limit, not a global nearest-point fallback.
    pub maximum_distance_m: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Guidance {
    #[serde(default)]
    #[schemars(required)]
    pub anatomy: Option<charloom_core::anatomy::AnatomyReview>,
    pub format: String,
    pub version: u32,
    pub coordinate_system: String,
    pub prior_hash: String,
    pub source_hash: String,
    pub author: String,
    pub landmarks: Vec<Landmark>,
    pub confirmed_landmarks: Vec<String>,
    pub regions: Vec<Region>,
         #[serde(default)]
    pub preserve_vertices: Vec<usize>,
}
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub mesh: MeshData,
    pub source: MeshData,
    pub guidance: Guidance,
}
/// Stable binary fingerprint, independent of JSON float formatting or object keys.
pub fn fingerprint(mesh: &MeshData) -> String {
    fingerprint_geometry(&mesh.vertices, &mesh.triangles)
}
pub fn fingerprint_geometry(vertices: &[[f64; 3]], triangles: &[[usize; 3]]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"charloom-mesh-v1\0");
    hash.update((vertices.len() as u64).to_le_bytes());
    hash.update((triangles.len() as u64).to_le_bytes());
    for &v in vertices.iter().flatten() {
        hash.update((if v == 0. { 0. } else { v }).to_le_bytes());
    }
    for &v in triangles.iter().flatten() {
        hash.update((v as u32).to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}
fn interval(value: [f64; 2]) -> bool {
    value.iter().all(|v| v.is_finite())
        && 0. <= value[0]
        && value[0] <= value[1]
        && value[1] <= 300.
}
fn validate_meshes(mesh: &MeshData, source: &MeshData) -> Result<()> {
    mesh.check()?;
    source.check()?;
    ensure!(
        mesh.vertices
            .iter()
            .chain(&source.vertices)
            .flatten()
            .all(|v| v.abs() <= 100.),
        "body_coordinates_require_metres_within_100m"
    );
    Ok(())
}
fn validate_guidance(
    mesh: &MeshData,
    source: &MeshData,
    g: &Guidance,
    prior_hash: &str,
    source_hash: &str,
) -> Result<()> {
    ensure!(
        g.format == "charloom-body-guidance"
            && g.version == 1
            && g.coordinate_system == "gltf_y_up_m",
        "body_guidance_version_or_coordinates"
    );
    ensure!(
        ["user", "agent", "mixed"].contains(&g.author.as_str()),
        "body_guidance_author"
    );
    ensure!(
        g.prior_hash == prior_hash,
        "body_prior_hash_mismatch: annotation belongs to a different body version"
    );
    ensure!(
        g.source_hash == source_hash,
        "body_source_hash_mismatch: annotation belongs to a different source surface"
    );
    ensure!(
        (4..=128).contains(&g.landmarks.len()),
        "body_landmark_count"
    );
    let mut ids = BTreeSet::new();
    ensure!(
        g.preserve_vertices.iter().all(|&i| i < mesh.vertices.len())
            && g.preserve_vertices.iter().collect::<BTreeSet<_>>().len()
                == g.preserve_vertices.len(),
        "body_invalid_preserved_vertices"
    );
    for l in &g.landmarks {
        ensure!(
            ids.insert(l.id.clone()) && !l.id.is_empty() && l.id.len() <= 128,
            "body_duplicate_landmark"
        );
        ensure!(
            l.prior
                .iter()
                .chain(&l.target)
                .all(|v| v.is_finite() && v.abs() <= 100.)
                && l.strength.is_finite()
                && l.strength > 0.
                && l.strength <= 100.,
            "body_invalid_landmark"
        );
        ensure!(
            g.confirmed_landmarks.contains(&l.id),
            "body_unconfirmed_landmark: {}",
            l.id
        );
    }
    let confirmed: BTreeSet<_> = g.confirmed_landmarks.iter().collect();
    ensure!(
        confirmed.len() == g.confirmed_landmarks.len()
            && confirmed.iter().all(|id| ids.contains(*id)),
        "body_invalid_confirmed_landmarks"
    );
    let center = g.landmarks.iter().map(|l| V::from(l.prior)).sum::<V>() / g.landmarks.len() as f64;
    let covariance = g.landmarks.iter().fold(nalgebra::Matrix3::zeros(), |a, l| {
        let d = V::from(l.prior) - center;
        a + d * d.transpose()
    });
    let mut eigen = covariance.symmetric_eigen().eigenvalues.as_slice().to_vec();
    eigen.sort_by(f64::total_cmp);
    ensure!(
        eigen[1] > 1e-6 && eigen[2] > 1e-5,
        "body_collinear_landmarks"
    );
    let mut region_ids = BTreeSet::new();
    let mut used = BTreeSet::new();
    ensure!(g.regions.len() <= 64, "body_region_budget");
    for r in &g.regions {
        ensure!(
            region_ids.insert(&r.id) && !r.id.is_empty() && r.id.len() <= 128,
            "body_duplicate_region"
        );
        ensure!(
            r.radial_scale.is_finite() && (0.6..=1.6).contains(&r.radial_scale),
            "body_invalid_girth"
        );
        ensure!(
            interval(r.material_thickness_mm) && interval(r.air_gap_mm),
            "body_invalid_clearance_interval"
        );
        ensure!(
            r.maximum_distance_m.is_finite() && (0.001..=0.3).contains(&r.maximum_distance_m),
            "body_invalid_capture_radius"
        );
        let a = g.landmarks.iter().find(|l| l.id == r.axis[0]);
        let b = g.landmarks.iter().find(|l| l.id == r.axis[1]);
        ensure!(a.is_some() && b.is_some(), "body_unknown_axis");
        let (a, b) = (a.unwrap(), b.unwrap());
        ensure!(
            (V::from(a.prior) - V::from(b.prior)).norm() > 0.005
                && (V::from(a.target) - V::from(b.target)).norm() > 0.005,
            "body_degenerate_axis"
        );
        ensure!(
            r.vertices
                .iter()
                .all(|&i| i < mesh.vertices.len() && used.insert(i)),
            "body_overlapping_or_invalid_regions"
        );
        ensure!(
            r.source_triangles.len() <= source.triangles.len()
                && r.source_triangles.iter().collect::<BTreeSet<_>>().len()
                    == r.source_triangles.len()
                && r.source_triangles
                    .iter()
                    .all(|&i| i < source.triangles.len()),
            "body_invalid_source_faces"
        );
        ensure!(
            r.evidence == Evidence::Unknown || !r.source_triangles.is_empty(),
            "body_observation_needs_explicit_faces: {}",
            r.id
        );
    }
    Ok(())
}
fn center_axis(p: V, a: V, b: V) -> (V, f64) {
    let d = b - a;
    let t = ((p - a).dot(&d) / d.norm_squared()).clamp(0., 1.);
    (a + d * t, t)
}

pub fn solve(input: Input) -> Result<Value> {
    PreparedBody::new(input.mesh, input.source)?.solve(&input.guidance)
}

/// Immutable meshes and their validated topology can serve many parameter edits.
/// Each solve still validates the guidance and checks the complete candidate.
pub struct PreparedBody {
    mesh: MeshData,
    source_mesh: MeshData,
    prior_hash: String,
    source_hash: String,
    before: Vec<V>,
    source: Vec<V>,
    edges: Vec<BTreeSet<usize>>,
    prior_crossings: BTreeSet<[usize; 2]>,
}
impl PreparedBody {
    pub fn new(mesh: MeshData, source_mesh: MeshData) -> Result<Self> {
        validate_meshes(&mesh, &source_mesh)?;
        let prior_hash = fingerprint(&mesh);
        let source_hash = fingerprint(&source_mesh);
        let before: Vec<V> = mesh.vertices.iter().copied().map(V::from).collect();
        let source = source_mesh.vertices.iter().copied().map(V::from).collect();
        let mut edges = vec![BTreeSet::new(); before.len()];
        for &[a, b, c] in &mesh.triangles {
            for (a, b) in [(a, b), (b, c), (c, a)] {
                edges[a].insert(b);
                edges[b].insert(a);
            }
        }
        let prior_crossings = Surface::new(&before, &mesh.triangles).crossings();
        Ok(Self {
            mesh,
            source_mesh,
            prior_hash,
            source_hash,
            before,
            source,
            edges,
            prior_crossings,
        })
    }
    pub fn solve(&self, g: &Guidance) -> Result<Value> {
        validate_guidance(
            &self.mesh,
            &self.source_mesh,
            g,
            &self.prior_hash,
            &self.source_hash,
        )?;
        let anatomy = charloom_core::anatomy::preflight(&charloom_core::anatomy::AnatomyRequest {
            review: g.anatomy.clone(),
            source_hash: self.source_hash.clone(),
            prior_hash: self.prior_hash.clone(),
            usage: charloom_core::anatomy::AnatomyUse::FitBody,
            preserved_part_ids: vec![],
        })?;
        let before = &self.before;
        let source = &self.source;
        let edges = &self.edges;
        let prior_crossings = &self.prior_crossings;
        let warp = Warp::fit(&g.landmarks, 0.32, 0.0001)?;
        let noop = g.landmarks.iter().all(|l| l.prior == l.target)
            && g.regions
                .iter()
                .all(|r| r.radial_scale == 1. && r.evidence == Evidence::Unknown);
        let mut proposed: Vec<V> = if noop {
            before.to_vec()
        } else {
            self.mesh
                .vertices
                .iter()
                .map(|&p| V::from(warp.point(p)))
                .collect()
        };
        let mut delta = vec![V::zeros(); before.len()];
        let mut active = vec![false; before.len()];
        let controls: BTreeMap<_, _> = g.landmarks.iter().map(|l| (l.id.as_str(), l)).collect();
        let mut region_rows = Vec::new();
        let mut surfaces = Vec::new();
        for r in &g.regions {
            let a = controls[r.axis[0].as_str()];
            let b = controls[r.axis[1].as_str()];
            let faces = r
                .source_triangles
                .iter()
                .map(|&i| self.source_mesh.triangles[i])
                .collect::<Vec<_>>();
            let surface = Surface::new(source, &faces);
            let mut constrained = 0;
            let mut missing = 0;
            for &i in &r.vertices {
                let (_, t) = center_axis(before[i], V::from(a.prior), V::from(b.prior));
                let center = V::from(a.target) * (1. - t) + V::from(b.target) * t;
                let axis = (V::from(b.target) - V::from(a.target)).normalize();
                let d = proposed[i] - center;
                let radial = d - axis * d.dot(&axis);
                let joint_taper = (t / 0.15).min(1.) * ((1. - t) / 0.15).min(1.);
                let mut wanted = proposed[i] + radial * (r.radial_scale - 1.) * joint_taper;
                if r.evidence != Evidence::Unknown {
                    let radius = radial.norm();
                    if radius > 1e-6 {
                        let dir = radial / radius;
                        if let Some((hit, _, _)) =
                            surface.ray(center, dir, radius + r.maximum_distance_m)
                        {
                            let observed = (hit - center).dot(&dir);
                            if (observed - radius).abs() <= r.maximum_distance_m {
                                let clearance =
                                    (r.material_thickness_mm[0] + r.air_gap_mm[0]) / 1000.;
                                let target = if r.evidence == Evidence::Skin {
                                    observed
                                } else {
                                    ((wanted - center).dot(&dir))
                                        .min((observed - clearance).max(radius * 0.6))
                                };
                                wanted += dir * (target - (wanted - center).dot(&dir));
                                constrained += 1;
                            } else {
                                missing += 1;
                            }
                        } else {
                            missing += 1;
                        }
                    }
                }
                delta[i] = wanted - proposed[i];
                active[i] = r.evidence != Evidence::Unknown || r.radial_scale != 1.;
            }
            region_rows.push(json!({"id":r.id,"evidence":r.evidence,"template_vertices":r.vertices.len(),"source_triangles":faces.len(),"constrained_vertices":constrained,"unobserved_vertices":missing,"radial_scale":r.radial_scale,"hidden_anatomy_estimated":r.evidence!=Evidence::Skin}));
            surfaces.push(surface);
        }
        // Smooth the correction field across neighbouring body vertices. Original
        // anatomy stays intact; there is no Laplacian shrinkage of positions.
        for _ in 0..8 {
            delta = delta
                .iter()
                .enumerate()
                .map(|(i, d)| {
                    if !active[i] || edges[i].is_empty() {
                        return *d;
                    }
                    let average =
                        edges[i].iter().map(|&j| delta[j]).sum::<V>() / edges[i].len() as f64;
                    *d * 0.8 + average * 0.2
                })
                .collect();
        }
        for (p, d) in proposed.iter_mut().zip(delta) {
            *p += d;
        }
        for &i in &g.preserve_vertices {
            proposed[i] = before[i];
        }
        ensure!(
            proposed.iter().flatten().all(|v| v.is_finite()),
            "body_nonfinite_result"
        );
        let mut fraction = 1.;
        let mut attempts = Vec::new();
        let (positions, crossings) = loop {
            let candidate = before
                .iter()
                .zip(&proposed)
                .map(|(a, b)| a + (b - a) * fraction)
                .collect::<Vec<_>>();
            let folds = self
                .mesh
                .triangles
                .iter()
                .filter(|&&[a, b, c]| {
                    let old = (before[b] - before[a]).cross(&(before[c] - before[a]));
                    let new = (candidate[b] - candidate[a]).cross(&(candidate[c] - candidate[a]));
                    old.norm() > 1e-12 && (new.dot(&old) <= 0. || new.norm() < old.norm() * 0.05)
                })
                .count();
            let crossings = Surface::new(&candidate, &self.mesh.triangles).crossings();
            let introduced = crossings.difference(prior_crossings).count();
            attempts.push(json!({"fraction":fraction,"folded_or_collapsed_triangles":folds,"new_crossings":introduced}));
            if folds == 0 && introduced == 0 {
                break (candidate, crossings);
            }
            fraction *= 0.5;
            ensure!(
                fraction >= 0.03125,
                "body_fit_rejected: no safe candidate; annotations remain usable"
            );
        };
        let mut unresolved = 0;
        for (ri, r) in g.regions.iter().enumerate() {
            let a = controls[r.axis[0].as_str()];
            let b = controls[r.axis[1].as_str()];
            let mut misses = 0;
            let mut residuals = Vec::<f64>::new();
            for &i in &r.vertices {
                if r.evidence == Evidence::Unknown {
                    continue;
                }
                let (_, t) = center_axis(before[i], V::from(a.prior), V::from(b.prior));
                let center = V::from(a.target) * (1. - t) + V::from(b.target) * t;
                let axis = (V::from(b.target) - V::from(a.target)).normalize();
                let p = positions[i];
                let radial = p - center - axis * (p - center).dot(&axis);
                let radius = radial.norm();
                if radius < 1e-6 {
                    continue;
                }
                if let Some((hit, _, _)) =
                    surfaces[ri].ray(center, radial / radius, radius + r.maximum_distance_m)
                {
                    let observed = (hit - center).norm();
                    let residual = if r.evidence == Evidence::Skin {
                        (radius - observed).abs()
                    } else {
                        (radius + (r.material_thickness_mm[0] + r.air_gap_mm[0]) / 1000. - observed)
                            .max(0.)
                    };
                    residuals.push(residual);
                    if residual > 0.003 {
                        unresolved += 1;
                    }
                } else {
                    misses += 1;
                }
            }
            region_rows[ri]["unobserved_after_fit"] = json!(misses);
            region_rows[ri]["maximum_surface_residual_mm"] =
                json!(residuals.iter().copied().fold(0f64, f64::max) * 1000.);
            region_rows[ri]["unresolved_vertices"] =
                json!(residuals.iter().filter(|&&v| v > 0.003).count());
        }
        let landmark_rows=g.landmarks.iter().map(|l|{
        let fitted=V::from(l.prior)+(V::from(warp.point(l.prior))-V::from(l.prior))*fraction;
        json!({"id":l.id,"target_m":l.target,"fitted_m":[fitted.x,fitted.y,fitted.z],"residual_mm":(fitted-V::from(l.target)).norm()*1000.})
    }).collect::<Vec<_>>();
        let moved = positions
            .iter()
            .zip(before)
            .filter(|(a, b)| (*a - *b).norm() > 1e-9)
            .count();
        let max_move = positions
            .iter()
            .zip(before)
            .map(|(a, b)| (a - b).norm())
            .fold(0f64, f64::max);
        let vertices = positions
            .iter()
            .map(|v| [v.x, v.y, v.z])
            .collect::<Vec<_>>();
        let output_hash = fingerprint(&MeshData {
            vertices: vertices.clone(),
            triangles: self.mesh.triangles.clone(),
            uv: vec![],
            materials: vec![],
        });
        Ok(
            json!({"vertices":vertices,"landmarks":landmark_rows,"report":{
                "method":"guided_proper_similarity_mls_with_explicit_regional_surface_bounds_v1",
                "quality_status":"needs_review","anatomy":anatomy,"accepted_fraction":fraction,"attempts":attempts,
                "landmark_count":g.landmarks.len(),"regions":region_rows,"moved_vertices":moved,"maximum_displacement_m":max_move,
                "preserved_vertices":g.preserve_vertices.len(),
                "existing_crossings":prior_crossings.len(),"remaining_crossings":crossings.len(),"new_crossings":0,"folded_or_collapsed_triangles":0,
                "crossing_definition":"strict noncoplanar triangle crossings; coplanar contacts excluded",
                "unresolved_vertices":unresolved,"topology_unchanged":true,"unique_hidden_anatomy_recovery":false,
                "prior_hash":g.prior_hash,"source_hash":g.source_hash,"result_hash":output_hash,"input_author":g.author,
                "coordinate_system":"gltf_y_up_m","neutral_pose_required":true,
                "clothing_policy":"outer bound only; unknown regions retain the selected prior","clearance_policy":"minimum of the authored interval is enforced; no inflation to loose clothing",
                "user_approved":false,"animation_validated":false,"runtime_adoption":false
            }}),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn cylinder(radius: f64) -> MeshData {
        let mut vertices = Vec::new();
        let mut triangles = Vec::new();
        for j in 0..=4 {
            for i in 0..16 {
                let theta = i as f64 * std::f64::consts::TAU / 16.;
                vertices.push([radius * theta.cos(), j as f64 / 4., radius * theta.sin()]);
            }
        }
        for j in 0..4 {
            for i in 0..16 {
                let a = j * 16 + i;
                let b = j * 16 + (i + 1) % 16;
                triangles.push([a, b, a + 16]);
                triangles.push([b, b + 16, a + 16]);
            }
        }
        MeshData {
            vertices,
            triangles,
            uv: vec![],
            materials: vec![],
        }
    }
    fn fixture() -> Input {
        let mesh = cylinder(0.1);
        let source = cylinder(0.15);
        let landmarks = vec![
            ("a", [0., 0., 0.]),
            ("b", [0., 1., 0.]),
            ("c", [0.3, 0., 0.]),
            ("d", [-0.3, 1., 0.]),
        ]
        .into_iter()
        .map(|(id, p)| Landmark {
            id: id.into(),
            prior: p,
            target: p,
            strength: 1.,
        })
        .collect::<Vec<_>>();
        let guidance = Guidance {
            anatomy: Some(serde_json::from_value(serde_json::json!({
                "version":1,"anatomy_class":"nonhuman","source_hash":fingerprint(&source),"prior_hash":fingerprint(&mesh),"author":"agent",
                "evidence":"Explicit synthetic cylindrical observation and prior",
                "parts": (["head","body","hand","foot","chain"].map(|kind|serde_json::json!({"id":kind,"kind":kind,"source_form":"synthetic","prior_form":"synthetic","strategy":"fit_prior","evidence":"Synthetic compatible fixture","confirmed":true,"preserved_part_ids":[]})))
            })).unwrap()),
            format: "charloom-body-guidance".into(),
            version: 1,
            coordinate_system: "gltf_y_up_m".into(),
            prior_hash: fingerprint(&mesh),
            source_hash: fingerprint(&source),
            author: "user".into(),
            preserve_vertices: vec![],
            confirmed_landmarks: landmarks.iter().map(|l| l.id.clone()).collect(),
            landmarks,
            regions: vec![Region {
                id: "torso".into(),
                axis: ["a".into(), "b".into()],
                vertices: (0..mesh.vertices.len()).collect(),
                source_triangles: (0..source.triangles.len()).collect(),
                evidence: Evidence::Unknown,
                radial_scale: 1.,
                material_thickness_mm: [1., 3.],
                air_gap_mm: [3., 15.],
                maximum_distance_m: 0.1,
            }],
        };
        Input {
            mesh,
            source,
            guidance,
        }
    }
    fn positions(value: &Value) -> Vec<[f64; 3]> {
        serde_json::from_value(value["vertices"].clone()).unwrap()
    }
    #[test]
    fn body_fit_keeps_explicit_reviewed_vertices_exact() {
        let mut input = fixture();
        let before = input.mesh.vertices.clone();
        input.guidance.preserve_vertices = (64..80).collect();
        for l in &mut input.guidance.landmarks {
            l.target[0] += 0.002;
        }
        let result = solve(input.clone()).unwrap();
        let after = positions(&result);
        for &i in &input.guidance.preserve_vertices {
            assert_eq!(after[i], before[i]);
        }
        assert!(
            after[..16]
                .iter()
                .zip(&before)
                .any(|(a, b)| (a[0] - b[0]).abs() > 0.0001)
        );
        assert_eq!(result["report"]["preserved_vertices"], 16);
        input.guidance.preserve_vertices.push(usize::MAX);
        assert!(solve(input).is_err());
    }
    #[test]
    fn unchanged_prior_is_exact_and_translation_preserves_shape() {
        let mut input = fixture();
        let before = input.mesh.vertices.clone();
        let result = solve(input.clone()).unwrap();
        assert_eq!(positions(&result), before);
        assert_eq!(result["report"]["moved_vertices"], 0);
        for l in &mut input.guidance.landmarks {
            l.target[0] += 0.03;
            l.target[2] -= 0.01;
        }
        let result = solve(input).unwrap();
        for (p, q) in positions(&result).iter().zip(before) {
            assert!((p[0] - q[0] - 0.03).abs() < 1e-10);
            assert!((p[1] - q[1]).abs() < 1e-10);
            assert!((p[2] - q[2] + 0.01).abs() < 1e-10);
        }
        assert_eq!(result["report"]["accepted_fraction"], 1.);
        assert_eq!(result["report"]["user_approved"], false);
    }
    #[test]
    fn loose_garment_never_inflates_the_hidden_body_but_skin_fits() {
        let mut input = fixture();
        input.guidance.regions[0].evidence = Evidence::Clothing;
        let clothing = positions(&solve(input.clone()).unwrap());
        for p in clothing {
            assert!((p[0].hypot(p[2]) - 0.1).abs() < 1e-10);
        }
        input.guidance.regions[0].evidence = Evidence::Skin;
        let skin = solve(input).unwrap();
        for p in positions(&skin) {
            assert!(p[0].hypot(p[2]) > 0.145);
        }
        assert_eq!(skin["report"]["new_crossings"], 0);
    }
    #[test]
    fn tight_garment_contracts_and_reports_residuals_without_hiding_them() {
        let mut input = fixture();
        input.source = cylinder(0.09);
        input.guidance.source_hash = fingerprint(&input.source);
        input.guidance.anatomy.as_mut().unwrap().source_hash = input.guidance.source_hash.clone();
        input.guidance.regions[0].evidence = Evidence::Clothing;
        let result = solve(input).unwrap();
        assert!(positions(&result).iter().all(|p| p[0].hypot(p[2]) < 0.09));
        assert!(
            result["report"]["regions"][0]["maximum_surface_residual_mm"]
                .as_f64()
                .unwrap()
                < 3.
        );
        assert_eq!(result["report"]["quality_status"], "needs_review");
    }
    #[test]
    fn unknown_regions_do_not_receive_neighbours_surface_corrections() {
        let mut input = fixture();
        input.guidance.regions[0].vertices = (0..32).collect();
        input.guidance.regions[0].evidence = Evidence::Skin;
        let mut unknown = input.guidance.regions[0].clone();
        unknown.id = "unknown".into();
        unknown.vertices = (32..80).collect();
        unknown.evidence = Evidence::Unknown;
        input.guidance.regions.push(unknown);
        let result = positions(&solve(input.clone()).unwrap());
        for i in 32..80 {
            for k in 0..3 {
                assert!((result[i][k] - input.mesh.vertices[i][k]).abs() < 1e-10);
            }
        }
    }
    #[test]
    fn rejects_stale_unconfirmed_duplicate_and_extreme_annotations() {
        let base = fixture();
        let mut input = base.clone();
        input.mesh.vertices[0][0] += 0.001;
        assert!(
            solve(input)
                .unwrap_err()
                .to_string()
                .contains("hash_mismatch")
        );
        let mut input = base.clone();
        input.guidance.confirmed_landmarks.pop();
        assert!(
            solve(input)
                .unwrap_err()
                .to_string()
                .contains("unconfirmed")
        );
        let mut input = base.clone();
        input.guidance.regions[0].source_triangles.push(0);
        assert!(
            solve(input)
                .unwrap_err()
                .to_string()
                .contains("source_faces")
        );
        let mut input = base.clone();
        input.guidance.landmarks[0].target[0] = 1e300;
        assert!(solve(input).unwrap_err().to_string().contains("landmark"));
        let mut input = base;
        input.guidance.confirmed_landmarks.push("a".into());
        assert!(solve(input).unwrap_err().to_string().contains("confirmed"));
    }
    #[test]
    fn backward_rotation_triggers_fraction_guard() {
        let mut input = fixture();
        input.guidance.regions.clear();
        for l in &mut input.guidance.landmarks {
            l.target[0] = -l.prior[0];
            l.target[2] = -l.prior[2];
        }
        let result = solve(input).unwrap();
        assert!(result["report"]["accepted_fraction"].as_f64().unwrap() < 1.);
        assert_eq!(result["report"]["new_crossings"], 0);
        assert_eq!(result["report"]["folded_or_collapsed_triangles"], 0);
    }
    #[test]
    fn shared_envelope_routes_body_and_rejects_version_and_unknown_fields() {
        let input = fixture();
        let mut request = json!({"schema_version":1,"operation":"fit_body","input":input});
        let result: Value =
            serde_json::from_str(&crate::authoring::process(&request.to_string()).unwrap())
                .unwrap();
        assert_eq!(result["operation"], "fit_body");
        request["schema_version"] = json!(2);
        assert!(crate::authoring::process(&request.to_string()).is_err());
        request["schema_version"] = json!(1);
        request["input"]["guidance"]["unexpected"] = json!(true);
        assert!(crate::authoring::process(&request.to_string()).is_err());
    }
}
