 use crate::surface_topology::{SurfaceTopology, mean_weights};
use anyhow::{Context, Result, ensure};
use charloom_core::*;
use charloom_io::{GlbDocument, load_gltf, write_json};
use nalgebra::{Matrix4, Vector3};
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeSet, HashMap},
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct RepairConfig {
    pub clips: Vec<String>,
    pub samples_per_clip: usize,
    pub extra_times: Vec<f64>,
    pub max_edge_growth_m: f64,
    pub max_seam_gap_m: f64,
         pub max_rounds: Option<usize>,
    /// Optional additive gate, frozen after installation in an existing session.
    pub surface_contacts: Option<crate::surface_contacts::SurfaceContactConfig>,
}
impl Default for RepairConfig {
    fn default() -> Self {
        Self {
            clips: vec![],
            samples_per_clip: 33,
            extra_times: vec![],
            max_edge_growth_m: 0.03,
            max_seam_gap_m: 0.001,
            max_rounds: None,
            surface_contacts: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RepairStrategy {
    RelaxStrain,
    ExcludeInfluences,
    /// Couple render duplicates proven to share a manifold geometric seam.
    WeldSeams,
    /// Preserve reference-side ordering of triangles that cross in sampled poses.
    SeparateSurfaces,
    /// Change only the optional round budget, with a persistent audit record.
    UpdateBudget,
    Accept,
    /// Preserve the best numeric result while explicitly failing visual review.
    ReportUnresolved,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RepairRegion {
    #[serde(default)]
    pub mask: Option<crate::surface_region::SurfaceMask>,
    pub min: Vec3,
    pub max: Vec3,
    #[serde(default)]
    pub exclude_joints: Vec<String>,
    #[serde(default = "default_feather")]
    pub feather_m: f64,
}
fn default_feather() -> f64 {
    0.01
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct RepairPlan {
    /// Must equal the number of completed correction rounds in session.json.
    pub expected_round: usize,
    pub strategy: RepairStrategy,
    pub diagnosis: String,
    pub evidence: Vec<String>,
    pub regions: Vec<RepairRegion>,
    pub protect_joints: Vec<String>,
    pub iterations: usize,
    pub max_weight_change_l1: f64,
    /// Explicitly continue an unresolved review without resetting the ledger.
    pub resume_after_review: bool,
    pub round_budget: Option<usize>,
         pub continue_from_round: Option<usize>,
    pub expected_candidate_hash: Option<String>,
    /// Installs an additive surface-crossing gate; cannot relax an existing one.
    pub surface_contacts: Option<crate::surface_contacts::SurfaceContactConfig>,
    /// Additional L1 trust radius around the start of this surface-repair step.
    pub contact_weight_change_l1: f64,
    /// Smooth corrections, not the original weights, across adjacent surface vertices.
    pub contact_smoothing: f64,
}
impl Default for RepairPlan {
    fn default() -> Self {
        Self {
            expected_round: 0,
            strategy: RepairStrategy::RelaxStrain,
            diagnosis: String::new(),
            evidence: vec![],
            regions: vec![],
            protect_joints: vec![],
            iterations: 32,
            max_weight_change_l1: 0.35,
            resume_after_review: false,
            round_budget: None,
            continue_from_round: None,
            expected_candidate_hash: None,
            surface_contacts: None,
            contact_weight_change_l1: 0.05,
            contact_smoothing: 0.5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hotspot {
    pub clip: String,
    pub time_s: f64,
    pub mesh: usize,
    pub vertices: [usize; 2],
    pub rest_edge_m: f64,
    pub growth_m: f64,
    pub rest_positions: [Vec3; 2],
    pub influences: [Vec<(String, f64)>; 2],
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkinDiagnostics {
    pub status: String,
    pub max_edge_growth_m: f64,
    pub threshold_m: f64,
    pub violating_edge_samples: usize,
    pub pose_samples: usize,
    pub vertices_per_pose: usize,
    pub hotspots: Vec<Hotspot>,
    pub limits: Vec<String>,
    #[serde(default)]
    pub seam_groups: usize,
    #[serde(default)]
    pub max_seam_gap_m: f64,
    #[serde(default)]
    pub seam_threshold_m: f64,
    #[serde(default)]
    pub violating_seam_samples: usize,
    #[serde(default)]
    pub max_seam_weight_l1: f64,
    #[serde(default)]
    pub surface_contact_audit: Option<Value>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairRound {
    #[serde(default)]
    pub algorithm: String,
    pub round: usize,
    pub artifact: String,
    pub kept_as_best: bool,
    pub changed_vertices: usize,
    pub max_weight_change_l1: f64,
    pub diagnostics: SkinDiagnostics,
    #[serde(default)]
    pub artifact_hash: String,
    #[serde(default)]
    pub started_from: String,
    #[serde(default)]
    pub elapsed_seconds: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairSession {
    pub config: RepairConfig,
    pub source: PathBuf,
    pub source_hash: String,
    pub current_artifact: String,
    pub current_hash: String,
    pub initial_hash: String,
    pub status: String,
    pub rounds: Vec<RepairRound>,
    pub diagnostics: SkinDiagnostics,
    pub agent_review: Option<Value>,
    #[serde(default)]
    pub review_history: Vec<Value>,
    #[serde(default)]
    pub budget_history: Vec<Value>,
    #[serde(default)]
    pub quality_gate_history: Vec<Value>,
    pub user_visual_review: String,
    pub runtime_adoption: bool,
}

struct Edge {
    a: usize,
    b: usize,
    length: f64,
}
struct Frame {
    clip: String,
    time: f64,
    matrices: Vec<Vec<Matrix4<f64>>>,
}
struct Prepared {
    frames: Vec<Frame>,
    edges: Vec<Vec<Edge>>,
    rest: Vec<Vec<Vec3>>,
    topology: Vec<SurfaceTopology>,
    seam_threshold_m: f64,
}
#[derive(Clone)]
struct Violation {
    frame: usize,
    mesh: usize,
    edge: usize,
    growth: f64,
}

impl Prepared {
    fn new(asset: &Asset, config: &RepairConfig) -> Result<Self> {
        ensure!(
            config.max_rounds.is_none_or(|n| n > 0),
            "repair_round_budget_must_be_positive_or_null"
        );
        ensure!(
            (3..=129).contains(&config.samples_per_clip),
            "repair_sample_count_out_of_range"
        );
        ensure!(
            config.max_edge_growth_m.is_finite() && config.max_edge_growth_m > 0.0,
            "invalid_repair_threshold"
        );
        ensure!(
            config.max_seam_gap_m.is_finite()
                && config.max_seam_gap_m > 0.0
                && config.max_seam_gap_m <= 0.005,
            "invalid_seam_threshold"
        );
        ensure!(!asset.skins.is_empty(), "repair_requires_existing_skin");
        ensure!(
            asset.meshes.iter().all(|m| m.morph_targets.is_empty()),
            "skin_repair_morphs_not_supported"
        );
        ensure!(
            asset
                .meshes
                .iter()
                .filter(|m| m.skin.is_some())
                .all(|m| m.influences.iter().all(|r| !r.is_empty() && r.len() <= 4)),
            "repair_requires_at_most_four_influences"
        );
        ensure!(
            config.extra_times.len() <= 64
                && config
                    .extra_times
                    .iter()
                    .all(|t| t.is_finite() && *t >= 0.0),
            "invalid_repair_extra_times"
        );
        let names = if config.clips.is_empty() {
            asset
                .clips
                .iter()
                .map(|c| c.name.clone())
                .collect::<Vec<_>>()
        } else {
            config.clips.clone()
        };
        ensure!(
            !names.is_empty() && names.len() <= 24,
            "invalid_repair_clips"
        );
        let mut frames = Vec::new();
        for name in names {
            let matching: Vec<_> = asset.clips.iter().filter(|c| c.name == name).collect();
            ensure!(matching.len() == 1, "repair_clip_must_be_unique: {name}");
            let clip = matching[0];
            let [start, end] = clip.time_range();
            let mut times: Vec<_> = (0..config.samples_per_clip)
                .map(|i| start + (end - start) * i as f64 / (config.samples_per_clip - 1) as f64)
                .collect();
            times.extend(
                config
                    .extra_times
                    .iter()
                    .copied()
                    .filter(|t| *t >= start && *t <= end),
            );
            times.sort_by(f64::total_cmp);
            times.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
            for time in times {
                let world = asset
                    .skeleton
                    .world_matrices(&clip.sample(&asset.skeleton, time)?)?;
                let matrices = asset
                    .skins
                    .iter()
                    .map(|s| {
                        let mut rows = vec![Matrix4::identity(); world.len()];
                        for (&joint, ibm) in s.joints.iter().zip(&s.inverse_bind_matrices) {
                            rows[joint] = world[joint] * matrix_from(*ibm);
                        }
                        rows
                    })
                    .collect();
                frames.push(Frame {
                    clip: name.clone(),
                    time,
                    matrices,
                });
            }
        }
        ensure!(frames.len() <= 512, "repair_pose_budget_exceeded");
        let rest = asset.world_meshes()?;
        let edges = asset
            .meshes
            .iter()
            .enumerate()
            .map(|(m, mesh)| {
                if mesh.skin.is_none() {
                    return vec![];
                }
                let mut pairs = Vec::with_capacity(mesh.triangles.len() * 3);
                for &[a, b, c] in &mesh.triangles {
                    for (a, b) in [(a, b), (b, c), (c, a)] {
                        if a != b {
                            pairs.push((a.min(b) as usize, a.max(b) as usize));
                        }
                    }
                }
                pairs.sort_unstable();
                pairs.dedup();
                pairs
                    .into_iter()
                    .map(|(a, b)| Edge {
                        a,
                        b,
                        length: (v3(rest[m][a]) - v3(rest[m][b])).norm(),
                    })
                    .collect()
            })
            .collect();
        Ok(Self {
            frames,
            edges,
            rest,
            topology: asset.meshes.iter().map(SurfaceTopology::new).collect(),
            seam_threshold_m: config.max_seam_gap_m,
        })
    }

    fn scan(&self, asset: &Asset, threshold: f64) -> (SkinDiagnostics, Vec<Violation>) {
        let results: Vec<_> = self
            .frames
            .par_iter()
            .enumerate()
            .map(|(f, frame)| {
                let mut maximum: f64 = 0.0;
                let mut violations = Vec::new();
                let mut count = 0;
                let mut seam_maximum: f64 = 0.0;
                let mut seam_count = 0;
                for (m, mesh) in asset.meshes.iter().enumerate() {
                    let Some(skin) = mesh.skin else {
                        continue;
                    };
                    let matrices = &frame.matrices[skin];
                    let points: Vec<_> = mesh
                        .positions
                        .iter()
                        .zip(&mesh.influences)
                        .map(|(&p, w)| skinned(p, w, matrices))
                        .collect();
                    for (e, edge) in self.edges[m].iter().enumerate() {
                        let growth = (points[edge.a] - points[edge.b]).norm() - edge.length;
                        maximum = maximum.max(growth);
                        if growth > threshold {
                            count += 1;
                            violations.push(Violation {
                                frame: f,
                                mesh: m,
                                edge: e,
                                growth,
                            });
                        }
                    }
                    for group in &self.topology[m].groups {
                        for &i in &group[1..] {
                            let gap = (points[i] - points[group[0]]).norm();
                            seam_maximum = seam_maximum.max(gap);
                            if gap > self.seam_threshold_m {
                                seam_count += 1;
                            }
                        }
                    }
                }
                violations.sort_by(|a, b| b.growth.total_cmp(&a.growth));
                violations.truncate(20_000);
                (maximum, count, violations, seam_maximum, seam_count)
            })
            .collect();
        let maximum = results.iter().map(|r| r.0).fold(0.0, f64::max);
        let count = results.iter().map(|r| r.1).sum();
        let seam_maximum = results.iter().map(|r| r.3).fold(0.0, f64::max);
        let seam_count = results.iter().map(|r| r.4).sum();
        let mut violations: Vec<_> = results.into_iter().flat_map(|r| r.2).collect();
        violations.sort_by(|a, b| {
            b.growth
                .total_cmp(&a.growth)
                .then(a.frame.cmp(&b.frame))
                .then(a.mesh.cmp(&b.mesh))
                .then(a.edge.cmp(&b.edge))
        });
        violations.truncate(100_000);
        let hotspots = violations
            .iter()
            .take(48)
            .map(|v| {
                let e = &self.edges[v.mesh][v.edge];
                let frame = &self.frames[v.frame];
                Hotspot {
                    clip: frame.clip.clone(),
                    time_s: frame.time,
                    mesh: v.mesh,
                    vertices: [e.a, e.b],
                    rest_edge_m: e.length,
                    growth_m: v.growth,
                    rest_positions: [self.rest[v.mesh][e.a], self.rest[v.mesh][e.b]],
                    influences: [e.a, e.b].map(|i| {
                        asset.meshes[v.mesh].influences[i]
                            .iter()
                            .map(|w| (asset.skeleton.joints[w.joint].name.clone(), w.weight))
                            .collect()
                    }),
                }
            })
            .collect();
        (SkinDiagnostics { status: if count == 0 && seam_count == 0 { "passed" } else { "failed" }.into(), max_edge_growth_m: maximum, threshold_m: threshold, violating_edge_samples: count,
            pose_samples: self.frames.len(), vertices_per_pose: asset.meshes.iter().filter(|m| m.skin.is_some()).map(|m| m.positions.len()).sum(), hotspots,
            seam_groups:self.topology.iter().zip(&asset.meshes).filter(|(_,m)|m.skin.is_some()).map(|(t,_)|t.groups.len()).sum(), max_seam_gap_m:seam_maximum,seam_threshold_m:self.seam_threshold_m,violating_seam_samples:seam_count,
            max_seam_weight_l1:self.topology.iter().zip(&asset.meshes).filter(|(_,m)|m.skin.is_some()).map(|(t,m)|t.weight_l1_max(&m.influences)).fold(0.0,f64::max),surface_contact_audit:None,
            limits: vec!["Sampled strain and seam continuity are not anatomy, collision, grasp, or visual approval.".into(),"Joint transforms, animation tracks, geometry and materials are held fixed; seam repair can share the existing neighboring joint palettes.".into()] }, violations)
    }

    fn local_violations(
        &self,
        asset: &Asset,
        tracked: &[BTreeSet<usize>],
        threshold: f64,
    ) -> Vec<Violation> {
        self.frames
            .par_iter()
            .enumerate()
            .flat_map_iter(|(f, frame)| {
                let mut violations = Vec::new();
                for (m, edges) in tracked.iter().enumerate() {
                    let mesh = &asset.meshes[m];
                    let Some(skin) = mesh.skin else {
                        continue;
                    };
                    let matrices = &frame.matrices[skin];
                    for &e in edges {
                        let edge = &self.edges[m][e];
                        let a = skinned(mesh.positions[edge.a], &mesh.influences[edge.a], matrices);
                        let b = skinned(mesh.positions[edge.b], &mesh.influences[edge.b], matrices);
                        let growth = (a - b).norm() - edge.length;
                        if growth > threshold {
                            violations.push(Violation {
                                frame: f,
                                mesh: m,
                                edge: e,
                                growth,
                            });
                        }
                    }
                }
                violations
            })
            .collect()
    }
}

fn skinned(p: Vec3, row: &[Influence], matrices: &[Matrix4<f64>]) -> Vector3<f64> {
    row.iter()
        .map(|w| v3(transform_point(&matrices[w.joint], p)) * w.weight)
        .sum()
}
pub(crate) fn simplex(values: &mut [f64]) {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let mut sum = 0.0;
    let mut theta = 0.0;
    for (i, &v) in sorted.iter().enumerate() {
        sum += v;
        let t = (sum - 1.0) / (i + 1) as f64;
        if v > t {
            theta = t;
        }
    }
    for v in values {
        *v = (*v - theta).max(0.0);
    }
}
fn region_factor(p: Vec3, mesh: usize, vertex: usize, region: &RepairRegion) -> f64 {
    if region
        .mask
        .as_ref()
        .is_some_and(|m| m.mesh != mesh || m.vertices.binary_search(&vertex).is_err())
    {
        return 0.0;
    }
    let d = (0..3)
        .map(|i| (p[i] - region.min[i]).min(region.max[i] - p[i]))
        .fold(f64::INFINITY, f64::min);
    let t = (d / region.feather_m).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
fn l1(a: &[Influence], b: &[Influence]) -> f64 {
    a.iter()
        .map(|w| {
            (w.weight
                - b.iter()
                    .find(|v| v.joint == w.joint)
                    .map_or(0.0, |v| v.weight))
            .abs()
        })
        .sum::<f64>()
        + b.iter()
            .filter(|w| !a.iter().any(|v| v.joint == w.joint))
            .map(|w| w.weight)
            .sum::<f64>()
}

fn add_contact_gate(
    asset: &Asset,
    config: &RepairConfig,
    diagnostics: &mut SkinDiagnostics,
) -> Result<()> {
    if let Some(check) = &config.surface_contacts {
        let mut check = check.clone();
        check.maximum_reported_pairs = 0;
        let audit = crate::surface_contacts::audit_asset(asset, &check)?;
        if audit["samples"].as_array().unwrap().iter().any(|s| {
            s["new_crossing_pairs"].as_u64().unwrap() > 0
                || s["collapsed_triangles"].as_u64().unwrap() > 0
        }) {
            diagnostics.status = "failed".into();
        }
        diagnostics.surface_contact_audit = Some(audit);
    }
    Ok(())
}
fn contact_count(diagnostics: &SkinDiagnostics) -> usize {
    diagnostics.surface_contact_audit.as_ref().map_or(0, |a| {
        a["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["new_crossing_pairs"].as_u64().unwrap() as usize)
            .sum()
    })
}
fn collapsed_count(diagnostics: &SkinDiagnostics) -> usize {
    diagnostics.surface_contact_audit.as_ref().map_or(0, |a| {
        a["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["collapsed_triangles"].as_u64().unwrap_or(0) as usize)
            .sum()
    })
}

struct ContactConstraint {
    frame: usize,
    mesh: usize,
    vertex: usize,
    other_mesh: usize,
    face: [usize; 3],
    side: f64,
    gap: f64,
}
fn contact_constraints(
    asset: &Asset,
    prepared: &Prepared,
    check: &crate::surface_contacts::SurfaceContactConfig,
) -> Result<Vec<ContactConstraint>> {
    ensure!(
        !check.times.is_empty() && check.times.len() <= 16,
        "repair_contact_pose_budget_exceeded"
    );
    let mut selected = BTreeSet::new();
    for mask in &check.regions {
        crate::surface_region::validate_mask(mask, &asset.meshes)?;
        for (t, tri) in asset.meshes[mask.mesh].triangles.iter().enumerate() {
            if tri
                .iter()
                .any(|v| mask.vertices.binary_search(&(*v as usize)).is_ok())
            {
                selected.insert([mask.mesh, t]);
            }
        }
    }
    let baseline = crate::surface_query::SurfaceQuery::new(&asset.meshes, &prepared.rest)
        .crossing_pairs(&selected, check.epsilon_m);
    let mut constraints = Vec::new();
    let mut unique = BTreeSet::new();
    for &time in &check.times {
        let frame = prepared
            .frames
            .iter()
            .position(|f| Some(&f.clip) == check.clip.as_ref() && (f.time - time).abs() < 1e-6)
            .context("contact_time_must_match_frozen_repair_pose")?;
        let clip = asset
            .clips
            .iter()
            .find(|c| Some(&c.name) == check.clip.as_ref())
            .context("contact_clip_missing")?;
        let positions = asset.deformed_positions(
            &clip.sample(&asset.skeleton, time)?,
            &clip.sample_morphs(time)?,
        )?;
        let pairs = crate::surface_query::SurfaceQuery::new(&asset.meshes, &positions)
            .crossing_pairs(&selected, check.epsilon_m);
        for pair in pairs.difference(&baseline) {
            for (a, b) in [(pair[0], pair[1]), (pair[1], pair[0])] {
                let (mesh, other_mesh) = (&asset.meshes[a[0]], &asset.meshes[b[0]]);
                if mesh.skin.is_none() || other_mesh.skin.is_none() {
                    continue;
                }
                let face = other_mesh.triangles[b[1]].map(|v| v as usize);
                let ps = face.map(|v| v3(prepared.rest[b[0]][v]));
                let normal = (ps[1] - ps[0]).cross(&(ps[2] - ps[0]));
                if normal.norm() < 1e-12 {
                    continue;
                }
                let normal = normal.normalize();
                let vertices = mesh.triangles[a[1]].map(|v| v as usize);
                let distances = vertices.map(|v| (v3(prepared.rest[a[0]][v]) - ps[0]).dot(&normal));
                let side = if distances.iter().all(|d| *d > check.epsilon_m) {
                    1.0
                } else if distances.iter().all(|d| *d < -check.epsilon_m) {
                    -1.0
                } else {
                    continue;
                };
                for (vertex, distance) in vertices.into_iter().zip(distances) {
                    if unique.insert([frame, a[0], vertex, b[0], b[1]]) {
                        constraints.push(ContactConstraint {
                            frame,
                            mesh: a[0],
                            vertex,
                            other_mesh: b[0],
                            face,
                            side,
                            gap: (distance.abs() * 0.25).min(0.0005),
                        });
                    }
                }
            }
        }
    }
    Ok(constraints)
}

struct WeightProjection<'a> {
    baseline: &'a Asset,
    prepared: &'a Prepared,
    editable: &'a [Vec<bool>],
    maximum: f64,
    origin: &'a [Vec<Vec<Influence>>],
    local_maximum: f64,
}
impl WeightProjection<'_> {
    fn apply(
        &self,
        asset: &mut Asset,
        mesh: usize,
        vertex: usize,
        gradient: &[f64],
        step: f64,
    ) -> Vec<usize> {
        let topology = &self.prepared.topology[mesh];
        let members = topology.group_index[vertex]
            .map(|g| topology.groups[g].clone())
            .unwrap_or_else(|| vec![vertex]);
        if !members.iter().all(|&v| self.editable[mesh][v]) {
            return vec![];
        }
        let row = &asset.meshes[mesh].influences[vertex];
        let mut values: Vec<_> = row
            .iter()
            .zip(gradient)
            .map(|(w, g)| w.weight - step * g)
            .collect();
        simplex(&mut values);
        let valid = |ratio: f64| {
            let candidate: Vec<_> = row
                .iter()
                .zip(&values)
                .map(|(w, v)| Influence {
                    joint: w.joint,
                    weight: w.weight + (v - w.weight) * ratio,
                })
                .collect();
            members.iter().all(|&v| {
                l1(&candidate, &self.baseline.meshes[mesh].influences[v]) <= self.maximum + 1e-9
                    && l1(&candidate, &self.origin[mesh][v]) <= self.local_maximum + 1e-9
            })
        };
        let mut ratio = 1.0;
        if !valid(ratio) {
            let (mut low, mut high) = (0.0, 1.0);
            for _ in 0..24 {
                let mid = (low + high) * 0.5;
                if valid(mid) {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            ratio = low;
        }
        let mut updated = row.clone();
        for (w, v) in updated.iter_mut().zip(values) {
            w.weight += (v - w.weight) * ratio;
        }
        for &v in &members {
            asset.meshes[mesh].influences[v] = updated.clone();
        }
        members
    }
}

fn smooth_contact_deltas(
    asset: &mut Asset,
    vertices: &BTreeSet<(usize, usize)>,
    projection: &WeightProjection<'_>,
    strength: f64,
) -> Vec<(usize, usize)> {
    let updates: Vec<_> = vertices
        .iter()
        .map(|&(m, v)| {
            let row = &asset.meshes[m].influences[v];
            let weight = |weights: &[Influence], joint| {
                weights
                    .iter()
                    .find(|w| w.joint == joint)
                    .map_or(0.0, |w| w.weight)
            };
            let adjacent = &projection.prepared.topology[m].adjacency[v];
            let gradient: Vec<_> = row
                .iter()
                .map(|w| {
                    let own_delta = w.weight - weight(&projection.origin[m][v], w.joint);
                    let mean_delta = (own_delta
                        + adjacent
                            .iter()
                            .map(|&n| {
                                weight(&asset.meshes[m].influences[n], w.joint)
                                    - weight(&projection.origin[m][n], w.joint)
                            })
                            .sum::<f64>())
                        / (adjacent.len() + 1) as f64;
                    (own_delta - mean_delta) * strength
                })
                .collect();
            (m, v, gradient)
        })
        .collect();
    updates
        .into_iter()
        .flat_map(|(m, v, g)| {
            projection
                .apply(asset, m, v, &g, 1.0)
                .into_iter()
                .map(move |v| (m, v))
        })
        .collect()
}
fn separate_contacts(
    asset: &mut Asset,
    constraints: &[ContactConstraint],
    projection: &WeightProjection<'_>,
) -> Vec<(usize, usize)> {
    let mut changed = Vec::new();
    for c in constraints {
        let frame = &projection.prepared.frames[c.frame];
        let mesh = &asset.meshes[c.mesh];
        let other = &asset.meshes[c.other_mesh];
        let matrices = &frame.matrices[mesh.skin.unwrap()];
        let other_matrices = &frame.matrices[other.skin.unwrap()];
        let p = skinned(
            mesh.positions[c.vertex],
            &mesh.influences[c.vertex],
            matrices,
        );
        let q = c
            .face
            .map(|v| skinned(other.positions[v], &other.influences[v], other_matrices));
        let normal = (q[1] - q[0]).cross(&(q[2] - q[0]));
        if normal.norm() < 1e-12 {
            continue;
        }
        let normal = normal.normalize() * c.side;
        let excess = c.gap - (p - (q[0] + q[1] + q[2]) / 3.0).dot(&normal);
        if excess <= 1e-7 {
            continue;
        }
        let endpoints = [
            (c.mesh, c.vertex, 1.0),
            (c.other_mesh, c.face[0], -1.0 / 3.0),
            (c.other_mesh, c.face[1], -1.0 / 3.0),
            (c.other_mesh, c.face[2], -1.0 / 3.0),
        ];
        let gradients: Vec<Vec<f64>> = endpoints
            .iter()
            .map(|&(m, v, scale)| {
                let mesh = &asset.meshes[m];
                let row = &mesh.influences[v];
                if !projection.editable[m][v] {
                    return vec![0.0; row.len()];
                }
                let matrices = &frame.matrices[mesh.skin.unwrap()];
                let mut values: Vec<_> = row
                    .iter()
                    .map(|w| {
                        normal.dot(&v3(transform_point(&matrices[w.joint], mesh.positions[v])))
                            * scale
                    })
                    .collect();
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                for value in &mut values {
                    *value -= mean;
                }
                values
            })
            .collect();
        let norm = gradients.iter().flatten().map(|g| g * g).sum::<f64>();
        if norm < 1e-14 {
            continue;
        }
        let step = -0.65 * excess / norm;
        for ((m, v, _), gradient) in endpoints.into_iter().zip(gradients) {
            changed.extend(
                projection
                    .apply(asset, m, v, &gradient, step)
                    .into_iter()
                    .map(|v| (m, v)),
            );
        }
    }
    changed
}

fn correct(
    asset: &mut Asset,
    baseline: &Asset,
    prepared: &Prepared,
    config: &RepairConfig,
    plan: &RepairPlan,
) -> Result<(usize, f64)> {
    ensure!(
        (1..=64).contains(&plan.iterations),
        "invalid_repair_iterations"
    );
    ensure!(
        plan.max_weight_change_l1.is_finite()
            && plan.max_weight_change_l1 > 0.0
            && plan.max_weight_change_l1 <= 0.5,
        "repair_weight_change_budget_exceeded"
    );
    let ids = |names: &[String]| -> Result<Vec<usize>> {
        names
            .iter()
            .map(|name| {
                let matches: Vec<_> = asset
                    .skeleton
                    .joints
                    .iter()
                    .enumerate()
                    .filter(|(_, j)| &j.name == name)
                    .map(|(i, _)| i)
                    .collect();
                ensure!(matches.len() == 1, "repair_joint_must_be_unique: {name}");
                Ok(matches[0])
            })
            .collect()
    };
    let protected = ids(&plan.protect_joints)?;
    let excluded: Vec<_> = plan
        .regions
        .iter()
        .map(|r| ids(&r.exclude_joints))
        .collect::<Result<_>>()?;
    for r in &plan.regions {
        if let Some(mask) = &r.mask {
            crate::surface_region::validate_mask(mask, &asset.meshes)?;
        }
        ensure!(
            r.feather_m.is_finite()
                && r.feather_m > 0.0
                && (0..3)
                    .all(|i| r.min[i].is_finite() && r.max[i].is_finite() && r.min[i] < r.max[i]),
            "invalid_repair_region"
        );
    }
    let before: Vec<_> = asset.meshes.iter().map(|m| m.influences.clone()).collect();
    let editable: Vec<Vec<_>> = asset
        .meshes
        .iter()
        .enumerate()
        .map(|(m, mesh)| {
            mesh.influences
                .iter()
                .enumerate()
                .map(|(i, row)| {
                    (row.len() > 1
                        || matches!(
                            plan.strategy,
                            RepairStrategy::WeldSeams | RepairStrategy::SeparateSurfaces
                        ))
                        && row
                            .iter()
                            .filter(|w| protected.contains(&w.joint))
                            .map(|w| w.weight)
                            .sum::<f64>()
                            < 0.5
                        && (plan.regions.is_empty()
                            || plan
                                .regions
                                .iter()
                                .any(|r| region_factor(prepared.rest[m][i], m, i, r) > 0.0))
                })
                .collect()
        })
        .collect();
    if matches!(plan.strategy, RepairStrategy::ExcludeInfluences) {
        ensure!(
            !plan.regions.is_empty() && excluded.iter().any(|e| !e.is_empty()),
            "exclusion_requires_regions_and_joints"
        );
        for (m, mesh) in asset.meshes.iter_mut().enumerate() {
            for (i, row) in mesh.influences.iter_mut().enumerate() {
                if !editable[m][i] {
                    continue;
                }
                for (r, joints) in plan.regions.iter().zip(&excluded) {
                    let f = region_factor(prepared.rest[m][i], m, i, r);
                    for w in row.iter_mut() {
                        if joints.contains(&w.joint) {
                            w.weight *= 1.0 - f;
                        }
                    }
                }
                let sum: f64 = row.iter().map(|w| w.weight).sum();
                ensure!(
                    sum > 1e-10,
                    "exclusion_removes_all_influences: mesh {m}, vertex {i}"
                );
                for w in row.iter_mut() {
                    w.weight /= sum;
                }
                ensure!(
                    l1(row, &baseline.meshes[m].influences[i]) <= plan.max_weight_change_l1 + 1e-9,
                    "exclusion_exceeds_weight_change_budget: mesh {m}, vertex {i}"
                );
            }
        }
    }
    let couple_seams = matches!(
        plan.strategy,
        RepairStrategy::WeldSeams | RepairStrategy::SeparateSurfaces
    );
    if couple_seams {
        ensure!(
            excluded.iter().all(Vec::is_empty),
            "weld_seams_exclusions_require_prevalidated_binding_regions"
        );
        for (m, mesh) in asset.meshes.iter_mut().enumerate() {
            if mesh.skin.is_none() {
                continue;
            }
            for group in &prepared.topology[m].groups {
                if !group.iter().all(|&i| editable[m][i]) {
                    continue;
                }
                let row = mean_weights(group.iter().map(|&i| &mesh.influences[i]), 4);
                if matches!(plan.strategy, RepairStrategy::SeparateSurfaces) {
                    ensure!(
                        group
                            .iter()
                            .all(|&i| l1(&row, &before[m][i])
                                <= plan.contact_weight_change_l1 + 1e-9),
                        "surface_separation_requires_seam_consistent_input_within_local_budget"
                    );
                }
                ensure!(
                    group
                        .iter()
                        .all(|&i| l1(&row, &baseline.meshes[m].influences[i])
                            <= plan.max_weight_change_l1 + 1e-9),
                    "seam_consensus_exceeds_global_weight_budget: mesh {m}, vertex {}",
                    group[0]
                );
                for &i in group {
                    mesh.influences[i] = row.clone();
                }
            }
        }
    }
    let target = config.max_edge_growth_m * 0.75;
    let (_, mut constraints) = prepared.scan(asset, target);
    let mut tracked = vec![BTreeSet::new(); asset.meshes.len()];
    let mut known: BTreeSet<_> = constraints
        .iter()
        .map(|c| (c.frame, c.mesh, c.edge))
        .collect();
    let mut incident: Vec<Vec<Vec<usize>>> = asset
        .meshes
        .iter()
        .map(|m| vec![vec![]; m.positions.len()])
        .collect();
    for (m, edges) in prepared.edges.iter().enumerate() {
        for (e, edge) in edges.iter().enumerate() {
            incident[m][edge.a].push(e);
            incident[m][edge.b].push(e);
        }
    }
     
    let relax_iterations = if matches!(
        plan.strategy,
        RepairStrategy::RelaxStrain | RepairStrategy::WeldSeams | RepairStrategy::SeparateSurfaces
    ) {
        plan.iterations
    } else {
        0
    };
    let contacts = if matches!(plan.strategy, RepairStrategy::SeparateSurfaces) {
        ensure!(
            plan.contact_weight_change_l1.is_finite()
                && plan.contact_weight_change_l1 > 0.0
                && plan.contact_weight_change_l1 <= 0.5
                && plan.contact_smoothing.is_finite()
                && (0.0..=1.0).contains(&plan.contact_smoothing),
            "invalid_contact_repair_trust_parameters"
        );
        contact_constraints(
            asset,
            prepared,
            config
                .surface_contacts
                .as_ref()
                .context("surface_separation_requires_contact_gate")?,
        )?
    } else {
        vec![]
    };
    let projection = WeightProjection {
        baseline,
        prepared,
        editable: &editable,
        maximum: plan.max_weight_change_l1,
        origin: &before,
        local_maximum: plan.contact_weight_change_l1,
    };
    let mut contact_vertices = BTreeSet::new();
    for c in &contacts {
        for (m, v) in
            std::iter::once((c.mesh, c.vertex)).chain(c.face.into_iter().map(|v| (c.other_mesh, v)))
        {
            contact_vertices.insert((m, prepared.topology[m].representative[v]));
            contact_vertices.extend(prepared.topology[m].adjacency[v].iter().map(|&n| (m, n)));
        }
    }
    for iteration in 0..relax_iterations {
        for (mesh, vertex) in separate_contacts(asset, &contacts, &projection) {
            tracked[mesh].extend(incident[mesh][vertex].iter().copied());
        }
        if !contacts.is_empty() {
            for (mesh, vertex) in smooth_contact_deltas(
                asset,
                &contact_vertices,
                &projection,
                plan.contact_smoothing,
            ) {
                tracked[mesh].extend(incident[mesh][vertex].iter().copied());
            }
        }
        // Correcting a vertex also changes every adjacent edge. Discover new
        // constraints in that expanding neighborhood across ALL frozen poses,
        // not just the initially failing edges or the currently worst pose.
        if iteration > 0 && iteration % 4 == 0 {
            let mut additional = prepared.local_violations(asset, &tracked, target);
            additional.sort_by(|a, b| {
                b.growth
                    .total_cmp(&a.growth)
                    .then(a.frame.cmp(&b.frame))
                    .then(a.mesh.cmp(&b.mesh))
                    .then(a.edge.cmp(&b.edge))
            });
            for c in additional {
                if known.insert((c.frame, c.mesh, c.edge)) {
                    constraints.push(c);
                }
            }
        }
        for c in &constraints {
            let mesh = &asset.meshes[c.mesh];
            let e = &prepared.edges[c.mesh][c.edge];
            let matrices = &prepared.frames[c.frame].matrices[mesh.skin.unwrap()];
            let qa = skinned(mesh.positions[e.a], &mesh.influences[e.a], matrices);
            let qb = skinned(mesh.positions[e.b], &mesh.influences[e.b], matrices);
            let delta = qa - qb;
            let length = delta.norm();
            let excess = length - e.length - target;
            if excess <= 1e-7 || length < 1e-10 {
                continue;
            }
            let direction = delta / length;
            let gradients: [Vec<f64>; 2] = [(e.a, 1.0), (e.b, -1.0)].map(|(i, sign)| {
                if !editable[c.mesh][i] {
                    return vec![0.0; mesh.influences[i].len()];
                }
                let mut g: Vec<_> = mesh.influences[i]
                    .iter()
                    .map(|w| {
                        direction.dot(&v3(transform_point(&matrices[w.joint], mesh.positions[i])))
                            * sign
                    })
                    .collect();
                let mean = g.iter().sum::<f64>() / g.len() as f64;
                for x in &mut g {
                    *x -= mean;
                }
                g
            });
            let norm: f64 = gradients.iter().flatten().map(|g| g * g).sum();
            if norm < 1e-14 {
                continue;
            }
            let step = 0.8 * excess / norm;
            for (i, g) in [e.a, e.b].into_iter().zip(gradients) {
                if !editable[c.mesh][i] {
                    continue;
                }
                let members = if couple_seams {
                    prepared.topology[c.mesh].group_index[i]
                        .map(|g| prepared.topology[c.mesh].groups[g].clone())
                        .unwrap_or_else(|| vec![i])
                } else {
                    vec![i]
                };
                if !members.iter().all(|&v| editable[c.mesh][v]) {
                    continue;
                }
                let row = &asset.meshes[c.mesh].influences[i];
                for &v in &members {
                    tracked[c.mesh].extend(incident[c.mesh][v].iter().copied());
                }
                let mut values: Vec<_> = row
                    .iter()
                    .zip(g)
                    .map(|(w, g)| w.weight - step * g)
                    .collect();
                simplex(&mut values);
                let previous: Vec<_> = row.iter().map(|w| w.weight).collect();
                let distance = |ratio: f64| -> f64 {
                    let candidate: Vec<_> = row
                        .iter()
                        .zip(&values)
                        .zip(&previous)
                        .map(|((w, &v), &p)| Influence {
                            joint: w.joint,
                            weight: p + (v - p) * ratio,
                        })
                        .collect();
                    members
                        .iter()
                        .map(|&v| {
                            let global = l1(&candidate, &baseline.meshes[c.mesh].influences[v])
                                / plan.max_weight_change_l1;
                            if matches!(plan.strategy, RepairStrategy::SeparateSurfaces) {
                                global.max(
                                    l1(&candidate, &before[c.mesh][v])
                                        / plan.contact_weight_change_l1,
                                )
                            } else {
                                global
                            }
                        })
                        .fold(0.0, f64::max)
                };
                let mut ratio = 1.0;
                if distance(ratio) > 1.0 {
                    let (mut low, mut high) = (0.0, 1.0);
                    for _ in 0..24 {
                        let mid = (low + high) * 0.5;
                        if distance(mid) <= 1.0 {
                            low = mid;
                        } else {
                            high = mid;
                        }
                    }
                    ratio = low;
                }
                let mut updated = row.clone();
                for ((w, v), p) in updated.iter_mut().zip(values).zip(previous) {
                    w.weight = p + (v - p) * ratio;
                }
                for v in members {
                    asset.meshes[c.mesh].influences[v] = updated.clone();
                }
            }
        }
    }
    let mut changed = 0;
    let mut maximum: f64 = 0.0;
    for (m, mesh) in asset.meshes.iter().enumerate() {
        for (i, row) in mesh.influences.iter().enumerate() {
            if l1(row, &before[m][i]) > 1e-7 {
                changed += 1;
            }
            maximum = maximum.max(l1(row, &baseline.meshes[m].influences[i]));
        }
    }
    Ok((changed, maximum))
}

pub(crate) fn write_weights(document: &mut GlbDocument, asset: &Asset) -> Result<()> {
    let mut mesh_copies = HashMap::new();
    for mesh in &asset.meshes {
        let Some(skin_index) = mesh.skin else {
            continue;
        };
        let skin = &asset.skins[skin_index];
        let map: HashMap<_, _> = skin
            .joints
            .iter()
            .enumerate()
            .map(|(i, &j)| (j, i))
            .collect();
        let mut joints = Vec::with_capacity(mesh.positions.len() * 8);
        let mut weights = Vec::with_capacity(mesh.positions.len() * 16);
        for row in &mesh.influences {
            for k in 0..4 {
                joints.extend_from_slice(
                    &(row.get(k).map_or(0, |w| map[&w.joint]) as u16).to_le_bytes(),
                );
            }
            for k in 0..4 {
                weights.extend_from_slice(
                    &(row.get(k).map_or(0.0, |w| w.weight) as f32).to_le_bytes(),
                );
            }
        }
        let j = document.append_accessor(
            &joints,
            5123,
            "VEC4",
            mesh.positions.len(),
            Some(34962),
            None,
        );
        let w = document.append_accessor(
            &weights,
            5126,
            "VEC4",
            mesh.positions.len(),
            Some(34962),
            None,
        );
        let mesh_index = if let Some(&index) = mesh_copies.get(&mesh.node) {
            index
        } else {
            let original = document.json["nodes"][mesh.node]["mesh"]
                .as_u64()
                .context("missing_mesh_node")? as usize;
            let copy = document.json["meshes"][original].clone();
            let index = document.array("meshes").len();
            document.array("meshes").push(copy);
            document.json["nodes"][mesh.node]["mesh"] = json!(index);
            mesh_copies.insert(mesh.node, index);
            index
        };
        let attrs = document.json["meshes"][mesh_index]["primitives"][mesh.primitive]["attributes"]
            .as_object_mut()
            .context("missing_mesh_attributes")?;
        attrs.insert("JOINTS_0".into(), json!(j));
        attrs.insert("WEIGHTS_0".into(), json!(w));
    }
    Ok(())
}

pub fn start(input: &Path, config: &RepairConfig, out: &Path) -> Result<Value> {
    let source = load_gltf(input)?;
    let prepared = Prepared::new(&source.asset, config)?;
    let (mut diagnostics, _) = prepared.scan(&source.asset, config.max_edge_growth_m);
    add_contact_gate(&source.asset, config, &mut diagnostics)?;
    let mut state = RepairSession {
        config: config.clone(),
        source: fs::canonicalize(input)?,
        source_hash: source.asset.source_hash.clone(),
        current_artifact: "initial.glb".into(),
        current_hash: source.asset.source_hash.clone(),
        initial_hash: String::new(),
        status: if diagnostics.status == "passed" {
            "awaiting_agent_visual_review"
        } else {
            "awaiting_agent_repair"
        }
        .into(),
        rounds: vec![],
        diagnostics,
        agent_review: None,
        review_history: vec![],
        budget_history: vec![],
        quality_gate_history: vec![],
        user_visual_review: "pending".into(),
        runtime_adoption: false,
    };
    crate::pipeline::publish_bundle(out, |dir| {
        source.document.save(&dir.join("initial.glb"))?;
        state.current_hash = load_gltf(dir.join("initial.glb"))?.asset.source_hash;
        state.initial_hash = state.current_hash.clone();
        write_json(&dir.join("session.json"), &state)
    })?;
    Ok(json!({"session":fs::canonicalize(out)?,"state":state}))
}

pub fn inspect(session: &Path) -> Result<Value> {
    let state: RepairSession = serde_json::from_slice(&fs::read(session.join("session.json"))?)?;
    Ok(json!({"session":fs::canonicalize(session)?,"state":state}))
}
struct Lock(PathBuf);
impl Drop for Lock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn atomic_state(path: &Path, state: &RepairSession) -> Result<()> {
    let mut temp = tempfile::NamedTempFile::new_in(path.parent().unwrap())?;
    serde_json::to_writer_pretty(&mut temp, state)?;
    temp.as_file().sync_all()?;
    temp.persist(path)?;
    Ok(())
}

pub fn step(session: &Path, plan: &RepairPlan) -> Result<Value> {
    let started = std::time::Instant::now();
    let session = fs::canonicalize(session)?;
    let lock_path = session.join(".repair-lock");
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock_path)
        .context("repair_session_busy")?;
    let _lock = Lock(lock_path);
    let state_path = session.join("session.json");
    let mut state: RepairSession = serde_json::from_slice(&fs::read(&state_path)?)?;
    ensure!(
        plan.expected_round == state.rounds.len(),
        "stale_repair_plan"
    );
    ensure!(
        !plan.diagnosis.trim().is_empty() && !plan.evidence.is_empty(),
        "repair_requires_diagnosis_and_evidence"
    );
    ensure!(
        state.current_artifact == "initial.glb"
            || state
                .rounds
                .iter()
                .any(|r| r.artifact == state.current_artifact),
        "invalid_repair_artifact"
    );
    let mut current = load_gltf(session.join(&state.current_artifact))?;
    ensure!(
        current.asset.source_hash == state.current_hash,
        "repair_candidate_changed"
    );
    if matches!(plan.strategy, RepairStrategy::UpdateBudget) {
        ensure!(
            plan.round_budget.is_none_or(|n| n > state.rounds.len()),
            "new_budget_must_exceed_completed_rounds"
        );
        state.budget_history.push(json!({"after_round":state.rounds.len(),"previous":state.config.max_rounds,"updated":plan.round_budget,"reason":plan.diagnosis,"evidence":plan.evidence}));
        state.config.max_rounds = plan.round_budget;
        if state.agent_review.is_none() {
            state.status = "awaiting_agent_repair".into();
        }
        atomic_state(&state_path, &state)?;
        return inspect(&session);
    }
    if let Some(review) = state.agent_review.as_ref() {
        ensure!(
            plan.resume_after_review
                && review["decision"] == "unresolved"
                && matches!(
                    plan.strategy,
                    RepairStrategy::RelaxStrain
                        | RepairStrategy::ExcludeInfluences
                        | RepairStrategy::WeldSeams
                        | RepairStrategy::SeparateSurfaces
                ),
            "reviewed_repair_session_is_closed"
        );
        ensure!(
            state
                .config
                .max_rounds
                .is_none_or(|limit| state.rounds.len() < limit),
            "repair_round_budget_exhausted"
        );
        state
            .review_history
            .push(state.agent_review.take().unwrap());
    }
    if matches!(plan.strategy, RepairStrategy::ReportUnresolved) {
        state.agent_review = Some(
            json!({"decision":"unresolved","notes":plan.diagnosis,"evidence":plan.evidence,"candidate_hash":state.current_hash}),
        );
        state.status = "unresolved_after_agent_review".into();
        atomic_state(&state_path, &state)?;
        return inspect(&session);
    }
    if matches!(plan.strategy, RepairStrategy::Accept) {
        state.diagnostics = Prepared::new(&current.asset, &state.config)?
            .scan(&current.asset, state.config.max_edge_growth_m)
            .0;
        add_contact_gate(&current.asset, &state.config, &mut state.diagnostics)?;
        ensure!(
            state.diagnostics.status == "passed",
            "cannot_accept_failed_numeric_gate"
        );
        state.agent_review = Some(
            json!({"decision":"accepted","notes":plan.diagnosis,"evidence":plan.evidence,"candidate_hash":state.current_hash}),
        );
        state.status = "passed_agent_review_pending_user".into();
        atomic_state(&state_path, &state)?;
        return inspect(&session);
    }
    ensure!(
        state
            .config
            .max_rounds
            .is_none_or(|limit| state.rounds.len() < limit),
        "repair_round_budget_exhausted"
    );
    let baseline = load_gltf(session.join("initial.glb"))?;
    ensure!(
        baseline.asset.source_hash == state.initial_hash,
        "repair_baseline_changed"
    );
    if let Some(check) = &plan.surface_contacts {
        ensure!(
            matches!(plan.strategy, RepairStrategy::SeparateSurfaces),
            "contact_gate_requires_surface_separation_plan"
        );
        if let Some(existing) = &state.config.surface_contacts {
            ensure!(
                serde_json::to_value(existing)? == serde_json::to_value(check)?,
                "repair_contact_gate_is_frozen"
            );
        } else {
            state.quality_gate_history.push(json!({"after_round":state.rounds.len(),"addition":"sampled_surface_crossings","reason":plan.diagnosis,"evidence":plan.evidence,"config":check}));
            state.config.surface_contacts = Some(check.clone());
        }
    }
    let mut started_from = state.current_artifact.clone();
    let prepared = Prepared::new(&current.asset, &state.config)?;
     
    state.diagnostics = prepared
        .scan(&current.asset, state.config.max_edge_growth_m)
        .0;
    add_contact_gate(&current.asset, &state.config, &mut state.diagnostics)?;
    if let Some(round) = plan.continue_from_round {
        let prior = state
            .rounds
            .iter()
            .find(|r| r.round == round)
            .context("unknown_repair_start_round")?;
        let expected = plan
            .expected_candidate_hash
            .as_ref()
            .context("continued_candidate_requires_expected_hash")?;
        let candidate = load_gltf(session.join(&prior.artifact))?;
        ensure!(
            &candidate.asset.source_hash == expected
                && (prior.artifact_hash.is_empty() || prior.artifact_hash == *expected),
            "continued_repair_candidate_changed"
        );
        ensure!(
            serde_json::to_value(&candidate.asset.skeleton)?
                == serde_json::to_value(&current.asset.skeleton)?
                && serde_json::to_value(&candidate.asset.clips)?
                    == serde_json::to_value(&current.asset.clips)?
                && serde_json::to_value(&candidate.asset.skins)?
                    == serde_json::to_value(&current.asset.skins)?
                && candidate.asset.meshes.len() == current.asset.meshes.len()
                && candidate
                    .asset
                    .meshes
                    .iter()
                    .zip(&current.asset.meshes)
                    .all(|(a, b)| a.positions == b.positions
                        && a.triangles == b.triangles
                        && a.node == b.node
                        && a.skin == b.skin),
            "continued_candidate_structure_changed"
        );
        started_from = prior.artifact.clone();
        current = candidate;
    }
    let (changed, maximum) = correct(
        &mut current.asset,
        &baseline.asset,
        &prepared,
        &state.config,
        plan,
    )?;
    write_weights(&mut current.document, &current.asset)?;
    let round = state.rounds.len() + 1;
    let relative = format!("round-{round:02}/character.glb");
    let folder = session.join(format!("round-{round:02}"));
    let mut round_report = None;
    let mut candidate_hash = String::new();
    crate::pipeline::publish_bundle(&folder, |dir| {
        current.document.save(&dir.join("character.glb"))?;
        // Gate the actual f32 export, not only the f64 optimizer state.
        let exported = load_gltf(dir.join("character.glb"))?;
        candidate_hash = exported.asset.source_hash.clone();
        let (mut diagnostics, _) = prepared.scan(&exported.asset, state.config.max_edge_growth_m);
        add_contact_gate(&exported.asset, &state.config, &mut diagnostics)?;
        let kept = (diagnostics.status == "passed" && state.diagnostics.status != "passed")
            || (diagnostics.max_edge_growth_m <= state.diagnostics.max_edge_growth_m + 1e-7
                && diagnostics.violating_edge_samples <= state.diagnostics.violating_edge_samples
                && diagnostics.max_seam_gap_m <= state.diagnostics.max_seam_gap_m + 1e-7
                && diagnostics.violating_seam_samples <= state.diagnostics.violating_seam_samples
                && contact_count(&diagnostics) <= contact_count(&state.diagnostics))
                && collapsed_count(&diagnostics) <= collapsed_count(&state.diagnostics)
            || (contact_count(&diagnostics) < contact_count(&state.diagnostics)
                && diagnostics.violating_edge_samples == 0
                && diagnostics.violating_seam_samples == 0
                && collapsed_count(&diagnostics) <= collapsed_count(&state.diagnostics));
        let report = RepairRound {
            algorithm: if matches!(plan.strategy, RepairStrategy::SeparateSurfaces) {
                "regularized_reference_side_surface_separation_v2"
            } else if matches!(plan.strategy, RepairStrategy::WeldSeams) {
                "seam_consensus_and_coupled_strain_v1"
            } else {
                "projected_strain_with_neighbor_constraints_v2"
            }
            .into(),
            round,
            artifact: relative.clone(),
            kept_as_best: kept,
            changed_vertices: changed,
            max_weight_change_l1: maximum,
            diagnostics,
            artifact_hash: candidate_hash.clone(),
            started_from: started_from.clone(),
            elapsed_seconds: started.elapsed().as_secs_f64(),
        };
        write_json(&dir.join("plan.json"), plan)?;
        write_json(&dir.join("report.json"), &report)?;
        round_report = Some(report);
        Ok(())
    })?;
    let report = round_report.unwrap();
    if report.kept_as_best {
        state.current_artifact = relative;
        state.current_hash = candidate_hash;
        state.diagnostics = report.diagnostics.clone();
    }
    state.rounds.push(report);
    state.status = if state.diagnostics.status == "passed" {
        "awaiting_agent_visual_review"
    } else if state
        .config
        .max_rounds
        .is_some_and(|limit| state.rounds.len() >= limit)
    {
        "needs_manual_review"
    } else {
        "awaiting_agent_repair"
    }
    .into();
    atomic_state(&state_path, &state)?;
    inspect(&session)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_separation_corrects_a_crossing_without_changing_geometry_or_motion() {
        let skeleton = Skeleton {
            joints: ["fixed", "moving"]
                .into_iter()
                .map(|name| Joint {
                    id: name.into(),
                    name: name.into(),
                    parent: None,
                    rest: Transform::default(),
                    matrix: None,
                    deform: true,
                })
                .collect(),
        };
        let mesh = Mesh {
            name: "two_layers".into(),
            node: 0,
            primitive: 0,
            positions: vec![
                [0., 0., 0.],
                [1., 0., 0.],
                [0., 1., 0.],
                [0.1, 0.1, 0.1],
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
            ],
            triangles: vec![[0, 1, 2], [3, 4, 5]],
            skin: Some(0),
            influences: [0.0, 0.0, 0.0, 0.7, 0.2, 0.2]
                .into_iter()
                .map(|moving| {
                    vec![
                        Influence {
                            joint: 0,
                            weight: 1.0 - moving,
                        },
                        Influence {
                            joint: 1,
                            weight: moving,
                        },
                    ]
                })
                .collect(),
            morph_targets: vec![],
            morph_weights: vec![],
        };
        let mut asset = Asset {
            name: "layers".into(),
            source_hash: "test".into(),
            skeleton,
            meshes: vec![mesh],
            skins: vec![Skin {
                name: "skin".into(),
                joints: vec![0, 1],
                inverse_bind_matrices: vec![matrix_array(&Matrix4::identity()); 2],
            }],
            clips: vec![MotionClip {
                name: "cross".into(),
                tracks: vec![Track {
                    joint: 1,
                    property: Property::Translation,
                    interpolation: Interpolation::Linear,
                    times: vec![0., 1.],
                    values: vec![[0.; 4], [0., 0., -0.2, 0.]],
                }],
                morph_tracks: vec![],
            }],
            issues: vec![],
        };
        let original = asset.clone();
        let check = crate::surface_contacts::SurfaceContactConfig {
            regions: vec![crate::surface_region::SurfaceMask {
                mesh: 0,
                geometry_hash: crate::surface_region::geometry_hash(&asset.meshes[0]),
                vertices: vec![3, 4, 5],
            }],
            clip: Some("cross".into()),
            times: vec![1.0],
            ..Default::default()
        };
        let config = RepairConfig {
            samples_per_clip: 3,
            surface_contacts: Some(check.clone()),
            ..Default::default()
        };
        let prepared = Prepared::new(&asset, &config).unwrap();
        let mut diagnostic = prepared.scan(&asset, config.max_edge_growth_m).0;
        add_contact_gate(&asset, &config, &mut diagnostic).unwrap();
        assert_eq!(contact_count(&diagnostic), 1);
        correct(
            &mut asset,
            &original,
            &prepared,
            &config,
            &RepairPlan {
                strategy: RepairStrategy::SeparateSurfaces,
                contact_weight_change_l1: 0.5,
                contact_smoothing: 0.0,
                max_weight_change_l1: 0.5,
                iterations: 32,
                ..Default::default()
            },
        )
        .unwrap();
        let mut diagnostic = prepared.scan(&asset, config.max_edge_growth_m).0;
        add_contact_gate(&asset, &config, &mut diagnostic).unwrap();
        assert_eq!(
            diagnostic.status, "passed",
            "{diagnostic:?}; weights {:?}",
            asset.meshes[0].influences
        );
        assert_eq!(contact_count(&diagnostic), 0);
        assert_eq!(asset.meshes[0].positions, original.meshes[0].positions);
        assert_eq!(
            serde_json::to_value(&asset.clips).unwrap(),
            serde_json::to_value(&original.clips).unwrap()
        );
        assert!(
            asset.meshes[0]
                .influences
                .iter()
                .zip(&original.meshes[0].influences)
                .all(|(a, b)| l1(a, b) <= 0.5 + 1e-8)
        );
    }

    fn fixture(path: &Path) {
        let mut doc = charloom_io::mesh_document(
            "cloth",
            &[
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.0, 0.01, 0.0],
                [0.01, 0.01, 0.0],
            ],
            &[[0, 1, 2], [1, 3, 2]],
        )
        .unwrap();
        let skeleton = Skeleton {
            joints: ["fixed", "moving"]
                .into_iter()
                .map(|name| Joint {
                    id: name.into(),
                    name: name.into(),
                    parent: None,
                    rest: Transform::default(),
                    matrix: None,
                    deform: true,
                })
                .collect(),
        };
        let offset = doc.append_skeleton(&skeleton).unwrap();
        let inverse = doc.append_matrices(&[matrix_array(&Matrix4::identity()); 2]);
        doc.json["skins"] = json!([{"joints":[offset, offset+1],"inverseBindMatrices":inverse}]);
        doc.json["nodes"][0]["skin"] = json!(0);
        let joints: Vec<u8> = (0..4)
            .flat_map(|_| [0u16, 1, 0, 0])
            .flat_map(u16::to_le_bytes)
            .collect();
        let weights: Vec<u8> = [
            [0.8f32, 0.2, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0],
            [0.2, 0.8, 0.0, 0.0],
        ]
        .into_iter()
        .flatten()
        .flat_map(f32::to_le_bytes)
        .collect();
        let j = doc.append_accessor(&joints, 5123, "VEC4", 4, Some(34962), None);
        let w = doc.append_accessor(&weights, 5126, "VEC4", 4, Some(34962), None);
        doc.json["meshes"][0]["primitives"][0]["attributes"]["JOINTS_0"] = json!(j);
        doc.json["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_0"] = json!(w);
        doc.append_animation(
            &MotionClip {
                name: "pull".into(),
                tracks: vec![Track {
                    joint: offset + 1,
                    property: Property::Translation,
                    interpolation: Interpolation::Linear,
                    times: vec![0.0, 1.0],
                    values: vec![[0.0; 4], [0.0, 0.2, 0.0, 0.0]],
                }],
                morph_tracks: vec![],
            },
            0,
        )
        .unwrap();
        doc.save(path).unwrap();
    }

    fn plan(round: usize) -> RepairPlan {
        RepairPlan {
            expected_round: round,
            diagnosis: "Adjacent garment vertices split between moving and fixed anchors".into(),
            evidence: vec!["session.json#/diagnostics/hotspots".into()],
            max_weight_change_l1: 0.5,
            ..Default::default()
        }
    }

    #[test]
    fn correction_reduces_animated_strain_with_fixed_geometry_and_global_budget() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input.glb");
        fixture(&input);
        let original = fs::read(&input).unwrap();
        let session = temp.path().join("session");
        let started = start(
            &input,
            &RepairConfig {
                samples_per_clip: 5,
                ..Default::default()
            },
            &session,
        )
        .unwrap();
        assert_eq!(started["state"]["diagnostics"]["status"], "failed");
        let accept = RepairPlan {
            strategy: RepairStrategy::Accept,
            ..plan(0)
        };
        assert!(
            step(&session, &accept)
                .unwrap_err()
                .to_string()
                .contains("cannot_accept_failed")
        );
        let result = step(&session, &plan(0)).unwrap();
        assert_eq!(result["state"]["diagnostics"]["status"], "passed");
        assert!(
            step(&session, &plan(0))
                .unwrap_err()
                .to_string()
                .contains("stale_repair_plan")
        );
        let old = load_gltf(&input).unwrap();
        let fixed = load_gltf(session.join("round-01/character.glb")).unwrap();
        assert_eq!(
            serde_json::to_value(&old.asset.clips).unwrap(),
            serde_json::to_value(&fixed.asset.clips).unwrap()
        );
        assert_eq!(
            serde_json::to_value(&old.asset.skeleton).unwrap(),
            serde_json::to_value(&fixed.asset.skeleton).unwrap()
        );
        for (a, b) in old.asset.meshes.iter().zip(&fixed.asset.meshes) {
            assert_eq!(a.positions, b.positions);
            assert_eq!(a.triangles, b.triangles);
            for (before, after) in a.influences.iter().zip(&b.influences) {
                assert!(l1(before, after) <= 0.500001);
                assert!((after.iter().map(|w| w.weight).sum::<f64>() - 1.0).abs() < 1e-6);
            }
        }
        assert_eq!(fs::read(input).unwrap(), original);
        let result = step(
            &session,
            &RepairPlan {
                expected_round: 1,
                ..accept
            },
        )
        .unwrap();
        assert_eq!(result["state"]["user_visual_review"], "pending");
        assert_eq!(result["state"]["runtime_adoption"], false);
    }

    #[test]
    fn explicit_round_budget_is_enforced_without_a_hardcoded_three_round_limit() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input.glb");
        fixture(&input);
        let session = temp.path().join("session");
        start(
            &input,
            &RepairConfig {
                max_edge_growth_m: 0.001,
                max_rounds: Some(3),
                samples_per_clip: 3,
                ..Default::default()
            },
            &session,
        )
        .unwrap();
        for round in 0..3 {
            step(
                &session,
                &RepairPlan {
                    max_weight_change_l1: 0.01,
                    ..plan(round)
                },
            )
            .unwrap();
        }
        assert_eq!(
            inspect(&session).unwrap()["state"]["status"],
            "needs_manual_review"
        );
        assert!(
            step(&session, &plan(3))
                .unwrap_err()
                .to_string()
                .contains("budget_exhausted")
        );
        assert!(!session.join("round-04").exists());
        let expanded = step(
            &session,
            &RepairPlan {
                strategy: RepairStrategy::UpdateBudget,
                round_budget: None,
                ..plan(3)
            },
        )
        .unwrap();
        assert_eq!(expanded["state"]["rounds"].as_array().unwrap().len(), 3);
        assert!(expanded["state"]["config"]["max_rounds"].is_null());
        assert_eq!(
            expanded["state"]["budget_history"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            step(&session, &plan(3)).unwrap()["state"]["rounds"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
    }

    #[test]
    fn exclusions_are_not_reintroduced_by_strain_relaxation() {
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("input.glb");
        fixture(&input);
        let mut loaded = load_gltf(&input).unwrap();
        let baseline = loaded.asset.clone();
        let config = RepairConfig {
            samples_per_clip: 3,
            ..Default::default()
        };
        let prepared = Prepared::new(&loaded.asset, &config).unwrap();
         
        let mut p = plan(0);
        p.strategy = RepairStrategy::ExcludeInfluences;
        p.regions = vec![RepairRegion {
            mask: None,
            min: [-0.01, -0.01, -0.01],
            max: [0.005, 0.02, 0.01],
            exclude_joints: vec!["moving".into()],
            feather_m: 0.001,
        }];
        correct(&mut loaded.asset, &baseline, &prepared, &config, &p).unwrap();
        for i in [0, 2] {
            assert!(
                loaded.asset.meshes[0].influences[i]
                    .iter()
                    .filter(|w| loaded.asset.skeleton.joints[w.joint].name == "moving")
                    .all(|w| w.weight == 0.0)
            );
        }
    }

    #[test]
    fn numeric_pass_can_be_closed_with_unresolved_visual_issues() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("input.glb");
        fixture(&input);
        let session = dir.path().join("session");
        start(
            &input,
            &RepairConfig {
                samples_per_clip: 3,
                ..Default::default()
            },
            &session,
        )
        .unwrap();
        step(&session, &plan(0)).unwrap();
        let result = step(
            &session,
            &RepairPlan {
                strategy: RepairStrategy::ReportUnresolved,
                diagnosis: "Cloth layers intersect; edge strain does not certify separation".into(),
                ..plan(1)
            },
        )
        .unwrap();
        assert_eq!(result["state"]["diagnostics"]["status"], "passed");
        assert_eq!(result["state"]["status"], "unresolved_after_agent_review");
        assert_eq!(result["state"]["agent_review"]["decision"], "unresolved");
        assert_eq!(result["state"]["rounds"].as_array().unwrap().len(), 1);
        assert!(
            step(
                &session,
                &RepairPlan {
                    strategy: RepairStrategy::Accept,
                    ..plan(1)
                }
            )
            .is_err()
        );
        let resumed = step(
            &session,
            &RepairPlan {
                resume_after_review: true,
                ..plan(1)
            },
        )
        .unwrap();
        assert_eq!(resumed["state"]["rounds"].as_array().unwrap().len(), 2);
        assert_eq!(
            resumed["state"]["review_history"].as_array().unwrap().len(),
            1
        );
        assert!(resumed["state"]["agent_review"].is_null());
        step(&session, &plan(2)).unwrap();
        assert_eq!(
            step(&session, &plan(3)).unwrap()["state"]["rounds"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
    }

    #[test]
    fn split_render_seams_are_audited_and_share_weights_after_export() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("split.glb");
        fixture(&input);
        let mut source = load_gltf(&input).unwrap();
        let mesh = &mut source.asset.meshes[0];
        mesh.positions
            .extend([mesh.positions[1], mesh.positions[2]]);
        mesh.triangles = vec![[0, 1, 2], [4, 3, 5]];
        let a = mesh.influences[0][0].joint;
        let b = mesh.influences[0][1].joint;
        mesh.influences = [0.5, 0.6, 0.6, 0.5, 0.4, 0.4]
            .map(|x| {
                vec![
                    Influence {
                        joint: a,
                        weight: x,
                    },
                    Influence {
                        joint: b,
                        weight: 1.0 - x,
                    },
                ]
            })
            .into();
        let positions: Vec<u8> = mesh
            .positions
            .iter()
            .flatten()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect();
        let indices: Vec<u8> = mesh
            .triangles
            .iter()
            .flatten()
            .flat_map(|&i| i.to_le_bytes())
            .collect();
        let p = source
            .document
            .append_accessor(&positions, 5126, "VEC3", 6, Some(34962), None);
        source.document.json["accessors"][p]["min"] = json!([0.0, 0.0, 0.0]);
        source.document.json["accessors"][p]["max"] = json!([0.01, 0.01, 0.0]);
        let i = source
            .document
            .append_accessor(&indices, 5125, "SCALAR", 6, Some(34963), None);
        source.document.json["meshes"][0]["primitives"][0]["attributes"]["POSITION"] = json!(p);
        source.document.json["meshes"][0]["primitives"][0]["indices"] = json!(i);
        write_weights(&mut source.document, &source.asset).unwrap();
        source.document.save(&input).unwrap();
        let bytes = fs::read(&input).unwrap();
        let session = dir.path().join("session");
        let initial = start(
            &input,
            &RepairConfig {
                samples_per_clip: 3,
                ..Default::default()
            },
            &session,
        )
        .unwrap();
        assert_eq!(initial["state"]["diagnostics"]["seam_groups"], 2);
        assert!(
            initial["state"]["diagnostics"]["max_seam_gap_m"]
                .as_f64()
                .unwrap()
                > 0.039
        );
        let fixed = step(
            &session,
            &RepairPlan {
                strategy: RepairStrategy::WeldSeams,
                ..plan(0)
            },
        )
        .unwrap();
        assert_eq!(fixed["state"]["diagnostics"]["status"], "passed");
        assert!(
            fixed["state"]["diagnostics"]["max_seam_gap_m"]
                .as_f64()
                .unwrap()
                < 1e-8
        );
        let after = load_gltf(session.join("round-01/character.glb")).unwrap();
        let before = load_gltf(&input).unwrap();
        assert_eq!(
            after.asset.meshes[0].positions,
            before.asset.meshes[0].positions
        );
        assert_eq!(
            after.asset.meshes[0].triangles,
            before.asset.meshes[0].triangles
        );
        assert_eq!(fs::read(input).unwrap(), bytes);
    }
}
