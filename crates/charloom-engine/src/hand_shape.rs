//! Preserve target digit proportions and bound new skin-edge elongation.
//! This is a pose guard, not a grasp generator or a change to skin weights.
use anyhow::{Result, ensure};
use charloom_core::*;
use nalgebra::{Matrix4, Vector3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

fn yes() -> bool {
    true
}
fn ratio() -> f64 {
    1.2
}
fn growth() -> f64 {
    0.004
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct DigitShape {
    pub name: String,
    /// Ordered base-to-tip chain, including a nondeforming tip when available.
    pub joints: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::UnitQuaternion;

    #[test]
    fn five_digits_keep_target_lengths_and_limit_surface_stretch() {
        let joint = |name: String, parent, position| Joint {
            id: name.clone(),
            name,
            parent,
            rest: Transform::translated(position),
            matrix: None,
            deform: true,
        };
        let mut skeleton = Skeleton {
            joints: vec![joint("wrist".into(), None, [0.0; 3])],
        };
        let mut digits = Vec::new();
        let mut positions = Vec::new();
        let mut influences = Vec::new();
        let mut triangles = Vec::new();
        for (i, name) in ["thumb", "index", "middle", "ring", "little"]
            .into_iter()
            .enumerate()
        {
            let first = skeleton.joints.len();
            let x = i as f64 * 0.04;
            let length = 0.04 + i as f64 * 0.003;
            skeleton
                .joints
                .push(joint(format!("{name}1"), Some(0), [x, 0.07, 0.0]));
            skeleton
                .joints
                .push(joint(format!("{name}2"), Some(first), [0.0, length, 0.0]));
            digits.push(DigitShape {
                name: name.into(),
                joints: vec![format!("{name}1"), format!("{name}2")],
            });
            let base = positions.len() as u32;
            positions.extend([
                [x + 0.005, 0.07 + length - 0.01, 0.0],
                [x - 0.005, 0.07 + length - 0.008, 0.0],
                [x + 0.004, 0.07 + length - 0.012, 0.004],
            ]);
            influences.extend([
                vec![Influence {
                    joint: first,
                    weight: 1.0,
                }],
                vec![Influence {
                    joint: first + 1,
                    weight: 1.0,
                }],
                vec![Influence {
                    joint: first,
                    weight: 1.0,
                }],
            ]);
            triangles.push([base, base + 1, base + 2]);
        }
        let world = skeleton.world_matrices(&skeleton.rest_pose()).unwrap();
        let asset = Asset {
            name: "five_different_digit_lengths".into(),
            source_hash: String::new(),
            skins: vec![Skin {
                name: "hand".into(),
                joints: (0..skeleton.joints.len()).collect(),
                inverse_bind_matrices: world
                    .iter()
                    .map(|w| matrix_array(&w.try_inverse().unwrap()))
                    .collect(),
            }],
            skeleton,
            meshes: vec![Mesh {
                name: "hand".into(),
                node: 0,
                primitive: 0,
                positions,
                triangles,
                skin: Some(0),
                influences,
                morph_targets: vec![],
                morph_weights: vec![],
            }],
            clips: vec![],
            issues: vec![],
        };
        let guard = PreparedHandShape::prepare(
            &asset,
            &HandShapeConstraint {
                wrist: "wrist".into(),
                digits,
                max_edge_ratio: 1.05,
                max_edge_growth_m: 0.0001,
                rebind_missing_digits: false,
                allow_rotation_fallback: true,
                surface_repair: None,
                surface_corrective: None,
            },
        )
        .unwrap();
        let mut pose = asset.skeleton.rest_pose();
        for i in (2..pose.len()).step_by(2) {
            pose[i].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::z_axis(), -1.8));
            pose[i].translation[1] *= 1.8;
            pose[i].scale = [1.2; 3];
        }
        let wrist = pose[0];
        let report = guard.apply(&asset, &mut pose, &HashMap::new()).unwrap();
        assert_eq!(report.status, "converged");
        assert_eq!(report.reset_translation_or_scale_joints, 5);
        assert_eq!(report.digits.len(), 5);
        assert_eq!(pose[0], wrist);
        for d in &report.digits {
            assert!(d.max_segment_length_error_m < 1e-12);
            assert_eq!(d.violating_edges, 0);
            assert!(d.retained_rotation_fraction < 1.0);
            assert!(d.after_max_edge_growth_m < d.before_max_edge_growth_m);
        }
        // The model's distinct ratios are preserved, not replaced by average anatomy.
        assert_ne!(
            report.digits[0].target_segment_lengths_m[1],
            report.digits[4].target_segment_lengths_m[1]
        );
        let mut merged = asset.clone();
        for row in &mut merged.meshes[0].influences {
            *row = vec![Influence {
                joint: 5,
                weight: 1.0,
            }];
        }
        let mut config = guard.config.clone();
        config.rebind_missing_digits = true;
        let mut posed_default = merged.clone();
        posed_default.skeleton.joints[5].rest.rotation =
            a4(UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 1.7));
        let binding = rebind_missing_digits(&mut merged, &config).unwrap();
        rebind_missing_digits(&mut posed_default, &config).unwrap();
        assert_eq!(
            serde_json::to_value(&merged.meshes[0].influences).unwrap(),
            serde_json::to_value(&posed_default.meshes[0].influences).unwrap(),
            "digit partition must be independent of the display pose"
        );
        assert_eq!(binding["changed_vertices"], 15);
        assert!(
            binding["digit_surface_vertices_after"]
                .as_array()
                .unwrap()
                .iter()
                .all(|v| v.as_u64().unwrap() > 0)
        );
        assert_eq!(merged.meshes[0].positions, asset.meshes[0].positions);
        for row in &merged.meshes[0].influences {
            assert!((row.iter().map(|w| w.weight).sum::<f64>() - 1.0).abs() < 1e-12);
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HandShapeConstraint {
    pub wrist: String,
    pub digits: Vec<DigitShape>,
    /// Explicitly repartition existing digit-weighted surface if some named
    /// chains have no independent weights. The wrist/digit region is repartitioned; body weights are preserved.
    #[serde(default)]
    pub rebind_missing_digits: bool,
    /// If false, retain authored finger rotations and report surface failures.
    /// Use this for grips, where opening a fist is not a valid repair.
    #[serde(default = "yes")]
    pub allow_rotation_fallback: bool,
         #[serde(default)]
    pub surface_repair: Option<crate::hand_surface::HandSurfaceRepair>,
    #[serde(default)]
    pub surface_corrective: Option<crate::hand_corrective::HandSurfaceCorrective>,
    /// An edge may grow to reference_length * ratio + growth_m.
    #[serde(default = "ratio")]
    pub max_edge_ratio: f64,
    #[serde(default = "growth")]
    pub max_edge_growth_m: f64,
}

pub fn rebind_missing_digits(
    asset: &mut Asset,
    config: &HandShapeConstraint,
) -> Result<serde_json::Value> {
    let joint = |name: &str| -> Result<usize> {
        let ids: Vec<_> = asset
            .skeleton
            .joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.name == name)
            .map(|(i, _)| i)
            .collect();
        ensure!(ids.len() == 1, "digit_binding_joint_must_be_unique: {name}");
        Ok(ids[0])
    };
    let chains: Vec<Vec<usize>> = config
        .digits
        .iter()
        .map(|d| d.joints.iter().map(|n| joint(n)).collect())
        .collect::<Result<_>>()?;
    let all: HashSet<usize> = chains.iter().flatten().copied().collect();
    let wrist = joint(&config.wrist)?;
    let mut family = all.clone();
    family.insert(wrist);
    let coverage = |asset: &Asset| -> Vec<usize> {
        chains
            .iter()
            .map(|chain| {
                asset
                    .meshes
                    .iter()
                    .flat_map(|m| &m.influences)
                    .filter(|row| {
                        row.iter()
                            .filter(|w| chain.contains(&w.joint))
                            .map(|w| w.weight)
                            .sum::<f64>()
                            >= 0.2
                    })
                    .count()
            })
            .collect()
    };
    let before = coverage(asset);
    if !before.contains(&0) || !config.rebind_missing_digits {
        return Ok(
            serde_json::json!({"wrist":config.wrist,"digit_surface_vertices_before":before,"changed_vertices":0,"requested":config.rebind_missing_digits}),
        );
    }
    let mut changed = 0;
    for mesh in &mut asset.meshes {
        let Some(skin) = mesh.skin else { continue };
        if !mesh
            .influences
            .iter()
            .any(|row| row.iter().any(|w| all.contains(&w.joint) && w.weight > 0.0))
        {
            continue;
        }
        ensure!(
            all.iter().all(|j| asset.skins[skin].joints.contains(j)),
            "digit_binding_requires_chains_in_existing_skin"
        );
        let mut points = vec![[0.0; 3]; asset.skeleton.joints.len()];
        for (&j, bind) in asset.skins[skin]
            .joints
            .iter()
            .zip(&asset.skins[skin].inverse_bind_matrices)
        {
            let world = matrix_from(*bind)
                .try_inverse()
                .ok_or_else(|| anyhow::anyhow!("singular_digit_bind"))?;
            points[j] = transform_point(&world, [0.0; 3]);
        }
        let segments: Vec<Vec<_>> = chains
            .iter()
            .map(|chain| {
                chain
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &j)| {
                        let a = v3(points[j]);
                        let b = if i + 1 < chain.len() {
                            v3(points[chain[i + 1]])
                        } else if i > 0 {
                            a + (a - v3(points[chain[i - 1]])) * 0.65
                        } else {
                            return None;
                        };
                        ((b - a).norm() > 1e-7).then_some((j, a, b))
                    })
                    .collect()
            })
            .collect();
        ensure!(
            segments.iter().all(|s| !s.is_empty()),
            "digit_binding_requires_nonzero_chains"
        );
        // Euclidean nearest-bone classification can jump across the air gap
        // between fingers. Propagate distal seeds along the actual surface,
        // welding only verified render seams, before assigning a digit family.
        let topology = crate::surface_topology::SurfaceTopology::new(mesh);
        let eligible: Vec<_> = mesh
            .influences
            .iter()
            .map(|row| {
                row.iter()
                    .filter(|w| family.contains(&w.joint))
                    .map(|w| w.weight)
                    .sum::<f64>()
                    >= 0.98
            })
            .collect();
        let mut geodesic = Vec::new();
        let mut seeds = HashSet::new();
        for chain in &segments {
            let (_, a, b) = *chain.last().unwrap();
            let tip = a + (b - a) * 0.8;
            let seed = (0..mesh.positions.len())
                .filter(|&v| eligible[v] && !seeds.contains(&topology.representative[v]))
                .min_by(|&a, &b| {
                    (v3(mesh.positions[a]) - tip)
                        .norm_squared()
                        .total_cmp(&(v3(mesh.positions[b]) - tip).norm_squared())
                });
            let mut distance = vec![f64::INFINITY; mesh.positions.len()];
            if let Some(seed) = seed {
                let seed = topology.representative[seed];
                seeds.insert(seed);
                distance[seed] = 0.0;
                let mut queue = std::collections::BinaryHeap::new();
                queue.push(std::cmp::Reverse((0_u64, seed)));
                while let Some(std::cmp::Reverse((bits, v))) = queue.pop() {
                    let d = f64::from_bits(bits);
                    if d > distance[v] {
                        continue;
                    }
                    for &next in &topology.adjacency[v] {
                        if !eligible[next] {
                            continue;
                        }
                        let nd = d + (v3(mesh.positions[v]) - v3(mesh.positions[next])).norm();
                        if nd < distance[next] {
                            distance[next] = nd;
                            queue.push(std::cmp::Reverse((nd.to_bits(), next)));
                        }
                    }
                }
            }
            geodesic.push(distance);
        }
        for (vi, row) in mesh.influences.iter_mut().enumerate() {
            let mass: f64 = row
                .iter()
                .filter(|w| family.contains(&w.joint))
                .map(|w| w.weight)
                .sum();
            if mass < 0.98 || row.iter().filter(|w| !family.contains(&w.joint)).count() > 1 {
                continue;
            }
            let point = v3(mesh.positions[vi]);
            let distances: Vec<Vec<_>> = segments
                .iter()
                .map(|chain| {
                    chain
                        .iter()
                        .map(|&(j, a, b)| {
                            let delta = b - a;
                            let t =
                                ((point - a).dot(&delta) / delta.norm_squared()).clamp(0.0, 1.0);
                            (j, (point - a - delta * t).norm_squared(), delta.norm())
                        })
                        .collect()
                })
                .collect();
            let digit = distances
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.iter()
                        .map(|v| v.1)
                        .fold(f64::INFINITY, f64::min)
                        .total_cmp(&b.iter().map(|v| v.1).fold(f64::INFINITY, f64::min))
                })
                .unwrap()
                .0;
            let representative = topology.representative[vi];
            let digit = geodesic
                .iter()
                .enumerate()
                .filter(|(_, d)| d[representative].is_finite())
                .min_by(|(_, a), (_, b)| a[representative].total_cmp(&b[representative]))
                .map_or(digit, |(i, _)| i);
            let mut candidates = distances[digit].clone();
            candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
            candidates.truncate(2);
            let mut result: Vec<_> = row
                .iter()
                .copied()
                .filter(|w| !family.contains(&w.joint))
                .collect();

            let minimum = candidates[0].1;
            let sigma = (candidates[0].2 * 0.18).max(0.003);
            let palm_distance = chains
                .iter()
                .map(|chain| {
                    let a = v3(points[wrist]);
                    let d = v3(points[chain[0]]) - a;
                    let t = ((point - a).dot(&d) / d.norm_squared()).clamp(0.0, 1.0);
                    (point - a - d * t).norm()
                })
                .fold(f64::INFINITY, f64::min);
            let fraction = ((palm_distance - minimum.sqrt()) / (candidates[0].2 * 0.25).max(0.004))
                .clamp(0.0, 1.0);
            let fraction = fraction * fraction * (3.0 - 2.0 * fraction);
            result.push(Influence {
                joint: wrist,
                weight: mass * (1.0 - fraction),
            });
            let sum: f64 = candidates
                .iter()
                .map(|c| (-(c.1 - minimum) / (2.0 * sigma * sigma)).exp())
                .sum();
            result.extend(candidates.iter().map(|c| Influence {
                joint: c.0,
                weight: mass * fraction * (-(c.1 - minimum) / (2.0 * sigma * sigma)).exp() / sum,
            }));
            result.retain(|w| w.weight > 1e-12);
            result.sort_by(|a, b| b.weight.total_cmp(&a.weight));
            *row = result;
            changed += 1;
        }
    }
    let after = coverage(asset);
    ensure!(
        !after.contains(&0),
        "digit_binding_geometry_has_unresolved_chains"
    );
    Ok(
        serde_json::json!({"wrist":config.wrist,"method":"explicit_chain_geodesic_bind_space_partition_v4","digit_surface_vertices_before":before,"digit_surface_vertices_after":after,"changed_vertices":changed,"limits":["Explicit wrist and digit family only; body weights and original geometry are preserved.","Closed or intersecting fingers can be geometrically ambiguous; visual review is required."]}),
    )
}

struct Sample {
    node: usize,
    influences: Vec<(usize, f64, Vec3, Vec<Vec3>)>,
    default_morphs: Vec<f64>,
}
impl Sample {
    fn point(&self, world: &[Matrix4<f64>], morphs: &HashMap<usize, Vec<f64>>) -> Vector3<f64> {
        let weights = morphs.get(&self.node).unwrap_or(&self.default_morphs);
        self.influences
            .iter()
            .fold(Vector3::zeros(), |sum, (j, weight, base, targets)| {
                let p = targets
                    .iter()
                    .zip(weights)
                    .fold(v3(*base), |p, (delta, w)| p + v3(*delta) * *w);
                sum + v3(transform_point(&world[*j], a3(p))) * *weight
            })
    }
}
struct Digit {
    name: String,
    joints: Vec<usize>,
    edges: Vec<(usize, usize, f64)>,
    lengths: Vec<f64>,
    reference_rotations: Vec<Quat>,
}
pub struct PreparedHandShape {
    wrist: usize,
    digits: Vec<Digit>,
    samples: Vec<Sample>,
    config: HandShapeConstraint,
}

#[derive(Debug, Clone, Serialize)]
pub struct DigitShapeReport {
    pub name: String,
    pub target_segment_lengths_m: Vec<f64>,
    pub max_segment_length_error_m: f64,
    pub surface_edges: usize,
    pub before_max_edge_growth_m: f64,
    pub after_max_edge_growth_m: f64,
    pub violating_edges: usize,
    pub retained_rotation_fraction: f64,
}
#[derive(Debug, Clone, Serialize)]
pub struct HandShapeReport {
    pub wrist: String,
    pub status: String,
    pub reset_translation_or_scale_joints: usize,
    pub digits: Vec<DigitShapeReport>,
}

impl PreparedHandShape {
    pub fn prepare(asset: &Asset, config: &HandShapeConstraint) -> Result<Self> {
        ensure!(
            !config.digits.is_empty() && config.digits.len() <= 16,
            "invalid_digit_count"
        );
        ensure!(
            config.max_edge_ratio.is_finite()
                && config.max_edge_ratio >= 1.0
                && config.max_edge_growth_m.is_finite()
                && config.max_edge_growth_m >= 0.0,
            "invalid_hand_surface_limits"
        );
        let joint = |name: &str| -> Result<usize> {
            let ids: Vec<_> = asset
                .skeleton
                .joints
                .iter()
                .enumerate()
                .filter(|(_, j)| j.name == name)
                .map(|(i, _)| i)
                .collect();
            ensure!(ids.len() == 1, "hand_shape_joint_must_be_unique: {name}");
            Ok(ids[0])
        };
        let wrist = joint(&config.wrist)?;
        let rest = asset.skeleton.rest_pose();
        let world = asset.skeleton.world_matrices(&rest)?;

        let mut used = HashSet::new();
        let mut digits = Vec::new();
        for spec in &config.digits {
            ensure!(
                !spec.name.is_empty() && (2..=8).contains(&spec.joints.len()),
                "invalid_digit_chain"
            );
            let joints: Vec<_> = spec
                .joints
                .iter()
                .map(|s| joint(s))
                .collect::<Result<_>>()?;
            let mut parent = wrist;
            let mut lengths = Vec::new();
            for &j in &joints {
                ensure!(
                    used.insert(j)
                        && asset.skeleton.joints[j].parent == Some(parent)
                        && asset.skeleton.joints[j].matrix.is_none(),
                    "digit_requires_disjoint_direct_TRS_chain"
                );
                lengths.push(
                    (v3(transform_point(&world[j], [0.0; 3]))
                        - v3(transform_point(&world[parent], [0.0; 3])))
                    .norm(),
                );
                parent = j;
            }
            let bind_skin = asset
                .skins
                .iter()
                .find(|skin| {
                    std::iter::once(&wrist)
                        .chain(&joints)
                        .all(|j| skin.joints.contains(j))
                })
                .ok_or_else(|| anyhow::anyhow!("digit_chain_missing_from_skin"))?;
            let bind_world = |j: usize| -> Result<Matrix4<f64>> {
                let at = bind_skin
                    .joints
                    .iter()
                    .position(|&k| k == j)
                    .ok_or_else(|| anyhow::anyhow!("digit_parent_missing_from_skin"))?;
                matrix_from(bind_skin.inverse_bind_matrices[at])
                    .try_inverse()
                    .ok_or_else(|| anyhow::anyhow!("singular_digit_reference"))
            };
            let reference_rotations = joints
                .iter()
                .map(|&j| {
                    let parent = asset.skeleton.joints[j].parent.unwrap();
                    Ok(a4(crate::solve::world_rotation(
                        &(bind_world(parent)?.try_inverse().unwrap() * bind_world(j)?),
                    )?))
                })
                .collect::<Result<Vec<_>>>()?;
            digits.push(Digit {
                name: spec.name.clone(),
                reference_rotations,
                joints,
                edges: Vec::new(),
                lengths,
            });
        }
        let mut samples = Vec::new();
        for mesh in &asset.meshes {
            let Some(skin_index) = mesh.skin else {
                continue;
            };
            let skin = &asset.skins[skin_index];
            let binds: HashMap<_, _> = skin
                .joints
                .iter()
                .copied()
                .zip(skin.inverse_bind_matrices.iter().map(|m| matrix_from(*m)))
                .collect();
            let mut sampled = HashMap::new();
            for digit in &mut digits {
                let weight: Vec<_> = mesh
                    .influences
                    .iter()
                    .map(|weights| {
                        weights
                            .iter()
                            .filter(|w| digit.joints.contains(&w.joint))
                            .map(|w| w.weight)
                            .sum::<f64>()
                    })
                    .collect();
                let mut seen = HashSet::new();
                for tri in &mesh.triangles {
                    for k in 0..3 {
                        let a = tri[k].min(tri[(k + 1) % 3]) as usize;
                        let b = tri[k].max(tri[(k + 1) % 3]) as usize;
                        if weight[a].max(weight[b]) < 0.2 || !seen.insert((a, b)) {
                            continue;
                        }
                        let length = (v3(mesh.positions[a]) - v3(mesh.positions[b])).norm();
                        if length < 1e-7 {
                            continue;
                        }
                        let mut ids = [0; 2];
                        for (slot, vi) in [a, b].into_iter().enumerate() {
                            ids[slot] = *sampled.entry(vi).or_insert_with(|| {
                                let index = samples.len();
                                samples.push(Sample {
                                    node: mesh.node,
                                    default_morphs: mesh.morph_weights.clone(),
                                    influences: mesh.influences[vi]
                                        .iter()
                                        .map(|w| {
                                            let bind = &binds[&w.joint];
                                            let origin = v3(transform_point(bind, [0.0; 3]));
                                            (
                                                w.joint,
                                                w.weight,
                                                transform_point(bind, mesh.positions[vi]),
                                                mesh.morph_targets
                                                    .iter()
                                                    .map(|t| {
                                                        a3(v3(transform_point(bind, t[vi]))
                                                            - origin)
                                                    })
                                                    .collect(),
                                            )
                                        })
                                        .collect(),
                                });
                                index
                            });
                        }
                        digit.edges.push((ids[0], ids[1], length));
                    }
                }
            }
        }
        ensure!(
            digits.iter().all(|d| !d.edges.is_empty()),
            "digit_requires_weighted_surface_edges"
        );
        Ok(Self {
            wrist,
            digits,
            samples,
            config: config.clone(),
        })
    }

    fn audit(
        &self,
        asset: &Asset,
        pose: &[Transform],
        morphs: &HashMap<usize, Vec<f64>>,
    ) -> Result<Vec<(usize, f64)>> {
        let world = asset.skeleton.world_matrices(pose)?;
        let points: Vec<_> = self
            .samples
            .iter()
            .map(|s| s.point(&world, morphs))
            .collect();
        Ok(self
            .digits
            .iter()
            .map(|d| {
                let mut violations = 0;
                let mut growth = 0.0_f64;
                for &(a, b, rest) in &d.edges {
                    let actual = (points[a] - points[b]).norm();
                    growth = growth.max(actual - rest);
                    if actual
                        > rest * self.config.max_edge_ratio + self.config.max_edge_growth_m + 1e-9
                    {
                        violations += 1;
                    }
                }
                (violations, growth)
            })
            .collect())
    }

    pub fn apply(
        &self,
        asset: &Asset,
        pose: &mut [Transform],
        morphs: &HashMap<usize, Vec<f64>>,
    ) -> Result<HandShapeReport> {
        ensure!(
            pose.len() == asset.skeleton.joints.len(),
            "hand_shape_pose_size_mismatch"
        );
        let mut reset = 0;
        for digit in &self.digits {
            for &j in &digit.joints {
                let rest = asset.skeleton.joints[j].rest;
                if pose[j].translation != rest.translation || pose[j].scale != rest.scale {
                    reset += 1;
                }
                pose[j].translation = rest.translation;
                pose[j].scale = rest.scale;
            }
        }
        let before = self.audit(asset, pose, morphs)?;
        let original = pose.to_vec();
        let mut fractions = vec![1.0; self.digits.len()];
         
        for _ in 0..if self.config.allow_rotation_fallback {
            2
        } else {
            0
        } {
            let state = self.audit(asset, pose, morphs)?;
            for (di, digit) in self.digits.iter().enumerate() {
                if state[di].0 == 0 {
                    continue;
                }
                for fraction in [0.75, 0.5, 0.25, 0.0] {
                    if fraction >= fractions[di] {
                        continue;
                    }
                    for (&j, &reference) in digit.joints.iter().zip(&digit.reference_rotations) {
                        pose[j].rotation =
                            a4(unit_quat(reference)
                                .slerp(&unit_quat(original[j].rotation), fraction));
                    }
                    fractions[di] = fraction;
                    if self.audit(asset, pose, morphs)?[di].0 == 0 {
                        break;
                    }
                }
            }
        }
        let mut after = self.audit(asset, pose, morphs)?;
        if after.iter().map(|v| v.0).sum::<usize>() > before.iter().map(|v| v.0).sum::<usize>()
            || after.iter().map(|v| v.1).fold(0.0_f64, f64::max)
                > before.iter().map(|v| v.1).fold(0.0_f64, f64::max) + 1e-9
        {
            pose.copy_from_slice(&original);
            fractions.fill(1.0);
            after = before.clone();
        }
        let world = asset.skeleton.world_matrices(pose)?;
        let mut reports = Vec::new();
        for (di, digit) in self.digits.iter().enumerate() {
            let mut parent = self.wrist;
            let mut error = 0.0_f64;
            for (&j, &length) in digit.joints.iter().zip(&digit.lengths) {
                let current = (v3(transform_point(&world[j], [0.0; 3]))
                    - v3(transform_point(&world[parent], [0.0; 3])))
                .norm();
                error = error.max((current - length).abs());
                parent = j;
            }
            reports.push(DigitShapeReport {
                name: digit.name.clone(),
                target_segment_lengths_m: digit.lengths.clone(),
                max_segment_length_error_m: error,
                surface_edges: digit.edges.len(),
                before_max_edge_growth_m: before[di].1,
                after_max_edge_growth_m: after[di].1,
                violating_edges: after[di].0,
                retained_rotation_fraction: fractions[di],
            });
        }
        Ok(HandShapeReport {
            wrist: asset.skeleton.joints[self.wrist].name.clone(),
            status: if reports
                .iter()
                .all(|r| r.violating_edges == 0 && r.max_segment_length_error_m < 1e-5)
            {
                "converged"
            } else {
                "unresolved"
            }
            .into(),
            reset_translation_or_scale_joints: reset,
            digits: reports,
        })
    }
}
