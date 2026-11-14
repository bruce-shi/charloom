 use crate::{hand_shape::HandShapeConstraint, surface_topology::SurfaceTopology};
use anyhow::{Result, ensure};
use charloom_core::*;
use nalgebra::Vector3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HandSurfaceRepair {
    pub reference_clips: Vec<String>,
    pub samples_per_clip: usize,
    #[serde(default)]
    pub include_keys_and_midpoints: bool,
    pub iterations: usize,
    pub max_weight_change_l1: f64,
}

pub fn repair(asset: &mut Asset, hand: &HandShapeConstraint) -> Result<Value> {
    let Some(config) = &hand.surface_repair else {
        return Ok(json!({"requested":false,"changed_vertices":0}));
    };
    ensure!(
        !config.reference_clips.is_empty()
            && (3..=65).contains(&config.samples_per_clip)
            && (1..=256).contains(&config.iterations)
            && config.max_weight_change_l1.is_finite()
            && config.max_weight_change_l1 > 0.0
            && config.max_weight_change_l1 <= 2.0,
        "invalid_hand_surface_repair"
    );
    let named = |name: &str| -> Result<usize> {
        let found: Vec<_> = asset
            .skeleton
            .joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.name == name)
            .map(|(i, _)| i)
            .collect();
        ensure!(
            found.len() == 1,
            "hand_surface_joint_must_be_unique: {name}"
        );
        Ok(found[0])
    };
    let wrist = named(&hand.wrist)?;
    let digits: HashSet<_> = hand
        .digits
        .iter()
        .flat_map(|d| &d.joints)
        .map(|n| named(n))
        .collect::<Result<_>>()?;
    let mut family = digits.clone();
    family.insert(wrist);
    // Existing forearm influences at the palm transition may also need repair.
    // Never import a new forearm influence into a digit-only vertex.
    if let Some(parent) = asset.skeleton.joints[wrist].parent {
        family.insert(parent);
    }
    let mut frames = Vec::new();
    for name in &config.reference_clips {
        let clips: Vec<_> = asset.clips.iter().filter(|c| &c.name == name).collect();
        ensure!(clips.len() == 1, "hand_surface_clip_must_be_unique: {name}");
        let clip = clips[0];
        let range = clip.time_range();
        let mut times: Vec<_> = (0..config.samples_per_clip)
            .map(|i| {
                range[0] + (range[1] - range[0]) * i as f64 / (config.samples_per_clip - 1) as f64
            })
            .collect();
        if config.include_keys_and_midpoints {
            let mut keys: Vec<_> = clip
                .tracks
                .iter()
                .flat_map(|t| t.times.iter().copied())
                .collect();
            keys.sort_by(f64::total_cmp);
            keys.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
            times.extend(keys.iter().copied());
            times.extend(keys.windows(2).map(|p| (p[0] + p[1]) * 0.5));
        }
        times.sort_by(f64::total_cmp);
        times.dedup_by(|a, b| (*a - *b).abs() < 1e-8);
        ensure!(times.len() <= 10000, "too_many_hand_reference_samples");
        for t in times {
            frames.push((
                asset
                    .skeleton
                    .world_matrices(&clip.sample(&asset.skeleton, t)?)?,
                clip.sample_morphs(t)?,
            ));
        }
    }
    let mut reports = Vec::new();
    let mut changed = 0;
    for mesh in &mut asset.meshes {
        let Some(skin_id) = mesh.skin else { continue };
        if !mesh.influences.iter().any(|row| {
            row.iter()
                .any(|w| digits.contains(&w.joint) && w.weight >= 0.01)
        }) {
            continue;
        }
        let selected: Vec<_> = mesh
            .influences
            .iter()
            .map(|r| {
                r.iter()
                    .any(|w| (digits.contains(&w.joint) || w.joint == wrist) && w.weight >= 0.01)
            })
            .collect();
        if !selected.iter().any(|v| *v) {
            continue;
        }
        let mut original = mesh.influences.clone();
        // A rigid vertex may need its immediate anatomical neighbor to form
        // a joint transition. Never import a different finger's palette.
        for row in &mut original {
            if !row
                .iter()
                .any(|w| family.contains(&w.joint) && w.weight > 0.0)
            {
                continue;
            }
            let existing = row.clone();
            for w in &existing {
                if w.joint != wrist
                    && digits.contains(&w.joint)
                    && let Some(parent) = asset.skeleton.joints[w.joint].parent
                    && family.contains(&parent)
                    && !row.iter().any(|w| w.joint == parent)
                    && row.len() < 8
                {
                    row.push(Influence {
                        joint: parent,
                        weight: 0.0,
                    });
                }
            }
        }
        for row in &mut original {
            if row
                .iter()
                .any(|w| family.contains(&w.joint) && w.weight > 0.0)
                && !row.iter().any(|w| w.joint == wrist)
            {
                row.push(Influence {
                    joint: wrist,
                    weight: 0.0,
                });
            }
        }
        let topology = SurfaceTopology::new(mesh);
        let same_row = |a: usize, b: usize| {
            original[a].len() == original[b].len()
                && original[a]
                    .iter()
                    .zip(&original[b])
                    .all(|(a, b)| a.joint == b.joint && (a.weight - b.weight).abs() < 1e-8)
        };
        let representative: Vec<_> = topology
            .representative
            .iter()
            .enumerate()
            .map(|(i, &r)| if same_row(i, r) { r } else { i })
            .collect();
        let mut group = vec![Vec::new(); mesh.positions.len()];
        for (i, &r) in representative.iter().enumerate() {
            group[r].push(i);
        }
        let mut unique = HashSet::new();
        let mut edges = Vec::new();
        let mut used = HashSet::new();
        for tri in &mesh.triangles {
            for k in 0..3 {
                let (i, j) = (tri[k] as usize, tri[(k + 1) % 3] as usize);
                if !selected[i] && !selected[j] {
                    continue;
                }
                let (a, b) = (
                    representative[i].min(representative[j]),
                    representative[i].max(representative[j]),
                );
                if a == b || !unique.insert((a, b)) {
                    continue;
                }
                let rest = (v3(mesh.positions[a]) - v3(mesh.positions[b])).norm();
                if rest < 1e-7 {
                    continue;
                }
                edges.push((
                    a,
                    b,
                    rest * hand.max_edge_ratio + hand.max_edge_growth_m * 0.75,
                ));
                used.extend([a, b]);
            }
        }
        let mut vertices: Vec<_> = used.into_iter().collect();
        vertices.sort_unstable();
        let index: HashMap<_, _> = vertices.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let edges: Vec<_> = edges
            .into_iter()
            .map(|(a, b, l)| (index[&a], index[&b], l))
            .collect();
        let editable: Vec<_> = vertices
            .iter()
            .map(|&v| {
                original[v]
                    .iter()
                    .filter(|w| family.contains(&w.joint))
                    .count()
                    > 1
            })
            .collect();
        let mut weights: Vec<Vec<_>> = vertices
            .iter()
            .map(|&v| original[v].iter().map(|w| w.weight).collect())
            .collect();
        let prior = weights.clone();
        let skin = &asset.skins[skin_id];
        let binds: HashMap<_, _> = skin
            .joints
            .iter()
            .copied()
            .zip(skin.inverse_bind_matrices.iter().map(|m| matrix_from(*m)))
            .collect();
        // Global motion does not change edge lengths. Deduplicate the
        // actual local skin transforms before allocating per-vertex samples.
        let used_joints: HashSet<_> = vertices
            .iter()
            .flat_map(|&v| original[v].iter().map(|w| w.joint))
            .collect();
        let mut used_joints: Vec<_> = used_joints.into_iter().collect();
        used_joints.sort_unstable();
        let mut seen_frames = HashSet::new();
        let mut frame_ids = Vec::new();
        for (index, (world, morphs)) in frames.iter().enumerate() {
            let rotation = crate::solve::world_rotation(&world[wrist])
                .unwrap_or_else(|_| nalgebra::UnitQuaternion::identity());
            let neutral = rotation.inverse().to_homogeneous()
                * nalgebra::Translation3::from(-v3(transform_point(&world[wrist], [0.0; 3])))
                    .to_homogeneous();
            let mut key: Vec<i64> = used_joints
                .iter()
                .flat_map(|&j| {
                    (neutral * world[j])
                        .as_slice()
                        .iter()
                        .map(|x| (x * 1e8).round() as i64)
                        .collect::<Vec<_>>()
                })
                .collect();
            key.extend(
                morphs
                    .get(&mesh.node)
                    .unwrap_or(&mesh.morph_weights)
                    .iter()
                    .map(|x| (x * 1e8).round() as i64),
            );
            if seen_frames.insert(key) {
                frame_ids.push((index, neutral));
            }
        }
        let world_frames = &frames;
        let frames = frame_ids;
        let transformed: Vec<Vec<Vec<Vector3<f64>>>> = frames
            .iter()
            .map(|(index, neutral)| {
                let (world, morphs) = &world_frames[*index];
                let morph = morphs.get(&mesh.node).unwrap_or(&mesh.morph_weights);
                vertices
                    .iter()
                    .map(|&v| {
                        let mut p = v3(mesh.positions[v]);
                        for (target, &weight) in mesh.morph_targets.iter().zip(morph) {
                            p += v3(target[v]) * weight;
                        }
                        original[v]
                            .iter()
                            .map(|w| {
                                v3(transform_point(
                                    &(neutral * world[w.joint] * binds[&w.joint]),
                                    a3(p),
                                ))
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let point = |frame: usize, v: usize, w: &Vec<Vec<f64>>| {
            transformed[frame][v]
                .iter()
                .zip(&w[v])
                .fold(Vector3::zeros(), |sum, (p, w)| sum + p * *w)
        };
        let mut points: Vec<Vec<_>> = (0..frames.len())
            .map(|f| (0..vertices.len()).map(|v| point(f, v, &weights)).collect())
            .collect();
        let audit = |points: &Vec<Vec<Vector3<f64>>>| -> (f64, usize, f64) {
            let mut worst = 0.0_f64;
            let mut count = 0;
            let mut energy = 0.0;
            for p in points {
                for &(a, b, limit) in &edges {
                    let excess = (p[a] - p[b]).norm() - limit;
                    worst = worst.max(excess);
                    if excess > 1e-7 {
                        count += 1;
                        energy += excess * excess;
                    }
                }
            }
            (worst, count, energy)
        };
        let before = audit(&points);
        let mut best = weights.clone();
        let mut best_score = before;
        let mut feasible_prior = prior.clone();
        let movable: Vec<Vec<bool>> = vertices
            .iter()
            .map(|&v| {
                original[v]
                    .iter()
                    .map(|w| family.contains(&w.joint))
                    .collect()
            })
            .collect();
        let masses: Vec<f64> = prior
            .iter()
            .zip(&movable)
            .map(|(w, m)| w.iter().zip(m).filter(|(_, ok)| **ok).map(|(w, _)| w).sum())
            .collect();
        let mut allowed = movable.clone();
        let optimization_tolerance = hand.max_edge_growth_m * 0.1 + 1e-8;
        let mut iteration = 0;
        while iteration < config.iterations + 128 {
            if iteration == config.iterations {
                weights = best.clone();
                for (v, row) in weights.iter_mut().enumerate() {
                    if masses[v] < 1e-12 {
                        continue;
                    }
                    let mut order: Vec<_> = (0..row.len()).filter(|&i| movable[v][i]).collect();
                    let capacity = 4 - prior[v]
                        .iter()
                        .zip(&movable[v])
                        .filter(|(w, m)| !**m && **w > 1e-12)
                        .count();
                    order.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                    let dropped = order
                        .iter()
                        .skip(capacity)
                        .map(|&i| prior[v][i])
                        .sum::<f64>();
                    if 2.0 * dropped > config.max_weight_change_l1 {
                        order.sort_by(|&a, &b| prior[v][b].total_cmp(&prior[v][a]));
                    }
                    for &i in order.iter().skip(capacity) {
                        allowed[v][i] = false;
                        row[i] = 0.0;
                        feasible_prior[v][i] = 0.0;
                    }
                    for values in [&mut *row, &mut feasible_prior[v]] {
                        let sum = values
                            .iter()
                            .zip(&movable[v])
                            .filter(|(_, ok)| **ok)
                            .map(|(w, _)| w)
                            .sum::<f64>();
                        if sum > 1e-12 {
                            for (w, ok) in values.iter_mut().zip(&movable[v]) {
                                if *ok {
                                    *w *= masses[v] / sum;
                                }
                            }
                        }
                    }
                    if row
                        .iter()
                        .zip(&prior[v])
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f64>()
                        > config.max_weight_change_l1
                    {
                        row.clone_from(&feasible_prior[v]);
                    }
                }
                for (f, p) in points.iter_mut().enumerate() {
                    for (v, x) in p.iter_mut().enumerate() {
                        *x = point(f, v, &weights);
                    }
                }
                best = weights.clone();
                best_score = audit(&points);
            }
            for f in 0..frames.len() {
                for (v, p) in points[f].iter_mut().enumerate() {
                    *p = point(f, v, &weights);
                }
                for &(a, b, limit) in &edges {
                    let delta = points[f][a] - points[f][b];
                    let length = delta.norm();
                    let excess = length - limit;
                    if excess <= 1e-7 || length < 1e-10 {
                        continue;
                    }
                    let direction = delta / length;
                    let gradients: [Vec<f64>; 2] = [(a, 1.0), (b, -1.0)].map(|(v, sign)| {
                        if !editable[v] || masses[v] < 1e-12 {
                            return vec![0.0; weights[v].len()];
                        }
                        let mut g: Vec<_> = transformed[f][v]
                            .iter()
                            .map(|p| direction.dot(p) * sign)
                            .collect();
                        let mean = g
                            .iter()
                            .zip(&allowed[v])
                            .filter(|(_, ok)| **ok)
                            .map(|(g, _)| g)
                            .sum::<f64>()
                            / allowed[v].iter().filter(|ok| **ok).count() as f64;
                        for (x, ok) in g.iter_mut().zip(&allowed[v]) {
                            *x = if *ok { *x - mean } else { 0.0 };
                        }
                        g
                    });
                    let norm = gradients.iter().flatten().map(|g| g * g).sum::<f64>();
                    if norm < 1e-14 {
                        continue;
                    }
                    let step = 0.65 * excess / norm;
                    for (v, g) in [a, b].into_iter().zip(gradients) {
                        if !editable[v] || masses[v] < 1e-12 {
                            continue;
                        }
                        let mut candidate: Vec<_> = weights[v]
                            .iter()
                            .zip(g)
                            .map(|(w, g)| w - step * g)
                            .collect();
                        let mut selected: Vec<_> = candidate
                            .iter()
                            .zip(&allowed[v])
                            .filter(|(_, ok)| **ok)
                            .map(|(w, _)| *w)
                            .collect();
                        for w in &mut selected {
                            *w /= masses[v];
                        }
                        crate::skin_repair::simplex(&mut selected);
                        for w in &mut selected {
                            *w *= masses[v];
                        }
                        let mut at = 0;
                        for (i, (w, ok)) in candidate.iter_mut().zip(&allowed[v]).enumerate() {
                            *w = if *ok {
                                let value = selected[at];
                                at += 1;
                                value
                            } else if movable[v][i] {
                                0.0
                            } else {
                                prior[v][i]
                            };
                        }
                        let distance = candidate
                            .iter()
                            .zip(&prior[v])
                            .map(|(a, b)| (a - b).abs())
                            .sum::<f64>();
                        if distance > config.max_weight_change_l1 {
                            let mut low = 0.0;
                            let mut high = 1.0;
                            for _ in 0..24 {
                                let t = (low + high) * 0.5;
                                let d = candidate
                                    .iter()
                                    .zip(&feasible_prior[v])
                                    .zip(&prior[v])
                                    .map(|((w, b), p)| (b + (w - b) * t - p).abs())
                                    .sum::<f64>();
                                if d > config.max_weight_change_l1 {
                                    high = t;
                                } else {
                                    low = t;
                                }
                            }
                            for (w, b) in candidate.iter_mut().zip(&feasible_prior[v]) {
                                *w = b + (*w - b) * low;
                            }
                        }
                        weights[v] = candidate;
                        points[f][v] = point(f, v, &weights);
                    }
                }
            }
            for (f, p) in points.iter_mut().enumerate() {
                for (v, x) in p.iter_mut().enumerate() {
                    *x = point(f, v, &weights);
                }
            }
            let score = audit(&points);
            if score.2 < best_score.2 {
                best_score = score;
                best = weights.clone();
            }
            if score.0 <= optimization_tolerance {
                if iteration < config.iterations {
                    iteration = config.iterations;
                    continue;
                }
                break;
            }
            iteration += 1;
        }
        let mut local_changed = 0;
        let mut max_l1 = 0.0_f64;
        for (i, &v) in vertices.iter().enumerate() {
            let l1 = best[i]
                .iter()
                .zip(&prior[i])
                .map(|(a, b)| (a - b).abs())
                .sum::<f64>();
            if l1 <= 1e-9 {
                continue;
            }
            max_l1 = max_l1.max(l1);
            for &member in &group[v] {
                mesh.influences[member] = original[v]
                    .iter()
                    .zip(&best[i])
                    .filter(|(_, weight)| **weight > 1e-12)
                    .map(|(w, &weight)| Influence {
                        joint: w.joint,
                        weight,
                    })
                    .collect();
                local_changed += 1;
            }
        }
        changed += local_changed;
        reports.push(json!({"mesh":mesh.name,"changed_vertices":local_changed,"max_weight_change_l1":max_l1,"before_max_excess_m":before.0,"after_max_excess_m":best_score.0,"remaining_edge_samples":best_score.1,"distinct_local_pose_samples":frames.len(),"optimization_tolerance_m":optimization_tolerance,"edge_growth_reserve_m":hand.max_edge_growth_m*0.25}));
    }
    Ok(
        json!({"wrist":hand.wrist,"requested":true,"method":"fixed_pose_hand_weight_projection_v2","pose_samples":frames.len(),"changed_vertices":changed,"meshes":reports,"geometry_unchanged":true,"animation_unchanged":true}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hand_shape::DigitShape;
    use nalgebra::UnitQuaternion;
    #[test]
    fn fixed_grip_weights_reduce_strain_without_opening_or_stretching_the_finger() {
        let joint = |name: &str, parent, position| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(position),
            matrix: None,
            deform: true,
        };
        let skeleton = Skeleton {
            joints: vec![
                joint("wrist", None, [0.0; 3]),
                joint("finger1", Some(0), [0.0, 1.0, 0.0]),
                joint("finger2", Some(1), [0.0, 0.5, 0.0]),
            ],
        };
        let world = skeleton.world_matrices(&skeleton.rest_pose()).unwrap();
        let rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 1.1));
        let clip = MotionClip {
            name: "grip".into(),
            tracks: vec![Track {
                joint: 2,
                property: Property::Rotation,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 1.0],
                values: vec![rotation; 2],
            }],
            morph_tracks: vec![],
        };
        let asset = Asset {
            name: "hand".into(),
            source_hash: String::new(),
            skeleton,
            skins: vec![Skin {
                name: "hand".into(),
                joints: vec![0, 1, 2],
                inverse_bind_matrices: world
                    .iter()
                    .map(|w| matrix_array(&w.try_inverse().unwrap()))
                    .collect(),
            }],
            meshes: vec![Mesh {
                name: "hand".into(),
                node: 0,
                primitive: 0,
                positions: vec![[0.01, 1.8, 0.0], [-0.01, 1.8, 0.0], [0.0, 1.81, 0.01]],
                triangles: vec![[0, 1, 2]],
                skin: Some(0),
                influences: vec![
                    vec![
                        Influence {
                            joint: 1,
                            weight: 0.75,
                        },
                        Influence {
                            joint: 2,
                            weight: 0.25,
                        },
                    ],
                    vec![
                        Influence {
                            joint: 1,
                            weight: 0.25,
                        },
                        Influence {
                            joint: 2,
                            weight: 0.75,
                        },
                    ],
                    vec![
                        Influence {
                            joint: 1,
                            weight: 0.5,
                        },
                        Influence {
                            joint: 2,
                            weight: 0.5,
                        },
                    ],
                ],
                morph_targets: vec![],
                morph_weights: vec![],
            }],
            clips: vec![clip],
            issues: vec![],
        };
        let baseline = asset;
        for foreign in [0.0, 0.1] {
            let mut asset = baseline.clone();
            asset.skeleton.joints.push(joint("cloth", None, [0.0; 3]));
            asset.skins[0].joints.push(3);
            asset.skins[0]
                .inverse_bind_matrices
                .push(matrix_array(&nalgebra::Matrix4::identity()));
            if foreign > 0.0 {
                for row in &mut asset.meshes[0].influences {
                    for w in row.iter_mut() {
                        w.weight *= 1.0 - foreign;
                    }
                    row.push(Influence {
                        joint: 3,
                        weight: foreign,
                    });
                }
            }
            let before = asset.clone();
            let report = repair(
                &mut asset,
                &HandShapeConstraint {
                    wrist: "wrist".into(),
                    digits: vec![DigitShape {
                        name: "finger".into(),
                        joints: vec!["finger1".into(), "finger2".into()],
                    }],
                    rebind_missing_digits: false,
                    allow_rotation_fallback: false,
                    max_edge_ratio: 1.2,
                    max_edge_growth_m: 0.004,
                    surface_corrective: None,
                    surface_repair: Some(HandSurfaceRepair {
                        reference_clips: vec!["grip".into()],
                        samples_per_clip: 3,
                        include_keys_and_midpoints: true,
                        iterations: 64,
                        max_weight_change_l1: 0.6,
                    }),
                },
            )
            .unwrap();
            assert!(report["meshes"][0]["before_max_excess_m"].as_f64().unwrap() > 0.05);
            assert!(
                report["meshes"][0]["after_max_excess_m"].as_f64().unwrap()
                    < report["meshes"][0]["edge_growth_reserve_m"]
                        .as_f64()
                        .unwrap(),
                "{report}"
            );
            let pose = asset.clips[0].sample(&asset.skeleton, 0.5).unwrap();
            let world = asset.skeleton.world_matrices(&pose).unwrap();
            let mesh = &asset.meshes[0];
            let points: Vec<_> = mesh
                .positions
                .iter()
                .zip(&mesh.influences)
                .map(|(&p, row)| {
                    row.iter().fold(Vector3::zeros(), |sum, w| {
                        sum + v3(transform_point(
                            &(world[w.joint]
                                * matrix_from(asset.skins[0].inverse_bind_matrices[w.joint])),
                            p,
                        )) * w.weight
                    })
                })
                .collect();
            for [a, b, c] in &mesh.triangles {
                for (a, b) in [(*a, *b), (*b, *c), (*c, *a)] {
                    let (a, b) = (a as usize, b as usize);
                    let bind = (v3(mesh.positions[a]) - v3(mesh.positions[b])).norm();
                    assert!((points[a] - points[b]).norm() <= bind * 1.2 + 0.004);
                }
            }
            assert_eq!(asset.meshes[0].positions, before.meshes[0].positions);
            assert_eq!(asset.skeleton.rest_pose(), before.skeleton.rest_pose());
            assert_eq!(
                serde_json::to_value(&asset.clips).unwrap(),
                serde_json::to_value(&before.clips).unwrap()
            );
            for (new, old) in asset.meshes[0]
                .influences
                .iter()
                .zip(&before.meshes[0].influences)
            {
                assert!((new.iter().map(|w| w.weight).sum::<f64>() - 1.0).abs() < 1e-12);
                assert!(new.len() <= 4);
                assert!(
                    (new.iter()
                        .filter(|w| w.joint == 3)
                        .map(|w| w.weight)
                        .sum::<f64>()
                        - foreign)
                        .abs()
                        < 1e-12
                );
                assert!(crate::surface_topology::weight_l1(new, old) <= 0.6 + 1e-10);
            }
        }
    }
}
