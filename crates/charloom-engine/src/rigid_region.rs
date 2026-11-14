//! Preserve rigid bone cores while retaining feathered joint transitions.
//! Regions are explicit characterization inputs, not material classification.
use anyhow::{Result, ensure};
use charloom_core::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RigidBoneRegion {
    pub joint: String,
    pub end: String,
    pub influences: Vec<String>,
    /// Rigid interval along the bone, followed by a smooth falloff on each side.
    pub interval: [f64; 2],
    pub feather: [f64; 2],
    pub minimum_region_weight: f64,
    pub radius_quantile: f64,
}

pub fn apply_regions(
    asset: &mut Asset,
    regions: &[RigidBoneRegion],
) -> Result<Vec<serde_json::Value>> {
    ensure!(regions.len() <= 64, "too_many_rigid_regions");
    let mut reports = Vec::new();
    for region in regions {
        ensure!(
            region
                .interval
                .iter()
                .chain(&region.feather)
                .all(|v| v.is_finite())
                && region.interval[0] < region.interval[1]
                && region.feather.iter().all(|v| *v > 0.0)
                && (0.5..=1.0).contains(&region.minimum_region_weight)
                && (0.5..=1.0).contains(&region.radius_quantile),
            "invalid_rigid_region"
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
                "rigid_region_joint_must_be_unique: {name}"
            );
            Ok(found[0])
        };
        let joint = named(&region.joint)?;
        let end = named(&region.end)?;
        ensure!(
            asset.skeleton.joints[end].parent == Some(joint),
            "rigid_region_requires_direct_bone"
        );
        let selected: HashSet<_> = region
            .influences
            .iter()
            .map(|n| named(n))
            .collect::<Result<_>>()?;
        ensure!(
            selected.contains(&joint),
            "rigid_region_owner_must_be_selected"
        );
        let smooth = |x: f64| {
            let x = x.clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        };
        let mut changed = 0;
        let mut rigid = 0;
        let mut maximum_l1 = 0.0_f64;
        let mut bind_regions = Vec::new();
        for (skin_index, skin) in asset.skins.iter().enumerate() {
            let bind_point = |node: usize| -> Result<Option<nalgebra::Vector3<f64>>> {
                let Some(i) = skin.joints.iter().position(|&j| j == node) else {
                    return Ok(None);
                };
                let Some(world) = matrix_from(skin.inverse_bind_matrices[i]).try_inverse() else {
                    anyhow::bail!("singular_rigid_region_bind")
                };
                Ok(Some(v3(transform_point(&world, [0.0; 3]))))
            };
            let (Some(start), Some(end)) = (bind_point(joint)?, bind_point(end)?) else {
                continue;
            };
            let delta = end - start;
            ensure!(delta.norm() > 1e-8, "rigid_region_zero_length");
             
            let project = |p: Vec3| {
                let offset = v3(p) - start;
                let t = offset.dot(&delta) / delta.norm_squared();
                (t, (offset - delta * t).norm())
            };
            let mut radii = Vec::new();
            for mesh in asset.meshes.iter().filter(|m| m.skin == Some(skin_index)) {
                for (p, row) in mesh.positions.iter().zip(&mesh.influences) {
                    let weight = row
                        .iter()
                        .filter(|w| w.joint == joint)
                        .map(|w| w.weight)
                        .sum::<f64>();
                    let (t, r) = project(*p);
                    if weight >= 0.5 && t >= region.interval[0] && t <= region.interval[1] {
                        radii.push(r);
                    }
                }
            }
            if radii.is_empty() {
                continue;
            }
            radii.sort_by(f64::total_cmp);
            let radius =
                radii[((radii.len() - 1) as f64 * region.radius_quantile).round() as usize];
            bind_regions.push(serde_json::json!({"skin":skin_index,"radius_bind_units":radius}));
            for mesh in asset
                .meshes
                .iter_mut()
                .filter(|m| m.skin == Some(skin_index))
            {
                for (vi, row) in mesh.influences.iter_mut().enumerate() {
                    let mass = row
                        .iter()
                        .filter(|w| selected.contains(&w.joint))
                        .map(|w| w.weight)
                        .sum::<f64>();
                    if mass < region.minimum_region_weight {
                        continue;
                    }
                    let owner_participation = row
                        .iter()
                        .filter(|w| w.joint == joint)
                        .map(|w| w.weight)
                        .sum::<f64>();
                    let (t, r) = project(mesh.positions[vi]);
                    let fraction =
                        smooth((t - region.interval[0] + region.feather[0]) / region.feather[0])
                            * smooth(
                                (region.interval[1] + region.feather[1] - t) / region.feather[1],
                            )
                            * smooth((radius * 1.15 - r) / (radius * 0.15).max(1e-8))
                            * smooth(owner_participation / 0.05);
                    if fraction < 1e-8 {
                        continue;
                    }
                    let old_owner = row
                        .iter()
                        .filter(|w| w.joint == joint)
                        .map(|w| w.weight)
                        .sum::<f64>();
                    let moved = (mass - old_owner) * fraction;
                    if moved < 1e-10 {
                        continue;
                    }
                    maximum_l1 = maximum_l1.max(2.0 * moved);
                    for w in row.iter_mut() {
                        if selected.contains(&w.joint) {
                            w.weight *= 1.0 - fraction;
                        }
                    }
                    if let Some(owner) = row.iter_mut().find(|w| w.joint == joint) {
                        owner.weight += mass * fraction;
                    } else {
                        row.push(Influence {
                            joint,
                            weight: mass * fraction,
                        });
                    }
                    row.retain(|w| w.weight > 1e-10);
                    row.sort_by(|a, b| b.weight.total_cmp(&a.weight));
                    row.truncate(4);
                    let sum = row.iter().map(|w| w.weight).sum::<f64>();
                    for w in row.iter_mut() {
                        w.weight /= sum;
                    }
                    changed += 1;
                    if row.len() == 1 && row[0].joint == joint {
                        rigid += 1;
                    }
                }
            }
        }
        ensure!(
            !bind_regions.is_empty(),
            "rigid_region_has_no_weighted_surface"
        );
        reports.push(serde_json::json!({"joint":region.joint,"end":region.end,"bind_regions":bind_regions,"coordinate_space":"skin_bind","changed_vertices":changed,"fully_rigid_vertices":rigid,"maximum_weight_change_l1":maximum_l1,"geometry_changed":false}));
    }
    Ok(reports)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};
    #[test]
    fn cuff_core_keeps_its_diameter_under_twist_with_a_flexible_wrist_edge() {
        let skeleton = Skeleton {
            joints: vec![
                Joint {
                    id: "forearm".into(),
                    name: "forearm".into(),
                    parent: None,
                    rest: Transform::default(),
                    matrix: None,
                    deform: true,
                },
                Joint {
                    id: "hand".into(),
                    name: "hand".into(),
                    parent: Some(0),
                    rest: Transform::translated([0., 1., 0.]),
                    matrix: None,
                    deform: true,
                },
            ],
        };
        let world = skeleton.world_matrices(&skeleton.rest_pose()).unwrap();
        let mut asset = Asset {
            name: "cuff".into(),
            source_hash: String::new(),
            skins: vec![Skin {
                name: "arm".into(),
                joints: vec![0, 1],
                inverse_bind_matrices: world
                    .iter()
                    .map(|w| matrix_array(&w.try_inverse().unwrap()))
                    .collect(),
            }],
            skeleton,
            meshes: vec![Mesh {
                name: "cuff".into(),
                node: 0,
                primitive: 0,
                skin: Some(0),
                positions: vec![
                    [0.1, 0.5, 0.],
                    [-0.1, 0.5, 0.],
                    [0., 0.5, 0.1],
                    [0., 0.5, -0.1],
                    [0.1, 0.98, 0.],
                    [-0.1, 0.98, 0.],
                ],
                triangles: vec![[0, 1, 2], [0, 2, 3]],
                influences: vec![
                    vec![
                        Influence {
                            joint: 0,
                            weight: 0.6
                        },
                        Influence {
                            joint: 1,
                            weight: 0.4
                        }
                    ];
                    6
                ],
                morph_targets: vec![],
                morph_weights: vec![],
            }],
            clips: vec![],
            issues: vec![],
        };
        let original = asset.clone();
        let region = RigidBoneRegion {
            joint: "forearm".into(),
            end: "hand".into(),
            influences: vec!["forearm".into(), "hand".into()],
            interval: [0.2, 0.9],
            feather: [0.1, 0.1],
            minimum_region_weight: 0.98,
            radius_quantile: 1.0,
        };
        let mut posed_default = asset.clone();
        posed_default.skeleton.joints[1].rest.rotation =
            a4(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 2.2));
        let report = apply_regions(&mut asset, std::slice::from_ref(&region)).unwrap();
        apply_regions(&mut posed_default, &[region]).unwrap();
        assert_eq!(
            serde_json::to_value(&asset.meshes[0].influences).unwrap(),
            serde_json::to_value(&posed_default.meshes[0].influences).unwrap(),
            "weight regions must use inverse binds, not an arbitrary displayed pose"
        );
        assert_eq!(report[0]["fully_rigid_vertices"], 4);
        assert_eq!(asset.meshes[0].positions, original.meshes[0].positions);
        assert_eq!(asset.meshes[0].influences[4].len(), 2);
        for angle in [0., 1., 2., std::f64::consts::PI] {
            let mut pose = asset.skeleton.rest_pose();
            pose[1].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle));
            let vertices = asset.skinned_positions(&pose).unwrap();
            assert!(((v3(vertices[0][0]) - v3(vertices[0][1])).norm() - 0.2).abs() < 1e-12);
        }
    }
}
