//! Reference-pose-aware retargeting. Explicitly reports incomplete mappings.
use crate::solve::world_rotation;
use anyhow::{Context, Result, ensure};
use charloom_core::*;
use nalgebra::{Matrix3, Matrix4, Rotation3, Unit, UnitQuaternion, Vector3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JointMapping {
    pub source: usize,
    pub target: usize,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MappingReport {
    pub mappings: Vec<JointMapping>,
    pub unmapped_source: Vec<String>,
    pub unmapped_target: Vec<String>,
    pub target_deform_coverage: f64,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LimbTrajectory {
    /// Root, middle, orientation joint, end joint. The first three form a
    /// direct two-bone IK chain; orientation limits below are opt-in.
    pub target_chain: [String; 4],
    /// A mapped body anchor, for example Hips. Source endpoints/anchor are
    /// resolved from the verified mapping, not guessed from species names.
    pub target_anchor: String,
    /// Scale the source end-point displacement around the observed target
    /// reference offset. Defaults to the configured root-motion scale.
    #[serde(default)]
    pub position_scale: Option<f64>,
    /// Optional limit around the observed orientation transported with the
    /// body anchor. A long raised hock must not inherit an unrestricted human
    /// heel-strike roll. None retains the full FK orientation.
    #[serde(default)]
    pub max_orientation_deviation_rad: Option<f64>,
    /// Optional end-joint world-orientation limit around the observed end
    /// reference transported with the current body anchor. None preserves the
    /// existing FK end orientation after the proximal orientation correction.
    /// This is an authored adaptation policy, not species recognition.
    #[serde(default)]
    pub max_end_orientation_deviation_rad: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct LimbTrajectoryReport {
    pub target_chain: [String; 4],
    pub max_position_error_m: f64,
    #[serde(default)]
    pub max_orientation_correction_rad: f64,
    #[serde(default)]
    pub max_end_orientation_correction_rad: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct RetargetConfig {
    pub fps: f64,
    pub start: Option<f64>,
    pub end: Option<f64>,
    pub name: String,
    /// Target-name -> source-name overrides. Names must be unique.
    pub joint_map: BTreeMap<String, String>,
    /// A world rotation aligning source reference axes to target reference axes.
    pub alignment: Quat,
    pub root_motion: bool,
    pub root_scale: Option<f64>,
    pub min_target_coverage: f64,
    /// Optional exact target motion-joint roster required by this clip's
    /// authored scope. This changes only the coverage gate, not mappings or
    /// emitted motion. None keeps the full target deform-joint denominator.
    /// Zero-weight roots/controls may be required too; they need an actual
    /// mapping, such as an explicit joint_map entry. Auxiliary inheritance,
    /// digits and facial motion need separate review.
    pub target_joint_scope: Option<Vec<String>>,
         pub align_reference_chains: bool,
         pub distribute_palm_twist: bool,
         pub preserve_reference_joints: Vec<String>,
         pub preserve_local_pose_joints: Vec<String>,
         pub rest_local_reference_joints: Vec<String>,
         pub limb_trajectories: Vec<LimbTrajectory>,
    /// Put axial hand rotation on the forearm while preserving the world hand
    /// pose. Requires uniquely mapped direct forearm/hand pairs.
    pub distribute_wrist_twist: bool,
    /// Explicit serial support joints followed by the mapped Hand, excluding
    /// the mapped ForeArm. For example [Twist1, Twist2, Twist3, Hand]. Supports
    /// remain unmapped and redistribute wrist roll without rotating ForeArm.
    /// Empty keeps the existing retarget behavior.
    pub wrist_twist_chains: Vec<Vec<String>>,
}
impl Default for RetargetConfig {
    fn default() -> Self {
        Self {
            fps: 30.0,
            start: None,
            end: None,
            name: "retargeted".into(),
            joint_map: BTreeMap::new(),
            alignment: [0.0, 0.0, 0.0, 1.0],
            root_motion: true,
            root_scale: None,
            min_target_coverage: 0.5,
            target_joint_scope: None,
            align_reference_chains: false,
            distribute_palm_twist: false,
            preserve_reference_joints: Vec::new(),
            preserve_local_pose_joints: Vec::new(),
            rest_local_reference_joints: Vec::new(),
            limb_trajectories: Vec::new(),
            distribute_wrist_twist: false,
            wrist_twist_chains: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RetargetReport {
    pub mapping: MappingReport,
    /// Explicit clip coverage, when requested. The mapping report above always
    /// retains coverage over every target deform joint, including out-of-scope
    /// joints; scoped success must never be presented as full-rig coverage.
    #[serde(default)]
    pub target_joint_scope: Option<TargetJointScopeReport>,
    pub root_scale: f64,
    pub motion_roots: Vec<String>,
    pub sample_count: usize,
    #[serde(default)]
    pub preserved_local_pose_joints: Vec<String>,
    pub max_local_bone_length_error: f64,
    pub issues: Vec<String>,
    pub reference_chain_corrections: Vec<String>,
    pub twist_distribution_joints: Vec<String>,
    pub max_wrist_twist_transferred_rad: f64,
    #[serde(default)]
    pub wrist_twist_chains: Vec<WristTwistChainReport>,
    #[serde(default)]
    pub limb_trajectories: Vec<LimbTrajectoryReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TargetJointScopeReport {
    pub joints: Vec<TargetJointScopeJoint>,
    pub mapped_joints: Vec<String>,
    pub unmapped_joints: Vec<String>,
    pub target_joint_coverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct TargetJointScopeJoint {
    pub name: String,
    pub deform: bool,
    pub mapped: bool,
}

fn target_joint_scope_report(
    target: &Skeleton,
    mapping: &MappingReport,
    names: &[String],
) -> Result<TargetJointScopeReport> {
    ensure!(!names.is_empty(), "target_joint_scope_must_not_be_empty");
    let mapped: HashSet<_> = mapping.mappings.iter().map(|m| m.target).collect();
    let mut seen = HashSet::new();
    let mut joints = Vec::new();
    let mut mapped_joints = Vec::new();
    let mut unmapped_joints = Vec::new();
    for name in names {
        ensure!(
            seen.insert(name),
            "duplicate_target_joint_scope_name: {name}"
        );
        let joint = unique_name(target, name)?;
        joints.push(TargetJointScopeJoint {
            name: name.clone(),
            deform: target.joints[joint].deform,
            mapped: mapped.contains(&joint),
        });
        if mapped.contains(&joint) {
            mapped_joints.push(name.clone());
        } else {
            unmapped_joints.push(name.clone());
        }
    }
    Ok(TargetJointScopeReport {
        joints,
        target_joint_coverage: mapped_joints.len() as f64 / names.len() as f64,
        mapped_joints,
        unmapped_joints,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct WristTwistChainReport {
    pub forearm: String,
    pub joints: Vec<String>,
    /// Maximum roll moved by any individual child-to-support transfer.
    pub max_transferred_rad: f64,
    /// Maximum rest-relative axial step left on any support or the Hand.
    /// Hand swing is preserved and is not counted as an axial twist step.
    pub max_twist_step_rad: f64,
    pub singular_swing_samples: usize,
}

struct PreparedWristTwistChain {
    forearm: usize,
    joints: Vec<usize>,
    transfers: Vec<(crate::articulation::PreparedTwist, f64)>,
}

fn uniform_positive_twist_scale(scale: Vec3) -> bool {
    scale.iter().all(|s| s.is_finite() && *s > 0.)
        && scale
            .iter()
            .all(|s| (s - scale[0]).abs() <= 1e-10 * s.max(scale[0]))
}

impl PreparedWristTwistChain {
    fn prepare(target: &Skeleton, forearm: usize, joints: Vec<usize>) -> Result<Self> {
        for &i in std::iter::once(&forearm).chain(joints.iter()) {
            ensure!(
                target.joints[i].matrix.is_none(),
                "serial_wrist_twist_requires_trs"
            );
            ensure!(
                uniform_positive_twist_scale(target.joints[i].rest.scale),
                "serial_wrist_twist_requires_uniform_positive_scale"
            );
        }
        for (at, &support) in joints[..joints.len() - 1].iter().enumerate() {
            let child = joints[at + 1];
            ensure!(
                target.joints[child].parent == Some(support),
                "serial_wrist_twist_requires_direct_chain"
            );
            let children: Vec<_> = target
                .joints
                .iter()
                .enumerate()
                .filter(|(_, j)| j.parent == Some(support))
                .map(|(i, _)| i)
                .collect();
            ensure!(children == [child], "serial_wrist_twist_support_has_branch");
            let incoming = v3(target.joints[support].rest.translation)
                .try_normalize(1e-9)
                .context("serial_wrist_twist_zero_axis")?;
            let outgoing = unit_quat(target.joints[support].rest.rotation)
                * v3(target.joints[child].rest.translation)
                    .try_normalize(1e-9)
                    .context("serial_wrist_twist_zero_axis")?;
            ensure!(
                incoming.dot(&outgoing) > 1.0 - 1e-10,
                "serial_wrist_twist_axes_must_be_collinear_and_forward"
            );
        }
        let transfers = (1..joints.len())
            .map(|i| {
                Ok((
                    crate::articulation::PreparedTwist::prepare(target, joints[i])?,
                    i as f64 / (i + 1) as f64,
                ))
            })
            .collect::<Result<_>>()?;
        Ok(Self {
            forearm,
            joints,
            transfers,
        })
    }

    fn apply(&self, target: &Skeleton, pose: &mut [Transform]) -> Result<(f64, f64, bool)> {
        for &i in std::iter::once(&self.forearm).chain(self.joints.iter()) {
            ensure!(
                pose[i].is_valid() && uniform_positive_twist_scale(pose[i].scale),
                "serial_wrist_twist_requires_uniform_positive_pose_scale"
            );
        }
        let hand = *self.joints.last().unwrap();
        let before = target.world_matrices(pose)?;
        let mut maximum = 0.0_f64;
        let mut step = 0.0_f64;
        let mut first_support = 0.;
        let mut singular = false;
        // N-1/N, N-2/N-1, ..., 1/2 leaves the same axial step on every
        // support and Hand. The ForeArm is never a transfer destination.
        for (transfer, fraction) in self.transfers.iter().rev() {
            let report = transfer.apply_fraction(target, pose, *fraction)?;
            maximum = maximum.max(report.transferred_rad);
            step = step.max(report.transferred_rad * (1.0 - fraction) / fraction);
            first_support = report.transferred_rad;
            singular |= report.singular_swing;
        }
        step = step.max(first_support);
        let after = target.world_matrices(pose)?;
        ensure!(
            (before[hand] - after[hand]).norm() < 1e-9 * before[hand].norm().max(1.)
                && (before[self.forearm] - after[self.forearm]).norm() < 1e-12,
            "serial_wrist_twist_changed_hand_or_forearm_world"
        );
        Ok((maximum, step, singular))
    }
}

/// The two independent directions are needed: wrist-to-middle alone leaves an
/// arbitrary roll which can turn a copied finger curl through the back of the
/// target hand. Index-to-little uses the same anatomical ordering on both sides.
fn palm_frame(
    wrist: Vector3<f64>,
    middle: Vector3<f64>,
    index: Vector3<f64>,
    little: Vector3<f64>,
) -> Option<UnitQuaternion<f64>> {
    let distal = (middle - wrist).try_normalize(1e-7)?;
    let spread = index - little;
    let lateral = (spread - distal * spread.dot(&distal)).try_normalize(1e-7)?;
    let normal = lateral.cross(&distal);
    Some(UnitQuaternion::from_rotation_matrix(
        &Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[lateral, distal, normal])),
    ))
}

/// A facial forward marker disambiguates a turned head in a posed GLB default.
/// If the source has no facial marker, a caller-selected humanoid reference
/// uses its torso facing; the report records that fallback explicitly.
fn marked_head_frame(
    skeleton: &Skeleton,
    world: &[nalgebra::Matrix4<f64>],
    head: usize,
    fallback_forward: Option<Vector3<f64>>,
) -> Option<UnitQuaternion<f64>> {
    let point = |i| v3(transform_point(&world[i], [0.0; 3]));
    let child = |names: &[&str]| {
        skeleton.joints.iter().enumerate().find_map(|(i, joint)| {
            (joint.parent == Some(head) && names.contains(&compact(&joint.name).as_str()))
                .then_some(i)
        })
    };
    let up = point(child(&["headend", "headtopend"])?) - point(head);
    let forward = child(&["headfront"])
        .map(|i| point(i) - point(head))
        .or(fallback_forward)?
        .try_normalize(1e-7)?;
    let up = (up - forward * up.dot(&forward)).try_normalize(1e-7)?;
    let right = up.cross(&forward);
    Some(UnitQuaternion::from_rotation_matrix(
        &Rotation3::from_matrix_unchecked(Matrix3::from_columns(&[right, up, forward])),
    ))
}

fn digit_rank(key: &str) -> Option<usize> {
    let mut fields = key.split('_');
    let side = fields.next()?;
    let digit = fields.next()?;
    let rank = fields.next()?.parse::<usize>().ok()?;
    if !["left", "right"].contains(&side)
        || !["thumb", "index", "middle", "ring", "little"].contains(&digit)
        || fields.next().is_some()
    {
        return None;
    }
    Some(rank)
}

/// Some rigs omit terminal finger tips. Only infer their forward axis when the
/// preceding two phalanges provide a consistent local-axis convention. Never
/// assume every imported rig uses +Y, or treat an arbitrary leaf as a finger.
fn terminal_digit_direction(
    skeleton: &Skeleton,
    world: &[nalgebra::Matrix4<f64>],
    keys: &[String],
    joint: usize,
) -> Result<Option<Vector3<f64>>> {
    let Some(rank) = digit_rank(&keys[joint]).filter(|rank| *rank >= 3) else {
        return Ok(None);
    };
    let Some(parent) = skeleton.joints[joint].parent else {
        return Ok(None);
    };
    let stem = keys[joint].rsplit_once('_').unwrap().0;
    if keys[parent] != format!("{stem}_{}", rank - 1) {
        return Ok(None);
    }
    let Some(a) = v3(skeleton.joints[joint].rest.translation).try_normalize(1e-7) else {
        return Ok(None);
    };
    let Some(b) = v3(skeleton.joints[parent].rest.translation).try_normalize(1e-7) else {
        return Ok(None);
    };
    if a.dot(&b) < 0.98 {
        return Ok(None);
    }
    Ok(Some(world_rotation(&world[joint])? * (a + b).normalize()))
}

fn align_reference_pose(
    source: &Skeleton,
    target: &Skeleton,
    mappings: &[JointMapping],
    alignment: UnitQuaternion<f64>,
    preserved: &HashSet<usize>,
    rest_local: &HashSet<usize>,
    distribute_palm_twist: bool,
) -> Result<(Vec<Transform>, Vec<String>, Vec<usize>)> {
    let mut supports = Vec::new();
    let source_world = source.world_matrices(&source.rest_pose())?;
    let mut pose = target.rest_pose();
    let observed_world = target.world_matrices(&pose)?;
    let source_keys = semantic_keys(source);
    let target_keys = semantic_keys(target);
    let by_target: HashMap<_, _> = mappings.iter().map(|m| (m.target, m.source)).collect();
    let mut order: Vec<_> = mappings.iter().collect();
    order.sort_by_key(|m| depth(target, m.target));
    let mut corrected = Vec::new();
    for mapping in order {
        let world = target.world_matrices(&pose)?;
        if preserved.contains(&mapping.target) {
            ensure!(
                target.joints[mapping.target].matrix.is_none(),
                "reference preservation requires a TRS joint"
            );
            let parent = target.joints[mapping.target]
                .parent
                .map(|p| world_rotation(&world[p]))
                .transpose()?
                .unwrap_or_default();
            pose[mapping.target].rotation =
                a4(parent.inverse() * world_rotation(&observed_world[mapping.target])?);
            corrected.push(format!(
                "{}: observed target reference world posture preserved by explicit request",
                target.joints[mapping.target].name
            ));
            continue;
        }
        if rest_local.contains(&mapping.target) {
            // `pose` already holds the rest local transform.
            corrected.push(format!(
                "{}: rest pose against its parent kept as the reference by explicit request",
                target.joints[mapping.target].name
            ));
            continue;
        }
        let source_key = &source_keys[mapping.source];
        if source_key == "name:head" {
            let find = |key: &str| mappings.iter().find(|m| source_keys[m.source] == key);
            let point = |matrices: &[nalgebra::Matrix4<f64>], i| {
                v3(transform_point(&matrices[i], [0.0; 3]))
            };
            let source_forward = (|| {
                let left = find("left_clavicle")?;
                let right = find("right_clavicle")?;
                let pelvis = find("pelvis")?;
                let lateral =
                    point(&source_world, left.source) - point(&source_world, right.source);
                let up = point(&source_world, mapping.source) - point(&source_world, pelvis.source);
                lateral.cross(&up).try_normalize(1e-7)
            })();
            if let (Some(from), Some(to)) = (
                marked_head_frame(target, &world, mapping.target, None),
                marked_head_frame(source, &source_world, mapping.source, source_forward),
            ) {
                let correction = alignment * to * from.inverse();
                let parent = target.joints[mapping.target]
                    .parent
                    .map(|p| world_rotation(&world[p]))
                    .transpose()?
                    .unwrap_or_default();
                pose[mapping.target].rotation =
                    a4(parent.inverse() * correction * world_rotation(&world[mapping.target])?);
                corrected.push(format!("{}: {:.5} rad marked head-facing alignment (headfront; source marker or torso-facing reference)", target.joints[mapping.target].name, correction.angle()));
                continue;
            }
        }
        if let Some(side) = source_key.strip_suffix("_hand") {
            let root = |digit: &str| {
                mappings
                    .iter()
                    .find(|m| source_keys[m.source] == format!("{side}_{digit}_1"))
            };
            if let (Some(index), Some(middle), Some(little)) =
                (root("index"), root("middle"), root("little"))
            {
                let point = |matrices: &[nalgebra::Matrix4<f64>], i| {
                    v3(transform_point(&matrices[i], [0.0; 3]))
                };
                let from = palm_frame(
                    point(&world, mapping.target),
                    point(&world, middle.target),
                    point(&world, index.target),
                    point(&world, little.target),
                );
                let to = palm_frame(
                    point(&source_world, mapping.source),
                    point(&source_world, middle.source),
                    point(&source_world, index.source),
                    point(&source_world, little.source),
                );
                if let (Some(from), Some(to)) = (from, to) {
                    let correction = alignment * to * from.inverse();
                    let hand_world = correction * world_rotation(&world[mapping.target])?;
                    let mut spread = String::new();
                    if distribute_palm_twist
                        && let Some(chain) =
                            forearm_twist_chain(target, &by_target, &source_keys, mapping.target)
                    {
                        let (twist, rolled) =
                            spread_palm_twist(target, &mut pose, &chain, correction)?;
                        supports.extend(&chain[1..chain.len() - 1]);
                        let names: Vec<_> = chain
                            .iter()
                            .map(|&j| target.joints[j].name.as_str())
                            .collect();
                        spread = format!(
                            "; its {twist:.5} rad axial twist spread over {} ({rolled:.5} rad left on the hand)",
                            names.join(", ")
                        );
                    }
                    let world = target.world_matrices(&pose)?;
                    let parent = target.joints[mapping.target]
                        .parent
                        .map(|p| world_rotation(&world[p]))
                        .transpose()?
                        .unwrap_or_default();
                    pose[mapping.target].rotation = a4(parent.inverse() * hand_world);
                    corrected.push(format!(
                        "{}: {:.5} rad palm-frame alignment (wrist, middle, index, little){spread}",
                        target.joints[mapping.target].name,
                        correction.angle()
                    ));
                    continue;
                }
            }
        }
        let mut children: Vec<_> = mappings
            .iter()
            .filter(|other| {
                let mut parent = target.joints[other.target].parent;
                while let Some(p) = parent {
                    if p == mapping.target {
                        return true;
                    }
                    if by_target.contains_key(&p) {
                        return false;
                    }
                    parent = target.joints[p].parent;
                }
                false
            })
            .collect();
        if children.len() > 1 {
            // At the torso, prefer the continuing axial chain over limbs.
            // At a hand, the middle finger gives a stable palm direction.
            let axial: Vec<_> = children
                .iter()
                .copied()
                .filter(|child| {
                    let key = &source_keys[child.source];
                    key.starts_with("spine_")
                        || key == "name:neck"
                        || key == "name:head"
                        || key.ends_with("_middle_1")
                })
                .collect();
            if axial.len() == 1 {
                children = axial;
            }
        }
        let bilateral_role = if source_key == "pelvis" {
            Some("thigh")
        } else if source_key.starts_with("spine_") {
            Some("clavicle")
        } else {
            None
        };
        if let Some(role) = bilateral_role
            && children.len() == 1
        {
            let find = |side: &str| {
                mappings
                    .iter()
                    .find(|m| source_keys[m.source] == format!("{side}_{role}"))
            };
            if let (Some(left), Some(right)) = (find("left"), find("right")) {
                let point = |matrices: &[nalgebra::Matrix4<f64>], i| {
                    v3(transform_point(&matrices[i], [0.0; 3]))
                };
                let child = children[0];
                let from = palm_frame(
                    point(&world, mapping.target),
                    point(&world, child.target),
                    point(&world, left.target),
                    point(&world, right.target),
                );
                let to = palm_frame(
                    point(&source_world, mapping.source),
                    point(&source_world, child.source),
                    point(&source_world, left.source),
                    point(&source_world, right.source),
                );
                if let (Some(from), Some(to)) = (from, to) {
                    let correction = alignment * to * from.inverse();
                    let parent = target.joints[mapping.target]
                        .parent
                        .map(|p| world_rotation(&world[p]))
                        .transpose()?
                        .unwrap_or_default();
                    pose[mapping.target].rotation =
                        a4(parent.inverse() * correction * world_rotation(&world[mapping.target])?);
                    corrected.push(format!(
                        "{}: {:.5} rad bilateral axial-frame alignment (left/right {role})",
                        target.joints[mapping.target].name,
                        correction.angle()
                    ));
                    continue;
                }
            }
        }
        let (direction, desired, evidence) = if children.len() == 1 {
            let child = children[0];
            (
                v3(transform_point(&world[child.target], [0.0; 3]))
                    - v3(transform_point(&world[mapping.target], [0.0; 3])),
                alignment
                    * (v3(transform_point(&source_world[child.source], [0.0; 3]))
                        - v3(transform_point(&source_world[mapping.source], [0.0; 3]))),
                format!("child {}", target.joints[child.target].name),
            )
        } else if children.is_empty() && target.joints[mapping.target].deform {
            let target_direction =
                terminal_digit_direction(target, &world, &target_keys, mapping.target)?;
            let source_children: Vec<_> = source
                .joints
                .iter()
                .enumerate()
                .filter(|(_, joint)| joint.parent == Some(mapping.source))
                .map(|(i, _)| i)
                .collect();
            let source_direction = if source_children.len() == 1 {
                Some(
                    v3(transform_point(&source_world[source_children[0]], [0.0; 3]))
                        - v3(transform_point(&source_world[mapping.source], [0.0; 3])),
                )
            } else if source_children.is_empty() {
                terminal_digit_direction(source, &source_world, &source_keys, mapping.source)?
            } else {
                None
            };
            let (Some(direction), Some(desired)) = (target_direction, source_direction) else {
                continue;
            };
            (
                direction,
                alignment * desired,
                "terminal digit axis inferred from two consistent phalanges".into(),
            )
        } else {
            continue;
        };
        if direction.norm() < 1e-7 || desired.norm() < 1e-7 {
            continue;
        }
        let from = direction.normalize();
        let to = desired.normalize();
        let swing = UnitQuaternion::rotation_between(&from, &to).unwrap_or_else(|| {
            let axis = from.cross(&if from.x.abs() < 0.8 {
                Vector3::x()
            } else {
                Vector3::y()
            });
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), std::f64::consts::PI)
        });
        if swing.angle() < 1e-5 {
            continue;
        }
        let parent = target.joints[mapping.target]
            .parent
            .map(|p| world_rotation(&world[p]))
            .transpose()?
            .unwrap_or_default();
        pose[mapping.target].rotation =
            a4(parent.inverse() * swing * world_rotation(&world[mapping.target])?);
        corrected.push(format!(
            "{}: {:.5} rad reference swing ({})",
            target.joints[mapping.target].name,
            swing.angle(),
            evidence
        ));
    }
    Ok((pose, corrected, supports))
}

/// The mapped forearm above a mapped hand and the unmapped twist joints
/// between them: `[forearm, supports..., hand]`. None unless the forearm maps
/// to the same side's source forearm and every support is a plain link (one
/// child, TRS).
fn forearm_twist_chain(
    target: &Skeleton,
    by_target: &HashMap<usize, usize>,
    source_keys: &[String],
    hand: usize,
) -> Option<Vec<usize>> {
    let side = source_keys[*by_target.get(&hand)?].strip_suffix("_hand")?;
    let mut chain = vec![hand];
    let mut joint = target.joints[hand].parent?;
    while !by_target.contains_key(&joint) {
        let children = target.joints.iter().filter(|j| j.parent == Some(joint));
        if chain.len() > 4 || children.count() != 1 || target.joints[joint].matrix.is_some() {
            return None;
        }
        chain.push(joint);
        joint = target.joints[joint].parent?;
    }
    (source_keys[by_target[&joint]] == format!("{side}_forearm")
        && target.joints[joint].matrix.is_none())
    .then(|| {
        chain.push(joint);
        chain.reverse();
        chain
    })
}

 fn palm_twist_shares(supports: usize) -> Vec<f64> {
    if supports == 0 {
        return vec![0.5, 0.5];
    }
    let mut shares = vec![0.25];
    shares.extend(std::iter::repeat_n(0.5 / supports as f64, supports));
    shares.push(0.25);
    shares
}

/// Rolls the forearm and its twist supports of `chain` (see
/// [`forearm_twist_chain`]) by their shares ([`palm_twist_shares`]) of the
/// palm `correction`'s twist about the forearm-to-hand axis, each about the
/// axis to its next joint so that joint stays put; the hand then takes the
/// rest when its world pose is set. Returns the twist and the hand's share
/// (rad).
fn spread_palm_twist(
    target: &Skeleton,
    pose: &mut [Transform],
    chain: &[usize],
    correction: UnitQuaternion<f64>,
) -> Result<(f64, f64)> {
    let point = |matrices: &[Matrix4<f64>], i: usize| v3(transform_point(&matrices[i], [0.0; 3]));
    let world = target.world_matrices(pose)?;
    let hand = *chain.last().unwrap();
    let Some(axis) = (point(&world, hand) - point(&world, chain[0])).try_normalize(1e-7) else {
        return Ok((0.0, 0.0));
    };
    let q = correction.into_inner();
    let q = if q.w < 0.0 { -q } else { q };
    let twist = 2.0 * q.imag().dot(&axis).atan2(q.w);
    let shares = palm_twist_shares(chain.len() - 2);
    for (k, &joint) in chain[..chain.len() - 1].iter().enumerate() {
        let world = target.world_matrices(pose)?;
        let axis = if k == 0 {
            axis
        } else {
            match (point(&world, chain[k + 1]) - point(&world, joint)).try_normalize(1e-7) {
                Some(axis) => axis,
                None => continue,
            }
        };
        let roll = UnitQuaternion::from_axis_angle(&Unit::new_unchecked(axis), twist * shares[k]);
        let parent = target.joints[joint]
            .parent
            .map(|p| world_rotation(&world[p]))
            .transpose()?
            .unwrap_or_default();
        pose[joint].rotation = a4(parent.inverse() * roll * world_rotation(&world[joint])?);
    }
    Ok((twist, twist * shares[shares.len() - 1]))
}

fn compact(name: &str) -> String {
    name.rsplit([':', '|'])
        .next()
        .unwrap_or(name)
        .chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn depth(skeleton: &Skeleton, mut i: usize) -> usize {
    let mut n = 0;
    while let Some(p) = skeleton.joints[i].parent {
        n += 1;
        i = p;
    }
    n
}

/// Known name conventions supply evidence; unknown joints retain exact-name keys.
/// Spine ordering follows ancestry, because exporters disagree on numbering.
pub fn semantic_keys(skeleton: &Skeleton) -> Vec<String> {
    // MPFB's default rig uses split limb bones, five spine segments and
    // numbered fingers. Recognize the complete convention, never a generic
    // node named Root on its own. Auxiliary and facial bones keep local poses.
    let mpfb = [
        "root",
        "spine05",
        "spine01",
        "upperarm01.L",
        "upperarm01.R",
        "lowerarm02.L",
        "lowerarm02.R",
        "wrist.L",
        "wrist.R",
        "upperleg01.L",
        "upperleg01.R",
        "foot.L",
        "foot.R",
    ]
    .iter()
    .all(|name| {
        skeleton
            .joints
            .iter()
            .filter(|j| compact(&j.name) == compact(name))
            .count()
            == 1
    });
    let mut spines: Vec<_> = skeleton
        .joints
        .iter()
        .enumerate()
        .filter(|(_, j)| {
            let n = compact(&j.name);
            n.starts_with("spine") && n[5..].chars().all(|c| c.is_ascii_digit())
        })
        .map(|(i, _)| i)
        .collect();
    spines.sort_by_key(|&i| depth(skeleton, i));
    skeleton
        .joints
        .iter()
        .enumerate()
        .map(|(i, joint)| {
            let n = compact(&joint.name);
            if mpfb {
                match joint.name.as_str() {
                    "root" => return "pelvis".into(),
                    "spine05" => return "spine_0".into(),
                    "spine03" => return "spine_1".into(),
                    "spine01" => return "spine_2".into(),
                    "spine04" | "spine02" => return format!("mpfb_aux:{n}"),
                    _ => {}
                }
                // Three.js sanitizes dots from node names during GLB loading.
                // Both upperarm01.L and upperarm01L are the same MPFB convention.
                for (suffix, side) in [("l", "left"), ("r", "right")] {
                    if let Some(stem) = n.strip_suffix(suffix) {
                        let role = match stem {
                            "upperarm01" => Some("upperarm"),
                            "lowerarm01" => Some("forearm"),
                            "wrist" => Some("hand"),
                            "upperleg01" => Some("thigh"),
                            "lowerleg01" => Some("shin"),
                            "foot" => Some("foot"),
                            "clavicle" => Some("clavicle"),
                            "toe11" => Some("toe"),
                            "toe12" => Some("toe_end"),
                            _ => None,
                        };
                        if let Some(role) = role {
                            return format!("{side}_{role}");
                        }
                        if let Some(digit) = stem.strip_prefix("finger")
                            && digit.len() == 2
                            && let (Ok(finger), Ok(rank)) =
                                (digit[..1].parse::<usize>(), digit[1..].parse::<usize>())
                            && (1..=5).contains(&finger)
                            && (1..=3).contains(&rank)
                        {
                            let role = ["thumb", "index", "middle", "ring", "little"][finger - 1];
                            return format!("{side}_{role}_{rank}");
                        }
                    }
                }
            }
            if let Some(rank) = spines.iter().position(|&j| j == i) {
                return format!("spine_{rank}");
            }
            if ["hips", "pelvis"].contains(&n.as_str()) {
                return "pelvis".into();
            }
            if n == "neck01" {
                return "name:neck".into();
            }
            let suffix_side = joint.name.to_ascii_lowercase();
            let short_side = if suffix_side.ends_with("_l") || suffix_side.ends_with(".l") {
                Some("left")
            } else if suffix_side.ends_with("_r") || suffix_side.ends_with(".r") {
                Some("right")
            } else {
                None
            };
            let side = if n.contains("left") {
                Some("left")
            } else if n.contains("right") {
                Some("right")
            } else {
                short_side
            };
            if let Some(side) = side {
                let rest = if short_side.is_some() && !n.contains(side) {
                    n[..n.len() - 1].to_owned()
                } else {
                    n.replace(side, "")
                };
                let rest = rest.strip_prefix("hc").unwrap_or(&rest);
                for (alias, role) in [
                    ("upleg", "thigh"),
                    ("thigh", "thigh"),
                    ("leg", "shin"),
                    ("calf", "shin"),
                    ("foot", "foot"),
                    ("toebase", "toe"),
                    ("ball", "toe"),
                    ("ballleaf", "toe_end"),
                    ("toeend", "toe_end"),
                    ("toebaseend", "toe_end"),
                    ("shoulder", "clavicle"),
                    ("clavicle", "clavicle"),
                    ("arm", "upperarm"),
                    ("upperarm", "upperarm"),
                    ("forearm", "forearm"),
                    ("lowerarm", "forearm"),
                    ("hand", "hand"),
                ] {
                    if rest == alias {
                        return format!("{side}_{role}");
                    }
                }
                let digit = rest.strip_prefix("hand").unwrap_or(rest);
                for (alias, role) in [
                    ("index", "index"),
                    ("middle", "middle"),
                    ("ring", "ring"),
                    ("pinky", "little"),
                    ("little", "little"),
                    ("thumb", "thumb"),
                ] {
                    if let Some(number) = digit
                        .strip_prefix(alias)
                        .map(|n| n.strip_suffix("leaf").unwrap_or(n))
                        && !number.is_empty()
                        && number.chars().all(|c| c.is_ascii_digit())
                        && let Ok(number) = number.parse::<u32>()
                    {
                        return format!("{side}_{role}_{number}");
                    }
                }
            }
            format!("name:{n}")
        })
        .collect()
}

fn unique_name(skeleton: &Skeleton, name: &str) -> Result<usize> {
    let matches: Vec<_> = skeleton
        .joints
        .iter()
        .enumerate()
        .filter(|(_, j)| j.name == name)
        .map(|(i, _)| i)
        .collect();
    ensure!(
        matches.len() == 1,
        "joint name must identify exactly one node: {name}"
    );
    Ok(matches[0])
}

pub fn match_joints(
    source: &Skeleton,
    target: &Skeleton,
    overrides: &BTreeMap<String, String>,
) -> Result<MappingReport> {
    source.validate()?;
    target.validate()?;
    let source_keys = semantic_keys(source);
    let target_keys = semantic_keys(target);
    let mut source_index: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut target_counts = HashMap::new();
    for (i, key) in source_keys.iter().enumerate() {
        if source.joints[i].deform {
            source_index.entry(key).or_default().push(i);
        }
    }
    for (i, key) in target_keys.iter().enumerate() {
        if target.joints[i].deform {
            *target_counts.entry(key).or_insert(0) += 1;
        }
    }
    let mut mappings = Vec::new();
    let mut used = HashSet::new();
    let mut warnings = Vec::new();
    let overrides: HashMap<usize, usize> = overrides
        .iter()
        .map(|(t, s)| Ok((unique_name(target, t)?, unique_name(source, s)?)))
        .collect::<Result<_>>()?;
    for (i, key) in target_keys.iter().enumerate() {
        let (candidate, evidence) = if let Some(&s) = overrides.get(&i) {
            (Some(s), "explicit_override".to_owned())
        } else if target.joints[i].deform && target_counts.get(key) == Some(&1) {
            (
                source_index
                    .get(key.as_str())
                    .filter(|v| v.len() == 1)
                    .map(|v| v[0]),
                format!("unique_semantic_key:{key}"),
            )
        } else {
            (None, String::new())
        };
        if let Some(s) = candidate {
            ensure!(
                used.insert(s),
                "source joint mapped more than once: {}",
                source.joints[s].name
            );
            ensure!(
                target.joints[i].matrix.is_none(),
                "target joint uses a matrix: {}",
                target.joints[i].name
            );
            mappings.push(JointMapping {
                source: s,
                target: i,
                evidence,
            });
        }
    }
     
    let by_target: HashMap<_, _> = mappings.iter().map(|m| (m.target, m.source)).collect();
    let mut remove = HashSet::new();
    for m in &mappings {
        let mut parent = target.joints[m.target].parent;
        while let Some(p) = parent {
            if let Some(&s) = by_target.get(&p) {
                let mut ancestor = source.joints[m.source].parent;
                while ancestor.is_some_and(|a| a != s) {
                    ancestor = source.joints[ancestor.unwrap()].parent;
                }
                if ancestor.is_none() {
                    ensure!(
                        !overrides.contains_key(&m.target),
                        "explicit mapping contradicts source ancestry at {}",
                        target.joints[m.target].name
                    );
                    remove.insert(m.target);
                    warnings.push(format!(
                        "incompatible_ancestry:{}",
                        target.joints[m.target].name
                    ));
                }
                break;
            }
            parent = target.joints[p].parent;
        }
    }
    mappings.retain(|m| !remove.contains(&m.target));
    let source_mapped: HashSet<_> = mappings.iter().map(|m| m.source).collect();
    let target_mapped: HashSet<_> = mappings.iter().map(|m| m.target).collect();
    let deform_count = target.joints.iter().filter(|j| j.deform).count();
    let mapped_deform = target_mapped
        .iter()
        .filter(|&&i| target.joints[i].deform)
        .count();
    Ok(MappingReport {
        mappings,
        unmapped_source: source
            .joints
            .iter()
            .enumerate()
            .filter(|(i, j)| j.deform && !source_mapped.contains(i))
            .map(|(_, j)| j.name.clone())
            .collect(),
        unmapped_target: target
            .joints
            .iter()
            .enumerate()
            .filter(|(i, j)| j.deform && !target_mapped.contains(i))
            .map(|(_, j)| j.name.clone())
            .collect(),
        target_deform_coverage: mapped_deform as f64 / deform_count.max(1) as f64,
        warnings,
    })
}

pub fn sample_times(start: f64, end: f64, fps: f64) -> Result<Vec<f64>> {
    ensure!(
        start.is_finite() && end.is_finite() && start >= 0.0 && end > start,
        "invalid time range"
    );
    ensure!(
        fps.is_finite() && (1.0..=240.0).contains(&fps),
        "fps must be 1..240"
    );
    let count = ((end - start) * fps).floor() as usize;
    ensure!(count <= 100_000, "resource_limit: at most 100000 frames");
    let mut times: Vec<_> = (0..=count)
        .map(|i| (start + i as f64 / fps).min(end))
        .collect();
    // A float32 GLB duration commonly lies a few nanoseconds after a regular
    // sample. Keep the exact end without exporting colliding float32 keys.
    if times.len() > 1 && ((times.last().unwrap() - start) as f32) == ((end - start) as f32) {
        *times.last_mut().unwrap() = end;
    } else {
        times.push(end);
    }
    Ok(times)
}

pub fn clip_from_poses(
    name: &str,
    times: &[f64],
    poses: &[Vec<Transform>],
    joints: &[usize],
) -> Result<MotionClip> {
    ensure!(
        !poses.is_empty() && poses.len() == times.len(),
        "sample count mismatch"
    );
    let mut tracks = Vec::new();
    for &joint in joints {
        for property in [Property::Translation, Property::Rotation, Property::Scale] {
            let mut values: Vec<[f64; 4]> = poses
                .iter()
                .map(|p| {
                    let t = p[joint];
                    match property {
                        Property::Translation => {
                            [t.translation[0], t.translation[1], t.translation[2], 0.0]
                        }
                        Property::Rotation => t.rotation,
                        Property::Scale => [t.scale[0], t.scale[1], t.scale[2], 0.0],
                    }
                })
                .collect();
            if property == Property::Rotation {
                for i in 1..values.len() {
                    if values[i]
                        .iter()
                        .zip(values[i - 1])
                        .map(|(a, b)| a * b)
                        .sum::<f64>()
                        < 0.0
                    {
                        values[i] = values[i].map(|v| -v);
                    }
                }
            }
            tracks.push(Track {
                joint,
                property,
                interpolation: Interpolation::Linear,
                times: times.to_vec(),
                values,
            });
        }
    }
    Ok(MotionClip {
        name: name.into(),
        tracks,
        morph_tracks: Vec::new(),
    })
}

pub fn retarget(
    source: &Skeleton,
    clip: &MotionClip,
    target: &Skeleton,
    config: &RetargetConfig,
) -> Result<(MotionClip, RetargetReport)> {
    clip.validate(source)?;
    ensure!(
        !config.distribute_wrist_twist || config.wrist_twist_chains.is_empty(),
        "serial_wrist_twist_conflicts_with_legacy_distribution"
    );
    ensure!(
        config.alignment.iter().all(|x| x.is_finite())
            && config.alignment.iter().map(|x| x * x).sum::<f64>() > 1e-12,
        "invalid alignment"
    );
    ensure!(
        config.min_target_coverage.is_finite() && (0.0..=1.0).contains(&config.min_target_coverage),
        "invalid coverage threshold"
    );
    let mapping = match_joints(source, target, &config.joint_map)?;
    let target_joint_scope = config
        .target_joint_scope
        .as_ref()
        .map(|names| target_joint_scope_report(target, &mapping, names))
        .transpose()?;
    if let Some(scope) = &target_joint_scope {
        ensure!(
            !mapping.mappings.is_empty()
                && scope.target_joint_coverage >= config.min_target_coverage,
            "insufficient_mapping: scoped target coverage {:.3}, required {:.3}; global target coverage {:.3}; unmapped scope: {}",
            scope.target_joint_coverage,
            config.min_target_coverage,
            mapping.target_deform_coverage,
            scope.unmapped_joints.join(", ")
        );
    } else {
        ensure!(
            !mapping.mappings.is_empty()
                && mapping.target_deform_coverage >= config.min_target_coverage,
            "insufficient_mapping: target coverage {:.3}, required {:.3}; inspect mapping or supply explicit overrides",
            mapping.target_deform_coverage,
            config.min_target_coverage
        );
    }
    let [clip_start, clip_end] = clip.time_range();
    let start = config.start.unwrap_or(clip_start);
    let end = config.end.unwrap_or(clip_end);
    ensure!(
        start >= clip_start && end <= clip_end + 1e-6,
        "retarget range outside source clip"
    );
    let source_times = sample_times(start, end, config.fps)?;
    let times: Vec<_> = source_times.iter().map(|t| t - start).collect();
    let sr = source.world_matrices(&source.rest_pose())?;
    let align = unit_quat(config.alignment);
    ensure!(
        config.align_reference_chains || config.preserve_reference_joints.is_empty(),
        "reference_preservation_requires_chain_alignment"
    );
    let preserved: HashSet<_> = config
        .preserve_reference_joints
        .iter()
        .map(|name| unique_name(target, name))
        .collect::<Result<_>>()?;
    ensure!(
        preserved.len() == config.preserve_reference_joints.len(),
        "duplicate_preserved_reference_joint"
    );
    ensure!(
        preserved
            .iter()
            .all(|joint| mapping.mappings.iter().any(|m| m.target == *joint)),
        "preserved_reference_joint_must_be_mapped"
    );
    let rest_local: HashSet<_> = config
        .rest_local_reference_joints
        .iter()
        .map(|name| unique_name(target, name))
        .collect::<Result<_>>()?;
    ensure!(
        rest_local
            .iter()
            .all(|joint| mapping.mappings.iter().any(|m| m.target == *joint) && !preserved.contains(joint)),
        "rest_local_reference_joint_must_be_mapped_and_not_preserved"
    );
    ensure!(
        config.align_reference_chains || !config.distribute_palm_twist,
        "palm_twist_distribution_requires_chain_alignment"
    );
    let (target_reference, reference_chain_corrections, twist_supports) =
        if config.align_reference_chains {
            align_reference_pose(
                source,
                target,
                &mapping.mappings,
                align,
                &preserved,
                &rest_local,
                config.distribute_palm_twist,
            )?
        } else {
            (target.rest_pose(), Vec::new(), Vec::new())
        };
    let tr = target.world_matrices(&target_reference)?;
    let source_rot: Vec<_> = sr.iter().map(world_rotation).collect::<Result<_>>()?;
    let target_rot: Vec<_> = tr.iter().map(world_rotation).collect::<Result<_>>()?;
    let by_target: HashMap<_, _> = mapping
        .mappings
        .iter()
        .map(|m| (m.target, m.source))
        .collect();
    let locally_preserved: HashSet<_> = config
        .preserve_local_pose_joints
        .iter()
        .map(|name| unique_name(target, name))
        .collect::<Result<_>>()?;
    ensure!(
        locally_preserved.len() == config.preserve_local_pose_joints.len(),
        "duplicate_preserved_local_pose_joint"
    );
    ensure!(
        locally_preserved
            .iter()
            .all(|i| by_target.contains_key(i) && target.joints[*i].matrix.is_none()),
        "local_pose_preservation_requires_mapped_trs_joints"
    );
    let mut ratios = Vec::new();
    for m in &mapping.mappings {
        if let Some(tp) = target.joints[m.target].parent
            && let Some(&sp) = by_target.get(&tp)
        {
            let sl = (v3(transform_point(&sr[m.source], [0.0; 3]))
                - v3(transform_point(&sr[sp], [0.0; 3])))
            .norm();
            let tl = (v3(transform_point(&tr[m.target], [0.0; 3]))
                - v3(transform_point(&tr[tp], [0.0; 3])))
            .norm();
            if sl > 1e-6 && tl > 1e-6 {
                ratios.push(tl / sl);
            }
        }
    }
    ratios.sort_by(f64::total_cmp);
    let root_scale = config
        .root_scale
        .unwrap_or_else(|| ratios.get(ratios.len() / 2).copied().unwrap_or(1.0));
    ensure!(
        root_scale.is_finite() && root_scale > 0.0,
        "invalid root scale"
    );
    let roots: HashSet<_> = mapping
        .mappings
        .iter()
        .filter(|m| {
            let mut parent = target.joints[m.target].parent;
            while let Some(p) = parent {
                if by_target.contains_key(&p) {
                    return false;
                }
                parent = target.joints[p].parent;
            }
            true
        })
        .map(|m| m.target)
        .collect();
    let observed_world = target.world_matrices(&target.rest_pose())?;
    let mut trajectory_joints = HashSet::new();
    let mut trajectories = Vec::new();
    for trajectory in &config.limb_trajectories {
        let chain = trajectory
            .target_chain
            .iter()
            .map(|name| unique_name(target, name))
            .collect::<Result<Vec<_>>>()?;
        for pair in chain.windows(2) {
            ensure!(
                target.joints[pair[1]].parent == Some(pair[0]),
                "trajectory_requires_a_direct_four_joint_chain"
            );
        }
        ensure!(
            chain.iter().all(|j| target.joints[*j].matrix.is_none()),
            "trajectory_requires_trs_joints"
        );
        ensure!(
            chain.iter().all(|j| trajectory_joints.insert(*j)),
            "overlapping_limb_trajectories"
        );
        let anchor = unique_name(target, &trajectory.target_anchor)?;
        let source_end = *by_target
            .get(&chain[3])
            .context("trajectory_end_must_be_mapped")?;
        let source_anchor = *by_target
            .get(&anchor)
            .context("trajectory_anchor_must_be_mapped")?;
        let mut ancestor = target.joints[chain[0]].parent;
        while ancestor.is_some_and(|j| j != anchor) {
            ancestor = target.joints[ancestor.unwrap()].parent;
        }
        ensure!(
            ancestor == Some(anchor),
            "trajectory_anchor_must_be_above_chain"
        );
        let scale = trajectory.position_scale.unwrap_or(root_scale);
        ensure!(
            scale.is_finite() && scale > 0.,
            "invalid_trajectory_position_scale"
        );
        ensure!(
            trajectory.max_orientation_deviation_rad.is_none_or(
                |limit| limit.is_finite() && (0. ..=std::f64::consts::PI).contains(&limit)
            ),
            "invalid_trajectory_orientation_limit"
        );
        ensure!(
            trajectory.max_end_orientation_deviation_rad.is_none_or(
                |limit| limit.is_finite() && (0. ..=std::f64::consts::PI).contains(&limit)
            ),
            "invalid_trajectory_end_orientation_limit"
        );
        let point = |matrices: &[Matrix4<f64>], i| v3(transform_point(&matrices[i], [0.; 3]));
        trajectories.push((
            chain,
            anchor,
            source_end,
            source_anchor,
            scale,
            point(
                &observed_world,
                unique_name(target, &trajectory.target_chain[3])?,
            ) - point(&observed_world, anchor),
            point(&sr, source_end) - point(&sr, source_anchor),
        ));
    }
    let mut trajectory_reports: Vec<_> = config
        .limb_trajectories
        .iter()
        .map(|t| LimbTrajectoryReport {
            target_chain: t.target_chain.clone(),
            max_position_error_m: 0.,
            max_orientation_correction_rad: 0.,
            max_end_orientation_correction_rad: 0.,
        })
        .collect();
    ensure!(
        locally_preserved.is_disjoint(&trajectory_joints),
        "local_pose_preservation_conflicts_with_limb_trajectory"
    );
    let mut order: Vec<_> = (0..target.joints.len()).collect();
    order.sort_by_key(|&i| depth(target, i));
    let mut poses = Vec::new();
    let mut twist_distribution = Vec::new();
    let mut twist_distribution_joints = Vec::new();
    if config.distribute_wrist_twist {
        let keys = semantic_keys(target);
        for m in &mapping.mappings {
            if let Some(side) = keys[m.target].strip_suffix("_hand")
                && let Some(parent) = target.joints[m.target].parent
                && keys[parent] == format!("{side}_forearm")
                && by_target.contains_key(&parent)
            {
                ensure!(
                    !locally_preserved.contains(&m.target) && !locally_preserved.contains(&parent),
                    "local_pose_preservation_conflicts_with_wrist_twist"
                );
                twist_distribution.push(crate::articulation::PreparedTwist::prepare(
                    target, m.target,
                )?);
                twist_distribution_joints.push(target.joints[m.target].name.clone());
            }
        }
        ensure!(
            !twist_distribution.is_empty(),
            "no_direct_mapped_wrist_twist_pairs"
        );
    }
    let source_keys = semantic_keys(source);
    let mut serial_nodes = HashSet::new();
    let mut serial_twists = Vec::new();
    let mut wrist_twist_reports = Vec::new();
    for names in &config.wrist_twist_chains {
        ensure!(
            (2..=8).contains(&names.len()),
            "serial_wrist_twist_chain_length_must_be_2_to_8"
        );
        let joints = names
            .iter()
            .map(|name| unique_name(target, name))
            .collect::<Result<Vec<_>>>()?;
        let hand = *joints.last().unwrap();
        let forearm = target.joints[joints[0]]
            .parent
            .context("serial_wrist_twist_requires_forearm_parent")?;
        let source_hand = *by_target
            .get(&hand)
            .context("serial_wrist_twist_requires_mapped_hand")?;
        let source_forearm = *by_target
            .get(&forearm)
            .context("serial_wrist_twist_requires_mapped_forearm")?;
        let side = source_keys[source_hand]
            .strip_suffix("_hand")
            .context("serial_wrist_twist_end_must_map_to_hand")?;
        ensure!(
            source_keys[source_forearm] == format!("{side}_forearm"),
            "serial_wrist_twist_parent_must_map_to_same_forearm"
        );
        ensure!(
            joints[..joints.len() - 1]
                .iter()
                .all(|i| !by_target.contains_key(i)),
            "serial_wrist_twist_support_must_be_unmapped"
        );
        for &i in std::iter::once(&forearm).chain(joints.iter()) {
            ensure!(serial_nodes.insert(i), "serial_wrist_twist_chains_overlap");
            ensure!(
                !locally_preserved.contains(&i),
                "local_pose_preservation_conflicts_with_serial_wrist_twist"
            );
            ensure!(
                !trajectory_joints.contains(&i)
                    && trajectories.iter().all(|(_, anchor, ..)| *anchor != i),
                "limb_trajectory_conflicts_with_serial_wrist_twist"
            );
        }
        let prepared = PreparedWristTwistChain::prepare(target, forearm, joints)?;
        wrist_twist_reports.push(WristTwistChainReport {
            forearm: target.joints[forearm].name.clone(),
            joints: names.clone(),
            max_transferred_rad: 0.,
            max_twist_step_rad: 0.,
            singular_swing_samples: 0,
        });
        serial_twists.push(prepared);
    }
    let mut max_wrist_twist_transferred_rad = 0.0_f64;
    for time in source_times {
        let sw = source.world_matrices(&clip.sample(source, time)?)?;
        // Unmapped joints keep their reference pose: the rest pose, or the
        // share of a palm twist a forearm twist support took.
        let mut pose = target.rest_pose();
        for &i in &twist_supports {
            pose[i] = target_reference[i];
        }
        let mut worlds = tr.clone();
        for &i in &order {
            let parent = target.joints[i].parent;
            if let Some(&s) = by_target.get(&i)
                && !locally_preserved.contains(&i)
            {
                let delta =
                    align * world_rotation(&sw[s])? * source_rot[s].inverse() * align.inverse();
                let desired = delta * target_rot[i];
                let parent_rotation = parent
                    .map(|p| world_rotation(&worlds[p]))
                    .transpose()?
                    .unwrap_or_else(UnitQuaternion::identity);
                pose[i].rotation = a4(parent_rotation.inverse() * desired);
                if config.root_motion && roots.contains(&i) {
                    let offset = align
                        * (v3(transform_point(&sw[s], [0.0; 3]))
                            - v3(transform_point(&sr[s], [0.0; 3])))
                        * root_scale;
                    let position = a3(v3(transform_point(&tr[i], [0.0; 3])) + offset);
                    pose[i].translation = if let Some(p) = parent {
                        transform_point(
                            &worlds[p].try_inverse().context("singular parent frame")?,
                            position,
                        )
                    } else {
                        position
                    };
                }
            }
            let local = target.joints[i]
                .matrix
                .map(matrix_from)
                .unwrap_or_else(|| pose[i].matrix());
            worlds[i] = parent.map_or(local, |p| worlds[p] * local);
        }
        for (
            index,
            (chain, anchor, source_end, source_anchor, scale, target_offset, source_offset),
        ) in trajectories.iter().enumerate()
        {
            let world = target.world_matrices(&pose)?;
            let point = |matrices: &[Matrix4<f64>], i| v3(transform_point(&matrices[i], [0.; 3]));
            let goal = point(&world, *anchor)
                + target_offset
                + align
                    * (point(&sw, *source_end) - point(&sw, *source_anchor) - source_offset)
                    * *scale;
            let fk_orientation = world_rotation(&world[chain[2]])?;
            let mut terminal_orientation = world_rotation(&world[chain[3]])?;
            let reference_orientation = world_rotation(&world[*anchor])?
                * world_rotation(&observed_world[*anchor])?.inverse()
                * world_rotation(&observed_world[chain[2]])?;
            let mut orientation = fk_orientation;
            if let Some(limit) = config.limb_trajectories[index].max_orientation_deviation_rad {
                let angle = (reference_orientation.inverse() * fk_orientation).angle();
                if angle > limit {
                    orientation = reference_orientation.slerp(&fk_orientation, limit / angle);
                    trajectory_reports[index].max_orientation_correction_rad = trajectory_reports
                        [index]
                        .max_orientation_correction_rad
                        .max(angle - limit);
                }
            }
            let local_terminal =
                fk_orientation.inverse() * (point(&world, chain[3]) - point(&world, chain[2]));
            let terminal_offset = orientation * local_terminal;
            let pole = point(&world, chain[1]);
            crate::solve::two_bone(
                target,
                &mut pose,
                chain[0],
                chain[1],
                chain[2],
                a3(goal - terminal_offset),
                Some(a3(pole)),
            )?;
            let updated = target.world_matrices(&pose)?;
            pose[chain[2]].rotation =
                a4(world_rotation(&updated[chain[1]])?.inverse() * orientation);
             
            if let Some(limit) = config.limb_trajectories[index].max_end_orientation_deviation_rad {
                let end_reference_orientation = world_rotation(&world[*anchor])?
                    * world_rotation(&observed_world[*anchor])?.inverse()
                    * world_rotation(&observed_world[chain[3]])?;
                let angle = (end_reference_orientation.inverse() * terminal_orientation).angle();
                if angle > limit {
                    terminal_orientation =
                        end_reference_orientation.slerp(&terminal_orientation, limit / angle);
                    trajectory_reports[index].max_end_orientation_correction_rad =
                        trajectory_reports[index]
                            .max_end_orientation_correction_rad
                            .max(angle - limit);
                }
            }
            pose[chain[3]].rotation = a4(orientation.inverse() * terminal_orientation);
            let final_world = target.world_matrices(&pose)?;
            let error = (point(&final_world, chain[3]) - goal).norm();
            trajectory_reports[index].max_position_error_m =
                trajectory_reports[index].max_position_error_m.max(error);
        }
        for distribution in &twist_distribution {
            let report = distribution.apply(target, &mut pose)?;
            max_wrist_twist_transferred_rad =
                max_wrist_twist_transferred_rad.max(report.transferred_rad);
        }
        for (chain, report) in serial_twists.iter().zip(&mut wrist_twist_reports) {
            let (transferred, step, singular) = chain.apply(target, &mut pose)?;
            report.max_transferred_rad = report.max_transferred_rad.max(transferred);
            report.max_twist_step_rad = report.max_twist_step_rad.max(step);
            report.singular_swing_samples += usize::from(singular);
            max_wrist_twist_transferred_rad = max_wrist_twist_transferred_rad.max(transferred);
        }
        poses.push(pose);
    }
    let mut issues = mapping.warnings.clone();
    if target_joint_scope.is_some() {
        issues.push("Explicit target joint scope gates this clip only; global rig mapping remains incomplete when reported. Out-of-scope motion and auxiliary inheritance require separate validation.".into());
    }
    if !clip.morph_tracks.is_empty() {
        issues.push("Source morph targets are not transferred across characters; target default morph weights are preserved.".into());
    }
    if !mapping.unmapped_target.is_empty() {
        issues.push(if serial_twists.is_empty() {
            "Unmapped target joints keep their local reference pose and inherit parent motion."
        } else {
            "Unmapped target joints keep their local reference pose except explicitly authored serial wrist-twist supports."
        }.into());
    }
    if clip.tracks.iter().any(|t| t.property == Property::Scale) {
        issues.push("Source scale animation is not transferred; target reference scales and proportions are retained.".into());
    }
    issues.push("Contacts and anatomical correctness require separate validation; no automatic foot locking is applied.".into());
    let drift = poses
        .iter()
        .flat_map(|p| p.iter().enumerate())
        .filter(|(i, _)| !roots.contains(i))
        .map(|(i, t)| (v3(t.translation) - v3(target.joints[i].rest.translation)).norm())
        .fold(0.0, f64::max);
    let mut joints: Vec<_> = mapping.mappings.iter().map(|m| m.target).collect();
    joints.extend(&twist_supports);
    // Supports deliberately have no source mapping, but their newly authored
    // rotations must be exported as tracks rather than left in memory only.
    for chain in &serial_twists {
        joints.extend(&chain.joints[..chain.joints.len() - 1]);
    }
    let mut seen = HashSet::new();
    joints.retain(|j| seen.insert(*j));
    let output = clip_from_poses(&config.name, &times, &poses, &joints)?;
    output.validate(target)?;
    let mut motion_roots: Vec<_> = roots
        .iter()
        .map(|&i| target.joints[i].name.clone())
        .collect();
    motion_roots.sort();
    Ok((
        output,
        RetargetReport {
            root_scale,
            motion_roots,
            sample_count: poses.len(),
            preserved_local_pose_joints: config.preserve_local_pose_joints.clone(),
            max_local_bone_length_error: drift,
            mapping,
            target_joint_scope,
            issues,
            reference_chain_corrections,
            twist_distribution_joints,
            max_wrist_twist_transferred_rad,
            wrist_twist_chains: wrist_twist_reports,
            limb_trajectories: trajectory_reports,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn serial_wrist_fixture(
        count: usize,
        angle: f64,
    ) -> (Skeleton, Skeleton, MotionClip, RetargetConfig) {
        let joint = |name: &str, parent, translation| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(translation),
            matrix: None,
            deform: true,
        };
        let mut source = Skeleton {
            joints: vec![
                joint("Hips", None, [0.2, 0.8, -0.1]),
                joint("LeftForeArm", Some(0), [0.1, 0.2, 0.05]),
                joint("LeftHand", Some(1), [0., 0.4, 0.]),
                joint("LeftHandMiddle1", Some(2), [0.03, 0.1, 0.]),
            ],
        };
        source.joints[0].rest.rotation = a4(UnitQuaternion::from_euler_angles(0.3, -0.2, 0.15));
        source.joints[0].rest.scale = [1.2; 3];
        source.joints[1].rest.rotation = a4(UnitQuaternion::from_euler_angles(-0.2, 0.3, 0.1));
        source.joints[1].rest.scale = [1.1; 3];
        source.joints[2].rest.rotation = a4(UnitQuaternion::from_euler_angles(0.2, -0.1, 0.25));
        source.joints[2].rest.scale = [0.9; 3];
        let mut target = Skeleton {
            joints: source.joints[..2].to_vec(),
        };
        let mut parent = 1;
        let mut names = Vec::new();
        for i in 1..count {
            let name = format!("LeftForeArmTwist{i}");
            target
                .joints
                .push(joint(&name, Some(parent), [0., 0.4 / count as f64, 0.]));
            parent = target.joints.len() - 1;
            names.push(name);
        }
        let mut hand = source.joints[2].clone();
        hand.parent = Some(parent);
        hand.rest.translation = [0., 0.4 / count as f64, 0.];
        target.joints.push(hand);
        let mut finger = source.joints[3].clone();
        finger.parent = Some(target.joints.len() - 1);
        target.joints.push(finger);
        names.push("LeftHand".into());
        let initial = source.rest_pose();
        let mut moved = initial.clone();
        moved[0].translation = [0.3, 0.85, -0.05];
        moved[0].rotation =
            a4(unit_quat(initial[0].rotation) * UnitQuaternion::from_euler_angles(-0.1, 0.2, 0.15));
        moved[1].rotation = a4(unit_quat(initial[1].rotation)
            * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.3));
        moved[2].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle)
            * unit_quat(initial[2].rotation));
        moved[3].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.8));
        let clip =
            clip_from_poses("wrist_roll", &[0., 1.], &[initial, moved], &[0, 1, 2, 3]).unwrap();
        let config = RetargetConfig {
            fps: 4.,
            root_scale: Some(1.),
            min_target_coverage: 0.2,
            preserve_local_pose_joints: vec!["LeftHandMiddle1".into()],
            wrist_twist_chains: if count > 1 { vec![names] } else { Vec::new() },
            ..Default::default()
        };
        (source, target, clip, config)
    }

    #[test]
    fn serial_wrist_twist_preserves_hand_forearm_and_distributes_small_steps() {
        for count in [2, 4, 8] {
            for angle in [0., std::f64::consts::FRAC_PI_2, 2.6, -2.6] {
                let (source, target, clip, config) = serial_wrist_fixture(count, angle);
                let mut ordinary = config.clone();
                ordinary.wrist_twist_chains.clear();
                let (baseline, _) = retarget(&source, &clip, &target, &ordinary).unwrap();
                let (output, report) = retarget(&source, &clip, &target, &config).unwrap();
                let hand = count + 1;
                let finger = count + 2;
                assert_eq!(baseline.time_range(), output.time_range());
                for track in &baseline.tracks {
                    let other = output
                        .tracks
                        .iter()
                        .find(|t| t.joint == track.joint && t.property == track.property)
                        .unwrap();
                    assert_eq!(track.times, other.times);
                }
                for i in 2..=count {
                    assert!(
                        output
                            .tracks
                            .iter()
                            .any(|t| t.joint == i && t.property == Property::Rotation)
                    );
                }
                for time in [0., 0.125, 0.25, 0.37, 0.5, 0.625, 0.75, 0.875, 1.] {
                    let before = baseline.sample(&target, time).unwrap();
                    let after = output.sample(&target, time).unwrap();
                    let a = target.world_matrices(&before).unwrap();
                    let b = target.world_matrices(&after).unwrap();
                    for i in [1, hand, finger] {
                        assert!((a[i] - b[i]).norm() < 1e-9);
                    }
                    for i in 0..target.joints.len() {
                        assert_eq!(before[i].translation, after[i].translation);
                        assert_eq!(before[i].scale, after[i].scale);
                        assert!(
                            (v3(transform_point(&a[i], [0.; 3]))
                                - v3(transform_point(&b[i], [0.; 3])))
                            .norm()
                                < 1e-9
                        );
                    }
                    let forearm = world_rotation(&b[1]).unwrap();
                    for i in 2..=hand {
                        let mut relative = forearm.inverse() * world_rotation(&b[i]).unwrap();
                        if i == hand {
                            relative *= unit_quat(target.joints[hand].rest.rotation).inverse();
                        }
                        let expected = UnitQuaternion::from_axis_angle(
                            &Vector3::y_axis(),
                            angle * time * (i - 1) as f64 / count as f64,
                        );
                        assert!(relative.angle_to(&expected) < 1e-8);
                        let delta = unit_quat(after[i].rotation)
                            * unit_quat(target.joints[i].rest.rotation).inverse();
                        assert!((delta.angle() - (angle * time / count as f64).abs()).abs() < 1e-8);
                    }
                }
                let chain = &report.wrist_twist_chains[0];
                assert_eq!(chain.forearm, "LeftForeArm");
                assert_eq!(chain.joints, config.wrist_twist_chains[0]);
                assert!((chain.max_twist_step_rad - angle.abs() / count as f64).abs() < 1e-8);
                assert!(
                    (chain.max_transferred_rad - angle.abs() * (count - 1) as f64 / count as f64)
                        .abs()
                        < 1e-8
                );
                assert_eq!(chain.singular_swing_samples, 0);
            }
        }
    }

    #[test]
    fn serial_wrist_twist_rejects_invalid_structure_mapping_and_conflicts() {
        let (source, target, clip, config) = serial_wrist_fixture(4, 2.6);
        let rejected =
            |t: &Skeleton, c: &RetargetConfig| assert!(retarget(&source, &clip, t, c).is_err());
        let mut bad = config.clone();
        bad.wrist_twist_chains[0] = vec!["LeftHand".into()];
        rejected(&target, &bad);
        bad = config.clone();
        bad.wrist_twist_chains[0] = vec!["LeftHand".into(); 9];
        rejected(&target, &bad);
        bad = config.clone();
        bad.wrist_twist_chains[0][1] = "missing".into();
        rejected(&target, &bad);
        bad = config.clone();
        bad.wrist_twist_chains
            .push(bad.wrist_twist_chains[0].clone());
        rejected(&target, &bad);
        bad = config.clone();
        bad.wrist_twist_chains[0].swap(1, 2);
        rejected(&target, &bad);
        bad = config.clone();
        bad.distribute_wrist_twist = true;
        rejected(&target, &bad);
        for name in ["LeftForeArm", "LeftHand"] {
            bad = config.clone();
            bad.preserve_local_pose_joints.push(name.into());
            rejected(&target, &bad);
        }
        bad = config.clone();
        bad.limb_trajectories.push(LimbTrajectory {
            target_chain: [
                "LeftForeArmTwist1",
                "LeftForeArmTwist2",
                "LeftForeArmTwist3",
                "LeftHand",
            ]
            .map(str::to_owned),
            target_anchor: "Hips".into(),
            position_scale: Some(1.),
            max_orientation_deviation_rad: None,
            max_end_orientation_deviation_rad: None,
        });
        rejected(&target, &bad);
        for support in [2, 3, 4] {
            let mut t = target.clone();
            t.joints.push(Joint {
                id: "branch".into(),
                name: "branch".into(),
                parent: Some(support),
                rest: Transform::translated([0.1, 0., 0.]),
                matrix: None,
                deform: false,
            });
            rejected(&t, &config);
        }
        for index in [1, 2, 5] {
            for scale in [[1., 1.2, 1.], [-1.; 3], [0.; 3]] {
                let mut t = target.clone();
                t.joints[index].rest.scale = scale;
                rejected(&t, &config);
            }
        }
        let mut t = target.clone();
        t.joints[2].matrix = Some(matrix_array(&t.joints[2].rest.matrix()));
        rejected(&t, &config);
        for translation in [[0.01, 0.1, 0.], [0., -0.1, 0.], [0.; 3]] {
            let mut t = target.clone();
            t.joints[3].rest.translation = translation;
            rejected(&t, &config);
        }
        let mut t = target.clone();
        t.joints[2].rest.rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.2));
        rejected(&t, &config);
        for index in [1, 5] {
            let mut t = target.clone();
            t.joints[index].deform = false;
            rejected(&t, &config);
        }

        // A helper that has acquired an explicit source mapping is forbidden,
        // even when source/target ancestry and source Hand mapping are valid.
        let mut mapped_source = source.clone();
        mapped_source.joints.push(Joint {
            id: "marker".into(),
            name: "marker".into(),
            parent: Some(1),
            rest: Transform::translated([0., 0.2, 0.]),
            matrix: None,
            deform: true,
        });
        mapped_source.joints[2].parent = Some(4);
        mapped_source.joints[2].rest.translation = [0., 0.2, 0.];
        let rest = mapped_source.rest_pose();
        let mapped_clip =
            clip_from_poses("mapped", &[0., 1.], &[rest.clone(), rest], &[0, 1, 2, 3, 4]).unwrap();
        bad = config.clone();
        bad.joint_map
            .insert("LeftForeArmTwist1".into(), "marker".into());
        assert!(
            retarget(&mapped_source, &mapped_clip, &target, &bad)
                .unwrap_err()
                .to_string()
                .contains("support_must_be_unmapped")
        );
    }

    #[test]
    fn omitted_serial_wrist_config_preserves_legacy_direct_distribution() {
        let (source, target, clip, mut config) = serial_wrist_fixture(1, 2.6);
        config.distribute_wrist_twist = true;
        let (explicit, report) = retarget(&source, &clip, &target, &config).unwrap();
        let mut value = serde_json::to_value(&config).unwrap();
        value.as_object_mut().unwrap().remove("wrist_twist_chains");
        let legacy: RetargetConfig = serde_json::from_value(value).unwrap();
        assert!(legacy.wrist_twist_chains.is_empty());
        let (omitted, _) = retarget(&source, &clip, &target, &legacy).unwrap();
        assert_eq!(
            serde_json::to_value(&explicit).unwrap(),
            serde_json::to_value(&omitted).unwrap()
        );
        assert!(report.wrist_twist_chains.is_empty());
        assert_eq!(report.twist_distribution_joints, ["LeftHand"]);
        assert!(report.max_wrist_twist_transferred_rad > 2.5);
    }

    #[test]
    fn selective_reference_preservation_keeps_bent_legs_while_aligning_arms() {
        let joint = |name: &str, parent, translation| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(translation),
            matrix: None,
            deform: true,
        };
        let source = Skeleton {
            joints: vec![
                joint("Hips", None, [0., 1., 0.]),
                joint("LeftUpLeg", Some(0), [0.1, -0.1, 0.]),
                joint("LeftLeg", Some(1), [0., -0.4, 0.]),
                joint("LeftFoot", Some(2), [0., -0.4, 0.]),
                joint("LeftToeBase", Some(3), [0., -0.05, 0.15]),
                joint("RightUpLeg", Some(0), [-0.1, -0.1, 0.]),
                joint("RightLeg", Some(5), [0., -0.4, 0.]),
                joint("RightFoot", Some(6), [0., -0.4, 0.]),
                joint("RightToeBase", Some(7), [0., -0.05, 0.15]),
                joint("LeftArm", Some(0), [0.2, 0.4, 0.]),
                joint("LeftForeArm", Some(9), [0.3, 0., 0.]),
                joint("LeftHand", Some(10), [0.3, 0., 0.]),
            ],
        };
        let mut target = source.clone();
        for (knee, hock, toe) in [(2, 3, 4), (6, 7, 8)] {
            target.joints[knee].rest.translation = [0., -0.28, 0.10];
            target.joints[hock].rest.translation = [0., -0.22, -0.13];
            target.joints[toe].rest.translation = [0., -0.18, 0.16];
        }
        target.joints[10].rest.translation = [0.25, -0.18, 0.];
        target.joints[11].rest.translation = [0.25, -0.18, 0.];
        let rest = source.rest_pose();
        let mut moved = rest.clone();
        moved[1].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -0.2));
        let clip = clip_from_poses(
            "walk",
            &[0., 1.],
            &[rest, moved],
            &(0..source.joints.len()).collect::<Vec<_>>(),
        )
        .unwrap();
        let preserved: Vec<_> = (1..=8).map(|i| target.joints[i].name.clone()).collect();
        let config = RetargetConfig {
            align_reference_chains: true,
            preserve_reference_joints: preserved,
            root_motion: false,
            ..Default::default()
        };
        let (output, report) = retarget(&source, &clip, &target, &config).unwrap();
        let initial = output.sample(&target, 0.).unwrap();
        let expected = target.positions(&target.rest_pose()).unwrap();
        let actual = target.positions(&initial).unwrap();
        for i in 1..=8 {
            assert!((v3(expected[i]) - v3(actual[i])).norm() < 1e-8);
        }
        assert!(
            report
                .reference_chain_corrections
                .iter()
                .any(|s| s.contains("LeftArm:") && s.contains("reference swing"))
        );
        let animated = output.sample(&target, 1.).unwrap();
        assert!((v3(target.positions(&animated).unwrap()[4]) - v3(actual[4])).norm() > 0.05);
        assert_eq!(report.max_local_bone_length_error, 0.);
        let mut adapted = config.clone();
        adapted.root_scale = Some(0.35);
        adapted.limb_trajectories = vec![LimbTrajectory {
            target_chain: ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"].map(str::to_owned),
            target_anchor: "Hips".into(),
            position_scale: None,
            max_orientation_deviation_rad: None,
            max_end_orientation_deviation_rad: None,
        }];
        let (adapted_clip, adapted_report) = retarget(&source, &clip, &target, &adapted).unwrap();
        let source_reference = source.positions(&source.rest_pose()).unwrap();
        let source_animated = source
            .positions(&clip.sample(&source, 1.).unwrap())
            .unwrap();
        let desired = v3(expected[4])
            + ((v3(source_animated[4]) - v3(source_animated[0]))
                - (v3(source_reference[4]) - v3(source_reference[0])))
                * 0.35;
        let adapted_pose = adapted_clip.sample(&target, 1.).unwrap();
        assert!((v3(target.positions(&adapted_pose).unwrap()[4]) - desired).norm() < 1e-7);
        let fk_world = target.world_matrices(&animated).unwrap();
        let ik_world = target.world_matrices(&adapted_pose).unwrap();
        assert!(
            (world_rotation(&fk_world[3]).unwrap().inverse()
                * world_rotation(&ik_world[3]).unwrap())
            .angle()
                < 1e-7
        );
        assert!(adapted_report.limb_trajectories[0].max_position_error_m < 1e-7);
        assert_eq!(adapted_report.max_local_bone_length_error, 0.);
        let mut limited = adapted.clone();
        limited.limb_trajectories[0].max_orientation_deviation_rad = Some(0.);
        let (limited_clip, limited_report) = retarget(&source, &clip, &target, &limited).unwrap();
        let limited_pose = limited_clip.sample(&target, 1.).unwrap();
        let limited_world = target.world_matrices(&limited_pose).unwrap();
        assert!(world_rotation(&limited_world[3]).unwrap().angle() < 1e-7);
        assert!(
            (world_rotation(&fk_world[4]).unwrap().inverse()
                * world_rotation(&limited_world[4]).unwrap())
            .angle()
                < 1e-7
        );
        assert!((v3(target.positions(&limited_pose).unwrap()[4]) - desired).norm() < 1e-7);
        assert!(limited_report.limb_trajectories[0].max_orientation_correction_rad > 0.19);
        limited.limb_trajectories[0].max_orientation_deviation_rad = Some(-1.);
        assert!(retarget(&source, &clip, &target, &limited).is_err());
        let mut unreachable = adapted.clone();
        unreachable.limb_trajectories[0].position_scale = Some(1000.);
        let (_, unreachable_report) = retarget(&source, &clip, &target, &unreachable).unwrap();
        assert!(unreachable_report.limb_trajectories[0].max_position_error_m > 1.);
        let mut invalid_path = adapted.clone();
        invalid_path.limb_trajectories[0].target_chain[3] = "LeftHand".into();
        assert!(retarget(&source, &clip, &target, &invalid_path).is_err());
        invalid_path = adapted.clone();
        invalid_path
            .limb_trajectories
            .push(invalid_path.limb_trajectories[0].clone());
        assert!(retarget(&source, &clip, &target, &invalid_path).is_err());
        let mut invalid = config.clone();
        invalid.align_reference_chains = false;
        assert!(retarget(&source, &clip, &target, &invalid).is_err());
        invalid = config.clone();
        invalid
            .preserve_reference_joints
            .push("missing_hock".into());
        assert!(retarget(&source, &clip, &target, &invalid).is_err());
        invalid = config.clone();
        invalid.preserve_reference_joints.push("LeftFoot".into());
        assert!(retarget(&source, &clip, &target, &invalid).is_err());
    }

    fn end_orientation_fixture() -> (Skeleton, Skeleton, MotionClip, RetargetConfig) {
        let joint = |name: &str, parent, translation| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(translation),
            matrix: None,
            deform: true,
        };
        let source = Skeleton {
            joints: vec![
                joint("Hips", None, [0., 1., 0.]),
                joint("LeftUpLeg", Some(0), [0.1, -0.1, 0.]),
                joint("LeftLeg", Some(1), [0., -0.4, 0.]),
                joint("LeftFoot", Some(2), [0., -0.4, 0.]),
                joint("LeftToeBase", Some(3), [0., -0.05, 0.15]),
            ],
        };
        let mut target = source.clone();
        target.joints[2].rest.translation = [0., -0.28, 0.12];
        target.joints[3].rest.translation = [0., -0.22, -0.14];
        target.joints[4].rest.translation = [0., -0.17, 0.14];
        target.joints[4].rest.rotation =
            a4(UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.2));
        let initial = source.rest_pose();
        let mut moved = initial.clone();
        moved[0].translation = [0.05, 1.02, 0.1];
        moved[1].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -0.15));
        moved[3].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.7));
        moved[4].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -0.8));
        let clip = clip_from_poses("walk", &[0., 1.], &[initial, moved], &[0, 1, 2, 3, 4]).unwrap();
        let config = RetargetConfig {
            fps: 4.,
            root_scale: Some(0.2),
            min_target_coverage: 1.,
            limb_trajectories: vec![LimbTrajectory {
                target_chain: ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"]
                    .map(str::to_owned),
                target_anchor: "Hips".into(),
                position_scale: Some(0.05),
                max_orientation_deviation_rad: Some(0.1),
                max_end_orientation_deviation_rad: None,
            }],
            ..Default::default()
        };
        (source, target, clip, config)
    }

    fn assert_same_trajectory_geometry(target: &Skeleton, before: &MotionClip, after: &MotionClip) {
        assert_eq!(before.time_range(), after.time_range());
        assert_eq!(before.tracks.len(), after.tracks.len());
        for (a, b) in before.tracks.iter().zip(&after.tracks) {
            assert_eq!(a.joint, b.joint);
            assert_eq!(a.times, b.times);
        }
        for time in [0., 0.25, 0.5, 0.75, 1.] {
            let a = before.sample(target, time).unwrap();
            let b = after.sample(target, time).unwrap();
            let pa = target.positions(&a).unwrap();
            let pb = target.positions(&b).unwrap();
            for (i, joint) in target.joints.iter().enumerate() {
                assert_eq!(a[i].translation, b[i].translation);
                assert_eq!(a[i].scale, b[i].scale);
                assert!((v3(pa[i]) - v3(pb[i])).norm() < 1e-10);
                if let Some(parent) = joint.parent {
                    assert!(
                        ((v3(pb[i]) - v3(pb[parent])).norm() - v3(joint.rest.translation).norm())
                            .abs()
                            < 1e-10
                    );
                }
            }
        }
    }

    #[test]
    fn end_orientation_limit_is_opt_in_and_preserves_geometry_and_timing() {
        let (source, target, clip, config) = end_orientation_fixture();
        let (baseline, baseline_report) = retarget(&source, &clip, &target, &config).unwrap();
        let mut fk_config = config.clone();
        fk_config.limb_trajectories.clear();
        let (fk, _) = retarget(&source, &clip, &target, &fk_config).unwrap();
        let observed = target.world_matrices(&target.rest_pose()).unwrap();
        let reference = world_rotation(&observed[4]).unwrap();
        let mut max_fk_deviation = 0.0_f64;
        for time in [0., 0.25, 0.5, 0.75, 1.] {
            let expected = target
                .world_matrices(&fk.sample(&target, time).unwrap())
                .unwrap();
            let actual = target
                .world_matrices(&baseline.sample(&target, time).unwrap())
                .unwrap();
            let end = world_rotation(&actual[4]).unwrap();
            assert!((world_rotation(&expected[4]).unwrap().inverse() * end).angle() < 1e-8);
            max_fk_deviation = max_fk_deviation.max((reference.inverse() * end).angle());
        }
        assert!(max_fk_deviation > 0.24);
        assert_eq!(
            baseline_report.limb_trajectories[0].max_end_orientation_correction_rad,
            0.
        );

         
        let mut legacy = serde_json::to_value(&config.limb_trajectories[0]).unwrap();
        legacy
            .as_object_mut()
            .unwrap()
            .remove("max_end_orientation_deviation_rad");
        let mut omitted_config = config.clone();
        omitted_config.limb_trajectories[0] = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            omitted_config.limb_trajectories[0].max_end_orientation_deviation_rad,
            None
        );
        let (omitted, _) = retarget(&source, &clip, &target, &omitted_config).unwrap();
        assert_eq!(
            serde_json::to_value(&omitted).unwrap(),
            serde_json::to_value(&baseline).unwrap()
        );
        let mut inactive = config.clone();
        inactive.limb_trajectories[0].max_end_orientation_deviation_rad =
            Some(std::f64::consts::PI);
        let (inactive_clip, _) = retarget(&source, &clip, &target, &inactive).unwrap();
        assert_eq!(
            serde_json::to_value(&inactive_clip).unwrap(),
            serde_json::to_value(&baseline).unwrap()
        );

        for limit in [0., 0.1] {
            let mut limited = config.clone();
            limited.limb_trajectories[0].max_end_orientation_deviation_rad = Some(limit);
            let (output, report) = retarget(&source, &clip, &target, &limited).unwrap();
            assert_same_trajectory_geometry(&target, &baseline, &output);
            assert_eq!(report.sample_count, baseline_report.sample_count);
            assert_eq!(report.max_local_bone_length_error, 0.);
            assert!(report.limb_trajectories[0].max_position_error_m < 1e-8);
            for time in [0., 0.25, 0.5, 0.75, 1.] {
                let actual = target
                    .world_matrices(&output.sample(&target, time).unwrap())
                    .unwrap();
                assert!(
                    (reference.inverse() * world_rotation(&actual[4]).unwrap()).angle()
                        <= limit + 1e-8
                );
            }
            let end = target
                .world_matrices(&output.sample(&target, 1.).unwrap())
                .unwrap();
            assert!(
                ((reference.inverse() * world_rotation(&end[4]).unwrap()).angle() - limit).abs()
                    < 1e-8
            );
            assert!(
                (report.limb_trajectories[0].max_end_orientation_correction_rad
                    - (max_fk_deviation - limit))
                    .abs()
                    < 1e-8
            );
        }
    }

    #[test]
    fn end_orientation_limit_rejects_nonfinite_and_out_of_range_bounds() {
        let (source, target, clip, mut config) = end_orientation_fixture();
        for limit in [
            -0.1,
            std::f64::consts::PI + 0.1,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            config.limb_trajectories[0].max_end_orientation_deviation_rad = Some(limit);
            let error = retarget(&source, &clip, &target, &config).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("invalid_trajectory_end_orientation_limit")
            );
        }
    }

    #[test]
    fn end_orientation_zero_limit_transports_observed_pose_with_rotating_anchor() {
        let (mut source, mut target, _, mut config) = end_orientation_fixture();
        source.joints[0].rest.rotation = a4(UnitQuaternion::from_euler_angles(0.2, -0.15, 0.1));
        target.joints[0].rest.rotation = a4(UnitQuaternion::from_euler_angles(-0.3, 0.2, -0.15));
        target.joints[4].rest.rotation = a4(UnitQuaternion::from_euler_angles(0.1, 0.35, -0.2));
        let initial = source.rest_pose();
        let mut moved = initial.clone();
        moved[0].rotation =
            a4(unit_quat(initial[0].rotation) * UnitQuaternion::from_euler_angles(0.1, 0.5, -0.2));
        moved[1].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -0.15));
        moved[3].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.7));
        moved[4].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -0.8));
        let clip = clip_from_poses(
            "turning_walk",
            &[0., 1.],
            &[initial, moved],
            &[0, 1, 2, 3, 4],
        )
        .unwrap();
        let (baseline, _) = retarget(&source, &clip, &target, &config).unwrap();
        config.limb_trajectories[0].max_end_orientation_deviation_rad = Some(0.);
        let (output, report) = retarget(&source, &clip, &target, &config).unwrap();
        assert_same_trajectory_geometry(&target, &baseline, &output);
        let observed = target.world_matrices(&target.rest_pose()).unwrap();
        let mut max_transport = 0.0_f64;
        for time in [0., 0.25, 0.5, 0.75, 1.] {
            let actual = target
                .world_matrices(&output.sample(&target, time).unwrap())
                .unwrap();
            let expected = world_rotation(&actual[0]).unwrap()
                * world_rotation(&observed[0]).unwrap().inverse()
                * world_rotation(&observed[4]).unwrap();
            assert!((expected.inverse() * world_rotation(&actual[4]).unwrap()).angle() < 1e-8);
            max_transport = max_transport
                .max((world_rotation(&observed[4]).unwrap().inverse() * expected).angle());
        }
        assert!(max_transport > 0.4);
        assert!(report.limb_trajectories[0].max_end_orientation_correction_rad > 0.1);
    }

    #[test]
    fn mpfb_default_rig_has_unique_body_keys_without_mapping_auxiliary_bones() {
        let names = [
            "root",
            "spine05",
            "spine04",
            "spine03",
            "spine02",
            "spine01",
            "upperarm01.L",
            "upperarm01.R",
            "lowerarm02.L",
            "lowerarm02.R",
            "wrist.L",
            "wrist.R",
            "upperleg01.L",
            "upperleg01.R",
            "foot.L",
            "foot.R",
            "lowerarm01.L",
            "lowerleg01.L",
            "finger2-3.L",
            "toe1-1.R",
            "head",
        ];
        let skeleton = Skeleton {
            joints: names
                .iter()
                .map(|name| Joint {
                    id: name.to_string(),
                    name: name.to_string(),
                    parent: None,
                    rest: Transform::default(),
                    matrix: None,
                    deform: true,
                })
                .collect(),
        };
        let keys = semantic_keys(&skeleton);
        let key = |name: &str| keys[names.iter().position(|n| *n == name).unwrap()].as_str();
        assert_eq!(key("root"), "pelvis");
        assert_eq!(key("spine01"), "spine_2");
        assert_eq!(key("lowerarm01.L"), "left_forearm");
        assert_eq!(key("wrist.R"), "right_hand");
        assert_eq!(key("finger2-3.L"), "left_index_3");
        assert_eq!(key("toe1-1.R"), "right_toe");
        assert!(key("lowerarm02.L").starts_with("name:"));
        assert_eq!(keys.iter().collect::<HashSet<_>>().len(), keys.len());
        let mut browser_export = skeleton.clone();
        for joint in &mut browser_export.joints {
            joint.name = joint.name.replace('.', "");
        }
        assert_eq!(semantic_keys(&browser_export), keys);
        let mut unknown = skeleton.clone();
        unknown
            .joints
            .retain(|j| j.name == "root" || j.name == "upperarm01.L");
        assert_eq!(
            semantic_keys(&unknown),
            vec!["name:root", "name:upperarm01l"]
        );
    }

    fn finger_rig(target: bool) -> Skeleton {
        let mut joints = Vec::new();
        for (side, sign) in [("Left", 1.0), ("Right", -1.0)] {
            let hand = joints.len();
            joints.push(Joint {
                id: format!("{side}Hand"),
                name: format!("{side}Hand"),
                parent: None,
                rest: Transform {
                    translation: [sign * 0.4, 0.0, 0.0],
                    rotation: a4(UnitQuaternion::from_axis_angle(
                        &Vector3::y_axis(),
                        if target { sign * 1.3 } else { 0.0 },
                    )),
                    ..Transform::default()
                },
                matrix: None,
                deform: true,
            });
            for (digit, source_name, spread) in [
                ("index", "Index", 0.04),
                ("middle", "Middle", 0.0),
                ("little", "Pinky", -0.04),
            ] {
                let mut parent = hand;
                for rank in 1..=if target { 3 } else { 4 } {
                    let name = if target {
                        format!("HC_{side}_{digit}_{rank}")
                    } else {
                        format!("{side}Hand{source_name}{rank}")
                    };
                    let length_scale = if target { 1.4 } else { 1.0 };
                    joints.push(Joint {
                        id: name.clone(),
                        name,
                        parent: Some(parent),
                        rest: Transform {
                            translation: if rank == 1 {
                                [sign * spread * length_scale, 0.1 * length_scale, 0.0]
                            } else {
                                [0.0, 0.035 * length_scale, 0.0]
                            },
                            rotation: a4(UnitQuaternion::from_axis_angle(
                                &Vector3::x_axis(),
                                if target { sign * 0.8 } else { 0.0 },
                            )),
                            ..Transform::default()
                        },
                        matrix: None,
                        deform: rank < 4,
                    });
                    parent = joints.len() - 1;
                }
            }
        }
        Skeleton { joints }
    }

    #[test]
    fn explicit_local_pose_preservation_keeps_reviewed_fingers_and_parent_motion() {
        let source = finger_rig(false);
        let target = finger_rig(true);
        let mut moved = source.rest_pose();
        for (i, pose) in moved.iter_mut().enumerate() {
            if source.joints[i].name.contains("Hand") {
                pose.rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.8));
            }
        }
        let clip = clip_from_poses(
            "walk",
            &[0., 1.],
            &[source.rest_pose(), moved],
            &(0..source.joints.len()).collect::<Vec<_>>(),
        )
        .unwrap();
        let keys = semantic_keys(&target);
        let names: Vec<_> = target
            .joints
            .iter()
            .enumerate()
            .filter(|(i, _)| digit_rank(&keys[*i]).is_some_and(|rank| rank < 4))
            .map(|(_, j)| j.name.clone())
            .collect();
        assert!(!names.is_empty());
        let base = RetargetConfig {
            align_reference_chains: true,
            ..Default::default()
        };
        let (ordinary, _) = retarget(&source, &clip, &target, &base).unwrap();
        let config = RetargetConfig {
            preserve_local_pose_joints: names.clone(),
            ..base.clone()
        };
        let (preserved, report) = retarget(&source, &clip, &target, &config).unwrap();
        assert_eq!(report.preserved_local_pose_joints, names);
        assert_eq!(ordinary.time_range(), preserved.time_range());
        for time in [0., 0.37, 1.] {
            let pose = preserved.sample(&target, time).unwrap();
            let regular = ordinary.sample(&target, time).unwrap();
            for (i, joint) in target.joints.iter().enumerate() {
                if names.contains(&joint.name) {
                    assert!(
                        unit_quat(pose[i].rotation).angle_to(&unit_quat(joint.rest.rotation))
                            < 1e-7
                    );
                    assert_eq!(pose[i].translation, joint.rest.translation);
                    assert_eq!(pose[i].scale, joint.rest.scale);
                } else {
                    assert!(
                        unit_quat(pose[i].rotation).angle_to(&unit_quat(regular[i].rotation))
                            < 1e-7
                    );
                }
            }
        }
        let mut bad = config.clone();
        bad.preserve_local_pose_joints.push(names[0].clone());
        assert!(
            retarget(&source, &clip, &target, &bad)
                .unwrap_err()
                .to_string()
                .contains("duplicate_preserved_local_pose_joint")
        );
        bad.preserve_local_pose_joints = vec!["missing".into()];
        assert!(retarget(&source, &clip, &target, &bad).is_err());
        let omitted: RetargetConfig =
            serde_json::from_str(r#"{"align_reference_chains":true}"#).unwrap();
        let (legacy, _) = retarget(&source, &clip, &target, &omitted).unwrap();
        assert_eq!(
            serde_json::to_value(legacy).unwrap(),
            serde_json::to_value(ordinary).unwrap()
        );
    }

    #[test]
    fn retarget_aligns_both_palm_planes_and_missing_terminal_tips() {
        let source = finger_rig(false);
        let target = finger_rig(true);
        let keys = semantic_keys(&source);
        let mut curl = source.rest_pose();
        for (i, pose) in curl.iter_mut().enumerate() {
            let sign = if keys[i].starts_with("left_") {
                1.0
            } else {
                -1.0
            };
            pose.rotation = a4(UnitQuaternion::from_axis_angle(
                &Vector3::x_axis(),
                if digit_rank(&keys[i]).is_some_and(|rank| rank < 4) {
                    sign * 0.55
                } else {
                    0.0
                },
            ));
        }
        let clip = clip_from_poses(
            "curl",
            &[0.0, 1.0],
            &[source.rest_pose(), curl],
            &(0..source.joints.len()).collect::<Vec<_>>(),
        )
        .unwrap();
        let (result, report) = retarget(
            &source,
            &clip,
            &target,
            &RetargetConfig {
                align_reference_chains: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            report
                .reference_chain_corrections
                .iter()
                .filter(|s| s.contains("palm-frame"))
                .count(),
            2
        );
        assert_eq!(
            report
                .reference_chain_corrections
                .iter()
                .filter(|s| s.contains("terminal digit"))
                .count(),
            6
        );
        for time in [0.0, 0.37, 1.0] {
            let sw = source
                .world_matrices(&clip.sample(&source, time).unwrap())
                .unwrap();
            let pose = result.sample(&target, time).unwrap();
            let tw = target.world_matrices(&pose).unwrap();
            for m in &report.mapping.mappings {
                // Every phalanx, including terminal bones with no target tip,
                // must point in the same anatomical direction after transfer.
                if digit_rank(&keys[m.source]).is_some() {
                    let expected = world_rotation(&sw[m.source]).unwrap() * Vector3::y();
                    let actual = world_rotation(&tw[m.target]).unwrap() * Vector3::y();
                    assert!(
                        expected.dot(&actual) > 0.999999,
                        "{} at {time}",
                        keys[m.source]
                    );
                }
                assert_eq!(
                    pose[m.target].translation,
                    target.joints[m.target].rest.translation
                );
            }
            for side in ["left", "right"] {
                let mapped = |name: &str| {
                    report
                        .mapping
                        .mappings
                        .iter()
                        .find(|m| keys[m.source] == format!("{side}_{name}"))
                        .unwrap()
                };
                let hand = mapped("hand");
                let middle = mapped("middle_1");
                let index = mapped("index_1");
                let little = mapped("little_1");
                let point = |w: &[nalgebra::Matrix4<f64>], i| v3(transform_point(&w[i], [0.0; 3]));
                let sf = palm_frame(
                    point(&sw, hand.source),
                    point(&sw, middle.source),
                    point(&sw, index.source),
                    point(&sw, little.source),
                )
                .unwrap();
                let tf = palm_frame(
                    point(&tw, hand.target),
                    point(&tw, middle.target),
                    point(&tw, index.target),
                    point(&tw, little.target),
                )
                .unwrap();
                assert!(sf.angle_to(&tf) < 1e-7, "{side} palm at {time}");
            }
        }
    }

    /// A forearm, an unmapped twist support and a hand whose palm is turned
    /// 1.2 rad about the forearm (a palms-forward target) against a source
    /// arm with a plain palm.
    fn twisted_palm_arm(target: bool) -> Skeleton {
        let joint = |name: &str, parent: Option<usize>, t: [f64; 3], roll: f64| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform {
                translation: t,
                rotation: a4(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), roll)),
                ..Transform::default()
            },
            matrix: None,
            deform: true,
        };
        let mut joints = vec![joint("LeftForeArm", None, [0.0; 3], 0.0)];
        if target {
            // Slightly off the forearm line, as a refitted rig's is.
            joints.push(joint("LeftForeArmTwist", Some(0), [0.12, 0.01, 0.0], 0.0));
            joints.push(joint("LeftHand", Some(1), [0.13, -0.01, 0.0], 1.2));
        } else {
            joints.push(joint("LeftHand", Some(0), [0.25, 0.0, 0.0], 0.0));
        }
        let hand = joints.len() - 1;
        for (name, spread) in [("Index", 0.03), ("Middle", 0.0), ("Pinky", -0.03)] {
            joints.push(joint(
                &format!("LeftHand{name}1"),
                Some(hand),
                [0.1, 0.0, spread],
                0.0,
            ));
        }
        Skeleton { joints }
    }

    #[test]
    fn palm_twist_is_spread_over_the_forearm_without_moving_the_hand() {
        let source = twisted_palm_arm(false);
        let target = twisted_palm_arm(true);
        let mut bent = source.rest_pose();
        bent[0].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.5));
        bent[1].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.3));
        let clip = clip_from_poses(
            "bend",
            &[0.0, 1.0],
            &[source.rest_pose(), bent],
            &(0..source.joints.len()).collect::<Vec<_>>(),
        )
        .unwrap();
        let run = |distribute_palm_twist| {
            retarget(
                &source,
                &clip,
                &target,
                &RetargetConfig {
                    align_reference_chains: true,
                    distribute_palm_twist,
                    ..Default::default()
                },
            )
            .unwrap()
        };
        let (plain, _) = run(false);
        let (spread, report) = run(true);
        assert!(
            report
                .reference_chain_corrections
                .iter()
                .any(|c| c.contains("spread over LeftForeArm, LeftForeArmTwist, LeftHand")),
            "{:?}",
            report.reference_chain_corrections
        );
        let axial = |q: UnitQuaternion<f64>, axis: Vector3<f64>| {
            let q = if q.w < 0.0 {
                -q.into_inner()
            } else {
                q.into_inner()
            };
            2.0 * q.imag().dot(&axis).atan2(q.w)
        };
        let point = |w: &[Matrix4<f64>], i: usize| v3(transform_point(&w[i], [0.0; 3]));
        for time in [0.0, 0.5, 1.0] {
            let a = plain.sample(&target, time).unwrap();
            let b = spread.sample(&target, time).unwrap();
            let (wa, wb) = (
                target.world_matrices(&a).unwrap(),
                target.world_matrices(&b).unwrap(),
            );
            // The hand and fingers end where they did; the forearm points the
            // same way.
            for j in 2..target.joints.len() {
                assert!((wa[j] - wb[j]).norm() < 1e-9, "joint {j} at {time}");
            }
            let dir = |w: &[Matrix4<f64>]| (point(w, 2) - point(w, 0)).normalize();
            assert!(dir(&wa).dot(&dir(&wb)) > 1.0 - 1e-12);
            // Without the spread the wrist takes all of the 1.2 rad; with
            // it the forearm and the hand take a quarter, the support half.
            let rest = |j: usize| unit_quat(target.joints[j].rest.rotation);
            let local_axis = v3(target.joints[2].rest.translation).normalize();
            let wring =
                |p: &[Transform]| axial(unit_quat(p[2].rotation) * rest(2).inverse(), local_axis);
            assert!((wring(&a).abs() - 1.2).abs() < 0.02, "{}", wring(&a));
            assert!((wring(&b).abs() - 0.3).abs() < 0.02, "{}", wring(&b));
            let support_axis = v3(target.joints[1].rest.translation).normalize();
            let support = axial(unit_quat(b[1].rotation) * rest(1).inverse(), support_axis);
            assert!((support.abs() - 0.6).abs() < 0.02, "{support}");
        }
    }

    #[test]
    fn sampling_preserves_exact_end_without_float32_collisions() {
        let end = 5.433333396911621;
        let times = sample_times(0.0, end, 60.0).unwrap();
        assert_eq!(*times.last().unwrap(), end);
        assert!(times.windows(2).all(|w| (w[0] as f32) < (w[1] as f32)));
        assert_eq!(times.len(), 327);
    }

    #[test]
    fn posed_reference_aligns_hip_and_shoulder_spread_and_marked_face() {
        let joint = |name: &str, parent, translation, deform| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(translation),
            matrix: None,
            deform,
        };
        let source = Skeleton {
            joints: vec![
                joint("Hips", None, [0.0, 1.0, 0.0], true),
                joint("LeftUpLeg", Some(0), [0.12, -0.08, 0.0], true),
                joint("RightUpLeg", Some(0), [-0.12, -0.08, 0.0], true),
                joint("Spine", Some(0), [0.0, 0.18, 0.0], true),
                joint("Spine1", Some(3), [0.0, 0.18, 0.0], true),
                joint("LeftShoulder", Some(4), [0.1, 0.08, 0.0], true),
                joint("RightShoulder", Some(4), [-0.1, 0.08, 0.0], true),
                joint("neck", Some(4), [0.0, 0.15, 0.0], true),
                joint("Head", Some(7), [0.0, 0.1, 0.0], true),
                joint("headfront", Some(8), [0.0, 0.0, 0.06], false),
                joint("head_end", Some(8), [0.0, 0.2, 0.02], false),
            ],
        };
        let mut target = source.clone();
        // An exported archer stance: hips face sideways, chest twists, head
        // counter-turns. Aligning only the spine's up vector leaves the legs
        // one in front of the other, despite correct individual leg swings.
        target.joints[0].rest.rotation = a4(UnitQuaternion::from_axis_angle(
            &Vector3::y_axis(),
            std::f64::consts::FRAC_PI_2,
        ));
        target.joints[4].rest.rotation =
            a4(UnitQuaternion::from_axis_angle(&Vector3::y_axis(), -0.6));
        target.joints[8].rest.rotation = a4(UnitQuaternion::from_axis_angle(
            &Vector3::y_axis(),
            0.6 - std::f64::consts::FRAC_PI_2,
        ));
        for joint in &mut target.joints[1..] {
            joint.rest.translation[0] *= 0.8;
            joint.rest.translation[1] *= 1.2;
        }
        let mut pose = source.rest_pose();
        pose[0].rotation = a4(UnitQuaternion::from_euler_angles(0.04, 0.2, 0.08));
        pose[3].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.3));
        pose[4].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.12));
        pose[8].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.35));
        let clip = clip_from_poses(
            "body",
            &[0.0, 1.0],
            &[source.rest_pose(), pose],
            &(0..source.joints.len()).collect::<Vec<_>>(),
        )
        .unwrap();
        for source_has_face_marker in [true, false] {
            let mut source = source.clone();
            if !source_has_face_marker {
                source.joints[9].name = "unknown_marker".into();
            }
            let (result, report) = retarget(
                &source,
                &clip,
                &target,
                &RetargetConfig {
                    align_reference_chains: true,
                    root_motion: false,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                report
                    .reference_chain_corrections
                    .iter()
                    .filter(|s| s.contains("bilateral axial-frame"))
                    .count(),
                3
            );
            assert!(
                report
                    .reference_chain_corrections
                    .iter()
                    .any(|s| s.contains("marked head-facing"))
            );
            for time in [0.0, 0.37, 1.0] {
                let sp = source
                    .positions(&clip.sample(&source, time).unwrap())
                    .unwrap();
                let tp = result.sample(&target, time).unwrap();
                let points = target.positions(&tp).unwrap();
                for (a, b) in [(1, 2), (5, 6), (9, 8)] {
                    let expected = (v3(sp[a]) - v3(sp[b])).normalize();
                    let actual = (v3(points[a]) - v3(points[b])).normalize();
                    assert!(
                        expected.dot(&actual) > 0.999999,
                        "landmarks {a}/{b} at {time}, marker={source_has_face_marker}"
                    );
                }
                for (i, transform) in tp.iter().enumerate() {
                    assert_eq!(transform.translation, target.joints[i].rest.translation);
                }
            }
        }
    }
    #[test]
    fn aligns_a_pose_to_source_chain_without_changing_bone_lengths() {
        let joint = |name: &str, parent, translation| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(translation),
            matrix: None,
            deform: true,
        };
        let source = Skeleton {
            joints: vec![
                joint("LeftArm", None, [0.0; 3]),
                joint("LeftForeArm", Some(0), [1.0, 0.0, 0.0]),
                joint("LeftHand", Some(1), [1.0, 0.0, 0.0]),
            ],
        };
        let mut target = source.clone();
        target.joints[0].rest.rotation =
            a4(UnitQuaternion::from_axis_angle(&Vector3::z_axis(), -0.6));
        target.joints[1].rest.translation = [0.7, 0.0, 0.0];
        target.joints[2].rest.translation = [0.5, 0.0, 0.0];
        let clip = clip_from_poses(
            "neutral",
            &[0.0, 1.0],
            &[source.rest_pose(), source.rest_pose()],
            &[0, 1, 2],
        )
        .unwrap();
        let config = RetargetConfig {
            align_reference_chains: true,
            ..Default::default()
        };
        let (result, report) = retarget(&source, &clip, &target, &config).unwrap();
        let pose = result.sample(&target, 0.0).unwrap();
        let hand = target.positions(&pose).unwrap()[2];
        assert!((hand[0] - 1.2).abs() < 1e-8 && hand[1].abs() < 1e-8);
        assert_eq!(pose[1].translation, target.joints[1].rest.translation);
        assert_eq!(pose[2].translation, target.joints[2].rest.translation);
        assert!(!report.reference_chain_corrections.is_empty());
    }
    fn rig(rotated: bool, length: f64) -> Skeleton {
        let r = if rotated {
            a4(UnitQuaternion::from_axis_angle(
                &nalgebra::Vector3::z_axis(),
                0.7,
            ))
        } else {
            [0.0, 0.0, 0.0, 1.0]
        };
        Skeleton {
            joints: vec![
                Joint {
                    id: "a".into(),
                    name: "Hips".into(),
                    parent: None,
                    rest: Transform {
                        rotation: r,
                        ..Transform::default()
                    },
                    matrix: None,
                    deform: true,
                },
                Joint {
                    id: "b".into(),
                    name: "LeftUpLeg".into(),
                    parent: Some(0),
                    rest: Transform::translated([0.0, -length, 0.0]),
                    matrix: None,
                    deform: true,
                },
            ],
        }
    }

    fn scoped_body_fixture() -> (Skeleton, Skeleton, MotionClip) {
        let source = rig(false, 1.0);
        let mut target = rig(true, 2.0);
        for name in ["authored_lid", "authored_jaw", "authored_cheek"] {
            target.joints.push(Joint {
                id: name.into(),
                name: name.into(),
                parent: Some(0),
                rest: Transform::translated([0.1, 0.2, 0.3]),
                matrix: None,
                deform: true,
            });
        }
        target.joints.push(Joint {
            id: "control_only".into(),
            name: "control_only".into(),
            parent: Some(0),
            rest: Transform::default(),
            matrix: None,
            deform: false,
        });
        let mut moved = source.rest_pose();
        moved[0].translation = [0.1, 0.05, 0.];
        moved[0].rotation = a4(UnitQuaternion::from_euler_angles(0.2, -0.1, 0.3));
        moved[1].rotation = a4(UnitQuaternion::from_euler_angles(-0.3, 0.1, 0.));
        let clip = clip_from_poses(
            "body_only",
            &[0., 1.],
            &[source.rest_pose(), moved],
            &[0, 1],
        )
        .unwrap();
        (source, target, clip)
    }

    #[test]
    fn explicit_target_scope_gates_required_body_without_claiming_full_coverage() {
        let (source, target, clip) = scoped_body_fixture();
        // The full rig remains below the historical 0.5 gate.
        let error = retarget(&source, &clip, &target, &RetargetConfig::default()).unwrap_err();
        assert!(error.to_string().contains("target coverage 0.400"));
        let config = RetargetConfig {
            min_target_coverage: 1.,
            target_joint_scope: Some(vec!["LeftUpLeg".into(), "Hips".into()]),
            ..Default::default()
        };
        let (output, report) = retarget(&source, &clip, &target, &config).unwrap();
        assert_eq!(report.mapping.target_deform_coverage, 0.4);
        assert_eq!(report.mapping.unmapped_target.len(), 3);
        let scope = report.target_joint_scope.unwrap();
        let names: Vec<_> = scope.joints.iter().map(|j| j.name.clone()).collect();
        assert_eq!(names, config.target_joint_scope.unwrap());
        assert_eq!(scope.mapped_joints, names);
        assert!(scope.joints.iter().all(|j| j.deform && j.mapped));
        assert!(scope.unmapped_joints.is_empty());
        assert_eq!(scope.target_joint_coverage, 1.);
        for time in [0., 0.37, 1.] {
            let pose = output.sample(&target, time).unwrap();
            for (local, joint) in pose.iter().zip(&target.joints).skip(2) {
                assert_eq!(*local, joint.rest);
            }
        }
        assert_eq!(report.max_local_bone_length_error, 0.);
    }

    #[test]
    fn target_scope_is_not_inferred_from_successful_mappings_or_a_motion_mask() {
        let (source, target, clip) = scoped_body_fixture();
        let mut config = RetargetConfig {
            min_target_coverage: 1.,
            target_joint_scope: Some(vec!["Hips".into(), "authored_lid".into()]),
            ..Default::default()
        };
        let error = retarget(&source, &clip, &target, &config).unwrap_err();
        assert!(error.to_string().contains("scoped target coverage 0.500"));
        assert!(error.to_string().contains("unmapped scope: authored_lid"));
        config.min_target_coverage = 0.5;
        let (_, report) = retarget(&source, &clip, &target, &config).unwrap();
        let scope = report.target_joint_scope.unwrap();
        assert_eq!(scope.unmapped_joints, ["authored_lid"]);
        assert_eq!(scope.mapped_joints, ["Hips"]);
        assert_eq!(report.mapping.target_deform_coverage, 0.4);

        config.min_target_coverage = 1.;
        config.target_joint_scope = Some(vec!["Hips".into()]);
        let (scoped, _) = retarget(&source, &clip, &target, &config).unwrap();
        let (unscoped, _) = retarget(
            &source,
            &clip,
            &target,
            &RetargetConfig {
                min_target_coverage: 0.4,
                ..Default::default()
            },
        )
        .unwrap();
        // The unlisted mapped leg still moves exactly as before. Scope is only
        // a caller-authored denominator and cannot silently filter animation.
        assert_eq!(
            serde_json::to_value(scoped).unwrap(),
            serde_json::to_value(unscoped).unwrap()
        );
    }

    #[test]
    fn target_scope_rejects_empty_duplicate_and_unknown_rosters() {
        let (source, target, clip) = scoped_body_fixture();
        for (names, expected) in [
            (vec![], "target_joint_scope_must_not_be_empty"),
            (vec!["Hips", "Hips"], "duplicate_target_joint_scope_name"),
            (
                vec!["missing_joint"],
                "joint name must identify exactly one node",
            ),
        ] {
            let config = RetargetConfig {
                target_joint_scope: Some(names.into_iter().map(String::from).collect()),
                ..Default::default()
            };
            let error = retarget(&source, &clip, &target, &config).unwrap_err();
            assert!(error.to_string().contains(expected), "{error}");
        }
    }

    #[test]
    fn scope_requires_a_real_mapping_for_zero_weight_root_and_exports_its_motion() {
        let source = rig(false, 1.);
        let mut target = rig(true, 2.);
        target.joints[0].deform = false;
        // Reproduce a real glTF palette with only the weighted descendant.
        // The unweighted Hips is still a required locomotion control and is
        // an ancestor of every weighted vertex, despite deform=false on load.
        let mut doc = charloom_io::mesh_document(
            "weighted_descendant",
            &[[0., -2., 0.], [0.1, -2., 0.], [0., -1.9, 0.]],
            &[[0, 1, 2]],
        )
        .unwrap();
        let offset = doc.append_skeleton(&target).unwrap();
        let worlds = target.world_matrices(&target.rest_pose()).unwrap();
        let inverse = doc.append_matrices(&[matrix_array(&worlds[1].try_inverse().unwrap())]);
        doc.json["skins"] = serde_json::json!([{
            "joints":[offset + 1], "inverseBindMatrices":inverse
        }]);
        doc.json["nodes"][0]["skin"] = serde_json::json!(0);
        let joints = doc.append_accessor(&[0; 24], 5123, "VEC4", 3, Some(34962), None);
        let weight_bytes: Vec<_> = [1f32, 0., 0., 0.]
            .into_iter()
            .cycle()
            .take(12)
            .flat_map(f32::to_le_bytes)
            .collect();
        let weights = doc.append_accessor(&weight_bytes, 5126, "VEC4", 3, Some(34962), None);
        doc.json["meshes"][0]["primitives"][0]["attributes"]["JOINTS_0"] =
            serde_json::json!(joints);
        doc.json["meshes"][0]["primitives"][0]["attributes"]["WEIGHTS_0"] =
            serde_json::json!(weights);
        let temp = tempfile::tempdir().unwrap();
        let input = temp.path().join("root-not-in-palette.glb");
        doc.save(&input).unwrap();
        let mut loaded = charloom_io::load_gltf(&input).unwrap();
        let root = unique_name(&loaded.asset.skeleton, "Hips").unwrap();
        let leg = unique_name(&loaded.asset.skeleton, "LeftUpLeg").unwrap();
        assert!(!loaded.asset.skeleton.joints[root].deform);
        assert_eq!(loaded.asset.skeleton.joints[leg].parent, Some(root));
        assert!(
            loaded.asset.meshes[0]
                .influences
                .iter()
                .all(|row| { row.len() == 1 && row[0].joint == leg && row[0].weight == 1. })
        );
        let mut moved = source.rest_pose();
        moved[0].translation = [0.1, 0.05, 0.];
        let clip = clip_from_poses(
            "locomotion",
            &[0., 1.],
            &[source.rest_pose(), moved],
            &[0, 1],
        )
        .unwrap();
        let mut config = RetargetConfig {
            target_joint_scope: Some(vec!["Hips".into(), "LeftUpLeg".into()]),
            min_target_coverage: 1.,
            root_scale: Some(2.),
            ..Default::default()
        };
        let error = retarget(&source, &clip, &loaded.asset.skeleton, &config).unwrap_err();
        assert!(error.to_string().contains("unmapped scope: Hips"));
        config.joint_map.insert("Hips".into(), "Hips".into());
        let (motion, report) = retarget(&source, &clip, &loaded.asset.skeleton, &config).unwrap();
        let scope = report.target_joint_scope.unwrap();
        assert_eq!(scope.target_joint_coverage, 1.);
        assert!(!scope.joints[0].deform && scope.joints[0].mapped);
        assert!(scope.joints[1].deform && scope.joints[1].mapped);
        assert_eq!(report.mapping.target_deform_coverage, 1.);
        assert_eq!(report.motion_roots, ["Hips"]);
        assert!(
            motion
                .tracks
                .iter()
                .any(|track| { track.joint == root && track.property == Property::Translation })
        );
        loaded.document.append_animation(&motion, 0).unwrap();
        let output = temp.path().join("root-motion-readback.glb");
        loaded.document.save(&output).unwrap();
        let readback = charloom_io::load_gltf(&output).unwrap();
        let motion = &readback.asset.clips[0];
        let initial = motion.sample(&readback.asset.skeleton, 0.).unwrap();
        let final_pose = motion.sample(&readback.asset.skeleton, 1.).unwrap();
        assert!((v3(final_pose[root].translation) - Vector3::new(0.2, 0.1, 0.)).norm() < 1e-7);
        let before = readback.asset.skinned_positions(&initial).unwrap();
        let after = readback.asset.skinned_positions(&final_pose).unwrap();
        for (before, after) in before[0].iter().zip(&after[0]) {
            assert!((v3(*after) - v3(*before) - Vector3::new(0.2, 0.1, 0.)).norm() < 1e-7);
        }
        for (joint, pose) in readback.asset.skeleton.joints.iter().zip(&final_pose) {
            assert_eq!(pose.scale, joint.rest.scale);
        }
    }

    #[test]
    fn omitted_and_null_target_scope_keep_legacy_global_gate_and_motion() {
        let (source, target, clip) = scoped_body_fixture();
        let old: RetargetConfig = serde_json::from_str(r#"{"min_target_coverage":0.4}"#).unwrap();
        let explicit_none: RetargetConfig =
            serde_json::from_str(r#"{"min_target_coverage":0.4,"target_joint_scope":null}"#)
                .unwrap();
        assert!(old.target_joint_scope.is_none());
        let (old_motion, old_report) = retarget(&source, &clip, &target, &old).unwrap();
        let (new_motion, new_report) = retarget(&source, &clip, &target, &explicit_none).unwrap();
        assert!(new_report.target_joint_scope.is_none());
        assert_eq!(new_report.mapping.target_deform_coverage, 0.4);
        assert_eq!(
            serde_json::to_value(old_motion).unwrap(),
            serde_json::to_value(new_motion).unwrap()
        );
        assert_eq!(
            serde_json::to_value(old_report).unwrap(),
            serde_json::to_value(new_report).unwrap()
        );
        let strict: RetargetConfig = serde_json::from_str("{}").unwrap();
        assert!(retarget(&source, &clip, &target, &strict).is_err());
    }

    #[test]
    fn retarget_keeps_target_proportions_and_corrects_reference_axes() {
        let s = rig(false, 1.0);
        let t = rig(true, 2.0);
        let clip = MotionClip {
            name: "s".into(),
            morph_tracks: Vec::new(),
            tracks: vec![Track {
                joint: 0,
                property: Property::Rotation,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 1.0],
                values: vec![
                    [0.0, 0.0, 0.0, 1.0],
                    a4(UnitQuaternion::from_axis_angle(
                        &nalgebra::Vector3::x_axis(),
                        0.6,
                    )),
                ],
            }],
        };
        let (out, report) = retarget(&s, &clip, &t, &RetargetConfig::default()).unwrap();
        let pose = out.sample(&t, 1.0).unwrap();
        assert_eq!(pose[1].translation, [0.0, -2.0, 0.0]);
        assert!(
            (world_rotation(&t.world_matrices(&pose).unwrap()[0])
                .unwrap()
                .angle_to(
                    &(UnitQuaternion::from_axis_angle(&nalgebra::Vector3::x_axis(), 0.6)
                        * unit_quat(t.joints[0].rest.rotation))
                ))
                < 1e-8
        );
        assert!((report.root_scale - 2.0).abs() < 1e-10);
        assert_eq!(report.max_local_bone_length_error, 0.0);
    }
    #[test]
    fn digit_aliases_match_across_exporters() {
        let mut s = rig(false, 1.0);
        let mut t = s.clone();
        s.joints[1].name = "HC_Left_little_2".into();
        t.joints[1].name = "mixamorig:LeftHandPinky2".into();
        assert_eq!(
            match_joints(&s, &t, &BTreeMap::new())
                .unwrap()
                .mappings
                .len(),
            2
        );
        s.joints[1].name = "pinky_02_l".into();
        assert_eq!(semantic_keys(&s)[1], semantic_keys(&t)[1]);
        s.joints[1].name = "middle_04_leaf_r".into();
        t.joints[1].name = "mixamorig:RightHandMiddle4".into();
        assert_eq!(semantic_keys(&s)[1], semantic_keys(&t)[1]);
    }
}
