//! Kinematic hand/prop attachments with explicit calibration and release policy.
use crate::{
    pipeline::select_clip,
    solve::{Effector, SolveConfig, solve_pose, world_rotation},
};
use anyhow::{Context, Result, ensure};
use charloom_core::*;
use nalgebra::Matrix4;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::HashSet, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PropAsset {
    pub path: PathBuf,
    /// Unique name of the new scene root, used by Attachment::node.
    pub name: String,
    #[serde(default)]
    pub transform: Transform,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ReleasePolicy {
    /// Keep the last held world transform after end; no gravity is simulated.
    #[default]
    HoldWorld,
    /// Resume the underlying animation after end; transitions are caller-authored.
    Source,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct GripFollower {
    pub joint: String,
    /// Restricts solving to this chain so it cannot move the primary hand.
    pub chain_root: String,
    /// Desired wrist/fingertip frame in prop-local coordinates. If omitted,
    /// capture it from the attachment's explicit reference pose.
    #[serde(default)]
    pub offset: Option<Transform>,
    #[serde(default = "yes")]
    pub orientation: bool,
    #[serde(default)]
    pub pole: Option<Vec3>,
}
fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Attachment {
    /// An existing rigid scene root or a weapon's existing driving bone.
    pub node: String,
    pub driver: String,
    /// Optional arm root: allow a bounded translation of the primary grip so
    /// both hands remain reachable after retargeting to different proportions.
    #[serde(default)]
    pub driver_chain_root: Option<String>,
    #[serde(default = "driver_limit")]
    pub max_driver_translation_m: f64,
    /// Optional common rigid-prop rotation budget during constrained two-hand IK.
    #[serde(default)]
    pub max_driver_rotation_rad: f64,
    /// Optional temporal bound on each solved arm joint's local rotation.
    #[serde(default)]
    pub max_joint_angular_speed_rad_s: Option<f64>,
    /// Squared meter-equivalent scale for orientation residuals during IK.
    #[serde(default = "orientation_weight")]
    pub orientation_weight: f64,
    pub start: f64,
    pub end: f64,
    /// Prop-root transform relative to the driving hand. If omitted, calibrate
    /// from reference_clip/reference_time, or from the default node pose.
    #[serde(default)]
    pub driver_offset: Option<Transform>,
    #[serde(default)]
    pub reference_clip: Option<String>,
    #[serde(default)]
    pub reference_time: f64,
    #[serde(default)]
    pub followers: Vec<GripFollower>,
    #[serde(default)]
    pub release: ReleasePolicy,
}
fn orientation_weight() -> f64 {
    1.0
}
fn driver_limit() -> f64 {
    0.25
}

pub(crate) fn joint_id(s: &Skeleton, name: &str) -> Result<usize> {
    let ids: Vec<_> = s
        .joints
        .iter()
        .enumerate()
        .filter(|(_, j)| j.name == name)
        .map(|(i, _)| i)
        .collect();
    ensure!(
        ids.len() == 1,
        "joint must have a unique exact name: {name}"
    );
    Ok(ids[0])
}

fn descendant(s: &Skeleton, mut node: usize, ancestor: usize) -> bool {
    loop {
        if node == ancestor {
            return true;
        }
        match s.joints[node].parent {
            Some(parent) => node = parent,
            None => return false,
        }
    }
}

fn decompose(matrix: &Matrix4<f64>) -> Result<Transform> {
    let rotation = world_rotation(matrix)?;
    let transform = Transform {
        translation: transform_point(matrix, [0.0; 3]),
        rotation: a4(rotation),
        scale: std::array::from_fn(|i| matrix.fixed_view::<3, 1>(0, i).norm()),
    };
    ensure!(
        transform.is_valid()
            && (transform.matrix() - matrix).norm() < 1e-6 * matrix.norm().max(1.0),
        "unsupported_attachment_affine_transform"
    );
    Ok(transform)
}

fn inverse(matrix: &Matrix4<f64>) -> Result<Matrix4<f64>> {
    matrix.try_inverse().context("singular_attachment_frame")
}

fn put_world(
    skeleton: &Skeleton,
    pose: &mut [Transform],
    node: usize,
    world: &Matrix4<f64>,
) -> Result<()> {
    let current = skeleton.world_matrices(pose)?;
    let local = match skeleton.joints[node].parent {
        Some(parent) => inverse(&current[parent])? * world,
        None => *world,
    };
    pose[node] = decompose(&local)?;
    Ok(())
}

struct Follower {
    joint: usize,
    root: usize,
    offset: Transform,
    orientation: bool,
    pole: Option<Vec3>,
}

pub(crate) struct PreparedAttachment {
    node: usize,
    driver: usize,
    driver_root: Option<usize>,
    offset: Transform,
    followers: Vec<Follower>,
    active: HashSet<usize>,
    spec: Attachment,
    held_world: Option<Matrix4<f64>>,
    previous_pose: Option<(f64, Vec<Transform>)>,
    source_driver_world: Option<Matrix4<f64>>,
}

impl PreparedAttachment {
    pub(crate) fn capture_source_driver(
        &mut self,
        skeleton: &Skeleton,
        pose: &[Transform],
    ) -> Result<()> {
        self.source_driver_world = Some(skeleton.world_matrices(pose)?[self.driver]);
        Ok(())
    }
    pub(crate) fn apply_all(
        attachments: &mut [Self],
        skeleton: &Skeleton,
        pose: &mut Vec<Transform>,
        time: f64,
        config: &SolveConfig,
    ) -> Result<Vec<Value>> {
        let mut reports = Vec::new();
        for index in 0..attachments.len() {
            // A later pickup supersedes an older world hold on the same object.
            if attachments[index + 1..]
                .iter()
                .any(|later| later.node == attachments[index].node && time >= later.spec.start)
            {
                continue;
            }
            if let Some(report) = attachments[index].apply(skeleton, pose, time, config)? {
                reports.push(report);
            }
        }
        Ok(reports)
    }

    pub(crate) fn prepare(asset: &Asset, specs: &[Attachment], duration: f64) -> Result<Vec<Self>> {
        ensure!(specs.len() <= 16, "at_most_16_attachments");
        let s = &asset.skeleton;
        let mut out = Vec::new();
        for spec in specs {
            ensure!(
                spec.start.is_finite()
                    && spec.end.is_finite()
                    && spec.start >= 0.0
                    && spec.end >= spec.start
                    && spec.end <= duration,
                "invalid_attachment_interval"
            );
            ensure!(spec.followers.len() <= 32, "at_most_32_grip_followers");
            ensure!(
                spec.reference_time.is_finite(),
                "invalid_attachment_reference_time"
            );
            let node = joint_id(s, &spec.node)?;
            let driver = joint_id(s, &spec.driver)?;
            let driver_root = spec
                .driver_chain_root
                .as_ref()
                .map(|name| joint_id(s, name))
                .transpose()?;
            ensure!(
                spec.max_driver_translation_m.is_finite() && spec.max_driver_translation_m >= 0.0,
                "invalid_driver_translation_limit"
            );
            ensure!(
                spec.max_driver_rotation_rad.is_finite()
                    && (0.0..=std::f64::consts::PI).contains(&spec.max_driver_rotation_rad),
                "invalid_driver_rotation_limit"
            );
            ensure!(
                spec.orientation_weight.is_finite() && spec.orientation_weight > 0.0,
                "invalid_grip_orientation_weight"
            );
            ensure!(
                spec.max_joint_angular_speed_rad_s
                    .is_none_or(|v| v.is_finite() && v > 0.0),
                "invalid_grip_joint_angular_speed"
            );
            if let Some(root) = driver_root {
                ensure!(
                    root != driver && descendant(s, driver, root),
                    "driver_chain_root_not_ancestor"
                );
            }
            ensure!(
                s.joints[node].matrix.is_none(),
                "attachment_node_requires_TRS"
            );
            ensure!(
                !descendant(s, driver, node),
                "attachment_dependency_cycle: object drives its primary hand"
            );
            let reference = if let Some(name) = &spec.reference_clip {
                let clip = select_clip(asset, name)?;
                let range = clip.time_range();
                ensure!(
                    spec.reference_time >= range[0] && spec.reference_time <= range[1],
                    "attachment_reference_time_outside_clip"
                );
                clip.sample(s, spec.reference_time)?
            } else {
                ensure!(
                    spec.reference_time == 0.0,
                    "reference_time_requires_reference_clip"
                );
                s.rest_pose()
            };
            let world = s.world_matrices(&reference)?;
            let offset = match spec.driver_offset {
                Some(offset) => offset,
                None => decompose(&(inverse(&world[driver])? * world[node]))?,
            };
            ensure!(offset.is_invertible(), "invalid_driver_offset");
            let mut followers = Vec::new();
            let mut active = HashSet::new();
            let mut used = HashSet::new();
            for f in &spec.followers {
                let joint = joint_id(s, &f.joint)?;
                let root = joint_id(s, &f.chain_root)?;
                ensure!(used.insert(joint), "duplicate_grip_follower");
                ensure!(descendant(s, joint, root), "grip_chain_root_not_ancestor");
                ensure!(
                    !descendant(s, driver, root)
                        && !descendant(s, node, root)
                        && !descendant(s, joint, node),
                    "grip_chain_moves_driver_or_prop"
                );
                if let Some(driver_root) = driver_root {
                    ensure!(
                        !descendant(s, root, driver_root) && !descendant(s, driver_root, root),
                        "primary_and_secondary_grip_chains_must_be_independent"
                    );
                }
                let offset = match f.offset {
                    Some(offset) => offset,
                    None => decompose(&(inverse(&world[node])? * world[joint]))?,
                };
                ensure!(
                    offset.is_invertible()
                        && f.pole.is_none_or(|p| p.iter().all(|x| x.is_finite())),
                    "invalid_grip_target"
                );
                let mut at = joint;
                loop {
                    active.insert(at);
                    if at == root {
                        break;
                    }
                    at = s.joints[at].parent.context("missing grip ancestor")?;
                }
                followers.push(Follower {
                    joint,
                    root,
                    offset,
                    orientation: f.orientation,
                    pole: f.pole,
                });
            }
            if let Some(root) = driver_root {
                let mut at = driver;
                loop {
                    active.insert(at);
                    if at == root {
                        break;
                    }
                    at = s.joints[at].parent.context("missing driver ancestor")?;
                }
            }
            out.push(Self {
                node,
                driver,
                driver_root,
                offset,
                followers,
                active,
                spec: spec.clone(),
                held_world: None,
                previous_pose: None,
                source_driver_world: None,
            });
        }
        out.sort_by(|a, b| a.spec.start.total_cmp(&b.spec.start));
        for (i, a) in out.iter().enumerate() {
            for b in out.iter().skip(i + 1) {
                if a.spec.end < b.spec.start || b.spec.end < a.spec.start {
                    continue;
                }
                ensure!(a.node != b.node, "overlapping_attachment_intervals");
                ensure!(
                    !descendant(s, a.driver, b.node)
                        && !descendant(s, b.driver, a.node)
                        && !descendant(s, a.node, b.node)
                        && !descendant(s, b.node, a.node),
                    "coupled_attachment_graph_unsupported"
                );
                for (x, y) in [(a, b), (b, a)] {
                    ensure!(
                        x.followers
                            .iter()
                            .all(|f| !descendant(s, y.driver, f.root)
                                && !descendant(s, y.node, f.root))
                            && x.active.is_disjoint(&y.active),
                        "coupled_grip_chains_unsupported"
                    );
                }
            }
        }
        Ok(out)
    }

    pub(crate) fn calibration(&self) -> Value {
        json!({"node":self.spec.node,"driver":self.spec.driver,"start":self.spec.start,"end":self.spec.end,
            "driver_chain_root":self.spec.driver_chain_root,"max_driver_translation_m":self.spec.max_driver_translation_m,
            "reference_clip":self.spec.reference_clip,"reference_time":self.spec.reference_time,"driver_offset":self.offset,"release":self.spec.release,
            "followers":self.followers.iter().map(|f| json!({"joint":f.joint,"chain_root":f.root,"offset":f.offset,"orientation":f.orientation})).collect::<Vec<_>>()})
    }

    pub(crate) fn apply(
        &mut self,
        skeleton: &Skeleton,
        pose: &mut Vec<Transform>,
        time: f64,
        config: &SolveConfig,
    ) -> Result<Option<Value>> {
        if time < self.spec.start {
            return Ok(None);
        }
        if time > self.spec.end {
            if self.spec.release == ReleasePolicy::HoldWorld
                && let Some(world) = self.held_world
            {
                put_world(skeleton, pose, self.node, &world)?;
            }
            return Ok(None);
        }
        let world = skeleton.world_matrices(pose)?;
        let source_driver = self
            .source_driver_world
            .filter(|_| self.driver_root.is_some())
            .unwrap_or(world[self.driver]);
        let original = source_driver * self.offset.matrix();
        let mut desired = original;
        let primary_frame = inverse(&self.offset.matrix())?;
        if let Some(root) = self.driver_root {
            let mut reaches = vec![(root, self.driver, primary_frame)];
            reaches.extend(
                self.followers
                    .iter()
                    .map(|f| (f.root, f.joint, f.offset.matrix())),
            );
             
            for _ in 0..64 {
                let mut corrected = false;
                for (root, joint, frame) in &reaches {
                    let mut at = *joint;
                    let mut lengths = Vec::new();
                    while at != *root {
                        let parent = skeleton.joints[at]
                            .parent
                            .context("invalid reachability chain")?;
                        lengths.push(
                            (v3(transform_point(&world[at], [0.0; 3]))
                                - v3(transform_point(&world[parent], [0.0; 3])))
                            .norm(),
                        );
                        at = parent;
                    }
                    let total = lengths.iter().sum::<f64>();
                    if total < 1e-8 {
                        continue;
                    }
                    let radius = (total - config.tolerance_m * 0.5).max(total * 0.99);
                    let minimum =
                        (2.0 * lengths.iter().copied().fold(0.0, f64::max) - total).max(0.0);
                    let delta = v3(transform_point(&(desired * frame), [0.0; 3]))
                        - v3(transform_point(&world[*root], [0.0; 3]));
                    let distance = delta.norm();
                    let adjusted = distance.clamp(minimum + 1e-7, radius.max(minimum + 1e-7));
                    if (adjusted - distance).abs() > 1e-7 {
                        let direction = if distance > 1e-10 {
                            delta / distance
                        } else {
                            nalgebra::Vector3::x()
                        };
                        for axis in 0..3 {
                            desired[(axis, 3)] += direction[axis] * (adjusted - distance);
                        }
                        corrected = true;
                    }
                }
                if !corrected {
                    break;
                }
            }
            let delta =
                v3(transform_point(&desired, [0.0; 3])) - v3(transform_point(&original, [0.0; 3]));
            if delta.norm() > self.spec.max_driver_translation_m {
                let bounded = delta.normalize() * self.spec.max_driver_translation_m;
                for axis in 0..3 {
                    desired[(axis, 3)] = original[(axis, 3)] + bounded[axis];
                }
            }
        }
        put_world(skeleton, pose, self.node, &desired)?;
        let mut effectors = self
            .followers
            .iter()
            .map(|f| {
                let target = desired * f.offset.matrix();
                Ok(Effector {
                    joint: f.joint,
                    position: transform_point(&target, [0.0; 3]),
                    position_weight: 1.0,
                    position_axes: [true; 3],
                    rotation: if f.orientation {
                        Some(a4(world_rotation(&target)?))
                    } else {
                        None
                    },
                    rotation_weight: self.spec.orientation_weight,
                    chain_root: Some(f.root),
                    pole: f.pole,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(root) = self.driver_root {
            let target = desired * primary_frame;
            effectors.push(Effector {
                joint: self.driver,
                position: transform_point(&target, [0.0; 3]),
                position_weight: 1.0,
                position_axes: [true; 3],
                rotation: Some(a4(world_rotation(&target)?)),
                rotation_weight: self.spec.orientation_weight,
                chain_root: Some(root),
                pole: None,
            });
        }
        if !effectors.is_empty() {
            let mut config = config.clone();
            config.root_translation = false;
            config
                .limits
                .retain(|limit| self.active.contains(&limit.joint));
            if let Some(speed) = self.spec.max_joint_angular_speed_rad_s
                && let Some((last, previous)) = &self.previous_pose
                && time > *last
                && time - last <= 0.1
            {
                for &joint in &self.active {
                    config.limits.push(crate::solve::JointLimit {
                        joint,
                        max_angle_rad: (speed * (time - last)).min(std::f64::consts::PI),
                        reference_rotation: Some(previous[joint].rotation),
                    });
                }
            }
            let score = |r: &crate::solve::SolveResult| {
                r.residuals
                    .iter()
                    .map(|r| {
                        r.position_error_m.powi(2)
                            + r.rotation_error_rad.unwrap_or(0.0).powi(2)
                                * self.spec.orientation_weight
                    })
                    .sum::<f64>()
                    + r.collision_residuals
                        .iter()
                        .map(|r| {
                            r.penetration_m.powi(2)
                                * config.collision.as_ref().map_or(0.0, |c| c.weight)
                        })
                        .sum::<f64>()
            };
            let relative = (!config.articulations.is_empty()
                && self.driver_root.is_some()
                && !self.followers.is_empty())
            .then(|| crate::relative_grip::RelativeGrip {
                driver: self.driver,
                source: source_driver,
                followers: self
                    .followers
                    .iter()
                    .map(|f| {
                        (
                            f.joint,
                            self.offset.matrix() * f.offset.matrix(),
                            f.orientation,
                        )
                    })
                    .collect(),
                active: {
                    let mut a: Vec<_> = self.active.iter().copied().collect();
                    a.sort_unstable();
                    a
                },
                max_translation: self.spec.max_driver_translation_m,
                max_rotation: self.spec.max_driver_rotation_rad,
                orientation_weight: self.spec.orientation_weight,
                posture: pose.clone(),
            });
            let mut relative_solved = false;
            if let Some(task) = &relative {
                let mut start = pose.clone();
                if let Some((last, previous)) = &self.previous_pose
                    && time > *last
                    && time - last <= 0.1
                {
                    for &j in &self.active {
                        start[j].rotation = previous[j].rotation;
                    }
                }
                let result = task.solve(skeleton, &start, &config)?;
                if result.converged {
                    *pose = result.pose;
                    relative_solved = true;
                }
            }
            for attempt in 0..if relative_solved { 0 } else { 32 } {
                let mut best = solve_pose(skeleton, pose, &effectors, &config)?;
                if attempt == 0
                    && !config.articulations.is_empty()
                    && let Some((last, previous)) = &self.previous_pose
                    && time > *last
                    && time - last <= 0.1
                {
                    let mut warm = pose.clone();
                    for &j in &self.active {
                        warm[j].rotation = previous[j].rotation;
                    }
                    let candidate = solve_pose(skeleton, &warm, &effectors, &config)?;
                    if candidate.status == "converged"
                        || (best.status != "converged" && score(&candidate) < score(&best))
                    {
                        best = candidate;
                    }
                }
                if attempt == 0 && best.status != "converged" && !config.articulations.is_empty() {
                    // Elbow position has a circular family of solutions. Try
                    // bounded deterministic pole alternatives when the local
                    // solution is trapped against the body or a joint limit.
                    for effector in &effectors {
                        let Some(root) = effector.chain_root else {
                            continue;
                        };
                        let Some(middle) = skeleton.joints[effector.joint].parent else {
                            continue;
                        };
                        if skeleton.joints[middle].parent != Some(root) {
                            continue;
                        }
                        let seed = best.pose.clone();
                        let world = skeleton.world_matrices(&seed)?;
                        let origin = v3(transform_point(&world[root], [0.0; 3]));
                        let Some(axis) = (v3(effector.position) - origin).try_normalize(1e-8)
                        else {
                            continue;
                        };
                        let elbow = v3(transform_point(&world[middle], [0.0; 3])) - origin;
                        let perpendicular = elbow - axis * elbow.dot(&axis);
                        let perpendicular =
                            perpendicular.try_normalize(1e-8).unwrap_or_else(|| {
                                axis.cross(&nalgebra::Vector3::x())
                                    .try_normalize(1e-8)
                                    .unwrap_or(nalgebra::Vector3::z())
                            });
                        for angle in [0.6, -0.6, 1.2, -1.2, 1.8, -1.8, std::f64::consts::PI] {
                            let rotation = nalgebra::UnitQuaternion::from_axis_angle(
                                &nalgebra::Unit::new_normalize(axis),
                                angle,
                            );
                            let pole = a3(origin + rotation * perpendicular);
                            let mut candidate = seed.clone();
                            crate::solve::two_bone(
                                skeleton,
                                &mut candidate,
                                root,
                                middle,
                                effector.joint,
                                effector.position,
                                Some(pole),
                            )?;
                            if let Some(q) = effector.rotation {
                                let world = skeleton.world_matrices(&candidate)?;
                                candidate[effector.joint].rotation =
                                    a4(world_rotation(&world[middle])?.inverse() * unit_quat(q));
                            }
                            let mut twists = crate::articulation::PreparedTwist::prepare_named(
                                skeleton,
                                &[skeleton.joints[effector.joint].name.clone()],
                            )?;
                            for twist in &mut twists {
                                twist.set_reference(&config.articulations);
                                twist.apply(skeleton, &mut candidate)?;
                            }
                            for limit in &config.articulations {
                                limit.distribute_bend_plane(skeleton, &mut candidate)?;
                            }
                            let result = solve_pose(skeleton, &candidate, &effectors, &config)?;
                            if score(&result) < score(&best) {
                                best = result;
                            }
                            if best.status == "converged" {
                                break;
                            }
                        }
                        if best.status == "converged" {
                            break;
                        }
                    }
                }
                *pose = best.pose;
                if config.collision.is_none() || self.driver_root.is_none() {
                    break;
                }
                let world = skeleton.world_matrices(pose)?;
                let residual = crate::solve::residuals(skeleton, pose, &effectors)?;
                if residual.iter().all(|r| {
                    r.position_error_m <= config.tolerance_m
                        && r.rotation_error_rad
                            .is_none_or(|a| a <= config.rotation_tolerance_rad)
                }) {
                    break;
                }
                // Fit one rigid prop to the feasible hand frames. Rotation
                // and translation are bounded relative to this frame's source;
                // neither hand gets an independent sliding grip target.
                let old_rotation = world_rotation(&desired)?;
                let frames: Vec<_> = self
                    .followers
                    .iter()
                    .map(|f| f.offset.matrix())
                    .chain(std::iter::once(primary_frame))
                    .collect();
                let prop_origin = v3(transform_point(&desired, [0.0; 3]));
                let local_centroid = effectors
                    .iter()
                    .map(|e| v3(e.position) - prop_origin)
                    .sum::<nalgebra::Vector3<f64>>()
                    / effectors.len() as f64;
                let mean_error = effectors
                    .iter()
                    .map(|e| v3(transform_point(&world[e.joint], [0.0; 3])) - v3(e.position))
                    .sum::<nalgebra::Vector3<f64>>()
                    / effectors.len() as f64;
                let mut normal = nalgebra::Matrix3::identity() * 1e-8;
                let mut rhs = nalgebra::Vector3::zeros();
                for (e, frame) in effectors.iter().zip(&frames) {
                    let p = v3(e.position) - prop_origin - local_centroid;
                    let residual = v3(transform_point(&world[e.joint], [0.0; 3]))
                        - v3(e.position)
                        - mean_error;
                    normal += nalgebra::Matrix3::identity() * p.norm_squared() - p * p.transpose();
                    rhs += p.cross(&residual);
                    if e.rotation.is_some() {
                        let proposed =
                            world_rotation(&world[e.joint])? * world_rotation(frame)?.inverse();
                        normal += nalgebra::Matrix3::identity() * self.spec.orientation_weight;
                        rhs += (proposed * old_rotation.inverse()).scaled_axis()
                            * self.spec.orientation_weight;
                    }
                }
                let mut turn = normal
                    .cholesky()
                    .context("singular_rigid_grip_fit")?
                    .solve(&rhs);
                if turn.norm() > 0.12 {
                    turn = turn.normalize() * 0.12;
                }
                let mut rotation = nalgebra::UnitQuaternion::from_scaled_axis(turn) * old_rotation;
                let original_rotation = world_rotation(&original)?;
                let angle = original_rotation.angle_to(&rotation);
                if angle > self.spec.max_driver_rotation_rad {
                    rotation = original_rotation
                        .slerp(&rotation, self.spec.max_driver_rotation_rad / angle);
                }
                let current = v3(transform_point(&desired, [0.0; 3]));
                let origin = v3(transform_point(&original, [0.0; 3]));
                let scale = std::array::from_fn(|i| desired.fixed_view::<3, 1>(0, i).norm());
                let orientation = Transform {
                    rotation: a4(rotation),
                    scale,
                    ..Default::default()
                }
                .matrix();
                let fit = effectors.iter().zip(&frames).fold(
                    nalgebra::Vector3::zeros(),
                    |sum, (e, frame)| {
                        sum + v3(transform_point(&world[e.joint], [0.0; 3]))
                            - v3(transform_point(&(orientation * frame), [0.0; 3]))
                    },
                ) / effectors.len() as f64;
                let mut delta = fit - origin;
                if delta.norm() > self.spec.max_driver_translation_m {
                    delta = delta.normalize() * self.spec.max_driver_translation_m;
                }
                let position = origin + delta;
                if (position - current).norm() < 1e-7 && old_rotation.angle_to(&rotation) < 1e-7 {
                    break;
                }
                desired = Transform {
                    translation: a3(position),
                    rotation: a4(rotation),
                    scale,
                }
                .matrix();
                for (e, frame) in effectors.iter_mut().zip(&frames) {
                    let target = desired * frame;
                    e.position = transform_point(&target, [0.0; 3]);
                    if e.rotation.is_some() {
                        e.rotation = Some(a4(world_rotation(&target)?));
                    }
                }
            }
            if !relative_solved && let Some(task) = &relative {
                let result = task.solve(skeleton, pose, &config)?;
                if result.converged {
                    *pose = result.pose;
                    relative_solved = true;
                }
            }
            if relative_solved {
                let world = skeleton.world_matrices(pose)?;
                if let Some(primary) = effectors.last_mut() {
                    primary.position = transform_point(&world[self.driver], [0.0; 3]);
                    primary.rotation = Some(a4(world_rotation(&world[self.driver])?));
                }
            }
        }
        let solved = skeleton.world_matrices(pose)?;
        let final_prop = solved[self.driver] * self.offset.matrix();
        put_world(skeleton, pose, self.node, &final_prop)?;
        for (effector, follower) in effectors.iter_mut().zip(&self.followers) {
            let target = final_prop * follower.offset.matrix();
            effector.position = transform_point(&target, [0.0; 3]);
            if follower.orientation {
                effector.rotation = Some(a4(world_rotation(&target)?));
            }
        }
        let actual = skeleton.world_matrices(pose)?;
        let residuals = crate::solve::residuals(skeleton, pose, &effectors)?;
        let collision_residuals = config
            .collision
            .as_ref()
            .map(|c| c.residuals(skeleton, pose))
            .transpose()?
            .unwrap_or_default();
        let angular_step = self
            .previous_pose
            .as_ref()
            .filter(|(last, _)| time > *last && time - last <= 0.1)
            .map(|(_, previous)| {
                self.active
                    .iter()
                    .map(|&j| {
                        unit_quat(previous[j].rotation).angle_to(&unit_quat(pose[j].rotation))
                    })
                    .fold(0.0, f64::max)
            });
        let angular_limit = self
            .spec
            .max_joint_angular_speed_rad_s
            .zip(self.previous_pose.as_ref())
            .filter(|(_, (last, _))| time > *last && time - last <= 0.1)
            .map(|(speed, (last, _))| speed * (time - last));
        let temporal_ok = angular_step
            .zip(angular_limit)
            .is_none_or(|(a, b)| a <= b + 1e-5);
        let converged = temporal_ok
            && config.collision.as_ref().is_none_or(|c| {
                collision_residuals
                    .iter()
                    .all(|r| r.penetration_m <= c.tolerance_m)
            })
            && residuals.iter().all(|r| {
                r.position_error_m <= config.tolerance_m
                    && r.rotation_error_rad
                        .is_none_or(|a| a <= config.rotation_tolerance_rad)
            });
        self.held_world = Some(actual[self.node]);
        self.previous_pose = Some((time, pose.clone()));
        Ok(Some(
            json!({"node":self.spec.node,"time_s":time,"status":if converged {"converged"} else {"not_converged"},
            "driver_matrix_error":(actual[self.node] - final_prop).norm(),
            "maximum_joint_rotation_step_rad":angular_step,"joint_rotation_step_limit_rad":angular_limit,
            "driver_motion_correction_m":(v3(transform_point(&actual[self.driver],[0.0;3]))-v3(transform_point(&source_driver,[0.0;3]))).norm(),
            "followers":&residuals[..self.followers.len()],"primary_solve":if self.driver_root.is_some(){residuals.last()}else{None},"collision_residuals":collision_residuals}),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn both_hands_reach_the_same_rigid_prop_with_bounded_primary_adjustment() {
        let joint = |name: &str, parent, translation| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(translation),
            matrix: None,
            deform: true,
        };
        let asset = Asset {
            name: "two_arms".into(),
            source_hash: String::new(),
            meshes: vec![],
            skins: vec![],
            clips: vec![],
            issues: vec![],
            skeleton: Skeleton {
                joints: vec![
                    joint("body", None, [0.0; 3]),
                    joint("left_root", Some(0), [-0.3, 1.0, 0.0]),
                    joint("left_elbow", Some(1), [0.4, 0.0, 0.0]),
                    joint("left_hand", Some(2), [0.4, 0.0, 0.0]),
                    joint("right_root", Some(0), [0.3, 1.0, 0.0]),
                    joint("right_elbow", Some(4), [0.4, 0.0, 0.0]),
                    joint("right_hand", Some(5), [0.4, 0.0, 0.0]),
                    joint("object", None, [1.1, 1.0, 0.0]),
                ],
            },
        };
        for (limit, expected, capture) in [
            (0.3, "converged", false),
            (0.02, "not_converged", false),
            (0.3, "converged", true),
        ] {
            let spec=serde_json::from_value(json!({"node":"object","driver":"right_hand","driver_chain_root":"right_root","max_driver_translation_m":limit,"start":0,"end":1,
                "followers":[{"joint":"left_hand","chain_root":"left_root","offset":{"translation":[-0.4,0,0]},"orientation":false}]})).unwrap();
            let mut attachments = PreparedAttachment::prepare(&asset, &[spec], 1.0).unwrap();
            let mut pose = asset.skeleton.rest_pose();
            if capture {
                attachments[0]
                    .capture_source_driver(&asset.skeleton, &pose)
                    .unwrap();
                pose[4].rotation = a4(nalgebra::UnitQuaternion::from_axis_angle(
                    &nalgebra::Vector3::z_axis(),
                    0.5,
                ));
                pose[1].rotation = a4(nalgebra::UnitQuaternion::from_axis_angle(
                    &nalgebra::Vector3::z_axis(),
                    -0.3,
                ));
            }
            let report = attachments[0]
                .apply(&asset.skeleton, &mut pose, 0.0, &SolveConfig::default())
                .unwrap()
                .unwrap();
            assert_eq!(report["status"], expected);
            assert!(report["driver_motion_correction_m"].as_f64().unwrap() <= limit + 0.001);
            for (i, p) in pose.iter().enumerate().take(7) {
                assert_eq!(p.translation, asset.skeleton.joints[i].rest.translation);
            }
        }
    }
    fn fixture() -> Asset {
        let joint = |name: &str, parent, translation| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(translation),
            matrix: None,
            deform: true,
        };
        Asset {
            name: "grip_test".into(),
            source_hash: String::new(),
            meshes: vec![],
            skins: vec![],
            clips: vec![],
            issues: vec![],
            skeleton: Skeleton {
                joints: vec![
                    joint("body", None, [0.0; 3]),
                    joint("right", Some(0), [0.8, 0.0, 0.0]),
                    joint("prop", Some(0), [0.8, 0.0, 0.0]),
                    joint("left_arm", Some(0), [-0.8, 0.0, 0.0]),
                    joint("left_elbow", Some(3), [0.0, 0.5, 0.0]),
                    joint("left_hand", Some(4), [0.0, 0.5, 0.0]),
                ],
            },
        }
    }
    #[test]
    fn secondary_hand_reaches_prop_and_release_holds_world() {
        let asset = fixture();
        let spec:Attachment=serde_json::from_value(json!({"node":"prop","driver":"right","start":0.0,"end":1.0,
            "followers":[{"joint":"left_hand","chain_root":"left_arm","offset":{"translation":[-1.4,0.7,0.0]},"orientation":false}]})).unwrap();
        let mut attached = PreparedAttachment::prepare(&asset, &[spec], 2.0).unwrap();
        let s = &asset.skeleton;
        let mut pose = s.rest_pose();
        pose[1].translation[2] = 0.1;
        let before = s.world_matrices(&pose).unwrap();
        let report = attached[0]
            .apply(s, &mut pose, 1.0, &SolveConfig::default())
            .unwrap()
            .unwrap();
        let after = s.world_matrices(&pose).unwrap();
        assert_eq!(report["status"], "converged");
        assert!((before[1] - after[1]).norm() < 1e-12);
        assert!((transform_point(&after[2], [0.0; 3])[2] - 0.1).abs() < 1e-12);
        let held = after[2];
        pose = s.rest_pose();
        pose[0].translation = [2.0, 0.0, 0.0];
        attached[0]
            .apply(s, &mut pose, 1.5, &SolveConfig::default())
            .unwrap();
        assert!((s.world_matrices(&pose).unwrap()[2] - held).norm() < 1e-10);
    }
    #[test]
    fn temporal_joint_bound_reports_unreachable_motion_without_a_branch_jump() {
        let asset = fixture();
        let spec:Attachment=serde_json::from_value(json!({"node":"prop","driver":"right","start":0.0,"end":1.0,"max_joint_angular_speed_rad_s":1.0,
            "followers":[{"joint":"left_hand","chain_root":"left_arm","offset":{"translation":[-1.4,0.7,0.0]},"orientation":false}]})).unwrap();
        let mut attached = PreparedAttachment::prepare(&asset, &[spec], 1.0).unwrap();
        let mut pose = asset.skeleton.rest_pose();
        let first = attached[0]
            .apply(&asset.skeleton, &mut pose, 0.0, &SolveConfig::default())
            .unwrap()
            .unwrap();
        assert_eq!(first["status"], "converged");
        let previous = pose.clone();
        pose[1].translation[2] += 0.2;
        let second = attached[0]
            .apply(&asset.skeleton, &mut pose, 0.01, &SolveConfig::default())
            .unwrap()
            .unwrap();
        assert_eq!(second["status"], "not_converged");
        for j in [3, 4, 5] {
            assert!(
                unit_quat(pose[j].rotation).angle_to(&unit_quat(previous[j].rotation)) <= 0.010001
            );
            assert_eq!(pose[j].translation, previous[j].translation);
        }
    }
    #[test]
    fn rejects_hand_prop_cycles_and_coupled_chains() {
        let asset = fixture();
        for spec in [
            json!({"node":"body","driver":"right","start":0,"end":1}),
            json!({"node":"prop","driver":"right","start":0,"end":1,"followers":[{"joint":"left_hand","chain_root":"body"}]}),
        ] {
            let spec = serde_json::from_value(spec).unwrap();
            assert!(PreparedAttachment::prepare(&asset, &[spec], 1.0).is_err());
        }
    }
    #[test]
    fn unreachable_grip_is_reported_without_moving_primary_hand() {
        let asset = fixture();
        let spec=serde_json::from_value(json!({"node":"prop","driver":"right","start":0,"end":1,
            "followers":[{"joint":"left_hand","chain_root":"left_arm","offset":{"translation":[10,0,0]},"orientation":false}]})).unwrap();
        let mut attached = PreparedAttachment::prepare(&asset, &[spec], 1.0).unwrap();
        let mut pose = asset.skeleton.rest_pose();
        let report = attached[0]
            .apply(&asset.skeleton, &mut pose, 0.0, &SolveConfig::default())
            .unwrap()
            .unwrap();
        assert_eq!(report["status"], "not_converged");
        assert_eq!(pose[1], asset.skeleton.joints[1].rest);
    }
}
