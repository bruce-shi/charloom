//! Rotation distribution across a direct parent/child joint pair. This moves
//! axial roll to the parent without changing the child's world transform.
use anyhow::{Context, Result, ensure};
use charloom_core::*;
use nalgebra::{Quaternion, Unit, UnitQuaternion, Vector3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A characterized joint frame. The hinge axis is in the parent frame;
/// twist is about the outgoing bone in the joint frame.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArticulationLimit {
    pub joint: usize,
    pub reference_rotation: Quat,
    pub twist_axis: Vec3,
    pub twist_range_rad: [f64; 2],
    pub max_swing_rad: f64,
    #[serde(default)]
    pub hinge: Option<HingeLimit>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HingeLimit {
    pub axis_parent: Vec3,
    pub range_rad: [f64; 2],
}

fn signed_twist(q: UnitQuaternion<f64>, axis: Vector3<f64>) -> f64 {
    let mut angle = 2.0 * q.imag().dot(&axis).atan2(q.w);
    while angle > std::f64::consts::PI {
        angle -= std::f64::consts::TAU;
    }
    while angle < -std::f64::consts::PI {
        angle += std::f64::consts::TAU;
    }
    angle
}

impl ArticulationLimit {
    /// Re-express an elbow's bend plane as shoulder roll. A rotation about
    /// the upper-arm bone fixes the elbow position, and the inverse child
    /// rotation preserves the forearm, hand and every descendant in world.
    pub fn distribute_bend_plane(&self, skeleton: &Skeleton, pose: &mut [Transform]) -> Result<()> {
        let Some(hinge) = &self.hinge else {
            return Ok(());
        };
        let child = self.joint;
        let parent = skeleton.joints[child]
            .parent
            .context("hinge_requires_parent")?;
        let incoming = v3(pose[child].translation)
            .try_normalize(1e-8)
            .context("zero_hinge_parent_bone")?;
        let reference = unit_quat(self.reference_rotation) * v3(self.twist_axis).normalize();
        ensure!(
            incoming.dot(&reference) > 1.0 - 1e-6,
            "hinge_distribution_requires_straight_reference"
        );
        let scale = pose[parent].scale;
        ensure!(
            (scale[0] - scale[1]).abs() < 1e-8 && (scale[0] - scale[2]).abs() < 1e-8,
            "hinge_distribution_requires_uniform_parent_scale"
        );
        let q = unit_quat(pose[child].rotation);
        let direction = q * v3(self.twist_axis).normalize();
        let Some(bend) = (direction - incoming * incoming.dot(&direction)).try_normalize(1e-8)
        else {
            return Ok(());
        };
        let desired = v3(hinge.axis_parent).normalize().cross(&incoming);
        let angle = incoming
            .dot(&desired.cross(&bend))
            .atan2(desired.dot(&bend));
        let roll = UnitQuaternion::from_axis_angle(&Unit::new_normalize(incoming), angle);
        pose[parent].rotation = a4(unit_quat(pose[parent].rotation) * roll);
        pose[child].rotation = a4(roll.inverse() * q);
        Ok(())
    }

    pub fn validate(&self, skeleton: &Skeleton) -> Result<()> {
        ensure!(
            self.joint < skeleton.joints.len(),
            "articulation_joint_out_of_range"
        );
        ensure!(
            self.reference_rotation
                .iter()
                .chain(&self.twist_axis)
                .all(|v| v.is_finite())
                && v3(self.twist_axis).norm() > 1e-8
                && self.reference_rotation.iter().map(|x| x * x).sum::<f64>() > 1e-8,
            "invalid_articulation_frame"
        );
        let valid_range = |r: [f64; 2]| {
            r.iter()
                .all(|v| v.is_finite() && v.abs() <= std::f64::consts::PI)
                && r[0] <= r[1]
        };
        ensure!(
            valid_range(self.twist_range_rad)
                && self.max_swing_rad.is_finite()
                && (0.0..=std::f64::consts::PI).contains(&self.max_swing_rad),
            "invalid_articulation_range"
        );
        if let Some(h) = &self.hinge {
            ensure!(
                valid_range(h.range_rad)
                    && h.axis_parent.iter().all(|v| v.is_finite())
                    && v3(h.axis_parent).norm() > 1e-8
                    && v3(h.axis_parent)
                        .normalize()
                        .dot(
                            &(unit_quat(self.reference_rotation) * v3(self.twist_axis).normalize())
                        )
                        .abs()
                        < 1e-5,
                "hinge_axis_must_be_perpendicular_to_reference_bone"
            );
        }
        Ok(())
    }

    pub fn project_rotation(&self, q: UnitQuaternion<f64>) -> UnitQuaternion<f64> {
        let reference = unit_quat(self.reference_rotation);
        let axis = v3(self.twist_axis).normalize();
        let swing = if let Some(h) = &self.hinge {
            let normal = v3(h.axis_parent).normalize();
            let from = reference * axis;
            let to = q * axis;
            let theta = normal
                .dot(&from.cross(&to))
                .atan2(from.dot(&to))
                .clamp(h.range_rad[0], h.range_rad[1]);
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(normal), theta) * reference
        } else {
            let delta = reference.inverse() * q;
            let angle = signed_twist(delta, axis);
            let twist = UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), angle);
            let s = delta * twist.inverse();
            let s = if s.angle() > self.max_swing_rad {
                UnitQuaternion::identity().slerp(&s, self.max_swing_rad / s.angle())
            } else {
                s
            };
            reference * s
        };
        let twist = signed_twist(swing.inverse() * q, axis)
            .clamp(self.twist_range_rad[0], self.twist_range_rad[1]);
        swing * UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), twist)
    }

    pub fn local_axes(&self, q: UnitQuaternion<f64>) -> [Vector3<f64>; 3] {
        if let Some(h) = &self.hinge {
            [
                q.inverse() * v3(h.axis_parent).normalize(),
                v3(self.twist_axis).normalize(),
                Vector3::zeros(),
            ]
        } else {
            [Vector3::x(), Vector3::y(), Vector3::z()]
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ArmArticulation {
    pub upper_arm: String,
    pub forearm: String,
    pub wrist: String,
    /// Index, middle and little finger bases define the palm frame.
    pub palm_landmarks: [String; 3],
    pub neutral_palm_hinge_sign: f64,
    pub max_flexion_rad: f64,
    pub forearm_twist_range_rad: [f64; 2],
    pub wrist_swing_limit_rad: f64,
    pub wrist_twist_range_rad: [f64; 2],
}

impl ArmArticulation {
    pub fn prepare(&self, skeleton: &Skeleton) -> Result<Vec<ArticulationLimit>> {
        let named = |name: &str| -> Result<usize> {
            let found: Vec<_> = skeleton
                .joints
                .iter()
                .enumerate()
                .filter(|(_, j)| j.name == name)
                .map(|(i, _)| i)
                .collect();
            ensure!(
                found.len() == 1,
                "articulation_joint_must_be_unique: {name}"
            );
            Ok(found[0])
        };
        let upper = named(&self.upper_arm)?;
        let forearm = named(&self.forearm)?;
        let wrist = named(&self.wrist)?;
        ensure!(
            skeleton.joints[forearm].parent == Some(upper)
                && skeleton.joints[wrist].parent == Some(forearm),
            "articulation_requires_direct_arm_chain"
        );
        let incoming = v3(skeleton.joints[forearm].rest.translation)
            .try_normalize(1e-8)
            .context("zero_upper_arm_length")?;
        let axis = v3(skeleton.joints[wrist].rest.translation)
            .try_normalize(1e-8)
            .context("zero_forearm_length")?;
        let rest = unit_quat(skeleton.joints[forearm].rest.rotation);
        let outgoing = rest * axis;
        let hinge = incoming
            .cross(&outgoing)
            .try_normalize(1e-5)
            .context("straight_reference_requires_explicit_hinge_frame")?;
        let straighten = UnitQuaternion::rotation_between(&outgoing, &incoming)
            .context("antiparallel_arm_reference")?;
        let mut neutral = straighten * rest;
        let landmarks = self
            .palm_landmarks
            .iter()
            .map(|n| named(n))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            landmarks
                .iter()
                .all(|&j| skeleton.joints[j].parent == Some(wrist)),
            "palm_landmarks_must_be_direct_wrist_children"
        );
        ensure!(
            (self.neutral_palm_hinge_sign.abs() - 1.0).abs() < 1e-8,
            "palm_hinge_sign_must_be_unit"
        );
        let wrist_axis = v3(skeleton.joints[landmarks[1]].rest.translation).normalize();
        let across = v3(skeleton.joints[landmarks[0]].rest.translation)
            - v3(skeleton.joints[landmarks[2]].rest.translation);
        let palm_normal = across
            .cross(&wrist_axis)
            .try_normalize(1e-8)
            .context("degenerate_palm_frame")?;
        let wrist_rest = unit_quat(skeleton.joints[wrist].rest.rotation);
        let wrist_neutral = UnitQuaternion::rotation_between(&(wrist_rest * wrist_axis), &axis)
            .context("antiparallel_palm_reference")?
            * wrist_rest;
        let normal = neutral * wrist_neutral * palm_normal;
        let desired = hinge * self.neutral_palm_hinge_sign;
        let roll = incoming
            .dot(&normal.cross(&desired))
            .atan2(normal.dot(&desired));
        neutral = UnitQuaternion::from_axis_angle(&Unit::new_normalize(incoming), roll) * neutral;
        let limits = vec![
            ArticulationLimit {
                joint: forearm,
                reference_rotation: a4(neutral),
                twist_axis: a3(axis),
                twist_range_rad: self.forearm_twist_range_rad,
                max_swing_rad: self.max_flexion_rad,
                hinge: Some(HingeLimit {
                    axis_parent: a3(hinge),
                    range_rad: [0.0, self.max_flexion_rad],
                }),
            },
            ArticulationLimit {
                joint: wrist,
                reference_rotation: a4(wrist_neutral),
                twist_axis: a3(wrist_axis),
                twist_range_rad: self.wrist_twist_range_rad,
                max_swing_rad: self.wrist_swing_limit_rad,
                hinge: None,
            },
        ];
        for l in &limits {
            l.validate(skeleton)?;
        }
        Ok(limits)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct TwistDistributionReport {
    pub parent: String,
    pub child: String,
    pub transferred_rad: f64,
    pub singular_swing: bool,
}

pub struct PreparedTwist {
    child: usize,
    parent: usize,
    reference: UnitQuaternion<f64>,
}

impl PreparedTwist {
    pub fn set_reference(&mut self, limits: &[ArticulationLimit]) {
        if let Some(limit) = limits
            .iter()
            .find(|l| l.joint == self.child && l.hinge.is_none())
        {
            self.reference = unit_quat(limit.reference_rotation);
        }
    }
    pub fn prepare(skeleton: &Skeleton, child: usize) -> Result<Self> {
        ensure!(child < skeleton.joints.len(), "twist_child_out_of_bounds");
        let parent = skeleton.joints[child]
            .parent
            .context("twist_child_requires_parent")?;
        ensure!(
            skeleton.joints[child].matrix.is_none() && skeleton.joints[parent].matrix.is_none(),
            "twist_distribution_requires_TRS"
        );
        Ok(Self {
            child,
            parent,
            reference: unit_quat(skeleton.joints[child].rest.rotation),
        })
    }

    pub fn prepare_named(skeleton: &Skeleton, names: &[String]) -> Result<Vec<Self>> {
        ensure!(names.len() <= 64, "too_many_twist_distributions");
        let mut ids = std::collections::HashSet::new();
        names
            .iter()
            .map(|name| {
                let matches: Vec<_> = skeleton
                    .joints
                    .iter()
                    .enumerate()
                    .filter(|(_, joint)| joint.name == *name)
                    .map(|(i, _)| i)
                    .collect();
                ensure!(
                    matches.len() == 1 && ids.insert(matches[0]),
                    "twist_joint_must_be_unique: {name}"
                );
                Self::prepare(skeleton, matches[0])
            })
            .collect()
    }

    pub fn apply(
        &self,
        skeleton: &Skeleton,
        pose: &mut [Transform],
    ) -> Result<TwistDistributionReport> {
        self.apply_fraction(skeleton, pose, 1.0)
    }

         pub fn apply_fraction(
        &self,
        skeleton: &Skeleton,
        pose: &mut [Transform],
        fraction: f64,
    ) -> Result<TwistDistributionReport> {
        ensure!(
            fraction.is_finite() && (0.0..=1.0).contains(&fraction),
            "invalid_twist_distribution_fraction"
        );
        ensure!(
            pose.len() == skeleton.joints.len(),
            "twist_pose_size_mismatch"
        );
        let scale = pose[self.parent].scale;
        ensure!(
            (scale[0] - scale[1]).abs() < 1e-7
                && (scale[0] - scale[2]).abs() < 1e-7
                && scale[0] > 0.0,
            "twist_distribution_requires_uniform_positive_parent_scale"
        );
        let axis = v3(pose[self.child].translation)
            .try_normalize(1e-9)
            .context("twist_axis_has_zero_length")?;
        let delta = unit_quat(pose[self.child].rotation) * self.reference.inverse();
        let axial = delta.imag().dot(&axis);
        let raw = Quaternion::from_parts(delta.w, axis * axial);
        let singular = raw.norm_squared() < 1e-12;
        let transferred = if singular || fraction == 0.0 {
            0.0
        } else {
            let twist = UnitQuaternion::new_normalize(raw);
            // Keep the original full-transfer path, including quaternion
            // sign, when called through apply().
            let twist = if fraction == 1.0 {
                twist
            } else {
                UnitQuaternion::identity().slerp(&twist, fraction)
            };
            // R_parent T · T^-1 R_child = R_parent R_child, and T fixes
            // the child translation because its axis is that same vector.
            pose[self.parent].rotation = a4(unit_quat(pose[self.parent].rotation) * twist);
            pose[self.child].rotation = a4(twist.inverse() * unit_quat(pose[self.child].rotation));
            twist.angle()
        };
        Ok(TwistDistributionReport {
            parent: skeleton.joints[self.parent].name.clone(),
            child: skeleton.joints[self.child].name.clone(),
            transferred_rad: transferred,
            singular_swing: singular,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    #[test]
    fn shoulder_roll_aligns_the_elbow_hinge_without_moving_the_hand() {
        let joint = |name: &str, parent, translation| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(translation),
            matrix: None,
            deform: true,
        };
        let s = Skeleton {
            joints: vec![
                joint("upper", None, [0., 0., 0.]),
                joint("elbow", Some(0), [0., 0.3, 0.]),
                joint("hand", Some(1), [0., 0.25, 0.]),
            ],
        };
        let limit = ArticulationLimit {
            joint: 1,
            reference_rotation: [0., 0., 0., 1.],
            twist_axis: [0., 1., 0.],
            twist_range_rad: [-1.5, 1.5],
            max_swing_rad: 2.7,
            hinge: Some(HingeLimit {
                axis_parent: [0., 0., 1.],
                range_rad: [0., 2.7],
            }),
        };
        for azimuth in [-2.9, -1.0, 0.4, 2.8] {
            let mut pose = s.rest_pose();
            pose[1].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::y_axis(), azimuth)
                * UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 1.2)
                * UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.7));
            let before = s.world_matrices(&pose).unwrap();
            limit.distribute_bend_plane(&s, &mut pose).unwrap();
            let after = s.world_matrices(&pose).unwrap();
            for j in [1, 2] {
                assert!((before[j] - after[j]).norm() < 1e-10);
            }
            let q = unit_quat(pose[1].rotation);
            assert!(q.angle_to(&limit.project_rotation(q)) < 1e-10);
        }
    }

    #[test]
    fn transfers_roll_without_moving_the_hand_or_its_descendants() {
        let joint = |name: &str, parent, translation| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(translation),
            matrix: None,
            deform: true,
        };
        let mut skeleton = Skeleton {
            joints: vec![
                joint("forearm", None, [0.3, 0.6, -0.1]),
                joint("hand", Some(0), [0.0, 0.25, 0.0]),
                joint("finger", Some(1), [0.01, 0.08, 0.0]),
            ],
        };
        skeleton.joints[0].rest.rotation = a4(UnitQuaternion::from_euler_angles(0.2, -0.4, 0.3));
        skeleton.joints[1].rest.rotation = a4(UnitQuaternion::from_euler_angles(-0.1, 0.3, 0.2));
        let prepared = PreparedTwist::prepare(&skeleton, 1).unwrap();
        for twist in [0.0, 0.7, 2.9, -3.13, 3.13] {
            let mut pose = skeleton.rest_pose();
            pose[1].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::y_axis(), twist)
                * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.5)
                * unit_quat(pose[1].rotation));
            let before = skeleton.world_matrices(&pose).unwrap();
            prepared.apply(&skeleton, &mut pose).unwrap();
            let after = skeleton.world_matrices(&pose).unwrap();
            for i in [1, 2] {
                assert!((before[i] - after[i]).norm() < 1e-10);
            }
            let residual = unit_quat(pose[1].rotation) * prepared.reference.inverse();
            assert!(residual.imag().y.abs() < 1e-10);
            for (i, t) in pose.iter().enumerate() {
                assert_eq!(t.translation, skeleton.joints[i].rest.translation);
            }
        }
    }

    #[test]
    fn fractional_twist_preserves_world_subtree_and_full_transfer_compatibility() {
        let mut skeleton = Skeleton {
            joints: vec![
                Joint {
                    id: "forearm".into(),
                    name: "forearm".into(),
                    parent: None,
                    rest: Transform::translated([0.2, 0.6, -0.1]),
                    matrix: None,
                    deform: true,
                },
                Joint {
                    id: "hand".into(),
                    name: "hand".into(),
                    parent: Some(0),
                    rest: Transform::translated([0., 0.3, 0.]),
                    matrix: None,
                    deform: true,
                },
                Joint {
                    id: "finger".into(),
                    name: "finger".into(),
                    parent: Some(1),
                    rest: Transform::translated([0.02, 0.1, 0.]),
                    matrix: None,
                    deform: true,
                },
            ],
        };
        skeleton.joints[0].rest.rotation = a4(UnitQuaternion::from_euler_angles(0.2, -0.4, 0.3));
        skeleton.joints[0].rest.scale = [1.2; 3];
        skeleton.joints[1].rest.rotation = a4(UnitQuaternion::from_euler_angles(-0.1, 0.2, 0.3));
        skeleton.joints[1].rest.scale = [0.8; 3];
        let prepared = PreparedTwist::prepare(&skeleton, 1).unwrap();
        for angle in [0., 0.7, 2.6, -2.8] {
            for fraction in [0., 0.25, 0.75, 1.] {
                let mut before = skeleton.rest_pose();
                before[1].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::y_axis(), angle)
                    * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), 0.4)
                    * unit_quat(before[1].rotation));
                let mut pose = before.clone();
                let world_before = skeleton.world_matrices(&before).unwrap();
                let report = prepared
                    .apply_fraction(&skeleton, &mut pose, fraction)
                    .unwrap();
                let world_after = skeleton.world_matrices(&pose).unwrap();
                for i in [1, 2] {
                    assert!((world_before[i] - world_after[i]).norm() < 1e-10);
                }
                for i in 0..pose.len() {
                    assert_eq!(pose[i].translation, before[i].translation);
                    assert_eq!(pose[i].scale, before[i].scale);
                }
                assert!((report.transferred_rad - angle.abs() * fraction).abs() < 1e-10);
                if fraction == 0. {
                    assert_eq!(pose, before);
                }
                if fraction == 1. {
                    let mut legacy = before.clone();
                    prepared.apply(&skeleton, &mut legacy).unwrap();
                    assert_eq!(pose, legacy);
                }
            }
        }
        for fraction in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            let mut pose = skeleton.rest_pose();
            let before = pose.clone();
            assert!(
                prepared
                    .apply_fraction(&skeleton, &mut pose, fraction)
                    .is_err()
            );
            assert_eq!(pose, before);
        }
    }

    #[test]
    fn rejects_nonuniform_scale_and_reports_singular_swing() {
        let skeleton = Skeleton {
            joints: vec![
                Joint {
                    id: "a".into(),
                    name: "a".into(),
                    parent: None,
                    rest: Transform::default(),
                    matrix: None,
                    deform: true,
                },
                Joint {
                    id: "b".into(),
                    name: "b".into(),
                    parent: Some(0),
                    rest: Transform::translated([0.0, 1.0, 0.0]),
                    matrix: None,
                    deform: true,
                },
            ],
        };
        let prepared = PreparedTwist::prepare(&skeleton, 1).unwrap();
        let mut pose = skeleton.rest_pose();
        pose[0].scale = [1.0, 2.0, 1.0];
        assert!(prepared.apply(&skeleton, &mut pose).is_err());
        pose = skeleton.rest_pose();
        pose[1].rotation = a4(UnitQuaternion::from_axis_angle(
            &Vector3::x_axis(),
            std::f64::consts::PI,
        ));
        assert!(prepared.apply(&skeleton, &mut pose).unwrap().singular_swing);
    }
}
