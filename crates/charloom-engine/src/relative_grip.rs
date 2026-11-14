//! Joint-space closed-chain solve: the prop follows the driver while every
//! follower constrains that moving frame in the same numerical step.
use crate::solve::{SolveConfig, project_limits, world_rotation};
use anyhow::{Context, Result};
use charloom_core::*;
use nalgebra::{DMatrix, DVector, Matrix4, UnitQuaternion, Vector3};

pub(crate) struct RelativeGrip {
    pub driver: usize,
    pub source: Matrix4<f64>,
    pub followers: Vec<(usize, Matrix4<f64>, bool)>,
    pub active: Vec<usize>,
    pub max_translation: f64,
    pub max_rotation: f64,
    pub orientation_weight: f64,
    pub posture: Vec<Transform>,
}
pub(crate) struct ResultPose {
    pub pose: Vec<Transform>,
    pub converged: bool,
}
impl RelativeGrip {
    fn errors(
        &self,
        skeleton: &Skeleton,
        pose: &[Transform],
        config: &SolveConfig,
    ) -> Result<(DVector<f64>, bool)> {
        let world = skeleton.world_matrices(pose)?;
        let driver = world[self.driver];
        let mut error = Vec::new();
        let mut satisfied = true;
        for &(joint, offset, orientation) in &self.followers {
            let target = driver * offset;
            let delta = v3(transform_point(&target, [0.0; 3]))
                - v3(transform_point(&world[joint], [0.0; 3]));
            satisfied &= delta.norm() <= config.tolerance_m;
            error.extend(delta.iter().copied());
            if orientation {
                let turn = (world_rotation(&target)? * world_rotation(&world[joint])?.inverse())
                    .scaled_axis();
                satisfied &= turn.norm() <= config.rotation_tolerance_rad;
                error.extend(turn.iter().map(|v| v * self.orientation_weight.sqrt()));
            }
        }
        let displacement =
            v3(transform_point(&self.source, [0.0; 3])) - v3(transform_point(&driver, [0.0; 3]));
        let rotation =
            (world_rotation(&self.source)? * world_rotation(&driver)?.inverse()).scaled_axis();
        // Soft trajectory preference, with separate hard-budget residuals.
        error.extend(displacement.iter().map(|v| v * 0.01));
        error.extend(rotation.iter().map(|v| v * 0.001));
         
        for &j in &self.active {
            let turn = (unit_quat(pose[j].rotation).inverse()
                * unit_quat(self.posture[j].rotation))
            .scaled_axis();
            error.extend(turn.iter().map(|v| v * config.posture_weight.sqrt()));
        }
        let excess = (displacement.norm() - self.max_translation).max(0.0);
        let angular = (rotation.norm() - self.max_rotation).max(0.0);
        error.push(excess * 10.0);
        error.push(angular * 10.0);
        satisfied &= excess <= 1e-7 && angular <= 1e-7;
        if let Some(c) = &config.collision {
            for r in c.residuals(skeleton, pose)? {
                satisfied &= r.penetration_m <= c.tolerance_m;
                error.push(r.penetration_m * c.weight.sqrt());
            }
        }
        Ok((DVector::from_vec(error), satisfied))
    }
    pub fn solve(
        &self,
        skeleton: &Skeleton,
        initial: &[Transform],
        config: &SolveConfig,
    ) -> Result<ResultPose> {
        let mut pose = initial.to_vec();
        project_limits(skeleton, &mut pose, config);
        let epsilon = 1e-5;
        let columns = self.active.len() * 3;
        let mut feasible = None;
        for _ in 0..config.iterations.max(240) {
             
            let (error, done) = self.errors(skeleton, &pose, config)?;
            if done {
                feasible = Some(pose.clone());
            }
            let axes: Vec<_> = self
                .active
                .iter()
                .map(|&j| {
                    config
                        .articulations
                        .iter()
                        .find(|l| l.joint == j)
                        .map_or([Vector3::x(), Vector3::y(), Vector3::z()], |l| {
                            l.local_axes(unit_quat(pose[j].rotation))
                        })
                })
                .collect();
            let mut jacobian = DMatrix::<f64>::zeros(error.len(), columns);
            for column in 0..columns {
                let j = self.active[column / 3];
                let d = axes[column / 3][column % 3] * epsilon;
                if d.norm_squared() < 1e-20 {
                    continue;
                }
                let mut plus = pose.clone();
                let mut minus = pose.clone();
                plus[j].rotation =
                    a4(unit_quat(pose[j].rotation) * UnitQuaternion::from_scaled_axis(d));
                minus[j].rotation =
                    a4(unit_quat(pose[j].rotation) * UnitQuaternion::from_scaled_axis(-d));
                project_limits(skeleton, &mut plus, config);
                project_limits(skeleton, &mut minus, config);
                let p = self.errors(skeleton, &plus, config)?.0;
                let m = self.errors(skeleton, &minus, config)?.0;
                let derivative = (m - p) / (2.0 * epsilon);
                let gradient = derivative.dot(&error);
                let plus_free = unit_quat(pose[j].rotation).angle_to(&unit_quat(plus[j].rotation))
                    > epsilon * 0.05;
                let minus_free = unit_quat(pose[j].rotation)
                    .angle_to(&unit_quat(minus[j].rotation))
                    > epsilon * 0.05;
                if (gradient > 0.0 && !plus_free) || (gradient < 0.0 && !minus_free) {
                    continue;
                }
                jacobian.column_mut(column).copy_from(&derivative);
            }
            let system = jacobian.transpose() * &jacobian
                + DMatrix::identity(columns, columns) * config.damping.powi(2);
            let delta = system
                .cholesky()
                .context("singular_relative_grip")?
                .solve(&(jacobian.transpose() * &error));
            let max = (0..self.active.len())
                .map(|i| delta.rows(i * 3, 3).norm())
                .fold(0.0, f64::max);
            let scale = if max > config.max_step_rad {
                config.max_step_rad / max
            } else {
                1.0
            };
            let mut accepted = false;
            for attempt in 0..10 {
                let mut candidate = pose.clone();
                let factor = scale * 0.5_f64.powi(attempt);
                for (i, &j) in self.active.iter().enumerate() {
                    let d = axes[i][0] * delta[i * 3]
                        + axes[i][1] * delta[i * 3 + 1]
                        + axes[i][2] * delta[i * 3 + 2];
                    candidate[j].rotation =
                        a4(unit_quat(pose[j].rotation)
                            * UnitQuaternion::from_scaled_axis(d * factor));
                }
                project_limits(skeleton, &mut candidate, config);
                if self.errors(skeleton, &candidate, config)?.0.norm_squared()
                    < error.norm_squared()
                {
                    pose = candidate;
                    accepted = true;
                    break;
                }
            }
            if !accepted {
                break;
            }
        }
        let converged = self.errors(skeleton, &pose, config)?.1;
        if !converged && let Some(pose) = feasible {
            return Ok(ResultPose {
                pose,
                converged: true,
            });
        }
        Ok(ResultPose { pose, converged })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn closed_chain_preserves_bone_lengths_and_obeys_driver_budget() {
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
                joint("body", None, [0.0; 3]),
                joint("left_root", Some(0), [-0.3, 1.0, 0.0]),
                joint("left_elbow", Some(1), [0.4, 0.0, 0.0]),
                joint("left_hand", Some(2), [0.4, 0.0, 0.0]),
                joint("right_root", Some(0), [0.3, 1.0, 0.0]),
                joint("right_elbow", Some(4), [0.4, 0.0, 0.0]),
                joint("right_hand", Some(5), [0.4, 0.0, 0.0]),
            ],
        };
        let source = s.world_matrices(&s.rest_pose()).unwrap()[6];
        let mut pose = s.rest_pose();
        for j in [2, 5] {
            pose[j].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.2));
        }
        for (budget, expected) in [(0.3, true), (0.02, false)] {
            let task = RelativeGrip {
                driver: 6,
                source,
                followers: vec![(3, Transform::translated([-0.4, 0.0, 0.0]).matrix(), true)],
                active: vec![1, 2, 3, 4, 5, 6],
                max_translation: budget,
                max_rotation: 0.5,
                orientation_weight: 0.01,
                posture: pose.clone(),
            };
            let result = task
                .solve(
                    &s,
                    &pose,
                    &SolveConfig {
                        iterations: 500,
                        ..Default::default()
                    },
                )
                .unwrap();
            assert_eq!(result.converged, expected);
            for (a, b) in pose.iter().zip(&result.pose) {
                assert_eq!(a.translation, b.translation);
                assert_eq!(a.scale, b.scale);
            }
            if expected {
                let world = s.world_matrices(&result.pose).unwrap();
                assert!(
                    (v3(transform_point(&world[6], [0.0; 3]))
                        - v3(transform_point(&source, [0.0; 3])))
                    .norm()
                        <= budget + 1e-7
                );
            }
        }
    }
}
