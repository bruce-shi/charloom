 use anyhow::{Context, Result, ensure};
use charloom_core::*;
use nalgebra::{DMatrix, DVector, Matrix4, Rotation3, Unit, UnitQuaternion, Vector3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Effector {
    pub joint: usize,
    pub position: Vec3,
    #[serde(default = "one")]
    pub position_weight: f64,
    /// World axes constrained by this target. For grounded stance, X/Z may be
    /// locked while the independently measured sole controls vertical height.
    #[serde(default = "all_position_axes")]
    pub position_axes: [bool; 3],
    #[serde(default)]
    pub rotation: Option<Quat>,
    #[serde(default = "one")]
    pub rotation_weight: f64,
    #[serde(default)]
    pub chain_root: Option<usize>,
    #[serde(default)]
    pub pole: Option<Vec3>,
}
fn one() -> f64 {
    1.0
}
fn all_position_axes() -> [bool; 3] {
    [true; 3]
}
fn axis_mask(axes: [bool; 3]) -> Vector3<f64> {
    Vector3::from(axes.map(|active| if active { 1.0 } else { 0.0 }))
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JointLimit {
    pub joint: usize,
    /// A cone bound relative to the reference local rotation; not an anatomy prior.
    pub max_angle_rad: f64,
    #[serde(default)]
    pub reference_rotation: Option<Quat>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct SolveConfig {
    pub iterations: usize,
    pub tolerance_m: f64,
    pub rotation_tolerance_rad: f64,
    pub damping: f64,
    pub max_step_rad: f64,
    pub posture_weight: f64,
    pub root_translation: bool,
    pub warm_start_two_bone: bool,
    pub limits: Vec<JointLimit>,
    pub articulations: Vec<crate::articulation::ArticulationLimit>,
    pub collision: Option<crate::collision::CollisionConfig>,
}
impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            iterations: 80,
            tolerance_m: 0.001,
            rotation_tolerance_rad: 0.01,
            damping: 0.03,
            max_step_rad: 0.25,
            posture_weight: 0.00001,
            root_translation: false,
            warm_start_two_bone: true,
            limits: Vec::new(),
            articulations: Vec::new(),
            collision: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct EffectorResidual {
    pub joint: usize,
    pub position_error_m: f64,
    pub full_position_error_m: f64,
    pub position_axes: [bool; 3],
    pub rotation_error_rad: Option<f64>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SolveResult {
    pub status: String,
    pub iterations: usize,
    pub pose: Vec<Transform>,
    pub residuals: Vec<EffectorResidual>,
    pub max_bone_length_drift: f64,
    pub collision_residuals: Vec<crate::collision::CollisionResidual>,
}

pub fn world_rotation(matrix: &Matrix4<f64>) -> Result<UnitQuaternion<f64>> {
    let mut basis = matrix.fixed_view::<3, 3>(0, 0).into_owned();
    for i in 0..3 {
        let norm = basis.column(i).norm();
        ensure!(norm > 1e-12, "singular transform");
        basis.column_mut(i).scale_mut(1.0 / norm);
    }
    ensure!(
        (basis.transpose() * basis - nalgebra::Matrix3::identity()).norm() < 1e-4
            && basis.determinant() > 0.0,
        "unsupported_sheared_or_reflected_solve_frame: bake the rig into a rigid frame first"
    );
    Ok(UnitQuaternion::from_rotation_matrix(
        &Rotation3::from_matrix_unchecked(basis),
    ))
}

fn between(a: Vector3<f64>, b: Vector3<f64>) -> UnitQuaternion<f64> {
    let a = a.normalize();
    let b = b.normalize();
    UnitQuaternion::rotation_between(&a, &b).unwrap_or_else(|| {
        let axis = a.cross(&if a.x.abs() < 0.8 {
            Vector3::x()
        } else {
            Vector3::y()
        });
        UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), std::f64::consts::PI)
    })
}

pub fn two_bone(
    skeleton: &Skeleton,
    pose: &mut [Transform],
    root: usize,
    middle: usize,
    tip: usize,
    target: Vec3,
    pole: Option<Vec3>,
) -> Result<()> {
    ensure!(
        root < skeleton.joints.len()
            && middle < skeleton.joints.len()
            && tip < skeleton.joints.len(),
        "IK joint index out of bounds"
    );
    ensure!(
        skeleton.joints[middle].parent == Some(root) && skeleton.joints[tip].parent == Some(middle),
        "two_bone requires a direct three-joint chain"
    );
    let world = skeleton.world_matrices(pose)?;
    let a = v3(transform_point(&world[root], [0.0; 3]));
    let b = v3(transform_point(&world[middle], [0.0; 3]));
    let c = v3(transform_point(&world[tip], [0.0; 3]));
    let l1 = (b - a).norm();
    let l2 = (c - b).norm();
    ensure!(l1 > 1e-9 && l2 > 1e-9, "degenerate IK bone");
    let delta = v3(target) - a;
    let direction = delta
        .try_normalize(1e-12)
        .unwrap_or_else(|| (c - a).try_normalize(1e-12).unwrap_or(Vector3::y()));
    let distance = delta.norm().clamp((l1 - l2).abs() + 1e-9, (l1 + l2) - 1e-9);
    let mut pole_direction = pole.map_or(b - a, |p| v3(p) - a);
    pole_direction -= direction * pole_direction.dot(&direction);
    let perpendicular = pole_direction.try_normalize(1e-9).unwrap_or_else(|| {
        let reference = if direction.z.abs() < 0.8 {
            Vector3::z()
        } else {
            Vector3::x()
        };
        (reference - direction * reference.dot(&direction)).normalize()
    });
    let cosine =
        ((l1 * l1 + distance * distance - l2 * l2) / (2.0 * l1 * distance)).clamp(-1.0, 1.0);
    let desired_b = a
        + direction * (l1 * cosine)
        + perpendicular * (l1 * (1.0 - cosine * cosine).max(0.0).sqrt());
    let desired_c = a + direction * distance;
    let root_world = between(b - a, desired_b - a) * world_rotation(&world[root])?;
    let parent_world = skeleton.joints[root]
        .parent
        .map(|p| world_rotation(&world[p]))
        .transpose()?
        .unwrap_or_default();
    pose[root].rotation = a4(parent_world.inverse() * root_world);
    let world = skeleton.world_matrices(pose)?;
    let actual_b = v3(transform_point(&world[middle], [0.0; 3]));
    let actual_c = v3(transform_point(&world[tip], [0.0; 3]));
    let middle_world =
        between(actual_c - actual_b, desired_c - actual_b) * world_rotation(&world[middle])?;
    pose[middle].rotation = a4(world_rotation(&world[root])?.inverse() * middle_world);
    Ok(())
}

fn ancestry(skeleton: &Skeleton, effector: &Effector) -> Result<Vec<usize>> {
    let mut path = vec![effector.joint];
    let mut at = effector.joint;
    while Some(at) != effector.chain_root {
        let Some(parent) = skeleton.joints[at].parent else {
            break;
        };
        path.push(parent);
        at = parent;
    }
    if let Some(root) = effector.chain_root {
        ensure!(at == root, "chain_root is not an ancestor of effector");
    }
    Ok(path)
}

pub(crate) fn project_limits(skeleton: &Skeleton, pose: &mut [Transform], config: &SolveConfig) {
    let passes = if config.limits.iter().any(|l| l.reference_rotation.is_some())
        && !config.articulations.is_empty()
    {
        5
    } else {
        1
    };
    for _ in 0..passes {
        let limits = &config.limits;
        for limit in limits {
            let rest = unit_quat(
                limit
                    .reference_rotation
                    .unwrap_or(skeleton.joints[limit.joint].rest.rotation),
            );
            let delta = rest.inverse() * unit_quat(pose[limit.joint].rotation);
            let v = delta.scaled_axis();
            if v.norm() > limit.max_angle_rad {
                pose[limit.joint].rotation =
                    a4(rest
                        * UnitQuaternion::from_scaled_axis(v.normalize() * limit.max_angle_rad));
            }
        }
        for limit in &config.articulations {
            pose[limit.joint].rotation =
                a4(limit.project_rotation(unit_quat(pose[limit.joint].rotation)));
        }
    }
}

pub(crate) fn residuals(
    skeleton: &Skeleton,
    pose: &[Transform],
    effectors: &[Effector],
) -> Result<Vec<EffectorResidual>> {
    let world = skeleton.world_matrices(pose)?;
    effectors
        .iter()
        .map(|e| {
            Ok(EffectorResidual {
                joint: e.joint,
                position_error_m: (v3(transform_point(&world[e.joint], [0.0; 3])) - v3(e.position))
                    .component_mul(&axis_mask(e.position_axes))
                    .norm(),
                full_position_error_m: (v3(transform_point(&world[e.joint], [0.0; 3]))
                    - v3(e.position))
                .norm(),
                position_axes: e.position_axes,
                rotation_error_rad: e
                    .rotation
                    .map(|q| -> Result<f64> {
                        Ok((unit_quat(q) * world_rotation(&world[e.joint])?.inverse()).angle())
                    })
                    .transpose()?,
            })
        })
        .collect()
}

pub fn solve_pose(
    skeleton: &Skeleton,
    initial: &[Transform],
    effectors: &[Effector],
    config: &SolveConfig,
) -> Result<SolveResult> {
    skeleton.validate()?;
    ensure!(
        !effectors.is_empty() || config.collision.is_some(),
        "at least one effector or collision graph is required"
    );
    ensure!(
        config.iterations > 0 && config.iterations <= 1000,
        "iterations must be 1..1000"
    );
    ensure!(
        [
            config.tolerance_m,
            config.rotation_tolerance_rad,
            config.damping,
            config.max_step_rad
        ]
        .iter()
        .all(|v| v.is_finite() && *v > 0.0)
            && config.posture_weight.is_finite()
            && config.posture_weight >= 0.0,
        "invalid solve configuration"
    );
    ensure!(
        initial.len() == skeleton.joints.len(),
        "pose length mismatch"
    );
    for e in effectors {
        ensure!(
            e.joint < skeleton.joints.len() && e.position.iter().all(|v| v.is_finite()),
            "invalid effector"
        );
        ensure!(
            e.position_weight.is_finite()
                && e.position_weight >= 0.0
                && e.rotation_weight.is_finite()
                && e.rotation_weight >= 0.0,
            "invalid effector weight"
        );
        if let Some(q) = e.rotation {
            ensure!(
                q.iter().all(|v| v.is_finite()) && q.iter().map(|v| v * v).sum::<f64>() > 1e-20,
                "invalid target rotation"
            );
        }
    }
    for l in &config.limits {
        ensure!(
            l.joint < skeleton.joints.len()
                && l.max_angle_rad.is_finite()
                && l.reference_rotation
                    .is_none_or(|q| q.iter().all(|v| v.is_finite())
                        && q.iter().map(|v| v * v).sum::<f64>() > 1e-20)
                && (0.0..=std::f64::consts::PI).contains(&l.max_angle_rad),
            "invalid joint limit"
        );
    }
    for limit in &config.articulations {
        limit.validate(skeleton)?;
    }
    let paths: Vec<_> = effectors
        .iter()
        .map(|e| ancestry(skeleton, e))
        .collect::<Result<_>>()?;
    let mut active: Vec<_> = paths.iter().flatten().copied().collect();
    if let Some(collision) = &config.collision {
        active.extend(collision.validate(skeleton)?);
    }
    active.sort_unstable();
    active.dedup();
    ensure!(!active.is_empty(), "solve_has_no_editable_joints");
    let collision_ancestors = if config.collision.is_some() {
        (0..skeleton.joints.len())
            .map(|i| crate::collision::ancestors(skeleton, i, None))
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    ensure!(
        active.iter().all(|&j| skeleton.joints[j].matrix.is_none()),
        "matrix-based joint is not editable by the IK solver"
    );
    let mut pose = initial.to_vec();
    if config.warm_start_two_bone {
        for e in effectors {
            if e.position_weight == 0.0 || !e.position_axes.iter().all(|axis| *axis) {
                continue;
            }
            if let Some(middle) = skeleton.joints[e.joint].parent
                && let Some(root) = skeleton.joints[middle].parent
                && e.chain_root.is_none_or(|r| r == root)
            {
                // Warm starts are optional; zero-length attachment bones are common.
                let _ = two_bone(
                    skeleton, &mut pose, root, middle, e.joint, e.position, e.pole,
                );
            }
        }
    }
    project_limits(skeleton, &mut pose, config);
    let roots: Vec<_> = if config.root_translation {
        active
            .iter()
            .copied()
            .filter(|&i| skeleton.joints[i].parent.is_none())
            .collect()
    } else {
        Vec::new()
    };
    let columns = active.len() * 3 + roots.len() * 3;
    let effector_rows: usize = effectors
        .iter()
        .map(|e| if e.rotation.is_some() { 6 } else { 3 })
        .sum();
    let rows = effector_rows + config.collision.as_ref().map_or(0, |c| c.pairs.len());
    let mut iterations = 0;
    for iteration in 0..config.iterations {
        iterations = iteration;
        let residual = residuals(skeleton, &pose, effectors)?;
        let collision_satisfied = config
            .collision
            .as_ref()
            .map(|c| {
                Ok::<_, anyhow::Error>(
                    c.residuals(skeleton, &pose)?
                        .iter()
                        .all(|r| r.penetration_m <= c.tolerance_m),
                )
            })
            .transpose()?
            .unwrap_or(true);
        if collision_satisfied
            && residual.iter().zip(effectors).all(|(r, e)| {
                (e.position_weight == 0.0 || r.position_error_m <= config.tolerance_m)
                    && (e.rotation_weight == 0.0
                        || r.rotation_error_rad
                            .is_none_or(|a| a <= config.rotation_tolerance_rad))
            })
        {
            break;
        }
        let world = skeleton.world_matrices(&pose)?;
        let mut rotations = vec![UnitQuaternion::identity(); world.len()];
        for &joint in &active {
            rotations[joint] = world_rotation(&world[joint])?;
        }
        let axes: Vec<_> = pose
            .iter()
            .enumerate()
            .map(|(j, t)| {
                config
                    .articulations
                    .iter()
                    .find(|l| l.joint == j)
                    .map_or([Vector3::x(), Vector3::y(), Vector3::z()], |l| {
                        l.local_axes(unit_quat(t.rotation))
                    })
            })
            .collect();
        // Differentiate the constrained rotation map itself at joint bounds.
        // An unconstrained wrist Jacobian can point entirely outside its cone,
        // leaving a reachable arm stuck after every update gets clamped away.
        let epsilon = 1e-5;
        let mut projected_steps = std::collections::HashMap::new();
        for (column, &joint) in active.iter().enumerate() {
            if !config.articulations.iter().any(|l| l.joint == joint) {
                continue;
            }
            for (axis, local_axis) in axes[joint].iter().enumerate() {
                let mut plus = pose.clone();
                let mut minus = pose.clone();
                let d = local_axis * epsilon;
                plus[joint].rotation =
                    a4(unit_quat(pose[joint].rotation) * UnitQuaternion::from_scaled_axis(d));
                minus[joint].rotation =
                    a4(unit_quat(pose[joint].rotation) * UnitQuaternion::from_scaled_axis(-d));
                project_limits(skeleton, &mut plus, config);
                project_limits(skeleton, &mut minus, config);
                let q = unit_quat(pose[joint].rotation);
                let plus_free = q.angle_to(&unit_quat(plus[joint].rotation)) > epsilon * 0.05;
                let minus_free = q.angle_to(&unit_quat(minus[joint].rotation)) > epsilon * 0.05;
                projected_steps.insert(
                    column * 3 + axis,
                    (
                        skeleton.world_matrices(&plus)?,
                        skeleton.world_matrices(&minus)?,
                        plus_free,
                        minus_free,
                    ),
                );
            }
        }
        let mut jacobian = DMatrix::<f64>::zeros(rows, columns);
        let mut error = DVector::<f64>::zeros(rows);
        let mut row = 0;
        for (effector, path) in effectors.iter().zip(&paths) {
            let position = v3(transform_point(&world[effector.joint], [0.0; 3]));
            let mask = axis_mask(effector.position_axes);
            let e = (v3(effector.position) - position).component_mul(&mask)
                * effector.position_weight.sqrt();
            error.rows_mut(row, 3).copy_from(&e);
            let rotation_error = effector.rotation.map(|q| {
                (unit_quat(q) * rotations[effector.joint].inverse()).scaled_axis()
                    * effector.rotation_weight.sqrt()
            });
            if let Some(e) = rotation_error {
                error.rows_mut(row + 3, 3).copy_from(&e);
            }
            for (column, &joint) in active.iter().enumerate() {
                if !path.contains(&joint) {
                    continue;
                }
                let pivot = v3(transform_point(&world[joint], [0.0; 3]));
                for (axis, local_axis) in axes[joint].iter().enumerate() {
                    if let Some((plus, minus, _, _)) = projected_steps.get(&(column * 3 + axis)) {
                        let derivative = (v3(transform_point(&plus[effector.joint], [0.0; 3]))
                            - v3(transform_point(&minus[effector.joint], [0.0; 3])))
                            / (2.0 * epsilon);
                        jacobian
                            .view_mut((row, column * 3 + axis), (3, 1))
                            .copy_from(
                                &(derivative.component_mul(&mask)
                                    * effector.position_weight.sqrt()),
                            );
                        if rotation_error.is_some() {
                            let d = (world_rotation(&plus[effector.joint])?
                                * world_rotation(&minus[effector.joint])?.inverse())
                            .scaled_axis()
                                / (2.0 * epsilon);
                            jacobian
                                .view_mut((row + 3, column * 3 + axis), (3, 1))
                                .copy_from(&(d * effector.rotation_weight.sqrt()));
                        }
                        continue;
                    }
                    let basis = rotations[joint] * local_axis;
                    let derivative = basis.cross(&(position - pivot)).component_mul(&mask)
                        * effector.position_weight.sqrt();
                    jacobian
                        .view_mut((row, column * 3 + axis), (3, 1))
                        .copy_from(&derivative);
                    if rotation_error.is_some() {
                        jacobian
                            .view_mut((row + 3, column * 3 + axis), (3, 1))
                            .copy_from(&(basis * effector.rotation_weight.sqrt()));
                    }
                }
            }
            for (column, &root) in roots.iter().enumerate() {
                if path.contains(&root) {
                    for axis in 0..3 {
                        jacobian[(row + axis, active.len() * 3 + column * 3 + axis)] =
                            effector.position_weight.sqrt() * mask[axis];
                    }
                }
            }
            row += if effector.rotation.is_some() { 6 } else { 3 };
        }
        if let Some(collision) = &config.collision {
            let contacts = collision.rows(&world);
            // Elliptical support changes with both orientation and the contact
            // normal. Differentiate the complete residual, not only centers.
            let numeric =
                if collision.capsules.iter().any(|c| c.cross_section.is_some()) {
                    let mut derivatives = DMatrix::zeros(contacts.len(), columns);
                    let epsilon = 1e-5;
                    for column in 0..columns {
                        let mut plus = pose.clone();
                        let mut minus = pose.clone();
                        if column < active.len() * 3 {
                            let j = active[column / 3];
                            let axis = axes[j][column % 3] * epsilon;
                            plus[j].rotation = a4(unit_quat(pose[j].rotation)
                                * UnitQuaternion::from_scaled_axis(axis));
                            minus[j].rotation = a4(unit_quat(pose[j].rotation)
                                * UnitQuaternion::from_scaled_axis(-axis));
                        } else {
                            let offset = column - active.len() * 3;
                            let j = roots[offset / 3];
                            plus[j].translation[offset % 3] += epsilon;
                            minus[j].translation[offset % 3] -= epsilon;
                        }
                        project_limits(skeleton, &mut plus, config);
                        project_limits(skeleton, &mut minus, config);
                        let p = collision.rows(&skeleton.world_matrices(&plus)?);
                        let m = collision.rows(&skeleton.world_matrices(&minus)?);
                        for i in 0..contacts.len() {
                            derivatives[(i, column)] = (m[i].depth - p[i].depth) / (2.0 * epsilon);
                        }
                    }
                    Some(derivatives)
                } else {
                    None
                };
            for (contact_index, contact) in contacts.into_iter().enumerate() {
                if contact.depth <= 0.0 {
                    row += 1;
                    continue;
                }
                let weight = collision.weight.sqrt();
                error[row] = contact.depth * weight;
                if let Some(numeric) = &numeric
                    && contact.distance > 1e-7
                {
                    for column in 0..columns {
                        jacobian[(row, column)] = numeric[(contact_index, column)] * weight;
                    }
                    row += 1;
                    continue;
                }
                let a = &collision.capsules[contact.a];
                let b = &collision.capsules[contact.b];
                for (column, &joint) in active.iter().enumerate() {
                    let pivot = v3(transform_point(&world[joint], [0.0; 3]));
                    for (axis, local_axis) in axes[joint].iter().enumerate() {
                        let basis = rotations[joint] * local_axis;
                        let derivative = |capsule: &crate::collision::CollisionCapsule,
                                          fraction: f64| {
                            let mut value = Vector3::zeros();
                            for (end, coefficient) in
                                [(capsule.start, 1.0 - fraction), (capsule.end, fraction)]
                            {
                                if collision_ancestors[end].contains(&joint) {
                                    value += basis.cross(
                                        &(v3(transform_point(&world[end], [0.0; 3])) - pivot),
                                    ) * coefficient;
                                }
                            }
                            value
                        };
                        jacobian[(row, column * 3 + axis)] = contact.normal.dot(
                            &(derivative(a, contact.fraction_a)
                                - derivative(b, contact.fraction_b)),
                        ) * weight;
                    }
                }
                for (column, &root) in roots.iter().enumerate() {
                    let influence =
                        |capsule: &crate::collision::CollisionCapsule, fraction: f64| -> f64 {
                            (if collision_ancestors[capsule.start].contains(&root) {
                                1.0 - fraction
                            } else {
                                0.0
                            }) + (if collision_ancestors[capsule.end].contains(&root) {
                                fraction
                            } else {
                                0.0
                            })
                        };
                    for axis in 0..3 {
                        jacobian[(row, active.len() * 3 + column * 3 + axis)] = contact.normal
                            [axis]
                            * (influence(a, contact.fraction_a) - influence(b, contact.fraction_b))
                            * weight;
                    }
                }
                row += 1;
            }
        }
        // Freeze a fully blocked direction only when the objective pushes
        // outward. Other joints can then absorb the remaining task error.
        for (&column, (_, _, plus_free, minus_free)) in &projected_steps {
            let gradient = jacobian.column(column).dot(&error);
            if (gradient > 0.0 && !plus_free) || (gradient < 0.0 && !minus_free) {
                jacobian.column_mut(column).fill(0.0);
            }
        }
        let regularization = config.damping.powi(2) + config.posture_weight;
        let mut prior = DVector::zeros(columns);
        for (i, &joint) in active.iter().enumerate() {
            let d = (unit_quat(pose[joint].rotation).inverse()
                * unit_quat(initial[joint].rotation))
            .scaled_axis()
                * (config.posture_weight / regularization);
            for a in 0..3 {
                prior[i * 3 + a] = d.dot(&axes[joint][a]);
            }
        }
        let system =
            &jacobian * jacobian.transpose() + DMatrix::identity(rows, rows) * regularization;
        let rhs = &error - &jacobian * &prior;
        let solution = system
            .cholesky()
            .context("singular damped IK system")?
            .solve(&rhs);
        let delta = prior + jacobian.transpose() * solution;
        // Clamp the entire update by one factor. Clamping joints separately
        // changes the least-squares direction and can turn it into an ascent
        // direction, especially when a nearly straight elbow must bend while
        // the shoulder rotates the opposite way.
        let max_rotation_step = (0..active.len())
            .map(|i| delta.rows(i * 3, 3).norm())
            .fold(0.0, f64::max);
        let step_scale = if max_rotation_step > config.max_step_rad {
            config.max_step_rad / max_rotation_step
        } else {
            1.0
        };
        let previous = pose.clone();
        let before = error.norm_squared();
        let mut accepted = false;
        for attempt in 0..8 {
            let factor = step_scale * 0.5_f64.powi(attempt);
            pose.clone_from(&previous);
            for (i, &joint) in active.iter().enumerate() {
                let d = axes[joint][0] * delta[i * 3]
                    + axes[joint][1] * delta[i * 3 + 1]
                    + axes[joint][2] * delta[i * 3 + 2];
                pose[joint].rotation =
                    a4(unit_quat(pose[joint].rotation)
                        * UnitQuaternion::from_scaled_axis(d * factor));
            }
            for (i, &root) in roots.iter().enumerate() {
                let c = active.len() * 3 + i * 3;
                pose[root].translation = a3(v3(pose[root].translation)
                    + Vector3::new(delta[c], delta[c + 1], delta[c + 2]) * factor);
            }
            project_limits(skeleton, &mut pose, config);
            let mut after = residuals(skeleton, &pose, effectors)?
                .iter()
                .zip(effectors)
                .map(|(r, e)| {
                    r.position_error_m.powi(2) * e.position_weight
                        + r.rotation_error_rad.unwrap_or(0.0).powi(2) * e.rotation_weight
                })
                .sum::<f64>();
            if let Some(collision) = &config.collision {
                after += collision
                    .residuals(skeleton, &pose)?
                    .iter()
                    .map(|r| r.penetration_m.powi(2) * collision.weight)
                    .sum::<f64>();
            }
            if after < before {
                accepted = true;
                break;
            }
        }
        if !accepted {
            pose = previous;
            break;
        }
        iterations = iteration + 1;
    }
    let residuals = residuals(skeleton, &pose, effectors)?;
    let collision_residuals = config
        .collision
        .as_ref()
        .map(|c| c.residuals(skeleton, &pose))
        .transpose()?
        .unwrap_or_default();
    let collision_satisfied = config.collision.as_ref().is_none_or(|c| {
        collision_residuals
            .iter()
            .all(|r| r.penetration_m <= c.tolerance_m)
    });
    let converged = collision_satisfied
        && residuals.iter().zip(effectors).all(|(r, e)| {
            (e.position_weight == 0.0 || r.position_error_m <= config.tolerance_m)
                && (e.rotation_weight == 0.0
                    || r.rotation_error_rad
                        .is_none_or(|a| a <= config.rotation_tolerance_rad))
        });
    let drift = initial
        .iter()
        .zip(&pose)
        .map(|(a, b)| (v3(a.translation) - v3(b.translation)).norm())
        .enumerate()
        .filter(|(i, _)| skeleton.joints[*i].parent.is_some())
        .map(|(_, d)| d)
        .fold(0.0, f64::max);
    Ok(SolveResult {
        status: if converged {
            "converged"
        } else {
            "not_converged"
        }
        .into(),
        iterations,
        pose,
        residuals,
        max_bone_length_drift: drift,
        collision_residuals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn volume_constraints_separate_an_arm_without_stretching_bones() {
        use crate::collision::{CollisionCapsule, CollisionConfig};
        let joint = |name: &str, parent, position| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(position),
            matrix: None,
            deform: true,
        };
        for (radius, elliptical) in [(0.12, false), (0.25, false), (0.18, true)] {
            let s = Skeleton {
                joints: vec![
                    joint("shoulder", None, [0.5, 1.0, 0.0]),
                    joint("elbow", Some(0), [-0.4, 0.0, 0.0]),
                    joint("hand", Some(1), [-0.4, 0.0, 0.0]),
                    joint("body_low", None, [0.0, 0.2, 0.0]),
                    joint("body_high", None, [0.0, 1.5, 0.0]),
                ],
            };
            let collision = CollisionConfig {
                capsules: vec![
                    CollisionCapsule {
                        cross_section: elliptical.then_some(
                            crate::collision::CapsuleCrossSection {
                                lateral_axis_local: [0.0, 0.0, 1.0],
                                radii_m: [0.025, 0.08],
                            },
                        ),
                        name: "forearm".into(),
                        start: 1,
                        end: 2,
                        interval: [0.0, 1.0],
                        radius_m: 0.05,
                        chain_root: Some(0),
                    },
                    CollisionCapsule {
                        cross_section: elliptical.then_some(
                            crate::collision::CapsuleCrossSection {
                                lateral_axis_local: [1.0, 0.0, 0.0],
                                radii_m: [0.13, 0.23],
                            },
                        ),
                        name: "torso".into(),
                        start: 3,
                        end: 4,
                        interval: [0.0, 1.0],
                        radius_m: radius,
                        chain_root: None,
                    },
                ],
                pairs: vec![[0, 1]],
                margin_m: 0.005,
                weight: 100.0,
                tolerance_m: 0.001,
            };
            let result = solve_pose(
                &s,
                &s.rest_pose(),
                &[],
                &SolveConfig {
                    iterations: 200,
                    damping: 0.01,
                    collision: Some(collision.clone()),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                result.status, "converged",
                "radius {radius}: {:?}",
                result.collision_residuals
            );
            assert!(result.collision_residuals[0].penetration_m <= 0.001);
            let world = s.positions(&result.pose).unwrap();
            for (a, b) in [(0, 1), (1, 2)] {
                assert!(((v3(world[b]) - v3(world[a])).norm() - 0.4).abs() < 1e-10);
            }
            assert_eq!(result.pose[3], s.joints[3].rest);
            assert_eq!(result.pose[4], s.joints[4].rest);
            let impossible = solve_pose(
                &s,
                &s.rest_pose(),
                &[Effector {
                    joint: 2,
                    position: [-0.3, 1.0, 0.0],
                    position_weight: 1000.0,
                    position_axes: [true; 3],
                    rotation: None,
                    rotation_weight: 0.0,
                    chain_root: Some(0),
                    pole: None,
                }],
                &SolveConfig {
                    iterations: 120,
                    damping: 0.01,
                    collision: Some(collision),
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(impossible.status, "not_converged");
            assert!(
                impossible.residuals[0].position_error_m > 0.001
                    || impossible.collision_residuals[0].penetration_m > 0.001
            );
            assert_eq!(impossible.max_bone_length_drift, 0.0);
        }
    }
    fn chain() -> Skeleton {
        Skeleton {
            joints: (0..3)
                .map(|i| Joint {
                    id: format!("j{i}"),
                    name: format!("j{i}"),
                    parent: if i == 0 { None } else { Some(i - 1) },
                    rest: Transform::translated(if i == 0 { [0.0; 3] } else { [1.0, 0.0, 0.0] }),
                    matrix: None,
                    deform: true,
                })
                .collect(),
        }
    }
    fn effector(position: Vec3) -> Effector {
        Effector {
            joint: 2,
            position,
            position_weight: 1.0,
            position_axes: [true; 3],
            rotation: None,
            rotation_weight: 1.0,
            chain_root: Some(0),
            pole: Some([0.0, 1.0, 0.0]),
        }
    }
    #[test]
    fn straight_chain_bends_without_changing_lengths() {
        let s = chain();
        let result = solve_pose(
            &s,
            &s.rest_pose(),
            &[effector([1.0, 1.0, 0.0])],
            &SolveConfig::default(),
        )
        .unwrap();
        assert_eq!(result.status, "converged");
        assert!(result.residuals[0].position_error_m < 1e-6);
        assert_eq!(result.max_bone_length_drift, 0.0);
    }
    #[test]
    fn masked_position_axes_ignore_vertical_target_without_hiding_the_full_error() {
        let s = chain();
        let mut target = effector([1.0, 1000.0, 1.0]);
        target.position_axes = [true, false, true];
        let result = solve_pose(&s, &s.rest_pose(), &[target], &SolveConfig::default()).unwrap();
        assert_eq!(result.status, "converged");
        assert!(result.residuals[0].position_error_m < 0.001);
        assert!(result.residuals[0].full_position_error_m > 990.0);
        assert_eq!(result.residuals[0].position_axes, [true, false, true]);
        assert_eq!(result.max_bone_length_drift, 0.0);
        let old: Effector =
            serde_json::from_value(serde_json::json!({"joint":2,"position":[1.,1.,0.]})).unwrap();
        assert_eq!(old.position_axes, [true; 3]);
    }
    #[test]
    fn hidden_accessory_does_not_block_an_unrelated_ik_chain() {
        let mut s = chain();
        s.joints.push(Joint {
            id: "hidden_arrow".into(),
            name: "hidden_arrow".into(),
            parent: None,
            rest: Transform::default(),
            matrix: None,
            deform: false,
        });
        let mut pose = s.rest_pose();
        pose[3].scale = [0.0; 3];
        pose[1].rotation = a4(UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.4));
        let config = SolveConfig {
            warm_start_two_bone: false,
            posture_weight: 1e-8,
            ..Default::default()
        };
        let result = solve_pose(&s, &pose, &[effector([1.0, 1.0, 0.0])], &config).unwrap();
        assert_eq!(result.status, "converged", "{:?}", result.residuals);
        assert_eq!(result.pose[3].scale, [0.0; 3]);
        assert!(world_rotation(&s.world_matrices(&pose).unwrap()[3]).is_err());
    }

    #[test]
    fn unreachable_target_reports_residual_and_never_stretches() {
        let s = chain();
        let result = solve_pose(
            &s,
            &s.rest_pose(),
            &[effector([5.0, 0.0, 0.0])],
            &SolveConfig::default(),
        )
        .unwrap();
        assert_eq!(result.status, "not_converged");
        assert!(result.residuals[0].position_error_m > 2.99);
        assert_eq!(result.max_bone_length_drift, 0.0);
    }
    #[test]
    fn joint_limits_are_applied_to_warm_start_and_iterations() {
        let s = chain();
        let config = SolveConfig {
            limits: vec![
                JointLimit {
                    joint: 0,
                    max_angle_rad: 0.1,
                    reference_rotation: None,
                },
                JointLimit {
                    joint: 1,
                    max_angle_rad: 0.1,
                    reference_rotation: None,
                },
            ],
            ..Default::default()
        };
        let result = solve_pose(&s, &s.rest_pose(), &[effector([0.0, 1.0, 0.0])], &config).unwrap();
        assert_eq!(result.status, "not_converged");
        assert!(unit_quat(result.pose[0].rotation).angle() <= 0.100001);
        assert!(unit_quat(result.pose[1].rotation).angle() <= 0.100001);
    }
}
