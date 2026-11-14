 use charloom_core::{Skeleton, Transform};
use nalgebra::{UnitQuaternion, Vector3};

/// (name, [(joint, local axis, radians)]).
pub type Expression = (&'static str, &'static [(&'static str, char, f64)]);

/// Turns added on top of the rest pose. `CHARLOOM_FACE_POSE`
/// (`joint:axis:radians,...`) adds a `custom` expression for tuning.
pub const EXPRESSIONS: &[Expression] = &[
    ("neutral", &[]),
    ("jaw-open", &[("jaw", 'x', 0.30)]),
    (
        "smile",
        &[
            ("levator03.L", 'x', -0.5),
            ("levator03.R", 'x', -0.5),
            ("risorius03.L", 'x', -0.6),
            ("risorius03.R", 'x', -0.6),
        ],
    ),
    (
        "brows-up",
        &[
            ("oculi01.L", 'x', -0.5),
            ("oculi01.R", 'x', -0.5),
            ("oculi02.L", 'x', -0.5),
            ("oculi02.R", 'x', -0.5),
        ],
    ),
    (
        "frown",
        &[
            ("oculi02.L", 'x', 0.4),
            ("oculi02.R", 'x', 0.4),
            ("risorius03.L", 'x', 0.6),
            ("risorius03.R", 'x', 0.6),
        ],
    ),
    (
        "blink",
        &[
            ("orbicularis03.L", 'x', -0.5),
            ("orbicularis03.R", 'x', -0.5),
        ],
    ),
    ("look-left", &[("eye.L", 'Y', 0.35), ("eye.R", 'Y', 0.35)]),
    ("look-up", &[("eye.L", 'X', -0.3), ("eye.R", 'X', -0.3)]),
];

/// The tuning expression from `CHARLOOM_FACE_POSE`, if set.
pub fn custom() -> Option<Vec<(String, char, f64)>> {
    let spec = std::env::var("CHARLOOM_FACE_POSE").ok()?;
    let turns: Vec<(String, char, f64)> = spec
        .split(',')
        .filter_map(|t| {
            let mut p = t.split(':');
            let joint = p.next()?.trim().to_string();
            let axis = p.next()?.trim().chars().next()?;
            let angle = p.next()?.trim().parse().ok()?;
            Some((joint, axis, angle))
        })
        .collect();
    (!turns.is_empty()).then_some(turns)
}

/// `rest` (local transforms) with `turns` applied about each joint's local
/// axis ('x', 'y', 'z') or about the model's axis ('X', 'Y', 'Z'). Joints
/// the skeleton does not have are skipped.
pub fn pose<'a>(
    skeleton: &Skeleton,
    rest: &[Transform],
    turns: impl IntoIterator<Item = (&'a str, char, f64)>,
) -> Vec<Transform> {
    let mut pose = rest.to_vec();
    for (joint, axis, angle) in turns {
        let Some(j) = skeleton.joints.iter().position(|x| x.name == joint) else {
            continue;
        };
        let unit = match axis.to_ascii_lowercase() {
            'x' => Vector3::x_axis(),
            'y' => Vector3::y_axis(),
            _ => Vector3::z_axis(),
        };
        let local = charloom_core::unit_quat(pose[j].rotation);
        let axis = if axis.is_ascii_uppercase() {
            // The model's axis in the joint's rest frame.
            let mut world = charloom_core::unit_quat(rest[j].rotation);
            let mut parent = skeleton.joints[j].parent;
            while let Some(p) = parent {
                world = charloom_core::unit_quat(rest[p].rotation) * world;
                parent = skeleton.joints[p].parent;
            }
            nalgebra::Unit::new_normalize(world.inverse() * unit.into_inner())
        } else {
            unit
        };
        let q = local * UnitQuaternion::from_axis_angle(&axis, angle);
        pose[j].rotation = [q.i, q.j, q.k, q.w];
    }
    pose
}

/// Every expression, rest included, as (name, local pose).
pub fn poses(skeleton: &Skeleton, rest: &[Transform]) -> Vec<(String, Vec<Transform>)> {
    let mut out: Vec<(String, Vec<Transform>)> = EXPRESSIONS
        .iter()
        .map(|(name, turns)| {
            (
                name.to_string(),
                pose(skeleton, rest, turns.iter().map(|&(j, a, r)| (j, a, r))),
            )
        })
        .collect();
    if let Some(c) = custom() {
        out.push((
            "custom".into(),
            pose(skeleton, rest, c.iter().map(|(j, a, r)| (j.as_str(), *a, *r))),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use charloom_core::Joint;

    #[test]
    fn expression_turns_only_its_joints() {
        let joint = |name: &str, parent| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated([0.0, 0.1, 0.0]),
            matrix: None,
            deform: true,
        };
        let s = Skeleton {
            joints: vec![joint("head", None), joint("jaw", Some(0))],
        };
        let rest = s.rest_pose();
        let p = pose(&s, &rest, [("jaw", 'x', 0.3), ("missing", 'x', 1.0)]);
        assert_eq!(p[0].rotation, rest[0].rotation);
        let q = charloom_core::unit_quat(p[1].rotation);
        assert!((q.angle() - 0.3).abs() < 1e-9);
    }

    #[test]
    fn eyes_look_up_and_left_whatever_the_bone_roll() {
        let mut s = Skeleton { joints: vec![] };
        // A head leaning back and a rolled, mirrored eye bone under it.
        for (name, parent, turn) in [
            ("head", None, UnitQuaternion::from_euler_angles(0.4, 0.2, 0.0)),
            ("eye.L", Some(0), UnitQuaternion::from_euler_angles(1.3, 0.3, -2.8)),
        ] {
            let mut rest = Transform::translated([0.0, 0.1, 0.0]);
            rest.rotation = [turn.i, turn.j, turn.k, turn.w];
            s.joints.push(Joint {
                id: name.into(),
                name: name.into(),
                parent,
                rest,
                matrix: None,
                deform: true,
            });
        }
        let rest = s.rest_pose();
        let world = |p: &[Transform]| {
            charloom_core::unit_quat(p[0].rotation) * charloom_core::unit_quat(p[1].rotation)
        };
        // Which way a gaze that pointed forward (+Z) at rest points now.
        let gaze = |name: &str| {
            let turns = EXPRESSIONS.iter().find(|e| e.0 == name).unwrap().1;
            let p = pose(&s, &rest, turns.iter().copied());
            world(&p) * world(&rest).inverse() * Vector3::z()
        };
        let up = gaze("look-up");
        assert!(up.y > 0.25 && up.x.abs() < 1e-9, "{up:?}");
        let left = gaze("look-left");
        assert!(left.x > 0.3 && left.y.abs() < 1e-9, "{left:?}");
    }
}
