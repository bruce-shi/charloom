//! Capsule separation inequalities for the pose solver. These are geometric
//! body volumes, not rigid-body dynamics or a proof of mesh nonintersection.
use anyhow::{Context, Result, ensure};
use charloom_core::*;
use nalgebra::{Matrix4, Vector3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

fn interval() -> [f64; 2] {
    [0.0, 1.0]
}
fn weight() -> f64 {
    100.0
}
fn tolerance() -> f64 {
    0.001
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CollisionCapsule {
    pub name: String,
    pub start: usize,
    pub end: usize,
    /// Fractions along the start/end joint line; modest extrapolation allows
    /// a hip or shoulder volume to extend beyond the joint centers.
    #[serde(default = "interval")]
    pub interval: [f64; 2],
    pub radius_m: f64,
    /// Optional elliptical cross section, transported with the start joint.
    #[serde(default)]
    pub cross_section: Option<CapsuleCrossSection>,
    /// Only this root and its descendants may be activated by this volume.
    /// None makes it an obstacle; other explicit effectors may still move it.
    #[serde(default)]
    pub chain_root: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CapsuleCrossSection {
    pub lateral_axis_local: Vec3,
    /// Lateral and front/back radii. The smaller is used for the rounded cap.
    pub radii_m: [f64; 2],
}

impl CollisionCapsule {
    fn support_radius(&self, world: &[Matrix4<f64>], normal: Vector3<f64>) -> f64 {
        let Some(section) = &self.cross_section else {
            return self.radius_m;
        };
        let a = v3(transform_point(&world[self.start], [0.0; 3]));
        let b = v3(transform_point(&world[self.end], [0.0; 3]));
        let Some(y) = (b - a).try_normalize(1e-10) else {
            return self.radius_m;
        };
        let lateral = world[self.start].fixed_view::<3, 3>(0, 0) * v3(section.lateral_axis_local);
        let Some(x) = (lateral - y * lateral.dot(&y)).try_normalize(1e-10) else {
            return self.radius_m;
        };
        let z = y.cross(&x);
        ((section.radii_m[0] * normal.dot(&x)).powi(2)
            + (section.radii_m[1] * normal.dot(&z)).powi(2)
            + (section.radii_m[0].min(section.radii_m[1]) * normal.dot(&y)).powi(2))
        .sqrt()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CollisionConfig {
    pub capsules: Vec<CollisionCapsule>,
    pub pairs: Vec<[usize; 2]>,
    #[serde(default)]
    pub margin_m: f64,
    #[serde(default = "weight")]
    pub weight: f64,
    #[serde(default = "tolerance")]
    pub tolerance_m: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CollisionResidual {
    pub pair: [String; 2],
    pub surface_gap_m: f64,
    pub required_gap_m: f64,
    pub penetration_m: f64,
}

pub(crate) struct CollisionRow {
    pub a: usize,
    pub b: usize,
    pub fraction_a: f64,
    pub fraction_b: f64,
    pub normal: Vector3<f64>,
    pub depth: f64,
    pub distance: f64,
    pub radius_sum: f64,
}

pub(crate) fn ancestors(
    skeleton: &Skeleton,
    joint: usize,
    stop: Option<usize>,
) -> Result<Vec<usize>> {
    let mut path = vec![joint];
    let mut at = joint;
    while Some(at) != stop {
        let Some(parent) = skeleton.joints[at].parent else {
            break;
        };
        path.push(parent);
        at = parent;
    }
    ensure!(
        stop.is_none_or(|root| at == root),
        "collision_chain_root_not_ancestor"
    );
    Ok(path)
}

impl CollisionConfig {
    pub(crate) fn validate(&self, skeleton: &Skeleton) -> Result<Vec<usize>> {
        ensure!(
            !self.capsules.is_empty()
                && self.capsules.len() <= 128
                && !self.pairs.is_empty()
                && self.pairs.len() <= 512,
            "invalid_collision_graph_size"
        );
        ensure!(
            self.margin_m.is_finite()
                && self.margin_m >= 0.0
                && self.weight.is_finite()
                && self.weight > 0.0
                && self.tolerance_m.is_finite()
                && self.tolerance_m > 0.0,
            "invalid_collision_settings"
        );
        let mut names = std::collections::HashSet::new();
        let mut active = Vec::new();
        for c in &self.capsules {
            ensure!(
                names.insert(&c.name)
                    && !c.name.is_empty()
                    && c.start < skeleton.joints.len()
                    && c.end < skeleton.joints.len(),
                "invalid_collision_capsule_joint_or_name"
            );
            if let Some(section) = &c.cross_section {
                ensure!(
                    section.radii_m.iter().all(|r| r.is_finite() && *r > 0.0)
                        && section.lateral_axis_local.iter().all(|v| v.is_finite())
                        && v3(section.lateral_axis_local).norm() > 1e-7,
                    "invalid_capsule_cross_section"
                );
            }
            ensure!(
                c.radius_m.is_finite()
                    && c.radius_m > 0.0
                    && c.interval
                        .iter()
                        .all(|v| v.is_finite() && (-2.0..=3.0).contains(v))
                    && c.interval[0] <= c.interval[1],
                "invalid_collision_capsule_shape"
            );
            if let Some(root) = c.chain_root {
                ensure!(root < skeleton.joints.len(), "collision_root_out_of_bounds");
                active.extend(ancestors(skeleton, c.start, Some(root))?);
                active.extend(ancestors(skeleton, c.end, Some(root))?);
            }
        }
        let mut pairs = std::collections::HashSet::new();
        for &[a, b] in &self.pairs {
            ensure!(
                a < self.capsules.len()
                    && b < self.capsules.len()
                    && a != b
                    && pairs.insert([a.min(b), a.max(b)]),
                "invalid_or_duplicate_collision_pair"
            );
        }
        active.sort_unstable();
        active.dedup();
        Ok(active)
    }

    pub(crate) fn rows(&self, world: &[Matrix4<f64>]) -> Vec<CollisionRow> {
        let segments: Vec<_> = self
            .capsules
            .iter()
            .map(|c| {
                let start = v3(transform_point(&world[c.start], [0.0; 3]));
                let end = v3(transform_point(&world[c.end], [0.0; 3]));
                (
                    start.lerp(&end, c.interval[0]),
                    start.lerp(&end, c.interval[1]),
                )
            })
            .collect();
        self.pairs
            .iter()
            .map(|&[a, b]| {
                let (a0, a1) = segments[a];
                let (b0, b1) = segments[b];
                let (s, t) = closest_segments(a0, a1, b0, b1);
                let delta = a0.lerp(&a1, s) - b0.lerp(&b1, t);
                let distance = delta.norm();
                let normal = delta
                    .try_normalize(1e-10)
                    .or_else(|| (a1 - a0).cross(&(b1 - b0)).try_normalize(1e-10))
                    .or_else(|| ((a0 + a1) - (b0 + b1)).try_normalize(1e-10))
                    .unwrap_or(Vector3::x());
                let radius_sum = self.capsules[a].support_radius(world, normal)
                    + self.capsules[b].support_radius(world, -normal);
                CollisionRow {
                    a,
                    b,
                    fraction_a: self.capsules[a].interval[0]
                        + s * (self.capsules[a].interval[1] - self.capsules[a].interval[0]),
                    fraction_b: self.capsules[b].interval[0]
                        + t * (self.capsules[b].interval[1] - self.capsules[b].interval[0]),
                    normal,
                    depth: (radius_sum + self.margin_m - distance).max(0.0),
                    distance,
                    radius_sum,
                }
            })
            .collect()
    }

    pub fn residuals(
        &self,
        skeleton: &Skeleton,
        pose: &[Transform],
    ) -> Result<Vec<CollisionResidual>> {
        let world = skeleton.world_matrices(pose)?;
        Ok(self
            .rows(&world)
            .into_iter()
            .map(|r| CollisionResidual {
                pair: [
                    self.capsules[r.a].name.clone(),
                    self.capsules[r.b].name.clone(),
                ],
                surface_gap_m: r.distance - r.radius_sum,
                required_gap_m: self.margin_m,
                penetration_m: r.depth,
            })
            .collect())
    }
}

/// Closest parameters of two closed segments, including point capsules.
fn closest_segments(
    p: Vector3<f64>,
    q: Vector3<f64>,
    r: Vector3<f64>,
    s: Vector3<f64>,
) -> (f64, f64) {
    let u = q - p;
    let v = s - r;
    let w = p - r;
    let a = u.dot(&u);
    let b = u.dot(&v);
    let c = v.dot(&v);
    let d = u.dot(&w);
    let e = v.dot(&w);
    if a < 1e-16 && c < 1e-16 {
        return (0.0, 0.0);
    }
    if a < 1e-16 {
        return (0.0, (e / c).clamp(0.0, 1.0));
    }
    if c < 1e-16 {
        return ((-d / a).clamp(0.0, 1.0), 0.0);
    }
    let denom = a * c - b * b;
    let mut first = if denom > 1e-16 {
        ((b * e - c * d) / denom).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut second = (b * first + e) / c;
    if second < 0.0 {
        second = 0.0;
        first = (-d / a).clamp(0.0, 1.0);
    } else if second > 1.0 {
        second = 1.0;
        first = ((b - d) / a).clamp(0.0, 1.0);
    }
    (first, second)
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SurfaceCapsule {
    pub name: String,
    pub start: String,
    pub end: String,
    #[serde(default = "interval")]
    pub interval: [f64; 2],
    #[serde(default)]
    pub chain_root: Option<String>,
    /// Explicit radius, or estimate it from the actual weighted surface.
    #[serde(default)]
    pub radius_m: Option<f64>,
    #[serde(default)]
    pub skin_joints: Vec<String>,
    #[serde(default = "quantile")]
    pub radius_quantile: f64,
    /// Two joints defining a lateral direction in this model's reference pose.
    /// When supplied, fit lateral and front/back widths independently.
    #[serde(default)]
    pub lateral_reference: Option<[String; 2]>,
}
fn quantile() -> f64 {
    0.95
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct BodyClearance {
    pub capsules: Vec<SurfaceCapsule>,
    pub pairs: Vec<[String; 2]>,
    #[serde(default)]
    pub margin_m: f64,
    #[serde(default = "weight")]
    pub weight: f64,
    #[serde(default = "tolerance")]
    pub tolerance_m: f64,
}

impl BodyClearance {
    pub fn prepare(&self, asset: &Asset) -> Result<(CollisionConfig, serde_json::Value)> {
        let skeleton = &asset.skeleton;
        let joint = |name: &str| -> Result<usize> {
            let ids: Vec<_> = skeleton
                .joints
                .iter()
                .enumerate()
                .filter(|(_, j)| j.name == name)
                .map(|(i, _)| i)
                .collect();
            ensure!(ids.len() == 1, "clearance_joint_must_be_unique: {name}");
            Ok(ids[0])
        };
        let points = skeleton.positions(&skeleton.rest_pose())?;
        let surfaces = if self.capsules.iter().any(|c| c.radius_m.is_none()) {
            Some(asset.skinned_positions(&skeleton.rest_pose())?)
        } else {
            None
        };
        let mut capsules = Vec::new();
        let mut fitted = Vec::new();
        for spec in &self.capsules {
            ensure!(
                spec.radius_quantile.is_finite() && (0.5..=1.0).contains(&spec.radius_quantile),
                "invalid_capsule_radius_quantile"
            );
            let start = joint(&spec.start)?;
            let end = joint(&spec.end)?;
            let mut radii = Vec::new();
            let mut lateral_radii = Vec::new();
            let mut depth_radii = Vec::new();
            let lateral_axis = if let Some(names) = &spec.lateral_reference {
                let y = (v3(points[end]) - v3(points[start]))
                    .try_normalize(1e-10)
                    .context("elliptical_capsule_requires_nonzero_segment")?;
                let axis = v3(points[joint(&names[0])?]) - v3(points[joint(&names[1])?]);
                Some(
                    (axis - y * axis.dot(&y))
                        .try_normalize(1e-10)
                        .context("lateral_reference_parallel_to_capsule")?,
                )
            } else {
                None
            };
            let radius = if let Some(radius) = spec.radius_m {
                radius
            } else {
                ensure!(
                    !spec.skin_joints.is_empty(),
                    "surface_capsule_requires_skin_joints"
                );
                let selected: std::collections::HashSet<_> = spec
                    .skin_joints
                    .iter()
                    .map(|name| joint(name))
                    .collect::<Result<_>>()?;
                let a = v3(points[start]);
                let delta = v3(points[end]) - a;
                let length = delta.norm_squared();
                ensure!(length > 1e-12, "automatic_capsule_radius_requires_segment");
                for (mesh, vertices) in asset.meshes.iter().zip(surfaces.as_ref().unwrap()) {
                    for (weights, &point) in mesh.influences.iter().zip(vertices) {
                        if weights
                            .iter()
                            .filter(|w| selected.contains(&w.joint))
                            .map(|w| w.weight)
                            .sum::<f64>()
                            < 0.5
                        {
                            continue;
                        }
                        let offset = v3(point) - a;
                        let t = offset.dot(&delta) / length;
                        if t < spec.interval[0] || t > spec.interval[1] {
                            continue;
                        }
                        let radial = offset - delta * t;
                        radii.push(radial.norm());
                        if let Some(axis) = lateral_axis {
                            lateral_radii.push(radial.dot(&axis).abs());
                            depth_radii.push(radial.dot(&delta.normalize().cross(&axis)).abs());
                        }
                    }
                }
                ensure!(
                    radii.len() >= 8,
                    "insufficient_surface_capsule_samples: {}",
                    spec.name
                );
                radii.sort_by(f64::total_cmp);
                radii[((radii.len() - 1) as f64 * spec.radius_quantile).round() as usize]
            };
            let cross_section = if let Some(axis) = lateral_axis {
                ensure!(
                    spec.radius_m.is_none(),
                    "explicit_radius_cannot_fit_elliptical_section"
                );
                lateral_radii.sort_by(f64::total_cmp);
                depth_radii.sort_by(f64::total_cmp);
                let index = ((radii.len() - 1) as f64 * spec.radius_quantile).round() as usize;
                let world = skeleton.world_matrices(&skeleton.rest_pose())?;
                let local = crate::solve::world_rotation(&world[start])?.inverse() * axis;
                Some(CapsuleCrossSection {
                    lateral_axis_local: a3(local),
                    radii_m: [lateral_radii[index], depth_radii[index]],
                })
            } else {
                None
            };
            fitted.push(serde_json::json!({"name":spec.name,"radius_m":radius,"cross_section":cross_section,"surface_samples":radii.len(),"radius_quantile":spec.radius_quantile,"explicit_radius":spec.radius_m.is_some()}));
            capsules.push(CollisionCapsule {
                name: spec.name.clone(),
                start,
                end,
                interval: spec.interval,
                radius_m: radius,
                cross_section,
                chain_root: spec
                    .chain_root
                    .as_ref()
                    .map(|name| joint(name))
                    .transpose()?,
            });
        }
        let pair_id = |name: &str| {
            capsules
                .iter()
                .position(|c| c.name == name)
                .with_context(|| format!("unknown_collision_capsule: {name}"))
        };
        let pairs = self
            .pairs
            .iter()
            .map(|p| Ok([pair_id(&p[0])?, pair_id(&p[1])?]))
            .collect::<Result<_>>()?;
        let config = CollisionConfig {
            capsules,
            pairs,
            margin_m: self.margin_m,
            weight: self.weight,
            tolerance_m: self.tolerance_m,
        };
        config.validate(skeleton)?;
        Ok((
            config,
            serde_json::json!({"method":"weighted_surface_quantile_capsules_v1","capsules":fitted,"limits":["Capsules and fitted elliptical sections are coarse proxies; the chosen quantile does not bound every surface vertex. Inspect final skin surfaces separately.","Radii are measured from this asset, not copied between character models."]}),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn elliptical_volume_keeps_lateral_width_distinct_from_front_depth() {
        let capsule = CollisionCapsule {
            name: "limb".into(),
            start: 0,
            end: 1,
            interval: [0.0, 1.0],
            radius_m: 0.12,
            chain_root: None,
            cross_section: Some(CapsuleCrossSection {
                lateral_axis_local: [1.0, 0.0, 0.0],
                radii_m: [0.04, 0.12],
            }),
        };
        let mut world = vec![
            Matrix4::identity(),
            Transform::translated([0.0, 1.0, 0.0]).matrix(),
        ];
        assert!((capsule.support_radius(&world, Vector3::x()) - 0.04).abs() < 1e-12);
        assert!((capsule.support_radius(&world, Vector3::z()) - 0.12).abs() < 1e-12);
        let rotation = nalgebra::UnitQuaternion::from_axis_angle(
            &Vector3::y_axis(),
            std::f64::consts::FRAC_PI_2,
        )
        .to_homogeneous();
        for w in &mut world {
            *w = rotation * *w;
        }
        assert!((capsule.support_radius(&world, Vector3::x()) - 0.12).abs() < 1e-12);
        assert!((capsule.support_radius(&world, Vector3::z()) - 0.04).abs() < 1e-12);
    }
    #[test]
    fn radii_are_fitted_from_each_assets_weighted_surface() {
        let joint = |name: &str, parent, position| Joint {
            id: name.into(),
            name: name.into(),
            parent,
            rest: Transform::translated(position),
            matrix: None,
            deform: true,
        };
        for radius in [0.04, 0.12] {
            let positions: Vec<_> = (0..3)
                .flat_map(|row| {
                    (0..32).map(move |i| {
                        let angle = i as f64 * std::f64::consts::TAU / 32.0;
                        [
                            radius * angle.cos(),
                            0.25 + row as f64 * 0.25,
                            radius * angle.sin(),
                        ]
                    })
                })
                .collect();
            let count = positions.len();
            let asset = Asset {
                name: "sized_limb".into(),
                source_hash: String::new(),
                skeleton: Skeleton {
                    joints: vec![
                        joint("start", None, [0.0; 3]),
                        joint("end", Some(0), [0.0, 1.0, 0.0]),
                        joint("other", None, [1.0, 0.0, 0.0]),
                    ],
                },
                meshes: vec![Mesh {
                    name: "surface".into(),
                    node: 0,
                    primitive: 0,
                    positions,
                    triangles: vec![],
                    skin: Some(0),
                    influences: vec![
                        vec![Influence {
                            joint: 0,
                            weight: 1.0
                        }];
                        count
                    ],
                    morph_targets: vec![],
                    morph_weights: vec![],
                }],
                skins: vec![Skin {
                    name: "skin".into(),
                    joints: vec![0],
                    inverse_bind_matrices: vec![matrix_array(&Matrix4::identity())],
                }],
                clips: vec![],
                issues: vec![],
            };
            let config = BodyClearance {
                capsules: vec![
                    SurfaceCapsule {
                        name: "fitted".into(),
                        start: "start".into(),
                        end: "end".into(),
                        interval: [0.1, 0.9],
                        chain_root: Some("start".into()),
                        radius_m: None,
                        skin_joints: vec!["start".into()],
                        radius_quantile: 0.95,
                        lateral_reference: None,
                    },
                    SurfaceCapsule {
                        name: "other".into(),
                        start: "other".into(),
                        end: "other".into(),
                        interval: [0.0, 1.0],
                        chain_root: None,
                        radius_m: Some(0.1),
                        skin_joints: vec![],
                        radius_quantile: 0.95,
                        lateral_reference: None,
                    },
                ],
                pairs: vec![["fitted".into(), "other".into()]],
                margin_m: 0.005,
                weight: 100.0,
                tolerance_m: 0.001,
            };
            let (prepared, report) = config.prepare(&asset).unwrap();
            assert!((prepared.capsules[0].radius_m - radius).abs() < 1e-10);
            assert_eq!(report["capsules"][0]["surface_samples"], 96);
        }
    }
    #[test]
    fn segment_closest_points_cover_crossing_parallel_and_point_cases() {
        let p = Vector3::new;
        let cases = [
            (
                p(-1., 0., 0.),
                p(1., 0., 0.),
                p(0., -1., 0.3),
                p(0., 1., 0.3),
                0.3,
            ),
            (
                p(0., 0., 0.),
                p(1., 0., 0.),
                p(0.3, 0.2, 0.),
                p(0.7, 0.2, 0.),
                0.2,
            ),
            (
                p(0., 0., 0.),
                p(0., 0., 0.),
                p(-1., 1., 0.),
                p(1., 1., 0.),
                1.,
            ),
        ];
        for (a, b, c, d, expected) in cases {
            let (s, t) = closest_segments(a, b, c, d);
            assert!(((a.lerp(&b, s) - c.lerp(&d, t)).norm() - expected).abs() < 1e-10);
        }
    }
}
