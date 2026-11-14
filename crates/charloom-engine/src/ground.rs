//! Explicit surface floor projection. The default raises only; an authored
//! walking alignment can also lower the root. Horizontal contacts are separate.
use anyhow::{Context, Result, ensure};
use charloom_core::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SurfaceFloor {
    pub root: String,
    /// Exact skin joint names defining the surface to protect from the plane.
    pub support_joints: Vec<String>,
    pub height_m: f64,
    pub max_lift_m: f64,
    /// After collision solving, keep the source frame's support height. This
    /// allows a bounded lowering as well as raising, preserving airborne phases.
    #[serde(default)]
    pub preserve_source_height: bool,
    /// Explicit walking preview correction. Also lowers floating support to
    /// the plane; leave disabled for jumping or deliberately airborne motion.
    #[serde(default)]
    pub align_to_ground: bool,
}

pub struct PreparedFloor {
    root: usize,
    samples: Vec<(usize, usize)>,
    height: f64,
    max_lift: f64,
    align_to_ground: bool,
}

fn unique(skeleton: &Skeleton, name: &str) -> Result<usize> {
    let ids: Vec<_> = skeleton
        .joints
        .iter()
        .enumerate()
        .filter(|(_, j)| j.name == name)
        .map(|(i, _)| i)
        .collect();
    ensure!(ids.len() == 1, "floor_joint_must_be_unique: {name}");
    Ok(ids[0])
}

impl PreparedFloor {
    pub fn prepare(asset: &Asset, config: &SurfaceFloor) -> Result<Self> {
        ensure!(
            config.height_m.is_finite() && config.max_lift_m.is_finite() && config.max_lift_m > 0.0,
            "invalid_surface_floor"
        );
        ensure!(
            !config.support_joints.is_empty() && config.support_joints.len() <= 64,
            "invalid_floor_support_joints"
        );
        ensure!(
            !config.align_to_ground || !config.preserve_source_height,
            "conflicting_floor_height_modes"
        );
        let root = unique(&asset.skeleton, &config.root)?;
        ensure!(
            asset.skeleton.joints[root].matrix.is_none(),
            "surface_floor_root_has_static_matrix"
        );
        let supports: HashSet<_> = config
            .support_joints
            .iter()
            .map(|n| unique(&asset.skeleton, n))
            .collect::<Result<_>>()?;
        let descendant = |mut i: usize| -> bool {
            loop {
                if i == root {
                    return true;
                }
                match asset.skeleton.joints[i].parent {
                    Some(p) => i = p,
                    None => return false,
                }
            }
        };
        ensure!(
            supports.iter().all(|&i| descendant(i)),
            "floor_support_outside_root"
        );
        let mut samples = Vec::new();
        for (m, mesh) in asset.meshes.iter().enumerate() {
            if mesh.skin.is_none() {
                continue;
            }
            for (v, weights) in mesh.influences.iter().enumerate() {
                if weights
                    .iter()
                    .filter(|w| supports.contains(&w.joint))
                    .map(|w| w.weight)
                    .sum::<f64>()
                    >= 0.5
                {
                    ensure!(
                        weights
                            .iter()
                            .all(|w| w.weight == 0.0 || descendant(w.joint)),
                        "floor_vertex_spans_independent_roots"
                    );
                    samples.push((m, v));
                }
            }
        }
        ensure!(!samples.is_empty(), "surface_floor_has_no_weighted_samples");
        Ok(Self {
            root,
            samples,
            height: config.height_m,
            max_lift: config.max_lift_m,
            align_to_ground: config.align_to_ground,
        })
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    pub fn minimum_y(
        &self,
        asset: &Asset,
        pose: &[Transform],
        morphs: &HashMap<usize, Vec<f64>>,
    ) -> Result<f64> {
        let world = asset.skeleton.world_matrices(pose)?;
        let matrices: Vec<HashMap<_, _>> = asset
            .skins
            .iter()
            .map(|skin| {
                skin.joints
                    .iter()
                    .zip(&skin.inverse_bind_matrices)
                    .map(|(&j, ibm)| (j, world[j] * matrix_from(*ibm)))
                    .collect()
            })
            .collect();
        let mut minimum = f64::INFINITY;
        for &(m, v) in &self.samples {
            let mesh = &asset.meshes[m];
            let mut point = v3(mesh.positions[v]);
            let weights = morphs.get(&mesh.node).unwrap_or(&mesh.morph_weights);
            ensure!(
                weights.len() == mesh.morph_targets.len(),
                "floor_morph_dimensions_mismatch"
            );
            for (target, &w) in mesh.morph_targets.iter().zip(weights) {
                point += v3(target[v]) * w;
            }
            let skin = &matrices[mesh.skin.context("floor_sample_without_skin")?];
            let y = mesh.influences[v]
                .iter()
                .map(|w| transform_point(&skin[&w.joint], a3(point))[1] * w.weight)
                .sum::<f64>();
            ensure!(y.is_finite(), "nonfinite_floor_surface");
            minimum = minimum.min(y);
        }
        Ok(minimum)
    }

    pub fn apply(
        &self,
        asset: &Asset,
        pose: &mut [Transform],
        morphs: &HashMap<usize, Vec<f64>>,
    ) -> Result<f64> {
        let minimum = self.minimum_y(asset, pose, morphs)?;
        let delta = self.height - minimum;
        let lift = if self.align_to_ground {
            delta
        } else {
            delta.max(0.0)
        };
        self.translate(asset, pose, lift)
    }

    pub fn restore_height(
        &self,
        asset: &Asset,
        pose: &mut [Transform],
        morphs: &HashMap<usize, Vec<f64>>,
        height: f64,
    ) -> Result<f64> {
        let current = self.minimum_y(asset, pose, morphs)?;
        self.translate(asset, pose, height.max(self.height) - current)
    }

    fn translate(&self, asset: &Asset, pose: &mut [Transform], lift: f64) -> Result<f64> {
        ensure!(
            lift.abs() <= self.max_lift,
            "surface_floor_lift_exceeded: {lift} m exceeds {} m",
            self.max_lift
        );
        if lift != 0.0 {
            let world = asset.skeleton.world_matrices(pose)?;
            let mut position = transform_point(&world[self.root], [0.0; 3]);
            position[1] += lift;
            pose[self.root].translation =
                if let Some(parent) = asset.skeleton.joints[self.root].parent {
                    transform_point(
                        &world[parent]
                            .try_inverse()
                            .context("singular_floor_root_parent")?,
                        position,
                    )
                } else {
                    position
                };
        }
        Ok(lift)
    }
}
