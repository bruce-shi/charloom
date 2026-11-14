//! Deterministic clip layers and explicit contact constraints, baked to a GLB.
use crate::{
    attachment::{Attachment, PreparedAttachment, PropAsset},
    ground::{PreparedFloor, SurfaceFloor},
    motion::{RetargetConfig, clip_from_poses, retarget, sample_times},
    pipeline::{publish_bundle, select_clip},
    solve::{Effector, SolveConfig, solve_pose},
};
use anyhow::{Context, Result, ensure};
use charloom_core::*;
use charloom_io::{load_gltf, load_motion_asset, write_json};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct AnimationRecipe {
    pub schema_version: String,
    pub asset: PathBuf,
    pub name: String,
    pub duration: f64,
    #[serde(default = "fps")]
    pub fps: f64,
    pub layers: Vec<Layer>,
    #[serde(default)]
    pub contacts: Vec<Contact>,
    /// Explicit skin-region floor protection; raises the root only as needed.
    #[serde(default)]
    pub surface_floor: Option<SurfaceFloor>,
    #[serde(default)]
    pub props: Vec<PropAsset>,
    #[serde(default)]
    pub attachments: Vec<Attachment>,
    /// Exact child-joint names whose axial rotation is distributed to their
    /// direct parent after all constraints. The child's world pose is kept.
    #[serde(default)]
    pub distribute_twist: Vec<String>,
         #[serde(default)]
    pub body_clearance: Option<crate::collision::BodyClearance>,
    /// Target digit lengths plus a bound on new skin-edge elongation.
    #[serde(default)]
    pub hand_shapes: Vec<crate::hand_shape::HandShapeConstraint>,
    #[serde(default)]
    pub arm_articulations: Vec<crate::articulation::ArmArticulation>,
    #[serde(default)]
    pub rigid_regions: Vec<crate::rigid_region::RigidBoneRegion>,
    #[serde(default)]
    pub temporal_correction: Option<TemporalCorrection>,
    #[serde(default)]
    pub solver: SolveConfig,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct TemporalCorrection {
    pub joints: Vec<String>,
    pub decay_seconds: f64,
}

fn fps() -> f64 {
    30.0
}

fn contact_active(contact: &Contact, time: f64) -> bool {
    // Baked glTF keys store f32 seconds. Different f64 arithmetic paths for
    // 1/6s must not drop a planted contact at the first or last emitted key.
    let time = time as f32;
    time >= contact.start as f32 && time <= contact.end as f32
}

fn recipe_sample_times(recipe: &AnimationRecipe) -> Result<Vec<f64>> {
    let mut times = sample_times(0.0, recipe.duration, recipe.fps)?;
    for contact in &recipe.contacts {
        times.extend([contact.start, contact.end]);
    }
    for attachment in &recipe.attachments {
        times.extend([attachment.start, attachment.end]);
    }
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| {
        if (*a as f32) == (*b as f32) {
            *b = *a;
            true
        } else {
            false
        }
    });
    Ok(times)
}
fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    /// If omitted, use the target's own animation. Paths are relative to the recipe.
    #[serde(default)]
    pub source: Option<PathBuf>,
    pub clip: String,
    #[serde(default)]
    pub timeline_start: f64,
    #[serde(default)]
    pub source_start: Option<f64>,
    #[serde(default)]
    pub source_end: Option<f64>,
    #[serde(default = "one")]
    pub speed: f64,
    #[serde(default = "one")]
    pub weight: f64,
    #[serde(default)]
    pub fade_in: f64,
    #[serde(default)]
    pub fade_out: f64,
    #[serde(default)]
    pub repeat: bool,
    #[serde(default)]
    pub additive: bool,
    /// Empty means all animation channels. Otherwise exact, unique node names.
    #[serde(default)]
    pub joints: Vec<String>,
    #[serde(default)]
    pub retarget: RetargetConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Contact {
    pub joint: String,
    pub start: f64,
    pub end: f64,
    /// Omission locks to the first baked contact sample, recorded in the report.
    #[serde(default)]
    pub position: Option<Vec3>,
    #[serde(default = "all_contact_position_axes")]
    pub position_axes: [bool; 3],
    #[serde(default)]
    pub rotation: Option<Quat>,
    #[serde(default)]
    pub chain_root: Option<String>,
    #[serde(default)]
    pub pole: Option<Vec3>,
}
fn all_contact_position_axes() -> [bool; 3] {
    [true; 3]
}

fn resolve(base: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}
fn joint_id(s: &Skeleton, name: &str) -> Result<usize> {
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

struct Prepared {
    clip: MotionClip,
    start: f64,
    end: f64,
    mask: HashSet<usize>,
}

fn layer_sample(layer: &Layer, p: &Prepared, time: f64) -> Option<(f64, f64)> {
    let elapsed = time - layer.timeline_start;
    if elapsed < -1e-10 {
        return None;
    }
    let duration = (p.end - p.start) / layer.speed;
    // glTF time accessors are f32. A requested 0.7s endpoint can therefore
    // read back as 0.699999988s; dropping the layer there resets root motion.
    // Admit only the source time's storage precision, then clamp to its end.
    let time_tolerance =
        f64::from(f32::EPSILON) * p.start.abs().max(p.end.abs()).max(1.0) / layer.speed;
    if !layer.repeat && elapsed > duration + time_tolerance {
        return None;
    }
    let source_time = p.start
        + if layer.repeat {
            (elapsed * layer.speed).rem_euclid(p.end - p.start)
        } else {
            (elapsed * layer.speed).clamp(0.0, p.end - p.start)
        };
    let mut weight = layer.weight;
    if layer.fade_in > 0.0 {
        weight *= (elapsed / layer.fade_in).clamp(0.0, 1.0);
    }
    if !layer.repeat && layer.fade_out > 0.0 {
        weight *= ((duration - elapsed) / layer.fade_out).clamp(0.0, 1.0);
    }
    Some((source_time, weight))
}

fn layer_pose(
    skeleton: &Skeleton,
    layers: &[Layer],
    prepared: &[Prepared],
    time: f64,
) -> Result<Vec<Transform>> {
    let mut pose = skeleton.rest_pose();
    for (layer, p) in layers.iter().zip(prepared) {
        let Some((source_time, weight)) = layer_sample(layer, p, time) else {
            continue;
        };
        let sample = p.clip.sample(skeleton, source_time)?;
        // Blend only properties carried by the clip; unrelated channels stay intact.
        for track in &p.clip.tracks {
            let i = track.joint;
            if !p.mask.contains(&i) {
                continue;
            }
            let reference = skeleton.joints[i].rest;
            match track.property {
                Property::Rotation => {
                    let rotation = if layer.additive {
                        unit_quat(pose[i].rotation)
                            * (unit_quat(reference.rotation).inverse()
                                * unit_quat(sample[i].rotation))
                            .powf(weight)
                    } else {
                        unit_quat(pose[i].rotation).slerp(&unit_quat(sample[i].rotation), weight)
                    };
                    pose[i].rotation = a4(rotation);
                }
                Property::Translation => {
                    pose[i].translation = if layer.additive {
                        a3(v3(pose[i].translation)
                            + (v3(sample[i].translation) - v3(reference.translation)) * weight)
                    } else {
                        a3(v3(pose[i].translation) * (1.0 - weight)
                            + v3(sample[i].translation) * weight)
                    }
                }
                Property::Scale => {
                    pose[i].scale = std::array::from_fn(|a| {
                        if layer.additive {
                            pose[i].scale[a]
                                * (1.0 + (sample[i].scale[a] / reference.scale[a] - 1.0) * weight)
                        } else {
                            pose[i].scale[a] * (1.0 - weight) + sample[i].scale[a] * weight
                        }
                    })
                }
            }
        }
    }
    Ok(pose)
}

pub fn build(recipe_path: &Path, out: &Path) -> Result<serde_json::Value> {
    ensure!(!out.exists(), "output_exists");
    let mut recipe: AnimationRecipe = serde_json::from_slice(&std::fs::read(recipe_path)?)?;
    ensure!(
        recipe.schema_version == SCHEMA_VERSION,
        "unsupported_recipe_schema"
    );
    ensure!(
        !recipe.layers.is_empty() && recipe.layers.len() <= 64,
        "recipe must contain 1..64 layers"
    );
    ensure!(
        recipe.contacts.len() <= 32,
        "at most 32 simultaneous contacts"
    );
    let base = recipe_path.parent().unwrap_or(Path::new("."));
    recipe.asset = resolve(base, &recipe.asset).canonicalize()?;
    for layer in &mut recipe.layers {
        if let Some(source) = &layer.source {
            layer.source = Some(resolve(base, source).canonicalize()?);
        }
    }
    let mut asset = load_gltf(resolve(base, &recipe.asset))?;
    let original_hash = asset.asset.source_hash.clone();
    ensure!(recipe.props.len() <= 16, "at_most_16_external_props");
    let mut prop_sources = Vec::new();
    for prop in &mut recipe.props {
        prop.path = resolve(base, &prop.path).canonicalize()?;
        let loaded = load_gltf(&prop.path)?;
        let node = asset
            .document
            .append_rigid_prop(&loaded, &prop.name, prop.transform)?;
        prop_sources.push(json!({"path":prop.path,"name":prop.name,"root_node":node,"source_hash":loaded.asset.source_hash,"issues":loaded.asset.issues}));
    }
    if !recipe.props.is_empty() {
        let temporary = tempfile::tempdir()?;
        let assembled = temporary.path().join("assembled.glb");
        asset.document.save(&assembled)?;
        asset = load_gltf(&assembled)?;
    }
    ensure!(
        !asset.asset.clips.iter().any(|c| c.name == recipe.name),
        "animation_name_exists"
    );
    let rigid_regions =
        crate::rigid_region::apply_regions(&mut asset.asset, &recipe.rigid_regions)?;
    let digit_binding = recipe
        .hand_shapes
        .iter()
        .map(|c| crate::hand_shape::rebind_missing_digits(&mut asset.asset, c))
        .collect::<Result<Vec<_>>>()?;
    let mut hand_surface_repair = recipe
        .hand_shapes
        .iter()
        .map(|c| {
            if c.allow_rotation_fallback {
                crate::hand_surface::repair(&mut asset.asset, c)
            } else {
                Ok(json!({"deferred_until_final_pose":true,"changed_vertices":0}))
            }
        })
        .collect::<Result<Vec<_>>>()?;
    if hand_surface_repair
        .iter()
        .any(|r| r["changed_vertices"].as_u64().unwrap_or(0) > 0)
        || rigid_regions
            .iter()
            .any(|r| r["changed_vertices"].as_u64().unwrap_or(0) > 0)
        || digit_binding
            .iter()
            .any(|r| r["changed_vertices"].as_u64().unwrap_or(0) > 0)
    {
        crate::skin_repair::write_weights(&mut asset.document, &asset.asset)?;
    }
    let skeleton = &asset.asset.skeleton;
    let mut solve_config = recipe.solver.clone();
    for arm in &recipe.arm_articulations {
        solve_config.articulations.extend(arm.prepare(skeleton)?);
    }
    let temporal_joints = if let Some(c) = &recipe.temporal_correction {
        ensure!(
            c.decay_seconds.is_finite() && c.decay_seconds > 0.0 && c.decay_seconds <= 10.0,
            "invalid_temporal_decay"
        );
        c.joints
            .iter()
            .map(|n| joint_id(skeleton, n))
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let mut previous_correction: Option<(f64, Vec<nalgebra::UnitQuaternion<f64>>)> = None;
    let clearance_preparation = if let Some(clearance) = &recipe.body_clearance {
        ensure!(
            solve_config.collision.is_none(),
            "choose_named_body_clearance_or_numeric_solver_collision"
        );
        let (collision, report) = clearance.prepare(&asset.asset)?;
        solve_config.collision = Some(collision);
        Some(report)
    } else {
        None
    };
    let mut twist_distribution =
        crate::articulation::PreparedTwist::prepare_named(skeleton, &recipe.distribute_twist)?;
    for twist in &mut twist_distribution {
        twist.set_reference(&solve_config.articulations);
    }
    ensure!(
        recipe.hand_shapes.len() <= 16,
        "too_many_hand_shape_constraints"
    );
    let hand_shapes = recipe
        .hand_shapes
        .iter()
        .map(|c| crate::hand_shape::PreparedHandShape::prepare(&asset.asset, c))
        .collect::<Result<Vec<_>>>()?;
    let floor = recipe
        .surface_floor
        .as_ref()
        .map(|f| PreparedFloor::prepare(&asset.asset, f))
        .transpose()?;
    let times = recipe_sample_times(&recipe)?;
    let mut attachments =
        PreparedAttachment::prepare(&asset.asset, &recipe.attachments, recipe.duration)?;
    let attachment_calibration: Vec<_> = attachments
        .iter()
        .map(PreparedAttachment::calibration)
        .collect();
    let mut prepared = Vec::new();
    let mut sources = Vec::new();
    for layer in &recipe.layers {
        ensure!(
            [
                layer.timeline_start,
                layer.speed,
                layer.weight,
                layer.fade_in,
                layer.fade_out
            ]
            .iter()
            .all(|v| v.is_finite()),
            "non-finite layer setting"
        );
        ensure!(
            layer.speed > 0.0
                && (0.0..=1.0).contains(&layer.weight)
                && layer.timeline_start >= 0.0
                && layer.fade_in >= 0.0
                && layer.fade_out >= 0.0,
            "invalid layer setting"
        );
        let clip = if let Some(source) = &layer.source {
            let input = load_motion_asset(resolve(base, source))?;
            let source_clip = select_clip(&input, &layer.clip)?;
            let mut config = layer.retarget.clone();
            config.start = layer.source_start;
            config.end = layer.source_end;
            config.fps = recipe.fps;
            let (motion, report) = retarget(&input.skeleton, source_clip, skeleton, &config)?;
            sources.push(json!({"path":source,"hash":input.source_hash,"clip":layer.clip,"retarget":report,"issues":input.issues}));
            motion
        } else {
            sources.push(json!({"path":recipe.asset,"hash":asset.asset.source_hash,"clip":layer.clip,"retarget":false}));
            select_clip(&asset.asset, &layer.clip)?.clone()
        };
        let range = clip.time_range();
        let start = if layer.source.is_some() {
            range[0]
        } else {
            layer.source_start.unwrap_or(range[0])
        };
        let end = if layer.source.is_some() {
            range[1]
        } else {
            layer.source_end.unwrap_or(range[1])
        };
        ensure!(
            start.is_finite()
                && end.is_finite()
                && end > start
                && start >= range[0]
                && end <= range[1] + 1e-6,
            "invalid layer clip range"
        );
        let mask = if layer.joints.is_empty() {
            clip.tracks
                .iter()
                .map(|t| t.joint)
                .chain(clip.morph_tracks.iter().map(|t| t.node))
                .collect()
        } else {
            layer
                .joints
                .iter()
                .map(|n| joint_id(skeleton, n))
                .collect::<Result<_>>()?
        };
        prepared.push(Prepared {
            clip,
            start,
            end,
            mask,
        });
    }
    let mut anchors = Vec::new();
    let mut resolved_contacts = Vec::new();
    for contact in &recipe.contacts {
        ensure!(
            contact.start.is_finite()
                && contact.end.is_finite()
                && contact.start >= 0.0
                && contact.end >= contact.start
                && contact.end <= recipe.duration,
            "invalid contact interval"
        );
        let joint = joint_id(skeleton, &contact.joint)?;
        let chain_root = contact
            .chain_root
            .as_ref()
            .map(|n| joint_id(skeleton, n))
            .transpose()?;
        ensure!(
            times.iter().any(|t| contact_active(contact, *t)),
            "contact has no baked sample"
        );
        anchors.push(contact.position);
        resolved_contacts.push((joint, chain_root));
    }
    let mut poses = Vec::new();
    let mut max_residual = 0.0_f64;
    let mut max_full_residual = 0.0_f64;
    let mut failed_frames = Vec::new();
    let mut anchor_samples = Vec::new();
    let mut grip_frames = Vec::new();
    let mut floor_frames = Vec::new();
    let mut twist_frames = Vec::new();
    let mut hand_shape_frames = Vec::new();
    let mut clearance_frames = Vec::new();
    let morph_defaults: HashMap<_, _> = asset
        .asset
        .meshes
        .iter()
        .filter(|m| !m.morph_weights.is_empty())
        .map(|m| (m.node, m.morph_weights.clone()))
        .collect();
    let mut morph_samples: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();
    for p in &prepared {
        for track in &p.clip.morph_tracks {
            if p.mask.contains(&track.node) {
                morph_samples.entry(track.node).or_default();
            }
        }
    }
    for (frame, &time) in times.iter().enumerate() {
        let mut pose = layer_pose(skeleton, &recipe.layers, &prepared, time)?;
        let mut morphs = morph_defaults.clone();
        for (layer, p) in recipe.layers.iter().zip(&prepared) {
            let Some((source_time, weight)) = layer_sample(layer, p, time) else {
                continue;
            };
            for track in &p.clip.morph_tracks {
                if !p.mask.contains(&track.node) {
                    continue;
                }
                let sample = track.sample(source_time)?;
                let values = morphs
                    .get_mut(&track.node)
                    .context("morph layer targets missing geometry")?;
                let reference = &morph_defaults[&track.node];
                ensure!(
                    values.len() == sample.len(),
                    "morph layer target dimensions disagree"
                );
                for i in 0..values.len() {
                    values[i] = if layer.additive {
                        values[i] + (sample[i] - reference[i]) * weight
                    } else {
                        values[i] * (1.0 - weight) + sample[i] * weight
                    };
                }
            }
        }
        for (&node, samples) in &mut morph_samples {
            samples.push(
                morphs
                    .get(&node)
                    .context("morph node has no defaults")?
                    .clone(),
            );
        }
        let mut floor_lift = if let Some(floor) = &floor {
            floor.apply(&asset.asset, &mut pose, &morphs)?
        } else {
            0.0
        };
        let source_surface_y = if recipe
            .surface_floor
            .as_ref()
            .is_some_and(|f| f.preserve_source_height)
        {
            Some(
                floor
                    .as_ref()
                    .unwrap()
                    .minimum_y(&asset.asset, &pose, &morphs)?,
            )
        } else {
            None
        };
        let positions = skeleton.positions(&pose)?;
        let mut effectors = Vec::new();
        for (index, contact) in recipe.contacts.iter().enumerate() {
            if !contact_active(contact, time) {
                continue;
            }
            let (joint, chain_root) = resolved_contacts[index];
            if anchors[index].is_none() {
                anchors[index] = Some(positions[joint]);
                anchor_samples
                    .push(json!({"contact":index,"time_s":time,"position":positions[joint]}));
            }
            effectors.push(Effector {
                joint,
                position: anchors[index].context("missing contact anchor")?,
                position_weight: 1.0,
                position_axes: contact.position_axes,
                rotation: contact.rotation,
                rotation_weight: 1.0,
                chain_root,
                pole: contact.pole,
            });
        }
         
        let source_rotations: Vec<_> = temporal_joints
            .iter()
            .map(|&j| unit_quat(pose[j].rotation))
            .collect();
        if !solve_config.articulations.is_empty() {
            // Keep the authored grip trajectory as the target. Capturing it
            // after collision/joint projection turns their arbitrary IK
            // branch into a new target and can cause frame-to-frame jumps.
            for attachment in &mut attachments {
                attachment.capture_source_driver(skeleton, &pose)?;
            }
            for d in &twist_distribution {
                d.apply(skeleton, &mut pose)?;
            }
            for limit in &solve_config.articulations {
                limit.distribute_bend_plane(skeleton, &mut pose)?;
            }
            crate::solve::project_limits(skeleton, &mut pose, &solve_config);
        }

        if let Some((last, corrections)) = &previous_correction {
            let retain =
                (-(time - last) / recipe.temporal_correction.as_ref().unwrap().decay_seconds).exp();
            for (i, &j) in temporal_joints.iter().enumerate() {
                pose[j].rotation = a4(nalgebra::UnitQuaternion::identity()
                    .slerp(&corrections[i], retain)
                    * source_rotations[i]);
            }
        }
        let collision_before = solve_config
            .collision
            .as_ref()
            .map(|c| c.residuals(skeleton, &pose))
            .transpose()?;
        if !effectors.is_empty() || solve_config.collision.is_some() {
            let result = solve_pose(skeleton, &pose, &effectors, &solve_config)?;
            pose = result.pose;
        }
        if solve_config.collision.is_some()
            && let Some(floor) = &floor
        {
            floor_lift += if let Some(y) = source_surface_y {
                floor.restore_height(&asset.asset, &mut pose, &morphs, y)?
            } else {
                floor.apply(&asset.asset, &mut pose, &morphs)?
            };
        }
        let grips = PreparedAttachment::apply_all(
            &mut attachments,
            skeleton,
            &mut pose,
            time,
            &solve_config,
        )?;
        if !grips.is_empty() {
            grip_frames.push(json!({"frame":frame,"time_s":time,"attachments":grips}));
        }
        if !twist_distribution.is_empty() && solve_config.articulations.is_empty() {
            let reports = twist_distribution
                .iter()
                .map(|d| d.apply(skeleton, &mut pose))
                .collect::<Result<Vec<_>>>()?;
            twist_frames.push(json!({"frame":frame,"time_s":time,"joints":reports}));
        }
        if !hand_shapes.is_empty() {
            let hands = hand_shapes
                .iter()
                .map(|c| c.apply(&asset.asset, &mut pose, &morphs))
                .collect::<Result<Vec<_>>>()?;
            hand_shape_frames.push(json!({"frame":frame,"time_s":time,"hands":hands}));
        }
        if !temporal_joints.is_empty() {
            previous_correction = Some((
                time,
                temporal_joints
                    .iter()
                    .enumerate()
                    .map(|(i, &j)| unit_quat(pose[j].rotation) * source_rotations[i].inverse())
                    .collect(),
            ));
        }
        if let Some(collision) = &solve_config.collision {
            let final_residuals = collision.residuals(skeleton, &pose)?;
            clearance_frames.push(json!({"frame":frame,"time_s":time,"before":collision_before,"after":final_residuals,"status":if final_residuals.iter().all(|r|r.penetration_m<=collision.tolerance_m){"converged"}else{"not_converged"}}));
        }
        if let Some(floor) = &floor {
            // Contacts, release smoothing and grips can lower a sole after the
            // initial floor pass. Protect the actual exported pose, then report
            // any resulting contact conflict below rather than hiding it.
            let final_lift = if let Some(y) = source_surface_y {
                floor.restore_height(&asset.asset, &mut pose, &morphs, y)?
            } else {
                floor.apply(&asset.asset, &mut pose, &morphs)?
            };
            floor_lift += final_lift;
            floor_frames.push(json!({"frame":frame,"time_s":time,"root_lift_m":floor_lift,"post_constraint_root_lift_m":final_lift,"final_minimum_surface_y_m":floor.minimum_y(&asset.asset,&pose,&morphs)?}));
        }
        // Evaluate contacts again after grips. A conflicting grip must not hide
        // a hand/foot contact that became unsatisfied in the final exported pose.
        let final_residuals = crate::solve::residuals(skeleton, &pose, &effectors)?;
        max_full_residual = max_full_residual.max(
            final_residuals
                .iter()
                .map(|r| r.full_position_error_m)
                .fold(0.0, f64::max),
        );
        max_residual = max_residual.max(
            final_residuals
                .iter()
                .map(|r| r.position_error_m)
                .fold(0.0, f64::max),
        );
        if final_residuals.iter().any(|r| {
            r.position_error_m > recipe.solver.tolerance_m
                || r.rotation_error_rad
                    .is_some_and(|a| a > recipe.solver.rotation_tolerance_rad)
        }) {
            failed_frames.push(json!({"frame":frame,"time_s":time,"residuals":final_residuals}));
        }
        poses.push(pose);
    }
    let joints: Vec<_> = skeleton
        .joints
        .iter()
        .enumerate()
        .filter(|(_, j)| j.matrix.is_none())
        .map(|(i, _)| i)
        .collect();
    let mut clip = clip_from_poses(&recipe.name, &times, &poses, &joints)?;
    let mut morph_nodes: Vec<_> = morph_samples.keys().copied().collect();
    morph_nodes.sort_unstable();
    for node in morph_nodes {
        clip.morph_tracks.push(MorphTrack {
            node,
            interpolation: Interpolation::Linear,
            times: times.clone(),
            values: morph_samples.remove(&node).unwrap(),
        });
    }
    clip.validate(skeleton)?;
    // A grip/clearance solve changes the wrist relative to the forearm.
    // Fit immutable-grip skin against those final poses, not a stale input clip.
    let final_hands: Vec<_> = recipe
        .hand_shapes
        .iter()
        .enumerate()
        .filter(|(_, h)| !h.allow_rotation_fallback && h.surface_repair.is_some())
        .collect();
    if !final_hands.is_empty() {
        asset.asset.clips.push(clip.clone());
        for &(index, hand) in &final_hands {
            let mut hand = hand.clone();
            let repair = hand.surface_repair.as_mut().unwrap();
            if !repair.reference_clips.contains(&clip.name) {
                repair.reference_clips.push(clip.name.clone());
            }
            hand_surface_repair[index] = crate::hand_surface::repair(&mut asset.asset, &hand)?;
            hand_surface_repair[index]["includes_final_solved_clip"] = json!(true);
        }
        crate::skin_repair::write_weights(&mut asset.document, &asset.asset)?;
        for &(index, hand) in &final_hands {
            let guard = crate::hand_shape::PreparedHandShape::prepare(&asset.asset, hand)?;
            for (frame, (&time, pose)) in times.iter().zip(&poses).enumerate() {
                let mut checked = pose.clone();
                let morphs = clip.sample_morphs(time)?;
                let report = guard.apply(&asset.asset, &mut checked, &morphs)?;
                ensure!(checked == *pose, "fixed_grip_surface_pass_changed_pose");
                hand_shape_frames[frame]["hands"][index] = serde_json::to_value(report)?;
            }
        }
        if let Some(floor) = &floor {
            for (frame, (&time, pose)) in times.iter().zip(&poses).enumerate() {
                floor_frames[frame]["final_minimum_surface_y_m"] =
                    json!(floor.minimum_y(&asset.asset, pose, &clip.sample_morphs(time)?)?);
            }
        }
    }
    let mut hand_surface_correctives = Vec::new();
    for hand in &recipe.hand_shapes {
        hand_surface_correctives.push(crate::hand_corrective::apply(
            &mut asset.asset,
            &mut asset.document,
            &mut clip,
            hand,
        )?);
    }
    if recipe
        .hand_shapes
        .iter()
        .any(|h| h.surface_corrective.is_some())
    {
        for (index, hand) in recipe.hand_shapes.iter().enumerate() {
            let guard = crate::hand_shape::PreparedHandShape::prepare(&asset.asset, hand)?;
            for (frame, (&time, pose)) in times.iter().zip(&poses).enumerate() {
                // A compact hand mesh may add a nondeforming scene node.
                let mut checked = asset.asset.skeleton.rest_pose();
                checked[..pose.len()].copy_from_slice(pose);
                let before_check = checked.clone();
                let report = guard.apply(&asset.asset, &mut checked, &clip.sample_morphs(time)?)?;
                ensure!(
                    checked == before_check,
                    "surface_corrective_changed_joint_pose"
                );
                hand_shape_frames[frame]["hands"][index] = serde_json::to_value(report)?;
            }
        }
    }
    asset.document.append_animation(&clip, 0)?;
    let grip_nonconverged_frames: Vec<_> = grip_frames
        .iter()
        .filter(|frame| {
            frame["attachments"]
                .as_array()
                .unwrap()
                .iter()
                .any(|a| a["status"] != "converged")
        })
        .cloned()
        .collect();
    let max_grip_error = grip_frames
        .iter()
        .flat_map(|frame| frame["attachments"].as_array().unwrap())
        .flat_map(|a| a["followers"].as_array().unwrap())
        .filter_map(|f| f["position_error_m"].as_f64())
        .fold(0.0, f64::max);
    let report = json!({"operation":"animation_build","source_hash":original_hash,"quality_status":"needs_review","visual_review":"pending",
        "sample_count":times.len(),"sources":sources,"max_contact_residual_m":max_residual,"max_contact_full_position_residual_m":max_full_residual,"nonconverged_frames":failed_frames,
        "contact_anchor_samples":anchor_samples,"contact_position_axes":recipe.contacts.iter().map(|c|json!({"joint":c.joint,"start":c.start,"end":c.end,"position_axes":c.position_axes})).collect::<Vec<_>>(),"source_issues":asset.asset.issues,
        "floor_sample_vertices":floor.as_ref().map_or(0,PreparedFloor::sample_count),"floor_frames":floor_frames,
        "twist_distribution_frames":twist_frames,
        "hand_surface_correctives":hand_surface_correctives,"hand_shape_frames":hand_shape_frames,"hand_surface_repair":hand_surface_repair,
        "digit_binding":digit_binding,
        "rigid_regions":rigid_regions,"articulation_limits":solve_config.articulations,
        "body_clearance_preparation":clearance_preparation,"body_clearance_config":solve_config.collision,"body_clearance_frames":clearance_frames,
        "props":prop_sources,"attachment_calibration":attachment_calibration,"grip_frames":grip_frames,"max_grip_residual_m":max_grip_error,"grip_nonconverged_frames":grip_nonconverged_frames,
        "limits":["Repeat wraps the source clip; it does not synthesize a seamless cycle.","Contacts are explicitly supplied and are not automatically detected.","Surface-floor projection is an explicit vertical root lift based on selected weighted vertices. It does not detect contacts or lock horizontal feet; final surface height is reported after other constraints.","Native target morph channels are layered; morph targets are not retargeted across different characters.","Grips use explicit or calibrated frames, not automatic grasp geometry. Coupled grip graphs are rejected. Release holds the world pose or resumes source animation, without physics.","Grips are applied after contacts; contact residuals are recomputed on the final pose. Interval boundaries are baked exactly; transition blending is caller-authored."]});
    publish_bundle(out, |dir| {
        asset.document.save(&dir.join("character.glb"))?;
        let loaded = load_gltf(dir.join("character.glb"))?;
        select_clip(&loaded.asset, &recipe.name)?.validate(&loaded.asset.skeleton)?;
        write_json(&dir.join("report.json"), &report)?;
        write_json(&dir.join("recipe.json"), &recipe)
    })?;
    Ok(json!({"output":out,"character":out.join("character.glb"),"report":report}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nonrepeating_gltf_endpoint_does_not_reset_root_motion() {
        let skeleton = Skeleton {
            joints: vec![Joint {
                id: "root".into(),
                name: "root".into(),
                parent: None,
                rest: Transform::default(),
                matrix: None,
                deform: true,
            }],
        };
        for duration in [0.7f64, 31.0 / 30.0] {
            let stored_end = f64::from(duration as f32);
            for speed in [0.5, 1.0, 2.0] {
                let layer: Layer = serde_json::from_value(json!({
                    "clip":"move", "speed":speed, "timeline_start":0.25,
                }))
                .unwrap();
                let prepared = Prepared {
                    clip: MotionClip {
                        name: "move".into(),
                        morph_tracks: vec![],
                        tracks: vec![Track {
                            joint: 0,
                            property: Property::Translation,
                            interpolation: Interpolation::Linear,
                            times: vec![0.0, stored_end],
                            values: vec![[0.0; 4], [0.0, 0.0, 3.0, 0.0]],
                        }],
                    },
                    start: 0.0,
                    end: stored_end,
                    mask: HashSet::from([0]),
                };
                let end = layer.timeline_start + duration / speed;
                let pose = layer_pose(
                    &skeleton,
                    std::slice::from_ref(&layer),
                    std::slice::from_ref(&prepared),
                    end,
                )
                .unwrap();
                assert_eq!(pose[0].translation, [0.0, 0.0, 3.0]);
                // The tolerance must not keep an ended layer active.
                assert!(layer_sample(&layer, &prepared, end + 1e-4 / speed).is_none());
            }
        }
    }

    #[test]
    fn masked_additive_layer_does_not_reset_other_channels() {
        let skeleton = Skeleton {
            joints: vec![Joint {
                id: "r".into(),
                name: "root".into(),
                parent: None,
                rest: Transform::default(),
                matrix: None,
                deform: true,
            }],
        };
        let base = MotionClip {
            name: "move".into(),
            morph_tracks: Vec::new(),
            tracks: vec![Track {
                joint: 0,
                property: Property::Translation,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 1.0],
                values: vec![[0.0; 4], [2.0, 0.0, 0.0, 0.0]],
            }],
        };
        let turn = MotionClip {
            name: "turn".into(),
            morph_tracks: Vec::new(),
            tracks: vec![Track {
                joint: 0,
                property: Property::Rotation,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 1.0],
                values: vec![
                    [0.0, 0.0, 0.0, 1.0],
                    a4(nalgebra::UnitQuaternion::from_axis_angle(
                        &nalgebra::Vector3::y_axis(),
                        1.0,
                    )),
                ],
            }],
        };
        let layer: Layer = serde_json::from_value(json!({"clip":"move"})).unwrap();
        let mut additive = layer.clone();
        additive.additive = true;
        additive.weight = 0.5;
        let prepared = vec![
            Prepared {
                clip: base,
                start: 0.0,
                end: 1.0,
                mask: HashSet::from([0]),
            },
            Prepared {
                clip: turn,
                start: 0.0,
                end: 1.0,
                mask: HashSet::from([0]),
            },
        ];
        let pose = layer_pose(&skeleton, &[layer, additive], &prepared, 1.0).unwrap();
        assert_eq!(pose[0].translation, [2.0, 0.0, 0.0]);
        assert!((unit_quat(pose[0].rotation).angle() - 0.5).abs() < 1e-10);
    }
}
#[test]
fn contact_events_preserve_subframe_times_and_storage_precision() {
    let contact: Contact = serde_json::from_value(json!({
        "joint":"hoof", "start":0.16666666666666669, "end":0.3833333333333333,
    }))
    .unwrap();
    assert!(contact_active(&contact, 10.0 / 60.0));
    assert!(contact_active(&contact, 23.0 / 60.0));
    assert!(contact_active(&contact, f64::from(contact.start as f32)));
    assert!(contact_active(&contact, f64::from(contact.end as f32)));
    assert!(!contact_active(&contact, contact.start - 1e-4));
    assert!(!contact_active(&contact, contact.end + 1e-4));
    let recipe: AnimationRecipe = serde_json::from_value(json!({
        "schema_version":"0.1", "asset":"unused.glb", "name":"contact_events",
        "duration":1., "fps":30., "layers":[{"clip":"walk"}],
        "contacts":[{"joint":"hoof", "start":0.171, "end":0.381}],
    }))
    .unwrap();
    let times = recipe_sample_times(&recipe).unwrap();
    assert!(times.contains(&0.171) && times.contains(&0.381));
    assert!(times.windows(2).all(|t| (t[0] as f32) < (t[1] as f32)));
}
