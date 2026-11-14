use crate::{
    motion::{RetargetConfig, retarget},
    solve::{Effector, SolveConfig, solve_pose},
    validate::validate_asset,
};
use anyhow::{Result, ensure};
use charloom_core::*;
use charloom_io::{load_gltf, load_motion_asset, write_json};
use serde_json::{Value, json};
use std::path::Path;

pub fn capabilities() -> Value {
    json!({"schema_version":SCHEMA_VERSION,"engine_version":ENGINE_VERSION,"stage":"experimental_alpha",
        "input_formats":["gltf","glb","fbx_motion"],"output_formats":["glb","json"],
        "fbx":{"scope":"skeleton_and_transform_animation_only","geometry_and_materials":false,"bake_hz":60,"units":"meters","axes":"right_handed_y_up","backend":"ufbx_0.11.5"},
        "operations":["asset_inspect","motion_map","motion_retarget","pose_solve","animation_build","artifact_validate","skin_repair_start","skin_repair_step","skin_repair_inspect","surface_analyze","surface_contacts","mh_guide","mh_guide_refine","mh_guide_check","mh_transfer"],
        "skin_repair":{"agent_directed":true,"maximum_correction_rounds":null,"optional_round_budget":true,"fixed_sampled_quality_gate":true,"keep_best_candidate":true,"exact_manifold_seam_gate":true,"optional_sampled_surface_crossing_gate":true,"resume_preserves_round_budget":true,"strategies":["relax_strain","exclude_influences","weld_seams","separate_surfaces","update_budget","accept","report_unresolved"]},
        "motion":{"reference_pose_retarget":true,"explicit_joint_overrides":true,"multi_effector_damped_ik":true,"two_bone_ik":true,"surface_fitted_body_clearance":true,"elliptical_limb_cross_sections":true,"wrist_twist_distribution":true,"target_digit_length_guard":true,"digit_surface_strain_guard":true,"explicit_calibrated_hinge_and_wrist_limits":true,"bind_space_rigid_bone_regions":true,"fixed_pose_hand_weight_repair":true,"clip_scoped_hand_surface_correctives":{"volume_links":true,"explicit_capsule_obstacles":true,"sampled_pose_bases":true,"compact_hand_partitions":true,"skeletal_tracks_preserved":true,"physical_simulation":false},"recipe_layers":true,"explicit_contact_intervals":true,"explicit_support_surface_floor":{"vertical_root_only":true,"raise_only_by_default":true,"optional_source_height_preservation":true,"horizontal_foot_lock":false}},
        "props":{"external_static_rigid_gltf":true,"existing_weapon_bones":true,"primary_hand_attachment":true,"secondary_hand_and_fingertip_ik":true,"grip_intervals":true,"closed_chain_articulated_grip_solve":true,"explicit_per_frame_joint_speed_bound":true,"release_policies":["hold_world","source"],"automatic_grasp_geometry":false,"physics":false},
        "interfaces":["Rust","CLI_JSON","MCP_stdio"],
        "not_implemented":["validated_anatomical_joint_prediction","learned_morphology_backend","FBX_geometry_material_conversion","BVH","automatic_contact_detection","cross_character_morph_retarget","automatic_anatomical_joint_limit_inference","persistent_jobs","in_flight_cancellation"],
        "quality_contract":"Experimental outputs require visual and anatomical review. Structural validity is reported separately."})
}

pub fn validate_path(input: &Path) -> Result<Value> {
    validate_path_clip(input, None)
}

pub fn validate_path_clip(input: &Path, clip: Option<&str>) -> Result<Value> {
    let mut asset = load_gltf(input)?.asset;
    let original_clip_count = asset.clips.len();
    if let Some(name) = clip {
        select_clip(&asset, name)?;
        asset.clips.retain(|c| c.name == name);
    }
    let mut report = validate_asset(&asset)?;
    report["validation_scope"] = json!({"selected_clip":clip,"available_clip_count":original_clip_count,"validated_clip_count":asset.clips.len(),"all_clips_validated":clip.is_none() || original_clip_count == 1});
    Ok(report)
}

pub fn select_clip<'a>(asset: &'a Asset, name: &str) -> Result<&'a MotionClip> {
    let clips: Vec<_> = asset.clips.iter().filter(|c| c.name == name).collect();
    ensure!(
        clips.len() == 1,
        "clip must identify one animation by name: {name}"
    );
    Ok(clips[0])
}

pub fn retarget_path(
    source: &Path,
    target: &Path,
    clip_name: &str,
    config: &RetargetConfig,
    out: &Path,
) -> Result<Value> {
    ensure!(!out.exists(), "output_exists");
    let source = load_motion_asset(source)?;
    let mut target = load_gltf(target)?;
    let clip = select_clip(&source, clip_name)?;
    let (motion, report) = retarget(&source.skeleton, clip, &target.asset.skeleton, config)?;
    ensure!(
        !target.asset.clips.iter().any(|c| c.name == motion.name),
        "animation_name_exists: {}",
        motion.name
    );
    target.document.append_animation(&motion, 0)?;
    let result = serde_json::json!({"operation":"motion_retarget","retarget":report,"quality_status":"needs_review","visual_review":"pending",
        "source_hash":source.source_hash,"target_hash":target.asset.source_hash,"source_clip":clip_name,"source_issues":source.issues,"target_issues":target.asset.issues});
    publish_bundle(out, |dir| {
        target.document.save(&dir.join("character.glb"))?;
        let reloaded = load_gltf(dir.join("character.glb"))?;
        select_clip(&reloaded.asset, &motion.name)?.validate(&reloaded.asset.skeleton)?;
        write_json(&dir.join("report.json"), &result)?;
        write_json(&dir.join("config.json"), config)
    })?;
    Ok(json!({"output":out,"character":out.join("character.glb"),"report":result}))
}

pub fn solve_path(
    input: &Path,
    clip: Option<&str>,
    time: f64,
    effectors: &[Effector],
    config: &SolveConfig,
    out: &Path,
) -> Result<Value> {
    ensure!(!out.exists(), "output_exists");
    let mut source = load_gltf(input)?;
    let initial = if let Some(clip) = clip {
        select_clip(&source.asset, clip)?.sample(&source.asset.skeleton, time)?
    } else {
        source.asset.skeleton.rest_pose()
    };
    let result = solve_pose(&source.asset.skeleton, &initial, effectors, config)?;
    let joints: Vec<_> = source
        .asset
        .skeleton
        .joints
        .iter()
        .enumerate()
        .filter(|(_, j)| j.matrix.is_none())
        .map(|(i, _)| i)
        .collect();
    ensure!(
        !source
            .asset
            .clips
            .iter()
            .any(|c| c.name == "charloom_solved_pose"),
        "animation_name_exists: charloom_solved_pose"
    );
    let mut motion = crate::motion::clip_from_poses(
        "charloom_solved_pose",
        &[0.0, 1.0],
        &[result.pose.clone(), result.pose.clone()],
        &joints,
    )?;
    if let Some(clip) = clip {
        for (node, weights) in select_clip(&source.asset, clip)?.sample_morphs(time)? {
            motion.morph_tracks.push(MorphTrack {
                node,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 1.0],
                values: vec![weights.clone(), weights],
            });
        }
    }
    motion.morph_tracks.sort_by_key(|t| t.node);
    source.document.append_animation(&motion, 0)?;
    publish_bundle(out, |dir| {
        source.document.save(&dir.join("character.glb"))?;
        write_json(&dir.join("solve.json"), &result)?;
        write_json(&dir.join("effectors.json"), effectors)?;
        write_json(&dir.join("config.json"), config)
    })?;
    Ok(json!({"output":out,"solve":result,"quality_status":"needs_review"}))
}

/// Publish a complete directory only after all checks pass. Existing output is refused.
pub fn publish_bundle(out: &Path, write: impl FnOnce(&Path) -> Result<()>) -> Result<()> {
    ensure!(!out.exists(), "output_exists: {}", out.display());
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let temp = tempfile::Builder::new()
        .prefix(".charloom-")
        .tempdir_in(parent)?;
    write(temp.path())?;
    ensure!(!out.exists(), "output_exists: {}", out.display());
    std::fs::rename(temp.path(), out)?;
    Ok(())
}
