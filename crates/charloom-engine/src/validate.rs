use anyhow::Result;
use charloom_core::*;
use serde_json::{Value, json};

pub fn validate_asset(asset: &Asset) -> Result<Value> {
    asset.skeleton.validate()?;
    let mut issues = asset.issues.clone();
    let mut max_weight_error: f64 = 0.0;
    let mut max_influences = 0usize;
    for mesh in &asset.meshes {
        for weights in &mesh.influences {
            max_weight_error =
                max_weight_error.max((weights.iter().map(|w| w.weight).sum::<f64>() - 1.0).abs());
            max_influences = max_influences.max(weights.len());
        }
    }
    if max_influences > 4 {
        issues.push(Issue::warning(
            "more_than_four_influences",
            format!("source contains up to {max_influences} influences per vertex"),
        ));
    }
    let original = asset.world_meshes()?;
    let neutral = asset.skinned_positions(&asset.skeleton.rest_pose())?;
    let mut rest_delta: f64 = 0.0;
    for (a, b) in original.iter().flatten().zip(neutral.iter().flatten()) {
        rest_delta = rest_delta.max((v3(*a) - v3(*b)).norm());
    }
    let height = asset.bounds()?.map_or(1.0, |b| b.diagonal().max(1e-9));
    if rest_delta > height * 1e-4 {
        issues.push(Issue::warning("default_pose_differs_from_bind",format!("maximum rest skin displacement {rest_delta:.6} m; default node pose may differ from bind pose")));
    }
    let mut clip_results = Vec::new();
    for clip in &asset.clips {
        clip.validate(&asset.skeleton)?;
        let [start, end] = clip.time_range();
        let frames = ((end - start) * 30.0).ceil().clamp(1.0, 600.0) as usize;
        let mut max_displacement: f64 = 0.0;
        for frame in 0..=frames {
            let t = start + (end - start) * frame as f64 / frames as f64;
            let pose = clip.sample(&asset.skeleton, t)?;
            let deformed = asset.deformed_positions(&pose, &clip.sample_morphs(t)?)?;
            for (a, b) in neutral.iter().flatten().zip(deformed.iter().flatten()) {
                if !b.iter().all(|v| v.is_finite()) {
                    issues.push(Issue::error(
                        "non_finite_deformation",
                        format!("{} at {t}", clip.name),
                    ));
                    break;
                }
                max_displacement = max_displacement.max((v3(*a) - v3(*b)).norm());
            }
        }
        clip_results.push(json!({"name":clip.name,"time_range_s":[start,end],"sample_count":frames+1,"max_vertex_displacement_m":max_displacement}));
    }
    Ok(
        json!({"schema_version":SCHEMA_VERSION,"source_hash":asset.source_hash,"data_status":if issues.iter().any(|i|i.severity==Severity::Error){"failed"}else{"passed"},"quality_status":"needs_review","visual_review":"pending","metrics":{"max_weight_sum_error":max_weight_error,"max_influences":max_influences,"default_pose_skin_displacement_m":rest_delta},"clips":clip_results,"issues":issues,"limits":["Anatomical quality, foot contacts, and physical plausibility are not certified by structural validation."]}),
    )
}
