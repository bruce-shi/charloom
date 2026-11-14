//! Embed review clips sampled by the Gallery on the final exported rig.
use anyhow::{Context, Result, ensure};
use charloom_core::MotionClip;
use charloom_io::GlbDocument;
use serde_json::{Value, json};
use std::path::Path;

pub fn embed(model: &Path, tracks: &Path, out: &Path) -> Result<Value> {
    // Only container metadata and the new animation channels are needed. A review
    // assembly may include unreferenced meshes from a prior fitting stage.
    let bytes = std::fs::read(model)?;
    let glb = gltf::binary::Glb::from_slice(&bytes)?;
    let json: Value = serde_json::from_slice(&glb.json)?;
    let mut doc = GlbDocument {
        json,
        binary: glb.bin.context("missing binary chunk")?.into_owned(),
    };
    let size = doc.json["buffers"][0]["byteLength"]
        .as_u64()
        .context("missing buffer length")? as usize;
    ensure!(
        doc.binary.len() >= size && doc.binary.len() - size <= 3,
        "invalid embedded buffer length"
    );
    doc.binary.truncate(size);
    let clips: Vec<MotionClip> = super::read(tracks)?;
    let node_count = doc.json["nodes"].as_array().context("missing nodes")?.len();
    ensure!(
        clips.len() == 2 && clips[0].name == "face_test" && clips[1].name == "face_talk",
        "missing face review clips"
    );
    doc.array("animations")
        .retain(|a| a["name"] != "face_talk" && a["name"] != "face_test");
    for clip in &clips {
        for t in &clip.tracks {
            ensure!(t.joint < node_count, "bone outside document");
        }
        for t in &clip.morph_tracks {
            let mesh = doc.json["nodes"]
                .get(t.node)
                .and_then(|n| n["mesh"].as_u64())
                .context("morph target missing mesh")? as usize;
            let parts = doc.json["meshes"][mesh]["primitives"]
                .as_array()
                .context("missing morph primitives")?;
            let width = parts
                .first()
                .and_then(|p| p["targets"].as_array())
                .context("missing corrective shapes")?
                .len();
            ensure!(
                parts
                    .iter()
                    .all(|p| p["targets"].as_array().is_some_and(|a| a.len() == width))
                    && t.values.iter().all(|v| v.len() == width),
                "morph target width mismatch"
            );
        }
        doc.append_animation(clip, 0)?;
    }
    doc.save(out)?;
    Ok(json!({"output":out,"clips":clips.iter().map(|c|&c.name).collect::<Vec<_>>()}))
}
