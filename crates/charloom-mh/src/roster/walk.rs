//! Restore a source animation on its original bind skeleton. Node indices
//! may change when meshes are added, so map channels through the skin palette.
use anyhow::{Context, Result, ensure};
use charloom_io::load_gltf;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

pub fn restore(model: &Path, original: &Path, out: &Path) -> Result<Value> {
    let src = load_gltf(original)?;
    let target = load_gltf(model)?;
    let skin = src.asset.skins.first().context("missing source skin")?;
    let dest = target.asset.skins.first().context("missing target skin")?;
    ensure!(
        skin.joints.len() == dest.joints.len()
            && skin.inverse_bind_matrices == dest.inverse_bind_matrices,
        "incompatible bind palette"
    );
    let mut mapping = HashMap::new();
    let mut names = HashSet::new();
    for (&i, &j) in skin.joints.iter().zip(&dest.joints) {
        let (a, b) = (
            &src.asset.skeleton.joints[i],
            &target.asset.skeleton.joints[j],
        );
        ensure!(
            a.name == b.name && names.insert(&a.name),
            "ambiguous or incompatible joint {}",
            a.name
        );
        let mut parents = (Some(i), Some(j));
        while let (Some(si), Some(ti)) = parents {
            let (s, t) = (
                &src.asset.skeleton.joints[si],
                &target.asset.skeleton.joints[ti],
            );
            ensure!(
                s.name == t.name && s.rest == t.rest && s.matrix == t.matrix,
                "joint rest transform differs for {}",
                s.name
            );
            parents = (s.parent, t.parent);
        }
        ensure!(parents == (None, None), "joint hierarchy differs");
        mapping.insert(i, j);
    }
    let mut walk = src
        .asset
        .clips
        .into_iter()
        .find(|c| c.name == "walking")
        .context("source has no walking clip")?;
    ensure!(
        walk.morph_tracks.is_empty(),
        "expected skeletal walking clip"
    );
    for track in &mut walk.tracks {
        track.joint = *mapping
            .get(&track.joint)
            .context("walk channel targets a non-joint node")?;
    }
    let mut doc = target.document;
    doc.array("animations")
        .retain(|clip| clip["name"] != "walking");
    doc.append_animation(&walk, 0)?;
    doc.json["asset"]["extras"]["charloom_experimental_preview"]["neutral_only"] = false.into();
    doc.save(out)?;
    Ok(json!({"output":out,"channels":walk.tracks.len()}))
}
