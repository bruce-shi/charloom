//! FBX skeleton and animation input. Geometry/material conversion is deliberately
//! outside this adapter: an FBX motion can drive an existing glTF character.
use anyhow::{Context, Result, ensure};
use charloom_core::*;
use sha2::{Digest, Sha256};
use std::path::Path;

const MAX_NODES: usize = 4096;
const MAX_KEYS: usize = 2_000_000;
const MAX_DURATION_S: f64 = 600.0;
const BAKE_HZ: f64 = 60.0;

fn allocator() -> ufbx::AllocatorOpts {
    ufbx::AllocatorOpts {
        memory_limit: 256 * 1024 * 1024,
        ..Default::default()
    }
}

fn transform(t: ufbx::Transform) -> Transform {
    Transform {
        translation: [t.translation.x, t.translation.y, t.translation.z],
        rotation: [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w],
        scale: [t.scale.x, t.scale.y, t.scale.z],
    }
}

/// Loads motion sources in glTF/GLB or FBX. FBX output has no render meshes;
/// consumers must not mistake successful motion import for model conversion.
pub fn load_motion_asset(path: impl AsRef<Path>) -> Result<Asset> {
    let path = path.as_ref();
    if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("fbx"))
    {
        load_fbx_motion(path)
    } else {
        Ok(crate::load_gltf(path)?.asset)
    }
}

pub fn load_fbx_motion(path: impl AsRef<Path>) -> Result<Asset> {
    let path = path.as_ref();
    let meta = std::fs::metadata(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        meta.len() <= crate::MAX_FILE_BYTES as u64,
        "resource_limit: FBX exceeds 256 MiB"
    );
    let bytes = std::fs::read(path)?;
    let scene = ufbx::load_memory(
        &bytes,
        ufbx::LoadOpts {
            ignore_geometry: true,
            ignore_embedded: true,
            load_external_files: false,
            node_depth_limit: 256,
            file_format: ufbx::FileFormat::Fbx,
            target_axes: ufbx::CoordinateAxes::right_handed_y_up(),
            target_unit_meters: 1.0,
            // Helpers express FBX scale inheritance using an ordinary TRS tree.
            inherit_mode_handling: ufbx::InheritModeHandling::HelperNodes,
            geometry_transform_handling: ufbx::GeometryTransformHandling::HelperNodes,
            temp_allocator: allocator(),
            result_allocator: allocator(),
            ..Default::default()
        },
    )
    .map_err(|e| anyhow::anyhow!("fbx_import_failed: {}", e.description))?;
    ensure!(
        scene.nodes.len() <= MAX_NODES,
        "resource_limit: FBX has too many nodes"
    );
    ensure!(
        scene.anim_stacks.len() <= 256,
        "resource_limit: FBX has too many takes"
    );
    ensure!(
        scene.constraints.is_empty(),
        "unsupported_fbx_constraints: bake constraints in the source application first"
    );
    let skeleton = Skeleton {
        joints: scene
            .nodes
            .iter()
            .map(|node| Joint {
                id: format!("fbx_node_{}", node.element.typed_id),
                name: if node.is_root {
                    "__fbx_scene__".into()
                } else {
                    node.element.name.to_string()
                },
                parent: node.parent.as_ref().map(|p| p.element.typed_id as usize),
                rest: transform(node.local_transform),
                matrix: None,
                deform: node.bone.is_some(),
            })
            .collect(),
    };
    skeleton.validate()?;
    // Catch an unsupported FBX inheritance/affine representation before retargeting.
    let worlds = skeleton.world_matrices(&skeleton.rest_pose())?;
    for (i, node) in scene.nodes.iter().enumerate() {
        ensure!(
            node.element.typed_id as usize == i,
            "invalid_fbx_node_order"
        );
        let m = &node.node_to_world;
        let expected = [
            [m.m00, m.m10, m.m20, 0.0],
            [m.m01, m.m11, m.m21, 0.0],
            [m.m02, m.m12, m.m22, 0.0],
            [m.m03, m.m13, m.m23, 1.0],
        ];
        let expected = matrix_from(expected);
        ensure!(
            (worlds[i] - expected).norm() <= 1e-7 * expected.norm().max(1.0),
            "unsupported_fbx_transform: {} cannot be represented by the TRS hierarchy",
            node.element.name
        );
    }
    let mut issues = vec![Issue {
        code: "fbx_motion_only".into(),
        severity: Severity::Info,
        message: format!(
            "FBX skeleton/takes imported with ufbx 0.11.5; right-handed Y-up meters; Euler/pre/post rotations and layers baked at up to {BAKE_HZ} Hz. Render geometry, materials and non-transform properties are not imported. Reference pose is the FBX default node pose, separate from the first animated frame."
        ),
        location: None,
    }];
    for warning in &scene.metadata.warnings {
        issues.push(Issue::warning(
            "fbx_source_warning",
            warning.description.to_string(),
        ));
    }
    let mut clips = Vec::new();
    let mut total_keys = 0usize;
    let mut names = std::collections::HashSet::new();
    for (stack_index, stack) in scene.anim_stacks.iter().enumerate() {
        let duration = stack.time_end - stack.time_begin;
        ensure!(
            duration.is_finite() && (0.0..=MAX_DURATION_S).contains(&duration),
            "resource_limit: FBX take duration must be in 0..600 seconds"
        );
        let baked = ufbx::bake_anim(
            &scene,
            &stack.anim,
            ufbx::BakeOpts {
                trim_start_time: true,
                resample_rate: BAKE_HZ,
                maximum_sample_rate: 240.0,
                temp_allocator: allocator(),
                result_allocator: allocator(),
                ..Default::default()
            },
        )
        .map_err(|e| anyhow::anyhow!("fbx_bake_failed: {}", e.description))?;
        let original_name = stack.element.name.to_string();
        let mut name = if original_name.is_empty() {
            format!("take_{stack_index}")
        } else {
            original_name
        };
        if !names.insert(name.clone()) {
            name = format!("{name}#{stack_index}");
            ensure!(names.insert(name.clone()), "duplicate_fbx_take_name");
        }
        let mut clip = MotionClip {
            name,
            tracks: Vec::new(),
            morph_tracks: Vec::new(),
        };
        for node in &baked.nodes {
            for (property, keys) in [
                (Property::Translation, &node.translation_keys),
                (Property::Scale, &node.scale_keys),
            ] {
                if keys.is_empty() {
                    continue;
                }
                clip.tracks.push(Track {
                    joint: node.typed_id as usize,
                    property,
                    interpolation: Interpolation::Linear,
                    times: keys.iter().map(|k| k.time).collect(),
                    values: keys
                        .iter()
                        .map(|k| [k.value.x, k.value.y, k.value.z, 0.0])
                        .collect(),
                });
            }
            if !node.rotation_keys.is_empty() {
                clip.tracks.push(Track {
                    joint: node.typed_id as usize,
                    property: Property::Rotation,
                    interpolation: Interpolation::Linear,
                    times: node.rotation_keys.iter().map(|k| k.time).collect(),
                    values: node
                        .rotation_keys
                        .iter()
                        .map(|k| [k.value.x, k.value.y, k.value.z, k.value.w])
                        .collect(),
                });
            }
        }
        total_keys += clip.tracks.iter().map(|t| t.times.len()).sum::<usize>();
        ensure!(
            total_keys <= MAX_KEYS,
            "resource_limit: too many FBX animation keys"
        );
        clip.validate(&skeleton)?;
        if !baked.elements.is_empty() {
            issues.push(Issue::warning("fbx_non_transform_animation", format!("{} contains non-transform animated properties which are outside the motion-only adapter", clip.name)));
        }
        clips.push(clip);
    }
    Ok(Asset {
        name: path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("motion")
            .into(),
        source_hash: format!("{:x}", Sha256::digest(&bytes)),
        skeleton,
        meshes: Vec::new(),
        skins: Vec::new(),
        clips,
        issues,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_fbx_without_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.fbx");
        std::fs::write(&path, b"not an FBX file").unwrap();
        assert!(
            load_fbx_motion(&path)
                .unwrap_err()
                .to_string()
                .contains("fbx_import_failed")
        );
    }
}
