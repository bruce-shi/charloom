use crate::GlbDocument;
use anyhow::{Context, Result, ensure};
use charloom_core::*;
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_ID: AtomicU64 = AtomicU64::new(0);

pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)?;
    let name = path
        .file_name()
        .context("invalid output filename")?
        .to_string_lossy();
    let tmp = parent.join(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        TEMP_ID.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| -> Result<()> {
        let mut f = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp, path).with_context(|| format!("publish {}", path.display()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

pub fn write_json(path: &Path, value: &(impl serde::Serialize + ?Sized)) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    write_atomic(path, &bytes)
}

impl GlbDocument {
    pub fn new() -> Self {
        Self {
            json: json!({"asset":{"version":"2.0","generator":format!("charloom {}", ENGINE_VERSION)},"scene":0,"scenes":[{"nodes":[]}],"nodes":[],"meshes":[],"bufferViews":[],"accessors":[]}),
            binary: Vec::new(),
        }
    }

    pub fn array(&mut self, name: &str) -> &mut Vec<Value> {
        self.json
            .as_object_mut()
            .expect("glTF object")
            .entry(name)
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .expect("glTF array")
    }

    pub fn append_view(&mut self, bytes: &[u8], target: Option<u32>) -> usize {
        while !self.binary.len().is_multiple_of(4) {
            self.binary.push(0);
        }
        let offset = self.binary.len();
        self.binary.extend_from_slice(bytes);
        let mut view = json!({"buffer":0,"byteOffset":offset,"byteLength":bytes.len()});
        if let Some(target) = target {
            view["target"] = json!(target);
        }
        let views = self.array("bufferViews");
        let index = views.len();
        views.push(view);
        index
    }

    pub fn append_accessor(
        &mut self,
        bytes: &[u8],
        component_type: u32,
        shape: &str,
        count: usize,
        target: Option<u32>,
        bounds: Option<(Value, Value)>,
    ) -> usize {
        let view = self.append_view(bytes, target);
        let mut accessor =
            json!({"bufferView":view,"componentType":component_type,"count":count,"type":shape});
        if let Some((min, max)) = bounds {
            accessor["min"] = min;
            accessor["max"] = max;
        }
        let accessors = self.array("accessors");
        let index = accessors.len();
        accessors.push(accessor);
        index
    }

    pub fn append_vec3(
        &mut self,
        values: &[Vec3],
        target: Option<u32>,
        with_bounds: bool,
    ) -> usize {
        let bytes: Vec<u8> = values
            .iter()
            .flatten()
            .flat_map(|v| (*v as f32).to_le_bytes())
            .collect();
        let bounds = if with_bounds {
            Bounds::from_points(values.iter().copied()).map(|b| (json!(b.min), json!(b.max)))
        } else {
            None
        };
        self.append_accessor(&bytes, 5126, "VEC3", values.len(), target, bounds)
    }

    pub fn append_matrices(&mut self, values: &[Mat4]) -> usize {
        let bytes: Vec<u8> = values
            .iter()
            .flatten()
            .flatten()
            .flat_map(|v| (*v as f32).to_le_bytes())
            .collect();
        self.append_accessor(&bytes, 5126, "MAT4", values.len(), None, None)
    }

    pub fn append_animation(&mut self, clip: &MotionClip, joint_offset: usize) -> Result<usize> {
        let mut samplers = Vec::new();
        let mut channels = Vec::new();
        for track in &clip.tracks {
            track.validate()?;
            ensure!(
                track
                    .times
                    .windows(2)
                    .all(|w| (w[0] as f32) < (w[1] as f32)),
                "export_time_precision_collision: resample with distinct float32 times"
            );
            ensure!(
                track
                    .times
                    .iter()
                    .chain(track.values.iter().flatten())
                    .all(|v| (*v as f32).is_finite()),
                "export_value_exceeds_float32"
            );
            let time_bytes: Vec<u8> = track
                .times
                .iter()
                .flat_map(|v| (*v as f32).to_le_bytes())
                .collect();
            let input = self.append_accessor(
                &time_bytes,
                5126,
                "SCALAR",
                track.times.len(),
                None,
                Some((
                    json!([track.times[0]]),
                    json!([track.times[track.times.len() - 1]]),
                )),
            );
            let components = if track.property == Property::Rotation {
                4
            } else {
                3
            };
            let bytes: Vec<u8> = track
                .values
                .iter()
                .flat_map(|v| {
                    v[..components]
                        .iter()
                        .flat_map(|v| (*v as f32).to_le_bytes())
                })
                .collect();
            let output = self.append_accessor(
                &bytes,
                5126,
                if components == 4 { "VEC4" } else { "VEC3" },
                track.values.len(),
                None,
                None,
            );
            let interpolation = match track.interpolation {
                Interpolation::Step => "STEP",
                Interpolation::Linear => "LINEAR",
                Interpolation::CubicSpline => "CUBICSPLINE",
            };
            let path = match track.property {
                Property::Translation => "translation",
                Property::Rotation => "rotation",
                Property::Scale => "scale",
            };
            channels.push(json!({"sampler":samplers.len(),"target":{"node":track.joint+joint_offset,"path":path}}));
            samplers.push(json!({"input":input,"output":output,"interpolation":interpolation}));
        }
        for track in &clip.morph_tracks {
            track.validate()?;
            ensure!(
                track
                    .times
                    .windows(2)
                    .all(|w| (w[0] as f32) < (w[1] as f32)),
                "export_time_precision_collision: resample with distinct float32 times"
            );
            ensure!(
                track
                    .times
                    .iter()
                    .chain(track.values.iter().flatten())
                    .all(|v| (*v as f32).is_finite()),
                "export_value_exceeds_float32"
            );
            let time_bytes: Vec<_> = track
                .times
                .iter()
                .flat_map(|v| (*v as f32).to_le_bytes())
                .collect();
            let input = self.append_accessor(
                &time_bytes,
                5126,
                "SCALAR",
                track.times.len(),
                None,
                Some((
                    json!([track.times[0]]),
                    json!([track.times[track.times.len() - 1]]),
                )),
            );
            let bytes: Vec<_> = track
                .values
                .iter()
                .flatten()
                .flat_map(|v| (*v as f32).to_le_bytes())
                .collect();
            let output = self.append_accessor(
                &bytes,
                5126,
                "SCALAR",
                track.values.iter().map(Vec::len).sum(),
                None,
                None,
            );
            let interpolation = match track.interpolation {
                Interpolation::Step => "STEP",
                Interpolation::Linear => "LINEAR",
                Interpolation::CubicSpline => "CUBICSPLINE",
            };
            channels.push(json!({"sampler":samplers.len(),"target":{"node":track.node+joint_offset,"path":"weights"}}));
            samplers.push(json!({"input":input,"output":output,"interpolation":interpolation}));
        }
        ensure!(!channels.is_empty(), "cannot export an empty animation");
        let animations = self.array("animations");
        let index = animations.len();
        animations.push(json!({"name":clip.name,"samplers":samplers,"channels":channels}));
        Ok(index)
    }

    pub fn append_skeleton(&mut self, skeleton: &Skeleton) -> Result<usize> {
        skeleton.validate()?;
        let offset = self.array("nodes").len();
        for (i, joint) in skeleton.joints.iter().enumerate() {
            let children: Vec<_> = skeleton
                .joints
                .iter()
                .enumerate()
                .filter(|(_, j)| j.parent == Some(i))
                .map(|(i, _)| i + offset)
                .collect();
            let mut node = json!({"name":joint.name});
            if let Some(matrix) = joint.matrix {
                node["matrix"] = json!(matrix.into_iter().flatten().collect::<Vec<_>>());
            } else {
                node["translation"] = json!(joint.rest.translation);
                node["rotation"] = json!(joint.rest.rotation);
                node["scale"] = json!(joint.rest.scale);
            }
            if !children.is_empty() {
                node["children"] = json!(children);
            }
            self.array("nodes").push(node);
        }
        let scene_index = self.json.get("scene").and_then(Value::as_u64).unwrap_or(0) as usize;
        if self.array("scenes").is_empty() {
            self.array("scenes").push(json!({"nodes":[]}));
        }
        let roots = skeleton
            .joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.parent.is_none())
            .map(|(i, _)| json!(i + offset));
        let nodes = self.json["scenes"][scene_index]
            .as_object_mut()
            .context("invalid scene")?
            .entry("nodes")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .context("invalid scene nodes")?;
        nodes.extend(roots);
        Ok(offset)
    }

    pub fn to_glb(&self) -> Result<Vec<u8>> {
        ensure!(
            self.binary.len() <= u32::MAX as usize,
            "GLB buffer exceeds 4 GiB"
        );
        let mut json = self.json.clone();
        // Re-export legacy engine metadata under the current name without
        // mutating the source document or another tool's generator attribution.
        if let Some(legacy) = json["asset"]["generator"].as_str().and_then(|name| {
            name.strip_prefix("humanik ")
                .or_else(|| name.strip_prefix("Humanik "))
        }) {
            json["asset"]["generator"] = json!(format!("charloom {legacy}"));
        }
        json["buffers"] = json!([{"byteLength":self.binary.len()}]);
        let mut json_bytes = serde_json::to_vec(&json)?;
        while !json_bytes.len().is_multiple_of(4) {
            json_bytes.push(b' ');
        }
        let binary_size = (self.binary.len() + 3) & !3;
        let length = 12 + 8 + json_bytes.len() + 8 + binary_size;
        ensure!(length <= u32::MAX as usize, "GLB file exceeds 4 GiB");
        let mut output = Vec::with_capacity(length);
        output.extend_from_slice(b"glTF");
        output.extend_from_slice(&2u32.to_le_bytes());
        output.extend_from_slice(&(length as u32).to_le_bytes());
        output.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
        output.extend_from_slice(b"JSON");
        output.extend_from_slice(&json_bytes);
        output.extend_from_slice(&(binary_size as u32).to_le_bytes());
        output.extend_from_slice(b"BIN\0");
        output.extend_from_slice(&self.binary);
        output.resize(length, 0);
        // Structural validation independent from our JSON writer.
        gltf::Gltf::from_slice(&output).context("exported GLB failed validation")?;
        Ok(output)
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        write_atomic(path, &self.to_glb()?)
    }
}

impl Default for GlbDocument {
    fn default() -> Self {
        Self::new()
    }
}

pub fn mesh_document(
    name: &str,
    positions: &[Vec3],
    triangles: &[[u32; 3]],
) -> Result<GlbDocument> {
    ensure!(
        !positions.is_empty() && !triangles.is_empty(),
        "mesh must not be empty"
    );
    ensure!(
        positions.iter().flatten().all(|v| v.is_finite()),
        "non-finite mesh position"
    );
    ensure!(
        triangles
            .iter()
            .flatten()
            .all(|&v| (v as usize) < positions.len()),
        "mesh index out of bounds"
    );
    let mut doc = GlbDocument::new();
    let pos = doc.append_vec3(positions, Some(34962), true);
    let bytes: Vec<u8> = triangles
        .iter()
        .flatten()
        .flat_map(|i| i.to_le_bytes())
        .collect();
    let indices = doc.append_accessor(
        &bytes,
        5125,
        "SCALAR",
        triangles.len() * 3,
        Some(34963),
        None,
    );
    let mut normals = vec![nalgebra::Vector3::zeros(); positions.len()];
    for &[a, b, c] in triangles {
        let cross = (v3(positions[b as usize]) - v3(positions[a as usize]))
            .cross(&(v3(positions[c as usize]) - v3(positions[a as usize])));
        for i in [a, b, c] {
            normals[i as usize] += cross;
        }
    }
    let normals: Vec<_> = normals
        .into_iter()
        .map(|n| a3(n.try_normalize(1e-15).unwrap_or(nalgebra::Vector3::y())))
        .collect();
    let normal = doc.append_vec3(&normals, Some(34962), false);
    doc.json["materials"] = json!([{"name":"diagnostic","pbrMetallicRoughness":{"baseColorFactor":[0.24,0.62,0.64,1.0],"metallicFactor":0.0,"roughnessFactor":0.8}}]);
    doc.array("meshes").push(json!({"name":name,"primitives":[{"attributes":{"POSITION":pos,"NORMAL":normal},"indices":indices,"material":0}]}));
    doc.array("nodes").push(json!({"name":name,"mesh":0}));
    doc.json["scenes"][0]["nodes"] = json!([0]);
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reexport_renames_legacy_generator_without_changing_source_or_geometry() {
        let mut doc = mesh_document(
            "triangle",
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[[0, 1, 2]],
        )
        .unwrap();
        for (source, expected) in [
            ("humanik 0.1.0-alpha.1", "charloom 0.1.0-alpha.1"),
            (
                "Humanik read-only extraction",
                "charloom read-only extraction",
            ),
            ("Blender 5.2.1", "Blender 5.2.1"),
        ] {
            doc.json["asset"]["generator"] = json!(source);
            let original = doc.json.clone();
            let exported = gltf::Gltf::from_slice(&doc.to_glb().unwrap()).unwrap();
            assert_eq!(
                exported.document.as_json().asset.generator.as_deref(),
                Some(expected)
            );
            assert_eq!(exported.blob.as_ref().unwrap(), &doc.binary);
            assert_eq!(doc.json, original);
        }
    }

    #[test]
    fn glb_roundtrip_preserves_world_transform_and_animation() {
        let mut doc = mesh_document(
            "triangle",
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            &[[0, 1, 2]],
        )
        .unwrap();
        doc.json["nodes"][0]["translation"] = json!([3.0, 2.0, 1.0]);
        let clip = MotionClip {
            morph_tracks: Vec::new(),
            name: "move".into(),
            tracks: vec![Track {
                joint: 0,
                property: Property::Translation,
                interpolation: Interpolation::Linear,
                times: vec![0.0, 2.0],
                values: vec![[3.0, 2.0, 1.0, 0.0], [5.0, 2.0, 1.0, 0.0]],
            }],
        };
        doc.append_animation(&clip, 0).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("test.glb");
        doc.save(&path).unwrap();
        let loaded = crate::load_gltf(&path).unwrap();
        assert_eq!(loaded.asset.bounds().unwrap().unwrap().min, [3.0, 2.0, 1.0]);
        let pose = loaded.asset.clips[0]
            .sample(&loaded.asset.skeleton, 1.0)
            .unwrap();
        assert_eq!(pose[0].translation, [4.0, 2.0, 1.0]);
        assert_eq!(loaded.document.json["materials"], doc.json["materials"]);
    }
}
