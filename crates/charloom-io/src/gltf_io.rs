use anyhow::{Context, Result, bail, ensure};
use base64::Engine;
use charloom_core::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const MAX_FILE_BYTES: usize = 256 * 1024 * 1024;
pub const MAX_VERTICES: usize = 2_000_000;

#[derive(Clone)]
pub struct GlbDocument {
    pub json: Value,
    pub binary: Vec<u8>,
}

pub struct LoadedAsset {
    pub asset: Asset,
    pub document: GlbDocument,
    pub source_path: PathBuf,
}

fn read_limited(path: &Path) -> Result<Vec<u8>> {
    let metadata = std::fs::metadata(path).with_context(|| format!("read {}", path.display()))?;
    ensure!(
        metadata.len() <= MAX_FILE_BYTES as u64,
        "resource_limit: file exceeds 256 MiB"
    );
    std::fs::read(path).with_context(|| format!("read {}", path.display()))
}

fn decode_uri(base: &Path, uri: &str) -> Result<Vec<u8>> {
    if let Some(data) = uri.strip_prefix("data:") {
        let (header, encoded) = data.split_once(',').context("invalid data URI")?;
        ensure!(
            header.ends_with(";base64"),
            "unsupported_data_uri: expected base64"
        );
        ensure!(
            encoded.len() <= MAX_FILE_BYTES * 4 / 3 + 8,
            "resource_limit: data URI too large"
        );
        return base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .context("invalid base64 resource");
    }
    ensure!(
        !uri.contains("://") && !uri.starts_with("file:") && !Path::new(uri).is_absolute(),
        "external_uri: only relative files and data URIs are supported"
    );
    // Decode percent-escaped filenames, without treating '+' as a space.
    let mut decoded = Vec::new();
    let mut bytes = uri.bytes();
    while let Some(b) = bytes.next() {
        if b == b'%' {
            let a = bytes.next().context("truncated percent escape")?;
            let b = bytes.next().context("truncated percent escape")?;
            let hex = std::str::from_utf8(&[a, b])
                .context("invalid percent escape")?
                .to_owned();
            decoded.push(u8::from_str_radix(&hex, 16).context("invalid percent escape")?);
        } else {
            decoded.push(b);
        }
    }
    let relative = String::from_utf8(decoded).context("resource path is not UTF-8")?;
    let root = base.canonicalize()?;
    let path = root
        .join(relative)
        .canonicalize()
        .context("missing external resource")?;
    ensure!(
        path.starts_with(&root),
        "external_uri: resource escapes the asset directory"
    );
    read_limited(&path)
}

pub fn load_gltf(path: impl AsRef<Path>) -> Result<LoadedAsset> {
    let path = path.as_ref();
    let bytes = read_limited(path)?;
    let gltf = gltf::Gltf::from_slice(&bytes).context("invalid glTF/GLB document")?;
    let base = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let raw = if bytes.starts_with(b"glTF") {
        let glb = gltf::binary::Glb::from_slice(&bytes)?;
        serde_json::from_slice::<Value>(&glb.json)?
    } else {
        serde_json::from_slice::<Value>(&bytes)?
    };
    for extension in gltf.extensions_required() {
        ensure!(
            matches!(extension, "KHR_materials_unlit" | "KHR_texture_transform"),
            "unsupported_required_extension: {extension}"
        );
    }
    let mut hash = Sha256::new();
    hash.update(&bytes);
    let mut buffers = Vec::new();
    let mut total_bytes = bytes.len();
    for buffer in gltf.buffers() {
        let mut data = match buffer.source() {
            gltf::buffer::Source::Bin => gltf.blob.clone().context("missing GLB binary chunk")?,
            gltf::buffer::Source::Uri(uri) => decode_uri(base, uri)?,
        };
        ensure!(data.len() >= buffer.length(), "buffer_length_mismatch");
        data.truncate(buffer.length());
        total_bytes += data.len();
        ensure!(
            total_bytes <= MAX_FILE_BYTES * 2,
            "resource_limit: combined asset exceeds 512 MiB"
        );
        hash.update((data.len() as u64).to_le_bytes());
        hash.update(&data);
        buffers.push(data);
    }
    // Reject buffer-view ranges before the gltf readers can slice them.
    for view in gltf.views() {
        let buffer = buffers
            .get(view.buffer().index())
            .context("buffer index out of bounds")?;
        ensure!(
            view.offset()
                .checked_add(view.length())
                .is_some_and(|n| n <= buffer.len()),
            "buffer_view_out_of_bounds"
        );
    }
    let document = &gltf.document;
    let mut parents = vec![None; document.nodes().len()];
    for node in document.nodes() {
        for child in node.children() {
            ensure!(
                parents[child.index()].replace(node.index()).is_none(),
                "node_has_multiple_parents"
            );
        }
    }
    let mut deform = vec![false; parents.len()];
    for skin in document.skins() {
        for joint in skin.joints() {
            deform[joint.index()] = true;
        }
    }
    let skeleton = Skeleton {
        joints: document
            .nodes()
            .map(|node| {
                let (t, r, s) = node.transform().decomposed();
                Joint {
                    id: format!("node_{}", node.index()),
                    name: node
                        .name()
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("node_{}", node.index())),
                    parent: parents[node.index()],
                    rest: Transform {
                        translation: t.map(f64::from),
                        rotation: r.map(f64::from),
                        scale: s.map(f64::from),
                    },
                    matrix: match node.transform() {
                        gltf::scene::Transform::Matrix { matrix } => {
                            Some(matrix.map(|c| c.map(f64::from)))
                        }
                        _ => None,
                    },
                    deform: deform[node.index()],
                }
            })
            .collect(),
    };
    skeleton.validate()?;
    let skins: Vec<Skin> = document
        .skins()
        .map(|skin| {
            let joints: Vec<usize> = skin.joints().map(|j| j.index()).collect();
            let reader = skin.reader(|b| buffers.get(b.index()).map(Vec::as_slice));
            let inverse_bind_matrices = reader
                .read_inverse_bind_matrices()
                .map(|i| i.map(|m| m.map(|c| c.map(f64::from))).collect())
                .unwrap_or_else(|| {
                    vec![matrix_array(&nalgebra::Matrix4::identity()); joints.len()]
                });
            Skin {
                name: skin.name().unwrap_or("skin").into(),
                joints,
                inverse_bind_matrices,
            }
        })
        .collect();
    for skin in &skins {
        ensure!(
            skin.joints.len() == skin.inverse_bind_matrices.len(),
            "inverse_bind_matrix_count_mismatch"
        );
        ensure!(
            skin.inverse_bind_matrices
                .iter()
                .flatten()
                .flatten()
                .all(|x| x.is_finite()),
            "non_finite_inverse_bind_matrix"
        );
    }
    let mut issues = Vec::new();
    let mut meshes = Vec::new();
    let mut vertex_count = 0usize;
    let mut morph_vertex_count = 0usize;
    let mut morph_dense_count = 0usize;
    let mut active = vec![false; skeleton.joints.len()];
    let scene = document
        .default_scene()
        .or_else(|| document.scenes().next());
    if let Some(scene) = scene {
        let mut stack: Vec<_> = scene.nodes().collect();
        while let Some(node) = stack.pop() {
            active[node.index()] = true;
            stack.extend(node.children());
        }
    } else {
        active.fill(true);
    }
    for node in document.nodes().filter(|n| active[n.index()]) {
        let Some(mesh) = node.mesh() else {
            continue;
        };
        for primitive in mesh.primitives() {
            let reader = primitive.reader(|b| buffers.get(b.index()).map(Vec::as_slice));
            let positions: Vec<Vec3> = reader
                .read_positions()
                .context("unsupported_or_missing_positions")?
                .map(|p| p.map(f64::from))
                .collect();
            vertex_count += positions.len();
            ensure!(
                vertex_count <= MAX_VERTICES,
                "resource_limit: more than two million vertices"
            );
            ensure!(
                positions.iter().flatten().all(|x| x.is_finite()),
                "non_finite_vertex"
            );
            let indices: Vec<u32> = reader
                .read_indices()
                .map(|i| i.into_u32().collect())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());
            ensure!(
                indices.iter().all(|&i| (i as usize) < positions.len()),
                "vertex_index_out_of_bounds"
            );
            let triangles = match primitive.mode() {
                gltf::mesh::Mode::Triangles => {
                    ensure!(indices.len().is_multiple_of(3), "incomplete_triangle");
                    indices
                        .chunks_exact(3)
                        .map(|c| [c[0], c[1], c[2]])
                        .collect()
                }
                gltf::mesh::Mode::TriangleStrip => indices
                    .windows(3)
                    .enumerate()
                    .map(|(i, c)| {
                        if i % 2 == 0 {
                            [c[0], c[1], c[2]]
                        } else {
                            [c[1], c[0], c[2]]
                        }
                    })
                    .collect(),
                gltf::mesh::Mode::TriangleFan => indices[1..]
                    .windows(2)
                    .map(|c| [indices[0], c[0], c[1]])
                    .collect(),
                _ => {
                    issues.push(Issue::warning(
                        "non_surface_primitive",
                        format!(
                            "node {} primitive {} omitted from geometric analysis",
                            node.index(),
                            primitive.index()
                        ),
                    ));
                    continue;
                }
            };
            let skin = node.skin().map(|s| s.index());
            let mut influences = vec![Vec::new(); positions.len()];
            if let Some(skin_index) = skin {
                for set in 0..=1 {
                    match (reader.read_joints(set), reader.read_weights(set)) {
                        (Some(joints), Some(weights)) => {
                            let joints: Vec<_> = joints.into_u16().collect();
                            let weights: Vec<_> = weights.into_f32().collect();
                            ensure!(
                                joints.len() == positions.len() && weights.len() == positions.len(),
                                "skin_attribute_length_mismatch"
                            );
                            for ((dst, joints), weights) in
                                influences.iter_mut().zip(joints).zip(weights)
                            {
                                for (&index, &weight) in joints.iter().zip(&weights) {
                                    ensure!(
                                        weight.is_finite() && weight >= 0.0,
                                        "invalid_skin_weight"
                                    );
                                    if weight == 0.0 {
                                        continue;
                                    }
                                    let &joint = skins[skin_index]
                                        .joints
                                        .get(index as usize)
                                        .context("skin_joint_index_out_of_bounds")?;
                                    dst.push(Influence {
                                        joint,
                                        weight: f64::from(weight),
                                    });
                                }
                            }
                        }
                        (None, None) if set == 1 => (),
                        _ => bail!("missing_or_inconsistent_skin_attributes"),
                    }
                }
                for weights in &influences {
                    let sum: f64 = weights.iter().map(|w| w.weight).sum();
                    ensure!(
                        (sum - 1.0).abs() < 1e-3,
                        "skin_weights_not_normalized: sum={sum}"
                    );
                }
            } else {
                influences.clear();
            }
            // The limit counts the values a file stores: sparse targets (the
            // usual case for local body-shape morphs) only their moved
            // vertices. Expanded in memory they are bounded separately.
            let stored: usize = primitive
                .morph_targets()
                .map(|t| {
                    t.positions().map_or(0, |a| {
                        a.sparse().map_or(a.count(), |s| s.count() as usize)
                    })
                })
                .sum();
            morph_vertex_count += stored;
            morph_dense_count += primitive.morph_targets().len() * positions.len();
            ensure!(
                primitive.morph_targets().len() <= 64
                    && morph_vertex_count <= 12_000_000
                    && morph_dense_count <= 64_000_000,
                "resource_limit: morph geometry exceeds limit"
            );
            let morph_targets: Vec<Vec<Vec3>> = reader
                .read_morph_targets()
                .map(|(p, _, _)| {
                    p.map(|p| p.map(|v| v.map(f64::from)).collect())
                        .unwrap_or_else(|| vec![[0.0; 3]; positions.len()])
                })
                .collect();
            ensure!(
                morph_targets.iter().all(
                    |t| t.len() == positions.len() && t.iter().flatten().all(|v| v.is_finite())
                ),
                "invalid_morph_geometry"
            );
            let morph_weights: Vec<f64> = node
                .weights()
                .or_else(|| mesh.weights())
                .map(|w| w.iter().copied().map(f64::from).collect())
                .unwrap_or_else(|| vec![0.0; morph_targets.len()]);
            ensure!(
                morph_weights.len() == morph_targets.len()
                    && morph_weights.iter().all(|v| v.is_finite()),
                "invalid_default_morph_weights"
            );
            meshes.push(Mesh {
                name: mesh.name().unwrap_or("mesh").into(),
                node: node.index(),
                primitive: primitive.index(),
                positions,
                triangles,
                skin,
                influences,
                morph_targets,
                morph_weights,
            });
        }
    }
    let mut clips = Vec::new();
    for animation in document.animations() {
        let mut tracks = Vec::new();
        let mut morph_tracks = Vec::new();
        let mut collisions = 0;
        let mut max_collision_delta = 0.0_f64;
        for channel in animation.channels() {
            let reader = channel.reader(|b| buffers.get(b.index()).map(Vec::as_slice));
            let times: Vec<f64> = reader
                .read_inputs()
                .context("missing animation input")?
                .map(f64::from)
                .collect();
            let interpolation = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::Linear => Interpolation::Linear,
                gltf::animation::Interpolation::Step => Interpolation::Step,
                gltf::animation::Interpolation::CubicSpline => Interpolation::CubicSpline,
            };
            let (property, values) =
                match reader.read_outputs().context("missing animation output")? {
                    gltf::animation::util::ReadOutputs::Translations(v) => (
                        Property::Translation,
                        v.map(|p| [p[0] as f64, p[1] as f64, p[2] as f64, 0.0])
                            .collect(),
                    ),
                    gltf::animation::util::ReadOutputs::Rotations(v) => (
                        Property::Rotation,
                        v.into_f32().map(|q| q.map(f64::from)).collect(),
                    ),
                    gltf::animation::util::ReadOutputs::Scales(v) => (
                        Property::Scale,
                        v.map(|p| [p[0] as f64, p[1] as f64, p[2] as f64, 0.0])
                            .collect(),
                    ),
                    gltf::animation::util::ReadOutputs::MorphTargetWeights(v) => {
                        let values: Vec<_> = v.into_f32().map(f64::from).collect();
                        let stride = if interpolation == Interpolation::CubicSpline {
                            3
                        } else {
                            1
                        };
                        let target_node = channel.target().node();
                        let target_count = target_node
                            .mesh()
                            .context("morph channel targets a node without a mesh")?
                            .primitives()
                            .next()
                            .context("empty morph mesh")?
                            .morph_targets()
                            .count();
                        ensure!(
                            target_count > 0
                                && target_count <= 64
                                && values.len() == times.len() * stride * target_count,
                            "morph_animation_size_mismatch"
                        );
                        let mut track = MorphTrack {
                            node: target_node.index(),
                            interpolation,
                            times,
                            values: values
                                .chunks_exact(target_count)
                                .map(|v| v.to_vec())
                                .collect(),
                        };
                        let (removed, delta) = track.merge_time_collisions(1e-5)?;
                        collisions += removed;
                        max_collision_delta = max_collision_delta.max(delta);
                        morph_tracks.push(track);
                        continue;
                    }
                };
            let mut track = Track {
                joint: channel.target().node().index(),
                property,
                interpolation,
                times,
                values,
            };
            let (removed, delta) = track.merge_time_collisions(1e-5).with_context(|| {
                format!(
                    "animation {} node {} {:?}",
                    animation.name().unwrap_or("unnamed"),
                    track.joint,
                    track.property
                )
            })?;
            collisions += removed;
            max_collision_delta = max_collision_delta.max(delta);
            tracks.push(track);
        }
        let clip = MotionClip {
            name: animation
                .name()
                .map(str::to_owned)
                .unwrap_or_else(|| format!("animation_{}", animation.index())),
            tracks,
            morph_tracks,
        };
        clip.validate(&skeleton)?;
        if collisions > 0 {
            issues.push(Issue::warning("near_equal_time_collisions_merged_in_ir",format!("{}: merged {collisions} duplicate keys, maximum component delta {max_collision_delta}; last near-equivalent value retained in skeletal IR, source animation bytes preserved",clip.name)));
        }
        clips.push(clip);
    }
    // Pack all existing buffers and images without decoding or changing their pixels.
    let mut json = raw;
    let mut binary = Vec::new();
    let mut offsets = Vec::new();
    for buffer in &buffers {
        while !binary.len().is_multiple_of(4) {
            binary.push(0);
        }
        offsets.push(binary.len());
        binary.extend_from_slice(buffer);
    }
    if let Some(views) = json.get_mut("bufferViews").and_then(Value::as_array_mut) {
        for view in views {
            let buffer = view["buffer"].as_u64().context("invalid buffer view")? as usize;
            let offset = view["byteOffset"].as_u64().unwrap_or(0) as usize;
            view["byteOffset"] = json!(offset + offsets[buffer]);
            view["buffer"] = json!(0);
        }
    }
    let image_count = json
        .get("images")
        .and_then(Value::as_array)
        .map_or(0, Vec::len);
    for i in 0..image_count {
        if let Some(uri) = json["images"][i]["uri"].as_str().map(str::to_owned) {
            let data = decode_uri(base, &uri)?;
            total_bytes += data.len();
            ensure!(
                total_bytes <= MAX_FILE_BYTES * 2,
                "resource_limit: combined asset exceeds 512 MiB"
            );
            hash.update((data.len() as u64).to_le_bytes());
            hash.update(&data);
            let mime = if data.starts_with(&[137, 80, 78, 71]) {
                "image/png"
            } else if data.starts_with(&[255, 216, 255]) {
                "image/jpeg"
            } else if data.starts_with(b"RIFF") && data.get(8..12) == Some(b"WEBP") {
                "image/webp"
            } else {
                bail!("unsupported_external_image_format");
            };
            while !binary.len().is_multiple_of(4) {
                binary.push(0);
            }
            let offset = binary.len();
            binary.extend_from_slice(&data);
            let views = json
                .as_object_mut()
                .context("invalid glTF object")?
                .entry("bufferViews")
                .or_insert_with(|| json!([]))
                .as_array_mut()
                .unwrap();
            let view_index = views.len();
            views.push(json!({"buffer":0,"byteOffset":offset,"byteLength":data.len()}));
            let image = json["images"][i]
                .as_object_mut()
                .context("invalid image object")?;
            image.remove("uri");
            image.insert("bufferView".into(), json!(view_index));
            image.insert("mimeType".into(), json!(mime));
        }
    }
    let source_hash = format!("{:x}", hash.finalize());
    let asset = Asset {
        name: path
            .file_stem()
            .and_then(|x| x.to_str())
            .unwrap_or("asset")
            .into(),
        source_hash,
        skeleton,
        meshes,
        skins,
        clips,
        issues,
    };
    Ok(LoadedAsset {
        asset,
        document: GlbDocument { json, binary },
        source_path: path.to_owned(),
    })
}

pub fn inspect(asset: &Asset) -> Result<Value> {
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "name": asset.name,
        "source_hash": asset.source_hash,
        "bounds": asset.bounds()?,
        "mesh_primitives": asset.meshes.len(),
        "vertices": asset.meshes.iter().map(|m| m.positions.len()).sum::<usize>(),
        "triangles": asset.meshes.iter().map(|m| m.triangles.len()).sum::<usize>(),
        "nodes": asset.skeleton.joints.len(),
        "deform_joints": asset.skeleton.joints.iter().filter(|j|j.deform).count(),
        "skins": asset.skins.len(),
        "joints": asset.skeleton.joints.iter().enumerate().filter(|(_,j)|j.deform).map(|(i,j)|json!({"index":i,"id":j.id,"name":j.name,"parent":j.parent})).collect::<Vec<_>>(),
        "clips": asset.clips.iter().map(|c|json!({"name":c.name,"time_range_s":c.time_range(),"tracks":c.tracks.len(),"morph_tracks":c.morph_tracks.len()})).collect::<Vec<_>>(),
        "issues": asset.issues,
    }))
}
