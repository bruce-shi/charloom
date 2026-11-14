//! Compact hand submeshes for pose-dependent correctives. Every original
//! triangle and vertex attribute is retained exactly; shared border vertices
//! have less than 25% hand influence and stay pinned by the corrective solver.
use crate::hand_shape::HandShapeConstraint;
use anyhow::{Context, Result, ensure};
use charloom_core::*;
use charloom_io::GlbDocument;
use serde_json::{Value, json};
use std::collections::HashSet;

fn layout(a: &Value) -> Result<(usize, usize, u64)> {
    let component = a["componentType"]
        .as_u64()
        .context("partition_attribute_component")?;
    let bytes = match component {
        5120 | 5121 => 1,
        5122 | 5123 => 2,
        5125 | 5126 => 4,
        _ => anyhow::bail!("partition_attribute_component"),
    };
    let width = match a["type"].as_str() {
        Some("SCALAR") => 1,
        Some("VEC2") => 2,
        Some("VEC3") => 3,
        Some("VEC4") => 4,
        _ => anyhow::bail!("partition_attribute_shape"),
    };
    Ok((bytes, width, component))
}
fn attribute_bytes(document: &GlbDocument, id: usize) -> Result<Vec<u8>> {
    let a = &document.json["accessors"][id];
    let (size, width, _) = layout(a)?;
    let item = size * width;
    let count = a["count"].as_u64().context("partition_attribute_count")? as usize;
    let mut result = vec![0; count * item];
    if let Some(v) = a["bufferView"].as_u64() {
        let view = &document.json["bufferViews"][v as usize];
        let start = view["byteOffset"].as_u64().unwrap_or(0) as usize
            + a["byteOffset"].as_u64().unwrap_or(0) as usize;
        let stride = view["byteStride"].as_u64().map_or(item, |n| n as usize);
        for i in 0..count {
            result[i * item..(i + 1) * item].copy_from_slice(
                document
                    .binary
                    .get(start + i * stride..start + i * stride + item)
                    .context("partition_attribute_bounds")?,
            );
        }
    }
    if let Some(s) = a.get("sparse") {
        let iv = &document.json["bufferViews"][s["indices"]["bufferView"]
            .as_u64()
            .context("partition_sparse_indices")?
            as usize];
        let vv = &document.json["bufferViews"][s["values"]["bufferView"]
            .as_u64()
            .context("partition_sparse_values")?
            as usize];
        let io = iv["byteOffset"].as_u64().unwrap_or(0) as usize
            + s["indices"]["byteOffset"].as_u64().unwrap_or(0) as usize;
        let vo = vv["byteOffset"].as_u64().unwrap_or(0) as usize
            + s["values"]["byteOffset"].as_u64().unwrap_or(0) as usize;
        let size = match s["indices"]["componentType"].as_u64() {
            Some(5121) => 1,
            Some(5123) => 2,
            Some(5125) => 4,
            _ => anyhow::bail!("partition_sparse_index_type"),
        };
        for i in 0..s["count"].as_u64().context("partition_sparse_count")? as usize {
            let b = document
                .binary
                .get(io + i * size..io + (i + 1) * size)
                .context("partition_sparse_index_bounds")?;
            let at = match size {
                1 => b[0] as usize,
                2 => u16::from_le_bytes(b.try_into().unwrap()) as usize,
                _ => u32::from_le_bytes(b.try_into().unwrap()) as usize,
            };
            ensure!(at < count, "partition_sparse_index_range");
            result[at * item..(at + 1) * item].copy_from_slice(
                document
                    .binary
                    .get(vo + i * item..vo + (i + 1) * item)
                    .context("partition_sparse_value_bounds")?,
            );
        }
    }
    Ok(result)
}
pub(crate) fn subset_accessor(
    document: &mut GlbDocument,
    id: usize,
    vertices: &[usize],
) -> Result<usize> {
    let mut accessor = document.json["accessors"][id].clone();
    let (size, width, component) = layout(&accessor)?;
    let item = size * width;
    let source = attribute_bytes(document, id)?;
    let mut bytes = Vec::with_capacity(vertices.len() * item);
    for &v in vertices {
        bytes.extend_from_slice(
            source
                .get(v * item..(v + 1) * item)
                .context("partition_vertex_range")?,
        );
    }
    let view = document.append_view(&bytes, Some(34962));
    let a = accessor
        .as_object_mut()
        .context("partition_accessor_object")?;
    a.insert("bufferView".into(), json!(view));
    a.insert("count".into(), json!(vertices.len()));
    a.remove("byteOffset");
    a.remove("sparse");
    if a.contains_key("min") || a.contains_key("max") {
        let mut lo = vec![f64::INFINITY; width];
        let mut hi = vec![f64::NEG_INFINITY; width];
        for row in bytes.chunks_exact(item) {
            for k in 0..width {
                let b = &row[k * size..(k + 1) * size];
                let value = match component {
                    5120 => b[0] as i8 as f64,
                    5121 => b[0] as f64,
                    5122 => i16::from_le_bytes(b.try_into().unwrap()) as f64,
                    5123 => u16::from_le_bytes(b.try_into().unwrap()) as f64,
                    5125 => u32::from_le_bytes(b.try_into().unwrap()) as f64,
                    _ => f32::from_le_bytes(b.try_into().unwrap()) as f64,
                };
                lo[k] = lo[k].min(value);
                hi[k] = hi[k].max(value);
            }
        }
        a.insert("min".into(), json!(lo));
        a.insert("max".into(), json!(hi));
    }
    let index = document.array("accessors").len();
    document.array("accessors").push(accessor);
    Ok(index)
}
fn subset_mesh(
    document: &mut GlbDocument,
    source: &Mesh,
    json_mesh: usize,
    triangles: &[[u32; 3]],
    name: &str,
) -> Result<(Mesh, usize)> {
    let mut used = vec![false; source.positions.len()];
    for t in triangles {
        for &v in t {
            used[v as usize] = true;
        }
    }
    let vertices: Vec<_> = used
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| v.then_some(i))
        .collect();
    let mut remap = vec![usize::MAX; used.len()];
    for (i, &v) in vertices.iter().enumerate() {
        remap[v] = i;
    }
    let mut mesh = source.clone();
    mesh.name = name.into();
    mesh.primitive = 0;
    mesh.positions = vertices.iter().map(|&i| source.positions[i]).collect();
    mesh.influences = vertices
        .iter()
        .map(|&i| source.influences[i].clone())
        .collect();
    mesh.morph_targets = source
        .morph_targets
        .iter()
        .map(|target| vertices.iter().map(|&i| target[i]).collect())
        .collect();
    mesh.triangles = triangles
        .iter()
        .map(|t| t.map(|i| remap[i as usize] as u32))
        .collect();
    let mut definition = document.json["meshes"][json_mesh].clone();
    ensure!(
        definition["primitives"]
            .as_array()
            .is_some_and(|p| p.len() == 1),
        "hand_partition_requires_one_primitive_per_node"
    );
    definition["name"] = json!(name);
    let p = &mut definition["primitives"][0];
    let attributes = p["attributes"]
        .as_object()
        .context("partition_attributes")?
        .clone();
    for (name, id) in attributes {
        p["attributes"][name] = json!(subset_accessor(
            document,
            id.as_u64().context("partition_accessor_index")? as usize,
            &vertices
        )?);
    }
    if let Some(targets) = p.get_mut("targets").and_then(Value::as_array_mut) {
        for target in targets {
            let attributes = target.as_object().context("partition_target")?.clone();
            for (name, id) in attributes {
                target[name] = json!(subset_accessor(
                    document,
                    id.as_u64().context("partition_target_accessor")? as usize,
                    &vertices
                )?);
            }
        }
    }
    let indices: Vec<_> = mesh
        .triangles
        .iter()
        .flatten()
        .flat_map(|i| i.to_le_bytes())
        .collect();
    p["indices"] = json!(document.append_accessor(
        &indices,
        5125,
        "SCALAR",
        mesh.triangles.len() * 3,
        Some(34963),
        None
    ));
    p["mode"] = json!(4);
    let mapping: Vec<_> = vertices
        .iter()
        .flat_map(|i| (*i as u32).to_le_bytes())
        .collect();
    let mapping = document.append_accessor(&mapping, 5125, "SCALAR", vertices.len(), None, None);
    if !definition["extras"].is_object() {
        definition["extras"] = json!({});
    }
    definition["extras"]["charloom_partition"] = json!({"source_mesh":json_mesh,"source_node":source.node,"source_primitive":source.primitive,"source_vertices":mapping,"triangles_unchanged":true});
    let index = document.array("meshes").len();
    document.array("meshes").push(definition);
    Ok((mesh, index))
}

pub fn compact(
    asset: &mut Asset,
    document: &mut GlbDocument,
    clip: &mut MotionClip,
    hand: &HandShapeConstraint,
) -> Result<Value> {
    let names: HashSet<_> = std::iter::once(&hand.wrist)
        .chain(hand.digits.iter().flat_map(|d| &d.joints))
        .collect();
    let family: HashSet<_> = asset
        .skeleton
        .joints
        .iter()
        .enumerate()
        .filter_map(|(i, j)| names.contains(&j.name).then_some(i))
        .collect();
    let mut reports = Vec::new();
    let count = asset.meshes.len();
    for id in 0..count {
        let source = asset.meshes[id].clone();
        if source.skin.is_none() || source.positions.len() < 20000 {
            continue;
        }
        let mass: Vec<_> = source
            .influences
            .iter()
            .map(|r| {
                r.iter()
                    .filter(|w| family.contains(&w.joint))
                    .map(|w| w.weight)
                    .sum::<f64>()
            })
            .collect();
        let (hand_triangles, body_triangles): (Vec<_>, Vec<_>) = source
            .triangles
            .iter()
            .copied()
            .partition(|t| t.iter().any(|&v| mass[v as usize] >= 0.25));
        if hand_triangles.is_empty() || body_triangles.is_empty() {
            continue;
        }
        let json_mesh = document.json["nodes"][source.node]["mesh"]
            .as_u64()
            .context("partition_mesh_node")? as usize;
        let (body, body_json) =
            subset_mesh(document, &source, json_mesh, &body_triangles, &source.name)?;
        let name = format!("{}__{}_surface", source.name, hand.wrist);
        let (mut region, region_json) =
            subset_mesh(document, &source, json_mesh, &hand_triangles, &name)?;
        let node = asset.skeleton.joints.len();
        ensure!(
            node == document.array("nodes").len(),
            "partition_node_index_mismatch"
        );
        region.node = node;
        let mut definition = json!({"name":name,"mesh":region_json,"skin":source.skin.unwrap()});
        if !source.morph_weights.is_empty() {
            definition["weights"] = json!(source.morph_weights);
        }
        document.array("nodes").push(definition);
        let old = &mut document.json["nodes"][source.node];
        old["mesh"] = json!(body_json);
        old.as_object_mut()
            .unwrap()
            .entry("children")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .context("partition_children")?
            .push(json!(node));
        asset.skeleton.joints.push(Joint {
            id: format!("node:{node}"),
            name: name.clone(),
            parent: Some(source.node),
            rest: Transform::default(),
            matrix: None,
            deform: false,
        });
        for c in std::iter::once(&mut *clip).chain(asset.clips.iter_mut()) {
            let copies: Vec<_> = c
                .morph_tracks
                .iter()
                .filter(|t| t.node == source.node)
                .cloned()
                .map(|mut t| {
                    t.node = node;
                    t
                })
                .collect();
            c.morph_tracks.extend(copies);
        }
        if let Some(animations) = document.json["animations"].as_array_mut() {
            for animation in animations {
                if let Some(channels) = animation["channels"].as_array_mut() {
                    let copies: Vec<_> = channels
                        .iter()
                        .filter(|c| {
                            c["target"]["node"].as_u64() == Some(source.node as u64)
                                && c["target"]["path"] == "weights"
                        })
                        .cloned()
                        .map(|mut c| {
                            c["target"]["node"] = json!(node);
                            c
                        })
                        .collect();
                    channels.extend(copies);
                }
            }
        }
        reports.push(json!({"source_node":source.node,"new_hand_node":node,"hand_vertices":region.positions.len(),"body_vertices":body.positions.len(),"triangles_before":source.triangles.len(),"triangles_after":body.triangles.len()+region.triangles.len(),"vertex_attributes":"exact_source_subsets","source_mapping_in_mesh_extras":true}));
        asset.meshes[id] = body;
        asset.meshes.push(region);
    }
    Ok(json!({"partitions":reports,"surface_geometry_unchanged":true}))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_subsets_preserve_attributes_topology_and_absent_targets() {
        let mut document = GlbDocument {
            json: json!({"accessors":[],"bufferViews":[],"meshes":[]}),
            binary: vec![],
        };
        let positions = vec![[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]];
        let bytes: Vec<_> = positions
            .iter()
            .flatten()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect();
        let accessor = document.append_accessor(&bytes, 5126, "VEC3", 4, Some(34962), None);
        document
            .array("meshes")
            .push(json!({"primitives":[{"attributes":{"POSITION":accessor},"material":3}]}));
        let source = Mesh {
            name: "surface".into(),
            node: 2,
            primitive: 0,
            positions,
            triangles: vec![[0, 1, 2], [0, 2, 3]],
            skin: None,
            influences: vec![vec![]; 4],
            morph_targets: vec![],
            morph_weights: vec![],
        };
        let (part, id) = subset_mesh(&mut document, &source, 0, &[[0, 2, 3]], "part").unwrap();
        assert_eq!(
            part.positions,
            vec![
                source.positions[0],
                source.positions[2],
                source.positions[3]
            ]
        );
        assert_eq!(part.triangles, vec![[0, 1, 2]]);
        let p = &document.json["meshes"][id]["primitives"][0];
        assert!(
            p.get("targets").is_none(),
            "A missing target list must not become null"
        );
        assert_eq!(p["material"], 3);
        let actual = attribute_bytes(
            &document,
            p["attributes"]["POSITION"].as_u64().unwrap() as usize,
        )
        .unwrap();
        let expected: Vec<_> = part
            .positions
            .iter()
            .flatten()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect();
        assert_eq!(actual, expected);
        assert_eq!(&document.binary[..bytes.len()], bytes);
    }
}
