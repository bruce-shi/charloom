//! Import a static rigid prop without re-encoding its meshes, textures or materials.
use crate::{GlbDocument, LoadedAsset};
use anyhow::{Context, Result, ensure};
use charloom_core::Transform;
use serde_json::{Value, json};

fn bump(value: &mut Value, key: &str, offset: usize) -> Result<()> {
    if let Some(index) = value.get_mut(key) {
        let old = index.as_u64().context("invalid glTF resource index")?;
        *index = json!(old + offset as u64);
    }
    Ok(())
}

fn check_extensions(value: &Value) -> Result<()> {
    match value {
        Value::Object(object) => {
            if let Some(extensions) = object.get("extensions").and_then(Value::as_object) {
                for name in extensions.keys() {
                    ensure!(
                        matches!(
                            name.as_str(),
                            "KHR_materials_unlit" | "KHR_texture_transform"
                        ),
                        "unsupported_prop_extension: {name}"
                    );
                }
            }
            for (key, value) in object {
                if key != "extras" {
                    check_extensions(value)?;
                }
            }
        }
        Value::Array(values) => {
            for value in values {
                check_extensions(value)?;
            }
        }
        _ => {}
    }
    Ok(())
}

impl GlbDocument {
    /// Adds a named scene root for an external rigid GLB/glTF prop. Animated or
    /// skinned props must already belong to the character's own document.
    pub fn append_rigid_prop(
        &mut self,
        prop: &LoadedAsset,
        name: &str,
        transform: Transform,
    ) -> Result<usize> {
        ensure!(!name.is_empty() && name.len() <= 128, "invalid_prop_name");
        ensure!(transform.is_invertible(), "invalid_prop_transform");
        ensure!(
            prop.asset.skins.is_empty() && prop.asset.clips.is_empty(),
            "external_prop_must_be_static_rigid"
        );
        ensure!(!prop.asset.meshes.is_empty(), "empty_prop_geometry");
        let mut source = prop.document.json.clone();
        check_extensions(&source)?;
        ensure!(
            source
                .get("cameras")
                .and_then(Value::as_array)
                .is_none_or(Vec::is_empty),
            "prop_cameras_unsupported"
        );
        let names: std::collections::HashSet<_> = self.json["nodes"]
            .as_array()
            .context("missing nodes")?
            .iter()
            .filter_map(|n| n["name"].as_str())
            .collect();
        ensure!(!names.contains(name), "prop_name_already_exists: {name}");
        for (i, node) in source["nodes"]
            .as_array()
            .context("prop missing nodes")?
            .iter()
            .enumerate()
        {
            let child_name = format!("{name}/{}:{i}", node["name"].as_str().unwrap_or("node"));
            ensure!(
                !names.contains(child_name.as_str()),
                "prop_node_name_already_exists"
            );
        }
        while !self.binary.len().is_multiple_of(4) {
            self.binary.push(0);
        }
        let binary_offset = self.binary.len();
        let count = |key: &str| self.json[key].as_array().map_or(0, Vec::len);
        let (views, accessors, images, samplers, textures, materials, meshes, nodes) = (
            count("bufferViews"),
            count("accessors"),
            count("images"),
            count("samplers"),
            count("textures"),
            count("materials"),
            count("meshes"),
            count("nodes"),
        );
        for key in [
            "bufferViews",
            "accessors",
            "images",
            "samplers",
            "textures",
            "materials",
            "meshes",
            "nodes",
        ] {
            if let Some(values) = source.get_mut(key).and_then(Value::as_array_mut) {
                for (i, value) in values.iter_mut().enumerate() {
                    match key {
                        "bufferViews" => {
                            ensure!(
                                value["buffer"].as_u64() == Some(0),
                                "prop_buffer_not_packed"
                            );
                            value["byteOffset"] = json!(
                                value["byteOffset"].as_u64().unwrap_or(0) + binary_offset as u64
                            );
                        }
                        "accessors" => {
                            bump(value, "bufferView", views)?;
                            if let Some(sparse) = value.get_mut("sparse") {
                                bump(&mut sparse["indices"], "bufferView", views)?;
                                bump(&mut sparse["values"], "bufferView", views)?;
                            }
                        }
                        "images" => bump(value, "bufferView", views)?,
                        "textures" => {
                            bump(value, "source", images)?;
                            bump(value, "sampler", samplers)?;
                        }
                        "materials" => {
                            for pointer in [
                                "/pbrMetallicRoughness/baseColorTexture",
                                "/pbrMetallicRoughness/metallicRoughnessTexture",
                                "/normalTexture",
                                "/occlusionTexture",
                                "/emissiveTexture",
                            ] {
                                if let Some(info) = value.pointer_mut(pointer) {
                                    bump(info, "index", textures)?;
                                }
                            }
                        }
                        "meshes" => {
                            for primitive in value["primitives"]
                                .as_array_mut()
                                .context("missing primitives")?
                            {
                                bump(primitive, "indices", accessors)?;
                                bump(primitive, "material", materials)?;
                                for index in primitive["attributes"]
                                    .as_object_mut()
                                    .context("missing attributes")?
                                    .values_mut()
                                {
                                    *index = json!(
                                        index.as_u64().context("invalid attribute index")?
                                            + accessors as u64
                                    );
                                }
                                if let Some(targets) =
                                    primitive.get_mut("targets").and_then(Value::as_array_mut)
                                {
                                    for target in targets {
                                        for index in target
                                            .as_object_mut()
                                            .context("invalid morph target")?
                                            .values_mut()
                                        {
                                            *index = json!(
                                                index.as_u64().context("invalid morph index")?
                                                    + accessors as u64
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        "nodes" => {
                            bump(value, "mesh", meshes)?;
                            if let Some(children) =
                                value.get_mut("children").and_then(Value::as_array_mut)
                            {
                                for child in children {
                                    *child = json!(
                                        child.as_u64().context("invalid child")? + nodes as u64
                                    );
                                }
                            }
                            value["name"] = json!(format!(
                                "{name}/{}:{i}",
                                value["name"].as_str().unwrap_or("node")
                            ));
                        }
                        _ => {}
                    }
                }
            }
        }
        let scene_index = source["scene"].as_u64().unwrap_or(0) as usize;
        let roots = source["scenes"][scene_index]["nodes"]
            .as_array()
            .context("prop needs a default scene")?
            .iter()
            .map(|n| Ok(n.as_u64().context("invalid scene root")? + nodes as u64))
            .collect::<Result<Vec<_>>>()?;
        ensure!(!roots.is_empty(), "empty_prop_scene");
        for key in [
            "bufferViews",
            "accessors",
            "images",
            "samplers",
            "textures",
            "materials",
            "meshes",
            "nodes",
        ] {
            if let Some(values) = source.get_mut(key).and_then(Value::as_array_mut) {
                self.array(key).append(values);
            }
        }
        self.binary.extend_from_slice(&prop.document.binary);
        let root = self.array("nodes").len();
        self.array("nodes").push(json!({"name":name,"children":roots,"translation":transform.translation,"rotation":transform.rotation,"scale":transform.scale}));
        let scene = self.json["scene"].as_u64().unwrap_or(0) as usize;
        self.json["scenes"][scene]["nodes"]
            .as_array_mut()
            .context("character needs default scene")?
            .push(json!(root));
        for key in ["extensionsUsed", "extensionsRequired"] {
            if let Some(values) = source[key].as_array() {
                for value in values {
                    if !self.array(key).contains(value) {
                        self.array(key).push(value.clone());
                    }
                }
            }
        }
        Ok(root)
    }
}
