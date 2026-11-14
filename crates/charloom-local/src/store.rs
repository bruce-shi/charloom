use crate::model::*;
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
    sync::Mutex,
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::broadcast;

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
pub fn uid() -> String {
    uuid::Uuid::new_v4().to_string()
}
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 128
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}
pub fn valid_hash(id: &str) -> bool {
    id.len() == 64
        && id
            .bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
}
pub fn atomic_json(path: &Path, value: &impl serde::Serialize) -> Result<()> {
    let temp = path.with_extension(format!("{}.tmp", uid()));
    let result = (|| -> Result<()> {
        let mut f = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            f.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        // Mesh/morph arrays contain millions of small JSON tokens. Buffer the
        // serializer's writes while retaining flush -> fsync -> atomic rename.
        {
            let mut writer = BufWriter::with_capacity(1024 * 1024, &mut f);
            serde_json::to_writer(&mut writer, value)?;
            writer.write_all(b"\n")?;
            writer.flush()?;
        }
        f.sync_all()?;
        fs::rename(&temp, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}
fn segments(path: &str) -> Result<Vec<String>> {
    ensure!(
        path.starts_with('/') && path.len() <= 2048,
        "invalid_json_pointer"
    );
    path[1..]
        .split('/')
        .map(|s| {
            let mut out = String::new();
            let mut chars = s.chars();
            while let Some(c) = chars.next() {
                if c == '~' {
                    out.push(match chars.next() {
                        Some('0') => '~',
                        Some('1') => '/',
                        _ => bail!("invalid_json_pointer_escape"),
                    });
                } else {
                    out.push(c);
                }
            }
            ensure!(
                !["__proto__", "prototype", "constructor"].contains(&out.as_str()),
                "reserved_json_key"
            );
            Ok(out)
        })
        .collect()
}
pub fn change(value: &mut Value, operation: &Change) -> Result<()> {
    let path = segments(operation.path())?;
    let (last, parents) = path.split_last().context("empty_json_pointer")?;
    let mut at = value;
    for p in parents {
        at = match at {
            Value::Object(map) => map.get_mut(p).context("pointer_parent_missing")?,
            Value::Array(a) => a
                .get_mut(p.parse::<usize>()?)
                .context("pointer_index_out_of_bounds")?,
            _ => bail!("pointer_parent_not_container"),
        };
    }
    match at {
        Value::Object(map) => match operation {
            Change::Set { value, .. } => {
                map.insert(last.clone(), value.clone());
            }
            Change::Remove { .. } => {
                ensure!(map.remove(last).is_some(), "pointer_missing");
            }
        },
        Value::Array(a) => {
            if last == "-" {
                if let Change::Set { value, .. } = operation {
                    a.push(value.clone());
                } else {
                    bail!("invalid_array_remove");
                }
            } else {
                let i = last.parse::<usize>()?;
                match operation {
                    Change::Set { value, .. } => {
                        ensure!(i < a.len(), "pointer_index_out_of_bounds");
                        a[i] = value.clone();
                    }
                    Change::Remove { .. } => {
                        ensure!(i < a.len(), "pointer_index_out_of_bounds");
                        a.remove(i);
                    }
                }
            }
        }
        _ => bail!("pointer_parent_not_container"),
    }
    Ok(())
}
pub struct Store {
    pub root: PathBuf,
    pub events: broadcast::Sender<Notice>,
    io: Mutex<BTreeMap<String, Summary>>,
    _lease: File,
}
impl Store {
    pub fn new(root: &Path) -> Result<Self> {
        fs::create_dir_all(root)?;
        let root = root.canonicalize()?;
        let lease = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(root.join("server.lock"))?;
        lease
            .try_lock()
            .map_err(|e| anyhow::anyhow!("local_data_in_use: {e}"))?;
        for folder in ["projects", "blobs", "results"] {
            let path = root.join(folder);
            let existed = path.exists();
            fs::create_dir_all(&path)?;
            #[cfg(unix)]
            if !existed {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
            }
        }
        let mut index = BTreeMap::new();
        for entry in fs::read_dir(root.join("projects"))? {
            let entry = entry?;
            let path = entry.path().join("project.json");
            if path.is_file() {
                let p: Project = serde_json::from_slice(&fs::read(path)?)?;
                index.insert(p.id.clone(), Summary::from(&p));
            }
        }
        let (events, _) = broadcast::channel(128);
        Ok(Self {
            root,
            events,
            io: Mutex::new(index),
            _lease: lease,
        })
    }
    fn path(&self, id: &str) -> Result<PathBuf> {
        ensure!(valid_id(id), "invalid_project_id");
        Ok(self.root.join("projects").join(id).join("project.json"))
    }
    pub fn list(&self) -> Result<Vec<Summary>> {
        let mut rows: Vec<_> = self.io.lock().unwrap().values().cloned().collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.updated_at));
        Ok(rows)
    }
    pub fn read(&self, id: &str) -> Result<Project> {
        Ok(serde_json::from_slice(
            &fs::read(self.path(id)?).context("project_not_found")?,
        )?)
    }
    pub fn create(
        &self,
        name: String,
        id: Option<String>,
        document: Option<Value>,
    ) -> Result<Project> {
        self.create_with_assets(name, id, document, None)
    }
    pub fn create_with_assets(
        &self,
        name: String,
        id: Option<String>,
        document: Option<Value>,
        assets: Option<BTreeMap<String, Asset>>,
    ) -> Result<Project> {
        ensure!(
            !name.trim().is_empty() && name.len() <= 256,
            "invalid_project_name"
        );
        let id = id.unwrap_or_else(uid);
        let path = self.path(&id)?;
        let mut index = self.io.lock().unwrap();
        ensure!(!path.exists(), "project_exists");
        fs::create_dir_all(path.parent().unwrap())?;
        let mut doc=document.unwrap_or_else(||json!({"format":"charloom-project","version":1,"id":id,"name":name,"resources":[],"patches":[],"logs":[],"datasets":{}}));
        doc.as_object_mut()
            .context("invalid_project_document")?
            .entry("datasets")
            .or_insert_with(|| json!({}));
        let p = Project {
            id: id.clone(),
            name,
            revision: 0,
            updated_at: now(),
            document: doc,
            assets: assets.unwrap_or_default(),
            last_request_id: None,
            recent_requests: vec![],
        };
        self.validate(&p)?;
        atomic_json(&path, &p)?;
        index.insert(id.clone(), Summary::from(&p));
        let _ = self.events.send(Notice::Project(Receipt {
            project_id: id,
            revision: 0,
            updated_at: p.updated_at,
            actor: "service".into(),
            request_id: uid(),
            changes: vec![],
            assets_changed: true,
        }));
        Ok(p)
    }
    pub fn blob(&self, bytes: &[u8]) -> Result<String> {
        ensure!(bytes.len() <= 512 * 1024 * 1024, "asset_size_limit");
        let hash = format!("{:x}", Sha256::digest(bytes));
        let path = self.root.join("blobs").join(&hash);
        if !path.exists() {
            let temp = path.with_extension(uid());
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?;
            file.write_all(bytes)?;
            file.sync_all()?;
            fs::rename(temp, path)?;
        }
        Ok(hash)
    }
    pub fn blob_path(&self, hash: &str) -> Result<PathBuf> {
        ensure!(valid_hash(hash), "invalid_blob_id");
        let p = self.root.join("blobs").join(hash);
        ensure!(p.is_file(), "blob_not_found");
        Ok(p)
    }
    fn validate(&self, p: &Project) -> Result<()> {
        let d = &p.document;
        ensure!(
            d["format"] == "charloom-project" && d["version"] == 1 && d["id"] == p.id,
            "invalid_project_document"
        );
        ensure!(
            d["name"]
                .as_str()
                .is_some_and(|v| !v.trim().is_empty() && v.len() <= 256),
            "invalid_project_name"
        );
        ensure!(d["name"] == p.name, "project_name_mismatch");
        for (id, asset) in &p.assets {
            ensure!(valid_id(id), "invalid_resource_id");
            self.blob_path(&asset.blob)?;
            for hash in asset.companions.values() {
                self.blob_path(hash)?;
            }
        }
        let resources = d["resources"]
            .as_array()
            .context("project_resources_required")?;
        let mut ids = std::collections::BTreeSet::new();
        for r in resources {
            let id = r["id"].as_str().context("resource_id_required")?;
            ensure!(
                valid_id(id) && ids.insert(id) && p.assets.contains_key(id),
                "resource_asset_missing_or_duplicate"
            );
            ensure!(
                ["model", "source", "clothing", "reference"]
                    .contains(&r["role"].as_str().unwrap_or("")),
                "invalid_resource_role"
            );
        }
        ensure!(d["patches"].is_array(), "project_patches_required");
        if let Some(fit) = d.get("fit").filter(|v| !v.is_null()) {
            let input: charloom_face_fit::Input =
                serde_json::from_value(fit["input"].clone()).context("invalid_shared_fit_input")?;
            let base: Vec<[f64; 3]> = serde_json::from_value(fit["baseVertices"].clone())
                .context("invalid_shared_fit_base")?;
            ensure!(
                base.len() == input.template.len() && input.body_fit.len() == input.template.len(),
                "fit_vertex_count_mismatch"
            );
            ensure!(
                fit["modelResource"]
                    .as_str()
                    .is_some_and(|id| ids.contains(id)),
                "fit_model_resource_missing"
            );
            if let Some(id) = fit.get("sourceResource").filter(|v| !v.is_null()) {
                ensure!(
                    id.as_str().is_some_and(|id| ids.contains(id)),
                    "fit_source_resource_missing"
                );
            }
            let _: Vec<charloom_face_fit::Annotation> =
                serde_json::from_value(fit["annotations"].clone())
                    .context("invalid_shared_face_annotations")?;
            for b in fit["bindings"]
                .as_array()
                .context("fit_bindings_required")?
            {
                let count = b["count"].as_u64().context("invalid_binding_count")? as usize;
                let indices: Vec<usize> = serde_json::from_value(b["indices"].clone())
                    .context("invalid_binding_indices")?;
                let weights: Vec<f64> = serde_json::from_value(b["weights"].clone())
                    .context("invalid_binding_weights")?;
                ensure!(
                    count > 0
                        && indices.len() == weights.len()
                        && indices.len().is_multiple_of(count)
                        && indices.iter().all(|&i| i < base.len()),
                    "invalid_binding_shape"
                );
            }
        }
        for patch in d["patches"].as_array().unwrap() {
            ensure!(
                patch["key"].is_string() && patch["visible"].is_boolean(),
                "invalid_mesh_patch"
            );
            for (name, size) in [("position", 3), ("quaternion", 4), ("scale", 3)] {
                let values = patch[name].as_array().context("invalid_patch_transform")?;
                ensure!(
                    values.len() == size && values.iter().all(Value::is_number),
                    "invalid_patch_transform"
                );
            }
            let geometry = &patch["geometry"];
            let position = &geometry["attributes"]["position"];
            let positions: Vec<f64> = serde_json::from_value(position["array"].clone())
                .context("invalid_patch_positions")?;
            ensure!(
                position["itemSize"] == 3 && positions.len().is_multiple_of(3),
                "invalid_patch_position_shape"
            );
            if !geometry["index"].is_null() {
                let indices: Vec<usize> = serde_json::from_value(geometry["index"].clone())
                    .context("invalid_patch_indices")?;
                ensure!(
                    indices.iter().all(|&i| i < positions.len() / 3),
                    "invalid_patch_indices"
                );
            }
        }
        ensure!(d["logs"].is_array(), "project_logs_required");
        if let Some(g) = d.pointer("/fit/bodyGuidance").filter(|v| !v.is_null()) {
            let _: charloom_face_fit::body_guidance::Guidance =
                serde_json::from_value(g.clone()).context("invalid_shared_body_guidance")?;
        }
        if let Some(a) = d.pointer("/fit/annotations") {
            let _: Vec<charloom_face_fit::Annotation> =
                serde_json::from_value(a.clone()).context("invalid_shared_face_annotations")?;
        }
        if let Some(a) = d.pointer("/fit/input/anatomy").filter(|v| !v.is_null()) {
            let _: charloom_core::anatomy::AnatomyReview =
                serde_json::from_value(a.clone()).context("invalid_shared_anatomy_review")?;
        }
        if let Some(data) = d.get("datasets") {
            ensure!(data.is_object(), "project_datasets_must_be_object");
        }
        Ok(())
    }
    pub fn edit(
        &self,
        id: &str,
        revision: u64,
        changes: Vec<Change>,
        actor: String,
        request_id: String,
    ) -> Result<Receipt> {
        self.commit(id, revision, changes, actor, request_id, None)
    }
    pub fn edit_assets(
        &self,
        id: &str,
        revision: u64,
        changes: Vec<Change>,
        actor: String,
        request_id: String,
        assets: Option<BTreeMap<String, Asset>>,
    ) -> Result<Receipt> {
        self.commit(id, revision, changes, actor, request_id, assets)
    }
    fn commit(
        &self,
        id: &str,
        revision: u64,
        changes: Vec<Change>,
        actor: String,
        request_id: String,
        assets: Option<BTreeMap<String, Asset>>,
    ) -> Result<Receipt> {
        ensure!(
            changes.len() <= 100_000 && actor.len() <= 256 && request_id.len() <= 256,
            "mutation_limit"
        );
        let mut index = self.io.lock().unwrap();
        let mut p = self.read(id)?;
        // Persist a bounded idempotency window so another client's intervening
        // edit cannot cause a lost acknowledgement to append or remove twice.
        let digest = format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(&(&changes, &assets))?)
        );
        let retry = p
            .recent_requests
            .iter()
            .find(|stamp| stamp.id == request_id);
        if let Some(stamp) = retry {
            ensure!(
                stamp.digest == digest,
                "idempotency_key_reused_with_different_changes"
            );
        }
        if retry.is_some() || p.last_request_id.as_deref() == Some(&request_id) {
            return Ok(Receipt {
                project_id: id.into(),
                revision: p.revision,
                updated_at: p.updated_at,
                actor,
                request_id,
                changes: vec![],
                assets_changed: assets.is_some(),
            });
        }
        ensure!(p.revision == revision, "revision_conflict:{}", p.revision);
        for operation in &changes {
            ensure!(
                !["/id", "/format", "/version"].contains(&operation.path()),
                "immutable_project_identity"
            );
            change(&mut p.document, operation)?;
        }
        let assets_changed = assets.is_some();
        if let Some(assets) = assets {
            for (key, value) in assets {
                ensure!(valid_id(&key), "invalid_resource_id");
                self.blob_path(&value.blob)?;
                for blob in value.companions.values() {
                    self.blob_path(blob)?;
                }
                p.assets.insert(key, value);
            }
        }
        p.name = p.document["name"].as_str().unwrap_or("").into();
        p.revision += 1;
        p.updated_at = now();
        p.last_request_id = Some(request_id.clone());
        p.recent_requests.push(RequestStamp {
            id: request_id.clone(),
            digest,
        });
        if p.recent_requests.len() > 256 {
            p.recent_requests.remove(0);
        }
        self.validate(&p)?;
        atomic_json(&self.path(id)?, &p)?;
        index.insert(id.into(), Summary::from(&p));
        let receipt = Receipt {
            project_id: id.into(),
            revision: p.revision,
            updated_at: p.updated_at,
            actor,
            request_id,
            changes,
            assets_changed,
        };
         
        if let Ok(history) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("projects").join(id).join("changes.jsonl"))
        {
            let mut history = BufWriter::with_capacity(64 * 1024, history);
            let _ = serde_json::to_writer(&mut history, &receipt);
            let _ = history.write_all(b"\n");
            let _ = history.flush();
        }
        let _ = self.events.send(Notice::Project(receipt.clone()));
        Ok(receipt)
    }
    pub fn resources(
        &self,
        id: &str,
        revision: u64,
        assets: BTreeMap<String, Asset>,
        actor: String,
    ) -> Result<Receipt> {
        self.commit(id, revision, vec![], actor, uid(), Some(assets))
    }
    pub fn open_path(
        &self,
        path: &Path,
        project: Option<String>,
        revision: Option<u64>,
        role: Option<String>,
    ) -> Result<Project> {
        let path = path.canonicalize().context("local_path_not_found")?;
        ensure!(path.is_file(), "local_path_is_not_file");
        ensure!(
            fs::metadata(&path)?.len() <= 512 * 1024 * 1024,
            "asset_size_limit"
        );
        let bytes = fs::read(&path)?;
        let format = path
            .extension()
            .and_then(|v| v.to_str())
            .unwrap_or("")
            .to_lowercase();
        ensure!(
            ["glb", "gltf", "obj", "fbx", "stl", "json", "geometry"].contains(&format.as_str()),
            "unsupported_local_file"
        );
        let is_new = project.is_none();
        let mut p = if let Some(id) = project {
            let p = self.read(&id)?;
            ensure!(
                Some(p.revision) == revision,
                "revision_conflict:{}",
                p.revision
            );
            p
        } else {
            let id = uid();
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("Untitled")
                .to_string();
            Project {
                id: id.clone(),
                name: name.clone(),
                revision: 0,
                updated_at: now(),
                document: json!({"format":"charloom-project","version":1,"id":id,"name":name,"resources":[],"patches":[],"logs":[],"datasets":{}}),
                assets: BTreeMap::new(),
                last_request_id: None,
                recent_requests: vec![],
            }
        };
        let mut changes = vec![];
        let mut assets = BTreeMap::new();
        let mut entries = p.document["resources"].as_array().unwrap().clone();
        if format == "json" {
            let value: Value = serde_json::from_slice(&bytes)?;
            let input = value.get("input").unwrap_or(&value);
            if input.get("template").is_some()
                && input.get("body_fit").is_some()
                && input.get("source").is_some()
            {
                let _: charloom_face_fit::Input =
                    serde_json::from_value(input.clone()).context("invalid_fit_input")?;
                let mut ids = vec![];
                for (role, positions, triangles, name) in [
                    (
                        "model",
                        &input["body_fit"],
                        &input["triangles"],
                        "Standard_skin_body",
                    ),
                    (
                        "source",
                        &input["source"],
                        &input["landmark_triangles"],
                        "Source_surface",
                    ),
                ] {
                    let points: Vec<[f64; 3]> = serde_json::from_value(positions.clone())?;
                    let points: Vec<_> = points.iter().map(|&[x, y, z]| [x, z, -y]).collect();
                    let data = serde_json::to_vec(
                        &json!({"name":name,"vertices":points,"triangles":triangles}),
                    )?;
                    let id = uid();
                    let blob = self.blob(&data)?;
                    entries.push(json!({"id":id,"name":name,"format":"geometry","role":role,"file":format!("assets/{id}.geometry"),"companions":[]}));
                    assets.insert(
                        id.clone(),
                        Asset {
                            blob,
                            companions: BTreeMap::new(),
                        },
                    );
                    ids.push(id);
                }
                let n = input["body_fit"]
                    .as_array()
                    .context("invalid_fit_input")?
                    .len();
                changes.push(Change::Set{path:"/fit".into(),value:json!({"input":input,"annotations":value.get("annotations").cloned().unwrap_or(json!([])),"baseVertices":input["body_fit"],"modelResource":ids[0],"sourceResource":ids[1],"bindings":[{"name":"Standard_skin_body","key":format!("{}/0",ids[0]),"kind":"skin","count":n,"indices":(0..n).collect::<Vec<_>>(),"weights":vec![1;n],"maxDistance":0}]})});
            } else {
                let mut datasets = p.document["datasets"]
                    .as_object()
                    .cloned()
                    .unwrap_or_default();
                datasets.insert("local_data".into(), value);
                changes.push(Change::Set {
                    path: "/datasets".into(),
                    value: Value::Object(datasets),
                });
            }
        } else {
            let id = uid();
            let blob = self.blob(&bytes)?;
            let mut companions = BTreeMap::new();
            // Browser loaders receive sibling buffers/textures from the same local folder.
            if format != "glb" && format != "geometry" {
                for entry in fs::read_dir(path.parent().unwrap())?.take(512) {
                    let f = entry?.path();
                    if f.is_file()
                        && f != path
                        && fs::metadata(&f)?.len() <= 64 * 1024 * 1024
                        && ["bin", "mtl", "png", "jpg", "jpeg", "webp"].contains(
                            &f.extension()
                                .and_then(|s| s.to_str())
                                .unwrap_or("")
                                .to_lowercase()
                                .as_str(),
                        )
                    {
                        companions.insert(
                            f.file_name().unwrap().to_string_lossy().into_owned(),
                            self.blob(&fs::read(f)?)?,
                        );
                    }
                }
            }
            let role = role.unwrap_or_else(|| "model".into());
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            entries.push(json!({"id":id,"name":name,"format":format,"role":role,"file":format!("assets/{id}.{format}"),"companions":companions.keys().map(|name|json!({"name":name,"file":name})).collect::<Vec<_>>()}));
            assets.insert(id, Asset { blob, companions });
        }
        changes.push(Change::Set {
            path: "/resources".into(),
            value: Value::Array(entries),
        });
        if is_new {
            for op in changes {
                change(&mut p.document, &op)?;
            }
            return self.create_with_assets(p.name, Some(p.id), Some(p.document), Some(assets));
        }
        self.commit(
            &p.id,
            p.revision,
            changes,
            "local-path".into(),
            uid(),
            Some(assets),
        )?;
        p = self.read(&p.id)?;
        Ok(p)
    }
    pub fn publish_dataset(
        &self,
        id: &str,
        key: &str,
        value: Value,
        actor: &str,
    ) -> Result<Receipt> {
        ensure!(valid_id(key), "invalid_dataset_name");
        for _ in 0..10 {
            let p = self.read(id)?;
            let path = if p.document.get("datasets").is_some() {
                format!("/datasets/{key}")
            } else {
                "/datasets".into()
            };
            let value = if path == "/datasets" {
                json!({key:value.clone()})
            } else {
                value.clone()
            };
            match self.edit(
                id,
                p.revision,
                vec![Change::Set { path, value }],
                actor.into(),
                uid(),
            ) {
                Err(e) if e.to_string().starts_with("revision_conflict:") => continue,
                result => return result,
            }
        }
        bail!("dataset_publish_busy")
    }
    pub fn run(
        &self,
        id: &str,
        revision: u64,
        request: Option<Value>,
        pointer: Option<String>,
        bindings: Vec<Binding>,
        name: Option<String>,
    ) -> Result<Value> {
        let p = self.read(id)?;
        ensure!(p.revision == revision, "revision_conflict:{}", p.revision);
        let mut request = if let Some(request) = request {
            request
        } else {
            p.document
                .pointer(pointer.as_deref().context("request_or_pointer_required")?)
                .context("request_pointer_missing")?
                .clone()
        };
        for b in bindings {
            let value = p
                .document
                .pointer(&b.source)
                .context("binding_source_missing")?
                .clone();
            change(
                &mut request,
                &Change::Set {
                    path: b.target,
                    value,
                },
            )?;
        }
        let typed: charloom_face_fit::authoring::Request = serde_json::from_value(request.clone())?;
        let result = charloom_face_fit::authoring::run(typed)?;
        let result_id = uid();
        let saved = json!({"id":result_id,"project_id":id,"base_revision":revision,"request":request,"result":result});
        fs::create_dir_all(self.root.join("results").join(id))?;
        atomic_json(
            &self
                .root
                .join("results")
                .join(id)
                .join(format!("{result_id}.json")),
            &saved,
        )?;
        let name = name.unwrap_or_else(|| "latest_result".into());
        ensure!(valid_id(&name), "invalid_result_name");
        let mut datasets = p.document["datasets"]
            .as_object()
            .cloned()
            .unwrap_or_default();
        datasets.insert(name.clone(), result.clone());
        let changes = vec![
            Change::Set {
                path: "/datasets".into(),
                value: Value::Object(datasets),
            },
            Change::Set {
                path: "/report".into(),
                value: result.get("report").cloned().unwrap_or(json!({})),
            },
        ];
        match self.edit(
            id,
            revision,
            changes,
            "native-authoring".into(),
            result_id.clone(),
        ) {
            Ok(receipt) => Ok(
                json!({"status":"published","receipt":receipt,"result_id":result_id,"result":result,"result_name":name}),
            ),
            Err(e) if e.to_string().starts_with("revision_conflict:") => Ok(
                json!({"status":"stale_candidate","result_id":result_id,"base_revision":revision,"current_revision":self.read(id)?.revision,"result":result}),
            ),
            Err(e) => Err(e),
        }
    }
}
