//! Lifecycle adapter for the existing Bun native-job handlers. Public callers
//! use Task; the private worker never owns project documents or authentication.
use crate::{
    model::Task,
    store::{Store, uid, valid_id},
};
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::{
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    time::Duration,
};

struct Process {
    child: Child,
    url: String,
    token: String,
}
pub struct NativeWorker {
    pub source_root: PathBuf,
    data_dir: PathBuf,
    process: Mutex<Option<Process>>,
}
impl Drop for NativeWorker {
    fn drop(&mut self) {
        if let Ok(Some(p)) = self.process.get_mut() {
            let _ = p.child.kill();
            let _ = p.child.wait();
        }
    }
}
impl NativeWorker {
    pub fn new(data_dir: &Path) -> Self {
        Self {
            source_root: std::env::var_os("CHARLOOM_SOURCE_ROOT")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
                .canonicalize()
                .unwrap(),
            data_dir: data_dir.into(),
            process: Mutex::new(None),
        }
    }
    fn connection(&self) -> Result<(String, String)> {
        let mut state = self.process.lock().unwrap();
        if let Some(p) = state.as_mut() {
            if p.child.try_wait()?.is_none() {
                return Ok((p.url.clone(), p.token.clone()));
            }
            *state = None;
        }
        let socket = std::net::TcpListener::bind("127.0.0.1:0")?;
        let port = socket.local_addr()?.port();
        drop(socket);
        let url = format!("http://127.0.0.1:{port}");
        let token = uid();
        let exe = std::env::var_os("CHARLOOM_CLI")
            .map(PathBuf::from)
            .unwrap_or(std::env::current_exe()?);
        let binary = if exe
            .file_stem()
            .is_some_and(|s| s.to_string_lossy().contains("charloom-mcp"))
        {
            exe.parent().unwrap().join("charloom")
        } else {
            exe
        };
        let script = std::env::var_os("CHARLOOM_NATIVE_WORKER")
            .map(PathBuf::from)
            .unwrap_or_else(|| self.source_root.join("viewer/server/worker.ts"));
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.data_dir.join("native-jobs.log"))?;
        let mut child =
            Command::new(std::env::var_os("CHARLOOM_BUN").unwrap_or_else(|| "bun".into()))
                .arg(script)
                .current_dir(&self.source_root)
                .env("CHARLOOM_CLI", binary)
                .env("CHARLOOM_SOURCE_ROOT", &self.source_root)
                .env("CHARLOOM_JOB_DIR", self.data_dir.join("jobs"))
                .env("CHARLOOM_JOB_URL", "/files/local/jobs")
                .env("CHARLOOM_WORKER_PORT", port.to_string())
                .env("CHARLOOM_WORKER_TOKEN", &token)
                .stdin(Stdio::null())
                .stdout(log.try_clone()?)
                .stderr(log)
                .spawn()
                .context("Native tasks require Bun and viewer/server/worker.ts")?;
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_millis(500)))
            .build()
            .into();
        for _ in 0..60 {
            if agent
                .get(format!("{url}/health"))
                .header("authorization", format!("Bearer {token}"))
                .call()
                .is_ok()
            {
                *state = Some(Process {
                    child,
                    url: url.clone(),
                    token: token.clone(),
                });
                return Ok((url, token));
            }
            if let Some(status) = child.try_wait()? {
                bail!("native_worker_start_failed:{status}; see native-jobs.log");
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = child.kill();
        let _ = child.wait();
        bail!("native_worker_start_timeout")
    }
    pub fn call(&self, store: &Store, task: &Task) -> Result<Value> {
        let (method, path, fields, files, body) = match task {
            Task::MotionLibrary => (
                "GET",
                "/api/animations/library".into(),
                vec![],
                vec![],
                None,
            ),
            Task::MotionSource { blob, name } => {
                let name = Path::new(name)
                    .file_name()
                    .context("invalid_motion_name")?
                    .to_string_lossy()
                    .replace(['\r', '\n', '"'], "_");
                (
                    "POST",
                    "/api/animations/sources".into(),
                    vec![],
                    vec![("motion", name, blob)],
                    None,
                )
            }
            Task::MotionStart {
                model,
                source,
                clip,
                root_motion,
            } => (
                "POST",
                "/api/animations/jobs".into(),
                vec![
                    ("source", source.clone()),
                    ("clip", clip.clone()),
                    ("rootMotion", root_motion.to_string()),
                ],
                vec![("model", "target.glb".into(), model)],
                None,
            ),
            Task::MotionCorrect {
                base_job_id,
                contacts,
                surface_floor,
            } => (
                "POST",
                "/api/animations/jobs".into(),
                vec![],
                vec![],
                Some(
                    json!({"baseJobId":base_job_id,"contacts":contacts,"surface_floor":surface_floor}),
                ),
            ),
            Task::MotionStatus { id } => {
                ensure!(valid_id(id), "invalid_job_id");
                (
                    "GET",
                    format!("/api/animations/jobs/{id}"),
                    vec![],
                    vec![],
                    None,
                )
            }
            Task::MotionCancel { id } => {
                ensure!(valid_id(id), "invalid_job_id");
                (
                    "DELETE",
                    format!("/api/animations/jobs/{id}"),
                    vec![],
                    vec![],
                    None,
                )
            }
        };
        let (url, token) = self.connection()?;
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(1800)))
            .build()
            .into();
        let auth = format!("Bearer {token}");
        let address = format!("{url}{path}");
        let mut response = if method == "GET" {
            agent.get(address).header("authorization", &auth).call()?
        } else if method == "DELETE" {
            agent
                .delete(address)
                .header("authorization", &auth)
                .header("origin", &url)
                .call()?
        } else if let Some(body) = body {
            agent
                .post(address)
                .header("authorization", &auth)
                .header("origin", &url)
                .send_json(body)?
        } else {
            let boundary = format!("charloom{}", uid());
            let mut data = Vec::new();
            for (name, value) in fields {
                data.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
            }
            for (name, filename, hash) in files {
                let path = store.blob_path(hash)?;
                data.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n").as_bytes());
                data.extend_from_slice(&std::fs::read(path)?);
                data.extend_from_slice(b"\r\n");
            }
            data.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
            agent
                .post(address)
                .header("authorization", &auth)
                .header("origin", &url)
                .header(
                    "content-type",
                    format!("multipart/form-data; boundary={boundary}"),
                )
                .send(data)?
        };
        let status = response.status();
        let value: Value = response
            .body_mut()
            .with_config()
            .limit(64 * 1024 * 1024)
            .read_json()?;
        ensure!(
            status.is_success(),
            "native_task_error:{status}:{}",
            value["error"]
        );
        Ok(value)
    }
    pub fn execute(
        self: &Arc<Self>,
        store: &Arc<Store>,
        task: Task,
        project: Option<String>,
        revision: Option<u64>,
    ) -> Result<Value> {
        if let Some(id) = &project {
            let p = store.read(id)?;
            ensure!(
                Some(p.revision) == revision,
                "revision_conflict:{}",
                p.revision
            );
        }
        let value = self.call(store, &task)?;
        if matches!(&task, Task::MotionSource { .. }) {
            let _ = store
                .events
                .send(crate::model::Notice::Library("motion".into()));
        }
        if let Some(project) = project {
            let job = value
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| valid_id(id))
                .unwrap_or("latest")
                .to_string();
            let key = format!("task_{job}");
            let record = json!({"task":task,"base_revision":revision,"candidate":value,"user_approved":false});
            store.publish_dataset(&project, &key, record.clone(), "native-task")?;
            if value["status"] == "running" {
                let status = match task {
                    Task::MotionStart { .. } | Task::MotionCorrect { .. } => {
                        Some(Task::MotionStatus { id: job })
                    }
                    _ => None,
                };
                if let Some(status) = status {
                    let worker = self.clone();
                    let store = store.clone();
                    std::thread::spawn(move || {
                        for _ in 0..650 {
                            std::thread::sleep(Duration::from_secs(1));
                            match worker.call(&store, &status) {
                                Ok(v) if v["status"] != "running" => {
                                    let mut updated = record;
                                    updated["candidate"] = v;
                                    let _ = store.publish_dataset(
                                        &project,
                                        &key,
                                        updated,
                                        "native-task",
                                    );
                                    break;
                                }
                                Err(_) => break,
                                _ => {}
                            }
                        }
                    });
                }
            }
        }
        Ok(value)
    }
}
