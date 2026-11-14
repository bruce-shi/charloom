use crate::{API, model::*};
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Duration,
};

pub fn data_dir() -> PathBuf {
    std::env::var_os("CHARLOOM_DATA_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".charloom")
        })
}
pub fn web_dir() -> PathBuf {
    std::env::var_os("CHARLOOM_WEB_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../viewer/dist"))
}
pub fn check_url(url: &str) -> Result<()> {
    let parsed: axum::http::Uri = url.parse()?;
    ensure!(
        parsed.scheme_str() == Some("http")
            && ["127.0.0.1", "localhost", "[::1]"].contains(&parsed.host().unwrap_or(""))
            && parsed.path() == "/"
            && parsed.query().is_none(),
        "local_http_server_required"
    );
    Ok(())
}
#[derive(Clone)]
pub struct Client {
    pub url: String,
    agent: ureq::Agent,
    token: Option<String>,
}
impl Client {
    pub fn connect(url: &str) -> Result<Self> {
        check_url(url)?;
        let agent = ureq::Agent::config_builder()
            .http_status_as_error(false)
            .timeout_global(Some(Duration::from_secs(1800)))
            .build()
            .into();
        Ok(Self {
            url: url.trim_end_matches('/').into(),
            agent,
            token: std::env::var("CHARLOOM_TOKEN").ok(),
        })
    }
    pub fn from_info(info: &ServiceInfo) -> Result<Self> {
        let mut c = Self::connect(&info.url)?;
        c.token = info.auth_token.clone();
        Ok(c)
    }
    pub fn health(&self) -> Result<ServiceInfo> {
        let agent: ureq::Agent = ureq::Agent::config_builder()
            .timeout_global(Some(Duration::from_secs(1)))
            .build()
            .into();
        let mut request = agent.get(format!("{}{API}/health", self.url));
        if let Some(token) = &self.token {
            request = request.header("authorization", format!("Bearer {token}"));
        }
        let info: ServiceInfo = request.call()?.body_mut().read_json()?;
        ensure!(info.protocol == 1, "local_protocol_mismatch");
        Ok(info)
    }
    pub fn call(&self, request: &Request) -> Result<Value> {
        let mut call = self
            .agent
            .post(format!("{}{API}/action", self.url))
            .header("x-charloom-protocol", "1");
        if let Some(token) = &self.token {
            call = call.header("authorization", format!("Bearer {token}"));
        }
        let mut response = call.send_json(request)?;
        let status = response.status().as_u16();
        let value: Value = response
            .body_mut()
            .with_config()
            .limit(512 * 1024 * 1024)
            .read_json()?;
        if status >= 400 {
            bail!(
                "local_service_error:{status}:{}",
                serde_json::to_string(&value)?
            );
        }
        Ok(value)
    }
    pub fn upload(&self, bytes: &[u8]) -> Result<String> {
        let mut call = self
            .agent
            .post(format!("{}{API}/blobs", self.url))
            .header("x-charloom-protocol", "1")
            .header("content-type", "application/octet-stream");
        if let Some(token) = &self.token {
            call = call.header("authorization", format!("Bearer {token}"));
        }
        let mut response = call.send(bytes)?;
        ensure!(response.status().is_success(), "blob_upload_failed");
        let v: Value = response.body_mut().read_json()?;
        Ok(v["blob"].as_str().context("missing_blob_id")?.into())
    }
    pub fn ensure(directory: Option<PathBuf>, server: Option<String>) -> Result<Self> {
        if let Some(url) = server.or_else(|| std::env::var("CHARLOOM_SERVER_URL").ok()) {
            let dir = directory.clone().unwrap_or_else(data_dir);
            let info = std::fs::read(dir.join("server.json"))
                .ok()
                .and_then(|b| serde_json::from_slice::<ServiceInfo>(&b).ok());
            let c = if let Some(info) = info.filter(|i| i.url == url.trim_end_matches('/')) {
                Self::from_info(&info)?
            } else {
                Self::connect(&url)?
            };
            c.health()?;
            return Ok(c);
        }
        let directory = directory.unwrap_or_else(data_dir);
        std::fs::create_dir_all(&directory)?;
        let directory = directory.canonicalize()?;
        let registry = directory.join("server.json");
        let discover = || -> Option<Self> {
            let info: ServiceInfo = serde_json::from_slice(&std::fs::read(&registry).ok()?).ok()?;
            let c = Self::from_info(&info).ok()?;
            let actual = c.health().ok()?;
            if actual.data_dir == directory {
                Some(c)
            } else {
                None
            }
        };
        if let Some(c) = discover() {
            return Ok(c);
        }
        let current = std::env::current_exe()?;
        let executable = std::env::var_os("CHARLOOM_CLI")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                if current
                    .file_stem()
                    .is_some_and(|s| s.to_string_lossy().contains("charloom-mcp"))
                {
                    current.parent().unwrap().join(if cfg!(windows) {
                        "charloom.exe"
                    } else {
                        "charloom"
                    })
                } else {
                    current
                }
            });
        let log = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(directory.join("server.log"))?;
        let mut child = Command::new(executable);
        child
            .arg("serve")
            .arg("--data-dir")
            .arg(&directory)
            .arg("--port")
            .arg("0")
            .stdin(Stdio::null())
            .stdout(log.try_clone()?)
            .stderr(log);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            child.process_group(0);
        }
        let mut process = child
            .spawn()
            .context("Start charloom serve before connecting")?;
        for _ in 0..100 {
            if let Some(c) = discover() {
                return Ok(c);
            }
            let _ = process.try_wait()?;
            std::thread::sleep(Duration::from_millis(50));
        }
        bail!(
            "local_server_start_timeout; see {}",
            directory.join("server.log").display()
        )
    }
}
pub fn open_browser(url: &str) -> Result<()> {
    let mut cmd = if cfg!(target_os = "macos") {
        Command::new("open")
    } else if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", "start", ""]);
        c
    } else {
        Command::new("xdg-open")
    };
    ensure!(cmd.arg(url).status()?.success(), "browser_open_failed");
    Ok(())
}
pub fn resolve_path(path: &Path) -> Result<PathBuf> {
    path.canonicalize()
        .with_context(|| format!("local path not found: {}", path.display()))
}
