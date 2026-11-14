use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Change {
    Set { path: String, value: Value },
    Remove { path: String },
}
impl Change {
    pub fn path(&self) -> &str {
        match self {
            Self::Set { path, .. } | Self::Remove { path } => path,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Asset {
    pub blob: String,
    #[serde(default)]
    pub companions: BTreeMap<String, String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RequestStamp {
    pub id: String,
    pub digest: String,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub revision: u64,
    pub updated_at: u64,
    /// The exact Web IDE ProjectManifest. Solver parameters retain their native schemas.
    pub document: Value,
    pub assets: BTreeMap<String, Asset>,
    #[serde(default)]
    pub last_request_id: Option<String>,
    #[serde(default)]
    pub recent_requests: Vec<RequestStamp>,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Summary {
    pub id: String,
    pub name: String,
    pub revision: u64,
    pub updated_at: u64,
}
impl From<&Project> for Summary {
    fn from(p: &Project) -> Self {
        Self {
            id: p.id.clone(),
            name: p.name.clone(),
            revision: p.revision,
            updated_at: p.updated_at,
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Receipt {
    pub project_id: String,
    pub revision: u64,
    pub updated_at: u64,
    pub actor: String,
    pub request_id: String,
    pub changes: Vec<Change>,
    #[serde(default)]
    pub assets_changed: bool,
}
#[derive(Debug, Clone)]
pub enum Notice {
    Project(Receipt),
    Library(String),
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Binding {
    pub target: String,
    pub source: String,
}
/// Native operations use the same request through Web, CLI and MCP. Model and
/// motion payloads are content hashes in the service's immutable blob store.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Task {
    MotionLibrary,
    MotionSource {
        blob: String,
        name: String,
    },
    MotionStart {
        model: String,
        source: String,
        clip: String,
        #[serde(default)]
        root_motion: bool,
    },
    MotionCorrect {
        base_job_id: String,
        contacts: Vec<Value>,
        #[serde(default)]
        surface_floor: Option<Value>,
    },
    MotionStatus {
        id: String,
    },
    MotionCancel {
        id: String,
    },
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum Request {
    List,
    Create {
        name: String,
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        document: Option<Value>,
        #[serde(default)]
        assets: Option<BTreeMap<String, Asset>>,
    },
    Read {
        id: String,
        #[serde(default)]
        pointer: Option<String>,
    },
    Edit {
        id: String,
        expected_revision: u64,
        changes: Vec<Change>,
        #[serde(default)]
        assets: Option<BTreeMap<String, Asset>>,
        #[serde(default)]
        actor: Option<String>,
        #[serde(default)]
        request_id: Option<String>,
    },
    Resources {
        id: String,
        expected_revision: u64,
        assets: BTreeMap<String, Asset>,
        #[serde(default)]
        actor: Option<String>,
    },
    OpenPath {
        path: PathBuf,
        #[serde(default)]
        project: Option<String>,
        #[serde(default)]
        expected_revision: Option<u64>,
        #[serde(default)]
        role: Option<String>,
    },
    Browse {
        #[serde(default)]
        path: Option<PathBuf>,
    },
    Run {
        id: String,
        expected_revision: u64,
        #[serde(default)]
        request: Option<Value>,
        #[serde(default)]
        request_pointer: Option<String>,
        #[serde(default)]
        bindings: Vec<Binding>,
        #[serde(default)]
        result_name: Option<String>,
    },
    Task {
        task: Task,
        #[serde(default)]
        project: Option<String>,
        #[serde(default)]
        expected_revision: Option<u64>,
    },
    Preview {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        tool: Option<String>,
    },
}
/// The same action contract is multiplexed over one authenticated connection.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum SocketClient {
    Action { id: String, request: Request },
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum SocketServer {
    Ready {
        protocol: u32,
    },
    Result {
        id: String,
        value: Value,
    },
    Error {
        id: Option<String>,
        status: u16,
        detail: Value,
    },
    ProjectChanged {
        receipt: Receipt,
    },
    LibraryChanged {
        library: String,
    },
    Resync,
}
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ServiceInfo {
    pub protocol: u32,
    pub pid: u32,
    pub url: String,
    pub data_dir: PathBuf,
    pub web_dir: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub auth_token: Option<String>,
}

pub fn schema() -> Value {
    serde_json::json!({"protocol":1,"request":schemars::schema_for!(Request),"project":schemars::schema_for!(Project),"receipt":schemars::schema_for!(Receipt),"socket_client":schemars::schema_for!(SocketClient),"socket_server":schemars::schema_for!(SocketServer),
        "authoring_request":schemars::schema_for!(charloom_face_fit::authoring::Request),"face_annotation":schemars::schema_for!(charloom_face_fit::Annotation),"project_document_format":"charloom-project","project_document_version":1})
}
