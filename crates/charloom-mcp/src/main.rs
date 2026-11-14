//! Local stdio MCP adapter. The engine is shared with the CLI.
use charloom_engine::{
    motion::{RetargetConfig, match_joints},
    pipeline, recipe,
    skin_repair::{self, RepairConfig, RepairPlan},
    solve::{Effector, SolveConfig},
};
use charloom_io::{inspect, load_gltf, load_motion_asset};
use rmcp::{
    Json, ServerHandler, ServiceExt,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{Implementation, ServerCapabilities, ServerConfig},
    tool, tool_handler, tool_router,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf, sync::Arc};
use tokio::sync::Semaphore;

#[derive(Debug, Clone)]
struct Charloom {
    tool_router: ToolRouter<Self>,
    workers: Arc<Semaphore>,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct AuthoringFile {
    input: PathBuf,
    out: PathBuf,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct WorkspaceAction {
    #[serde(default)]
    data_dir: Option<PathBuf>,
    #[serde(default)]
    server: Option<String>,
    request: charloom_local::Request,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Input {
    input: PathBuf,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Validate {
    input: PathBuf,
    #[serde(default)]
    clip: Option<String>,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Map {
    source: PathBuf,
    target: PathBuf,
    #[serde(default)]
    joint_map: BTreeMap<String, String>,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Retarget {
    source: PathBuf,
    target: PathBuf,
    clip: String,
    out: PathBuf,
    #[serde(default)]
    config: RetargetConfig,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Solve {
    input: PathBuf,
    out: PathBuf,
    effectors: Vec<Effector>,
    #[serde(default)]
    config: SolveConfig,
    #[serde(default)]
    clip: Option<String>,
    #[serde(default)]
    time: f64,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct Build {
    recipe: PathBuf,
    out: PathBuf,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RepairStart {
    input: PathBuf,
    out: PathBuf,
    #[serde(default)]
    config: RepairConfig,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RepairStep {
    session: PathBuf,
    plan: RepairPlan,
}
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct RepairInspect {
    session: PathBuf,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SurfaceAnalyze {
    input: PathBuf,
    out: PathBuf,
}

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct SurfaceContacts {
    input: PathBuf,
    out: PathBuf,
    config: charloom_engine::surface_contacts::SurfaceContactConfig,
}

impl Charloom {
    async fn compute(
        &self,
        call: impl FnOnce() -> anyhow::Result<Value> + Send + 'static,
    ) -> Result<Json<Value>, String> {
        let permit = self
            .workers
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| e.to_string())?;
        tokio::task::spawn_blocking(move || {
            let _permit = permit;
            call()
        })
        .await
        .map_err(|e| e.to_string())?
        .map(Json)
        .map_err(|e| format!("{e:#}"))
    }
}

#[tool_router(router = tool_router)]
impl Charloom {
    fn new() -> Self {
        Self {
            tool_router: Self::tool_router(),
            workers: Arc::new(Semaphore::new(2)),
        }
    }

    #[tool(
        description = "Report implemented operations and experimental quality limits. Read before processing unfamiliar assets."
    )]
    async fn capabilities(&self) -> Json<Value> {
        let mut c = pipeline::capabilities();
        c["authoring"] = charloom_face_fit::authoring::capabilities();
        c["local_workspace"] = json!({"protocol":1,"tool":"workspace_action","storage":"local_service","web_updates":"live","shared_schema":"workspace_schema"});
        Json(c)
    }
    #[tool(
        description = "Read or modify the same on-disk project used by the Web IDE and CLI. Supports list/create/read/edit/open_path/browse/run/preview. Read the revision before editing; stale writes are rejected. Use canonical fit fields for landmarks, anatomy and fitting parameters, and datasets for visual observations. Changes are pushed live to Web clients. Run results are candidates, not visual approval; preview returns a local session URL."
    )]
    async fn workspace_action(
        &self,
        Parameters(p): Parameters<WorkspaceAction>,
    ) -> Result<Json<Value>, String> {
        self.compute(move || {
            charloom_local::client::Client::ensure(p.data_dir, p.server)?.call(&p.request)
        })
        .await
    }
    #[tool(
        description = "Read the common local-project request, record and mutation schemas, plus the exact shared Rust authoring parameter schema. Used by CLI, Web IDE and workspace_action."
    )]
    async fn workspace_schema(&self) -> Json<Value> {
        Json(charloom_local::model::schema())
    }

    #[tool(
        description = "Run a portable Charloom authoring request JSON file using the exact Rust implementation used by the browser WASM Worker. Supports explicit guided body fitting, face fitting and mesh operations. Anatomy preflight and template operations share reviewed per-part compatibility; unknown, unconfirmed and stale reviews are rejected. Source annotations are bound to geometry hashes; result approval remains pending. Output file must be new."
    )]
    async fn authoring_process(
        &self,
        Parameters(p): Parameters<AuthoringFile>,
    ) -> Result<Json<Value>, String> {
        self.compute(move || charloom_face_fit::authoring::run_file(&p.input, &p.out))
            .await
    }
    #[tool(
        description = "Read the shared browser, CLI and MCP authoring request JSON schema. Body annotations include confirmed landmarks, semantic surface regions and exact geometry hashes."
    )]
    async fn authoring_schema(&self) -> Json<Value> {
        Json(
            serde_json::to_value(schemars::schema_for!(charloom_face_fit::authoring::Request))
                .expect("schema serializes"),
        )
    }

    #[tool(
        description = "Inspect a local GLB/glTF: skeleton, clips, mesh counts, and import diagnostics."
    )]
    async fn asset_inspect(&self, Parameters(p): Parameters<Input>) -> Result<Json<Value>, String> {
        self.compute(move || inspect(&load_motion_asset(p.input)?))
            .await
    }

    #[tool(
        description = "Inspect a source-to-target joint mapping, including unmapped bones and ancestry conflicts."
    )]
    async fn motion_map(&self, Parameters(p): Parameters<Map>) -> Result<Json<Value>, String> {
        self.compute(move || {
            Ok(serde_json::to_value(match_joints(
                &load_motion_asset(p.source)?.skeleton,
                &load_gltf(p.target)?.asset.skeleton,
                &p.joint_map,
            )?)?)
        })
        .await
    }

    #[tool(
        description = "Retarget a named source clip to a target rig using reference poses while preserving target bone lengths. Missing mappings and contact limitations are reported."
    )]
    async fn motion_retarget(
        &self,
        Parameters(p): Parameters<Retarget>,
    ) -> Result<Json<Value>, String> {
        self.compute(move || {
            pipeline::retarget_path(&p.source, &p.target, &p.clip, &p.config, &p.out)
        })
        .await
    }

    #[tool(
        description = "Solve a pose for explicit world-space effectors, with optional joint limits. Unreachable goals return residuals without stretching the skeleton."
    )]
    async fn pose_solve(&self, Parameters(p): Parameters<Solve>) -> Result<Json<Value>, String> {
        self.compute(move || {
            pipeline::solve_path(
                &p.input,
                p.clip.as_deref(),
                p.time,
                &p.effectors,
                &p.config,
                &p.out,
            )
        })
        .await
    }

    #[tool(
        description = "Bake a local JSON animation recipe containing clip layers, masks, retiming and explicit contacts. Relative asset paths resolve from the recipe file."
    )]
    async fn animation_build(
        &self,
        Parameters(p): Parameters<Build>,
    ) -> Result<Json<Value>, String> {
        self.compute(move || recipe::build(&p.recipe, &p.out)).await
    }

    #[tool(
        description = "Validate structural data and bounded sampled skeletal deformations. Does not certify anatomy, physical contacts or visual quality."
    )]
    async fn artifact_validate(
        &self,
        Parameters(p): Parameters<Validate>,
    ) -> Result<Json<Value>, String> {
        self.compute(move || pipeline::validate_path_clip(&p.input, p.clip.as_deref()))
            .await
    }

    #[tool(
        description = "Analyze exact manifold render seams and connected geometric surfaces. Returns geometry-hashed seam groups and component masks without changing geometry. Coincident overlapping sheets and nearby layers are not merged; connectivity is not semantic clothing segmentation."
    )]
    async fn surface_analyze(
        &self,
        Parameters(p): Parameters<SurfaceAnalyze>,
    ) -> Result<Json<Value>, String> {
        self.compute(move || charloom_engine::surface_topology::analyze_path(&p.input, &p.out))
            .await
    }

    #[tool(
        description = "Audit sampled noncoplanar triangle crossings touching explicit geometry-hashed regions. Compares each pose to the rest surface and reports newly crossing face pairs. Excludes coplanar overlap and shared-vertex contact; does not certify collision-free motion, semantic clothing layers or physical simulation. Does not modify the asset."
    )]
    async fn surface_contacts(
        &self,
        Parameters(p): Parameters<SurfaceContacts>,
    ) -> Result<Json<Value>, String> {
        self.compute(move || {
            charloom_engine::surface_contacts::audit_path(&p.input, &p.config, &p.out)
        })
        .await
    }

    #[tool(
        description = "Start a local agent-directed skin repair session. Freeze sampled clips and numeric thresholds; optional round budget; no hardcoded three-round limit. Returns vertex/time/joint hotspots for choosing a repair plan."
    )]
    async fn skin_repair_start(
        &self,
        Parameters(p): Parameters<RepairStart>,
    ) -> Result<Json<Value>, String> {
        self.compute(move || skin_repair::start(&p.input, &p.config, &p.out))
            .await
    }
    #[tool(
        description = "Execute one evidence-backed repair plan at the expected round. Strategies: relax_strain, exclude_influences, weld_seams, update_budget, accept, report_unresolved. Geometry, poses and clips remain fixed. Check triangle strain and exact seam continuity; retain the best candidate and honor the configured optional round budget. Explicit resume_after_review may continue an unresolved session with its remaining budget and archived review; it cannot reset rounds or reopen acceptance. update_budget records a new optional limit (null for agent-directed stopping). continue_from_round plus its expected hash can continue a rejected candidate while preserving the best. Accept recomputes numeric gates and requires visual evidence; user approval remains separate."
    )]
    async fn skin_repair_step(
        &self,
        Parameters(p): Parameters<RepairStep>,
    ) -> Result<Json<Value>, String> {
        self.compute(move || skin_repair::step(&p.session, &p.plan))
            .await
    }
    #[tool(
        description = "Read a repair session's current candidate, diagnostics, remaining correction budget and prior rounds without changing it."
    )]
    async fn skin_repair_inspect(
        &self,
        Parameters(p): Parameters<RepairInspect>,
    ) -> Result<Json<Value>, String> {
        self.compute(move || skin_repair::inspect(&p.session)).await
    }
}

#[tool_handler(router = self.tool_router)]
impl ServerHandler for Charloom {
    fn get_info(&self) -> ServerConfig {
        ServerConfig::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::new("charloom-mcp", env!("CARGO_PKG_VERSION")).with_title("Charloom"))
            .with_instructions("Charloom experimental local engine. Inspect capabilities; preserve source assets. Output directories must be new. Structural validity and visual/anatomical approval are separate. CPU work is bounded to two concurrent operations; in-flight numerical work is not cancellable yet.")
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let service = Charloom::new().serve(rmcp::transport::stdio()).await?;
    service.waiting().await?;
    Ok(())
}
