use anyhow::{Result, ensure};
use charloom_engine::pipeline;
use charloom_io::{inspect, load_gltf, load_motion_asset, write_json};
use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "charloom",
    version,
    about = "Rust geometry, rigging, and animation tools for agents"
)]
struct Cli {
    #[arg(long, global = true, help = "Shared local project data directory")]
    data_dir: Option<PathBuf>,
    #[arg(
        long,
        global = true,
        help = "Connect to an existing loopback Charloom service"
    )]
    server: Option<String>,
    #[arg(
        long,
        global = true,
        help = "Emit compact JSON (default is pretty JSON)"
    )]
    json: bool,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the local project authority and Web IDE.
    Serve {
        #[arg(long, default_value_t = 5274)]
        port: u16,
        #[arg(long)]
        web_dir: Option<PathBuf>,
        #[arg(long)]
        dev_origin: Option<String>,
    },
    /// Start/reuse the local service and open the Web IDE.
    Ide {
        project: Option<String>,
        #[arg(long)]
        tool: Option<String>,
        #[arg(long)]
        no_browser: bool,
    },
    /// Shared on-disk projects, also used by Web IDE and MCP.
    Project {
        #[command(subcommand)]
        command: ProjectCommand,
    },
    Capabilities,
         MhTransfer {
        source: PathBuf,
                 #[arg(long)]
        prior: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        /// Motion files (FBX/GLB) to retarget and check; repeatable. Name = file stem.
        #[arg(long)]
        motion: Vec<PathBuf>,
        /// Vision-LLM guidance.json from `charloom mh-guide` (views.json beside it).
        #[arg(long)]
        guidance: Option<PathBuf>,
    },
    /// Apply sculpted mouth curves directly to a GLB.
    MhFitSculptMouth {
        model: PathBuf,
        curves: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Project a hash-bound lip trace into the transfer's working frame.
    MhMouthTrace {
        run: PathBuf,
        trace: PathBuf,
    },
    /// Assemble both native-head stages using the measured body and mouth.
    MhFitLips {
        run: PathBuf,
        trace: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Bind a reconstructed chart or volume surface to the facial joints and blink target.
    MhBindFace {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Fit measured body controls or repair a selected crossing patch.
    MhBodySurface {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Restore walking from the same bind skeleton on a final surface.
    MhRestoreWalk {
        model: PathBuf,
        original: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Bind a retained sculpt's lip rims, beard and dental arches.
    MhBindMouth {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Embed sampled facial joint and corrective-shape animations in a GLB.
    MhEmbedFace {
        model: PathBuf,
        tracks: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Cut the anatomical prior's head component and extract its skinned collar.
    MhBodyCollar {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Reconstruct a face chart's cranium and fit the skinned neck to the body.
    MhChartShell {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// JSON schema of the mh-transfer config.
    MhTransferSchema,
    /// Fast CPU review renders of an mh-transfer run into RUN_DIR/renders/
    /// (the files scripts/mh-render-review.py writes in Blender).
    MhRender {
        run: PathBuf,
        /// Groups to render: segment, body, dressed, face, walk, hands (default all).
        #[arg(long, value_delimiter = ',')]
        only: Vec<String>,
        /// Supersampling per axis.
        #[arg(long, default_value_t = 2)]
        ssaa: usize,
        /// Reference renders (e.g. Blender's) to put side by side with ours in RUN_DIR/renders-compare/.
        #[arg(long)]
        compare: Option<PathBuf>,
    },
    /// Bake MakeHuman body-shape morphs (build-weight, build-muscle, stomach-pregnant, hips-circ, ...; NAME=VALUE) into an
    /// mh-transfer run's body and re-fit its garments onto the new body:
    /// writes OUT/3-skin/body.glb, OUT/4-garments/character.glb and
    /// OUT/refit.json (poke-through before/after). OUT defaults to
    /// RUN_DIR/refit/<morphs>; render it with `charloom mh-render OUT`.
    MhRefitGarments {
        run: PathBuf,
        /// NAME=VALUE, repeatable: `weight=1`, `belly=-0.5` (the `belly-`
        /// target at 0.5), or an exact target `chest+=0.3`.
        #[arg(long)]
        morph: Vec<String>,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Walk frames for the poke-through check.
        #[arg(long, default_value_t = 6)]
        frames: usize,
    },
    /// Stage 0: render calibrated views + template for a vision LLM's guidance.
    MhGuide {
        source: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// Stage 0b: add face close-up views centred on the eyes already marked in guidance.json.
    MhGuideRefine {
        guidance: PathBuf,
    },
    /// Stage 0 self-check: warnings and overlays (views/check/*.png) for a guidance.json.
    MhGuideCheck {
        guidance: PathBuf,
    },
    /// Extra body views at these yaws (degrees; body_y045, body_y315), for
    /// the bodysuit edits' 3/4 silhouettes.
    MhGuideViews {
        guidance: PathBuf,
        #[arg(long, num_args = 1.., allow_negative_numbers = true, required = true)]
        yaw: Vec<f64>,
    },
    /// Stage 0 close-up: render a calibrated zoom view (and an oblique one) of a
    /// marked landmark or a pixel box into views/ and views.json.
    MhGuideZoom {
        guidance: PathBuf,
        /// Landmark already marked in guidance.json, e.g. eye_r_inner or wrist_l.
        #[arg(long, conflicts_with = "area")]
        landmark: Option<String>,
        /// Pixel box in a view: VIEW X0 Y0 X1 Y1.
        #[arg(long = "box", num_args = 5, allow_negative_numbers = true, value_names = ["VIEW", "X0", "Y0", "X1", "Y1"])]
        area: Option<Vec<String>>,
        /// Yaw angles (degrees) of the extra views around the area; default 30.
        #[arg(long, num_args = 1.., allow_negative_numbers = true)]
        orbit: Option<Vec<f64>>,
    },
         MhTextureExport {
        run: PathBuf,
        /// Patch folder (default RUN_DIR/inpaint).
        #[arg(long)]
        out: Option<PathBuf>,
    },
         MhTextureImport {
        run: PathBuf,
        /// Patch folder (default RUN_DIR/inpaint).
        #[arg(long)]
        patches: Option<PathBuf>,
    },
    /// Run the shared browser/CLI/MCP character authoring request.
    Authoring {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    /// JSON schema for portable character authoring requests.
    AuthoringSchema,
    Inspect {
        input: PathBuf,
    },
    Validate {
        input: PathBuf,
                 #[arg(long)]
        clip: Option<String>,
    },
    Sample {
        input: PathBuf,
        #[arg(long)]
        clip: Option<String>,
        #[arg(long, default_value_t = 0.0)]
        time: f64,
    },
    Roundtrip {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Map {
        source: PathBuf,
        target: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
    },
    Retarget {
        source: PathBuf,
        target: PathBuf,
        #[arg(long)]
        clip: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
    },
    Solve {
        input: PathBuf,
        #[arg(long)]
        effectors: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        clip: Option<String>,
        #[arg(long, default_value_t = 0.0)]
        time: f64,
        #[arg(long)]
        config: Option<PathBuf>,
    },
    Schema {
        #[arg(long)]
        out: Option<PathBuf>,
    },
    Build {
        recipe: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    SkinRepairStart {
        input: PathBuf,
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        out: PathBuf,
    },
    SkinRepairStep {
        session: PathBuf,
        #[arg(long)]
        plan: PathBuf,
    },
    SkinRepairInspect {
        session: PathBuf,
    },
    SurfaceAnalyze {
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    SurfaceContacts {
        input: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum ProjectCommand {
    List,
    Create {
        name: String,
        #[arg(long)]
        id: Option<String>,
    },
    Open {
        path: PathBuf,
        #[arg(long)]
        project: Option<String>,
        #[arg(long)]
        revision: Option<u64>,
        #[arg(long)]
        role: Option<String>,
    },
    Read {
        id: String,
        #[arg(long)]
        pointer: Option<String>,
    },
    Set {
        id: String,
        path: String,
        value: String,
        #[arg(long)]
        revision: u64,
    },
    Remove {
        id: String,
        path: String,
        #[arg(long)]
        revision: u64,
    },
    Edit {
        id: String,
        #[arg(long)]
        changes: String,
        #[arg(long)]
        revision: u64,
    },
    Run {
        id: String,
        #[arg(long)]
        revision: u64,
        #[arg(long)]
        request_json: Option<String>,
        #[arg(long)]
        request_pointer: Option<String>,
        #[arg(long)]
        bindings: Option<String>,
        #[arg(long)]
        result_name: Option<String>,
    },
    /// Send the exact common request JSON; '-' reads standard input.
    Action {
        request: String,
    },
    Blob {
        path: PathBuf,
    },
    Schema,
}
fn json_argument(value: &str) -> Result<Value> {
    let text = if value == "-" {
        std::io::read_to_string(std::io::stdin())?
    } else {
        value.to_owned()
    };
    Ok(serde_json::from_str(&text)?)
}
fn project_request(command: ProjectCommand) -> Result<charloom_local::Request> {
    use charloom_local::{Change, Request};
    Ok(match command {
        ProjectCommand::List => Request::List,
        ProjectCommand::Create { name, id } => Request::Create {
            name,
            id,
            document: None,
            assets: None,
        },
        ProjectCommand::Open {
            path,
            project,
            revision,
            role,
        } => Request::OpenPath {
            path: charloom_local::client::resolve_path(&path)?,
            project,
            expected_revision: revision,
            role,
        },
        ProjectCommand::Read { id, pointer } => Request::Read { id, pointer },
        ProjectCommand::Set {
            id,
            path,
            value,
            revision,
        } => Request::Edit {
            id,
            expected_revision: revision,
            changes: vec![Change::Set {
                path,
                value: json_argument(&value)?,
            }],
            assets: None,
            actor: Some("cli".into()),
            request_id: None,
        },
        ProjectCommand::Remove { id, path, revision } => Request::Edit {
            id,
            expected_revision: revision,
            changes: vec![Change::Remove { path }],
            assets: None,
            actor: Some("cli".into()),
            request_id: None,
        },
        ProjectCommand::Edit {
            id,
            changes,
            revision,
        } => Request::Edit {
            id,
            expected_revision: revision,
            changes: serde_json::from_value(json_argument(&changes)?)?,
            assets: None,
            actor: Some("cli".into()),
            request_id: None,
        },
        ProjectCommand::Run {
            id,
            revision,
            request_json,
            request_pointer,
            bindings,
            result_name,
        } => Request::Run {
            id,
            expected_revision: revision,
            request: request_json.map(|v| json_argument(&v)).transpose()?,
            request_pointer,
            bindings: bindings
                .map(|s| -> Result<Vec<charloom_local::Binding>> {
                    Ok(serde_json::from_value(json_argument(&s)?)?)
                })
                .transpose()?
                .unwrap_or_default(),
            result_name,
        },
        ProjectCommand::Action { request } => serde_json::from_value(json_argument(&request)?)?,
        ProjectCommand::Schema | ProjectCommand::Blob { .. } => unreachable!(),
    })
}

fn run(command: Command, data_dir: Option<PathBuf>, server: Option<String>) -> Result<Value> {
    match command {
        Command::Serve {
            port,
            web_dir,
            dev_origin,
        } => {
            tokio::runtime::Runtime::new()?.block_on(charloom_local::server::serve(
                charloom_local::server::Config {
                    port,
                    data_dir: data_dir.unwrap_or_else(charloom_local::client::data_dir),
                    web_dir: web_dir.unwrap_or_else(charloom_local::client::web_dir),
                    dev_origin,
                },
            ))?;
            Ok(json!({"status":"stopped"}))
        }
        Command::Ide {
            project,
            tool,
            no_browser,
        } => {
            let client = charloom_local::client::Client::ensure(data_dir, server)?;
            let value = client.call(&charloom_local::Request::Preview { id: project, tool })?;
            if !no_browser {
                charloom_local::client::open_browser(value["url"].as_str().unwrap())?;
            }
            Ok(value)
        }
        Command::Project {
            command: ProjectCommand::Blob { path },
        } => {
            let client = charloom_local::client::Client::ensure(data_dir, server)?;
            let bytes = std::fs::read(path)?;
            Ok(json!({"blob":client.upload(&bytes)?,"bytes":bytes.len()}))
        }
        Command::Project {
            command: ProjectCommand::Schema,
        } => Ok(charloom_local::model::schema()),
        Command::Project { command } => charloom_local::client::Client::ensure(data_dir, server)?
            .call(&project_request(command)?),
        Command::Capabilities => {
            let mut c = pipeline::capabilities();
            c["authoring"] = charloom_face_fit::authoring::capabilities();
            c["local_workspace"] = json!({"protocol":1,"storage":"local_service","entry_points":["serve","ide","project"],"schema":"charloom project schema","same_data_for":["web","cli","mcp"]});
            Ok(c)
        }
        Command::Authoring { input, out } => charloom_face_fit::authoring::run_file(&input, &out),
        Command::AuthoringSchema => Ok(serde_json::to_value(schemars::schema_for!(
            charloom_face_fit::authoring::Request
        ))?),
        Command::SurfaceAnalyze { input, out } => {
            charloom_engine::surface_topology::analyze_path(&input, &out)
        }
        Command::SurfaceContacts { input, config, out } => {
            charloom_engine::surface_contacts::audit_path(&input, &read_config(Some(config))?, &out)
        }
        Command::SkinRepairStart { input, config, out } => {
            charloom_engine::skin_repair::start(&input, &read_config(config)?, &out)
        }
        Command::SkinRepairStep { session, plan } => charloom_engine::skin_repair::step(
            &session,
            &serde_json::from_slice(&std::fs::read(plan)?)?,
        ),
        Command::SkinRepairInspect { session } => charloom_engine::skin_repair::inspect(&session),
        Command::Build { recipe, out } => charloom_engine::recipe::build(&recipe, &out),
        Command::Inspect { input } => inspect(&load_motion_asset(input)?),
        Command::Validate { input, clip } => pipeline::validate_path_clip(&input, clip.as_deref()),
        Command::Sample { input, clip, time } => {
            let loaded = load_motion_asset(input)?;
            let asset = &loaded;
            let selected = clip
                .as_ref()
                .map(|name| pipeline::select_clip(asset, name))
                .transpose()?;
            let pose = if let Some(clip) = selected {
                clip.sample(&asset.skeleton, time)?
            } else {
                asset.skeleton.rest_pose()
            };
            let morphs = if let Some(clip) = selected {
                clip.sample_morphs(time)?
            } else {
                Default::default()
            };
            let positions = asset.deformed_positions(&pose, &morphs)?;
            let meshes:Vec<_>=asset.meshes.iter().zip(positions).map(|(mesh,vertices)|{
                let count=vertices.len().min(32);
                json!({"node":mesh.node,"primitive":mesh.primitive,"vertex_count":vertices.len(),"samples":(0..count).map(|n|{
                    let i=n*(vertices.len()-1)/count.saturating_sub(1).max(1);json!({"index":i,"position":vertices[i]})
                }).collect::<Vec<_>>()})
            }).collect();
            Ok(
                json!({"source_hash":asset.source_hash,"time_s":time,"joints":asset.skeleton.positions(&pose)?,"world_matrices":asset.skeleton.world_matrices(&pose)?.iter().map(charloom_core::matrix_array).collect::<Vec<_>>(),"joint_names":asset.skeleton.joints.iter().map(|j|&j.name).collect::<Vec<_>>(),"meshes":meshes,"issues":asset.issues}),
            )
        }
        Command::Roundtrip { input, out } => {
            ensure!(!out.exists(), "output_exists");
            let source = load_gltf(input)?;
            source.document.save(&out)?;
            Ok(json!({"output":out,"inspection":inspect(&load_gltf(&out)?.asset)?}))
        }
        Command::Map {
            source,
            target,
            config,
        } => {
            let cfg: charloom_engine::motion::RetargetConfig = read_config(config)?;
            Ok(serde_json::to_value(
                charloom_engine::motion::match_joints(
                    &load_motion_asset(source)?.skeleton,
                    &load_gltf(target)?.asset.skeleton,
                    &cfg.joint_map,
                )?,
            )?)
        }
        Command::MhTransfer {
            source,
            prior,
            out,
            config,
            motion,
            guidance,
        } => {
            let mut config: charloom_mh::Config = read_config(config)?;
            if guidance.is_some() {
                config.guidance = guidance;
            }
            for path in motion {
                let name = path
                    .file_stem()
                    .map_or("motion".into(), |s| s.to_string_lossy().into_owned());
                config.motions.push(charloom_mh::Motion {
                    path,
                    clip: None,
                    name,
                });
            }
            let report = charloom_mh::run(&source, &prior, &out, &config)?;
            Ok(
                json!({"output": out, "outputs": report["outputs"], "stage1_segment": report["stage1_segment"], "stage2_body_fit": {"inverted_triangles": report["stage2_body_fit"]["inverted_triangles"], "edge_ratio": report["stage2_body_fit"]["distortion_vs_prior_rest"]["edge_ratio"], "scan_skin_to_body_m": report["stage2_body_fit"]["scan_skin_to_body_m"]}, "checks": report["stage5_checks"].as_array().map(|c| c.iter().map(|x| json!({"clip":x["clip"],"inside_fraction":x["inside_fraction_over_frames"]})).collect::<Vec<_>>()), "total_s": report["total_s"]}),
            )
        }
        Command::MhFitSculptMouth { model, curves, out } => {
            let rims = serde_json::from_slice(&std::fs::read(curves)?)?;
            charloom_mh::roster::retained_mouth::apply(&model, rims, &out)
        }
        Command::MhMouthTrace { run, trace } => {
            Ok(json!(charloom_mh::roster::mouth_trace(&run, &trace)?))
        }
        Command::MhFitLips { run, trace, out } => {
            charloom_mh::roster::fit_native(&run, &trace, &out)
        }
        Command::MhBindFace { input, out } => charloom_mh::roster::bind_face(&input, &out),
        Command::MhBodySurface { input, out } => {
            charloom_mh::roster::body_surface::run(&input, &out)
        }
        Command::MhRestoreWalk {
            model,
            original,
            out,
        } => charloom_mh::roster::walk::restore(&model, &original, &out),
        Command::MhBindMouth { input, out } => {
            charloom_mh::roster::retained_mouth::run(&input, &out)
        }
        Command::MhEmbedFace { model, tracks, out } => {
            charloom_mh::roster::clips::embed(&model, &tracks, &out)
        }
        Command::MhBodyCollar { input, out } => charloom_mh::roster::collar::run(&input, &out),
        Command::MhChartShell { input, out } => charloom_mh::roster::chart_shell::run(&input, &out),
        Command::MhRender {
            run,
            only,
            ssaa,
            compare,
        } => charloom_mh::render::run(
            &run,
            &charloom_mh::render::Options {
                only,
                ssaa,
                compare,
            },
        ),
        Command::MhRefitGarments {
            run,
            morph,
            out,
            frames,
        } => charloom_mh::sizing::refit(
            &run,
            &charloom_mh::sizing::parse_morphs(&morph)?,
            out.as_deref(),
            frames,
        ),
        Command::MhGuide { source, out } => charloom_mh::guide::prepare(&source, &out),
        Command::MhGuideRefine { guidance } => charloom_mh::guide::refine(&guidance),
        Command::MhGuideCheck { guidance } => charloom_mh::guide::check(&guidance),
        Command::MhTextureExport { run, out } => charloom_mh::inpaint::export(&run, out.as_deref()),
        Command::MhTextureImport { run, patches } => {
            charloom_mh::inpaint::import(&run, patches.as_deref())
        }
        Command::MhGuideViews { guidance, yaw } => charloom_mh::guide::body_views(&guidance, &yaw),
        Command::MhGuideZoom {
            guidance,
            landmark,
            area,
            orbit,
        } => {
            use charloom_mh::guide::ZoomTarget;
            let target = match (landmark, area) {
                (Some(name), None) => Ok(ZoomTarget::Landmark(name)),
                (None, Some(b)) => (|| {
                    let n = |i: usize| -> anyhow::Result<f64> {
                        b[i].parse()
                            .map_err(|_| anyhow::anyhow!("--box: {} is not a number", b[i]))
                    };
                    Ok(ZoomTarget::Box(b[0].clone(), [n(1)?, n(2)?, n(3)?, n(4)?]))
                })(),
                _ => Err(anyhow::anyhow!(
                    "give --landmark NAME or --box VIEW X0 Y0 X1 Y1"
                )),
            };
            target.and_then(|t| charloom_mh::guide::zoom_orbit(&guidance, &t, orbit.as_deref().unwrap_or(&[30.0])))
        }
        Command::MhTransferSchema => Ok(serde_json::to_value(schemars::schema_for!(
            charloom_mh::Config
        ))?),
        Command::Retarget {
            source,
            target,
            clip,
            out,
            config,
        } => pipeline::retarget_path(&source, &target, &clip, &read_config(config)?, &out),
        Command::Solve {
            input,
            effectors,
            out,
            clip,
            time,
            config,
        } => {
            let effectors = serde_json::from_slice::<Vec<charloom_engine::solve::Effector>>(
                &std::fs::read(effectors)?,
            )?;
            pipeline::solve_path(
                &input,
                clip.as_deref(),
                time,
                &effectors,
                &read_config(config)?,
                &out,
            )
        }
        Command::Schema { out } => {
            let schema = json!({"anatomy_preflight_request":schemars::schema_for!(charloom_core::anatomy::AnatomyRequest),"mh_transfer_config":schemars::schema_for!(charloom_mh::Config),"mh_guidance":schemars::schema_for!(charloom_mh::guide::Guidance),
                "retarget_config":schemars::schema_for!(charloom_engine::motion::RetargetConfig),"effector":schemars::schema_for!(charloom_engine::solve::Effector),"solve_config":schemars::schema_for!(charloom_engine::solve::SolveConfig),"animation_recipe":schemars::schema_for!(charloom_engine::recipe::AnimationRecipe),"repair_config":schemars::schema_for!(charloom_engine::skin_repair::RepairConfig),"repair_plan":schemars::schema_for!(charloom_engine::skin_repair::RepairPlan),"surface_contact_config":schemars::schema_for!(charloom_engine::surface_contacts::SurfaceContactConfig)});
            if let Some(out) = out {
                write_json(&out, &schema)?;
            }
            Ok(schema)
        }
    }
}

fn read_config<T: serde::de::DeserializeOwned + Default>(path: Option<PathBuf>) -> Result<T> {
    Ok(if let Some(path) = path {
        serde_json::from_slice(&std::fs::read(path)?)?
    } else {
        T::default()
    })
}

fn main() {
    let cli = Cli::parse();
    let (value, code) = match run(cli.command, cli.data_dir, cli.server) {
        Ok(result) => (result, 0),
        Err(error) => (
            json!({"error":{"message":format!("{error:#}"),"operation_status":"failed"}}),
            1,
        ),
    };
    println!(
        "{}",
        if cli.json {
            serde_json::to_string(&value)
        } else {
            serde_json::to_string_pretty(&value)
        }
        .expect("JSON response")
    );
    if code != 0 {
        std::process::exit(code);
    }
}
