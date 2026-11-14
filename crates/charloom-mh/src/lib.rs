 pub mod bodysize;
pub mod check;
pub mod creature;
pub mod export;
pub mod face_test;
mod eyeball;
mod graft;
mod piece_islands;
pub mod fingers;
pub mod fit;
pub mod garment;
pub mod geom;
pub mod guide;
pub mod hand_keep;
pub mod head_audit;
pub mod head_keep;
pub mod inpaint;
pub mod lashes;
pub mod legs;
pub mod neckline;
pub mod paint;
pub mod pose;
pub mod prior;
pub mod render;
pub mod roster;
pub mod scan;
pub mod segment;
pub mod shape;
pub mod sizing;
pub mod source;
pub mod strands;
pub mod subdivide;
pub mod symmetry;
pub mod texture;
pub mod transport;
pub mod underwear;

use anyhow::{Context, Result, ensure};
use charloom_core::{Influence, Joint, Skeleton};
use charloom_io::write_json;
use creature::SkinMesh;
use geom::{Weld, vertex_normals};
use prior::{PartKind, Prior};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum Facing {
    /// Character looks along +Z (glTF convention; Tripo and Meshy default).
    PosZ,
    NegZ,
    PosX,
    NegX,
    /// Try all four and keep the one that best overlays the prior.
    Auto,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Motion {
    pub path: PathBuf,
    /// Clip inside the file; first clip when omitted.
    #[serde(default)]
    pub clip: Option<String>,
    pub name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub facing: Facing,
    pub pose_iterations: usize,
    pub fit_iterations: usize,
    pub fit_stiffness_start: f64,
    pub fit_stiffness_end: f64,
    pub skin_weight: f64,
    pub envelope_weight: f64,
    /// Symmetric prior of the body fit (`symmetry.rs`): pulls torso, limb
    /// and scalp vertices with weak evidence towards their mirror partner's
    /// shape. 0 disables it.
    pub symmetry_weight: f64,
    /// Gap kept between body and cloth, metres.
    pub clearance_m: f64,
    /// Estimate the hidden body's size from the clothes (`bodysize.rs`):
    /// the cloth–body gap measured at garment openings (else
    /// `GarmentGuide::gap_mm`, else a default), then soft *shell − gap*
    /// targets under the garments in the shape fit (with limb girths) and,
    /// weakly, the fine fit. Off: the cloth is an upper bound only.
    pub body_size: bool,
    /// Weight of the body-size targets in the fine fit (only around visible
    /// skin while `hidden_shape_weight` is on).
    pub body_fill_weight: f64,
         pub hidden_stiffness: f64,
    /// Fold fairing on the whole body (`FitParams::body_fairing`).
    pub body_fairing: bool,
         pub body_clearance: bool,
    /// CIE ΔE (L halved) from the face colour below which surface is skin.
    pub skin_delta_e: f64,
    pub overrides: Vec<segment::LabelOverride>,
    /// Stage-0 vision guidance (`charloom mh-guide` output filled by an LLM);
    /// `views.json` must sit next to it. Sets facing, landmarks and garments.
    pub guidance: Option<PathBuf>,
    pub texture_size: u32,
    /// Remove body triangles fully covered by clothing in `character.glb`.
    pub hide_covered_body: bool,
    /// Embed the scan's textures in the review pieces (`1-segment/pieces`).
    /// Off by default: the review draws them in flat colours.
    pub piece_textures: bool,
    /// Keep the scan's painted eye (white and iris) inside the marked and
    /// fitted eye openings out of the skin bake, so it is not drawn on the
    /// lids next to MakeHuman's eyeball. The lids, eyeliner and lash line
    /// around the opening are baked from the scan.
    pub exclude_eyes_from_bake: bool,
    /// Fit the prior's MakeHuman shape morphs (belly, girths, head, hands,
    /// face features) before the fine fit. Needs a prior exported with
    /// scripts/mpfb/export-shape-targets.py (body-shape.glb).
    pub shape_fit: bool,
    /// Shape and place each ear to its marked outline (`ear_{l,r}_*`
    /// landmarks) with MakeHuman's ear targets (`ear-targets.glb` next to the
    /// prior), and bake the scan's ear only inside that outline. Off: the
    /// side-map registration to the scan's ear surface.
    pub ear_outline: bool,
         pub suit_contours: bool,
    /// The contour file in the guidance's `views/` the fit reads:
    /// `suit_contours.ensemble.json` (three more edits, weighted by how far
    /// they agree, `scripts/mh-suit-ensemble.py`), or where a character has
    /// none, `suit_contours.json` (one edit, `scripts/mh-suit-contours.py`).
    pub suit_contours_file: String,
    /// Fit the two marked eyes as a mirrored pair about the face midplane
    /// (`fit::eyes::symmetrize`): one opening shape, size and tilt,
    /// symmetric placement, one eyeball scale. Only a disagreement beyond
    /// marking noise that both eyes show consistently is kept (reported in
    /// `stage0_eye_symmetry`). Off: each eye from its own marks.
    pub eye_symmetry: bool,
    /// Open the lids to the marked eye outlines: MakeHuman's lids warped
    /// onto the marks before the fine fit and held there; the eyeball's
    /// cornea may bulge out of the scan's flat painted eye; socket lining
    /// left in front of the eyeball inside the opening goes behind it.
    /// Off: landmark pulls only (the lid rim and lining stayed behind, a
    /// skin shelf over a stylised eye).
    pub eye_openings: bool,
         pub eye_surface_fairing: bool,
    /// Human first: the body is MakeHuman's shape, set by its own controls
    /// (shape morphs within the human ranges, pose, bone lengths) plus only
    /// a smooth residual capped at `fit::HUMAN_RESIDUAL` on visible skin,
    /// none under the clothes; the head scales within human limits. Off: the
    /// per-vertex fit follows the scan wherever it can (cartoon proportions,
    /// and with them lumps, folds and crumpled heads).
    pub human_first: bool,
    /// Free head (`fit::sculpt`): the head's visible skin sculpted onto
    /// the scan after the fit, the measured skull (bald suit outline), eye
    /// openings and landmarks holding. Off: the face stays near MakeHuman's
    /// shape and the ears are rigid MakeHuman patches.
    pub free_head: bool,
         pub hidden_head: bool,
         pub head_fallback_run: Option<PathBuf>,
    /// Local, verified MakeHuman head/neck basis and frozen observations.
    pub anatomical_head: Option<PathBuf>,
         pub anatomical_head_zero: bool,
    /// Body from the guidance (implies `human_first`): the joint marks set
    /// the pose firmly, the face marks place and size the head, MakeHuman's
    /// shape morphs take only visible skin (cloth never shapes the body), and
    /// the marked body outlines size each station by a smooth deformation
    /// (`shape::apply_outlines`); the scan's surface adds only a small capped
    /// residual on visible skin.
    pub guided_body: bool,
    /// Registration (`fit::register`): MakeHuman's rest mesh, reposed by bone
    /// rotations only, takes its shape from one coarse-to-fine solve on scan
    /// skin, the suit outlines, the cloth bound and the landmarks. The morph
    /// fit, the outline stations and the staged fine fit are skipped.
    pub registration: bool,
         pub registration_fallback_m: f64,
    /// Project the scan's painted eye onto MakeHuman's eyeball. Off by
    /// default: the eye is MakeHuman's clean ball (sclera, round iris,
    /// pupil) with only the iris colour, the iris size against the opening
    /// and the gaze taken from the scan (`eyeball`). The projection also
    /// carried lid lines, eyeliner, lid shadow and white rims onto the ball,
    /// and they slid over it when the eye turned.
    pub eye_projection: bool,
    /// Export MakeHuman's eyebrow mesh. Off by default: the scan's painted
    /// brows are baked into the skin and a second pair would double them.
    pub prior_eyebrows: bool,
    /// Head route. `fit`: MakeHuman's head and face rig fitted to the scan,
    /// colour baked into MakeHuman's UVs (the default). `keep_scan`: the
    /// scan's own head (mesh, UVs, texture, painted eyes) cut at the neck,
    /// lofted onto the MakeHuman neck and skinned rigidly to `head` (exact
    /// face, no facial rig). Unset: `guidance.subject.head`, else `fit`.
    pub head: Option<head_keep::HeadRoute>,
    /// Hand route: `fit` (MakeHuman's hand mapped onto the hand's landmarks
    /// and fitted digit by digit) or `keep_scan` (the scan's own hands cut at
    /// the wrists, lofted onto the MakeHuman forearms and skinned to
    /// MakeHuman's hand bones). Unset: `guidance.subject.hands`, else `fit`.
    pub hands: Option<hand_keep::HandRoute>,
    /// MakeHuman eyelash length and opacity (fit head route).
    pub lashes: lashes::Lashes,
    pub motions: Vec<Motion>,
    pub check_frames: usize,
    /// Export MakeHuman's body-shape targets (every shape target of the
    /// prior, `sizing::body_morphs`: build weight and muscle, belly, hips,
    /// limbs, head, face; `+`/`-`) as morph targets on the body and,
    /// transported, on every garment under the same names. Needs a prior with shape targets
    /// (body-shape.glb). Geometry at weight 0 is unchanged.
    pub size_morphs: bool,
    /// Underwear on the naked body (`underwear.rs`): `"auto"` (female:
    /// punkduck sport bra + briefs, male: Mindfront trunks; sex from the
    /// guidance, else the prior's `preparation.json`), `"none"`, or a list
    /// of asset ids (`underwear::CATALOG`). Licensed MakeHuman MHCLO assets
    /// evaluated on the fitted body, rigged to the same skeleton, in
    /// `3-skin/body.glb` only (with `ATTRIBUTION.md`); `character.glb` has
    /// none. Skipped (reported) when `"auto"` finds no assets.
    pub underwear: underwear::Underwear,
    /// Folder with the MakeHuman asset packs (`underwear02/clothes/...`)
    /// and MakeHuman's `base.obj` (or the MPFB source beside it). Unset:
    /// `$CHARLOOM_UNDERWEAR_ASSETS`, else `out/tools/mpfb/underwear-candidates`
    /// in an ancestor of the prior.
    pub underwear_assets: Option<PathBuf>,
    /// Least gap between skin and underwear, metres (fabric closer than
    /// twice this is lifted smoothly, keeping stacked layers in order).
    pub underwear_clearance_m: f64,
    /// A stylised subject: place MakeHuman's knees and ankles on the
    /// centres of the scan's visible legs, grow those legs about their own
    /// centres with a knee girth, and keep the knees and ankles there in the
    /// rig (`legs.rs`). Off, a realistic subject or legs the scan does not
    /// show as skin: closest points and landmarks place the legs, girths
    /// grow about the bones.
    pub legs: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            facing: Facing::PosZ,
            pose_iterations: 30,
            fit_iterations: 40,
            fit_stiffness_start: 6.0,
            fit_stiffness_end: 0.5,
            skin_weight: 1.0,
            envelope_weight: 1.5,
            symmetry_weight: 1.0,
            clearance_m: 0.004,
            body_size: true,
            body_fill_weight: 0.1,
            hidden_stiffness: 1.5,
            body_fairing: true,
            body_clearance: false,
            skin_delta_e: 14.0,
            overrides: vec![],
            guidance: None,
            texture_size: 2048,
            hide_covered_body: true,
            piece_textures: false,
            exclude_eyes_from_bake: true,
            shape_fit: true,
            ear_outline: true,
            suit_contours: true,
            suit_contours_file: "suit_contours.ensemble.json".into(),
            eye_symmetry: true,
            eye_openings: true,
            eye_surface_fairing: false,
            human_first: false,
            free_head: false,
            hidden_head: true,
            head_fallback_run: None,
            anatomical_head: None,
            anatomical_head_zero: false,
            guided_body: false,
            registration: false,
            registration_fallback_m: 0.0,
            eye_projection: false,
            prior_eyebrows: false,
            head: None,
            hands: None,
            lashes: lashes::Lashes::default(),
            motions: vec![],
            check_frames: 24,
            size_morphs: true,
            underwear: underwear::Underwear::default(),
            underwear_assets: None,
            underwear_clearance_m: 0.005,
            legs: true,
        }
    }
}

fn yaw_of(f: Facing) -> f64 {
    use std::f64::consts::{FRAC_PI_2, PI};
    match f {
        Facing::PosZ | Facing::Auto => 0.0,
        Facing::NegZ => PI,
        Facing::PosX => -FRAC_PI_2,
        Facing::NegX => FRAC_PI_2,
    }
}

/// Skin joints plus their non-mesh ancestors, renumbered.
struct Compact {
    skeleton: Skeleton,
    old_to_new: HashMap<usize, usize>,
    new_to_old: Vec<usize>,
}

/// The rig refitted to the fitted body, with the hands' bones carried from
/// their posed world matrices (`fingers::refit_hand_bones`).
fn refit_rig(
    prior: &Prior,
    fitted: &[charloom_core::Vec3],
    eyes: &fit::EyeFit,
    held: &BTreeMap<usize, geom::V>,
    posed_worlds: &[nalgebra::Matrix4<f64>],
    posed_body: &[charloom_core::Vec3],
) -> Result<fit::Rig> {
    let mut rig = fit::refit_rig(prior, fitted, eyes, held)?;
    rig.report["hands_from_pose"] = fingers::refit_hand_bones(
        prior,
        &mut rig.worlds,
        &mut rig.local,
        posed_worlds,
        posed_body,
        fitted,
    );
    Ok(rig)
}

/// Neighbour-averaging passes over each garment's weights (`run`).
const CLOTH_SMOOTHING: usize = 6;

fn compact_skeleton(prior: &Prior, rig: &fit::Rig) -> Compact {
    let joints = &prior.skeleton().joints;
    let mut keep = vec![false; joints.len()];
    for &j in &prior.skin_joints {
        let mut at = Some(j);
        while let Some(k) = at {
            keep[k] = true;
            at = joints[k].parent;
        }
    }
    let new_to_old: Vec<usize> = (0..joints.len()).filter(|&j| keep[j]).collect();
    let old_to_new: HashMap<usize, usize> = new_to_old
        .iter()
        .enumerate()
        .map(|(n, &o)| (o, n))
        .collect();
    let skeleton = Skeleton {
        joints: new_to_old
            .iter()
            .map(|&o| Joint {
                id: joints[o].name.clone(),
                name: joints[o].name.clone(),
                parent: joints[o].parent.map(|p| old_to_new[&p]),
                rest: rig.local[o],
                matrix: None,
                deform: prior.skin_joints.contains(&o),
            })
            .collect(),
    };
    Compact {
        skeleton,
        old_to_new,
        new_to_old,
    }
}

fn remap(rows: &[Vec<Influence>], map: &HashMap<usize, usize>) -> Vec<Vec<Influence>> {
    rows.iter()
        .map(|r| {
            r.iter()
                .map(|i| Influence {
                    joint: map[&i.joint],
                    weight: i.weight,
                })
                .collect()
        })
        .collect()
}

fn label_colors(labels: &[segment::Label]) -> Vec<[f32; 4]> {
    labels
        .iter()
        .map(|l| match l {
            segment::Label::Skin => [0.95, 0.55, 0.35, 1.0],
            segment::Label::Hair => [0.45, 0.2, 0.6, 1.0],
            segment::Label::Garment => [0.2, 0.45, 0.9, 1.0],
        })
        .collect()
}

pub fn run(source: &Path, prior_path: &Path, out: &Path, config: &Config) -> Result<Value> {
    let report = run_once(source, prior_path, out, config)?;
    let residual = &report["stage2_body_fit"]["scan_skin_to_body_m"];
    let (p50, p95) = (residual["p50"].as_f64().unwrap_or(0.0), residual["p95"].as_f64().unwrap_or(0.0));
    if !config.registration || config.registration_fallback_m <= 0.0 || p95 <= config.registration_fallback_m {
        return Ok(report);
    }
    // `run_once` created `out` itself (it refuses an existing directory),
    // so the discarded attempt is this run's own output.
    std::fs::remove_dir_all(out).with_context(|| format!("discard the registered attempt in {}", out.display()))?;
    let staged = Config { registration: false, ..config.clone() };
    let mut report = run_once(source, prior_path, out, &staged)?;
    let note = json!({
        "reason": "registration left the scan's skin too far from the body; refitted with the staged fit",
        "registered_scan_skin_to_body_m": {"p50": p50, "p95": p95},
        "limit_p95_m": config.registration_fallback_m,
    });
    report["registration_fallback"] = note.clone();
    let path = out.join("report.json");
    let mut written: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
    written["registration_fallback"] = note;
    write_json(&path, &written)?;
    Ok(report)
}

fn run_once(source: &Path, prior_path: &Path, out: &Path, config: &Config) -> Result<Value> {
    ensure!(
        !out.exists(),
        "output_exists: {} (choose a new directory)",
        out.display()
    );
    let started = Instant::now();
    let mut timings = serde_json::Map::new();
    let mut lap = |name: &str, t: &mut Instant| {
        timings.insert(name.into(), json!(t.elapsed().as_secs_f64()));
        *t = Instant::now();
    };
    let mut t = Instant::now();
    let mut prior = Prior::load(prior_path).context("load MakeHuman prior")?;
    let eye_fairing_reference = config.eye_surface_fairing
        .then(|| fit::eye_fairing::Reference::new(&prior));
    let raw = scan::RawScan::load(source).context("load source model")?;
    let guidance = config.guidance.as_deref().map(guide::load).transpose()?;
    if let Some((g, _)) = &guidance {
        ensure!(
            g.source_hash == raw.source_hash,
            "guidance_source_mismatch: guidance was prepared for a different file"
        );
    }
    // Creature hands: MakeHuman's fingers past `subject.hand_digits` take no
    // data upstream and are cut from the exported skin (creature.rs).
    let digits = guidance
        .as_ref()
        .map_or([5, 5], |(g, _)| g.subject.digits());
    let cut_bones = creature::missing_bones(|n| prior.by_name.get(n).copied(), digits);
    let no_digit = creature::digit_mask(&prior.body().influences, &cut_bones);
    let body_rest = &prior.body().prim.positions;
    let (lo, hi) = body_rest.iter().fold(
        ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]),
        |(mut l, mut h), p| {
            for k in 0..3 {
                l[k] = l[k].min(p[k]);
                h[k] = h[k].max(p[k]);
            }
            (l, h)
        },
    );
    let center = [(lo[0] + hi[0]) * 0.5, (lo[2] + hi[2]) * 0.5];
    // The scan is scaled to the prior's height; body proportions then come
    // from the fit, so the output keeps metric MakeHuman scale.
    let fit_height = prior.height();
    let (scan, facing_report) = if let Some((g, _)) = &guidance {
        let yaw = guide::facing_yaw(g)?;
        let transform = raw.working_transform(yaw, fit_height, center);
        (
            raw.into_scan(transform),
            json!({"mode":"guidance","front_view":g.front_view,"yaw":yaw}),
        )
    } else if config.facing == Facing::Auto {
        let base = raw.working_transform(0.0, fit_height, center);
        let scan = raw.into_scan(base);
        let costs: Vec<(f64, f64)> = [
            0.0,
            std::f64::consts::PI,
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
        ]
        .iter()
        .map(|&y| (y, pose::facing_cost(&prior, &scan, -y)))
        .collect();
        let best = costs.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
        let rotation = nalgebra::Rotation3::from_axis_angle(&nalgebra::Vector3::y_axis(), best)
            .to_homogeneous();
        let shift =
            nalgebra::Matrix4::new_translation(&nalgebra::Vector3::new(center[0], 0.0, center[1]));
        let transform = shift * rotation * shift.try_inverse().unwrap() * scan.transform;
        let raw_again = scan::RawScan::load(source)?;
        (
            raw_again.into_scan(transform),
            json!({"mode":"auto","costs":costs,"chosen_yaw":best}),
        )
    } else {
        let transform = raw.working_transform(yaw_of(config.facing), fit_height, center);
        (raw.into_scan(transform), json!({"mode":config.facing}))
    };
    lap("load", &mut t);

    // The bodysuit edits' silhouettes (guided): the head's size controls and
    // the body's outline in the fit (`fit::silhouette`).
    let suit_contours = match (&guidance, config.guidance.as_deref()) {
        (Some((_, views)), Some(path)) if config.guided_body && config.suit_contours => {
            guide::load_suit_contours(path, &config.suit_contours_file, views, &scan)
        }
        _ => vec![],
    };
    // ---- Stage 0: resolve vision guidance on the working-frame scan ----
    let resolved = guidance
        .as_ref()
        .map(|(g, views)| guide::resolve(g, views, &scan))
        .transpose()?;
    let landmarks = resolved
        .as_ref()
        .map(|r| r.landmarks.clone())
        .unwrap_or_default();
    // The eyes as a mirrored pair: every later stage (lid fit, eyeball
    // placement, eyeball bake, skin-bake exclusion) reads these marks.
    let (landmarks, eye_symmetry_report) = if config.eye_symmetry {
        fit::eyes::symmetrize(&landmarks)
    } else {
        (landmarks, json!({"applied": false, "reason": "disabled"}))
    };
    let joint_guide = pose::JointGuide {
        points: landmarks.clone(),
        rays: resolved
            .as_ref()
            .map(|r| r.joint_rays.clone())
            .unwrap_or_default(),
        confidence: resolved
            .as_ref()
            .map(|r| r.confidence.clone())
            .unwrap_or_default(),
        stylized: guidance
            .as_ref()
            .is_some_and(|(g, _)| g.subject.kind.starts_with("stylized")),
         
        legs: config.legs && !config.human_first && !config.guided_body,
        guided: config.guided_body,
        extreme: config.guided_body
            && guidance
                .as_ref()
                .is_some_and(|(g, _)| g.subject.proportions() == guide::Proportions::Extreme),
        foot_lengths: ["foot_l", "foot_r"].map(|station| {
            if !config.guided_body {
                return None;
            }
            resolved.as_ref()?.outlines.iter().find(|o| o.station == station && o.kind == 'd').map(|o| {
                let (a, b, d) = (geom::v(o.a), geom::v(o.b), geom::v(o.dir).normalize());
                let span = b - a;
                (span - d * span.dot(&d)).norm()
            })
        }),
    };
    if let Some(r) = &resolved {
        let stage0 = out.join("0-guide");
        std::fs::create_dir_all(&stage0)?;
        write_json(&stage0.join("resolved.json"), &r.report)?;
    }
    lap("guidance", &mut t);

     
    let coarse = pose::fit(
        &prior,
        &scan,
        config.pose_iterations,
        None,
        None,
        &joint_guide,
    )?;
    let coarse_body = prior.skin_mesh(prior.body, &coarse.worlds);
    let first = segment::segment(
        &prior,
        &scan,
        &coarse_body,
        config.skin_delta_e,
        &config.overrides,
        resolved.as_ref(),
    );
    // Props (a bow across the body, a quiver; rigid pieces) are excluded
    // from the pose refinement like hair: an archer's bow next to her hand
    // turned the hand 90° across her hips.
    let pose_labels: Vec<segment::Label> = first
        .triangle_label
        .iter()
        .enumerate()
        .map(|(t, &l)| {
            if first.is_prop(t) {
                segment::Label::Hair
            } else {
                l
            }
        })
        .collect();
    let mut pose_fit = pose::fit(
        &prior,
        &scan,
        config.pose_iterations / 2,
        Some(&pose_labels),
        Some(coarse.pose.clone()),
        &joint_guide,
    )?;
    // Stage 1b: MakeHuman's fingers posed onto the finger landmarks.
    let mut finger_stretch = BTreeMap::new();
    let finger_report = match &resolved {
        Some(r) => fingers::fit(
            &prior,
            &scan,
            &mut pose_fit.pose,
            &r.hand_points,
            digits,
            &mut finger_stretch,
        )?,
        None => json!({"applied": false}),
    };
    pose_fit.worlds = prior.worlds(&pose_fit.pose)?;
    // The mapped fingers' bones stretch their mesh with their lengths.
    fingers::stretch_bones(&mut pose_fit.worlds, &finger_stretch);
    let posed_body = prior.skin_mesh(prior.body, &pose_fit.worlds);
    geom::dump_stage("0-pose", posed_body.iter().copied());
    lap("pose", &mut t);
    let seg = segment::segment(
        &prior,
        &scan,
        &posed_body,
        config.skin_delta_e,
        &config.overrides,
        resolved.as_ref(),
    );
    let mut garments = garment::extract(&scan, &seg);
    lap("segment", &mut t);
    // Skin-coloured texels on hair strands take the hair's colour, in a copy
    // of the scan texture that only the hair pieces use (strands.rs).
    let (hair_images, recolour_report) = strands::recolour_hair(&scan, &seg, fit_height);
    let hair_encoded = strands::encode_images(&scan, &hair_images)?;
    drop(hair_images);
    lap("hair_recolour", &mut t);
    let stage1 = out.join("1-segment");
    std::fs::create_dir_all(stage1.join("pieces"))?;
    {
        // Labels as vertex colours, with the scan's UVs and materials so the
        // review can show the scan texture from the same file.
        let mut b = export::Builder::new();
        let materials = b.import_materials(
            &scan.images,
            &scan.textures,
            &scan.samplers,
            &scan.materials,
        );
        let mut colors = label_colors(&seg.vertex_label);
         
        for (t, tri) in scan.triangles.iter().enumerate() {
            if seg.strand[t] {
                for &i in tri {
                    if seg.vertex_label[i as usize] == segment::Label::Hair {
                        colors[i as usize] = [0.95, 0.35, 0.75, 1.0];
                    }
                }
            }
        }
        let mut by_material: BTreeMap<Option<usize>, Vec<[u32; 3]>> = BTreeMap::new();
        for (t, tri) in scan.triangles.iter().enumerate() {
            by_material
                .entry(scan.triangle_material[t])
                .or_default()
                .push(*tri);
        }
        // Each material's mesh gets only its own vertices: the whole scan's
        // arrays per material made a 113 MB scan with many materials a
        // 685 MB file (over the loader's limit).
        for (material, triangles) in &by_material {
            let mut map = vec![u32::MAX; scan.positions.len()];
            let mut used: Vec<usize> = vec![];
            let local: Vec<[u32; 3]> = triangles
                .iter()
                .map(|t| {
                    t.map(|i| {
                        let i = i as usize;
                        if map[i] == u32::MAX {
                            map[i] = used.len() as u32;
                            used.push(i);
                        }
                        map[i]
                    })
                })
                .collect();
            let pick = |v: &[charloom_core::Vec3]| used.iter().map(|&i| v[i]).collect::<Vec<_>>();
            let uvs: Vec<[f32; 2]> = used.iter().map(|&i| scan.uvs[i]).collect();
            let cols: Vec<[f32; 4]> = used.iter().map(|&i| colors[i]).collect();
            b.add_mesh(&export::OutMesh {
                name: "segmentation",
                positions: &pick(&scan.positions),
                normals: &pick(&scan.normals),
                uvs: Some(&uvs),
                colors: Some(&cols),
                triangles: &local,
                influences: None,
                material: material.map(|m| materials[m]),
            })?;
        }
        b.save(&stage1.join("segmentation.glb"))?;
        let mut b = export::Builder::new();
        let w = Weld::new(&prior.body().bind, 1e-7);
        b.add_mesh(&export::OutMesh {
            name: "posed-prior",
            positions: &posed_body,
            normals: &vertex_normals(&posed_body, &prior.body().prim.triangles, &w),
            uvs: None,
            colors: None,
            triangles: &prior.body().prim.triangles,
            influences: None,
            material: None,
        })?;
        b.save(&stage1.join("posed-prior.glb"))?;
         
        for g in &garments {
            let mut b = export::Builder::new();
            let materials = if config.piece_textures {
                b.import_materials(
                    &scan.images,
                    &scan.textures,
                    &scan.samplers,
                    &scan.materials,
                )
            } else {
                vec![]
            };
            b.add_mesh(&export::OutMesh {
                name: &g.name,
                positions: &g.positions,
                normals: &g.normals,
                uvs: Some(&g.uvs),
                colors: None,
                triangles: &g.triangles,
                influences: None,
                material: g.material.and_then(|m| materials.get(m).copied()),
            })?;
            b.save(&stage1.join("pieces").join(format!("{}.glb", g.name)))?;
        }
    }
    lap("stage1_export", &mut t);

    // ---- Stage 2: body + rig fit ----
    let stylized = guidance
        .as_ref()
        .is_some_and(|(g, _)| g.subject.kind.starts_with("stylized"));
    // Human first: the shape stays within MakeHuman's human ranges whatever
    // the subject (`Config::human_first`).
    let human_first = config.human_first || config.guided_body;
    let shape_stylized = stylized && !human_first;
    // Guided: the LLM's judged proportions set the controls' range and the
    // residual cap (`shape::guided_limits`).
    let extreme = config.guided_body
        && guidance
            .as_ref()
            .is_some_and(|(g, _)| g.subject.proportions() == guide::Proportions::Extreme);
    let guided_limits = config.guided_body.then(|| {
        shape::guided_limits(
            guidance
                .as_ref()
                .map_or(guide::Proportions::Human, |(g, _)| g.subject.proportions()),
        )
    });
    let mut fallback_controls: Option<[f64; 3]> = None;
    let head_fallback = if let Some(run) = &config.head_fallback_run {
        let old_report: Value = serde_json::from_slice(&std::fs::read(run.join("report.json"))?)?;
        ensure!(old_report["prior"].as_str()==prior_path.to_str(), "head fallback uses a different prior");
        ensure!(old_report["prior_hash"] == json!(prior.loaded.asset.source_hash), "head fallback prior bytes differ");
        ensure!(old_report["source_hash"] == json!(scan.source_hash), "head fallback scan bytes differ");
        let frame: Vec<f64> = serde_json::from_value(old_report["working_frame"]["source_to_working"].clone())?;
        ensure!(frame.len()==16 && frame.iter().zip(scan.transform.as_slice()).all(|(a,b)|(a-b).abs()<1e-10), "head fallback working frame differs");
        fallback_controls = Some(serde_json::from_value(old_report["stage2_shape"]["head_from_suit"]["controls"].clone())?);
        let loaded = charloom_io::load_gltf(run.join("2-body-rig/fit-diagnostic.glb"))?;
        let positions = loaded.asset.skinned_positions(&loaded.asset.skeleton.rest_pose())?;
        let meshes = source::read_primitives(&loaded, &positions)?;
        let mesh = meshes.first().context("head fallback has no body")?;
        ensure!(mesh.positions.len()==prior.body().prim.positions.len() && mesh.triangles==prior.body().prim.triangles,
            "head fallback topology mismatch");
        Some((run.display().to_string(),mesh.positions.clone()))
    } else {None};
    let mut params = fit::FitParams {
        scale_range: if stylized { (0.6, 1.8) } else { (0.8, 1.25) },
        iterations: config.fit_iterations,
        stiffness_start: config.fit_stiffness_start,
        stiffness_end: config.fit_stiffness_end,
        clearance_m: config.clearance_m,
        skin_weight: config.skin_weight,
        envelope_weight: config.envelope_weight,
        no_data: no_digit.clone(),
        symmetry: config.symmetry_weight,
        fill: vec![],
        fill_weight: config.body_fill_weight,
        mapped_hands: ["hand_l", "hand_r"].map(|h| !finger_report["hands"][h].is_null()),
        kept_hands: config
            .hands
            .or(guidance.as_ref().and_then(|(g, _)| g.subject.hands))
            == Some(hand_keep::HandRoute::KeepScan),
        finger_tip_radius: ["l", "r"].map(|side| {
            std::array::from_fn(|f| {
                ["tip_radius_m", "mid_radius_m"].map(|key| {
                    finger_report["hands"][format!("hand_{side}")]["fingers"]
                        .as_array()?
                        .iter()
                        .find(|c| c["finger"] == format!("finger{}_{side}", f + 1))?[key]
                        .as_f64()
                })
            })
        }),
        hidden_stiffness: config.hidden_stiffness,
        body_fairing: config.body_fairing,
        body_clearance: config.body_clearance,
        eye_openings: config.eye_openings,
        human_first,
        guided_body: config.guided_body,
        feet_measured: joint_guide.foot_lengths.iter().all(|l| l.is_some()),
        skin_residual: guided_limits.map_or(0.0, |l| l.4),
        extreme,
        bare_feet: ["left", "right"].map(|side| {
            config.guided_body
                && pose_fit.report["feet_in_footwear"]["shoes"].as_array().is_some_and(|shoes| {
                    shoes.iter().any(|s| s["side"] == json!(side) && s["closed"] == json!(false))
                })
        }),
        contours: suit_contours.clone(),
        head_contours: match (&guidance, config.guidance.as_deref()) {
            (Some((_, views)), Some(path)) if config.guided_body && config.suit_contours =>
                guide::load_suit_head_contours(path, "suit_contours.ensemble-all.json", views, &scan),
            _ => vec![],
        },
        free_head: config.free_head,
        hidden_head: config.hidden_head,
        anatomical_head: config.anatomical_head.as_deref().map(|path|
            fit::anatomical_head::load(path, prior_path)).transpose()?,
        anatomical_head_zero: config.anatomical_head_zero,
        head_fallback,
        hidden_neck_limit_m: graft::GAP * fit_height,
        registration: config.registration,
        joint_rays: joint_guide.rays.iter().map(|(name, (origin, direction))| (name.clone(), *origin, *direction)).collect(),
        outlines: resolved.as_ref().map_or(vec![], |r| {
            r.outlines
                .iter()
                .filter_map(|o| {
                    let (m, c, a, b, lo, hi) =
                        shape::outline_frames(&prior, &no_digit, std::slice::from_ref(o))
                            .into_iter()
                            .next()??;
                    Some((m, c, a, b, lo, hi, o.confidence))
                })
                .collect()
        }),
    };
    // Hidden body size from the clothes: gaps at the garment openings, then
    // *shell − gap* targets under the garments (bodysize.rs).
    let size_axes = bodysize::Axes::new(&prior, &pose_fit.worlds);
    let size_samples = bodysize::Samples::new(
        &scan,
        &seg,
        &size_axes,
        &posed_body,
        &prior.body().prim.triangles,
        fit_height,
    );
    let (gaps, gap_profiles) = if config.body_size {
        bodysize::estimate_gaps(
            &seg,
            &size_axes,
            &size_samples,
            &posed_body,
            config.clearance_m,
            fit_height,
        )
    } else {
        (vec![], Default::default())
    };
    let shape_fill = (config.body_size && !config.guided_body).then(|| {
        bodysize::fill_targets(
            &scan,
            &seg,
            &size_axes,
            &gaps,
            &posed_body,
            &prior.body().prim.triangles,
            config.clearance_m,
            fit_height,
            shape_stylized,
        )
    });
    let girth_prior = bodysize::girths(
        &size_axes,
        &bodysize::body_points(&size_axes, &posed_body),
        fit_height,
    );
    // The lash mesh has no shape morphs: it sits on the unmorphed lids.
    let unmorphed_body = prior.body().prim.positions.clone();
    // MakeHuman's own shape targets first: proportions from its morphs, so the
    // fine fit starts from (and regularises towards) a fitting body.
    let shape_report = if config.shape_fit && !config.registration {
        // Mapped hands are sized by their bones, not by hand morphs; so are
        // feet set into oversized footwear (pose.rs).
        let mut skip: Vec<&str> = if finger_report["applied"] == json!(true) {
            fingers::HAND_MORPHS.to_vec()
        } else {
            vec![]
        };
        if pose_fit.report["feet_in_footwear"]["alignment"]
            .as_array()
            .is_some_and(|a| !a.is_empty())
        {
            skip.push("foot-scale");
        }
        // Debugging: `CHARLOOM_SHAPE_SKIP=name,name` leaves those controls out.
        let extra_skip = std::env::var("CHARLOOM_SHAPE_SKIP").unwrap_or_default();
        skip.extend(extra_skip.split(',').map(str::trim).filter(|s| !s.is_empty()));
        shape::fit_shape(
            &mut prior,
            &scan,
            &seg,
            &mut pose_fit.worlds,
            &skip,
            &no_digit,
            shape_fill.as_deref(),
            shape_stylized,
            // Only legs the pose placed on their seen centres (legs.rs): a
            // leg hidden in trousers or armour keeps the bone-centred
            // girths (centred, the guard's walk poke-through doubled).
            ["L", "R"]
                .iter()
                .any(|s| pose_fit.report["legs"][s]["applied"] == json!(true)),
            resolved.as_ref().map_or(&[][..], |r| r.outlines.as_slice()),
            config.guided_body,
            guided_limits.map(|l| (l.1, l.2, l.3)),
            &suit_contours,
            &params.head_contours,
            fallback_controls,
            match (landmarks.get("eye_l"), landmarks.get("eye_r")) {
                (Some(l), Some(r)) => Some((geom::v(*l) + geom::v(*r)) * 0.5),
                _ => None,
            },
        )?
    } else {
        json!({"applied": false, "reason": "disabled"})
    };
    let posed_body = prior.skin_mesh(prior.body, &pose_fit.worlds);
    geom::dump_stage("1-shape", posed_body.iter().copied());
    // Guided: the marked outlines size each station of the shaped body
    // (cloth sizes nothing; `shape::apply_outlines`).
    // The deformation goes into MakeHuman's bind (and rest) shape, not only
    // the posed body: the fine fit's shape reference is the rest shape, and
    // applied to the posed body alone it relaxed back (the flame's arms kept
    // a third of their marked size).
    let (posed_body, outline_report) = match (&resolved, config.guided_body && !config.registration) {
        (Some(r), true) => {
            let mut p = posed_body.clone();
            let cap = guided_limits.map(|l| l.0 * prior.height());
            let report = shape::apply_outlines(&prior, &no_digit, &r.outlines, &mut p, cap);
            let body = prior.body;
            let bind = prior.meshes[body].bind.clone();
            // Per vertex: the skinning's linear map (bind -> posed), by
            // finite differences, inverted to carry the posed change back.
            let columns: Vec<Vec<charloom_core::Vec3>> = (0..3)
                .map(|k| {
                    let shifted: Vec<charloom_core::Vec3> = bind
                        .iter()
                        .map(|q| {
                            let mut q = *q;
                            q[k] += 0.01;
                            q
                        })
                        .collect();
                    prior.skin_positions(body, &shifted, &pose_fit.worlds)
                })
                .collect();
            let new_bind: Vec<charloom_core::Vec3> = (0..bind.len())
                .map(|i| {
                    let d = geom::v(p[i]) - geom::v(posed_body[i]);
                    if d.norm() < 1e-9 {
                        return bind[i];
                    }
                    let m = nalgebra::Matrix3::from_columns(&[0, 1, 2].map(|k| {
                        (geom::v(columns[k][i]) - geom::v(posed_body[i])) / 0.01
                    }));
                    match m.try_inverse() {
                        Some(inv) => geom::a(geom::v(bind[i]) + inv * d),
                        None => bind[i],
                    }
                })
                .collect();
            let rest = prior.skin_positions(body, &new_bind, &prior.rest_world.clone());
            prior.meshes[body].bind = new_bind;
            prior.meshes[body].prim.positions = rest;
            (prior.skin_mesh(prior.body, &pose_fit.worlds), report)
        }
        _ => (posed_body, json!({"applied": false})),
    };
    if config.body_size && !config.guided_body {
        params.fill = bodysize::fill_targets(
            &scan,
            &seg,
            &size_axes,
            &gaps,
            &posed_body,
            &prior.body().prim.triangles,
            config.clearance_m,
            fit_height,
            shape_stylized,
        );
    }
    geom::dump_stage("2-outlines", posed_body.iter().copied());
    lap("shape_fit", &mut t);
    let fit_landmarks = if config.ear_outline {
        landmarks.clone()
    } else {
        landmarks
            .iter()
            .filter(|(n, _)| !guide::is_ear_landmark(n))
            .map(|(n, p)| (n.clone(), *p))
            .collect()
    };
    let mut body_fit = fit::fit_body(&prior, &scan, &seg, &posed_body, &params, &fit_landmarks)?;
    geom::dump_stage("fit-output",body_fit.positions.iter().copied());
     
    let pinned_head_field = body_fit.pinned_head_rig_reference.take().map(|reference| {
        assert_eq!(reference.len(), body_fit.positions.len());
        let fitted = std::mem::replace(&mut body_fit.positions, reference);
        fitted.iter().zip(&body_fit.positions).map(|(p,b)| geom::v(*p)-geom::v(*b)).collect::<Vec<_>>()
    });
    lap("body_fit", &mut t);
    let body_size_report = json!({
        "enabled": config.body_size,
        "gaps": bodysize::gap_report(&gaps),
        "pieces": seg.pieces,
        "label_profiles": gap_profiles,
        "targets_shape_fit": shape_fill.as_ref().map(|f| bodysize::fill_report(f, &seg)),
        "targets_fine_fit": bodysize::fill_report(&params.fill, &seg),
        "girths_cm": {
            "note": "convex-hull circumference in the scan pose; torso stations include the thighs, limbs are left/right means",
            "prior": girth_prior,
            "shaped": bodysize::girths(&size_axes, &bodysize::body_points(&size_axes, &posed_body), fit_height),
            "fitted": bodysize::girths(&size_axes, &bodysize::body_points(&size_axes, &body_fit.positions), fit_height),
            "scan_dressed": bodysize::girths(
                &size_axes,
                &size_samples.points.iter().map(|(k, c, _)| (*k, *c)).collect::<Vec<_>>(),
                fit_height,
            ),
        },
    });
    let mut eyes = fit::eye_fits(
        &prior,
        &body_fit.positions,
        &landmarks,
        stylized,
        config.eye_symmetry,
    );
    // Legs placed on the scan by the pose fit keep those joints (legs.rs).
    let held_joints = legs::held_joints(&prior, &pose_fit.worlds, &pose_fit.report["legs"]);
    let mut rig = refit_rig(
        &prior,
        &body_fit.positions,
        &eyes,
        &held_joints,
        &pose_fit.worlds,
        &posed_body,
    )?;
    // Eyeballs stay behind the face: the scan's surface is the outer skin.
    // Move an eye back by its protrusion. Behind the lids (outside the
    // marked opening) the eyeball stays 0.5 mm under the skin; inside the
    // opening the cornea may bulge out of the scan's flat painted eye by up
    // to CORNEA_M, as it does past real lid margins: held behind the
    // painted surface too, the eyeball sank 4 mm into the socket and the
    // socket lining in front of it showed as skin over the eye's corners.
    // A mirrored pair moves back together (the larger protrusion), so the
    // two eyeballs stay mirror images.
    const CORNEA_M: f64 = 0.0025;
    let mut eye_pushback = serde_json::Map::new();
    if let Some(mi) = prior.meshes.iter().position(|m| m.kind == PartKind::Eyes) {
        let paired = config.eye_symmetry && eyes.len() == 2;
        for _ in 0..2 {
            let mut outs: Vec<(String, f64)> = vec![];
            for joint in eyes.keys() {
                let Some(&j) = prior.by_name.get(joint) else {
                    continue;
                };
                let side = if joint == "eye.L" { "l" } else { "r" };
                let marked = MarkedOpening::new(&landmarks, side);
                let out = rig.meshes[mi]
                    .iter()
                    .zip(&prior.meshes[mi].influences)
                    .filter(|(_, row)| row.iter().any(|x| x.joint == j && x.weight > 0.5))
                    .filter_map(|(p, _)| {
                        let hit = scan.query.closest_point(*p, 0.01, None)?;
                        let out = (geom::v(*p) - geom::v(hit.point)).dot(&geom::v(hit.normal));
                        let inside = marked.as_ref().is_some_and(|m| {
                            let (s0, t0) = m.st(geom::v(*p));
                            m.opening.depth(s0, t0) > 0.0
                        });
                        Some(if inside && config.eye_openings {
                            out - CORNEA_M - 0.0005
                        } else {
                            out
                        })
                    })
                    .fold(f64::NEG_INFINITY, f64::max);
                outs.push((joint.clone(), out));
            }
            if paired {
                let worst = outs.iter().map(|o| o.1).fold(f64::NEG_INFINITY, f64::max);
                outs.iter_mut().for_each(|o| o.1 = worst);
            }
            let mut changed = false;
            for (joint, out) in outs {
                if out > -0.0005 {
                    let back = out + 0.0005;
                    if let Some(fit_eye) = eyes.get_mut(&joint) {
                        fit_eye.2 -= nalgebra::Vector3::z() * back;
                    }
                    *eye_pushback.entry(joint.clone()).or_insert(json!(0.0)) = json!(
                        eye_pushback
                            .get(&joint)
                            .and_then(Value::as_f64)
                            .unwrap_or(0.0)
                            + back
                    );
                    changed = true;
                }
            }
            if !changed {
                break;
            }
            rig = refit_rig(
                &prior,
                &body_fit.positions,
                &eyes,
                &held_joints,
                &pose_fit.worlds,
                &posed_body,
            )?;
        }
    }
    // Nothing of the body covers the eyeball inside the marked opening:
    // the socket lining (and any lid skin) in front of the placed eyeball
    // there goes behind it.
    let eye_clearing = if config.eye_openings {
        clear_eye_openings(&prior, &rig, &mut body_fit.positions, &landmarks)
    } else {
        EyeClearing {
            moved: 0,
            report: json!({"applied": false}),
            lining: vec![],
        }
    };
    geom::dump_stage("eye-clear-before-fairing", body_fit.positions.iter().copied());
    let eye_fairing = if config.eye_openings {
        match &eye_fairing_reference {
            Some(reference) => reference.apply(&prior, &mut body_fit.positions, &landmarks)?,
            None => json!({"applied": false, "reason": "disabled"}),
        }
    } else {
        json!({"applied": false, "reason": "eye openings disabled"})
    };
    if eye_clearing.moved > 0 || eye_fairing["applied"] == true {
        rig = refit_rig(
            &prior,
            &body_fit.positions,
            &eyes,
            &held_joints,
            &pose_fit.worlds,
            &posed_body,
        )?;
    }
    // The upper-lid turn that closes each fitted eye (reported, and in
    // character.glb's asset extras for blink poses).
    geom::dump_stage("post-eye-clear",rig.meshes[prior.body].iter().copied());
    let lid_closing = fit::eyes::lid_closing(
        &prior,
        &fit::eyes::openings(&prior),
        &body_fit.positions,
        &rig.worlds,
    );
    // The eyeballs: MakeHuman's clean ball dressed from the scan (iris
    // colour and size in the texture, gaze by the eye bones' rest rotation),
    // unless the scan's painted eye is projected onto it (below).
    let mut prior_images = source::read_images(&prior.loaded)?;
    let eye_dress = if config.eye_projection {
        Value::Null
    } else {
        eyeball::dress(&prior, &mut rig, &mut prior_images, &scan, &landmarks)?
    };
    // Lashes rebuilt from their lid-margin roots: shorter, lighter lower set.
    let lash_mesh = prior
        .meshes
        .iter()
        .position(|m| m.kind == PartKind::Eyelashes)
        .filter(|_| config.lashes.keep_makehuman);
    let lash_shape = lash_mesh.map(|mi| {
        lashes::shape(
            &prior,
            mi,
            &unmorphed_body,
            &body_fit.positions,
            &rig.meshes[mi],
            &config.lashes,
        )
    });
    if let (Some(mi), Some(s)) = (lash_mesh, &lash_shape) {
        rig.meshes[mi] = s.positions.clone();
    }
     
    let mouth_follow = mouth_follows_cavity(&prior, &mut rig.meshes, &body_fit.cavity);
    let lips_sealed = lips_seal(&prior, &mut rig.meshes, &body_fit.cavity, &scan);
    let mut mouth_pushback = mouth_behind_lips(&prior, &mut rig.meshes, &body_fit.cavity, &scan, &landmarks);
    if !mouth_pushback.is_null() {
        mouth_pushback["follow_cavity"] = mouth_follow;
        mouth_pushback["lips_sealed"] = lips_sealed;
    }
    geom::dump_stage("post-mouth",rig.meshes[prior.body].iter().copied());
    lap("rig_refit", &mut t);
    // Bone lengths left vs right: after the pose stage (chain scaling) and
    // after the rig refit (joints carried by the fitted body).
    {
        let skeleton = prior.skeleton();
        let names: Vec<&str> = skeleton.joints.iter().map(|j| j.name.as_str()).collect();
        let parents: Vec<Option<usize>> = skeleton.joints.iter().map(|j| j.parent).collect();
        let origins = |w: &[nalgebra::Matrix4<f64>]| -> Vec<geom::V> {
            w.iter()
                .map(|m| m.fixed_view::<3, 1>(0, 3).into_owned())
                .collect()
        };
        body_fit.report["asymmetry"]["bones"] = json!({
            "note": "relative left/right difference of each bone's length (distance to its parent joint)",
            "pose": symmetry::bone_lengths(&names, &parents, &origins(&pose_fit.worlds)),
            "rig": symmetry::bone_lengths(&names, &parents, &origins(&rig.worlds)),
        });
    }

    // QA overlay: guidance landmarks (red) vs fitted MakeHuman (green).
    if let (Some((g0, views)), Some(r)) = (&guidance, &resolved) {
        let g_front = &g0.front_view;
        let mut marks = vec![];
        for (l, j) in pose::LANDMARK_JOINTS {
            let p = rig.worlds[prior.joint(j)]
                .fixed_view::<3, 1>(0, 3)
                .into_owned();
            // Green (fitted) only where there is a red (marked) partner.
            if let Some(q) = r.landmarks.get(*l) {
                marks.push((geom::a(p), [40u8, 230, 60]));
                marks.push((*q, [255, 40, 40]));
            } else if let Some((o, _)) = r.joint_rays.get(*l) {
                marks.push((geom::a(p), [40u8, 230, 60]));
                marks.push((*o, [255, 40, 40]));
            }
        }
        for (name, set) in fit::face_landmark_sets(&prior) {
            let c = set.iter().fold(nalgebra::Vector3::zeros(), |s, &i| {
                s + geom::v(body_fit.positions[i])
            }) / set.len() as f64;
            if let Some(q) = r.landmarks.get(name) {
                marks.push((geom::a(c), [40, 230, 60]));
                marks.push((*q, [255, 40, 40]));
            }
        }
        // Named by direction from the guidance's front view, which is not
        // always body_y000.
        let front: u32 = g_front
            .strip_prefix("body_y")
            .and_then(|d| d.parse().ok())
            .unwrap_or(0);
        for (label, view) in [
            ("front", format!("body_y{front:03}")),
            ("side", format!("body_y{:03}", (front + 90) % 360)),
            ("head-front", format!("head_y{front:03}")),
            ("face", "face_y000".to_string()),
        ] {
            if let Some(img) = guide::overlay(views, &view, &scan, &marks) {
                img.save(out.join("0-guide").join(format!("qa-{label}.png")))?;
            }
        }
    }
    // Kept hands (hand_keep.rs): their bones on the scan's measured digits,
    // before the skeleton is compacted for every output.
    let guided_hands = guidance.as_ref().and_then(|(g, _)| g.subject.hands);
    let hand_route = config.hands.or(guided_hands).unwrap_or_default();
    let mut hand_bones = json!(null);
    if hand_route == hand_keep::HandRoute::KeepScan
        && let Some(r) = &resolved
    {
        hand_bones = json!([0u8, 1].map(|side| hand_keep::place_bones(&prior, &mut rig, side, digits[side as usize], &r.hand_points)));
    }
    // Compact skeleton and remapped weights shared by every output.
    let compact = compact_skeleton(&prior, &rig);
    let palette: Vec<usize> = prior
        .skin_joints
        .iter()
        .map(|j| compact.old_to_new[j])
        .collect();
    let compact_world: Vec<_> = compact
        .skeleton
        .world_matrices(&compact.skeleton.rest_pose())?;
    // Sanity: the compact rest pose reproduces the refitted world matrices.
    let rest_error = compact
        .new_to_old
        .iter()
        .enumerate()
        .map(|(n, &o)| (compact_world[n] - rig.worlds[o]).abs().max())
        .fold(0.0, f64::max);
    ensure!(rest_error < 1e-6, "rig_rest_mismatch: {rest_error}");

    let body = prior.body();
    let body_weld = Weld::new(&body.bind, 1e-7);
    if let Some(field) = &pinned_head_field {
        for ((p, fitted), delta) in rig.meshes[prior.body].iter_mut().zip(&mut body_fit.positions).zip(field) {
            *p = geom::a(geom::v(*p) + delta);
            *fitted = geom::a(geom::v(*fitted) + delta);
        }
        body_fit.report["hidden_head"]["anatomical"]["export_application"] = json!({
            "method": "same_solved_field_after_live_x110_rig_eye_mouth_processing",
            "post_rig_export_imported": false, "per_vertex_targets": false
        });
    }
    let body_positions = &rig.meshes[prior.body];
    geom::dump_stage("post-rig",body_positions.iter().copied());
    let body_influences = remap(&body.influences, &compact.old_to_new);

    // ---- Stage 3: garments, texture ----
    // Registration fits one continuous face. Its residual is not evidence
    // of an anatomical part: grafting it added rigid patches over brows,
    // cheeks and jaws while leaving most of the fitted face underneath.
    // Explicitly marked rigid parts (ears, horns, tusks) are already in
    // `garments`; only the legacy staged free-head route uses auto grafts.
    let mut graft_report = if config.free_head && !config.registration {
        let restored_nape = body_fit.report["hidden_head"]["applied"].as_bool().unwrap_or(false);
        let (grafts, report) = graft::uncovered(&prior, body_positions, &scan, &seg, &fit::eye_holes(&landmarks), fit_height, restored_nape);
        garments.extend(grafts);
        report
    } else if config.registration {
        json!({"applied": false, "reason": "registration_keeps_one_face_surface_use_explicit_rigid_parts", "pieces": 0, "body_triangles_hidden": 0})
    } else {
        Value::Null
    };
    // Garments (claws, gloves) never bind to fingers that will be cut.
    let mut garment_report = garment::bind(
        &mut garments,
        &garment::BodyRest {
            positions: body_positions,
            triangles: &creature::unmasked_triangles(&body.prim.triangles, &no_digit),
            influences: &body.influences,
            weld: &body_weld,
        },
        fit_height,
        config.clearance_m,
        // Rigid parts on the face follow the jaw (lower tusks, a beak's
        // lower half) or the head (horns, upper tusks), never a face-detail
        // bone that slides with an expression.
        |j| match (prior.by_name.get("jaw"), prior.by_name.get("head")) {
            (Some(&jaw), _) if prior.is_ancestor(jaw, j) => jaw,
            (_, Some(&head)) if prior.is_ancestor(head, j) => head,
            _ => j,
        },
         
        |j| {
            let joints = &prior.skeleton().joints;
            let mut at = Some(j);
            while let Some(k) = at {
                let name = joints[k].name.to_lowercase();
                if name.starts_with("foot") {
                    return Some(k);
                }
                if !name.starts_with("toe") {
                    return None;
                }
                at = joints[k].parent;
            }
            None
        },
        |j| prior.region[j] == Some(prior::Region::Hand),
    );
    if let Some(pieces) = garment_report["rigid_pieces"].as_array_mut() {
        for p in pieces {
            if let Some(j) = p["joint"].as_u64() {
                p["bone"] = json!(prior.skeleton().joints[j as usize].name);
            }
        }
    }
    // Optional route: keep the scan's own head (head_keep.rs). Falls back to
    // the fitted MakeHuman head when no clean neck seam is found.
    let guided_head = guidance.as_ref().and_then(|(g, _)| g.subject.head);
    let (head_route, head_source) = match (config.head, guided_head) {
        (Some(r), _) => (r, "config"),
        (None, Some(r)) => (r, "guidance"),
        (None, None) => (head_keep::HeadRoute::Fit, "default"),
    };
    let mut kept_head = None;
    let mut head_report = json!({"route": head_route, "source": head_source});
    if head_route == head_keep::HeadRoute::KeepScan {
        let input = head_keep::Input {
            prior: &prior,
            rig: &rig,
            scan: &scan,
            seg: &seg,
            height: fit_height,
        };
        match head_keep::keep_scan_head(&input, &mut garments) {
            Ok(k) => kept_head = Some(k),
            Err(e) => {
                eprintln!("keep_scan head route failed, using the fitted head: {e:#}");
                head_report = json!({"route": "fit", "requested": "keep_scan", "source": head_source, "fallback_reason": format!("{e:#}")});
            }
        }
    }
    // Optional route: keep the scan's own hands (hand_keep.rs), on the kept
    // head's body or MakeHuman's whole one. A hand that cannot be kept stays
    // MakeHuman's.
    let mut hand_report = json!({"route": hand_route, "source": if config.hands.is_some() { "config" } else if guided_hands.is_some() { "guidance" } else { "default" }, "bones_on_measured_digits": hand_bones});
    if hand_route == hand_keep::HandRoute::KeepScan {
        let mut kept = kept_head.take().unwrap_or_else(|| head_keep::KeptHead::from_body(&prior, &rig));
        let mut sides = vec![];
        for side in [0u8, 1] {
            let input = hand_keep::Input {
                prior: &prior,
                rig: &rig,
                scan: &scan,
                seg: &seg,
                height: fit_height,
                side,
                digits: digits[side as usize],
                measured: resolved.as_ref().map_or(0, |r| {
                    let tag = if side == 0 { "l" } else { "r" };
                    (1..=8)
                        .take_while(|k| ["base", "mid", "tip"].iter().all(|p| r.hand_points.contains_key(&format!("finger{k}_{tag}_{p}"))))
                        .count()
                }),
            };
            match hand_keep::keep_scan_hand(&mut kept, &input, &mut garments) {
                Ok(r) => sides.push(r),
                Err(e) => {
                    eprintln!("keep_scan hand route failed on side {side}, keeping MakeHuman's hand: {e:#}");
                    sides.push(json!({"route": "fit", "requested": "keep_scan", "fallback_reason": format!("{e:#}")}));
                }
            }
        }
        hand_report["hands"] = json!(sides);
        if kept.head || !kept.bridges.is_empty() {
            kept_head = Some(kept);
        }
    }
    for g in &mut garments {
        g.influences = remap(&g.influences, &compact.old_to_new);
    }
     
    let mut cloth_smoothing = serde_json::Map::new();
    for g in &mut garments {
        if g.rigid || g.hair || g.positions.is_empty() {
            continue;
        }
        let count = g.weld_id.iter().copied().max().map_or(0, |m| m + 1);
        let mut nbr: Vec<Vec<usize>> = vec![vec![]; count];
        for t in &g.triangles {
            let w = t.map(|i| g.weld_id[i as usize]);
            for k in 0..3 {
                if w[k] != w[(k + 1) % 3] {
                    nbr[w[k]].push(w[(k + 1) % 3]);
                    nbr[w[(k + 1) % 3]].push(w[k]);
                }
            }
        }
        for n in &mut nbr {
            n.sort_unstable();
            n.dedup();
        }
        let mut rep = vec![usize::MAX; count];
        for (i, &w) in g.weld_id.iter().enumerate() {
            if rep[w] == usize::MAX {
                rep[w] = i;
            }
        }
        let mut cur: Vec<Vec<Influence>> = rep.iter().map(|&i| if i == usize::MAX { vec![] } else { g.influences[i].clone() }).collect();
        for _ in 0..CLOTH_SMOOTHING {
            cur = {
                use rayon::prelude::*;
                (0..count).into_par_iter()
                .map(|w| {
                    if nbr[w].is_empty() || cur[w].is_empty() {
                        return cur[w].clone();
                    }
                    let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
                    for x in &cur[w] {
                        *acc.entry(x.joint).or_default() += 0.5 * x.weight;
                    }
                    let share = 0.5 / nbr[w].len() as f64;
                    for &j in &nbr[w] {
                        for x in &cur[j] {
                            *acc.entry(x.joint).or_default() += share * x.weight;
                        }
                    }
                    let mut out: Vec<Influence> = acc.into_iter().filter(|(_, v)| *v > 1e-6).map(|(joint, weight)| Influence { joint, weight }).collect();
                    out.sort_by(|x, y| y.weight.total_cmp(&x.weight).then(x.joint.cmp(&y.joint)));
                    out.truncate(4);
                    let sum: f64 = out.iter().map(|x| x.weight).sum();
                    for x in &mut out {
                        x.weight /= sum.max(1e-12);
                    }
                    out
                })
                .collect()
            };
        }
        for (i, &w) in g.weld_id.iter().enumerate() {
            if !cur[w].is_empty() {
                g.influences[i] = cur[w].clone();
            }
        }
        cloth_smoothing.insert(g.name.clone(), json!(CLOTH_SMOOTHING));
    }
     
    let mut footwear = vec![];
    for g in &mut garments {
        let n = g.name.to_lowercase();
        if !["shoe", "boot", "sandal", "footwear", "clog", "slipper"].iter().any(|k| n.contains(k)) {
            continue;
        }
        let mut changed = 0usize;
        for side in [".L", ".R"] {
            let Some(&foot) = prior.by_name.get(&format!("foot{side}")).and_then(|j| compact.old_to_new.get(j)) else {
                continue;
            };
            let ankle_y = compact_world[foot][(1, 3)];
            let band = fit_height * 0.04;
            for i in 0..g.positions.len() {
                let row = &g.influences[i];
                let on_side = row
                    .iter()
                    .max_by(|x, y| x.weight.total_cmp(&y.weight))
                    .is_some_and(|x| compact.skeleton.joints[x.joint].name.ends_with(side));
                if !on_side {
                    continue;
                }
                let t = ((g.positions[i][1] - ankle_y) / band).clamp(0.0, 1.0);
                let t = t * t * (3.0 - 2.0 * t);
                let mut acc: BTreeMap<usize, f64> = BTreeMap::new();
                *acc.entry(foot).or_default() += 1.0 - t;
                for x in row {
                    *acc.entry(x.joint).or_default() += x.weight * t;
                }
                let mut out: Vec<Influence> = acc
                    .into_iter()
                    .filter(|(_, w)| *w > 1e-6)
                    .map(|(joint, weight)| Influence { joint, weight })
                    .collect();
                out.sort_by(|x, y| y.weight.total_cmp(&x.weight).then(x.joint.cmp(&y.joint)));
                out.truncate(4);
                let sum: f64 = out.iter().map(|x| x.weight).sum();
                for x in &mut out {
                    x.weight /= sum.max(1e-12);
                }
                g.influences[i] = out;
                changed += 1;
            }
        }
        footwear.push(json!({"garment": g.name, "vertices_rigid_with_foot": changed}));
    }
    let hidden = if config.hide_covered_body {
        garment::covered_body_triangles(
            &garment::BodyRest {
                positions: body_positions,
                triangles: &body.prim.triangles,
                influences: &body.influences,
                weld: &body_weld,
            },
            &garments,
            fit_height * 0.06,
        )
    } else {
        vec![false; body.prim.triangles.len()]
    };
     
    let mut hidden = hidden;
    let mut grafted_over = vec![false; body.prim.triangles.len()];
    // Free head: the body's own skin on or just under a graft goes; the
    // graft is that skin (MakeHuman's ear, torn stretching towards the
    // scan's, showed in front of the grafted one).
    if config.free_head {
        let graft_tris: Vec<&garment::Garment> = garments.iter().filter(|g| g.name.starts_with("graft")).collect();
        if !graft_tris.is_empty() {
            let mut positions = vec![];
            let mut triangles = vec![];
            for g in graft_tris {
                let base = positions.len() as u32;
                positions.extend(g.positions.iter().copied());
                triangles.extend(g.triangles.iter().map(|t| t.map(|i| i + base)));
            }
            let q = scan::build_query(&positions, &triangles);
            let near: Vec<bool> = body_positions
                .iter()
                .map(|p| q.closest_point(*p, fit_height * graft::BODY_UNDER, None).is_some())
                .collect();
            let mut n = 0;
            for (k, t) in body.prim.triangles.iter().enumerate() {
                if t.iter().all(|&i| near[i as usize]) {
                    grafted_over[k] = true;
                    n += 1;
                }
            }
            graft_report["body_triangles_hidden"] = json!(n);
        }
    }
     
    let cell = fit_height * 0.015;
    let collar = fit_height * 0.015;
    let mut shoe_top: HashMap<(i64, i64), f64> = HashMap::new();
    for g in &garments {
        let n = g.name.to_lowercase();
        if !["shoe", "boot", "sandal", "footwear", "clog", "slipper"].iter().any(|k| n.contains(k)) {
            continue;
        }
        for p in &g.positions {
            let key = ((p[0] / cell).floor() as i64, (p[2] / cell).floor() as i64);
            let top = shoe_top.entry(key).or_insert(f64::NEG_INFINITY);
            *top = top.max(p[1]);
        }
    }
    let shoe_top_at = |p: geom::V| -> Option<f64> {
        let (x, z) = ((p.x / cell).floor() as i64, (p.z / cell).floor() as i64);
        let mut top: Option<f64> = None;
        for dx in -1..=1 {
            for dz in -1..=1 {
                if let Some(&y) = shoe_top.get(&(x + dx, z + dz)) {
                    top = Some(top.map_or(y, |t: f64| t.max(y)));
                }
            }
        }
        top
    };
    let mut foot_kept_above_shoe = serde_json::Map::new();
    if config.hide_covered_body {
        for side in [".l", ".r"] {
            let foot: Vec<usize> = body
                .prim
                .triangles
                .iter()
                .enumerate()
                .filter(|(_, t)| {
                    t.iter().all(|&i| {
                        let n = prior.skeleton().joints[prior.dominant_joint(prior.body, i as usize)].name.to_lowercase();
                        n.ends_with(side) && (n.starts_with("foot") || n.starts_with("toe"))
                    })
                })
                .map(|(k, _)| k)
                .collect();
             
            let name = if side == ".l" { "left" } else { "right" };
            let shod = body_fit.report["feet"]
                .as_array()
                .is_some_and(|f| f.iter().any(|x| x["side"] == name && x["shod"] == true && x["bare_fit"].is_null()));
            let covered = foot.iter().filter(|&&k| hidden[k]).count();
            if !foot.is_empty() && (shod || covered * 2 > foot.len()) {
                let ankle_y = prior
                    .by_name
                    .get(&format!("foot{}", side.to_uppercase()))
                    .and_then(|j| compact.old_to_new.get(j))
                    .map(|&j| compact_world[j][(1, 3)]);
                let mut kept = 0usize;
                for k in foot {
                    if hidden[k] {
                        continue;
                    }
                    let t = body.prim.triangles[k];
                    let c = t.iter().map(|&i| geom::v(body_positions[i as usize])).sum::<geom::V>() / 3.0;
                    let inside = match (shoe_top_at(c), ankle_y) {
                        (Some(top), _) => c.y < top - collar,
                        (None, Some(a)) => c.y < a,
                        (None, None) => true,
                    };
                    if inside {
                        hidden[k] = true;
                    } else {
                        kept += 1;
                    }
                }
                foot_kept_above_shoe.insert(side[1..].to_string(), json!(kept));
            }
        }
    }
    // The head is never removed, under hair or a helmet either: through any
    // gap an empty skull would show. Skin a graft replaces goes (the graft
    // is that skin).
    let hidden: Vec<bool> = hidden
        .iter()
        .zip(&body.prim.triangles)
        .zip(&grafted_over)
        .map(|((&h, t), &g)| {
            g || h && !t.iter().any(|&i| {
                prior.region[prior.dominant_joint(prior.body, i as usize)]
                    == Some(prior::Region::Head)
            })
        })
        .collect();
    let hidden = match &kept_head {
        Some(k) => k.hidden(&hidden),
        None => hidden,
    };
    lap("garments", &mut t);
    let body_normals = vertex_normals(body_positions, &body.prim.triangles, &body_weld);
     
    let iris_report = match if config.eye_projection {
        bake_eyes(&prior, &rig, &mut prior_images, &scan, &landmarks)?
    } else {
        eye_dress
    } {
        Value::Null => tint_iris(&prior, &mut prior_images, &scan, &landmarks)?,
        done => done,
    };
    let prior_materials = source::read_materials(&prior.loaded)?;
    let lash_texture = match (lash_mesh, &lash_shape) {
        (Some(mi), Some(s)) => lashes::fade_texture(
            &prior,
            mi,
            &s.lower,
            &prior_materials,
            &mut prior_images,
            &config.lashes,
        )?,
        _ => Value::Null,
    };
    let body_material = body.prim.material;
    let original_texture = body_material
        .and_then(|m| prior_materials[m].base_color)
        .map(|i| &prior_images[i].decoded);
    // Eye-socket exclusion: within ~1.6 eyeball radii of each fitted eye
    // joint, feathered out to ~2.4 radii.
    let mut exclude = vec![0.0f64; body_positions.len()];
    let mut eye_surround = vec![0.0f64; body_positions.len()];
    if config.exclude_eyes_from_bake {
        let sets = fit::face_landmark_sets(&prior);
        for (eye, side) in [("eye.L", "l"), ("eye.R", "r")] {
            let Some(&j) = prior.by_name.get(eye) else {
                continue;
            };
            // Marked eye opening: exclude an ellipse around the fitted
            // MakeHuman opening and one around the marked scan opening. Where
            // the lids stop short of the marks, the scan's painted white and
            // iris outside the fitted lids must not be baked onto the skin.
            let fitted = |k: &str| {
                let name = format!("eye_{side}_{k}");
                landmarks.contains_key(&name).then_some(())?;
                let set = &sets.iter().find(|s| s.0 == name.as_str())?.1;
                Some(
                    set.iter().fold(nalgebra::Vector3::zeros(), |acc, &i| {
                        acc + geom::v(body_positions[i])
                    }) / set.len() as f64,
                )
            };
            let marked = |k: &str| {
                landmarks
                    .get(&format!("eye_{side}_{k}"))
                    .map(|p| geom::v(*p))
            };
            let openings = [
                [
                    fitted("inner"),
                    fitted("outer"),
                    fitted("top"),
                    fitted("bottom"),
                ],
                [
                    marked("inner"),
                    marked("outer"),
                    marked("top"),
                    marked("bottom"),
                ],
            ];
            let mut excluded = false;
            for [inner, outer, top, bottom] in openings {
                let (Some(inner), Some(outer), Some(top), Some(bottom)) =
                    (inner, outer, top, bottom)
                else {
                    continue;
                };
                excluded = true;
                let c = (inner + outer + top + bottom) / 4.0;
                let u = (outer - inner).normalize();
                let vv = ((top - bottom) - u * (top - bottom).dot(&u)).normalize();
                let n = u.cross(&vv);
                 
                let (ha, hb) = ((outer - inner).norm() * 0.5, (top - bottom).norm() * 0.5);
                for ((e, s), p) in exclude
                    .iter_mut()
                    .zip(&mut eye_surround)
                    .zip(body_positions)
                {
                    let d = geom::v(*p) - c;
                    if d.dot(&n).abs() > ha {
                        continue;
                    }
                    let r = (d.dot(&u) / ha).powi(2) + (d.dot(&vv) / hb).powi(2);
                     
                    let t = ((1.0 - r) / 0.5).clamp(0.0, 1.0);
                    *e = e.max(t * t * (3.0 - 2.0 * t));
                    // The lid skin out to 1.3 opening radii (fading by 1.6)
                    // rejects sclera-like colour too: the scan's painted white
                    // reaches past the marks, and the lids warped onto the
                    // marks baked it as a white rim around the eye.
                    let t = ((2.6 - r) / 0.9).clamp(0.0, 1.0);
                    *s = s.max(t * t * (3.0 - 2.0 * t));
                }
            }
            if excluded {
                continue;
            }
            let center = rig.worlds[j].fixed_view::<3, 1>(0, 3).into_owned();
            let radius = prior
                .meshes
                .iter()
                .enumerate()
                .filter(|(_, m)| m.kind == PartKind::Eyes)
                .flat_map(|(mi, m)| {
                    m.influences
                        .iter()
                        .zip(&rig.meshes[mi])
                        .filter(move |(row, _)| row.iter().any(|w| w.joint == j && w.weight > 0.5))
                })
                .map(|(_, p)| (geom::v(*p) - center).norm())
                .fold(0.0, f64::max)
                .clamp(0.008, 0.02);
            for (e, p) in exclude.iter_mut().zip(body_positions) {
                let d = (geom::v(*p) - center).norm();
                let t = ((radius * 2.4 - d) / (radius * 0.8)).clamp(0.0, 1.0);
                *e = e.max(t * t * (3.0 - 2.0 * t));
            }
        }
    }
    // Marked ear outlines: the scan's ear is baked onto MakeHuman's pinna
    // inside the outline only; head skin there fills from MakeHuman's.
    for (e, x) in exclude.iter_mut().zip(&body_fit.ear_exclude) {
        *e = e.max(*x);
    }
    // Brow zones (around MakeHuman's eyebrow mesh, in its rest shape) and
    // the body's mirror partners: an occluded brow takes the other one.
    let (brow_zone, brow_mirror) = brow_zones(&prior);
    // ---- Stage 3: skin restoration ----
    let scan_weight = texture::skin_sample_weights(
        &scan,
        &seg.vertex_label,
        resolved.as_ref().map(|r| r.cleanup.as_slice()),
        fit_height * 0.009,
        fit_height * 0.0025,
    );
    let detail: Vec<f64> = (0..body_positions.len())
        .map(|i| {
            let head =
                prior.region[prior.dominant_joint(prior.body, i)] == Some(prior::Region::Head);
            if head && body_normals[i][2] > 0.2 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    // The scalp (MakeHuman's `head` bone, not the face's muscles): its
    // stubble is taken out of the reference texture (`texture::destubble`).
    let scalp: Vec<f64> = (0..body_positions.len())
        .map(|i| {
            let j = prior.dominant_joint(prior.body, i);
            if prior.skeleton().joints[j].name == "head" { 1.0 } else { 0.0 }
        })
        .collect();
    let baked = texture::bake(&texture::BakeInput {
        scan: &scan,
        labels: &seg.vertex_label,
        positions: body_positions,
        normals: &body_normals,
        uvs: &body.prim.uvs,
        triangles: &body.prim.triangles,
        original: original_texture,
        exclude: &exclude,
        eye_surround: &eye_surround,
        eye_lining: &eye_clearing.lining,
        brows: &brow_zone,
        mirror: &brow_mirror,
        marked_brow: resolved.as_ref().map_or(&[][..], |r| r.brow.as_slice()),
        sclera: iris_report["sclera_linear"].as_array().and_then(|a| {
            let c: Vec<f64> = a.iter().filter_map(Value::as_f64).collect();
            (c.len() == 3).then(|| [c[0], c[1], c[2]])
        }),
        scan_weight: &scan_weight,
        detail: &detail,
        cavity: &body_fit.cavity,
        hidden: &body_fit.shod_foot,
        scalp: &scalp,
        size: config.texture_size,
        reach: fit_height * 0.02,
    });
    let mut png = vec![];
    baked
        .image
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
    if let Some(k) = &mut kept_head {
        k.bake_bridge(&baked.image)?;
    }
    // The exported skin: MakeHuman's body, or (keep_scan) the cut body, the
    // scan head parts and the neck bridge. Influences in compact joints.
    let mut skin_body = match &kept_head {
        Some(k) => SkinMesh {
            positions: k.body_positions.clone(),
            normals: k.body_normals.clone(),
            uvs: k.body_uvs.clone(),
            triangles: k.body_triangles.clone(),
            influences: remap(&k.body_influences, &compact.old_to_new),
        },
        None => SkinMesh {
            positions: body_positions.clone(),
            normals: body_normals.clone(),
            uvs: body.prim.uvs.clone(),
            triangles: body.prim.triangles.clone(),
            influences: body_influences.clone(),
        },
    };
    // Creature hands: the missing digits are cut and capped here, after the
    // fit and the bake, so everything upstream kept MakeHuman's topology.
    let compact_cut_bones = creature::missing_bones(
        |n| {
            prior
                .by_name
                .get(n)
                .and_then(|j| compact.old_to_new.get(j))
                .copied()
        },
        digits,
    );
    let mut hidden = hidden;
    let digit_report = if compact_cut_bones.is_empty() {
        json!({"applied": false, "digits": digits})
    } else {
        let mut r = creature::cut_digits(&mut skin_body, &mut hidden, &compact_cut_bones);
        r["digits"] = json!(digits);
        // The fit's distortion without the cut fingers (they followed the
        // regulariser only and are gone from the output).
        let kept = creature::unmasked_triangles(&body.prim.triangles, &no_digit);
        r["kept_distortion_vs_prior_rest"] =
            geom::distortion(&body.prim.positions, body_positions, &kept);
        // Where the most stretched kept edge is (the stump or elsewhere).
        let worst = kept
            .iter()
            .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
            .filter_map(|(i, j)| {
                let (i, j) = (i as usize, j as usize);
                let rest =
                    (geom::v(body.prim.positions[i]) - geom::v(body.prim.positions[j])).norm();
                (rest > fit_height * 0.001).then(|| {
                    let now = (geom::v(body_positions[i]) - geom::v(body_positions[j])).norm();
                    (i, now / rest)
                })
            })
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((i, ratio)) = worst {
            r["worst_kept_edge"] = json!({"ratio": ratio, "joint": prior.skeleton().joints[prior.dominant_joint(prior.body, i)].name});
        }
        r["garment_vertices_reweighted"] = json!(
            garments
                .iter_mut()
                .map(|g| creature::redirect_weights(&mut g.influences, &compact_cut_bones))
                .sum::<usize>()
        );
        r
    };
    // Scan head parts, then the bridge.
    let head_influences: Vec<Vec<Vec<Influence>>> = kept_head
        .as_ref()
        .map(|k| {
            k.heads
                .iter()
                .map(|h| remap(&h.influences, &compact.old_to_new))
                .chain(k.bridges.iter().map(|b| remap(&b.influences, &compact.old_to_new)))
                .collect()
        })
        .unwrap_or_default();
    lap("texture", &mut t);

    // ---- Stage 4: body-size morphs carried onto the clothes ----
    let size = if config.size_morphs {
        let mut meshes: Vec<sizing::ClothMesh> = garments
            .iter()
            .map(|g| sizing::ClothMesh {
                name: &g.name,
                positions: &g.positions,
                normals: &g.normals,
                triangles: &g.triangles,
                hair: g.hair,
                rigid: g.rigid,
            })
            .collect();
        if let Some(k) = &kept_head {
            for h in &k.heads {
                meshes.push(sizing::ClothMesh {
                    name: &h.name,
                    positions: &h.positions,
                    normals: &h.normals,
                    triangles: &h.triangles,
                    hair: false,
                    rigid: false,
                });
            }
            for br in &k.bridges {
                meshes.push(sizing::ClothMesh {
                    name: &br.name,
                    positions: &br.positions,
                    normals: &br.normals,
                    triangles: &br.triangles,
                    hair: false,
                    rigid: false,
                });
            }
        }
        let hands = sizing::hand_triangles(&skin_body.triangles, &skin_body.influences, |j| {
            sizing::is_hand_name(&compact.skeleton.joints[j].name)
        });
        sizing::compute(
            &prior,
            body_positions,
            (&skin_body.positions, &skin_body.triangles, &hands),
            &meshes,
            fit_height,
            config.clearance_m,
        )
    } else {
        None
    };
    let size_targets = |name: &str| size.as_ref().and_then(|s| s.export.get(name));
    lap("size_morphs", &mut t);

    // ---- Underwear on the naked body (underwear.rs) ----
    let guided_sex = guidance
        .as_ref()
        .map(|(g, _)| g.subject.sex.to_lowercase())
        .filter(|s| s == "female" || s == "male");
    let (underwear, underwear_assets, mut underwear_report) = fit_underwear(
        config,
        prior_path,
        &prior,
        guided_sex,
        &underwear::BodyInput {
            positions: body_positions,
            triangles: &body.prim.triangles,
            influences: &body_influences,
            size_targets: size.as_ref().map_or(&[][..], |s| s.body.as_slice()),
            height: fit_height,
        },
    )?;
    lap("underwear", &mut t);

    // ---- Stage 4: motions ----
    let mut clips = vec![];
    let mut motion_reports = vec![];
    let rigid_wrists: Vec<String> = hand_report["hands"]
        .as_array()
        .map(|hands| {
            hands
                .iter()
                .filter(|h| h["rigid"] == true)
                .filter_map(|h| h["side"].as_str())
                .map(|s| format!("wrist.{}", s.to_uppercase()))
                .collect()
        })
        .unwrap_or_default();
    for m in &config.motions {
        let (clip, report) = check::load_and_retarget(
            &m.path,
            m.clip.as_deref(),
            &m.name,
            &compact.skeleton,
            prior.skeleton(),
            hand_route == hand_keep::HandRoute::KeepScan,
            &rigid_wrists,
        )?;
        motion_reports.push(report);
        clips.push(clip);
    }

    let face_clip = {
        let skeleton = &compact.skeleton;
        let rest = skeleton.rest_pose();
        let poses = face_test::poses(skeleton, &rest);
        // Hold each expression for half a second, half a second to the next.
        let mut times = vec![];
        let mut frames = vec![];
        for (k, (_, p)) in poses.iter().enumerate() {
            times.push(k as f64);
            frames.push(p.clone());
            times.push(k as f64 + 0.5);
            frames.push(p.clone());
        }
        times.push(poses.len() as f64);
        frames.push(rest.clone());
        let joints: Vec<usize> = (0..skeleton.joints.len())
            .filter(|&j| poses.iter().any(|(_, p)| p[j].rotation != rest[j].rotation))
            .collect();
        (!joints.is_empty())
            .then(|| charloom_engine::motion::clip_from_poses("face_test", &times, &frames, &joints))
            .transpose()?
    };
    let prior_textures = scan::json_array(&prior.loaded.document.json["textures"]);
    let prior_samplers = scan::json_array(&prior.loaded.document.json["samplers"]);
    let write_character = |path: &Path, clothed: bool| -> Result<()> {
        let mut b = export::Builder::new();
        b.set_skeleton(&compact.skeleton, &palette, &compact_world)?;
        b.doc.json["asset"]["extras"]["charloom_lid_closing_rad"] = lid_closing.clone();
        let mh_materials = b.import_materials(
            &prior_images,
            &prior_textures,
            &prior_samplers,
            &prior_materials,
        );
        // Eyeballs are solid: blending would let the background show through
        // the sclera. (Lashes and brows keep their alpha.)
        for mesh in prior.meshes.iter().filter(|m| m.kind == PartKind::Eyes) {
            if let Some(m) = mesh.prim.material {
                b.doc.json["materials"][mh_materials[m]]["alphaMode"] = json!("OPAQUE");
            }
        }
        let baked_image = b.add_image(&png, "image/png");
        let baked_texture = b.add_texture(baked_image);
        let mut body_json =
            body_material.map_or(json!({"name":"body"}), |m| prior_materials[m].json.clone());
        body_json["pbrMetallicRoughness"]["baseColorTexture"] = json!({"index": baked_texture});
        body_json["alphaMode"] = json!("OPAQUE");
        body_json["doubleSided"] = json!(false);
        let body_mat = b.add_material(body_json);
        // The guided body is subdivided once for output (`subdivide.rs`):
        // grown far past MakeHuman's sizes, its polygons showed. Not with
        // size morph targets (they would need subdividing too).
        let fine = (config.guided_body && size.is_none()).then(|| {
            subdivide::loop_subdivide(
                &skin_body.positions,
                &skin_body.uvs,
                &skin_body.influences,
                &skin_body.triangles,
            )
        });
        let triangles: Vec<[u32; 3]> = match &fine {
            Some(f) => f
                .triangles
                .iter()
                .zip(&f.parent)
                .filter(|(_, p)| !clothed || !hidden[**p])
                .map(|(t, _)| *t)
                .collect(),
            None => skin_body
                .triangles
                .iter()
                .zip(&hidden)
                .filter(|(_, h)| !clothed || !**h)
                .map(|(t, _)| *t)
                .collect(),
        };
        let node = b.add_mesh(&export::OutMesh {
            name: "body",
            positions: fine.as_ref().map_or(&skin_body.positions, |f| &f.positions),
            normals: fine.as_ref().map_or(&skin_body.normals, |f| &f.normals),
            uvs: Some(fine.as_ref().map_or(&skin_body.uvs, |f| &f.uvs)),
            colors: None,
            triangles: &triangles,
            influences: Some(fine.as_ref().map_or(&skin_body.influences, |f| &f.influences)),
            material: Some(body_mat),
        })?;
        let sized = |b: &mut export::Builder, node: usize, name: &str, role: Value| -> Result<()> {
            if size.is_some() {
                if let Some(t) = size_targets(name) {
                    b.add_morph_targets(node, t)?;
                }
                b.set_mesh_extra(node, sizing::ROLE_KEY, role);
            }
            Ok(())
        };
        sized(&mut b, node, "body", json!({"role": "body"}))?;
        // keep_scan imports the scan's materials here, for its head; the
        // fit route imports them after MakeHuman's parts, as it always did
        // (byte-identical output).
        let mut scan_materials = kept_head.is_some().then(|| {
            b.import_materials(
                &scan.images,
                &scan.textures,
                &scan.samplers,
                &scan.materials,
            )
        });
        if let (Some(k), Some(scan_materials)) = (&kept_head, &scan_materials) {
            for (h, infl) in k.heads.iter().zip(&head_influences) {
                let node = b.add_mesh(&export::OutMesh {
                    name: &h.name,
                    positions: &h.positions,
                    normals: &h.normals,
                    uvs: Some(&h.uvs),
                    colors: None,
                    triangles: &h.triangles,
                    influences: Some(infl),
                    material: h.material.map(|m| scan_materials[m]),
                })?;
                sized(&mut b, node, &h.name, json!({"role": "cloth"}))?;
            }
            for (bi, br) in k.bridges.iter().enumerate() {
                let image = b.add_image(&br.png, "image/png");
                let texture = b.add_texture(image);
                let mut json = body_material.map_or(json!({}), |m| prior_materials[m].json.clone());
                json["name"] = json!(br.name);
                json["pbrMetallicRoughness"]["baseColorTexture"] = json!({"index": texture});
                json["alphaMode"] = json!("OPAQUE");
                json["doubleSided"] = json!(false);
                let material = b.add_material(json);
                let node = b.add_mesh(&export::OutMesh {
                    name: &br.name,
                    positions: &br.positions,
                    normals: &br.normals,
                    uvs: Some(&br.uvs),
                    colors: None,
                    triangles: &br.triangles,
                    influences: head_influences.get(k.heads.len() + bi).map(Vec::as_slice),
                    material: Some(material),
                })?;
                sized(&mut b, node, &br.name, json!({"role": "cloth"}))?;
            }
        }
        for (mi, mesh) in prior.meshes.iter().enumerate() {
            if mi == prior.body
                || mesh.kind == PartKind::Other
                || (mesh.kind == PartKind::Eyebrows && !config.prior_eyebrows)
                || (mesh.kind == PartKind::Eyelashes && !config.lashes.keep_makehuman)
                || (kept_head.as_ref().is_some_and(|k| k.head) && head_keep::removes_part(mesh.kind))
            {
                continue;
            }
            let positions = &rig.meshes[mi];
            let w = Weld::new(positions, 1e-7);
            let normals = vertex_normals(positions, &mesh.prim.triangles, &w);
            let name = format!("{:?}", mesh.kind).to_lowercase();
            b.add_mesh(&export::OutMesh {
                name: &name,
                positions,
                normals: &normals,
                uvs: Some(&mesh.prim.uvs),
                colors: None,
                triangles: &mesh.prim.triangles,
                influences: Some(&remap(&mesh.influences, &compact.old_to_new)),
                material: mesh.prim.material.map(|m| mh_materials[m]),
            })?;
        }
        if !clothed {
            for piece in &underwear {
                let asset = &underwear_assets[piece.asset];
                let mut texture = |img: &Option<(Vec<u8>, String)>| {
                    img.as_ref().map(|(bytes, mime)| {
                        let i = b.add_image(bytes, mime);
                        b.add_texture(i)
                    })
                };
                let (diffuse, normal) = (texture(&asset.diffuse), texture(&asset.normal));
                let mut mat = json!({
                    "name": piece.name,
                    "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 0.85},
                    "alphaMode": "OPAQUE",
                    "doubleSided": true,
                });
                if let Some(t) = diffuse {
                    mat["pbrMetallicRoughness"]["baseColorTexture"] = json!({"index": t});
                }
                if let Some(t) = normal {
                    mat["normalTexture"] = json!({"index": t});
                }
                let material = b.add_material(mat);
                let node = b.add_mesh(&export::OutMesh {
                    name: &piece.name,
                    positions: &piece.positions,
                    normals: &piece.normals,
                    uvs: Some(&piece.uvs),
                    colors: None,
                    triangles: &piece.triangles,
                    influences: Some(&piece.influences),
                    material: Some(material),
                })?;
                if size.is_some() {
                    b.add_morph_targets(node, &piece.targets)?;
                    b.set_mesh_extra(node, sizing::ROLE_KEY, json!({"role": "underwear"}));
                }
                b.set_mesh_extra(
                    node,
                    "charloom_underwear",
                    json!({"asset": asset.id, "author": asset.mhclo.author, "license": asset.mhclo.license, "source": asset.source_page()}),
                );
            }
        }
        if clothed && scan_materials.is_none() {
            scan_materials = Some(b.import_materials(
                &scan.images,
                &scan.textures,
                &scan.samplers,
                &scan.materials,
            ));
        }
        let hair_materials = match &scan_materials {
            Some(m) if !hair_encoded.is_empty() => {
                strands::add_hair_materials(&mut b, &scan, &hair_encoded, m)
            }
            _ => BTreeMap::new(),
        };
        if let Some(scan_materials) = &scan_materials {
            // keep_scan: the scan has no scalp under its hair, so the naked
            // body keeps the hair pieces too.
            for g in garments
                .iter()
                .filter(|g| clothed || (g.hair && kept_head.as_ref().is_some_and(|k| k.head)))
            {
                let node = b.add_mesh(&export::OutMesh {
                    name: &g.name,
                    positions: &g.positions,
                    normals: &g.normals,
                    uvs: Some(&g.uvs),
                    colors: None,
                    triangles: &g.triangles,
                    influences: Some(&g.influences),
                    material: g.material.map(|m| match hair_materials.get(&m) {
                        Some(&h) if g.hair => h,
                        _ => scan_materials[m],
                    }),
                })?;
                sized(
                    &mut b,
                    node,
                    &g.name,
                    json!({"role": "cloth", "hair": g.hair, "rigid": g.rigid}),
                )?;
                // What the piece is, for viewers (the clothes-only view). A
                // rigid piece (a graft, kept ears, a fang) or one named as a
                // body part (tusks) is the character's
                // own body: untagged, it shows with the bare body and not
                // among the clothes.
                let own = g.name.to_lowercase().split(|c: char| !c.is_ascii_alphabetic())
                    .any(|word| ["tusk", "fang", "ear", "horn", "tooth", "teeth", "claw", "tail"].contains(&word.trim_end_matches('s')));
                if g.hair || !(g.rigid || own) {
                    b.set_mesh_extra(node, "charloom_part", json!(if g.hair { "hair" } else { "garment" }));
                }
            }
        }
        for clip in &clips {
            b.add_animation(clip)?;
        }
        // Face test expressions, one per second (`face_test.rs`), so the
        // viewer can play them and show how the face is mapped onto the rig.
        if let Some(face) = face_clip.as_ref() {
            b.add_animation(face)?;
        }
        b.save(path)
    };
    let stage2 = out.join("2-body-rig");
    std::fs::create_dir_all(&stage2)?;
    let stage3 = out.join("3-skin");
    std::fs::create_dir_all(&stage3)?;
    write_character(&stage3.join("body.glb"), false)?;
    if let Some(text) = underwear::attribution(&underwear_assets) {
        std::fs::write(stage3.join("ATTRIBUTION.md"), text)?;
    }
    {
        let mut b = export::Builder::new();
        let w = Weld::new(body_positions, 1e-7);
        b.add_mesh(&export::OutMesh {
            name: "fit-diagnostic",
            positions: body_positions,
            normals: &vertex_normals(body_positions, &body.prim.triangles, &w),
            uvs: None,
            colors: Some(&body_fit.diagnostic),
            triangles: &body.prim.triangles,
            influences: None,
            material: None,
        })?;
        b.save(&stage2.join("fit-diagnostic.glb"))?;
    }
    baked.image.save(stage3.join("body-basecolor.png"))?;
    baked.provenance.save(stage3.join("body-provenance.png"))?;
    let stage4 = out.join("4-garments");
    std::fs::create_dir_all(&stage4)?;
    write_character(&stage4.join("character.glb"), true)?;
    lap("export", &mut t);

    // ---- Stage 4 checks on the exported data ----
    // Poke-through is measured against MakeHuman's whole fitted body on both
    // routes (comparable numbers; with keep_scan the scan head and its hair
    // both follow `head`, and the scan head is an open shell under the hair,
    // so "inside" is undefined there). keep_scan also reports poke-through
    // against the exported body + neck bridge, and seam opening over the
    // whole exported skin.
    let mut check_skin = SkinMesh {
        positions: skin_body.positions.clone(),
        normals: vec![],
        uvs: vec![],
        triangles: skin_body.triangles.clone(),
        influences: skin_body.influences.clone(),
    };
    let mut full_skin = SkinMesh {
        positions: skin_body.positions.clone(),
        normals: vec![],
        uvs: vec![],
        triangles: vec![],
        influences: skin_body.influences.clone(),
    };
    if let Some(k) = &kept_head {
        let parts = k
            .heads
            .iter()
            .map(|h| (&h.positions, &h.triangles, false))
            .chain(k.bridges.iter().map(|b| (&b.positions, &b.triangles, true)));
        for ((p, tr, bridge), infl) in parts.zip(&head_influences) {
            let targets = if bridge {
                vec![&mut check_skin, &mut full_skin]
            } else {
                vec![&mut full_skin]
            };
            for skin in targets {
                let o = skin.positions.len() as u32;
                skin.positions.extend_from_slice(p);
                skin.triangles.extend(tr.iter().map(|x| x.map(|i| i + o)));
                skin.influences.extend(infl.iter().cloned());
            }
        }
    }
    let body_part = check::SkinnedPart {
        name: "body",
        positions: body_positions,
        triangles: &body.prim.triangles,
        influences: &body_influences,
        seam: None,
    };
    let exported_part = check::SkinnedPart {
        name: "body_and_bridge",
        positions: &check_skin.positions,
        triangles: &check_skin.triangles,
        influences: &check_skin.influences,
        seam: None,
    };
    let garment_parts: Vec<check::SkinnedPart> = garments
        .iter()
        .map(|g| check::SkinnedPart {
            name: &g.name,
            positions: &g.positions,
            triangles: &g.triangles,
            influences: &g.influences,
            seam: Some(&g.weld_id),
        })
        .collect();
    let mut checks = vec![];
    let mut seam_gaps = vec![];
    let mut exported_checks = vec![];
    let rig_view = check::RigView {
        skeleton: &compact.skeleton,
        palette: &palette,
        rest_world: &compact_world,
    };
    for clip in &clips {
        checks.push(check::check_clip(
            &rig_view,
            clip,
            &body_part,
            &garment_parts,
            config.check_frames,
            fit_height,
            Some(&hidden),
        )?);
        if kept_head.is_some() {
            seam_gaps.push(head_keep::seam_gap(
                &rig_view,
                clip,
                config.check_frames,
                &full_skin.positions,
                &full_skin.influences,
            )?);
            let c = check::check_clip(
                &rig_view,
                clip,
                &exported_part,
                &garment_parts,
                config.check_frames,
                fit_height,
                None,
            )?;
            exported_checks.push(json!({"clip": c["clip"], "inside_fraction_over_frames": c["inside_fraction_over_frames"], "worst_frame_garments": c["worst_frame"]["garments"].as_array().map(|g| g.iter().map(|x| json!({"garment": x["garment"], "inside_fraction": x["inside_fraction"]})).collect::<Vec<_>>())}));
        }
    }
    // Underwear against the naked body over the clips (2 mm inside counts).
    if !underwear.is_empty() {
        let parts: Vec<check::SkinnedPart> = underwear
            .iter()
            .map(|p| check::SkinnedPart {
                name: &p.name,
                positions: &p.positions,
                triangles: &p.triangles,
                influences: &p.influences,
                seam: None,
            })
            .collect();
        let mut rows = vec![];
        for clip in &clips {
            let c = check::check_clip(
                &rig_view,
                clip,
                &body_part,
                &parts,
                config.check_frames,
                fit_height,
                None,
            )?;
            rows.push(json!({"clip": c["clip"], "rest": c["rest"]["garments"], "inside_fraction_over_frames": c["inside_fraction_over_frames"], "worst_frame": c["worst_frame"]["garments"]}));
        }
        underwear_report["checks"] = json!(rows);
    }
    if let Some(k) = kept_head.as_ref().filter(|k| k.head) {
        head_report = k.report.clone();
        head_report["source"] = json!(head_source);
        head_report["seam_gap_over_clips"] = json!(seam_gaps);
        head_report["poke_through_vs_exported_skin"] = json!(exported_checks);
    } else if kept_head.is_some() {
        hand_report["seam_gap_over_clips"] = json!(seam_gaps);
        hand_report["poke_through_vs_exported_skin"] = json!(exported_checks);
    }
    // Round trip: the written character reloads and its skin reproduces the fit.
    let reloaded = charloom_io::load_gltf(stage4.join("character.glb"))?;
    let reposed = reloaded
        .asset
        .skinned_positions(&reloaded.asset.skeleton.rest_pose())?;
    // Every skin mesh (body first, then the scan head parts and the bridge).
    let mut expected: Vec<&[charloom_core::Vec3]> = vec![&skin_body.positions];
    if let Some(k) = &kept_head {
        expected.extend(k.heads.iter().map(|h| h.positions.as_slice()));
        expected.extend(k.bridges.iter().map(|b| b.positions.as_slice()));
    }
    let roundtrip_error = reposed
        .iter()
        .zip(expected)
        .flat_map(|(r, e)| r.iter().zip(e))
        .map(|(a, b)| (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max))
        .fold(0.0, f64::max);
    lap("checks", &mut t);
    let size_report = match &size {
        Some(s) => {
            let checks = sizing::check(
                &sizing::CheckInput {
                    rig: &rig_view,
                    clip: clips.first(),
                    body_positions,
                    body_triangles: &body.prim.triangles,
                    body_influences: &body_influences,
                    garments: garments
                        .iter()
                        .map(|g| {
                            (
                                g.name.as_str(),
                                g.positions.as_slice(),
                                g.triangles.as_slice(),
                                g.influences.as_slice(),
                            )
                        })
                        .collect(),
                    frames: config.check_frames.min(8),
                    height: fit_height,
                },
                s,
                &(0..garments.len()).collect::<Vec<_>>(),
            )?;
            let mut r = s.report.clone();
            r["names"] = json!(s.names);
            r["clearance_m"] = json!(config.clearance_m);
            r["checks"] = checks;
            write_json(&stage4.join("size-morphs.json"), &r)?;
            r
        }
        None => {
            json!({"applied": false, "reason": if config.size_morphs { "prior has no shape morphs" } else { "disabled" }})
        }
    };
    lap("size_checks", &mut t);
    let stage5 = out.join("5-animation");
    std::fs::create_dir_all(&stage5)?;

    garment_report["grafts"] = graft_report;
    let report = json!({
        "operation": "makehuman_transfer",
        "source": source,
        "source_hash": scan.source_hash,
        "prior": prior_path,
        "prior_hash": prior.loaded.asset.source_hash,
        "working_frame": {"height_m": fit_height, "facing": facing_report, "source_to_working": scan.transform.as_slice()},
        "stage1_pose": {"coarse": coarse.report, "refined_without_hair": pose_fit.report},
        "stage1_fingers": finger_report,
        "stage1_segment": seg.report,
        "stage1_hair_recolour": recolour_report,
        "stage0_guidance": resolved.as_ref().map(|r| r.report.clone()),
        "stage2_shape": shape_report,
        "stage2_outlines": outline_report,
        "stage2_body_size": body_size_report,
        "stage2_body_fit": body_fit.report,
        "stage2_rig": rig.report,
        "stage0_eye_symmetry": eye_symmetry_report,
        "stage2_eye_pushback_m": eye_pushback,
        "stage2_eye_opening_clearing": eye_clearing.report,
        "stage2_lid_closing_rad": lid_closing,
        "stage2_lashes": json!({
            "shape": lash_shape.as_ref().map(|s| s.report.clone()),
            "texture": lash_texture,
        }),
        "stage2_head": head_report,
        "stage2_hands": hand_report,
        "stage4_garment_weights": {"footwear_rigid_with_foot": footwear, "cloth_smoothing_passes": cloth_smoothing, "shod_foot_kept_above_shoe": foot_kept_above_shoe},
        "stage3_skin": baked.report,
        "stage3_iris": iris_report,
        "stage3_underwear": underwear_report,
        "stage4_garments": garment_report,
        "stage4_hidden_body_triangles": hidden.iter().filter(|h| **h).count(),
        "stage4_digits": digit_report,
        "stage4_size_morphs": size_report,
        "stage5_motions": motion_reports,
        "stage5_checks": checks,
        "export_roundtrip_max_error_m": roundtrip_error,
        "timings_s": timings,
        "total_s": started.elapsed().as_secs_f64(),
        "outputs": {
            "segmentation": stage1.join("segmentation.glb"),
            "pieces": stage1.join("pieces"),
            "body": stage3.join("body.glb"),
            "character": stage4.join("character.glb"),
        },
    });
    let mut report = report;
    report["stage2_mouth_pushback"] = mouth_pushback;
    report["stage2_eye_surface_fairing"] = eye_fairing;
    write_json(&stage5.join("checks.json"), &report["stage5_checks"])?;
    write_json(&out.join("report.json"), &report)?;
    write_json(&out.join("config.json"), config)?;
    Ok(report)
}

 fn mouth_follows_cavity(prior: &Prior, meshes: &mut [Vec<charloom_core::Vec3>], cavity: &[bool]) -> Value {
    use geom::{a, v};
    let parts: Vec<usize> = prior.meshes.iter().enumerate()
        .filter(|(_, m)| matches!(m.kind, PartKind::Teeth | PartKind::Tongue)).map(|(i, _)| i).collect();
    let count: usize = parts.iter().map(|&p| prior.meshes[p].prim.positions.len()).sum();
    if count == 0 {
        return json!(null);
    }
    let centre = parts.iter().flat_map(|&p| prior.meshes[p].prim.positions.iter())
        .fold(nalgebra::Vector3::zeros(), |s, q| s + v(*q)) / count as f64;
    let rest = &prior.body().prim.positions;
    let radius = 0.045 * prior.height() / 1.75;
    let ids: Vec<usize> = (0..rest.len()).filter(|&i| cavity[i] && (v(rest[i]) - centre).norm() < radius).collect();
    if ids.len() < 20 {
        return json!({"applied": false, "lining_vertices": ids.len()});
    }
    let fitted = &meshes[prior.body];
    let mean = |points: &[charloom_core::Vec3]| ids.iter().fold(nalgebra::Vector3::zeros(), |s, &i| s + v(points[i])) / ids.len() as f64;
    let (ma, mb) = (mean(rest), mean(fitted));
    let h = ids.iter().fold(nalgebra::Matrix3::zeros(), |s, &i| s + (v(fitted[i]) - mb) * (v(rest[i]) - ma).transpose());
    let r = geom::proper_rotation(h);
    let scale = ids.iter().map(|&i| (v(fitted[i]) - mb).dot(&(r * (v(rest[i]) - ma)))).sum::<f64>()
        / ids.iter().map(|&i| (v(rest[i]) - ma).norm_squared()).sum::<f64>();
    let mut moved = 0.0f64;
    for &p in &parts {
        let source = prior.meshes[p].prim.positions.clone();
        for (q, s0) in meshes[p].iter_mut().zip(&source) {
            let placed = r * (v(*s0) - ma) * scale + mb;
            moved = moved.max((placed - v(*q)).norm());
            *q = a(placed);
        }
    }
    json!({"applied": true, "lining_vertices": ids.len(), "scale": scale, "largest_move_m": moved})
}

 fn lips_seal(
    prior: &Prior,
    meshes: &mut [Vec<charloom_core::Vec3>],
    cavity: &[bool],
    scan: &scan::Scan,
) -> Value {
    use geom::{a, v};
    const OVERLAP_M: f64 = 0.0001;
    const RINGS: usize = 4;
    let scale = prior.height() / 1.75;
    let names = &prior.skeleton().joints;
    // Which lip a body vertex belongs to, by the face joint that moves it.
    let lip = |i: usize| -> i8 {
        let name = names[prior.dominant_joint(prior.body, i)].name.as_str();
        match name.trim_end_matches(".L").trim_end_matches(".R") {
            "oris03" | "oris04" | "oris05" => 1,
            "oris01" | "oris02" | "oris07" => -1,
            // oris06 is the upper lip's middle joint, oris06.L/.R the lower lip's.
            "oris06" => if name == "oris06" { 1 } else { -1 },
            _ => 0,
        }
    };
    let triangles = &prior.meshes[prior.body].prim.triangles;
    let n = meshes[prior.body].len();
    let is_cavity = |i: usize| cavity.get(i).copied().unwrap_or(false);
    // The lips' inner edge: skin vertices that touch the mouth's lining.
    let mut rim = vec![false; n];
    let mut next: Vec<Vec<u32>> = vec![vec![]; n];
    for t in triangles {
        let lined = t.iter().any(|&i| is_cavity(i as usize));
        for k in 0..3 {
            let (i, j) = (t[k] as usize, t[(k + 1) % 3]);
            next[i].push(j);
            next[j as usize].push(i as u32);
            if lined && !is_cavity(i) && lip(i) != 0 {
                rim[i] = true;
            }
        }
    }
    let rest = &prior.body().prim.positions;
    let (upper, lower): (Vec<usize>, Vec<usize>) = (0..n).filter(|&i| rim[i]).partition(|&i| lip(i) > 0);
    if upper.is_empty() || lower.is_empty() {
        return json!({"applied": false, "upper": upper.len(), "lower": lower.len()});
    }
    let nearest = |i: usize, others: &[usize]| -> usize {
        *others.iter().min_by(|&&x, &&y| (v(rest[x]) - v(rest[i])).norm().total_cmp(&(v(rest[y]) - v(rest[i])).norm())).unwrap()
    };
    let pairs: Vec<(usize, usize)> = upper.iter().map(|&u| (u, nearest(u, &lower)))
        .chain(lower.iter().map(|&l| (nearest(l, &upper), l)))
        .collect();
    let body = &meshes[prior.body];
    let front = [0.0, 0.0, 1.0];
    let tolerance = LIP_DEPTH_TOLERANCE_M * scale;
    // Vertical move per vertex, the largest any of its pairs asks for.
    let mut shift = vec![0.0f64; n];
    let (mut sealed, mut largest, mut open) = (0usize, 0.0f64, 0usize);
    for &(u, l) in &pairs {
        let gap = body[u][1] - body[l][1];
        if gap <= -2.0 * OVERLAP_M {
            continue;
        }
        let limit = if gap <= SLIT_M * scale {
            SLIT_M * scale
        } else {
            // The scan's lips are closed here: it has surface at the fitted
            // lips' depth straight in front of the gap, not an opening.
            let middle = (v(body[u]) + v(body[l])) * 0.5;
            let from = middle - geom::V::z() * 0.01 * scale;
            let closed = scan.query.first_ray_distance(a(from), front, 0.03 * scale)
                .is_some_and(|d| (from.z + d - middle.z).abs() <= tolerance);
            if closed { 2.0 * SLIT_M * scale } else { 0.0 }
        };
        if gap > limit {
            open += 1;
            continue;
        }
        let half = gap * 0.5 + OVERLAP_M;
        shift[u] = shift[u].min(-half);
        shift[l] = shift[l].max(half);
        sealed += 1;
        largest = largest.max(gap);
    }
    if sealed == 0 {
        return json!({"applied": false, "pairs": pairs.len(), "open_pairs": open});
    }
    // The surrounding skin and lining follow their nearest edge vertex.
    let mut ring = vec![usize::MAX; n];
    let mut frontier: Vec<usize> = (0..n).filter(|&i| shift[i] != 0.0).collect();
    frontier.iter().for_each(|&i| ring[i] = 0);
    let mut moved = frontier.len();
    for r in 1..=RINGS {
        let mut grown = vec![];
        for &i in &frontier {
            for &j in &next[i] {
                let j = j as usize;
                // The other lip's edge is not dragged along.
                if ring[j] != usize::MAX || rim[j] {
                    continue;
                }
                ring[j] = r;
                let fade = 1.0 - r as f64 / (RINGS + 1) as f64;
                shift[j] = shift[i] / (1.0 - (r - 1) as f64 / (RINGS + 1) as f64) * fade;
                grown.push(j);
            }
        }
        moved += grown.len();
        frontier = grown;
    }
    for (p, s) in meshes[prior.body].iter_mut().zip(&shift) {
        p[1] += s;
    }
    json!({"applied": true, "pairs": pairs.len(), "sealed_pairs": sealed, "open_pairs": open,
        "largest_gap_m": largest, "moved_vertices": moved})
}

 fn mouth_behind_lips(
    prior: &Prior,
    meshes: &mut [Vec<charloom_core::Vec3>],
    cavity: &[bool],
    scan: &scan::Scan,
    landmarks: &std::collections::BTreeMap<String, charloom_core::Vec3>,
) -> Value {
    const MARGIN_M: f64 = 0.002;
    let parts: Vec<usize> = prior
        .meshes
        .iter()
        .enumerate()
        .filter(|(_, m)| matches!(m.kind, PartKind::Teeth | PartKind::Tongue))
        .map(|(i, _)| i)
        .collect();
    if parts.is_empty() {
        return json!(null);
    }
    let reach = prior.height() * 0.06;
    // The outer skin only: from a tooth in its place the mouth's cavity
    // lining behind it is entered from the cavity (outside the body), which
    // read as a protrusion and sent every tooth 7 cm back.
    let outer: Vec<[u32; 3]> = prior.meshes[prior.body]
        .prim
        .triangles
        .iter()
        .filter(|t| {
            t.iter()
                .all(|&i| !cavity.get(i as usize).copied().unwrap_or(false))
        })
        .copied()
        .collect();
    let body = scan::build_query(&meshes[prior.body], &outer);
    // MakeHuman faces +Z.
    let back = geom::v([0.0, 0.0, -1.0]);
    let depth_of = |p: charloom_core::Vec3| -> Option<f64> {
        let first = body.first_ray_distance(p, geom::a(back), reach)?;
        let exit = body.outward_ray_distance(p, geom::a(back), reach);
        // Entered from outside first: the point is in front of the skin.
        exit.is_none_or(|e| first < e - 1e-6).then_some(first)
    };
     
    let lip_thickness = LIP_THICKNESS_M * prior.height() / 1.75;
    let mut worst = 0.0f64;
    let mut poking = 0usize;
    let mut stray = 0usize;
    for &mi in &parts {
        for &p in &meshes[mi] {
            match depth_of(p) {
                Some(d) if d <= lip_thickness => {
                    worst = worst.max(d);
                    poking += 1;
                }
                Some(_) => stray += 1,
                None => {}
            }
        }
    }
    let shift = if poking > 0 { worst + MARGIN_M } else { 0.0 };
     
    let whole = scan::build_query(
        &meshes[prior.body],
        &prior.meshes[prior.body].prim.triangles,
    );
    let front = geom::v([0.0, 0.0, 1.0]);
    let seen_points = |moved: &dyn Fn(usize, usize) -> geom::V| -> Vec<geom::V> {
        parts
            .iter()
            .flat_map(|&mi| (0..meshes[mi].len()).map(move |k| (mi, k)))
            .map(|(mi, k)| moved(mi, k))
            .filter(|q| {
                whole
                    .first_ray_distance(geom::a(*q), geom::a(front), reach)
                    .is_none()
            })
            .collect()
    };
    let seen_at = |moved: &dyn Fn(usize, usize) -> geom::V| -> Vec<f64> {
        seen_points(moved).iter().map(|q| q.y).collect()
    };
    let base = |mi: usize, k: usize| geom::v(meshes[mi][k]) + back * shift;
     
    let lip_depth = match (landmarks.get("mouth_l"), landmarks.get("mouth_r")) {
        (Some(l), Some(r)) => Some((l[2] + r[2]) * 0.5),
        _ => None,
    };
    let lip_tolerance = LIP_DEPTH_TOLERANCE_M * prior.height() / 1.75;
    let seen_points0 = seen_points(&base);
    let scan_closed = lip_depth.map_or(0, |z| {
        seen_points0
            .iter()
            .filter(|q| {
                scan.query
                    .first_ray_distance(geom::a(**q), geom::a(front), reach)
                    .is_some_and(|d| q.z + d >= z - lip_tolerance)
            })
            .count()
    });
    let closed_lips = !seen_points0.is_empty() && scan_closed * 2 > seen_points0.len();
    let mut seen = seen_at(&base);
    seen.sort_by(f64::total_cmp);
    let seen_span = if seen.len() < 2 {
        0.0
    } else {
        seen[seen.len() * 95 / 100] - seen[seen.len() * 5 / 100]
    };
    let slit = SLIT_M * prior.height() / 1.75;
    // Per vertex: +1 moves up with the upper row, -1 down with the lower.
    let mut side: Vec<Vec<f64>> = parts
        .iter()
        .map(|&mi| vec![0.0; meshes[mi].len()])
        .collect();
    let mut parted = 0.0;
    // Only a row of teeth in a slit: a handful of seen points is not worth
    // moving the rows for (five of them moved the stocky woman's 5.5 mm
    // apart), and lips the fit left apart over a closed scan mouth are
    // closed by `lips_seal` (parting the rows for them, up to 11.5 mm each,
    // left the teeth out of place whenever the mouth opened).
    if seen.len() >= SEEN_MIN && seen_span < slit {
        let centre = seen[seen.len() / 2];
        for (pi, &mi) in parts.iter().enumerate() {
            if prior.meshes[mi].kind == PartKind::Tongue {
                side[pi].iter_mut().for_each(|x| *x = -1.0);
                continue;
            }
            // Each tooth (connected piece) moves with its row.
            let n = meshes[mi].len();
            let mut root: Vec<usize> = (0..n).collect();
            fn find(r: &mut [usize], mut i: usize) -> usize {
                while r[i] != i {
                    r[i] = r[r[i]];
                    i = r[i];
                }
                i
            }
            for t in &prior.meshes[mi].prim.triangles {
                let a0 = find(&mut root, t[0] as usize);
                for &j in &t[1..] {
                    let b0 = find(&mut root, j as usize);
                    root[b0] = a0;
                }
            }
            let mut sum: std::collections::HashMap<usize, (f64, usize)> = Default::default();
            for k in 0..n {
                let r = find(&mut root, k);
                let e = sum.entry(r).or_insert((0.0, 0));
                e.0 += geom::v(meshes[mi][k]).y;
                e.1 += 1;
            }
            for k in 0..n {
                let r = find(&mut root, k);
                let (y, c) = sum[&r];
                side[pi][k] = if y / c as f64 >= centre { 1.0 } else { -1.0 };
            }
        }
        let index: std::collections::HashMap<usize, usize> =
            parts.iter().enumerate().map(|(pi, &mi)| (mi, pi)).collect();
        let part_max = SLIT_PART_MAX_M * prior.height() / 1.75;
        while parted < part_max {
            parted += 0.0005;
            let moved =
                |mi: usize, k: usize| base(mi, k) + geom::V::y() * (side[index[&mi]][k] * parted);
            if seen_at(&moved).is_empty() {
                break;
            }
        }
    }
    for (pi, &mi) in parts.iter().enumerate() {
        for (k, p) in meshes[mi].iter_mut().enumerate() {
            *p = geom::a(geom::v(*p) + back * shift + geom::V::y() * (side[pi][k] * parted));
        }
    }
     
    let lip_z = lip_depth.unwrap_or_else(|| {
        parts
            .iter()
            .flat_map(|&mi| meshes[mi].iter())
            .map(|p| p[2])
            .fold(f64::NEG_INFINITY, f64::max)
    });
    let reach = prior.height();
    let inside = |p: charloom_core::Vec3| mouth_inside_outer_skin(&body, p, reach, MARGIN_M);
    // A purely forward ray can miss the head at the sides of a narrow
    // neck. In that case move toward a verified interior mouth point;
    // never stop outside or march through the front of the lips.
    let mouth_centre = match (landmarks.get("mouth_l"), landmarks.get("mouth_r")) {
        (Some(l), Some(r)) => geom::a((geom::v(*l) + geom::v(*r)) * 0.5),
        _ => {
            let ps: Vec<_> = parts.iter().flat_map(|&mi| meshes[mi].iter()).collect();
            geom::a(ps.iter().fold(geom::V::zeros(), |s, p| s + geom::v(**p)) / ps.len() as f64)
        }
    };
    let back_z = meshes[prior.body]
        .iter()
        .map(|p| p[2])
        .fold(lip_z, f64::min);
    let mut anchor = None;
    let mut anchor_clearance = 0.0f64;
    for k in 1..=((lip_z - back_z) / MARGIN_M).ceil() as usize {
        let p = [
            mouth_centre[0],
            mouth_centre[1],
            lip_z - k as f64 * MARGIN_M,
        ];
        if inside(p) {
            let clearance = body.closest_point(p, reach, None).unwrap().distance_m;
            if clearance > anchor_clearance {
                anchor = Some(p);
                anchor_clearance = clearance;
            }
        }
    }
    let mut out_back = 0usize;
    let mut fallback = 0usize;
    let mut max_forward = 0.0f64;
    let mut checked = 0usize;
    for &mi in &parts {
        for p in meshes[mi].iter_mut() {
            if p[2] >= lip_z {
                continue;
            }
            checked += 1;
            if !inside(*p) {
                let q0 = *p;
                let end = [p[0], p[1], lip_z];
                let found = mouth_first_inside(&body, q0, end, reach, MARGIN_M).or_else(|| {
                    let q = mouth_first_inside(&body, q0, anchor?, reach, MARGIN_M)?;
                    fallback += 1;
                    Some(q)
                });
                if let Some(q) = found {
                    max_forward = max_forward.max(q[2] - q0[2]);
                    *p = q;
                    out_back += 1;
                }
            }
        }
    }
     
    let mut surface_moved = 0usize;
    let mut surface_unresolved = 0usize;
    let mut surface_margin_shortfall = 0usize;
    for &mi in &parts {
        let (moved, outside, shortfall) = mouth_contain_surface(
            &body,
            &mut meshes[mi],
            &prior.meshes[mi].prim.triangles,
            lip_z,
            reach,
            MARGIN_M,
            anchor.map(|p| {
                (
                    p,
                    (anchor_clearance - MARGIN_M - 2e-6)
                        .min(lip_z - p[2] - 2e-6)
                        .max(0.0),
                )
            }),
        );
        surface_moved += moved;
        surface_unresolved += outside;
        surface_margin_shortfall += shortfall;
    }
    // Audit the final positions too, including the surface corrections.
    let mut unresolved = 0usize;
    let mut min_clearance = f64::INFINITY;
    for &mi in &parts {
        for &p in &meshes[mi] {
            if p[2] < lip_z {
                if inside(p) {
                    min_clearance =
                        min_clearance.min(body.closest_point(p, reach, None).unwrap().distance_m);
                } else {
                    unresolved += 1;
                }
            }
        }
    }
    json!({"poking_vertices": poking, "stray_vertices": stray, "seen_vertices": seen.len(), "seen_span_m": seen_span, "scan_closed_vertices": scan_closed, "closed_lips": closed_lips, "parted_m": parted, "shift_m": shift, "out_back_vertices": out_back, "out_back_checked_vertices": checked, "out_back_unresolved_vertices": unresolved, "out_back_max_forward_m": max_forward, "out_back_margin_m": MARGIN_M, "out_back_min_clearance_m": min_clearance, "out_back_interior_fallback_vertices": fallback, "out_back_surface_moved_vertices": surface_moved, "out_back_surface_outside_samples": surface_unresolved, "out_back_surface_margin_shortfall_samples": surface_margin_shortfall})
}

/// Inside the outer mouth/head envelope even when the lips are open.
/// Distance to the nearest triangle alone has no inside/outside meaning
/// at a concave nape or at an open lip edge. A rear exit or two opposed
/// lateral exits establish containment; the closest surface supplies the
/// clearance separately.
fn mouth_inside_outer_skin(
    body: &charloom_engine::surface_query::SurfaceQuery,
    p: charloom_core::Vec3,
    reach: f64,
    margin: f64,
) -> bool {
    if body.closest_point(p, reach, None).is_none_or(|h| h.distance_m < margin + 1e-6) {
        return false;
    }
    let exits = |direction| {
        let Some(first) = body.first_ray_distance(p, direction, reach) else { return false };
        let Some(exit) = body.outward_ray_distance(p, direction, reach) else { return false };
        (first - exit).abs() < 1e-6
    };
    // Removing cavity faces leaves openings in the query envelope. A ray
    // through one is not proof that a point is outside. In that case two
    // opposed lateral exit crossings certify the cheek/neck cross-section.
    exits([0.0, 0.0, -1.0]) || (exits([-1.0, 0.0, 0.0]) && exits([1.0, 0.0, 0.0]))

}

 fn mouth_first_inside(
    body: &charloom_engine::surface_query::SurfaceQuery,
    start: charloom_core::Vec3,
    end: charloom_core::Vec3,
    reach: f64,
    margin: f64,
) -> Option<charloom_core::Vec3> {
    let start = geom::v(start);
    let delta = geom::v(end) - start;
    let steps = (delta.norm() / (margin * 0.25)).ceil().max(1.0) as usize;
    let point = |t: f64| geom::a(start + delta * t);
    let inside = |t| mouth_inside_outer_skin(body, point(t), reach, margin);
    if inside(0.0) {
        return Some(point(0.0));
    }
    for k in 1..=steps {
        let mut hi = k as f64 / steps as f64;
        if !inside(hi) {
            continue;
        }
        let mut lo = (k - 1) as f64 / steps as f64;
        for _ in 0..12 {
            let mid = (lo + hi) * 0.5;
            if inside(mid) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return Some(point(hi));
    }
    None
}

fn mouth_contain_surface(
    body: &charloom_engine::surface_query::SurfaceQuery,
    points: &mut [charloom_core::Vec3],
    triangles: &[[u32; 3]],
    lip_z: f64,
    reach: f64,
    margin: f64,
    interior_ball: Option<(charloom_core::Vec3, f64)>,
) -> (usize, usize, usize) {
    let mut moved = vec![false; points.len()];
     
    loop {
        let mut repair = vec![false; points.len()];
        let mut bad = 0usize;
        let mut outside = 0usize;
        for tri in triangles {
            let p = tri.map(|i| geom::v(points[i as usize]));
            let edge = (p[1] - p[0])
                .norm()
                .max((p[2] - p[1]).norm())
                .max((p[0] - p[2]).norm());
            let divisions = (edge / margin).ceil().max(1.0) as usize;
            let mut check = |q: geom::V| {
                if q.z >= lip_z || mouth_inside_outer_skin(body, geom::a(q), reach, margin) {
                    return;
                }
                bad += 1;
                if !mouth_inside_outer_skin(body, geom::a(q), reach, 0.0) {
                    outside += 1;
                }
                for &i in tri {
                    repair[i as usize] = true;
                }
            };
            check((p[0] + p[1] + p[2]) / 3.0);
            for i in 0..=divisions {
                for j in 0..=divisions - i {
                    let u = i as f64 / divisions as f64;
                    let v = j as f64 / divisions as f64;
                    check(p[0] * (1.0 - u - v) + p[1] * u + p[2] * v);
                }
            }
        }
        let result = (moved.iter().filter(|v| **v).count(), outside, bad - outside);
        if bad == 0 {
            return result;
        }
        let Some((centre, radius)) = interior_ball else {
            return result;
        };
        let mut changed = false;
        for (i, p) in points.iter_mut().enumerate() {
            if !repair[i] || moved[i] || p[2] >= lip_z {
                continue;
            }
            let delta = geom::v(*p) - geom::v(centre);
            if delta.norm() > radius {
                *p = geom::a(geom::v(centre) + delta.normalize() * radius);
                moved[i] = true;
                changed = true;
            }
        }
        if !changed {
            return result;
        }
    }
}

/// Teeth seen through a lip opening shorter than this (at 1.75 m) are
/// peeking through closed lips (`mouth_behind_lips`).
const SLIT_M: f64 = 0.004;
/// The furthest each row moves away from such a slit.
const SLIT_PART_MAX_M: f64 = 0.006;
/// Fewer seen teeth points than this are left alone.
const SEEN_MIN: usize = 20;
/// The thickest lips (at 1.75 m): how far behind a tooth the skin it pokes
/// through can be.
const LIP_THICKNESS_M: f64 = 0.02;
/// Scan surface in front of a seen tooth this far behind the marked mouth
/// corners' depth (at 1.75 m) is still the scan's lips.
const LIP_DEPTH_TOLERANCE_M: f64 = 0.003;

/// The underwear for the naked body: chosen assets fitted to `body`, the
/// loaded assets (for materials and credits) and the report.
fn fit_underwear(
    config: &Config,
    prior_path: &Path,
    prior: &Prior,
    guided_sex: Option<String>,
    body: &underwear::BodyInput,
) -> Result<(Vec<underwear::Piece>, Vec<underwear::Asset>, Value)> {
    let (sex, sex_source) = match guided_sex {
        Some(s) => (s, "guidance"),
        None => match underwear::prior_sex(prior_path) {
            Some(s) => (s.to_string(), "prior preparation.json"),
            None => ("unknown".to_string(), "default"),
        },
    };
    let ids = underwear::choose(&config.underwear, &sex);
    let mut report =
        json!({"choice": config.underwear, "sex": sex, "sex_source": sex_source, "assets": ids});
    if ids.is_empty() {
        report["applied"] = json!(false);
        return Ok((vec![], vec![], report));
    }
    let auto = config.underwear == underwear::Underwear::default();
    let root = underwear::find_assets(config.underwear_assets.as_deref(), prior_path);
    let base = root.as_deref().and_then(underwear::find_base_mesh);
    let (Some(root), Some(base_path)) = (root, base) else {
        ensure!(
            auto,
            "underwear_assets_missing: set underwear_assets (asset packs and MakeHuman's base.obj)"
        );
        report["applied"] = json!(false);
        report["reason"] = json!("assets or MakeHuman base.obj not found");
        return Ok((vec![], vec![], report));
    };
    let assets = ids
        .iter()
        .map(|id| underwear::Asset::load(&root, id))
        .collect::<Result<Vec<_>>>()?;
    let base_bytes = std::fs::read(&base_path)?;
    let base = underwear::parse_obj(&String::from_utf8_lossy(&base_bytes))?;
    let pb = prior.body();
    let hm = underwear::Hm08::new(&base, &pb.prim.uvs, &pb.prim.triangles)?;
    let (pieces, rows) = underwear::fit(
        &assets,
        &hm,
        body,
        &pb.prim.positions,
        config.underwear_clearance_m,
    )?;
    report["applied"] = json!(true);
    report["assets_dir"] = json!(root);
    report["base_mesh"] = json!({"path": base_path, "sha256": format!("{:x}", <sha2::Sha256 as sha2::Digest>::digest(&base_bytes))});
    report["hm08_map"] = hm.report.clone();
    report["pieces"] = rows;
    Ok((pieces, assets, report))
}

/// Median scan colour in a small disc around each pupil landmark, applied as
/// a per-channel gain to the iris pixels (darker than the sclera) of
/// MakeHuman's eye texture.
fn tint_iris(
    prior: &Prior,
    images: &mut [source::ImageData],
    scan: &scan::Scan,
    landmarks: &std::collections::BTreeMap<String, charloom_core::Vec3>,
) -> Result<Value> {
    let (Some(el), Some(er)) = (landmarks.get("eye_l"), landmarks.get("eye_r")) else {
        return Ok(Value::Null);
    };
    let span = (geom::v(*el) - geom::v(*er)).norm();
    // The iris is a ring around the pupil: sample an annulus (inner radius
    // past the pupil, outer about the iris edge, ~12 mm iris on a ~62 mm
    // interocular span) and drop the lightest quarter (sclera, highlights).
    let (inner, outer) = (span * 0.035, span * 0.08);
    let mut samples: Vec<[f64; 3]> = vec![];
    for eye in [el, er] {
        for ring in 0..4 {
            let r = inner + (outer - inner) * ring as f64 / 3.0;
            for k in 0..24 {
                let phi = std::f64::consts::TAU * k as f64 / 24.0;
                let p = [
                    eye[0] + r * phi.cos(),
                    eye[1] + r * phi.sin(),
                    eye[2] + 0.005,
                ];
                if let Some(hit) = scan.query.closest_point(p, 0.02, None)
                    && let Some(c) = scan.color_at(hit.source[1], hit.barycentric)
                {
                    samples.push(c);
                }
            }
        }
    }
    if samples.len() < 10 {
        return Ok(Value::Null);
    }
    let luminance = |c: &[f64; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    samples.sort_by(|a0, b0| luminance(a0).total_cmp(&luminance(b0)));
    samples.truncate(samples.len() * 3 / 4);
    let median = |k: usize| {
        let mut x: Vec<f64> = samples.iter().map(|c| c[k]).collect();
        x.sort_by(f64::total_cmp);
        x[x.len() / 2]
    };
    let target = [median(0), median(1), median(2)].map(source::srgb_to_linear);
    let materials = source::read_materials(&prior.loaded)?;
    let Some(index) = prior
        .meshes
        .iter()
        .find(|m| m.kind == PartKind::Eyes)
        .and_then(|m| m.prim.material)
        .and_then(|m| materials[m].base_color)
    else {
        return Ok(Value::Null);
    };
    let img = &mut images[index].decoded;
    let lum =
        |p: &image::Rgba<u8>| 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64;
    let mut ls: Vec<f64> = img.pixels().map(lum).collect();
    ls.sort_by(f64::total_cmp);
    let sclera = ls[ls.len() * 9 / 10];
    let is_iris = |p: &image::Rgba<u8>| lum(p) < sclera * 0.6;
    let (mut sum, mut n) = ([0.0; 3], 0usize);
    for p in img.pixels().filter(|p| is_iris(p)) {
        for k in 0..3 {
            sum[k] += source::srgb_to_linear(p[k] as f64 / 255.0);
        }
        n += 1;
    }
    if n == 0 {
        return Ok(Value::Null);
    }
    // Keep MakeHuman's iris pattern (relative luminance), take the scan's
    // colour: pixel = target colour × (pixel luminance / mean iris luminance).
    let mean_lum = (0.2126 * sum[0] + 0.7152 * sum[1] + 0.0722 * sum[2]) / n as f64;
    let gain: [f64; 3] = std::array::from_fn(|k| target[k] / sum[k].max(1e-9) * n as f64);
    for p in img.pixels_mut() {
        if is_iris(p) {
            let lin: [f64; 3] =
                std::array::from_fn(|k| source::srgb_to_linear(p[k] as f64 / 255.0));
            let rel = (0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]) / mean_lum.max(1e-6);
            for k in 0..3 {
                p[k] = (source::linear_to_srgb(target[k] * rel.min(3.0)) * 255.0).round() as u8;
            }
        }
    }
    let mut png = vec![];
    image::DynamicImage::ImageRgba8(img.clone())
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
    images[index].bytes = png;
    images[index].mime = "image/png".into();
    Ok(
        json!({"scan_iris_srgb":[median(0), median(1), median(2)],"samples":samples.len(),"iris_pixels":n,"gain_linear":gain}),
    )
}

 struct Opening {
    upper: Vec<(f64, f64)>,
    lower: Vec<(f64, f64)>,
}

impl Opening {
    /// Lid margin height at `s`: piecewise linear through four or more
    /// points; the parabola through three (corner, top or bottom, corner).
    fn margin(points: &[(f64, f64)], s: f64) -> Option<f64> {
        let (first, last) = (points.first()?, points.last()?);
        if s < first.0 || s > last.0 {
            return None;
        }
        if let [a0, b0, c0] = points {
            let l = |x: f64, (xi, _): (f64, f64), (xj, _): (f64, f64), (xk, _): (f64, f64)| {
                // The denominator's magnitude clamped, its sign kept (a
                // clamp before the sign turned the middle point's negative
                // denominator into -1e-12: margins of millions of metres
                // for every eye marked without lid points).
                let den = (xi - xj) * (xi - xk);
                (x - xj) * (x - xk) / den.abs().max(1e-12).copysign(den)
            };
            return Some(
                a0.1 * l(s, *a0, *b0, *c0)
                    + b0.1 * l(s, *b0, *a0, *c0)
                    + c0.1 * l(s, *c0, *a0, *b0),
            );
        }
        let k = points.windows(2).position(|w| s <= w[1].0)?;
        let (p, q) = (points[k], points[k + 1]);
        let f = if q.0 - p.0 > 1e-12 {
            (s - p.0) / (q.0 - p.0)
        } else {
            0.0
        };
        Some(p.1 + (q.1 - p.1) * f)
    }

    /// How deep (s, t) lies inside the opening: 1 on the line midway between
    /// the lids, 0 on a lid margin, negative outside.
    fn depth(&self, s: f64, t: f64) -> f64 {
        let (Some(up), Some(lo)) = (Self::margin(&self.upper, s), Self::margin(&self.lower, s))
        else {
            return -1.0;
        };
        let half = (up - lo) * 0.5;
        if half <= 1e-9 {
            return -1.0;
        }
        ((up - t).min(t - lo) / half).min(1.0)
    }
}

/// Per body vertex: weight in the left / right brow zone (full within 6 mm
/// of MakeHuman's eyebrow mesh in its rest shape, none beyond 12 mm), and
/// every body vertex's mirror partner (itself where unmatched). Empty without
/// an eyebrow mesh.
fn brow_zones(prior: &Prior) -> (Vec<[f64; 2]>, Vec<u32>) {
    let Some(brows) = prior.meshes.iter().find(|m| m.kind == PartKind::Eyebrows) else {
        return (vec![], vec![]);
    };
    let rest: Vec<nalgebra::Vector3<f64>> = prior
        .body()
        .prim
        .positions
        .iter()
        .map(|p| geom::v(*p))
        .collect();
    let mirror = symmetry::Mirror::new(&rest, prior.height() * 0.0006);
    let partner: Vec<u32> = (0..rest.len())
        .map(|i| {
            if mirror.valid[i] {
                mirror.partner[i]
            } else {
                i as u32
            }
        })
        .collect();
    let brow: Vec<nalgebra::Vector3<f64>> =
        brows.prim.positions.iter().map(|p| geom::v(*p)).collect();
    let (lo, hi) = brow.iter().fold(
        (
            nalgebra::Vector3::repeat(f64::INFINITY),
            nalgebra::Vector3::repeat(f64::NEG_INFINITY),
        ),
        |(l, h), p| (l.inf(p), h.sup(p)),
    );
    let zones = rest
        .iter()
        .map(|p| {
            if p.iter()
                .zip(lo.iter().zip(hi.iter()))
                .any(|(x, (l, h))| *x < l - 0.012 || *x > h + 0.012)
            {
                return [0.0, 0.0];
            }
            let d = brow
                .iter()
                .map(|b| (b - p).norm())
                .fold(f64::INFINITY, f64::min);
            let t = ((0.012 - d) / 0.006).clamp(0.0, 1.0);
            let w = t * t * (3.0 - 2.0 * t);
            if p.x >= mirror.plane {
                [w, 0.0]
            } else {
                [0.0, w]
            }
        })
        .collect();
    (zones, partner)
}

struct EyeClearing {
    moved: usize,
    report: Value,
    /// Per body vertex: socket lining inside the marked opening, [0, 1].
    lining: Vec<f64>,
}

/// Body vertices whose front projection lies inside a marked eye opening
/// (clear of its lid margins) and in front of the placed eyeball there are
/// moved straight back to just behind the eyeball surface: the socket
/// lining between a stylised (wide, eyeball-pushed-back) opening and the
/// eyeball otherwise showed as skin over the eye.
fn clear_eye_openings(
    prior: &Prior,
    rig: &fit::Rig,
    body: &mut [charloom_core::Vec3],
    landmarks: &std::collections::BTreeMap<String, charloom_core::Vec3>,
) -> EyeClearing {
    let Some(mi) = prior.meshes.iter().position(|m| m.kind == PartKind::Eyes) else {
        return EyeClearing {
            moved: 0,
            report: Value::Null,
            lining: vec![],
        };
    };
    let eyes = &prior.meshes[mi];
    let mut moved = 0usize;
    let mut report = serde_json::Map::new();
    let mut lining = vec![0.0f64; body.len()];
    for (side, joint) in [("l", "eye.L"), ("r", "eye.R")] {
        let (Some(marked), Some(&j)) = (
            MarkedOpening::new(landmarks, side),
            prior.by_name.get(joint),
        ) else {
            continue;
        };
        let on_eye: Vec<bool> = eyes
            .influences
            .iter()
            .map(|row| row.iter().any(|x| x.joint == j && x.weight > 0.5))
            .collect();
        let tris: Vec<[u32; 3]> = eyes
            .prim
            .triangles
            .iter()
            .filter(|t| t.iter().all(|&i| on_eye[i as usize]))
            .copied()
            .collect();
        if tris.is_empty() {
            continue;
        }
        let ball = scan::build_query(&rig.meshes[mi], &tris);
        // Lining only: within 0.6 eyeball radii in front of the eyeball
        // (further out is face, e.g. a big nose seen over the inner corner).
        let pts: Vec<nalgebra::Vector3<f64>> = tris
            .iter()
            .flatten()
            .map(|&i| geom::v(rig.meshes[mi][i as usize]))
            .collect();
        let centre = pts.iter().sum::<nalgebra::Vector3<f64>>() / pts.len() as f64;
        let radius = pts.iter().map(|p| (p - centre).norm()).fold(0.0, f64::max);
        let (mut count, mut deepest) = (0usize, 0.0f64);
         
        let mut hood: Vec<(f64, f64)> = ["inner", "upper_1", "upper_2", "upper_3", "top", "outer"]
            .iter()
            .filter_map(|k| landmarks.get(&format!("eye_{side}_{k}")))
            .map(|p| (marked.st(geom::v(*p)).0, p[2]))
            .collect();
        hood.sort_by(|a0, b0| a0.0.total_cmp(&b0.0));
        let hood_z = |s: f64| -> f64 {
            let Some(k) = hood.iter().position(|h| h.0 >= s) else {
                return hood.last().map_or(f64::NEG_INFINITY, |h| h.1);
            };
            if k == 0 {
                return hood[0].1;
            }
            let (a0, b0) = (hood[k - 1], hood[k]);
            a0.1 + (b0.1 - a0.1) * ((s - a0.0) / (b0.0 - a0.0).max(1e-9))
        };
         
        let outline: Vec<[f64; 2]> = {
            let chain = |keys: &[&str]| -> Vec<(f64, [f64; 2])> {
                let mut c: Vec<(f64, [f64; 2])> = keys.iter()
                    .filter_map(|k| landmarks.get(&format!("eye_{side}_{k}")))
                    .map(|p| (marked.st(geom::v(*p)).0, [p[0], p[1]]))
                    .collect();
                c.sort_by(|a0, b0| a0.0.total_cmp(&b0.0));
                c
            };
            let upper = chain(&["inner", "upper_1", "upper_2", "upper_3", "outer"]);
            let upper = if upper.len() == 5 { upper } else { chain(&["inner", "top", "outer"]) };
            let lower = chain(&["lower_1", "lower_2", "lower_3"]);
            let lower = if lower.len() == 3 { lower } else { chain(&["bottom"]) };
            upper.into_iter().chain(lower.into_iter().rev()).map(|c| c.1).collect()
        };
        let shown_inside = |q: nalgebra::Vector3<f64>| -> bool {
            let mut inside = false;
            for k in 0..outline.len() {
                let (a0, b0) = (outline[k], outline[(k + 1) % outline.len()]);
                if (a0[1] > q.y) != (b0[1] > q.y) && q.x < a0[0] + (b0[0] - a0[0]) * (q.y - a0[1]) / (b0[1] - a0[1]) {
                    inside = !inside;
                }
            }
            inside
        };
        for p in body.iter_mut() {
            let q = geom::v(*p);
            let (s0, t0) = marked.st(q);
            // The lid margins too (they rest on the eyeball), just short.
            let depth = marked.opening.depth(s0, t0);
            if depth <= -0.05 {
                continue;
            }
            let behind = |r: nalgebra::Vector3<f64>| -> Option<f64> {
                let origin = [r.x, r.y, r.z + 0.05];
                let d = ball.first_ray_distance(origin, [0.0, 0.0, -1.0], 0.1)?;
                Some(origin[2] - d)
            };
            let Some(front) = behind(q) else {
                continue;
            };
            let covering = q.z < front + 2.0 * radius && shown_inside(q);
            if q.z <= front - 0.0002 || (!covering && q.z >= (front + 0.6 * radius).max(hood_z(s0) + 0.002)) {
                continue;
            }
             
            let (Some(up), Some(lo)) = (
                Opening::margin(&marked.opening.upper, s0),
                Opening::margin(&marked.opening.lower, s0),
            ) else {
                continue;
            };
            let t_new = if (up - t0).abs() < (t0 - lo).abs() {
                up
            } else {
                lo
            };
            let r = q + marked.up * (t_new - t0);
            let z = behind(r).unwrap_or(front).min(r.z) - 0.0003;
            let r = nalgebra::Vector3::new(r.x, r.y, z.min(q.z));
            deepest = deepest.max((r - q).norm());
            *p = geom::a(r);
            count += 1;
        }
         
        {
            let mesh = prior.body();
            let weld = geom::Weld::new(&mesh.bind, 1e-7);
            let free: Vec<bool> = body.iter().map(|p| {
                let q = geom::v(*p);
                if (q - centre).norm() > 1.8 * radius {
                    return false;
                }
                let (s0, t0) = marked.st(q);
                marked.opening.depth(s0, t0) > 0.0
                    && ball.first_ray_distance([q.x, q.y, q.z + 0.05], [0.0, 0.0, -1.0], 0.1).is_none()
            }).collect();
            let mut loose = vec![false; weld.representative.len()];
            for (i, f) in free.iter().enumerate() {
                if *f {
                    loose[weld.id[i]] = true;
                }
            }
            let mut around: Vec<Vec<usize>> = vec![vec![]; loose.len()];
            for t in &mesh.prim.triangles {
                let w = t.map(|i| weld.id[i as usize]);
                for k in 0..3 {
                    if loose[w[k]] {
                        around[w[k]].extend([w[(k + 1) % 3], w[(k + 2) % 3]]);
                    }
                }
            }
            let mut at: Vec<nalgebra::Vector3<f64>> = weld.representative.iter().map(|&i| geom::v(body[i])).collect();
            for _ in 0..40 {
                let next: Vec<(usize, nalgebra::Vector3<f64>)> = (0..at.len()).filter(|&w| loose[w] && !around[w].is_empty()).map(|w| {
                    (w, around[w].iter().fold(nalgebra::Vector3::zeros(), |s, &j| s + at[j]) / around[w].len() as f64)
                }).collect();
                for (w, q) in next {
                    at[w] = q;
                }
            }
            for (i, p) in body.iter_mut().enumerate() {
                if loose[weld.id[i]] {
                    *p = geom::a(at[weld.id[i]]);
                    count += 1;
                }
            }
        }
        moved += count;
         
        let mut lined = 0usize;
        for (l, p) in lining.iter_mut().zip(body.iter()) {
            let q = geom::v(*p);
            if (q - centre).norm() > 1.6 * radius {
                continue;
            }
            let (s0, t0) = marked.st(q);
            let w = ((marked.opening.depth(s0, t0) - 0.01) / 0.06).clamp(0.0, 1.0);
            if w > 0.0 {
                lined += 1;
            }
            *l = l.max(w);
        }
        report.insert(
            side.into(),
            json!({"vertices": count, "max_moved_m": deepest, "lining_vertices": lined}),
        );
    }
    EyeClearing {
        moved,
        report: Value::Object(report),
        lining,
    }
}

 struct MarkedOpening {
    centre: nalgebra::Vector3<f64>,
    u: nalgebra::Vector3<f64>,
    up: nalgebra::Vector3<f64>,
    opening: Opening,
}

impl MarkedOpening {
    fn new(
        landmarks: &std::collections::BTreeMap<String, charloom_core::Vec3>,
        side: &str,
    ) -> Option<Self> {
        let get = |k: &str| {
            landmarks
                .get(&format!("eye_{side}_{k}"))
                .map(|p| geom::v(*p))
        };
        let (inner, outer, top, bottom) =
            (get("inner")?, get("outer")?, get("top")?, get("bottom")?);
        let centre = (inner + outer + top + bottom) / 4.0;
        let u = (outer - inner).normalize();
        let up = ((top - bottom) - u * (top - bottom).dot(&u)).normalize();
        let st = |p: nalgebra::Vector3<f64>| ((p - centre).dot(&u), (p - centre).dot(&up));
        let lid = |part: &str, extreme: nalgebra::Vector3<f64>| -> Vec<(f64, f64)> {
            let marked: Vec<nalgebra::Vector3<f64>> = (1..=3)
                .filter_map(|k| get(&format!("{part}_{k}")))
                .collect();
            let mut pts = vec![st(inner)];
            if marked.len() == 3 {
                pts.extend(marked.iter().map(|p| st(*p)));
            } else {
                pts.push(st(extreme));
            }
            pts.push(st(outer));
            pts.sort_by(|a0, b0| a0.0.total_cmp(&b0.0));
            pts
        };
        Some(Self {
            centre,
            u,
            up,
            opening: Opening {
                upper: lid("upper", top),
                lower: lid("lower", bottom),
            },
        })
    }

         fn st(&self, p: nalgebra::Vector3<f64>) -> (f64, f64) {
        (
            (p - self.centre).dot(&self.u),
            (p - self.centre).dot(&self.up),
        )
    }
}

/// Bakes the scan's painted eyes onto the front hemisphere of MakeHuman's
/// eyeball texture: each eyeball texel is carried forward (+Z, the face
/// direction) to the scan; only hits inside the marked eye opening (and just
/// past its margin) are used, the rest is the scan's sclera colour. Both
/// eyeballs share one texture, so the two eyes' samples are averaged.
fn bake_eyes(
    prior: &Prior,
    rig: &fit::Rig,
    images: &mut [source::ImageData],
    scan: &scan::Scan,
    landmarks: &std::collections::BTreeMap<String, charloom_core::Vec3>,
) -> Result<Value> {
    use nalgebra::Vector3;
    let Some((mi, eyes)) = prior
        .meshes
        .iter()
        .enumerate()
        .find(|(_, m)| m.kind == PartKind::Eyes)
    else {
        return Ok(Value::Null);
    };
    let materials = source::read_materials(&prior.loaded)?;
    let Some(index) = eyes.prim.material.and_then(|m| materials[m].base_color) else {
        return Ok(Value::Null);
    };
    let positions = &rig.meshes[mi];
    let (w, h) = (
        images[index].decoded.width() as usize,
        images[index].decoded.height() as usize,
    );
    let mut acc = vec![[0f64; 4]; w * h];
    // Inner-opening samples (r < 0.75) for the sclera colour estimate.
    let mut inner_samples: Vec<[f64; 3]> = vec![];
    // Front texels whose forward ray leaves the scan's opening: filled with
    // the scan's sclera colour afterwards.
    let mut outside = vec![false; w * h];
    let mut used_eyes = vec![];
    for (side, joint) in [("l", "eye.L"), ("r", "eye.R")] {
        let Some(marked) = MarkedOpening::new(landmarks, side) else {
            continue;
        };
        let Some(&j) = prior.by_name.get(joint) else {
            continue;
        };
        let st = |p: Vector3<f64>| marked.st(p);
        let opening = &marked.opening;
        let on_eye: Vec<bool> = eyes
            .influences
            .iter()
            .map(|row| row.iter().any(|x| x.joint == j && x.weight > 0.5))
            .collect();
        used_eyes.push(side);
        for tri in eyes
            .prim
            .triangles
            .iter()
            .filter(|t| t.iter().all(|&i| on_eye[i as usize]))
        {
            let p3 = tri.map(|i| geom::v(positions[i as usize]));
            let normal = (p3[1] - p3[0]).cross(&(p3[2] - p3[0]));
            // Everything that can show through a wide (stylised) opening.
            if normal.norm() < 1e-14 || normal.normalize().z < -0.3 {
                continue;
            }
            let uv = tri.map(|i| {
                let t = eyes.prim.uvs[i as usize];
                [t[0] as f64 * w as f64, t[1] as f64 * h as f64]
            });
            let area = (uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1])
                - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]);
            if area.abs() < 1e-12 {
                continue;
            }
            let x0 = uv
                .iter()
                .map(|p| p[0])
                .fold(f64::INFINITY, f64::min)
                .floor()
                .max(0.0) as usize;
            let x1 = (uv
                .iter()
                .map(|p| p[0])
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil() as usize)
                .min(w - 1);
            let y0 = uv
                .iter()
                .map(|p| p[1])
                .fold(f64::INFINITY, f64::min)
                .floor()
                .max(0.0) as usize;
            let y1 = (uv
                .iter()
                .map(|p| p[1])
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil() as usize)
                .min(h - 1);
            for y in y0..=y1 {
                for x in x0..=x1 {
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    let l1 = ((uv[2][0] - px) * (uv[0][1] - py)
                        - (uv[0][0] - px) * (uv[2][1] - py))
                        / area;
                    let l2 = ((uv[0][0] - px) * (uv[1][1] - py)
                        - (uv[1][0] - px) * (uv[0][1] - py))
                        / area;
                    let l0 = 1.0 - l1 - l2;
                    if l0 < -1e-6 || l1 < -1e-6 || l2 < -1e-6 {
                        continue;
                    }
                    let p = p3[0] * l0 + p3[1] * l1 + p3[2] * l2;
                    // Forward to the scan surface in front of the eyeball.
                    let origin = geom::a(p - Vector3::z() * 0.002);
                    let Some(t) = scan.query.first_ray_distance(origin, [0.0, 0.0, 1.0], 0.03)
                    else {
                        outside[y * w + x] = true;
                        continue;
                    };
                    let hit_p = geom::v(origin) + Vector3::z() * t;
                    let (s_hit, t_hit) = st(hit_p);
                    // 0 on the middle line, 1 at the drawn lid margin. A little
                    // past the margin is sampled too: where the fitted lid
                    // stops just outside the marks, the eyeball shows the
                    // scan's lid line there, not sclera.
                    let r = 1.0 - opening.depth(s_hit, t_hit);
                    if r > 1.2 {
                        outside[y * w + x] = true;
                        continue;
                    }
                    let Some(hit) = scan.query.closest_point(geom::a(hit_p), 0.002, None) else {
                        continue;
                    };
                    let Some(col) = scan.color_at(hit.source[1], hit.barycentric) else {
                        continue;
                    };
                    // Full weight up to the margin: feathering the outer
                    // quarter towards the sclera colour painted out the
                    // scan's eyeliner and lash line, and with the lids opened
                    // to the marks it showed as a white rim inside them.
                    let wgt = 1.0;
                    if r < 0.75 {
                        inner_samples.push(col.map(source::srgb_to_linear));
                    }
                    let k = y * w + x;
                    for q in 0..3 {
                        acc[k][q] += source::srgb_to_linear(col[q]) * wgt;
                    }
                    acc[k][3] += wgt;
                }
            }
        }
    }
    if used_eyes.is_empty() {
        return Ok(Value::Null);
    }
    // Scan sclera: the lightest of the inner-opening samples (iris, pupil and
    // eyeliner are dark). The least saturated of the lighter half picked the
    // pink lid-margin paint on some scans: a skin-coloured band on the white.
    let lum = |c: &[f64; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    // A large painted iris leaves little white: even the lightest 15 % then
    // held pink lid paint, and the eyeball outside the drawn opening came out
    // skin-coloured. Take the lightest 6 %, then the less saturated half.
    let mut light = inner_samples.clone();
    light.sort_by(|a0, b0| lum(a0).total_cmp(&lum(b0)));
    let mut light = light.split_off(light.len() - (light.len() * 6 / 100).max(1).min(light.len()));
    let chroma = |c: &[f64; 3]| {
        let (hi, lo) = (c[0].max(c[1]).max(c[2]), c[0].min(c[1]).min(c[2]));
        (hi - lo) / hi.max(1e-6)
    };
    light.sort_by(|a0, b0| chroma(a0).total_cmp(&chroma(b0)));
    light.truncate((light.len() / 2).max(1));
    let sclera: Option<[f64; 3]> = (inner_samples.len() > 40).then(|| {
        let slice = &light[..];
        std::array::from_fn(|q| {
            let mut x: Vec<f64> = slice.iter().map(|c| c[q]).collect();
            x.sort_by(f64::total_cmp);
            x[x.len() / 2]
        })
    });
    if let Some(sc) = sclera {
        for k in 0..w * h {
            if outside[k] && acc[k][3] <= 0.0 {
                acc[k] = [sc[0], sc[1], sc[2], 1.0];
            } else if acc[k][3] > 0.0 && acc[k][3] < 1.0 {
                // Feathered opening rim: towards the scan's white, never
                // MakeHuman's own painted iris.
                let f = 1.0 - acc[k][3];
                for q in 0..3 {
                    acc[k][q] += sc[q] * f;
                }
                acc[k][3] = 1.0;
            }
        }
    }
    let img = &mut images[index].decoded;
    let mut baked = 0usize;
    for (k, a0) in acc.iter().enumerate() {
        if a0[3] <= 0.0 {
            continue;
        }
        // The two eyeballs have their own UV islands: full weight is 1.
        let m = a0[3].min(1.0);
        let p = img.get_pixel_mut((k % w) as u32, (k / w) as u32);
        for q in 0..3 {
            let scan_c = a0[q] / a0[3];
            let old = source::srgb_to_linear(p[q] as f64 / 255.0);
            p[q] = (source::linear_to_srgb(scan_c * m + old * (1.0 - m)) * 255.0).round() as u8;
        }
        baked += 1;
    }
    if baked < 50 {
        return Ok(Value::Null);
    }
    let mut png = vec![];
    image::DynamicImage::ImageRgba8(img.clone())
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)?;
    images[index].bytes = png;
    images[index].mime = "image/png".into();
    Ok(
        json!({"method":"scan_eye_baked_onto_eyeball","eyes":used_eyes,"texels":baked,"texture":[w, h],"sclera_linear":sclera}),
    )
}

#[cfg(test)]
mod tests {
    use super::Opening;

    fn mouth_box(with_back: bool) -> charloom_engine::surface_query::SurfaceQuery {
        let p = [
            [-0.05, -0.05, -0.1], [0.05, -0.05, -0.1],
            [0.05, 0.05, -0.1], [-0.05, 0.05, -0.1],
            [-0.05, -0.05, 0.1], [0.05, -0.05, 0.1],
            [0.05, 0.05, 0.1], [-0.05, 0.05, 0.1],
        ];
        let t = [
            [0, 2, 1], [0, 3, 2], [0, 1, 5], [0, 5, 4],
            [3, 7, 6], [3, 6, 2], [0, 4, 7], [0, 7, 3],
            [1, 2, 6], [1, 6, 5],
        ];
        crate::scan::build_query(&p, if with_back { &t } else { &t[2..] })
    }

    #[test]
    fn mouth_query_opening_uses_two_lateral_exits() {
        let body = mouth_box(false);
        assert!(super::mouth_inside_outer_skin(&body, [0.0, 0.0, 0.0], 1.0, 0.002));
        assert!(!super::mouth_inside_outer_skin(&body, [0.08, 0.0, 0.0], 1.0, 0.002));
    }

    #[test]
    fn mouth_rear_escape_over_50mm_ends_inside_with_margin() {
        let body = mouth_box(true);
        let p = super::mouth_first_inside(&body, [0.0, 0.0, -0.18], [0.0, 0.0, 0.09], 1.0, 0.002).unwrap();
        assert!(p[2] > -0.098 && p[2] < -0.0979);
        assert!(super::mouth_inside_outer_skin(&body, p, 1.0, 0.002));
        assert!(body.closest_point(p, 1.0, None).unwrap().distance_m >= 0.002);
    }

    #[test]
    fn mouth_open_front_keeps_valid_teeth_and_neck_side_has_fallback() {
        let body = mouth_box(true);
        let tooth = [0.0, 0.0, 0.08];
        assert_eq!(super::mouth_first_inside(&body, tooth, [0.0, 0.0, 0.09], 1.0, 0.002), Some(tooth));
        let side = [0.06, 0.0, -0.08];
        assert!(super::mouth_first_inside(&body, side, [0.06, 0.0, 0.09], 1.0, 0.002).is_none());
        let p = super::mouth_first_inside(&body, side, [0.0, 0.0, 0.0], 1.0, 0.002).unwrap();
        assert!(super::mouth_inside_outer_skin(&body, p, 1.0, 0.002));
    }

    #[test]
    fn mouth_contains_triangle_across_concave_nape() {
        // A V-shaped indentation in the nape: all tongue corners can be
        // inside while the triangle's middle crosses the skin.
        let p = [
            [-0.05, -0.05, -0.1], [0.05, -0.05, -0.1],
            [-0.05, 0.0, 0.0], [0.05, 0.0, 0.0],
            [-0.05, 0.05, -0.1], [0.05, 0.05, -0.1],
        ];
        let body = crate::scan::build_query(&p, &[[0, 3, 1], [0, 2, 3], [2, 5, 3], [2, 4, 5]]);
        let mut tongue = [[-0.01, -0.04, -0.07], [0.01, 0.04, -0.07], [0.01, -0.04, -0.07]];
        assert!(tongue.iter().all(|p| super::mouth_inside_outer_skin(&body, *p, 1.0, 0.002)));
        assert!(!super::mouth_inside_outer_skin(&body, [0.0, 0.0, -0.07], 1.0, 0.002));
        let centre = [0.0, 0.0, 0.05];
        let radius = body.closest_point(centre, 1.0, None).unwrap().distance_m - 0.002002;
        let (moved, outside, shortfall) = super::mouth_contain_surface(&body, &mut tongue, &[[0, 1, 2]], 0.1, 1.0, 0.002, Some((centre, radius)));
        assert_eq!(moved, 3);
        assert_eq!(outside, 0);
        assert_eq!(shortfall, 0);
        let middle = crate::geom::a((crate::geom::v(tongue[0]) + crate::geom::v(tongue[1])) * 0.5);
        assert!(super::mouth_inside_outer_skin(&body, middle, 1.0, 0.002));
    }

    #[test]
    fn three_point_lid_margin_is_the_parabola() {
        let upper = [(-1.0, 0.0), (0.0, 0.5), (1.0, 0.0)];
        let m = |s| Opening::margin(&upper, s).unwrap();
        assert!((m(0.0) - 0.5).abs() < 1e-12);
        assert!((m(0.5) - 0.375).abs() < 1e-12);
        let o = Opening {
            upper: upper.to_vec(),
            lower: vec![(-1.0, 0.0), (0.0, -0.5), (1.0, 0.0)],
        };
        assert!((o.depth(0.0, 0.0) - 1.0).abs() < 1e-12);
        assert!(o.depth(0.0, 0.6) < 0.0);
    }
}
