//! Body-size morphs that carry the clothes along.
//!
//! MakeHuman's body-shape targets (every shape target of the prior: build
//! weight and muscle, belly, girths, head and face; `+` and `-`) are taken
//! from the prior (`body-shape.glb`, scripts/mpfb/export-shape-targets.py),
//! carried from MakeHuman's rest to the fitted body by the fit's local deformation
//! gradient, and exported as glTF morph targets on the body. Every garment
//! (and, on the keep_scan route, the scan head parts and the neck bridge)
//! gets the identically named targets, computed by the local-frame
//! transport (`transport.rs`) with a clearance pass at weight 1, so an
//! engine drives body and clothes together by name. `mh-refit-garments`
//! bakes any combination into new GLBs, re-transporting the clothes onto
//! the combined body (not just summing the targets) and re-checking
//! poke-through.
use crate::check::{RigView, SkinnedPart, check_clip};
use crate::geom::{Weld, a, percentiles, v, vertex_normals};
use crate::prior::Prior;
use crate::scan::build_query;
use crate::transport::{self, Body, Cloth, ClothPart, Settings};
use anyhow::{Context, Result, bail, ensure};
use charloom_core::{Influence, MotionClip, Vec3};
use charloom_io::{GlbDocument, load_gltf, write_json};
use nalgebra::Matrix3;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

/// Body-shape bars: every MakeHuman shape target the prior carries
/// (`body-shape.glb`: build weight and muscle, breast, belly, waist, hips,
/// bust, chest and back muscles, buttocks, arm, thigh, calf, neck, hands,
/// feet, head, nose, mouth, eyes, cheeks), one morph per side under the
/// prior's own name (`build-weight+`, `nose-volume-`, ...). Weight 1 is the
/// target's full MakeHuman value, so a bar from -1 to 1 drives `NAME-` or
/// `NAME+` as MakeHuman's own slider does. Targets that move the skeleton
/// (height, age, gender, proportions) are not in the prior.
/// Smallest displacement worth a morph target on a mesh, metres.
const MIN_TARGET_M: f64 = 3e-4;
/// Per-vertex displacement stored as zero, metres.
const MIN_VERTEX_M: f64 = 1e-4;
/// Stored morph values per file (sparse targets count their moved
/// vertices), kept under charloom-io's loader limit (12 M).
const MORPH_VERTEX_BUDGET: usize = 11_000_000;

pub type Targets = Vec<(String, Vec<Vec3>)>;

 fn deformation_gradients(
    rest: &[Vec3],
    fitted: &[Vec3],
    triangles: &[[u32; 3]],
    weld: &Weld,
) -> Vec<Matrix3<f64>> {
    let mut acc = vec![Matrix3::zeros(); weld.count];
    let mut area = vec![0.0f64; weld.count];
    for t in triangles {
        let [p0, p1, p2] = t.map(|i| v(rest[i as usize]));
        let [q0, q1, q2] = t.map(|i| v(fitted[i as usize]));
        let (e1, e2) = (p1 - p0, p2 - p0);
        let (f1, f2) = (q1 - q0, q2 - q0);
        let (n, m) = (e1.cross(&e2), f1.cross(&f2));
        let (an, am) = (n.norm(), m.norm());
        if an < 1e-12 || am < 1e-12 || !(0.05..20.0).contains(&(am / an)) {
            continue;
        }
        let e = Matrix3::from_columns(&[e1, e2, n / an]);
        let f = Matrix3::from_columns(&[f1, f2, m / am * (am / an).sqrt()]);
        let Some(inv) = e.try_inverse() else { continue };
        let g = f * inv;
        if g.determinant() <= 0.0 {
            continue;
        }
        for &i in t {
            acc[weld.id[i as usize]] += g * an;
            area[weld.id[i as usize]] += an;
        }
    }
    let mut g: Vec<Matrix3<f64>> = acc
        .iter()
        .zip(&area)
        .map(|(m, &w)| if w > 0.0 { m / w } else { Matrix3::identity() })
        .collect();
    let adjacency = crate::geom::adjacency(triangles, weld);
    for _ in 0..2 {
        g = (0..weld.count)
            .map(|w| {
                let nb = &adjacency[w];
                let s = nb.iter().fold(g[w], |s, &j| s + g[j as usize]);
                s / (1 + nb.len()) as f64
            })
            .collect();
    }
    weld.id.iter().map(|&w| g[w]).collect()
}

/// The size morphs as displacements of the fitted MakeHuman body (rest
/// positions `fitted`, same vertices as the prior body).
pub fn body_morphs(prior: &Prior, fitted: &[Vec3]) -> (Targets, Value) {
    let body = prior.body();
    if body.morphs.is_empty() {
        return (
            vec![],
            json!({"applied": false, "reason": "prior has no shape morphs (use body-shape.glb)"}),
        );
    }
    let weld = Weld::new(&body.bind, 1e-7);
    let gradients =
        deformation_gradients(&body.prim.positions, fitted, &body.prim.triangles, &weld);
    let rest = prior.skin_positions(prior.body, &body.bind, &prior.rest_world);
    let mut out = vec![];
    let mut report = serde_json::Map::new();
    for (label, d) in &body.morphs {
        if !(label.ends_with('+') || label.ends_with('-')) {
            continue;
        }
        let bind: Vec<Vec3> = body
            .bind
            .iter()
            .zip(d.iter())
            .map(|(b, x)| [b[0] + x[0], b[1] + x[1], b[2] + x[2]])
            .collect();
        let moved = prior.skin_positions(prior.body, &bind, &prior.rest_world);
        let deltas: Vec<Vec3> = (0..fitted.len())
            .map(|i| a(gradients[i] * (v(moved[i]) - v(rest[i]))))
            .collect();
        let max = deltas.iter().map(|d| v(*d).norm()).fold(0.0, f64::max);
        if max <= MIN_TARGET_M {
            continue;
        }
        report.insert(label.clone(), json!({"max_m": max}));
        out.push((label.clone(), deltas));
    }
    let applied = !out.is_empty();
    (out, json!({"applied": applied, "morphs": report}))
}

/// One transported mesh of the export.
pub struct ClothMesh<'a> {
    pub name: &'a str,
    pub positions: &'a [Vec3],
    pub normals: &'a [Vec3],
    pub triangles: &'a [[u32; 3]],
    pub hair: bool,
    pub rigid: bool,
}

/// Morph targets of the export, per mesh name; the unpruned per-target
/// displacements of every cloth mesh for the checks.
pub struct SizeMorphs {
    pub names: Vec<String>,
    /// Fitted MakeHuman body (the check body).
    pub body: Targets,
    /// Every cloth mesh's displacement per target (same order as `names`).
    pub cloth: Vec<Vec<Vec<Vec3>>>,
         pub export: HashMap<String, Targets>,
    pub report: Value,
}

/// Morphs for the skin, the body and every cloth mesh. `skin` is the exported
/// skin (the MakeHuman body, or its cut variant: positions, triangles, hand
/// triangles from [`hand_triangles`]) and the surface the cloth is bound to.
pub fn compute(
    prior: &Prior,
    fitted: &[Vec3],
    skin: (&[Vec3], &[[u32; 3]], &[bool]),
    meshes: &[ClothMesh],
    height: f64,
    clearance: f64,
) -> Option<SizeMorphs> {
    let (body, mut report) = body_morphs(prior, fitted);
    if body.is_empty() {
        return None;
    }
    let names: Vec<String> = body.iter().map(|b| b.0.clone()).collect();
    // The exported skin's own displacements.
    let same = skin.0.len() == fitted.len() && skin.0.iter().zip(fitted).all(|(x, y)| x == y);
    let skin_deltas: Vec<Vec<Vec3>> = if same {
        body.iter().map(|b| b.1.clone()).collect()
    } else {
        let fields: Vec<Vec<Vec3>> = body.iter().map(|b| b.1.clone()).collect();
        transport::interpolate_on_body(
            &Body {
                positions: fitted,
                triangles: &prior.body().prim.triangles,
                hands: None,
            },
            skin.0,
            &fields,
        )
    };
    let (cloth, displacements, transport_report) =
        carry(skin, &skin_deltas, meshes, height, clearance);
    report["transport"] = transport_report;
    let mut export: HashMap<String, Targets> = HashMap::new();
    export.insert(
        "body".into(),
        names
            .iter()
            .cloned()
            .zip(skin_deltas.iter().cloned())
            .collect(),
    );
    for (mi, m) in meshes.iter().enumerate() {
        export.insert(
            m.name.to_string(),
            names
                .iter()
                .cloned()
                .zip(displacements.iter().map(|d| d[mi].clone()))
                .collect(),
        );
    }
    let pruning = prune(&mut export);
    report["export"] = pruning;
    let cloth_by_mesh: Vec<Vec<Vec<Vec3>>> = (0..meshes.len())
        .map(|mi| displacements.iter().map(|d| d[mi].clone()).collect())
        .collect();
    drop(cloth);
    Some(SizeMorphs {
        names,
        body,
        cloth: cloth_by_mesh,
        export,
        report,
    })
}

/// Triangles whose vertices all hang mostly on wrist / finger bones.
pub fn hand_triangles(
    triangles: &[[u32; 3]],
    influences: &[Vec<Influence>],
    is_hand: impl Fn(usize) -> bool,
) -> Vec<bool> {
    let on: Vec<bool> = influences
        .iter()
        .map(|r| {
            r.iter()
                .filter(|w| is_hand(w.joint))
                .map(|w| w.weight)
                .sum::<f64>()
                > 0.5
        })
        .collect();
    triangles
        .iter()
        .map(|t| {
            t.iter()
                .all(|&i| on.get(i as usize).copied().unwrap_or(false))
        })
        .collect()
}

/// Hand bone by MakeHuman name (wrist, fingers, metacarpals).
pub fn is_hand_name(name: &str) -> bool {
    let n = name.to_lowercase();
    n.starts_with("wrist") || n.starts_with("finger") || n.starts_with("metacarpal")
}

/// Binds `meshes` to the skin and transports them onto `skin + delta` for
/// every delta field. Returns `displacements[target][mesh][vertex]`.
fn carry(
    skin: (&[Vec3], &[[u32; 3]], &[bool]),
    deltas: &[Vec<Vec3>],
    meshes: &[ClothMesh],
    height: f64,
    clearance: f64,
) -> (Cloth, Vec<Vec<Vec<Vec3>>>, Value) {
    let parts: Vec<ClothPart> = meshes
        .iter()
        .map(|m| ClothPart {
            positions: m.positions,
            normals: m.normals,
            triangles: m.triangles,
            hair: m.hair,
            rigid: m.rigid,
        })
        .collect();
    let cloth = Cloth::new(&parts);
    let body = Body {
        positions: skin.0,
        triangles: skin.1,
        hands: Some(skin.2),
    };
    let (binding, bind_report) = transport::bind(&body, &cloth, height);
    let targets: Vec<Vec<Vec3>> = deltas
        .iter()
        .map(|d| {
            skin.0
                .iter()
                .zip(d)
                .map(|(p, x)| [p[0] + x[0], p[1] + x[1], p[2] + x[2]])
                .collect()
        })
        .collect();
    let (welded, reports) = transport::transport(
        &binding,
        &body,
        &cloth,
        &targets,
        &Settings { height, clearance },
    );
    let per_target: Vec<Value> = welded
        .iter()
        .zip(&reports)
        .map(|(d, r)| {
            json!({"clearance": r, "roughness": transport::roughness(&cloth, d), "max_m": d.iter().map(|x| x.norm()).fold(0.0, f64::max)})
        })
        .collect();
    let displacements = welded
        .iter()
        .map(|d| {
            let a3: Vec<Vec3> = d.iter().map(|x| a(*x)).collect();
            cloth.split(&a3)
        })
        .collect();
    (
        cloth,
        displacements,
        json!({"binding": bind_report, "targets": per_target}),
    )
}

/// Drops targets that barely move a mesh, then, if the files would exceed
/// the loader's morph budget, the targets that move their mesh least.
fn prune(export: &mut HashMap<String, Targets>) -> Value {
    let mut dropped = vec![];
    // Below a tenth of a millimetre a vertex does not visibly move: zeroed,
    // it is not stored (targets are written sparse, export.rs).
    for targets in export.values_mut() {
        for (_, d) in targets.iter_mut() {
            for x in d.iter_mut() {
                if v(*x).norm() < MIN_VERTEX_M {
                    *x = [0.0; 3];
                }
            }
        }
    }
    for (mesh, targets) in export.iter_mut() {
        targets.retain(|(name, d)| {
            let keep = d.iter().any(|x| v(*x).norm() > MIN_TARGET_M);
            if !keep {
                dropped.push(json!({"mesh": mesh, "target": name, "reason": "below_0.3mm"}));
            }
            keep
        });
    }
    // Values the file stores: a target moving under half its mesh is sparse.
    let stored = |d: &Vec<Vec3>| {
        let moved = d.iter().filter(|x| **x != [0.0; 3]).count();
        if moved * 2 < d.len() { moved } else { d.len() }
    };
    let total = |e: &HashMap<String, Targets>| -> usize {
        e.values().flat_map(|t| t.iter().map(|x| stored(&x.1))).sum()
    };
    // Over budget: drop the (mesh, target) pairs that move their mesh least
    // first; the body keeps all of its targets.
    let mut impact: Vec<(f64, String, String)> = export
        .iter()
        .filter(|(mesh, _)| mesh.as_str() != "body")
        .flat_map(|(mesh, t)| {
            t.iter().map(move |(name, d)| {
                let max = d.iter().map(|x| v(*x).norm()).fold(0.0, f64::max);
                (max, mesh.clone(), name.clone())
            })
        })
        .collect();
    impact.sort_by(|x, y| x.0.total_cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    for (max, mesh, name) in impact {
        if total(export) <= MORPH_VERTEX_BUDGET {
            break;
        }
        export.get_mut(&mesh).unwrap().retain(|t| t.0 != name);
        dropped.push(
            json!({"mesh": mesh, "target": name, "reason": "morph_vertex_budget", "max_m": max}),
        );
    }
    let kept: BTreeMap<&String, Vec<&String>> = export
        .iter()
        .map(|(k, t)| (k, t.iter().map(|x| &x.0).collect()))
        .collect();
    json!({"morph_vertices": total(export), "budget": MORPH_VERTEX_BUDGET, "targets_per_mesh": kept, "dropped": dropped})
}

/// Poke-through and stretch of the clothed body at every size target (and
/// a combined "fatter" one), at rest and over `clip`, with the linear sum of
/// morph targets an engine would play.
/// A skinned garment for the checks: name, rest positions, triangles, weights.
pub type CheckGarment<'a> = (&'a str, &'a [Vec3], &'a [[u32; 3]], &'a [Vec<Influence>]);

pub struct CheckInput<'a> {
    pub rig: &'a RigView<'a>,
    pub clip: Option<&'a MotionClip>,
    pub body_positions: &'a [Vec3],
    pub body_triangles: &'a [[u32; 3]],
    pub body_influences: &'a [Vec<Influence>],
    pub garments: Vec<CheckGarment<'a>>,
    pub frames: usize,
    pub height: f64,
}

pub fn check(input: &CheckInput, morphs: &SizeMorphs, garment_index: &[usize]) -> Result<Value> {
    let Some(clip) = input.clip else {
        return Ok(json!(null));
    };
    let mut cases: Vec<(String, Vec<(usize, f64)>)> = vec![("rest_shape".into(), vec![])];
    // Over the clip: the whole-body bars (MakeHuman's build macros) and the
    // belly, the ones that push the body into its clothes; every bar is
    // still transported with the clearance pass at weight 1.
    let find = |n: &str| morphs.names.iter().position(|x| x == n);
    for n in ["build-weight+", "build-weight-", "build-muscle+", "stomach-pregnant+"] {
        if let Some(i) = find(n) {
            cases.push((n.to_string(), vec![(i, 1.0)]));
        }
    }
    if let (Some(w), Some(b)) = (find("build-weight+"), find("stomach-pregnant+")) {
        cases.push(("fatter(build-weight+ stomach-pregnant+)".into(), vec![(w, 1.0), (b, 1.0)]));
    }
    let apply = |base: &[Vec3], fields: &[&Vec<Vec3>], weights: &[(usize, f64)]| -> Vec<Vec3> {
        base.iter()
            .enumerate()
            .map(|(i, p)| {
                let mut q = *p;
                for (&(_, w), f) in weights.iter().zip(fields) {
                    for c in 0..3 {
                        q[c] += f[i][c] * w;
                    }
                }
                q
            })
            .collect()
    };
    let mut rows = vec![];
    for (label, weights) in &cases {
        let body_fields: Vec<&Vec<Vec3>> =
            weights.iter().map(|(i, _)| &morphs.body[*i].1).collect();
        let body = apply(input.body_positions, &body_fields, weights);
        let garments: Vec<Vec<Vec3>> = input
            .garments
            .iter()
            .zip(garment_index)
            .map(|(g, &mi)| {
                let fields: Vec<&Vec<Vec3>> =
                    weights.iter().map(|(i, _)| &morphs.cloth[mi][*i]).collect();
                apply(g.1, &fields, weights)
            })
            .collect();
        let parts: Vec<SkinnedPart> = input
            .garments
            .iter()
            .zip(&garments)
            .map(|(g, p)| SkinnedPart {
                name: g.0,
                positions: p,
                triangles: g.2,
                influences: g.3,
                seam: None,
            })
            .collect();
        let c = check_clip(
            input.rig,
            clip,
            &SkinnedPart {
                name: "body",
                positions: &body,
                triangles: input.body_triangles,
                influences: input.body_influences,
                seam: None,
            },
            &parts,
            input.frames,
            input.height,
            None,
        )?;
        let worst_garments: Vec<Value> = c["worst_frame"]["garments"]
            .as_array()
            .map(|g| {
                g.iter()
                    .map(|x| json!({"garment": x["garment"], "inside_fraction": x["inside_fraction"], "edge_ratio_max": x["edge_ratio"]["max"]}))
                    .collect()
            })
            .unwrap_or_default();
        rows.push(json!({
            "case": label,
            "rest_inside_fraction": c["rest"]["inside_fraction"],
            "walk_inside_fraction": c["inside_fraction_over_frames"],
            "worst_frame_garments": worst_garments,
        }));
    }
    Ok(
        json!({"clip": clip.name, "frames": input.frames, "note": "garment vertices more than 2 mm inside the body (body poking through), linear morph-target sum as an engine plays it", "cases": rows}),
    )
}

// ---------------------------------------------------------------------------
// Offline refit: `charloom mh-refit-garments RUN_DIR --morph NAME=VALUE`.

/// Extras key marking exported meshes for the refit.
pub const ROLE_KEY: &str = "charloom_size";

fn read_vec3(doc: &GlbDocument, accessor: usize) -> Result<Vec<Vec3>> {
    let acc = &doc.json["accessors"][accessor];
    ensure!(
        acc["componentType"] == json!(5126) && acc["type"] == json!("VEC3"),
        "accessor {accessor} is not float VEC3"
    );
    ensure!(acc.get("sparse").is_none(), "sparse base accessor");
    let count = acc["count"].as_u64().context("accessor count")? as usize;
    let view = &doc.json["bufferViews"][acc["bufferView"].as_u64().context("bufferView")? as usize];
    let offset = view["byteOffset"].as_u64().unwrap_or(0) as usize
        + acc["byteOffset"].as_u64().unwrap_or(0) as usize;
    let stride = view["byteStride"].as_u64().unwrap_or(12) as usize;
    (0..count)
        .map(|i| {
            let o = offset + i * stride;
            let bytes = doc
                .binary
                .get(o..o + 12)
                .context("accessor out of bounds")?;
            Ok(std::array::from_fn(|c| {
                f32::from_le_bytes(bytes[c * 4..c * 4 + 4].try_into().unwrap()) as f64
            }))
        })
        .collect()
}

struct FileMesh {
    name: String,
    json_mesh: usize,
    role: String,
    hair: bool,
    rigid: bool,
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    targets: Targets,
}

fn file_meshes(path: &Path) -> Result<(charloom_io::LoadedAsset, Vec<FileMesh>)> {
    let loaded = load_gltf(path).with_context(|| format!("load {}", path.display()))?;
    let mut out = vec![];
    for m in &loaded.asset.meshes {
        let Some(json_mesh) = loaded.document.json["nodes"][m.node]["mesh"].as_u64() else {
            continue;
        };
        let jm = &loaded.document.json["meshes"][json_mesh as usize];
        let role = &jm["extras"][ROLE_KEY];
        let Some(kind) = role["role"].as_str() else {
            continue;
        };
        let normals = read_vec3(
            &loaded.document,
            jm["primitives"][0]["attributes"]["NORMAL"]
                .as_u64()
                .context("NORMAL")? as usize,
        )?;
        let names: Vec<String> = jm["extras"]["targetNames"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        ensure!(
            names.len() == m.morph_targets.len(),
            "{}: targetNames do not match its morph targets",
            m.name
        );
        out.push(FileMesh {
            name: m.name.clone(),
            json_mesh: json_mesh as usize,
            role: kind.to_string(),
            hair: role["hair"].as_bool().unwrap_or(false),
            rigid: role["rigid"].as_bool().unwrap_or(false),
            positions: m.positions.clone(),
            normals,
            triangles: m.triangles.clone(),
            targets: names
                .into_iter()
                .zip(m.morph_targets.iter().cloned())
                .collect(),
        });
    }
    Ok((loaded, out))
}

 pub fn parse_morphs(args: &[String]) -> Result<Vec<(String, f64)>> {
    args.iter()
        .map(|s| {
            let (name, value) = s
                .split_once('=')
                .with_context(|| format!("--morph {s}: expected NAME=VALUE"))?;
            let value: f64 = value
                .trim()
                .parse()
                .with_context(|| format!("--morph {s}: not a number"))?;
            ensure!(value.is_finite(), "--morph {s}: not finite");
            let name = name.trim();
            Ok(if name.ends_with('+') || name.ends_with('-') {
                (name.to_string(), value)
            } else if value >= 0.0 {
                (format!("{name}+"), value)
            } else {
                (format!("{name}-"), -value)
            })
        })
        .collect()
}

fn geometric_normals(positions: &[Vec3], triangles: &[[u32; 3]]) -> Vec<Vec3> {
    let weld = Weld::new(positions, 1e-7);
    vertex_normals(positions, triangles, &weld)
}

/// Original normals turned by the change of the geometric normal.
fn moved_normals(
    old: &[Vec3],
    before: &[Vec3],
    after: &[Vec3],
    triangles: &[[u32; 3]],
) -> Vec<Vec3> {
    let g0 = geometric_normals(before, triangles);
    let g1 = geometric_normals(after, triangles);
    (0..old.len())
        .map(|i| {
            let n = v(old[i]) + v(g1[i]) - v(g0[i]);
            a(n.try_normalize(1e-12).unwrap_or_else(|| v(old[i])))
        })
        .collect()
}

/// Fraction of cloth vertices more than 2 mm inside the body, per frame of
/// the walk (rest first), measured on the files themselves. Inside is
/// judged by the interpolated vertex normal at the closest point within
/// `transport::CONTACT` of the height (the stage-5 check's face normal and
/// 5 % reach count cloth several centimetres from a convex corner or a
/// concave crease as inside).
fn file_poke(body: &Path, character: &Path, frames: usize) -> Result<Value> {
    let b = load_gltf(body)?;
    let c = load_gltf(character)?;
    let role = |l: &charloom_io::LoadedAsset, m: &charloom_core::Mesh| -> Option<String> {
        let mesh = l.document.json["nodes"][m.node]["mesh"].as_u64()?;
        l.document.json["meshes"][mesh as usize]["extras"][ROLE_KEY]["role"]
            .as_str()
            .map(String::from)
    };
    let body_index = b
        .asset
        .meshes
        .iter()
        .position(|m| role(&b, m).as_deref() == Some("body"))
        .context("body.glb has no body mesh")?;
    let cloth: Vec<usize> = (0..c.asset.meshes.len())
        .filter(|&i| role(&c, &c.asset.meshes[i]).as_deref() == Some("cloth"))
        .collect();
    let height = {
        let ys = b.asset.meshes[body_index].positions.iter().map(|p| p[1]);
        let (lo, hi) = ys.fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), y| {
            (l.min(y), h.max(y))
        });
        hi - lo
    };
    let walk = |l: &charloom_io::LoadedAsset| -> Option<MotionClip> {
        l.asset
            .clips
            .iter()
            .find(|x| x.name == "walking")
            .or(l.asset.clips.first())
            .cloned()
    };
    let (clip, body_clip) = (walk(&c), walk(&b));
    let clip = clip.as_ref();
    let mut times: Vec<Option<f64>> = vec![None];
    if let Some(clip) = clip {
        let [t0, t1] = clip.time_range();
        for f in 0..frames {
            times.push(Some(
                t0 + (t1 - t0) * f as f64 / frames.max(2).saturating_sub(1) as f64,
            ));
        }
    }
    let mut rows = vec![];
    for t in times {
        // Each file has its own node list: sample the clip per file.
        let pose = match (t, clip) {
            (Some(t), Some(clip)) => clip.sample(&c.asset.skeleton, t)?,
            _ => c.asset.skeleton.rest_pose(),
        };
        let body_pose = match (t, &body_clip) {
            (Some(t), Some(clip)) => clip.sample(&b.asset.skeleton, t)?,
            _ => b.asset.skeleton.rest_pose(),
        };
        let bp = &b.asset.skinned_positions(&body_pose)?[body_index];
        let body_triangles = &b.asset.meshes[body_index].triangles;
        let query = build_query(bp, body_triangles);
        let normals = vertex_normals(bp, body_triangles, &Weld::new(bp, 1e-7));
        let cp = c.asset.skinned_positions(&pose)?;
        let mut per = vec![];
        let (mut inside, mut total) = (0usize, 0usize);
        for &i in &cloth {
            let pts = &cp[i];
            let stride = (pts.len() / 20_000).max(1);
            let n = pts
                .par_iter()
                .step_by(stride)
                .filter(|p| {
                    query
                        .closest_point(**p, height * transport::CONTACT, None)
                        .is_some_and(|h| {
                            let t = body_triangles[h.source[1]];
                            let n = (0..3).fold(nalgebra::Vector3::zeros(), |s, j| {
                                s + v(normals[t[j] as usize]) * h.barycentric[j]
                            });
                            (v(**p) - v(h.point)).dot(&n) < -0.002
                        })
                })
                .count();
            let s = pts.len().div_ceil(stride);
            inside += n;
            total += s;
            per.push(json!({"mesh": c.asset.meshes[i].name, "inside_fraction": n as f64 / s.max(1) as f64}));
        }
        rows.push(json!({"time": t, "inside_fraction": inside as f64 / total.max(1) as f64, "meshes": per}));
    }
    let walk: Vec<f64> = rows
        .iter()
        .skip(1)
        .filter_map(|r| r["inside_fraction"].as_f64())
        .collect();
    Ok(
        json!({"rest_inside_fraction": rows[0]["inside_fraction"], "walk_inside_fraction": percentiles(walk), "rest_meshes": rows[0]["meshes"]}),
    )
}

/// Bakes `morphs` into the body and re-transports the clothes onto it,
/// writing `out/3-skin/body.glb` and `out/4-garments/character.glb` (the
/// layout `charloom mh-render` reads). Default `out`:
/// `RUN/refit/<morphs>`.
pub fn refit(
    run: &Path,
    morphs: &[(String, f64)],
    out: Option<&Path>,
    frames: usize,
) -> Result<Value> {
    let body_path = run.join("3-skin/body.glb");
    let char_path = run.join("4-garments/character.glb");
    let report_path = run.join("report.json");
    let label = if morphs.is_empty() {
        "rest".to_string()
    } else {
        morphs
            .iter()
            .map(|(n, w)| format!("{n}{w}"))
            .collect::<Vec<_>>()
            .join("_")
    };
    let out: PathBuf = out.map_or_else(|| run.join("refit").join(&label), Path::to_path_buf);
    let (body_doc, body_meshes) = file_meshes(&body_path)?;
    let (char_doc, char_meshes) = file_meshes(&char_path)?;
    let skin = body_meshes
        .iter()
        .find(|m| m.role == "body")
        .context("body.glb has no size-morph body (re-run mh-transfer with size_morphs)")?;
    let known: Vec<&String> = skin.targets.iter().map(|t| &t.0).collect();
    for (n, _) in morphs {
        if !known.contains(&n) {
            bail!("unknown morph {n}; the body has {known:?}");
        }
    }
    let height = {
        let ys = skin.positions.iter().map(|p| p[1]);
        let (lo, hi) = ys.fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), y| {
            (l.min(y), h.max(y))
        });
        hi - lo
    };
    let clearance: f64 = std::fs::read(&report_path)
        .ok()
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|r| r["stage4_size_morphs"]["clearance_m"].as_f64())
        .unwrap_or(0.004);
    let mut delta = vec![[0.0; 3]; skin.positions.len()];
    for (n, w) in morphs {
        let d = &skin.targets.iter().find(|t| &t.0 == n).unwrap().1;
        for (x, y) in delta.iter_mut().zip(d) {
            for c in 0..3 {
                x[c] += y[c] * w;
            }
        }
    }
    let cloth_meshes: Vec<&FileMesh> = char_meshes.iter().filter(|m| m.role == "cloth").collect();
    let clothes: Vec<ClothMesh> = cloth_meshes
        .iter()
        .map(|m| ClothMesh {
            name: &m.name,
            positions: &m.positions,
            normals: &m.normals,
            triangles: &m.triangles,
            hair: m.hair,
            rigid: m.rigid,
        })
        .collect();
    let skin_influences = &body_doc
        .asset
        .meshes
        .iter()
        .find(|m| m.name == skin.name)
        .context("body mesh")?
        .influences;
    let joints = &body_doc.asset.skeleton.joints;
    let hands = hand_triangles(&skin.triangles, skin_influences, |j| {
        joints.get(j).is_some_and(|x| is_hand_name(&x.name))
    });
    let (_, displacements, transport_report) = carry(
        (&skin.positions, &skin.triangles, &hands),
        std::slice::from_ref(&delta),
        &clothes,
        height,
        clearance,
    );
    let mut moved: HashMap<String, Vec<Vec3>> = HashMap::new();
    let add = |p: &[Vec3], d: &[Vec3]| -> Vec<Vec3> {
        p.iter()
            .zip(d)
            .map(|(x, y)| [x[0] + y[0], x[1] + y[1], x[2] + y[2]])
            .collect()
    };
    moved.insert("body".into(), add(&skin.positions, &delta));
    for (m, d) in cloth_meshes.iter().zip(&displacements[0]) {
        moved.insert(m.name.clone(), add(&m.positions, d));
    }
    // Underwear on the naked body: its own targets are the MHCLO binding
    // re-evaluated on each morphed body (underwear.rs), so their sum is
    // the garment on the combined body.
    for m in body_meshes.iter().filter(|m| m.role == "underwear") {
        let mut p = m.positions.clone();
        for (n, w) in morphs {
            if let Some((_, d)) = m.targets.iter().find(|t| &t.0 == n) {
                for (x, y) in p.iter_mut().zip(d) {
                    for c in 0..3 {
                        x[c] += y[c] * w;
                    }
                }
            }
        }
        moved.insert(m.name.clone(), p);
    }
    let write =
        |loaded: &charloom_io::LoadedAsset, meshes: &[FileMesh], path: &Path| -> Result<()> {
            let mut doc = loaded.document.clone();
            for m in meshes {
                let key = if m.role == "body" {
                    "body"
                } else {
                    m.name.as_str()
                };
                let Some(p) = moved.get(key) else { continue };
                ensure!(
                    p.len() == m.positions.len(),
                    "{}: vertex count differs between files",
                    m.name
                );
                // The clothed body has covered triangles removed: normals from
                // the naked body's full surface.
                let tris = if m.role == "body" {
                    &skin.triangles
                } else {
                    &m.triangles
                };
                let normals = moved_normals(&m.normals, &m.positions, p, tris);
                let pa = doc.append_vec3(p, Some(34962), true);
                let na = doc.append_vec3(&normals, Some(34962), false);
                for prim in doc.json["meshes"][m.json_mesh]["primitives"]
                    .as_array_mut()
                    .context("primitives")?
                {
                    prim["attributes"]["POSITION"] = json!(pa);
                    prim["attributes"]["NORMAL"] = json!(na);
                }
            }
            doc.json["asset"]["extras"]["charloom_refit"] =
                json!({"source": run, "morphs": morphs});
            doc.save(path)
        };
    std::fs::create_dir_all(out.join("3-skin"))?;
    std::fs::create_dir_all(out.join("4-garments"))?;
    let new_body = out.join("3-skin/body.glb");
    let new_char = out.join("4-garments/character.glb");
    write(&body_doc, &body_meshes, &new_body)?;
    write(&char_doc, &char_meshes, &new_char)?;
    let before = file_poke(&body_path, &char_path, frames)?;
    let after = file_poke(&new_body, &new_char, frames)?;
    let report = json!({
        "operation": "mh_refit_garments",
        "run": run,
        "morphs": morphs,
        "clearance_m": clearance,
        "transport": transport_report,
        "body_max_displacement_m": delta.iter().map(|d| v(*d).norm()).fold(0.0, f64::max),
        "poke_through": {"before": before, "after": after},
        "outputs": {"body": new_body, "character": new_char},
    });
    write_json(&out.join("refit.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::V;

    #[test]
    fn morph_arguments_map_to_signed_targets() {
        let m =
            parse_morphs(&["weight=0.5".into(), "belly=-1".into(), "chest+=0.25".into()]).unwrap();
        assert_eq!(
            m,
            vec![
                ("weight+".to_string(), 0.5),
                ("belly-".to_string(), 1.0),
                ("chest+".to_string(), 0.25)
            ]
        );
        assert!(parse_morphs(&["weight".into()]).is_err());
    }

    /// A body moved by a known similarity (rotation + uniform scale): the
    /// deformation gradient carries a delta as the map does.
    #[test]
    fn deltas_follow_the_fit_deformation() {
        let rest: Vec<Vec3> = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let tris = vec![[0u32, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        let r = nalgebra::Rotation3::from_euler_angles(0.3, -0.2, 0.5).into_inner();
        let m = r * Matrix3::from_diagonal(&V::new(1.5, 1.5, 1.5));
        let fitted: Vec<Vec3> = rest.iter().map(|p| a(m * v(*p))).collect();
        let weld = Weld::new(&rest, 1e-9);
        let g = deformation_gradients(&rest, &fitted, &tris, &weld);
        let d = V::new(0.01, -0.02, 0.03);
        for gi in &g {
            assert!((gi * d - m * d).norm() < 1e-9, "{gi}");
        }
    }
}
