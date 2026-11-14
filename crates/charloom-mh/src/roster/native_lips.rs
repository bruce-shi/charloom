use super::rim_curve::RimCurve;
use super::{a, file_sha256, mouth_trace, read, v};
use anyhow::{Context, Result, ensure};
use charloom_core::Vec3;
use charloom_io::{GlbDocument, load_gltf, write_json};
use nalgebra::Vector3;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

type Rims<T> = BTreeMap<String, Vec<T>>;

fn rim_indices(uv: &[[f32; 2]]) -> Result<Rims<usize>> {
    // Each template rim position may have several UV copies at a seam.
    let reference: Rims<Vec<[f32; 2]>> = serde_json::from_str(include_str!("lip-rim-uv.json"))?;
    reference
        .into_iter()
        .map(|(side, rim)| {
            let indices = rim
                .iter()
                .map(|copies| {
                    let mut best = (f64::INFINITY, 0);
                    for &target in copies {
                        for (i, point) in uv.iter().enumerate() {
                            let d = (f64::from(point[0]) - f64::from(target[0])).powi(2)
                                + (f64::from(point[1]) - f64::from(target[1])).powi(2);
                            if d < best.0 {
                                best = (d, i);
                            }
                        }
                        if best.0 == 0. {
                            break;
                        }
                    }
                    ensure!(best.0 < 1e-12, "missing template {side} lip UV");
                    Ok(best.1)
                })
                .collect::<Result<_>>()?;
            Ok((side, indices))
        })
        .collect()
}

struct Fit {
    samples: Vec<(Vector3<f64>, Vector3<f64>)>,
    radius: f64,
    rims: Rims<Vec3>,
}

impl Fit {
    fn new(trace: &[Vec3], body: &[Vec3], ids: &Rims<usize>) -> Result<Self> {
        let curve = RimCurve::new(trace)?;
        let x = &curve.x;
        let mut samples = Vec::new();
        let mut rims = Rims::new();
        let width = x[x.len() - 1] - x[0];
        for (name, sign) in [("upper", 1.), ("lower", -1.)] {
            let indices = &ids[name];
            let start = body[indices[0]][0];
            let span = body[*indices.last().context("empty lip rim")?][0] - start;
            ensure!(span.abs() > 1e-9, "degenerate template mouth");
            let mut rim = Vec::new();
            for &i in indices {
                let u = ((body[i][0] - start) / span).clamp(0., 1.);
                let xx = x[0] + width * u;
                let mut target = curve.at(xx);
                target[1] += sign * 0.00025 * (std::f64::consts::PI * u).sin();
                rim.push(target);
                samples.push((v(body[i]), v(target) - v(body[i])));
            }
            rims.insert(name.to_owned(), rim);
        }
        Ok(Self {
            samples,
            rims,
            radius: (width * 0.55).max(0.022),
        })
    }

    fn transport(&self, original: &[Vec3]) -> Vec<Vec3> {
        original
            .par_iter()
            .map(|&p| {
                let pos = v(p);
                let mut nearest = [(f64::INFINITY, 0usize); 5];
                for (i, (point, _)) in self.samples.iter().enumerate() {
                    let d = (point - pos).norm_squared();
                    if d >= nearest[4].0 {
                        continue;
                    }
                    let slot = nearest.partition_point(|(prev, _)| *prev <= d);
                    nearest.copy_within(slot..4, slot + 1);
                    nearest[slot] = (d, i);
                }
                let t = (1. - nearest[0].0.sqrt() / self.radius).clamp(0., 1.);
                if t == 0. {
                    return p;
                }
                let mut sum = Vector3::zeros();
                let mut total = 0.;
                for (d, i) in nearest {
                    let w = 1. / d.max(1e-8);
                    total += w;
                    sum += self.samples[i].1 * w;
                }
                a(pos + sum * (t * t * (3. - 2. * t) / total))
            })
            .collect()
    }
}

fn assemble(original: &Path, corrected: &Path, out: &Path, trace: &[Vec3]) -> Result<Rims<Vec3>> {
    let loaded = load_gltf(original)?;
    let corrected = load_gltf(corrected)?;
    let base = crate::source::gltf_of(&loaded)?;
    let fit = crate::source::gltf_of(&corrected)?;
    let mesh = |doc: &gltf::Gltf| {
        doc.meshes()
            .position(|m| m.name() == Some("body"))
            .context("body mesh missing")
    };
    let body = mesh(&base)?;
    let p = base
        .meshes()
        .nth(body)
        .unwrap()
        .primitives()
        .next()
        .context("empty body")?;
    let cp = fit
        .meshes()
        .nth(mesh(&fit)?)
        .unwrap()
        .primitives()
        .next()
        .context("empty fitted body")?;
    let reader = p.reader(|_| Some(&loaded.document.binary));
    let cr = cp.reader(|_| Some(&corrected.document.binary));
    let coords = |input: Option<gltf::mesh::util::ReadTexCoords>| -> Result<Vec<[f32; 2]>> {
        Ok(input.context("missing body UVs")?.into_f32().collect())
    };
    let uv = coords(reader.read_tex_coords(0))?;
    ensure!(
        uv == coords(cr.read_tex_coords(0))?,
        "body UV correspondence differs"
    );
    let positions: Vec<Vec3> = cr
        .read_positions()
        .context("missing fitted positions")?
        .map(|p| p.map(f64::from))
        .collect();
    let mut normals: Vec<Vec3> = cr
        .read_normals()
        .context("missing fitted normals")?
        .map(|p| p.map(f64::from))
        .collect();
    ensure!(
        positions.len() == uv.len() && normals.len() == uv.len(),
        "body attribute count mismatch"
    );
    let indices: Vec<_> = reader
        .read_indices()
        .context("missing body indices")?
        .into_u32()
        .collect();
    ensure!(
        cr.read_indices()
            .context("missing fitted indices")?
            .into_u32()
            .eq(indices.iter().copied()),
        "body topology differs"
    );
    let ids = rim_indices(&uv)?;
    let fit = Fit::new(trace, &positions, &ids)?;
    let mut result = fit.transport(&positions);
    for side in ["upper", "lower"] {
        for (&id, &target) in ids[side].iter().zip(&fit.rims[side]) {
            for (i, value) in positions.iter().enumerate() {
                if *value == positions[id] {
                    result[i] = target;
                }
            }
        }
    }
    // Round to the exported f32 positions before recomputing area-weighted normals.
    for p in &mut result {
        *p = p.map(|n| f64::from(n as f32));
    }
    let mut accumulated = vec![Vector3::zeros(); result.len()];
    for tri in indices.chunks_exact(3) {
        let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| i as usize);
        ensure!(
            [a, b, c].into_iter().all(|i| i < result.len()),
            "body index out of range"
        );
        let normal = (v(result[b]) - v(result[a])).cross(&(v(result[c]) - v(result[a])));
        for i in [a, b, c] {
            accumulated[i] += normal;
        }
    }
    for (i, n) in accumulated.into_iter().enumerate() {
        if (v(result[i]) - v(positions[i])).norm() > 1e-6
            && let Some(unit) = n.try_normalize(1e-20)
        {
            normals[i] = a(unit);
        }
    }
    let mut doc = loaded.document;
    replace(&mut doc, body, 0, "POSITION", &result);
    replace(&mut doc, body, 0, "NORMAL", &normals);
    for mesh in base
        .meshes()
        .filter(|m| matches!(m.name(), Some("teeth" | "tongue")))
    {
        for p in mesh.primitives() {
            let rest = p
                .reader(|_| Some(&doc.binary))
                .read_positions()
                .context("missing oral mesh positions")?
                .map(|p| p.map(f64::from))
                .collect::<Vec<_>>();
            replace(
                &mut doc,
                mesh.index(),
                p.index(),
                "POSITION",
                &fit.transport(&rest),
            );
        }
    }
    let jaw = base
        .nodes()
        .find(|n| n.name() == Some("jaw"))
        .context("missing jaw")?;
    doc.json["nodes"][jaw.index()]["extras"]["charloomLipClearanceM"] = json!(0.0005);
    doc.save(out)?;
    Ok(fit.rims)
}

fn replace(doc: &mut GlbDocument, mesh: usize, primitive: usize, semantic: &str, values: &[Vec3]) {
    let accessor = doc.append_vec3(values, Some(34962), semantic == "POSITION");
    doc.json["meshes"][mesh]["primitives"][primitive]["attributes"][semantic] = json!(accessor);
}

pub fn fit_native(run: &Path, trace: &Path, out: &Path) -> Result<Value> {
    let points = mouth_trace(run, trace)?;
    let choice: Value = read(&run.join("body-selection.json"))?;
    let fitted = PathBuf::from(
        choice["candidate"]
            .as_str()
            .context("missing body fit candidate")?,
    );
    let mut exports = Vec::new();
    // Publish the directory as a unit so a failed dressed fit cannot replace the bare stage.
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    ensure!(
        !out.exists(),
        "output directory already exists: {}",
        out.display()
    );
    let staging = tempfile::tempdir_in(parent)?;
    for rel in ["3-skin/body.glb", "4-garments/character.glb"] {
        let destination = staging.path().join(rel);
        let rims = assemble(&run.join(rel), &fitted.join(rel), &destination, &points)?;
        write_json(&staging.path().join("mouth-curves.json"), &rims)?;
        exports.push(json!({"file":rel,"sha256":file_sha256(&destination)?}));
    }
    let info = json!({"source_run":run,"body_fit":choice,"face":"scan-registered native template with separately constrained upper/lower lip rims","exports":exports});
    write_json(&staging.path().join("assembly.json"), &info)?;
    std::fs::rename(staging.path(), out)?;
    Ok(info)
}
