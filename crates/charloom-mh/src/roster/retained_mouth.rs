//! Rebind the cut lip rims and attached beard on a retained sculpt. Place
//! the dental arches and tongue in the measured oral pocket.
use super::face_binding::smooth;
use super::read;
use crate::geom::{V, a, v};
use anyhow::{Context, Result, ensure};
use charloom_core::Vec3;
use nalgebra::Matrix3;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

#[derive(Deserialize)]
struct Part {
    name: String,
    positions: Vec<Vec3>,
    joints: Vec<[usize; 4]>,
    weights: Vec<[f64; 4]>,
    normals: Vec<Vec3>,
}
#[derive(Deserialize)]
struct Input {
    palette: Vec<String>,
    upper: Vec<Vec3>,
    lower: Vec<Vec3>,
    parts: Vec<Part>,
}
#[derive(Serialize)]
struct Skin {
    id: usize,
    joints: [usize; 4],
    weights: [f64; 4],
}
#[derive(Default, Serialize)]
struct Edits {
    skin: Vec<Skin>,
    lining: Vec<(usize, f64)>,
    positions: Vec<(usize, Vec3, Vec3)>,
}
struct Opening<'a> {
    upper: &'a [Vec3],
    lower: &'a [Vec3],
    mid: f64,
    radius: f64,
}
impl Opening<'_> {
    fn lookup(&self, x: f64) -> (f64, f64, f64) {
        let n = self.upper.len();
        let u = ((x - self.upper[0][0]) / (2. * self.radius)).clamp(0., 1.) * (n - 1) as f64;
        let i = (u.floor() as usize).min(n - 2);
        let t = u - i as f64;
        let at = |rim: &[Vec3], axis: usize| rim[i][axis] + (rim[i + 1][axis] - rim[i][axis]) * t;
        (at(self.upper, 1), at(self.lower, 1), at(self.upper, 2))
    }
}
fn percentile(values: impl Iterator<Item = f64>, q: f64) -> f64 {
    let mut values: Vec<_> = values.collect();
    values.sort_by(f64::total_cmp);
    let x = q * (values.len() - 1) as f64;
    let i = x.floor() as usize;
    values[i] + (values[x.ceil() as usize] - values[i]) * (x - i as f64)
}
fn dental_frame(points: &[V], upper: bool, target: V) -> Result<(Matrix3<f64>, V)> {
    ensure!(points.len() > 1, "empty dental arch");
    let center = points.iter().copied().sum::<V>() / points.len() as f64;
    let variance = points.iter().map(|p| (p.z - center.z).powi(2)).sum::<f64>();
    ensure!(variance > 0., "dental arch has no depth");
    let slope = points
        .iter()
        .map(|p| (p.z - center.z) * (p.y - center.y))
        .sum::<f64>()
        / variance;
    let (s, c) = slope.atan().sin_cos();
    let m = Matrix3::new(
        0.78,
        0.,
        0.,
        0.,
        0.70 * c,
        -0.70 * s,
        0.,
        0.72 * s,
        0.72 * c,
    );
    let coords: Vec<_> = points.iter().map(|p| m * p).collect();
    let cutoff = percentile(coords.iter().map(|p| p.z), 0.85);
    let tip = V::new(
        percentile(coords.iter().map(|p| p.x), 0.5),
        percentile(
            coords.iter().filter(|p| p.z >= cutoff).map(|p| p.y),
            if upper { 0.035 } else { 0.965 },
        ),
        coords.iter().map(|p| p.z).fold(f64::NEG_INFINITY, f64::max),
    );
    Ok((m, target - tip))
}

fn bind(input: &Input) -> Result<Vec<Edits>> {
    let Input {
        palette,
        upper,
        lower,
        parts,
    } = input;
    ensure!(
        upper.len() == lower.len() && upper.len() >= 2,
        "invalid sampled lip curves"
    );
    let start = upper[0][0];
    let end = upper.last().unwrap()[0];
    ensure!(end > start, "empty mouth span");
    let mouth = Opening {
        upper,
        lower,
        mid: (start + end) / 2.,
        radius: (end - start) / 2.,
    };
    let bone = |name: &str| {
        palette
            .iter()
            .position(|j| j == name)
            .with_context(|| format!("missing joint {name}"))
    };
    let head = bone("head")?;
    let jaw = bone("jaw")?;
    let corners = [bone("levator05.L")?, bone("levator05.R")?];
    let mids = [bone("oris05")?, bone("oris01")?];
    let outer = [
        [bone("oris03.L")?, bone("oris03.R")?],
        [bone("oris07.L")?, bone("oris07.R")?],
    ];
    let (top, bottom, depth) = mouth.lookup(mouth.mid);
    let cy = (top + bottom) * 0.5;
    let mut lower_frame = None;
    let mut result = Vec::new();
    for part in parts {
        let count = part.positions.len();
        ensure!(
            part.normals.len() == count
                && part.joints.len() == count
                && part.weights.len() == count,
            "incomplete mouth attributes"
        );
        let mut edits = Edits::default();
        if matches!(&*part.name, "teeth" | "tongue") {
            for (joint, upper) in [(head, true), (jaw, false)] {
                let tongue = part.name == "tongue";
                if tongue && upper {
                    continue;
                }
                let ids: Vec<_> = (0..count)
                    .filter(|&i| tongue || part.joints[i][0] == joint)
                    .collect();
                let (m, d) = if tongue {
                    lower_frame.context("tongue requires the lower dental arch")?
                } else {
                    let target = V::new(
                        mouth.mid,
                        cy + if upper { 0.0016 } else { -0.0035 },
                        depth - if upper { 0.012 } else { 0.016 },
                    );
                    dental_frame(
                        &ids.iter()
                            .map(|&i| v(part.positions[i]))
                            .collect::<Vec<_>>(),
                        upper,
                        target,
                    )?
                };
                if !upper {
                    lower_frame = Some((m, d));
                }
                let nm = m
                    .try_inverse()
                    .context("singular dental frame")?
                    .transpose();
                for i in ids {
                    let mut p = m * v(part.positions[i]) + d;
                    let (top, bottom, front) = mouth.lookup(p.x);
                    p.y += 0.65 * ((top + bottom) * 0.5 - cy);
                    p.z += 0.82 * (front - depth);
                    let normal = nm * v(part.normals[i]);
                    edits
                        .positions
                        .push((i, a(p), a(normal / normal.norm().max(1e-12))));
                }
            }
        } else {
            ensure!(
                matches!(&*part.name, "body" | "beard"),
                "unexpected mouth mesh"
            );
            let mut weights = vec![0.; palette.len()];
            for (id, &[x, y, z]) in part.positions.iter().enumerate() {
                let (top, bottom, front) = mouth.lookup(x);
                let center = (top + bottom) / 2.;
                let u = (x - mouth.mid).abs() / mouth.radius;
                let mask = smooth((1.65 - u) / 0.5)
                    * smooth((top + 0.011 - y) / 0.007)
                    * smooth((y - bottom + 0.085) / 0.035)
                    * smooth((z - front + 0.053) / 0.012);
                if part.name == "body" && z > front - 0.045 {
                    let lining = smooth((front - z - 0.001) / 0.003)
                        * smooth((0.0015 - (y - center).abs()) / 0.001)
                        * smooth((1. - u) / 0.03);
                    if lining > 0. {
                        edits.lining.push((id, lining));
                    }
                }
                if mask == 0. {
                    continue;
                }
                weights.fill(0.);
                let top_fraction = smooth((y - bottom) / (top - bottom).max(1e-6));
                let lip = smooth(
                    1. - (y - center).abs() / if part.name == "body" { 0.018 } else { 0.012 },
                );
                let active = lip * smooth((z - front + 0.043) / 0.025);
                let lateral = smooth(u / 0.64);
                let corner = smooth((u - 0.55) / 0.45);
                let side = usize::from(x <= mouth.mid);
                weights[corners[side]] = active * corner;
                for (k, fraction) in [top_fraction, 1. - top_fraction].into_iter().enumerate() {
                    let f = active * (1. - corner) * fraction;
                    weights[mids[k]] += f * (1. - lateral);
                    weights[outer[k][side]] += f * lateral;
                }
                let residual = 1. - weights.iter().sum::<f64>();
                let mandible = (1. - top_fraction) * (0.40 + 0.30 * lip);
                weights[jaw] += residual * mandible;
                weights[head] += residual * (1. - mandible);
                weights.iter_mut().for_each(|w| *w *= mask);
                for (&j, &w) in part.joints[id].iter().zip(&part.weights[id]) {
                    *weights.get_mut(j).context("skin joint out of range")? += w * (1. - mask);
                }
                let mut rank: Vec<_> = (0..weights.len()).collect();
                rank.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]));
                let joints = [rank[0], rank[1], rank[2], rank[3]];
                let values = joints.map(|j| weights[j]);
                let total = values.iter().sum::<f64>();
                ensure!(total > 0., "empty lip weights");
                edits.skin.push(Skin {
                    id,
                    joints,
                    weights: values.map(|w| w / total),
                });
            }
        }
        result.push(edits);
    }
    Ok(result)
}

pub fn run(input: &Path, output: &Path) -> Result<Value> {
    let result = bind(&read(input)?)?;
    charloom_io::write_json(output, &result)?;
    Ok(json!({"output":output,"parts":result.len()}))
}

#[derive(Deserialize)]
pub struct Rims {
    upper: Vec<Vec3>,
    lower: Vec<Vec3>,
}

/// Use the same horizontal grid and minimum opening as the sculpted aperture.
fn sample(rims: Rims) -> Result<(Vec<Vec3>, Vec<Vec3>, f64)> {
    use super::rim_curve::RimCurve;
    let ordered = |mut points: Vec<Vec3>| {
        points.sort_by(|a, b| a[0].total_cmp(&b[0]));
        points.dedup_by(|a, b| a[0] == b[0]);
        points
    };
    let (up, lo) = (ordered(rims.upper), ordered(rims.lower));
    let (up, lo) = (RimCurve::new(&up)?, RimCurve::new(&lo)?);
    let start = up.x[0].max(lo.x[0]);
    let end = up.x[up.x.len() - 1].min(lo.x[lo.x.len() - 1]);
    ensure!(end > start, "lip rims do not overlap");
    let (mut upper, mut lower) = (Vec::with_capacity(128), Vec::with_capacity(128));
    let mut clearance = 0.;
    for i in 0..128 {
        let u = i as f64 / 127.;
        let x = start + (end - start) * u;
        let (mut a, mut b) = (up.at(x), lo.at(x));
        let mid = (a[1] + b[1]) * 0.5;
        let gap = if i == 0 || i == 127 {
            0.
        } else {
            (a[1] - b[1]).max(0.0002 * (std::f64::consts::PI * u).sin())
        };
        a[1] = mid + gap * 0.5;
        b[1] = mid - gap * 0.5;
        upper.push(a);
        lower.push(b);
        if i == 63 || i == 64 {
            clearance += gap * 0.5;
        }
    }
    Ok((upper, lower, clearance))
}

pub fn apply(model: &Path, rims: Rims, out: &Path) -> Result<Value> {
    use charloom_io::load_gltf;
    let (upper, lower, clearance) = sample(rims)?;
    let mut doc = load_gltf(model)?.document;
    let gltf = gltf::Gltf::from_slice_without_validation(&serde_json::to_vec(&doc.json)?)?;
    let joints: Vec<_> = gltf
        .skins()
        .next()
        .context("missing skin")?
        .joints()
        .collect();
    let palette = joints
        .iter()
        .map(|j| j.name().context("unnamed skin joint").map(str::to_owned))
        .collect::<Result<_>>()?;
    let jaw = joints
        .iter()
        .find(|j| j.name() == Some("jaw"))
        .context("missing jaw")?;
    doc.json["nodes"][jaw.index()]["extras"]["charloomLipClearanceM"] = json!(clearance);
    let mut parts = Vec::new();
    let mut targets = Vec::new();
    for name in ["body", "beard", "teeth", "tongue"] {
        for mesh in gltf.meshes().filter(|mesh| mesh.name() == Some(name)) {
            for prim in mesh.primitives() {
                let r = prim.reader(|_| Some(&doc.binary));
                parts.push(Part {
                    name: name.into(),
                    positions: r
                        .read_positions()
                        .context("missing positions")?
                        .map(|p| p.map(f64::from))
                        .collect(),
                    normals: r
                        .read_normals()
                        .context("missing normals")?
                        .map(|p| p.map(f64::from))
                        .collect(),
                    joints: r
                        .read_joints(0)
                        .context("missing joints")?
                        .into_u16()
                        .map(|p| p.map(usize::from))
                        .collect(),
                    weights: r
                        .read_weights(0)
                        .context("missing weights")?
                        .into_f32()
                        .map(|p| p.map(f64::from))
                        .collect(),
                });
                targets.push((name, mesh.index(), prim));
            }
        }
    }
    let input = Input {
        palette,
        upper,
        lower,
        parts,
    };
    let edits = bind(&input)?;
    let mut changed = std::collections::BTreeMap::<_, usize>::new();
    for (((name, mesh, prim), edit), part) in targets.into_iter().zip(edits).zip(input.parts) {
        let mut attributes = Vec::new();
        if !edit.skin.is_empty() {
            *changed.entry(name).or_default() += edit.skin.len();
            let mut joints: Vec<_> = part.joints.iter().map(|p| p.map(|j| j as u16)).collect();
            let mut weights: Vec<_> = part.weights.iter().map(|p| p.map(|w| w as f32)).collect();
            for e in edit.skin {
                joints[e.id] = e.joints.map(|j| j as u16);
                weights[e.id] = e.weights.map(|w| w as f32);
            }
            let count = joints.len();
            let ji: Vec<_> = joints
                .into_iter()
                .flatten()
                .flat_map(u16::to_le_bytes)
                .collect();
            let wt: Vec<_> = weights
                .into_iter()
                .flatten()
                .flat_map(f32::to_le_bytes)
                .collect();
            attributes.push((
                "JOINTS_0",
                doc.append_accessor(&ji, 5123, "VEC4", count, Some(34962), None),
            ));
            attributes.push((
                "WEIGHTS_0",
                doc.append_accessor(&wt, 5126, "VEC4", count, Some(34962), None),
            ));
        }
        if !edit.lining.is_empty()
            && let Some(colors) = prim.reader(|_| Some(&doc.binary)).read_colors(0)
        {
            let mut colors: Vec<_> = colors.into_rgba_f32().collect();
            for (i, t) in edit.lining {
                for (a, to) in [0.077, 0.021, 0.024].iter().enumerate() {
                    colors[i][a] = ((1. - t) * f64::from(colors[i][a]) + t * to) as f32;
                }
            }
            let rgba = prim
                .get(&gltf::Semantic::Colors(0))
                .context("missing color accessor")?
                .dimensions()
                == gltf::accessor::Dimensions::Vec4;
            let width = if rgba { 4 } else { 3 };
            let bytes: Vec<_> = colors
                .iter()
                .flat_map(|p| p[..width].iter().flat_map(|n| n.to_le_bytes()))
                .collect();
            attributes.push((
                "COLOR_0",
                doc.append_accessor(
                    &bytes,
                    5126,
                    if rgba { "VEC4" } else { "VEC3" },
                    colors.len(),
                    Some(34962),
                    None,
                ),
            ));
        }
        if !edit.positions.is_empty() {
            let (mut positions, mut normals) = (part.positions, part.normals);
            for (i, p, n) in edit.positions {
                positions[i] = p.map(|x| f64::from(x as f32));
                normals[i] = n;
            }
            attributes.push(("POSITION", doc.append_vec3(&positions, Some(34962), true)));
            attributes.push(("NORMAL", doc.append_vec3(&normals, Some(34962), false)));
        }
        let attrs = &mut doc.json["meshes"][mesh]["primitives"][prim.index()]["attributes"];
        for (key, accessor) in attributes {
            attrs[key] = json!(accessor);
        }
    }
    doc.save(out)?;
    Ok(json!(changed))
}
