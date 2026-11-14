//! Reconstruct the covered cranium and neck behind a constrained frontal
//! chart. Loft onto the exact body collar, preserving its skin palette.
use crate::geom::{V, Weld, a, adjacency, v};
use anyhow::{Result, ensure};
use charloom_core::Vec3;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::Path;

#[derive(Deserialize)]
struct Input {
    positions: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    outline: Vec<usize>,
    rows: Vec<f64>,
    radius: f64,
    collar: Vec<Vec3>,
    collar_joints: Vec<[usize; 4]>,
    collar_weights: Vec<[f64; 4]>,
    head: usize,
    rims: Vec<usize>,
    eyes: [(Vec3, f64); 2],
}
fn align(points: &[V], mut ids: Vec<usize>) -> Vec<usize> {
    let mut area = 0.;
    for i in 0..ids.len() {
        let (a, b) = (points[ids[i]], points[ids[(i + 1) % ids.len()]]);
        area += a.x * b.z - b.x * a.z;
    }
    if area < 0. {
        ids.reverse();
    }
    let center = ids.iter().map(|&i| points[i]).sum::<V>() / ids.len() as f64;
    let angle = |i: usize| {
        let p = points[i] - center;
        p.z.atan2(p.x).abs()
    };
    let first = (0..ids.len())
        .min_by(|&i, &j| angle(ids[i]).total_cmp(&angle(ids[j])))
        .unwrap();
    ids.rotate_left(first);
    ids
}
fn perimeter(points: &[V], ids: &[usize]) -> Result<Vec<f64>> {
    let mut values = vec![0.];
    for i in 0..ids.len() {
        values.push(values[i] + (points[ids[i]] - points[ids[(i + 1) % ids.len()]]).norm());
    }
    let total = values[ids.len()];
    ensure!(total > 0., "collapsed collar");
    values.iter_mut().for_each(|x| *x /= total);
    Ok(values)
}
fn zipper(triangles: &mut Vec<[u32; 3]>, a: &[usize], b: &[usize]) {
    let (mut i, mut j) = (0, 0);
    while i < a.len() || j < b.len() {
        let (p, q) = (a[i % a.len()], b[j % b.len()]);
        let tri = if (i + 1) * b.len() < (j + 1) * a.len() {
            i += 1;
            [p, a[i % a.len()], q]
        } else {
            j += 1;
            [p, b[j % b.len()], q]
        };
        triangles.push(tri.map(|x| x as u32));
    }
}
fn interpolate_weights(
    head: usize,
    joints: [usize; 4],
    weights: [f64; 4],
    t: f64,
) -> ([usize; 4], [f64; 4]) {
    let mut combined = vec![(head, 1. - t)];
    for (j, w) in joints.into_iter().zip(weights) {
        if let Some((_, sum)) = combined.iter_mut().find(|(k, _)| *k == j) {
            *sum += w * t;
        } else {
            combined.push((j, w * t));
        }
    }
    combined.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut ids = [0; 4];
    let mut values = [0.; 4];
    for (k, (j, w)) in combined.into_iter().take(4).enumerate() {
        ids[k] = j;
        values[k] = w;
    }
    let total = values.iter().sum::<f64>();
    (ids, values.map(|x| x / total))
}
pub fn run(input: &Path, out: &Path) -> Result<Value> {
    let c: Input = super::read(input)?;
    let front = c.positions.len();
    let neck = c.collar.len();
    ensure!(
        neck >= 3 && c.collar_joints.len() == neck && c.collar_weights.len() == neck,
        "invalid body collar"
    );
    ensure!(
        c.rows.len() >= 2
            && c.rows.windows(2).all(|p| p[1] > p[0])
            && c.outline.len() == 2 * c.rows.len(),
        "invalid chart outline"
    );
    ensure!(
        c.outline.iter().chain(&c.rims).all(|&i| i < front)
            && c.triangles.iter().flatten().all(|&i| (i as usize) < front),
        "chart index out of bounds"
    );
    let mut verts: Vec<_> = c.positions.into_iter().map(v).collect();
    let mut tris = c.triangles;
    let mut previous = Vec::new();
    let mut cap = Vec::new();
    let throat = c.collar.iter().map(|&p| v(p)).sum::<V>() / neck as f64;
    let lo = c.rows[0];
    let hi = *c.rows.last().unwrap();
    for (i, &y) in c.rows.iter().enumerate() {
        let (il, ir) = (c.outline[i], c.outline[c.outline.len() - 1 - i]);
        let (left, right) = (verts[il], verts[ir]);
        let dorsal = left.z.min(right.z).min(throat.z)
            - 0.87
                * c.radius
                * (1. - ((2. * y - lo - hi) / (hi - lo)).powi(2))
                    .max(0.02)
                    .sqrt();
        let mut ring = vec![il];
        for k in 1..48 {
            let theta = k as f64 * (std::f64::consts::PI / 48.);
            let t = (1. - theta.cos()) / 2.;
            let mut p = left * (1. - t) + right * t;
            p.x -= ((y - hi + 33.) / 33.).clamp(0., 1.) * c.radius * 0.31 * (2. * theta).sin();
            p.z += (dorsal - (left.z + right.z) / 2.) * theta.sin();
            ring.push(verts.len());
            verts.push(p);
        }
        ring.push(ir);
        if previous.is_empty() {
            for k in 1..ring.len() - 1 {
                cap.push([ring[0], ring[k + 1], ring[k]].map(|j| j as u32));
            }
        } else {
            for (old, new) in previous.windows(2).zip(ring.windows(2)) {
                tris.push([old[0], old[1], new[0]].map(|j| j as u32));
                tris.push([old[1], new[1], new[0]].map(|j| j as u32));
            }
        }
        previous = ring;
    }
    tris.extend(cap);
    let upper = align(&verts, previous);
    let collar_start = verts.len();
    let collar: Vec<_> = (collar_start..collar_start + neck).collect();
    verts.extend(c.collar.iter().map(|&p| v(p)));
    let lower = align(&verts, collar.clone());
    let center_y = upper.iter().map(|&i| verts[i].y).sum::<f64>() / upper.len() as f64;
    let reach = (upper
        .iter()
        .map(|&i| verts[i].z)
        .fold(f64::NEG_INFINITY, f64::max)
        - throat.z)
        .max(0.001);
    for &i in &upper {
        verts[i].y = center_y + 0.018 * (1. - ((verts[i].z - throat.z) / reach).clamp(0., 1.));
    }
    let from = perimeter(&verts, &upper)?;
    let to = perimeter(&verts, &lower)?;
    let mut samples = Vec::new();
    for &step in &to[..neck] {
        let j = from.partition_point(|&u| u <= step).clamp(1, upper.len()) - 1;
        let width = from[j + 1] - from[j];
        ensure!(width > 0., "collapsed neck edge");
        let t = (step - from[j]) / width;
        samples.push(verts[upper[j]] * (1. - t) + verts[upper[(j + 1) % upper.len()]] * t);
    }
    let mut joints = vec![[c.head, 0, 0, 0]; verts.len()];
    let mut weights = vec![[1., 0., 0., 0.]; verts.len()];
    joints[collar_start..].copy_from_slice(&c.collar_joints);
    weights[collar_start..].copy_from_slice(&c.collar_weights);
    previous = upper;
    for t in [0.2, 0.42, 0.65, 0.84] {
        let mut ring = Vec::new();
        for (&p, &i) in samples.iter().zip(&lower) {
            ring.push(verts.len());
            verts.push(p * (1. - t) + verts[i] * t);
            let (js, ws) = interpolate_weights(c.head, joints[i], weights[i], t);
            joints.push(js);
            weights.push(ws);
        }
        zipper(&mut tris, &previous, &ring);
        previous = ring;
    }
    zipper(&mut tris, &previous, &lower);
    let count = verts.len();
    let weld = Weld {
        id: (0..count).collect(),
        count,
        representative: (0..count).collect(),
    };
    let adj = adjacency(&tris, &weld);
    let mut free = vec![true; count];
    for &i in c.rims.iter().chain(&collar) {
        free[i] = false;
    }
    for _ in 0..6 {
        let before: Vec<_> = verts.iter().map(|p| p.z).collect();
        for (i, neighbours) in adj.iter().enumerate() {
            if free[i] {
                let mean = neighbours.iter().map(|&j| before[j as usize]).sum::<f64>()
                    / neighbours.len().max(1) as f64;
                verts[i].z += 0.52 * (mean - before[i]);
            }
        }
    }
    for (center, radius) in c.eyes {
        ensure!(radius > 0., "invalid eyeball");
        for p in &mut verts[..front] {
            let r = radius * radius - (p.x - center[0]).powi(2) - (p.y - center[1]).powi(2);
            if r > 0. {
                p.z = p.z.max(center[2] + r.sqrt() + 0.0015);
            }
        }
    }
    let positions = verts.into_iter().map(a).collect::<Vec<_>>();
    charloom_io::write_json(
        out,
        &json!({"positions":positions,"triangles":tris,"collar":collar,"joints":joints,"weights":weights}),
    )?;
    Ok(json!({"output":out,"vertices":count}))
}
