 use crate::{hand_shape::HandShapeConstraint, surface_topology::SurfaceTopology};
use anyhow::{Context, Result, ensure};
use charloom_core::*;
use charloom_io::GlbDocument;
use nalgebra::{Matrix3, Matrix4, UnitQuaternion, Vector3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HandSurfaceCorrective {
    pub reference_time: f64,
    pub iterations: usize,
    pub global_iterations: usize,
    pub source_weight: f64,
    pub max_correction_m: f64,
    #[serde(default)]
    pub obstacles: Vec<HandCorrectiveObstacle>,
    #[serde(default)]
    pub pose_sampling: Option<HandCorrectiveSampling>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HandCorrectiveSampling {
    pub max_pose_error_rad: f64,
    pub max_samples: usize,
}

/// Explicit rigid capsule in an existing scene node's local coordinates.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct HandCorrectiveObstacle {
    pub node: String,
    pub start: Vec3,
    pub end: Vec3,
    pub radius_m: f64,
}

struct Capsule {
    start: Vector3<f64>,
    end: Vector3<f64>,
    radius: f64,
}

struct BoundaryBall {
    vertex: usize,
    center: Vector3<f64>,
    radius: f64,
}

fn project_boundaries(positions: &mut [Vector3<f64>], constraints: &[BoundaryBall]) {
    for c in constraints {
        let delta = positions[c.vertex] - c.center;
        let length = delta.norm();
        if length > c.radius {
            positions[c.vertex] = c.center + delta * (c.radius / length);
        }
    }
}

fn project_obstacles(positions: &mut [Vector3<f64>], movable: &[bool], obstacles: &[Capsule]) {
    for (i, p) in positions.iter_mut().enumerate() {
        if !movable[i] {
            continue;
        }
        for c in obstacles {
            let axis = c.end - c.start;
            let t = ((*p - c.start).dot(&axis) / axis.norm_squared()).clamp(0.0, 1.0);
            let center = c.start + axis * t;
            let delta = *p - center;
            let length = delta.norm();
            if length < c.radius && length > 1e-12 {
                *p = center + delta * (c.radius / length);
            }
        }
    }
}

 fn add_volume_links(
    rest: &[Vector3<f64>],
    normals: &[Vector3<f64>],
    eligible: &[bool],
    neighbors: &mut [Vec<usize>],
) -> usize {
    let cell = 0.045;
    let coords = |p: Vector3<f64>| {
        (
            (p.x / cell).floor() as i64,
            (p.y / cell).floor() as i64,
            (p.z / cell).floor() as i64,
        )
    };
    let mut grid: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
    for (i, &p) in rest.iter().enumerate() {
        if eligible[i] {
            grid.entry(coords(p)).or_default().push(i);
        }
    }
    let mut pairs = Vec::new();
    for (i, &p) in rest.iter().enumerate() {
        if !eligible[i] {
            continue;
        }
        let (x, y, z) = coords(p);
        let mut best = None;
        let mut score = f64::INFINITY;
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &j in grid.get(&(x + dx, y + dy, z + dz)).into_iter().flatten() {
                        if i == j || neighbors[i].contains(&j) || normals[i].dot(&normals[j]) > -0.3
                        {
                            continue;
                        }
                        let delta = rest[j] - p;
                        let distance = delta.norm();
                        if !(0.0025..=cell).contains(&distance) {
                            continue;
                        }
                        let a = -normals[i].dot(&delta) / distance;
                        let b = normals[j].dot(&delta) / distance;
                        if a < 0.5 || b < 0.5 {
                            continue;
                        }
                        let cost = distance * (2.0 - (a + b) * 0.5);
                        if cost < score {
                            score = cost;
                            best = Some(j);
                        }
                    }
                }
            }
        }
        if let Some(j) = best {
            pairs.push((i, j));
        }
    }
    for &(a, b) in &pairs {
        neighbors[a].push(b);
        neighbors[b].push(a);
    }
    for row in neighbors {
        row.sort_unstable();
        row.dedup();
    }
    pairs.len()
}

fn arap(
    rest: &[Vector3<f64>],
    source: &[Vector3<f64>],
    neighbors: &[Vec<usize>],
    movable: &[bool],
    config: &HandSurfaceCorrective,
    obstacles: &[Capsule],
    boundaries: &[BoundaryBall],
) -> Vec<Vector3<f64>> {
    let active: Vec<_> = (0..rest.len())
        .filter(|&i| movable[i] || neighbors[i].iter().any(|&j| movable[j]))
        .collect();
    let mut positions = source.to_vec();
    let mut rotations = vec![UnitQuaternion::identity(); rest.len()];
    let mut rhs = vec![Vector3::zeros(); rest.len()];
    for _ in 0..config.iterations {
        for &i in &active {
            let mut covariance = Matrix3::zeros();
            for &j in &neighbors[i] {
                covariance += (positions[i] - positions[j]) * (rest[i] - rest[j]).transpose();
            }
            let decomposition = covariance.svd(true, true);
            let mut u = decomposition.u.unwrap();
            let vt = decomposition.v_t.unwrap();
            if (u * vt).determinant() < 0.0 {
                u.column_mut(2).neg_mut();
            }
            rotations[i] = UnitQuaternion::from_matrix(&(u * vt));
        }
        for &i in &active {
            rhs[i] = Vector3::zeros();
            for &j in &neighbors[i] {
                let edge = rest[i] - rest[j];
                rhs[i] += (rotations[i] * edge + rotations[j] * edge) * 0.5;
            }
        }
        for _ in 0..config.global_iterations {
            for &i in &active {
                if !movable[i] || neighbors[i].is_empty() {
                    continue;
                }
                let mut sum = rhs[i] + source[i] * config.source_weight;
                for &j in &neighbors[i] {
                    sum += positions[j];
                }
                positions[i] = sum / (neighbors[i].len() as f64 + config.source_weight);
            }
            project_boundaries(&mut positions, boundaries);
            project_obstacles(&mut positions, movable, obstacles);
        }
    }
    positions
}

/// Remove residual surface stretch after the volume solve, keeping pinned
/// wrist vertices and the source grip. Interior links are not surface edges.
fn project_surface_limits(
    positions: &mut [Vector3<f64>],
    rest: &[Vector3<f64>],
    neighbors: &[Vec<usize>],
    movable: &[bool],
    ratio: f64,
    growth: f64,
) {
    for _ in 0..80 {
        let mut maximum = 0.0_f64;
        for (i, row) in neighbors.iter().enumerate() {
            for &j in row {
                if j <= i || !(movable[i] || movable[j]) {
                    continue;
                }
                let delta = positions[j] - positions[i];
                let length = delta.norm();
                let limit = (rest[j] - rest[i]).norm() * ratio + growth * 0.5;
                let excess = length - limit;
                if excess <= 0.0 || length < 1e-12 {
                    continue;
                }
                maximum = maximum.max(excess);
                let correction = delta * (excess / length)
                    / (usize::from(movable[i]) + usize::from(movable[j])) as f64;
                if movable[i] {
                    positions[i] += correction;
                }
                if movable[j] {
                    positions[j] -= correction;
                }
            }
        }
        if maximum < 1e-7 {
            break;
        }
    }
}

fn read_f32(document: &GlbDocument, accessor: usize, width: usize) -> Result<Vec<Vec<f64>>> {
    let a = &document.json["accessors"][accessor];
    ensure!(
        a["componentType"] == 5126,
        "corrective_requires_float_attribute"
    );
    let count = a["count"].as_u64().context("attribute_count")? as usize;
    let mut output = vec![vec![0.0; width]; count];
    let float = |at: usize| -> Result<f64> {
        Ok(f32::from_le_bytes(
            document
                .binary
                .get(at..at + 4)
                .context("attribute_bounds")?
                .try_into()
                .unwrap(),
        ) as f64)
    };
    if let Some(view) = a["bufferView"].as_u64() {
        let v = &document.json["bufferViews"][view as usize];
        let start = v["byteOffset"].as_u64().unwrap_or(0) as usize
            + a["byteOffset"].as_u64().unwrap_or(0) as usize;
        let stride = v["byteStride"].as_u64().map_or(width * 4, |n| n as usize);
        for (i, row) in output.iter_mut().enumerate() {
            for (k, value) in row.iter_mut().enumerate() {
                *value = float(start + i * stride + k * 4)?;
            }
        }
    }
    if let Some(sparse) = a.get("sparse") {
        let n = sparse["count"].as_u64().context("sparse_count")? as usize;
        let iv = &document.json["bufferViews"][sparse["indices"]["bufferView"]
            .as_u64()
            .context("sparse_indices")? as usize];
        let vv = &document.json["bufferViews"][sparse["values"]["bufferView"]
            .as_u64()
            .context("sparse_values")? as usize];
        let io = iv["byteOffset"].as_u64().unwrap_or(0) as usize
            + sparse["indices"]["byteOffset"].as_u64().unwrap_or(0) as usize;
        let vo = vv["byteOffset"].as_u64().unwrap_or(0) as usize
            + sparse["values"]["byteOffset"].as_u64().unwrap_or(0) as usize;
        let size = match sparse["indices"]["componentType"].as_u64() {
            Some(5121) => 1,
            Some(5123) => 2,
            Some(5125) => 4,
            _ => anyhow::bail!("invalid_sparse_index_type"),
        };
        for i in 0..n {
            let bytes = document
                .binary
                .get(io + i * size..io + (i + 1) * size)
                .context("sparse_index_bounds")?;
            let j = match size {
                1 => bytes[0] as usize,
                2 => u16::from_le_bytes(bytes.try_into().unwrap()) as usize,
                _ => u32::from_le_bytes(bytes.try_into().unwrap()) as usize,
            };
            ensure!(j < count, "sparse_index_range");
            for (k, value) in output[j].iter_mut().enumerate() {
                *value = float(vo + (i * width + k) * 4)?;
            }
        }
    }
    Ok(output)
}

fn append_sparse_vec3(document: &mut GlbDocument, values: &[Vec3], bounds: bool) -> usize {
    let mut selected: Vec<_> = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.iter().any(|x| x.abs() > 1e-12))
        .collect();
    if selected.is_empty() {
        selected.push((0, &values[0]));
    }
    let indices: Vec<_> = selected
        .iter()
        .flat_map(|(i, _)| (*i as u32).to_le_bytes())
        .collect();
    let data: Vec<_> = selected
        .iter()
        .flat_map(|(_, v)| v.iter().flat_map(|x| (*x as f32).to_le_bytes()))
        .collect();
    let iv = document.append_view(&indices, None);
    let vv = document.append_view(&data, None);
    let mut accessor = json!({"componentType":5126,"count":values.len(),"type":"VEC3","sparse":{"count":selected.len(),"indices":{"bufferView":iv,"componentType":5125},"values":{"bufferView":vv}}});
    if bounds && let Some(b) = Bounds::from_points(values.iter().copied()) {
        accessor["min"] = json!(b.min);
        accessor["max"] = json!(b.max);
    }
    let id = document.array("accessors").len();
    document.array("accessors").push(accessor);
    id
}

fn append_zero_to_existing_morph_tracks(
    document: &mut GlbDocument,
    node: usize,
    old_count: usize,
) -> Result<()> {
    if old_count == 0 {
        return Ok(());
    }
    let mut updates = Vec::new();
    for (ai, animation) in document.json["animations"]
        .as_array()
        .into_iter()
        .flatten()
        .enumerate()
    {
        for (ci, channel) in animation["channels"]
            .as_array()
            .into_iter()
            .flatten()
            .enumerate()
        {
            if channel["target"]["node"].as_u64() != Some(node as u64)
                || channel["target"]["path"] != "weights"
            {
                continue;
            }
            let sampler = channel["sampler"].as_u64().unwrap() as usize;
            let data = read_f32(
                document,
                animation["samplers"][sampler]["output"].as_u64().unwrap() as usize,
                1,
            )?;
            ensure!(
                data.len().is_multiple_of(old_count),
                "morph_track_width_mismatch"
            );
            let values: Vec<f64> = data
                .chunks_exact(old_count)
                .flat_map(|row| row.iter().map(|v| v[0]).chain([0.0]))
                .collect();
            updates.push((ai, ci, sampler, values));
        }
    }
    for (ai, ci, sampler, values) in updates {
        let bytes: Vec<_> = values
            .iter()
            .flat_map(|v| (*v as f32).to_le_bytes())
            .collect();
        let accessor = document.append_accessor(&bytes, 5126, "SCALAR", values.len(), None, None);
        let animation = &mut document.json["animations"][ai];
        let mut copy = animation["samplers"][sampler].clone();
        copy["output"] = json!(accessor);
        let samplers = animation["samplers"].as_array_mut().unwrap();
        let next = samplers.len();
        samplers.push(copy);
        animation["channels"][ci]["sampler"] = json!(next);
    }
    Ok(())
}

pub fn apply(
    asset: &mut Asset,
    document: &mut GlbDocument,
    clip: &mut MotionClip,
    hand: &HandShapeConstraint,
) -> Result<Value> {
    if let Some(sampling) = hand
        .surface_corrective
        .as_ref()
        .and_then(|c| c.pose_sampling.as_ref())
    {
        return apply_sampled(asset, document, clip, hand, sampling);
    }
    apply_static(asset, document, clip, hand)
}

fn apply_sampled(
    asset: &mut Asset,
    document: &mut GlbDocument,
    clip: &mut MotionClip,
    hand: &HandShapeConstraint,
    sampling: &HandCorrectiveSampling,
) -> Result<Value> {
    ensure!(
        (2..=64).contains(&sampling.max_samples)
            && sampling.max_pose_error_rad.is_finite()
            && (1e-4..=0.5).contains(&sampling.max_pose_error_rad),
        "invalid_hand_pose_sampling"
    );
    let digits: HashSet<_> = hand
        .digits
        .iter()
        .flat_map(|d| &d.joints)
        .map(|name| {
            asset
                .skeleton
                .joints
                .iter()
                .position(|j| j.name == *name)
                .with_context(|| format!("missing_corrective_digit: {name}"))
        })
        .collect::<Result<_>>()?;
    let partition = crate::hand_partition::compact(asset, document, clip, hand)?;
    let original = clip.clone();
    let range = original.time_range();
    let mut times: Vec<_> = original
        .tracks
        .iter()
        .filter(|t| digits.contains(&t.joint))
        .flat_map(|t| t.times.iter().copied())
        .chain(range)
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    let mids: Vec<_> = times.windows(2).map(|w| (w[0] + w[1]) * 0.5).collect();
    times.extend(mids);
    times.sort_by(f64::total_cmp);
    times.dedup();
    let poses: Vec<_> = times
        .iter()
        .map(|&t| original.sample(&asset.skeleton, t))
        .collect::<std::result::Result<_, _>>()?;
    let mut selected = vec![0, times.len() - 1];
    let mut max_error;
    loop {
        let mut worst = None;
        max_error = 0.0_f64;
        for pair in selected.windows(2) {
            let [a, b] = [pair[0], pair[1]];
            for k in a + 1..b {
                let t = (times[k] - times[a]) / (times[b] - times[a]);
                let error = digits
                    .iter()
                    .map(|&j| {
                        let rotation = unit_quat(poses[a][j].rotation)
                            .slerp(&unit_quat(poses[b][j].rotation), t);
                        rotation.angle_to(&unit_quat(poses[k][j].rotation))
                    })
                    .fold(0.0, f64::max);
                if error > max_error {
                    max_error = error;
                    worst = Some(k);
                }
            }
        }
        if max_error <= sampling.max_pose_error_rad {
            break;
        }
        ensure!(
            selected.len() < sampling.max_samples,
            "hand_pose_sample_budget_exceeded: {max_error}"
        );
        selected.push(worst.unwrap());
        selected.sort_unstable();
    }
    let mut sample_times: Vec<_> = selected.iter().map(|&i| times[i]).collect();
    let original_counts: Vec<_> = asset.meshes.iter().map(|m| m.morph_targets.len()).collect();
    let original_defaults: Vec<_> = asset
        .meshes
        .iter()
        .map(|m| m.morph_weights.clone())
        .collect();
    let mut reports = Vec::new();
    let mut fitted = 0;
    let mut surface_refinements = 0;
    let final_surface_error;
    loop {
        for &time in &sample_times[fitted..] {
            let mut work = original.clone();
            for track in &mut work.tracks {
                if digits.contains(&track.joint) {
                    let value = track.sample(time)?;
                    track.interpolation = Interpolation::Linear;
                    track.values = vec![value; track.times.len()];
                }
            }
             
            for mesh in &asset.meshes {
                if mesh.morph_targets.is_empty() {
                    continue;
                }
                if let Some(track) = work.morph_tracks.iter_mut().find(|t| t.node == mesh.node) {
                    for value in &mut track.values {
                        value.resize(mesh.morph_targets.len(), 0.0);
                    }
                } else {
                    let mut value = mesh.morph_weights.clone();
                    value.resize(mesh.morph_targets.len(), 0.0);
                    work.morph_tracks.push(MorphTrack {
                        node: mesh.node,
                        interpolation: Interpolation::Linear,
                        times: range.to_vec(),
                        values: vec![value.clone(), value],
                    });
                }
            }
            let mut config = hand.clone();
            let corrective = config.surface_corrective.as_mut().unwrap();
            corrective.reference_time = time;
            corrective.pose_sampling = None;
            reports.push(apply_static(asset, document, &mut work, &config)?);
        }
        fitted = sample_times.len();
        *clip = original.clone();
        let mut order: Vec<_> = (0..sample_times.len()).collect();
        order.sort_by(|&a, &b| sample_times[a].total_cmp(&sample_times[b]));
        for (id, mesh) in asset.meshes.iter().enumerate() {
            let old = original_counts[id];
            let added = mesh.morph_targets.len() - old;
            if added == 0 {
                continue;
            }
            ensure!(
                added == sample_times.len(),
                "inconsistent_hand_corrective_basis"
            );
            let mut keys = sample_times.clone();
            if let Some(track) = original.morph_tracks.iter().find(|t| t.node == mesh.node) {
                ensure!(
                    track.interpolation == Interpolation::Linear,
                    "pose_corrective_requires_baked_morph_tracks"
                );
                keys.extend(&track.times);
            }
            keys.sort_by(f64::total_cmp);
            keys.dedup();
            let mut values = Vec::new();
            for &time in &keys {
                let source = original.sample_morphs(time)?;
                let mut value = source
                    .get(&mesh.node)
                    .unwrap_or(&original_defaults[id])
                    .clone();
                value.resize(old + added, 0.0);
                let right = order
                    .partition_point(|&i| sample_times[i] <= time)
                    .min(added - 1);
                let left = right.saturating_sub(1);
                let (left, right) = (order[left], order[right]);
                if time <= sample_times[order[0]] {
                    value[old + order[0]] = 1.0;
                } else if time >= sample_times[order[added - 1]] {
                    value[old + order[added - 1]] = 1.0;
                } else {
                    let alpha =
                        (time - sample_times[left]) / (sample_times[right] - sample_times[left]);
                    value[old + left] = 1.0 - alpha;
                    value[old + right] = alpha;
                }
                values.push(value);
            }
            clip.morph_tracks.retain(|t| t.node != mesh.node);
            clip.morph_tracks.push(MorphTrack {
                node: mesh.node,
                interpolation: Interpolation::Linear,
                times: keys,
                values,
            });
        }
        let (error, time) = sampled_surface_error(asset, clip, hand, &times)?;
        if error <= 0.0001 || sample_times.iter().any(|&t| (t - time).abs() < 1e-8) {
            final_surface_error = error;
            break;
        }
        ensure!(
            sample_times.len() < sampling.max_samples,
            "hand_surface_sample_budget_exceeded: {error}"
        );
        sample_times.push(time);
        surface_refinements += 1;
    }
    Ok(
        json!({"requested":true,"method":"pose_sampled_volumetric_arap_correctives_v2","surface_refinement_samples":surface_refinements,"final_sampled_edge_excess_m":final_surface_error,"partition":partition,"sample_times_s":sample_times,"max_pose_interpolation_error_rad":max_error,"poses":reports,"base_geometry_unchanged":true,"skeleton_animation_unchanged":true,"limits":["Explicit named hand chains and a bounded sampled pose basis; final interpolated surfaces require independent verification."]}),
    )
}

/// Check the actual skinned interpolation, which can differ sharply from
/// quaternion interpolation when original blend matrices are ill-conditioned.
fn sampled_surface_error(
    asset: &Asset,
    clip: &MotionClip,
    hand: &HandShapeConstraint,
    times: &[f64],
) -> Result<(f64, f64)> {
    let names: HashSet<_> = std::iter::once(&hand.wrist)
        .chain(hand.digits.iter().flat_map(|d| &d.joints))
        .collect();
    let family: HashSet<_> = asset
        .skeleton
        .joints
        .iter()
        .enumerate()
        .filter_map(|(i, j)| names.contains(&j.name).then_some(i))
        .collect();
    let mut meshes = Vec::new();
    let mut all_edges = Vec::new();
    for mesh in &asset.meshes {
        if mesh.skin.is_none() {
            continue;
        }
        let selected: Vec<_> = mesh
            .influences
            .iter()
            .map(|row| {
                row.iter()
                    .filter(|w| family.contains(&w.joint))
                    .map(|w| w.weight)
                    .sum::<f64>()
                    >= 0.5
            })
            .collect();
        if !selected.iter().any(|&v| v) {
            continue;
        }
        let mut seen = HashSet::new();
        let mut edges = Vec::new();
        for triangle in &mesh.triangles {
            for k in 0..3 {
                let (a, b) = (triangle[k] as usize, triangle[(k + 1) % 3] as usize);
                let pair = (a.min(b), a.max(b));
                if (!selected[a] && !selected[b]) || !seen.insert(pair) {
                    continue;
                }
                let length = (v3(mesh.positions[a]) - v3(mesh.positions[b])).norm();
                if length < 1e-7 {
                    continue;
                }
                edges.push((a, b, length * hand.max_edge_ratio + hand.max_edge_growth_m));
            }
        }
        meshes.push(mesh.clone());
        all_edges.push(edges);
    }
    let subset = Asset {
        name: String::new(),
        source_hash: String::new(),
        skeleton: asset.skeleton.clone(),
        skins: asset.skins.clone(),
        meshes,
        clips: vec![],
        issues: vec![],
    };
    let mut worst = (0.0, 0.0);
    for &time in times {
        let points = subset.deformed_positions(
            &clip.sample(&subset.skeleton, time)?,
            &clip.sample_morphs(time)?,
        )?;
        for (positions, edges) in points.iter().zip(&all_edges) {
            for &(a, b, limit) in edges {
                let excess = (v3(positions[a]) - v3(positions[b])).norm() - limit;
                if excess > worst.0 {
                    worst = (excess, time);
                }
            }
        }
    }
    Ok(worst)
}

fn apply_static(
    asset: &mut Asset,
    document: &mut GlbDocument,
    clip: &mut MotionClip,
    hand: &HandShapeConstraint,
) -> Result<Value> {
    let Some(config) = &hand.surface_corrective else {
        return Ok(json!({"requested":false}));
    };
    let range = clip.time_range();
    ensure!(
        config.reference_time.is_finite()
            && config.reference_time >= range[0]
            && config.reference_time <= range[1] + 1e-6
            && (1..=100).contains(&config.iterations)
            && (1..=100).contains(&config.global_iterations)
            && config.source_weight.is_finite()
            && config.source_weight > 0.0
            && config.source_weight <= 1.0
            && config.max_correction_m.is_finite()
            && config.max_correction_m > 0.0
            && config.max_correction_m <= 0.25,
        "invalid_hand_surface_corrective"
    );
    let named = |name: &str| -> Result<usize> {
        let ids: Vec<_> = asset
            .skeleton
            .joints
            .iter()
            .enumerate()
            .filter(|(_, j)| j.name == name)
            .map(|(i, _)| i)
            .collect();
        ensure!(ids.len() == 1, "corrective_joint_must_be_unique: {name}");
        Ok(ids[0])
    };
    let wrist = named(&hand.wrist)?;
    let digits: HashSet<_> = hand
        .digits
        .iter()
        .flat_map(|d| &d.joints)
        .map(|n| named(n))
        .collect::<Result<_>>()?;
    // A static grip corrective must never be applied to an opening/closing hand.
    for track in clip.tracks.iter().filter(|t| digits.contains(&t.joint)) {
        let reference = track.sample(config.reference_time)?;
        for &time in &track.times {
            let value = track.sample(time)?;
            let error = match track.property {
                Property::Rotation => unit_quat(reference).angle_to(&unit_quat(value)),
                _ => v3([
                    reference[0] - value[0],
                    reference[1] - value[1],
                    reference[2] - value[2],
                ])
                .norm(),
            };
            ensure!(
                error < 1e-5,
                "fixed_grip_corrective_requires_constant_digits"
            );
        }
    }
    let world = asset
        .skeleton
        .world_matrices(&clip.sample(&asset.skeleton, config.reference_time)?)?;
    let obstacles: Vec<_> = config
        .obstacles
        .iter()
        .map(|o| -> Result<Capsule> {
            ensure!(
                o.radius_m.is_finite() && o.radius_m > 0.0 && o.radius_m <= 0.25,
                "invalid_corrective_obstacle_radius"
            );
            let node = named(&o.node)?;
            let start = v3(transform_point(&world[node], o.start));
            let end = v3(transform_point(&world[node], o.end));
            ensure!(
                (end - start).norm() > 1e-6,
                "invalid_corrective_obstacle_axis"
            );
            Ok(Capsule {
                start,
                end,
                radius: o.radius_m,
            })
        })
        .collect::<Result<_>>()?;
    let morphs = clip.sample_morphs(config.reference_time)?;
    let mut reports = Vec::new();
    for mesh_id in 0..asset.meshes.len() {
        let mesh = &asset.meshes[mesh_id];
        let Some(skin_id) = mesh.skin else {
            continue;
        };
        if !mesh.influences.iter().any(|r| {
            r.iter()
                .any(|w| digits.contains(&w.joint) && w.weight >= 0.1)
        }) {
            continue;
        }
        let skin = &asset.skins[skin_id];
        let bind_point = |joint: usize| -> Result<Vector3<f64>> {
            let at = skin
                .joints
                .iter()
                .position(|&j| j == joint)
                .context("corrective_bind_joint")?;
            let bind = matrix_from(skin.inverse_bind_matrices[at])
                .try_inverse()
                .context("corrective_bind_inverse")?;
            Ok(v3(transform_point(&bind, [0.0; 3])))
        };
        let bind_wrist = bind_point(wrist)?;
        let middle = hand
            .digits
            .iter()
            .find(|d| d.name == "middle")
            .or(hand.digits.first())
            .context("corrective_digit_chain")?;
        let palm = bind_point(named(&middle.joints[0])?)? - bind_wrist;
        let palm_length = palm.norm();
        ensure!(palm_length > 1e-6, "corrective_palm_axis");
        let palm_axis = palm / palm_length;
        let matrices: HashMap<_, _> = skin
            .joints
            .iter()
            .copied()
            .zip(&skin.inverse_bind_matrices)
            .map(|(j, b)| (j, world[j] * matrix_from(*b)))
            .collect();
        let transform: Vec<Matrix4<f64>> = mesh
            .influences
            .iter()
            .map(|row| {
                row.iter().fold(Matrix4::zeros(), |sum, w| {
                    sum + matrices[&w.joint] * w.weight
                })
            })
            .collect();
        let source_morphs = morphs.get(&mesh.node).unwrap_or(&mesh.morph_weights);
        let posed: Vec<_> = mesh
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let mut p = v3(*p);
                for (target, &w) in mesh.morph_targets.iter().zip(source_morphs) {
                    p += v3(target[i]) * w;
                }
                v3(transform_point(&transform[i], a3(p)))
            })
            .collect();
        let topology = SurfaceTopology::new(mesh);
        let representatives: Vec<_> = (0..mesh.positions.len())
            .filter(|&i| topology.representative[i] == i)
            .collect();
        let index: HashMap<_, _> = representatives
            .iter()
            .enumerate()
            .map(|(i, &v)| (v, i))
            .collect();
        let map: Vec<_> = topology.representative.iter().map(|r| index[r]).collect();
        let mut groups = vec![Vec::new(); representatives.len()];
        for (i, &v) in map.iter().enumerate() {
            groups[v].push(i);
        }
        let rest: Vec<_> = representatives
            .iter()
            .map(|&v| v3(mesh.positions[v]))
            .collect();
        let source: Vec<_> = groups
            .iter()
            .map(|g| g.iter().map(|&v| posed[v]).sum::<Vector3<f64>>() / g.len() as f64)
            .collect();
        let mut neighbors: Vec<Vec<usize>> = representatives
            .iter()
            .map(|&v| topology.adjacency[v].iter().map(|n| index[n]).collect())
            .collect();
        let movable: Vec<_> = groups
            .iter()
            .map(|g| {
                g.iter().all(|&v| {
                    let wrist_mass: f64 = mesh.influences[v]
                        .iter()
                        .filter(|w| w.joint == wrist)
                        .map(|w| w.weight)
                        .sum();
                    let family: f64 = mesh.influences[v]
                        .iter()
                        .filter(|w| w.joint == wrist || digits.contains(&w.joint))
                        .map(|w| w.weight)
                        .sum();
                    family >= 0.99999
                        && (wrist_mass < 0.98
                            || (!obstacles.is_empty()
                                && (v3(mesh.positions[v]) - bind_wrist).dot(&palm_axis)
                                    > palm_length * 0.15))
                })
            })
            .collect();
        if !movable.iter().any(|v| *v) {
            continue;
        }
        let mut surface_normals = vec![Vector3::zeros(); rest.len()];
        for tri in &mesh.triangles {
            let [a, b, c] = tri.map(|i| map[i as usize]);
            let n = (rest[b] - rest[a]).cross(&(rest[c] - rest[a]));
            for i in [a, b, c] {
                surface_normals[i] += n;
            }
        }
        for n in &mut surface_normals {
            if n.norm_squared() > 1e-20 {
                *n = n.normalize();
            }
        }
        let eligible: Vec<_> = groups
            .iter()
            .map(|g| {
                g.iter().all(|&v| {
                    mesh.influences[v]
                        .iter()
                        .filter(|w| w.joint == wrist || digits.contains(&w.joint))
                        .map(|w| w.weight)
                        .sum::<f64>()
                        >= 0.5
                })
            })
            .collect();
        let surface_neighbors = neighbors.clone();
        let volume_links = add_volume_links(&rest, &surface_normals, &eligible, &mut neighbors);
        // A static hand correction follows the wrist. Mixed forearm/hand
        // boundary vertices do not; constrain their whole trajectory in that
        // wrist frame so a good reference frame cannot hide later stretching.
        let mut boundary_edges = Vec::new();
        for (i, row) in surface_neighbors.iter().enumerate() {
            for &j in row {
                if movable[i] && !movable[j] && eligible[j] {
                    boundary_edges.push((
                        i,
                        j,
                        (rest[i] - rest[j]).norm() * hand.max_edge_ratio
                            + hand.max_edge_growth_m * 0.75,
                    ));
                }
            }
        }
        let mut times: Vec<_> = clip
            .tracks
            .iter()
            .flat_map(|t| t.times.iter().copied())
            .collect();
        times.push(config.reference_time);
        times.sort_by(f64::total_cmp);
        times.dedup();
        let mids: Vec<_> = times.windows(2).map(|p| (p[0] + p[1]) * 0.5).collect();
        times.extend(mids);
        let mut boundaries = Vec::new();
        let mut seen = HashSet::new();
        for time in times {
            if boundary_edges.is_empty() {
                break;
            }
            let sample_world = asset
                .skeleton
                .world_matrices(&clip.sample(&asset.skeleton, time)?)?;
            let align = world[wrist]
                * sample_world[wrist]
                    .try_inverse()
                    .context("corrective_wrist_inverse")?;
            let sampled_morphs = clip.sample_morphs(time)?;
            let weights = sampled_morphs
                .get(&mesh.node)
                .unwrap_or(&mesh.morph_weights);
            for &(i, j, radius) in &boundary_edges {
                let v = representatives[j];
                let mut p = v3(mesh.positions[v]);
                for (target, &w) in mesh.morph_targets.iter().zip(weights) {
                    p += v3(target[v]) * w;
                }
                let mut center = Vector3::zeros();
                for influence in &mesh.influences[v] {
                    let at = skin
                        .joints
                        .iter()
                        .position(|&j| j == influence.joint)
                        .context("corrective_boundary_joint")?;
                    let matrix = align
                        * sample_world[influence.joint]
                        * matrix_from(skin.inverse_bind_matrices[at]);
                    center += v3(transform_point(&matrix, a3(p))) * influence.weight;
                }
                let key = (i, j, center.map(|v| (v * 100000.0).round() as i64));
                if seen.insert(key) {
                    boundaries.push(BoundaryBall {
                        vertex: i,
                        center,
                        radius,
                    });
                }
            }
        }
        let mut solved = arap(
            &rest,
            &source,
            &neighbors,
            &movable,
            config,
            &obstacles,
            &boundaries,
        );
        project_surface_limits(
            &mut solved,
            &rest,
            &surface_neighbors,
            &movable,
            hand.max_edge_ratio,
            hand.max_edge_growth_m,
        );
        project_boundaries(&mut solved, &boundaries);
        project_obstacles(&mut solved, &movable, &obstacles);
        let max_correction = solved
            .iter()
            .zip(&source)
            .map(|(a, b)| (a - b).norm())
            .fold(0.0, f64::max);
        ensure!(
            max_correction <= config.max_correction_m,
            "hand_corrective_exceeds_displacement_budget: {max_correction}"
        );
        let corrected: Vec<_> = map
            .iter()
            .enumerate()
            .map(|(v, &i)| if movable[i] { solved[i] } else { posed[v] })
            .collect();
        let mut normal = vec![Vector3::zeros(); mesh.positions.len()];
        let mut normal_active = vec![false; mesh.positions.len()];
        for tri in &mesh.triangles {
            let [a, b, c] = tri.map(|i| i as usize);
            let n = (corrected[b] - corrected[a]).cross(&(corrected[c] - corrected[a]));
            for i in [a, b, c] {
                normal[i] += n;
                if movable[map[a]] || movable[map[b]] || movable[map[c]] {
                    normal_active[i] = true;
                }
            }
        }
        // Couple verified render seams for the corrective normal, just as for position.
        for group in &groups {
            let n = group.iter().map(|&v| normal[v]).sum::<Vector3<f64>>();
            for &v in group {
                normal[v] = n;
            }
        }
        let node = mesh.node;
        let primitive = mesh.primitive;
        let json_mesh = document.json["nodes"][node]["mesh"]
            .as_u64()
            .context("mesh_node")? as usize;
        ensure!(
            document.json["meshes"][json_mesh]["primitives"]
                .as_array()
                .unwrap()
                .len()
                == 1,
            "corrective_requires_one_primitive_per_node"
        );
        let normal_accessor =
            document.json["meshes"][json_mesh]["primitives"][primitive]["attributes"]["NORMAL"]
                .as_u64()
                .context("corrective_requires_normals")? as usize;
        let mut base_normals = read_f32(document, normal_accessor, 3)?;
        for (target, &weight) in
            document.json["meshes"][json_mesh]["primitives"][primitive]["targets"]
                .as_array()
                .into_iter()
                .flatten()
                .zip(source_morphs)
        {
            if weight != 0.0
                && let Some(accessor) = target["NORMAL"].as_u64()
            {
                let deltas = read_f32(document, accessor as usize, 3)?;
                for (normal, delta) in base_normals.iter_mut().zip(deltas) {
                    for k in 0..3 {
                        normal[k] += delta[k] * weight;
                    }
                }
            }
        }
        let mut delta = vec![[0.0; 3]; mesh.positions.len()];
        let mut normal_delta = delta.clone();
        for i in 0..delta.len() {
            if movable[map[i]] {
                let inv = transform[i]
                    .fixed_view::<3, 3>(0, 0)
                    .try_inverse()
                    .context("singular_corrective_skin_matrix")?;
                delta[i] = a3(inv * (corrected[i] - posed[i]));
            }
            if normal_active[i] && normal[i].norm_squared() > 1e-20 {
                let desired =
                    (transform[i].fixed_view::<3, 3>(0, 0).transpose() * normal[i]).normalize();
                normal_delta[i] =
                    a3(desired - v3([base_normals[i][0], base_normals[i][1], base_normals[i][2]]));
            }
        }
        let edge_error = |positions: &[Vector3<f64>]| {
            let mut worst = 0.0_f64;
            for (i, row) in neighbors.iter().enumerate() {
                for &j in row {
                    if movable[i] || movable[j] {
                        worst = worst.max(
                            ((positions[i] - positions[j]).norm() - (rest[i] - rest[j]).norm())
                                .abs(),
                        );
                    }
                }
            }
            worst
        };
        reports.push(json!({"mesh":mesh.name,"volume_links":volume_links,"movable_vertices":movable.iter().filter(|v|**v).count(),"maximum_correction_m":max_correction,"before_max_edge_length_error_m":edge_error(&source),"after_max_edge_length_error_m":edge_error(&solved)}));
        let old_count = mesh.morph_targets.len();
        append_zero_to_existing_morph_tracks(document, node, old_count)?;
        let position_accessor = append_sparse_vec3(document, &delta, true);
        let normal_accessor = append_sparse_vec3(document, &normal_delta, false);
        // Isolate a node's target list if its source mesh is shared.
        let mut json_mesh_value = document.json["meshes"][json_mesh].clone();
        let object = json_mesh_value["primitives"][primitive]
            .as_object_mut()
            .unwrap();
        object
            .entry("targets")
            .or_insert_with(|| json!([]))
            .as_array_mut()
            .unwrap()
            .push(json!({"POSITION":position_accessor,"NORMAL":normal_accessor}));
        let mut defaults = mesh.morph_weights.clone();
        defaults.resize(old_count, 0.0);
        defaults.push(0.0);
        json_mesh_value["weights"] = json!(defaults);
        let new_mesh = document.array("meshes").len();
        document.array("meshes").push(json_mesh_value);
        document.json["nodes"][node]["mesh"] = json!(new_mesh);
        document.json["nodes"][node]["weights"] = json!(defaults);
        let mesh = &mut asset.meshes[mesh_id];
        mesh.morph_targets.push(delta);
        mesh.morph_weights = defaults;
        for existing in &mut asset.clips {
            for track in &mut existing.morph_tracks {
                if track.node == node {
                    for values in &mut track.values {
                        values.push(0.0);
                    }
                }
            }
        }
        if let Some(track) = clip.morph_tracks.iter_mut().find(|t| t.node == node) {
            for values in &mut track.values {
                values.push(1.0);
            }
        } else {
            let mut values = mesh.morph_weights.clone();
            *values.last_mut().unwrap() = 1.0;
            clip.morph_tracks.push(MorphTrack {
                node,
                interpolation: Interpolation::Linear,
                times: vec![range[0], range[1]],
                values: vec![values.clone(), values],
            });
        }
    }
    Ok(
        json!({"requested":true,"method":"fixed_grip_volumetric_arap_corrective_v3","reference_time":config.reference_time,"meshes":reports,"base_geometry_unchanged":true,"skeleton_animation_unchanged":true,"limits":["Requires constant digit local transforms throughout the clip.","A clip-scoped morph corrects the hand surface; it is not a physical grasp or collision simulation."]}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capsule_projection_keeps_pinned_boundary_and_moves_only_penetrations() {
        let source = vec![
            Vector3::new(0.005, 0.03, 0.0),
            Vector3::new(0.03, 0.03, 0.0),
            Vector3::new(0.004, 0.0, 0.0),
        ];
        let mut points = source.clone();
        project_obstacles(
            &mut points,
            &[true, true, false],
            &[Capsule {
                start: Vector3::zeros(),
                end: Vector3::new(0.0, 0.1, 0.0),
                radius: 0.015,
            }],
        );
        assert!((points[0].x - 0.015).abs() < 1e-12);
        assert_eq!(points[1], source[1]);
        assert_eq!(points[2], source[2]);
        assert!((points[0].y - source[0].y).abs() < 1e-12);
    }

    #[test]
    fn volume_links_cross_a_finger_interior_without_joining_the_air_gap() {
        let mut points = Vec::new();
        let mut normals = Vec::new();
        let mut neighbors = Vec::new();
        for tube in 0..2 {
            for ring in 0..3 {
                for side in 0..8 {
                    let angle = side as f64 * std::f64::consts::TAU / 8.0;
                    let normal = Vector3::new(angle.cos(), 0.0, angle.sin());
                    points.push(
                        Vector3::new(tube as f64 * 0.014, ring as f64 * 0.01, 0.0) + normal * 0.005,
                    );
                    normals.push(normal);
                    let base = tube * 24 + ring * 8;
                    neighbors.push(vec![base + (side + 1) % 8, base + (side + 7) % 8]);
                }
            }
        }
        let count = add_volume_links(&points, &normals, &vec![true; points.len()], &mut neighbors);
        assert!(count >= 32);
        for (i, row) in neighbors.iter().enumerate() {
            assert!(
                row.iter().all(|&j| i / 24 == j / 24),
                "must not bridge the air between fingers"
            );
        }
    }

    #[test]
    fn rigid_rotation_is_preserved_and_stretched_surface_is_reduced() {
        let rest: Vec<_> = (0..8)
            .map(|i| {
                Vector3::new(
                    (i & 1) as f64 * 0.01,
                    ((i >> 1) & 1) as f64 * 0.01,
                    ((i >> 2) & 1) as f64 * 0.03,
                )
            })
            .collect();
        let neighbors: Vec<_> = (0..8)
            .map(|i| (0..8).filter(|&j| i != j).collect())
            .collect();
        let config = HandSurfaceCorrective {
            reference_time: 0.0,
            iterations: 30,
            global_iterations: 40,
            source_weight: 0.0001,
            max_correction_m: 0.1,
            obstacles: Vec::new(),
            pose_sampling: None,
        };
        let rotation = UnitQuaternion::from_euler_angles(0.4, -0.5, 1.1);
        let rigid: Vec<_> = rest
            .iter()
            .map(|p| rotation * p + Vector3::new(0.3, 0.8, 0.1))
            .collect();
        let fixed = arap(&rest, &rigid, &neighbors, &[true; 8], &config, &[], &[]);
        assert!(fixed.iter().zip(&rigid).all(|(a, b)| (a - b).norm() < 1e-5));
        let mut stretched = rigid.clone();
        for (i, p) in stretched.iter_mut().enumerate() {
            if i >= 4 {
                *p += rotation * Vector3::new(0., 0., 0.02);
            }
        }
        let repaired = arap(&rest, &stretched, &neighbors, &[true; 8], &config, &[], &[]);
        let error = |p: &[Vector3<f64>]| {
            (0..8)
                .flat_map(|i| (i + 1..8).map(move |j| (i, j)))
                .map(|(i, j)| ((p[i] - p[j]).norm() - (rest[i] - rest[j]).norm()).abs())
                .fold(0., f64::max)
        };
        assert!(error(&repaired) < error(&stretched) * 0.01);
    }
}
