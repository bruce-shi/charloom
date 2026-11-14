//! Keep the unobserved scalp inside the observed hair shell, then place the
//! fitted head in the existing body's attachment frame. These are separate from
//! facial identity fitting; neither operation welds provider skin into the body.
use crate::{FaceRegions, Landmark};
use anyhow::{Context, Result, ensure};
use nalgebra::{Matrix4, SMatrix, Vector4};
use std::collections::BTreeSet;

/// The lower head transition is the neck, not an entire horizontal slice of
/// the character. Deltoid tops can be higher than the lower neck boundary.
pub fn neck_blend(original: &[V], start: f64, end: f64) -> Vec<f64> {
    original
        .iter()
        .map(|p| {
            let vertical = crate::smooth((p.z - start) / (end - start));
            let transverse = if p.z >= end {
                1.
            } else {
                crate::smooth((0.13 - p.x.abs()) / 0.04)
            };
            vertical * transverse
        })
        .collect()
}

fn band_quality(body: &[V], points: &[V], faces: &[[usize; 3]]) -> (usize, f64) {
    let mut reversed = 0;
    let mut minimum_area_ratio = f64::INFINITY;
    for &[a, b, c] in faces {
        let before = (body[b] - body[a]).cross(&(body[c] - body[a]));
        let after = (points[b] - points[a]).cross(&(points[c] - points[a]));
        if before.norm() < 1e-12 {
            continue;
        }
        reversed += usize::from(before.dot(&after) <= 0.);
        minimum_area_ratio = minimum_area_ratio.min(after.norm() / before.norm());
    }
    (reversed, minimum_area_ratio)
}

 pub fn initial_neck(
    original: &[V],
    body: &[V],
    head: &[V],
    head_dz: V,
    blend: &[f64],
    body_ids: &[usize],
    faces: &[[usize; 3]],
    start: f64,
    end: f64,
) -> Result<(Vec<V>, serde_json::Value)> {
    let mut points: Vec<_> = body
        .iter()
        .zip(head)
        .zip(blend)
        .map(|((b, h), &w)| b * (1. - w) + h * w)
        .collect();
    let band: Vec<_> = faces
        .iter()
        .filter(|f| {
            f.iter()
                .all(|&i| original[i].z >= start && original[i].z <= end)
        })
        .copied()
        .collect();
    let (old_reversed, old_area) = band_quality(body, &points, &band);
    if old_reversed == 0 && old_area >= 0.05 {
        return Ok((
            points,
            serde_json::json!({
                "stage":"initial_neck_transition", "compression_repair_applied":false,
                "reversed_triangles_before":old_reversed, "minimum_area_ratio_before":old_area,
                "method":"unchanged position blend; no collapsed or reversed neck triangles detected"
            }),
        ));
    }
    let ids: Vec<_> = body_ids
        .iter()
        .copied()
        .filter(|&i| original[i].z > start && original[i].z < end && blend[i] > 0.)
        .collect();
    ensure!(
        ids.len() >= 12,
        "Insufficient neck support for compression repair"
    );
    let center = ids.iter().map(|&i| original[i]).sum::<V>() / ids.len() as f64;
    let mut gram = Matrix4::zeros();
    let mut rhs = SMatrix::<f64, 4, 3>::zeros();
    for &i in &ids {
        let q = original[i] - center;
        let a = Vector4::new(q.x, q.y, q.z, 1.);
        gram += a * a.transpose();
        rhs += a * body[i].transpose();
    }
    let affine = gram.lu().solve(&rhs).context("Neck support is singular")?;
    let body_dz = V::new(affine[(2, 0)], affine[(2, 1)], affine[(2, 2)]);
    let length = end - start;
    let mut minimum_rise = f64::INFINITY;
    for &i in &ids {
        let lower = body[i] + body_dz * (start - original[i].z);
        let upper = head[i] + head_dz * (end - original[i].z);
        minimum_rise = minimum_rise.min(upper.z - lower.z);
    }
    ensure!(
        minimum_rise > 1e-6,
        "Head is below the neck root; revise body/head placement"
    );
    let mut maximum_correction = 0f64;
    let mut changed = 0;
    for i in 0..points.len() {
        let u = (original[i].z - start) / length;
        if !(0. < u && u < 1.) || blend[i] == 0. {
            continue;
        }
        let lower = body[i] + body_dz * (start - original[i].z);
        let upper = head[i] + head_dz * (end - original[i].z);
        let mut m0 = body_dz * length;
        let mut m1 = head_dz * length;
        m0.z = m0.z.max(0.);
        m1.z = m1.z.max(0.);
        let slope_sum = m0.z + m1.z;
        let allowed = 3. * (upper.z - lower.z).max(0.);
        if slope_sum > allowed {
            m0.z *= allowed / slope_sum;
            m1.z *= allowed / slope_sum;
        }
        let curve = lower * (2. * u.powi(3) - 3. * u.powi(2) + 1.)
            + m0 * (u.powi(3) - 2. * u.powi(2) + u)
            + upper * (-2. * u.powi(3) + 3. * u.powi(2))
            + m1 * (u.powi(3) - u.powi(2));
        let transverse = (blend[i] / crate::smooth(u)).clamp(0., 1.);
        let next = body[i] + (curve - body[i]) * transverse;
        maximum_correction = maximum_correction.max((next - points[i]).norm());
        points[i] = next;
        changed += 1;
    }
    let (reversed, area) = band_quality(body, &points, &band);
    ensure!(
        reversed == 0 && area >= 0.05,
        "Initial neck compression could not be repaired safely"
    );
    Ok((
        points,
        serde_json::json!({
            "stage":"initial_neck_transition", "compression_repair_applied":true,
            "method":"boundary-station Hermite interpolation with monotone height; fixed lower body and upper head",
            "vertices":changed, "reversed_triangles_before":old_reversed,
            "reversed_triangles_after":reversed, "minimum_area_ratio_before":old_area,
            "minimum_area_ratio_after":area, "minimum_station_rise_mm":minimum_rise*1000.,
            "maximum_correction_mm":maximum_correction*1000., "global_collision_certified":false
        }),
    ))
}

 pub fn relax_neck_root(
    body: &[V],
    original: &[V],
    points: &mut [V],
    faces: &[[usize; 3]],
    regions: &FaceRegions,
    landmarks: &[Landmark],
) -> Result<serde_json::Value> {
    let chin = landmarks
        .iter()
        .find(|l| l.name == "chin")
        .context("Missing chin")?;
    let start = chin.original[2] - 0.16;
    let end = chin.original[2] + 0.035;
    let outer: BTreeSet<_> = regions.outer.iter().copied().collect();
    let cavity: BTreeSet<_> = regions
        .cavities
        .iter()
        .flatten()
        .filter(|i| !outer.contains(i))
        .copied()
        .collect();
    let frozen: BTreeSet<_> = landmarks.iter().map(|l| l.vertex).collect();
    let mut edges = BTreeSet::new();
    for &[a, b, c] in faces {
        if [a, b, c].iter().any(|i| cavity.contains(i)) {
            continue;
        }
        for (i, j) in [(a, b), (b, c), (c, a)] {
            edges.insert((i.min(j), i.max(j)));
        }
    }
    let edges: Vec<_> = edges.into_iter().collect();
    let mut degree = vec![0f64; points.len()];
    for &(a, b) in &edges {
        degree[a] += 1.;
        degree[b] += 1.;
    }
    let ids: Vec<_> = (0..points.len())
        .filter(|&i| {
            degree[i] > 0.
                && !frozen.contains(&i)
                && original[i].z > start
                && original[i].z < end
                && original[i].x.abs() < 0.13
                && !(original[i].z >= chin.original[2] - 0.003
                    && original[i].y <= chin.original[1] + 0.030)
        })
        .collect();
    ensure!(!ids.is_empty(), "Empty neck root patch");
    let laplacian = |values: &[V]| {
        let mut out = vec![V::zeros(); values.len()];
        for &(a, b) in &edges {
            let d = values[a] - values[b];
            out[a] += d;
            out[b] -= d;
        }
        out
    };
    let initial: Vec<_> = points.iter().zip(body).map(|(p, b)| p - b).collect();
    let mut fixed = initial.clone();
    for &i in &ids {
        fixed[i] = V::zeros();
    }
    let forcing = laplacian(&laplacian(&fixed));
    let fidelity = 0.002;
    let rhs: Vec<_> = ids
        .iter()
        .map(|&i| initial[i] * fidelity - forcing[i])
        .collect();
    let diagonal: Vec<_> = ids
        .iter()
        .map(|&i| degree[i] * degree[i] + degree[i] + fidelity)
        .collect();
    let apply = |x: &[V]| {
        let mut full = vec![V::zeros(); points.len()];
        for (&i, v) in ids.iter().zip(x) {
            full[i] = *v;
        }
        let squared = laplacian(&laplacian(&full));
        ids.iter()
            .enumerate()
            .map(|(k, &i)| squared[i] + x[k] * fidelity)
            .collect::<Vec<_>>()
    };
    let dot = |a: &[V], b: &[V]| a.iter().zip(b).map(|(a, b)| a.dot(b)).sum::<f64>();
    let mut x: Vec<_> = ids.iter().map(|&i| initial[i]).collect();
    let ax = apply(&x);
    let mut residual: Vec<_> = rhs.iter().zip(ax).map(|(b, a)| b - a).collect();
    let mut direction: Vec<_> = residual
        .iter()
        .zip(&diagonal)
        .map(|(r, d)| r / *d)
        .collect();
    let mut rz = dot(&residual, &direction);
    let initial_rz = rz;
    let mut iterations = 0;
    for step in 0..1200 {
        if rz < 1e-22 || rz < initial_rz * 1e-14 {
            break;
        }
        let ad = apply(&direction);
        let denominator = dot(&direction, &ad);
        ensure!(denominator > 0., "Invalid neck-root system");
        let alpha = rz / denominator;
        for k in 0..x.len() {
            x[k] += direction[k] * alpha;
            residual[k] -= ad[k] * alpha;
        }
        let z: Vec<_> = residual
            .iter()
            .zip(&diagonal)
            .map(|(r, d)| r / *d)
            .collect();
        let next = dot(&residual, &z);
        let beta = next / rz;
        for k in 0..direction.len() {
            direction[k] = z[k] + direction[k] * beta;
        }
        rz = next;
        iterations = step + 1;
    }
    ensure!(
        x.iter().all(|p| p.iter().all(|v| v.is_finite())),
        "Nonfinite neck-root solution"
    );
    ensure!(
        rz < 1e-18 || rz < initial_rz * 1e-10,
        "Neck-root solve did not converge"
    );
    let mut maximum_correction = 0f64;
    for (&i, d) in ids.iter().zip(x) {
        let next = body[i] + d;
        maximum_correction = maximum_correction.max((next - points[i]).norm());
        points[i] = next;
    }
    Ok(serde_json::json!({
        "method":"biharmonic displacement from body prior; fixed face and body anchors",
        "vertices":ids.len(), "vertex_ids":ids, "start_z":start, "end_z":end,
        "maximum_correction_mm":maximum_correction*1000., "iterations":iterations,
        "relative_preconditioned_residual":if initial_rz>0. {rz/initial_rz}else{0.},
        "global_collision_certified":false
    }))
}

/// Extend facial displacement through one connected neck surface. An authored
/// UV boundary is not an anatomical hinge. Smooth displacement relative to the
/// continuous aligned prior, not positions (which would shrink the neck).
pub fn neck_transition(
    base: &[V],
    original: &[V],
    p: &mut [V],
    faces: &[[usize; 3]],
    regions: &FaceRegions,
    landmarks: &[Landmark],
    start: f64,
    blend: &[f64],
) -> serde_json::Value {
    let chin = landmarks.iter().find(|l| l.name == "chin").unwrap();
    let end = chin.original[2] + 0.035;
    let outer: BTreeSet<_> = regions.outer.iter().copied().collect();
    let cavity: BTreeSet<_> = regions
        .cavities
        .iter()
        .flatten()
        .filter(|i| !outer.contains(i))
        .copied()
        .collect();
    let mut graph = vec![BTreeSet::new(); p.len()];
    for &[a, b, c] in faces {
        if [a, b, c].iter().any(|i| cavity.contains(i)) {
            continue;
        }
        for (i, j) in [(a, b), (b, c), (c, a)] {
            graph[i].insert(j);
            graph[j].insert(i);
        }
    }
    let frozen: BTreeSet<_> = landmarks.iter().map(|l| l.vertex).collect();
    let ids: Vec<_> = (0..p.len())
        .filter(|&i| {
            !graph[i].is_empty()
                && !frozen.contains(&i)
                && blend[i] > 0.
                && original[i].z > start
                && original[i].z < end
                // The forward chin/lower-lip surface belongs to the fitted
                // face. Keep it intact while extending into the neck behind it.
                && !(original[i].z >= chin.original[2]-0.003
                    && original[i].y <= chin.original[1]+0.030)
        })
        .collect();
    let previous = p.to_vec();
    let mut delta: Vec<_> = p.iter().zip(base).map(|(p, q)| p - q).collect();
    let initial = delta.clone();
     
    let weight = vec![0.015; ids.len()];
    for _ in 0..400 {
        let mut maximum = 0f64;
        for (k, &i) in ids.iter().enumerate() {
            let mut sum = initial[i] * weight[k];
            for &j in &graph[i] {
                sum += delta[j];
            }
            let next = sum / (graph[i].len() as f64 + weight[k]);
            maximum = maximum.max((next - delta[i]).norm());
            delta[i] = next;
        }
        if maximum < 1e-9 {
            break;
        }
    }
    let mut maximum = 0f64;
    for &i in &ids {
        p[i] = base[i] + delta[i];
        maximum = maximum.max((p[i] - previous[i]).norm());
    }
    serde_json::json!({"stage":"continuous_neck_displacement","vertices":ids.len(),"maximum_correction_mm":maximum*1000.,"method":"connected skin displacement regularization; facial landmarks and lower body anchored; no UV island hinge"})
}
use crate::surface::{Surface, V};

 fn scalp_field(
    neighbors: &[Vec<usize>],
    active: &[bool],
    upper: &[f64],
    target: &[f64],
    confidence: &[f64],
) -> Vec<f64> {
    let mut d = vec![0.; neighbors.len()];
    for _ in 0..1000 {
        let mut maximum = 0f64;
        for i in 0..d.len() {
            if !active[i] {
                continue;
            }
            let weight = confidence[i] * 2.;
            let next = ((neighbors[i].iter().map(|&j| d[j]).sum::<f64>() + target[i] * weight)
                / (neighbors[i].len() as f64 + weight + 0.015))
                .min(upper[i]);
            maximum = maximum.max((next - d[i]).abs());
            d[i] = next;
        }
        if maximum < 1e-8 {
            break;
        }
    }
    d
}

/// The lower local radial envelope estimates the head-supporting hair layer.
/// Raised tufts, knots and ornaments cannot pull a solitary patch toward them.
fn scalp_target(radius: f64, cap: f64, neighbor_ratios: &mut [f64]) -> f64 {
    neighbor_ratios.sort_by(f64::total_cmp);
    let envelope = if neighbor_ratios.len() >= 6 {
        radius * (neighbor_ratios[(neighbor_ratios.len() - 1) / 4] + 0.10)
    } else {
        cap
    };
    (cap.min(envelope) - radius).min(radius * 0.50).min(0.050)
}

/// Estimate broad cranial volume with three directional coefficients. Hair
/// grooves and a high swept fringe cannot become local skull bumps. IRLS limits
/// the effect of isolated raised hair while retaining both growth and shrinkage.
fn broad_scalp_fit(directions: &[V], target: &[f64], confidence: &[f64]) -> V {
    let count = confidence.iter().filter(|&&w| w > 0.).count().max(1) as f64;
    let mut coefficients = V::zeros();
    for _ in 0..16 {
        let mut gram = nalgebra::Matrix3::identity() * (count * 0.04);
        let mut rhs = V::zeros();
        for k in 0..target.len() {
            if confidence[k] == 0. {
                continue;
            }
            let a = directions[k].component_mul(&directions[k]);
            let residual = (a.dot(&coefficients) - target[k]).abs();
            let weight = 0.008 / residual.max(0.008);
            gram += a * a.transpose() * weight;
            rhs += a * target[k] * weight;
        }
        if let Some(next) = gram.lu().solve(&rhs) {
            coefficients = next.map(|v| v.clamp(-0.04, 0.05));
        }
    }
    coefficients
}

/// Hair clearance cannot justify folding the pinna into itself. Back off only
/// the proposed displacement near a new crossing or fold, along connected skin.
fn guard_scalp_revision(before: &[V], p: &mut [V], faces: &[[usize; 3]]) -> serde_json::Value {
    use std::collections::VecDeque;
    let prior = Surface::new(before, faces).crossings();
    let proposal = p.to_vec();
    let delta: Vec<_> = proposal.iter().zip(before).map(|(a, b)| a - b).collect();
    let moving: Vec<_> = delta.iter().map(|d| d.norm() > 1e-10).collect();
    let mut graph = vec![Vec::new(); p.len()];
    for &[a, b, c] in faces {
        for (a, b) in [(a, b), (b, c), (c, a)] {
            let length = (before[a] - before[b]).norm();
            graph[a].push((b, length));
            graph[b].push((a, length));
        }
    }
    let mut anchors = BTreeSet::new();
    let mut history = Vec::new();
    let mut fallback = false;
    for iteration in 0..=16 {
        let current = Surface::new(p, faces).crossings();
        let introduced: Vec<_> = current.difference(&prior).copied().collect();
        let folded: Vec<_> = faces
            .iter()
            .enumerate()
            .filter_map(|(k, &[a, b, c])| {
                let old = (before[b] - before[a]).cross(&(before[c] - before[a]));
                let new = (p[b] - p[a]).cross(&(p[c] - p[a]));
                ((moving[a] || moving[b] || moving[c])
                    && old.norm() > 1e-12
                    && (old.dot(&new) <= 0. || new.norm() < old.norm() * 0.05))
                    .then_some(k)
            })
            .collect();
        history.push(serde_json::json!({"iteration":iteration,"new_crossings":introduced.len(),"folded_or_collapsed_triangles":folded.len()}));
        if introduced.is_empty() && folded.is_empty() {
            break;
        }
        if iteration == 16 {
            p.copy_from_slice(before);
            fallback = true;
            break;
        }
        for face in introduced.iter().flatten().chain(folded.iter()) {
            for &i in &faces[*face] {
                if moving[i] {
                    anchors.insert(i);
                }
            }
        }
        let feather = 0.018;
        let mut distances = vec![f64::INFINITY; p.len()];
        let mut queue = VecDeque::new();
        for &i in &anchors {
            distances[i] = 0.;
            queue.push_back(i);
        }
        while let Some(i) = queue.pop_front() {
            for &(j, length) in &graph[i] {
                let next = distances[i] + length;
                if next < feather && next + 1e-12 < distances[j] {
                    distances[j] = next;
                    queue.push_back(j);
                }
            }
        }
        for i in 0..p.len() {
            p[i] = before[i] + delta[i] * crate::smooth(distances[i] / feather);
        }
    }
    let final_pairs = Surface::new(p, faces).crossings();
    serde_json::json!({"method":"local geodesic backoff of scalp displacement at new skin crossings or folds",
        "prior_crossings":prior.len(),"final_crossings":final_pairs.len(),"new_crossings":final_pairs.difference(&prior).count(),
        "full_proposal_reverted":fallback,"vertices_reduced":p.iter().zip(&proposal).filter(|(a,b)|(**a-**b).norm()>1e-10).count(),
        "iterations":history,"scope":"outer skin relative to the pre-scalp fit; hair containment and existing contacts remain separate diagnostics"})
}

pub fn contain_scalp(
    p: &mut [V],
    ids: &[usize],
    edges: &[(usize, usize)],
    faces: &[[usize; 3]],
    landmarks: &[Landmark],
    preserve: &[usize],
    hair: &Surface,
    complete: &Surface,
) -> serde_json::Value {
    let before = p.to_vec();
    let lm = |name: &str| V::from(landmarks.iter().find(|l| l.name == name).unwrap().observed);
    let eye = (lm("eye_upper_left") + lm("eye_upper_right")) * 0.5;
    let center = eye + V::new(0., 0.055, 0.015);
    let chin = lm("chin").z;
    let forehead = lm("forehead").z;
    let frozen: BTreeSet<_> = landmarks.iter().map(|l| l.vertex).collect();
    let preserved: BTreeSet<_> = preserve.iter().copied().collect();
    let fade = |q: V| {
        let front = crate::smooth((center.y + 0.015 - q.y) / 0.04);
        1. - front + front * crate::smooth((q.z - forehead) / 0.05)
    };
    let mut neighbors = vec![vec![]; ids.len()];
    for &(a, b) in edges {
        neighbors[a].push(b);
        neighbors[b].push(a);
    }
    let mut active = vec![false; ids.len()];
    let upper = vec![f64::INFINITY; ids.len()];
    let mut caps = vec![None; ids.len()];
    let mut radii = vec![0.; ids.len()];
    let mut directions = vec![V::zeros(); ids.len()];
    let mut boundary_fade = vec![0.; ids.len()];
    let mut rejected = 0;
    for (k, &i) in ids.iter().enumerate() {
        let q = p[i];
        // A hanging lock beside a cheek or neck is not a scalp measurement.
        boundary_fade[k] = fade(q);
        active[k] = !frozen.contains(&i)
            && !preserved.contains(&i)
            && q.z > chin + 0.025
            && (q.y > center.y - 0.005 || q.z > eye.z + 0.016);
        if preserved.contains(&i) {
            boundary_fade[k] = 0.;
        }
        if !active[k] {
            continue;
        }
        let radial = q - center;
        let radius = radial.norm();
        if radius < 0.025 {
            active[k] = false;
            continue;
        }
        let dir = radial / radius;
        radii[k] = radius;
        directions[k] = dir;
        let Some((h, n, _)) = hair.ray(center, dir, 0.25) else {
            continue;
        };
        let Some((front, _, _)) = complete.ray(center, dir, 0.25) else {
            continue;
        };
        if (h - front).norm() > 0.0015 || n.dot(&dir).abs() < 0.2 {
            continue;
        }
        let cap = (h - center).norm() - 0.0025;
        if cap < radius * 0.65 {
            rejected += 1;
            continue;
        }
        caps[k] = Some(cap);
        // No pointwise inward obstacle: it can carve a groove into healthy skin.
    }
    let mut target = vec![0.; ids.len()];
    let mut confidence = vec![0.; ids.len()];
    let mut raised = 0;
    let mut positive = 0;
    let mut negative = 0;
    for k in 0..ids.len() {
        if let Some(cap) = caps[k] {
            if preserved.contains(&ids[k]) || boundary_fade[k] == 0. {
                continue;
            }
            let mut local: Vec<_> = (0..ids.len())
                .filter_map(|j| {
                    caps[j]
                        .filter(|_| directions[k].dot(&directions[j]) > 0.9063)
                        .map(|c| c / radii[j])
                })
                .collect();
            target[k] = scalp_target(radii[k], cap, &mut local);
            confidence[k] = 1.;
            raised += usize::from(target[k] < cap - radii[k] - 0.005);
            positive += usize::from(target[k] > 0.);
            negative += usize::from(cap < radii[k]);
        }
    }
    let broad = broad_scalp_fit(&directions, &target, &confidence);
    for k in 0..ids.len() {
        if active[k] {
            target[k] = directions[k].component_mul(&directions[k]).dot(&broad) * boundary_fade[k];
            confidence[k] = 1.;
        }
    }
    let d = scalp_field(&neighbors, &active, &upper, &target, &confidence);
     
    let sample_constraints = 0;
    let sampling_rounds = 0;
    let mut maximum_out = 0f64;
    let mut maximum_in = 0f64;
    for (k, &i) in ids.iter().enumerate() {
        if active[k] {
            p[i] += directions[k] * d[k];
            maximum_out = maximum_out.max(d[k]);
            maximum_in = maximum_in.max(-d[k]);
        }
    }
    let revision_guard = guard_scalp_revision(&before, p, faces);
    maximum_out = 0.;
    maximum_in = 0.;
    for (k, &i) in ids.iter().enumerate() {
        let retained = (p[i] - before[i]).dot(&directions[k]);
        maximum_out = maximum_out.max(retained);
        maximum_in = maximum_in.max(-retained);
    }
    let unresolved_hair_rays = caps
        .iter()
        .enumerate()
        .filter(|(k, c)| c.is_some_and(|cap| (p[ids[*k]] - center).norm() > cap + 0.0005))
        .count();
    serde_json::json!({"stage":"scalp_hair_containment",
        "method":"smooth bidirectional cranial volume from broad hair envelope; detailed hair grooves never become skin obstacles",
        "unresolved_hair_rays":unresolved_hair_rays,"local_hair_indentations_allowed":false,
        "broad_volume_coefficients_mm":[broad.x*1000.,broad.y*1000.,broad.z*1000.],"ear_vertices_excluded_from_growth":preserve.len(),
        "ear_inward_containment_allowed":false,
        "observed_rays":caps.iter().filter(|c|c.is_some()).count(),
        "outward_target_rays":positive,"violated_rays_before":negative,
        "raised_hair_targets_limited":raised,"rejected_extreme_rays":rejected,
        "maximum_scalp_expansion_mm":maximum_out*1000.,"maximum_scalp_contraction_mm":maximum_in*1000.,
        "triangle_sample_constraints":sample_constraints,"triangle_sampling_rounds":sampling_rounds,
        "clearance_m":0.0025,"hair_geometry_changed":false,"facial_landmarks_changed":false,
        "revision_guard":revision_guard,"hair_root_welded":false,"hair_animation_validated":false})
}

/// Identity is solved in source coordinates. Attachment is a distinct result:
/// lower toward the existing body's chin height without raising a short-necked
/// source head to a taller generic prior. Only a translation is applied above
/// the transition, so fitted features and the copied hair move as one assembly.
pub fn assemble(body: &[V], fitted: &[V], blend: &[f64], chin: usize) -> (Vec<V>, V) {
    // A wide/stocky body prior can put its generic chin above the observed
    // character's short neck. Keep the observed height when the prior is higher.
    let delta = V::new(0., 0., (body[chin].z - fitted[chin].z).min(0.));
    let assembled = fitted
        .iter()
        .enumerate()
        .map(|(i, p)| p + delta * blend[i])
        .collect();
    (assembled, delta)
}

 pub fn neck_girth(
    original: &[V],
    points: &mut [V],
    body_ids: &[usize],
    faces: &[[usize; 3]],
    landmarks: &[Landmark],
    regions: &FaceRegions,
    shape: &crate::NeckGirth,
) -> Result<serde_json::Value> {
    ensure!(
        [shape.width_scale, shape.depth_scale]
            .iter()
            .all(|s| s.is_finite() && (0.75..=1.65).contains(s)),
        "Neck girth scales must be finite and within [0.75, 1.65]"
    );
    if shape.width_scale == 1. && shape.depth_scale == 1. {
        return Ok(serde_json::json!({"applied":false,"vertex_ids":[]}));
    }
    let chin = landmarks
        .iter()
        .find(|l| l.name == "chin")
        .context("Missing chin")?;
    let start = chin.original[2] - 0.145;
    let end = chin.original[2] + 0.055;
    let station = chin.original[2] - 0.025;
    let section: Vec<_> = body_ids
        .iter()
        .copied()
        .filter(|&i| {
            (original[i].z - station).abs() < 0.018
                && original[i].x.abs() < 0.10
                && original[i].y > chin.original[1] + 0.025
        })
        .collect();
    ensure!(
        section.len() >= 8,
        "Insufficient neck cross-section for girth prior"
    );
    let mut center = V::zeros();
    for axis in 0..2 {
        let mut values: Vec<_> = section.iter().map(|&i| points[i][axis]).collect();
        values.sort_by(f64::total_cmp);
        center[axis] =
            (values[(values.len() - 1) * 5 / 100] + values[(values.len() - 1) * 95 / 100]) * 0.5;
    }
    let frozen: BTreeSet<_> = landmarks
        .iter()
        .map(|l| l.vertex)
        .chain(regions.cavities.iter().flatten().copied())
        .chain(regions.scalp_preserve.iter().copied())
        .collect();
    let before = points.to_vec();
    let mut changed = BTreeSet::new();
    let mut maximum = 0f64;
    for &i in body_ids {
        let q = original[i];
        if frozen.contains(&i) {
            continue;
        }
        let axial = crate::smooth((q.z - start) / 0.09) * crate::smooth((end - q.z) / 0.055);
        // A compact polynomial has a broad derivative bound: a narrow
        // smoothstep cutoff folded the neck into the upper trapezius.
        let transverse = (1. - (q.x / 0.16).powi(2)).max(0.).powi(2);
        // Fade out before the anterior jaw, retaining the observed facial contour.
        let jaw = crate::smooth((q.z - (chin.original[2] - 0.025)) / 0.025);
        let behind_jaw = crate::smooth((q.y - (chin.original[1] + 0.025)) / 0.035);
        let weight = axial * transverse * (1. - jaw + jaw * behind_jaw);
        if weight <= 0. {
            continue;
        }
        points[i].x += (points[i].x - center.x) * (shape.width_scale - 1.) * weight;
        points[i].y += (points[i].y - center.y) * (shape.depth_scale - 1.) * weight;
        maximum = maximum.max((points[i] - before[i]).norm());
        changed.insert(i);
    }
    let affected: Vec<_> = faces
        .iter()
        .filter(|f| f.iter().any(|i| changed.contains(i)))
        .copied()
        .collect();
    let (reversed, area) = band_quality(&before, points, &affected);
    ensure!(
        reversed == 0 && area >= 0.05,
        "Neck girth prior introduces a fold or collapsed triangle"
    );
    ensure!(
        frozen.iter().all(|&i| points[i] == before[i]),
        "Neck girth changed a facial landmark, cavity, or preserved ear"
    );
    Ok(serde_json::json!({
        "applied":true,"method":"reviewed transverse neck-girth field with smooth root, jaw and shoulder boundaries",
        "width_scale":shape.width_scale,"depth_scale":shape.depth_scale,
        "center_xy_m":[center.x,center.y],"start_z":start,"end_z":end,
        "vertex_ids":changed,"maximum_displacement_mm":maximum*1000.,
        "reversed_triangles":reversed,"minimum_area_ratio":area,
        "head_height_changed":false,"facial_landmarks_changed":false,
        "interpretation":"visual volume prior for hidden anatomy, not a recovered measurement",
        "global_collision_certified":false
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn neutral_neck_girth_is_an_exact_noop() {
        let original = vec![V::new(0.1, 0.2, 1.4)];
        let mut fitted = original.clone();
        neck_girth(
            &original,
            &mut fitted,
            &[0],
            &[],
            &[],
            &FaceRegions::default(),
            &crate::NeckGirth::default(),
        )
        .unwrap();
        assert_eq!(fitted, original);
    }
    #[test]
    fn neck_girth_increases_volume_without_moving_head_face_or_shoulders() {
        let mut original = Vec::new();
        for j in 0..31 {
            for k in 0..32 {
                let a = k as f64 * std::f64::consts::TAU / 32.;
                original.push(V::new(
                    0.07 * a.cos(),
                    0.06 * a.sin(),
                    1.2 + j as f64 * 0.01,
                ));
            }
        }
        let chin = original.len();
        original.push(V::new(0., -0.12, 1.4));
        let face = original.len();
        original.push(V::new(0.04, -0.14, 1.43));
        let shoulder = original.len();
        original.push(V::new(0.2, 0., 1.35));
        let mouth_interior = original.len();
        original.push(V::new(0.025, -0.015, 1.43));
        let mut faces = Vec::new();
        for j in 0..30 {
            for k in 0..32 {
                let a = j * 32 + k;
                let b = j * 32 + (k + 1) % 32;
                faces.extend([[a, b, a + 32], [b, b + 32, a + 32]]);
            }
        }
        let landmarks = vec![Landmark {
            name: "chin".into(),
            vertex: chin,
            original: [0., -0.12, 1.4],
            observed: [0., -0.12, 1.4],
            weight: 1.,
        }];
        let ids = (0..original.len()).collect::<Vec<_>>();
        let mut fitted = original.clone();
        let report = neck_girth(
            &original,
            &mut fitted,
            &ids,
            &faces,
            &landmarks,
            &FaceRegions {
                cavities: vec![vec![mouth_interior]],
                ..FaceRegions::default()
            },
            &crate::NeckGirth {
                width_scale: 1.45,
                depth_scale: 1.3,
            },
        )
        .unwrap();
        assert!(
            (fitted[18 * 32].x
                - original[18 * 32].x
                    * (1. + 0.45 * (1.0_f64 - (0.07_f64 / 0.16).powi(2)).powi(2)))
            .abs()
                < 1e-10
        );
        assert!((fitted[18 * 32 + 8].y - original[18 * 32 + 8].y * 1.3).abs() < 1e-10);
        for i in [chin, face, shoulder, mouth_interior, 0, 30 * 32] {
            assert_eq!(fitted[i], original[i]);
        }
        assert!(fitted.iter().zip(&original).all(|(p, q)| p.z == q.z));
        assert_eq!(report["reversed_triangles"], 0);
    }
    fn neck_tube() -> (Vec<V>, Vec<[usize; 3]>) {
        let mut points = Vec::new();
        for j in 0..15 {
            for k in 0..20 {
                let angle = k as f64 * std::f64::consts::TAU / 20.;
                points.push(V::new(
                    0.1 * angle.cos(),
                    0.12 * angle.sin(),
                    j as f64 * 0.1 - 0.2,
                ));
            }
        }
        let mut faces = Vec::new();
        for j in 0..14 {
            for k in 0..20 {
                let a = j * 20 + k;
                let b = j * 20 + (k + 1) % 20;
                faces.extend([[a, b, a + 20], [b, b + 20, a + 20]]);
            }
        }
        (points, faces)
    }
    #[test]
    fn compressed_neck_stays_ordered_and_keeps_body_and_head_anchors() {
        let (body, faces) = neck_tube();
        let head: Vec<_> = body.iter().map(|p| p - V::z() * 0.75).collect();
        let blend: Vec<_> = body.iter().map(|p| crate::smooth(p.z)).collect();
        let ids: Vec<_> = (0..body.len()).collect();
        let (fitted, report) =
            initial_neck(&body, &body, &head, V::z(), &blend, &ids, &faces, 0., 1.).unwrap();
        assert!(report["reversed_triangles_before"].as_u64().unwrap() > 0);
        assert_eq!(report["reversed_triangles_after"], 0);
        for (i, p) in body.iter().enumerate() {
            if p.z <= 0. {
                assert_eq!(fitted[i], body[i]);
            }
            if p.z >= 1. {
                assert_eq!(fitted[i], head[i]);
            }
            if i >= 20 {
                assert!(fitted[i].z > fitted[i - 20].z);
            }
        }
    }
    #[test]
    fn valid_initial_neck_is_preserved_exactly() {
        let (body, faces) = neck_tube();
        let head: Vec<_> = body.iter().map(|p| p + V::z() * 0.1).collect();
        let blend: Vec<_> = body.iter().map(|p| crate::smooth(p.z)).collect();
        let ids: Vec<_> = (0..body.len()).collect();
        let (fitted, report) =
            initial_neck(&body, &body, &head, V::z(), &blend, &ids, &faces, 0., 1.).unwrap();
        assert_eq!(report["compression_repair_applied"], false);
        for i in ids {
            assert_eq!(fitted[i], body[i] * (1. - blend[i]) + head[i] * blend[i]);
        }
    }
    #[test]
    fn head_transition_keeps_deltoids_at_the_same_height_as_the_neck_fixed() {
        let original = vec![
            V::new(0.05, 0., 1.35),
            V::new(0.20, 0., 1.35),
            V::new(-0.20, 0., 1.35),
            V::new(0.08, 0., 1.45),
        ];
        let blend = neck_blend(&original, 1.32, 1.40);
        assert!(blend[0] > 0.);
        assert_eq!(blend[1], 0.);
        assert_eq!(blend[2], 0.);
        assert_eq!(blend[3], 1.);
        let head: Vec<_> = original.iter().map(|p| p - V::z() * 0.03).collect();
        let (points, _) = initial_neck(
            &original,
            &original,
            &head,
            V::z(),
            &blend,
            &[0, 1, 2, 3],
            &[],
            1.32,
            1.40,
        )
        .unwrap();
        assert_eq!(points[1], original[1]);
        assert_eq!(points[2], original[2]);
        assert_eq!(points[3], head[3]);
    }
    #[test]
    fn neck_root_relaxes_a_pinch_without_moving_anchors_or_chin() {
        let (mut body, faces) = neck_tube();
        for p in &mut body {
            *p *= 0.2;
        }
        let mut points = body.clone();
        for p in &mut points {
            if p.z > 0.08 && p.z < 0.12 {
                p.x *= 0.5;
                p.y *= 0.5;
            }
            if p.z > 0.14 {
                p.z -= 0.01;
            }
        }
        let before = points.clone();
        let vertex = 9 * 20 + 15;
        let landmark = Landmark {
            name: "chin".into(),
            vertex,
            original: crate::arr(body[vertex]),
            observed: crate::arr(points[vertex]),
            weight: 1.,
        };
        let regions = FaceRegions {
            outer: (0..body.len()).collect(),
            ..Default::default()
        };
        let report =
            relax_neck_root(&body, &body, &mut points, &faces, &regions, &[landmark]).unwrap();
        let changed: BTreeSet<_> = report["vertex_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        for i in 0..body.len() {
            if !changed.contains(&i) {
                assert_eq!(points[i], before[i]);
            }
        }
        assert_eq!(points[vertex], before[vertex]);
        let radius = |p: &[V]| p[140..160].iter().map(|v| v.x.hypot(v.y)).sum::<f64>();
        assert!(radius(&points) > radius(&before) * 1.2);
    }
    #[test]
    fn scalp_fills_visible_gaps_without_moving_face_or_crossing_hair() {
        let graph = vec![vec![1], vec![0, 2], vec![1, 3], vec![2, 4], vec![3]];
        let d = scalp_field(
            &graph,
            &[false, true, true, true, true],
            &[0., 0.04, 0.04, -0.01, 0.04],
            &[0., 0.03, 0.03, -0.01, 0.03],
            &[0., 1., 1., 1., 1.],
        );
        assert_eq!(d[0], 0.);
        assert!(d[1] > 0.01 && d[2] > 0.01 && d[4] > 0.01);
        assert!(d[3] <= -0.01);
        assert!(d.iter().all(|x| x.is_finite()));
    }
    #[test]
    fn hair_inside_the_ear_cannot_compress_the_authored_pinna() {
        let mut points = vec![
            V::new(0.1, 0.055, 0.03),
            V::new(0.1, 0.065, 0.03),
            V::new(-0.025, 0., 0.),
            V::new(0.025, 0., 0.),
            V::new(0., 0., -0.1),
            V::new(0., 0., 0.05),
        ];
        let before = points.clone();
        let landmarks: Vec<_> = ["eye_upper_left", "eye_upper_right", "chin", "forehead"]
            .iter()
            .enumerate()
            .map(|(i, name)| Landmark {
                name: (*name).into(),
                vertex: i + 2,
                original: crate::arr(points[i + 2]),
                observed: crate::arr(points[i + 2]),
                weight: 1.,
            })
            .collect();
        let hair = Surface::new(
            &[
                V::new(0.08, -1., -1.),
                V::new(0.08, 1., -1.),
                V::new(0.08, 0., 1.),
            ],
            &[[0, 1, 2]],
        );
        let report = contain_scalp(
            &mut points,
            &[0, 1],
            &[(0, 1)],
            &[],
            &landmarks,
            &[0],
            &hair,
            &hair,
        );
        assert_eq!(points[0], before[0]);
        assert!(points[1].x < 0.09);
        assert_eq!(report["ear_inward_containment_allowed"], false);
        for i in 2..points.len() {
            assert_eq!(points[i], before[i]);
        }
    }
    #[test]
    fn scalp_guard_prevents_ear_folding_without_reverting_distant_volume() {
        let before = vec![
            V::new(-1., 0., 0.),
            V::new(1., 0., 0.),
            V::new(0., 1., 0.),
            V::new(0., 0.2, 0.001),
            V::new(0.2, 0.2, 0.001),
            V::new(0., 0.7, 0.001),
            V::new(3., 3., 3.),
        ];
        let faces = vec![[0, 1, 2], [3, 4, 5]];
        let mut p = before.clone();
        p[3].z = -0.0001;
        p[4].z = 0.0001;
        p[5].z = 0.0001;
        p[6].x += 0.03;
        assert_eq!(Surface::new(&p, &faces).crossings().len(), 1);
        let report = guard_scalp_revision(&before, &mut p, &faces);
        assert_eq!(report["new_crossings"], 0);
        assert_eq!(report["full_proposal_reverted"], false);
        assert_eq!(p[6].x, before[6].x + 0.03);
        assert_eq!(&p[..6], &before[..6]);
    }
    #[test]
    fn raised_hair_does_not_define_the_local_skull_radius() {
        let mut ratios = vec![1.18, 1.20, 1.19, 1.21, 1.22, 1.20, 1.9];
        let d = scalp_target(0.1, 0.19, &mut ratios);
        assert!(d > 0.015 && d < 0.035);
        assert_eq!(scalp_target(0.1, 0.09, &mut ratios), -0.010000000000000009);
    }
    #[test]
    fn broad_cranial_size_rejects_isolated_high_hair_targets() {
        let mut dirs = Vec::new();
        let mut targets = Vec::new();
        let wanted = V::new(0.008, 0.015, 0.020);
        for i in 0..240 {
            let z = 1. - 2. * (i as f64 + 0.5) / 240.;
            let angle = i as f64 * 2.399963229728653;
            let r = (1. - z * z).sqrt();
            let d = V::new(r * angle.cos(), r * angle.sin(), z);
            dirs.push(d);
            targets.push(d.component_mul(&d).dot(&wanted) + if i % 10 == 0 { 0.08 } else { 0. });
        }
        let fitted = broad_scalp_fit(&dirs, &targets, &vec![1.; dirs.len()]);
        assert!((fitted - wanted).norm() < 0.006);
        assert!(fitted.iter().all(|&x| x > 0. && x < 0.03));
    }
    #[test]
    fn attachment_moves_head_and_hair_rigidly_and_preserves_body_anchor() {
        let body = vec![V::zeros(), V::new(0., 0., 1.), V::new(0.1, 0., 1.1)];
        let fitted = vec![body[0], body[1] + V::z() * 0.02, body[2] + V::z() * 0.02];
        let (p, offset) = assemble(&body, &fitted, &[0., 1., 1.], 1);
        assert_eq!(p[0], body[0]);
        assert!((p[1] - body[1]).norm() < 1e-12);
        assert!((p[2] - p[1] - (fitted[2] - fitted[1])).norm() < 1e-12);
        assert!((fitted[2] + offset - p[2]).norm() < 1e-12);
    }
    #[test]
    fn attachment_does_not_lengthen_observed_short_neck_to_stocky_prior() {
        let fitted = vec![
            V::new(0., 0., 1.35),
            V::new(0., -0.1, 1.49),
            V::new(0., 0., 1.70),
        ];
        let mut body = fitted.clone();
        body[1].z = 1.525;
        let (assembled, offset) = assemble(&body, &fitted, &[0., 1., 1.], 1);
        assert_eq!(offset, V::zeros());
        assert_eq!(assembled, fitted);
    }
    #[test]
    fn shell_query_finds_nearest_outward_hit_for_oblique_and_parallel_axes() {
        let points = vec![
            V::new(-2., 1., -2.),
            V::new(2., 1., -2.),
            V::new(0., 1., 2.),
            V::new(-2., 2., -2.),
            V::new(2., 2., -2.),
            V::new(0., 2., 2.),
        ];
        let shell = Surface::new(&points, &[[0, 1, 2], [3, 4, 5]]);
        let (q, _, id) = shell.ray(V::zeros(), V::new(0.1, 1., 0.1), 3.).unwrap();
        assert!((q.y - 1.).abs() < 1e-12);
        assert_eq!(id, 0);
        assert!(shell.ray(V::new(5., 0., 0.), V::y(), 3.).is_none());
        assert!(shell.ray(V::zeros(), -V::y(), 3.).is_none());
    }
}
