//! Limit a proposed outer-skin edit at new crossings, collapses or reversals.
use crate::{
    smooth,
    surface::{Surface, V},
};
use serde_json::{Value, json};
use std::collections::{BTreeSet, VecDeque};

 pub fn limit_update(
    reference: &[V],
    before: &[V],
    points: &mut [V],
    faces: &[[usize; 3]],
    graph: &[BTreeSet<usize>],
) -> Value {
    let prior = Surface::new(before, faces).crossings();
    let proposal = points.to_vec();
    let delta: Vec<_> = proposal.iter().zip(before).map(|(a, b)| a - b).collect();
    let mut anchors = BTreeSet::new();
    let mut fallback = false;
    let mut history = Vec::new();
    for iteration in 0..=16 {
        let current = Surface::new(points, faces).crossings();
        let added: Vec<_> = current.difference(&prior).copied().collect();
        let invalid: Vec<_> = faces
            .iter()
            .enumerate()
            .filter_map(|(k, &[a, b, c])| {
                if [a, b, c].iter().all(|&i| delta[i].norm() < 1e-10) {
                    return None;
                }
                let rest = (reference[b] - reference[a]).cross(&(reference[c] - reference[a]));
                let old = (before[b] - before[a]).cross(&(before[c] - before[a]));
                let new = (points[b] - points[a]).cross(&(points[c] - points[a]));
                ((old.dot(&rest) > 0. && new.dot(&rest) <= 0.)
                    || (old.norm() >= rest.norm() * 0.05 && new.norm() < rest.norm() * 0.05))
                    .then_some(k)
            })
            .collect();
        history.push(json!({"iteration":iteration,"new_crossings":added.len(),"new_reversed_or_collapsed_triangles":invalid.len()}));
        if added.is_empty() && invalid.is_empty() {
            break;
        }
        if iteration == 16 {
            points.copy_from_slice(before);
            fallback = true;
            break;
        }
        for &f in added.iter().flatten().chain(invalid.iter()) {
            for &i in &faces[f] {
                if delta[i].norm() > 1e-10 {
                    anchors.insert(i);
                }
            }
        }
        let mut distance = vec![f64::INFINITY; points.len()];
        let mut queue = VecDeque::new();
        for &i in &anchors {
            distance[i] = 0.;
            queue.push_back(i);
        }
        while let Some(i) = queue.pop_front() {
            for &j in &graph[i] {
                let next = distance[i] + (reference[i] - reference[j]).norm();
                if next < 0.012 && next + 1e-12 < distance[j] {
                    distance[j] = next;
                    queue.push_back(j);
                }
            }
        }
        for i in 0..points.len() {
            points[i] = before[i] + delta[i] * smooth(distance[i] / 0.012);
        }
    }
    json!({"new_crossings":Surface::new(points,faces).crossings().difference(&prior).count(),
        "full_proposal_reverted":fallback,"iterations":history,
        "scope":"outer skin relative to the pre-transition fit; existing feature and hair contacts remain separate"})
}
