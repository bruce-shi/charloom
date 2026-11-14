//! Cut the head component on the anatomical prior. Keep the original render
//! vertices at the resulting collar so skin and normals match across UV seams.
use super::read;
use crate::geom::{Weld, adjacency};
use anyhow::{Context, Result, ensure};
use charloom_core::Vec3;
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

#[derive(Deserialize)]
struct Input {
    // Rounded prior coordinates, in the order shared with the UV correspondence.
    prior: Vec<Vec3>,
    representative: Vec<usize>,
    weld: Vec<usize>,
    triangles: Vec<[u32; 3]>,
}

pub fn run(input: &Path, output: &Path) -> Result<Value> {
    let Input {
        prior,
        representative,
        weld,
        triangles,
    } = read(input)?;
    let count = prior.len();
    ensure!(
        count > 0
            && representative.len() == count
            && weld.iter().all(|&i| i < count)
            && representative
                .iter()
                .enumerate()
                .all(|(i, &r)| weld.get(r) == Some(&i))
            && triangles
                .iter()
                .flatten()
                .all(|&v| (v as usize) < weld.len()),
        "invalid anatomical prior correspondence"
    );
    let weld = Weld {
        id: weld,
        count,
        representative,
    };
    let graph = adjacency(&triangles, &weld);
    let crown = (0..count)
        .min_by(|&a, &b| prior[b][1].total_cmp(&prior[a][1]))
        .context("empty prior")?;
    let height = prior[crown][1] * 0.851;
    let mut head = vec![false; count];
    let mut stack = vec![crown];
    head[crown] = true;
    while let Some(vertex) = stack.pop() {
        for &other in &graph[vertex] {
            let other = other as usize;
            if !head[other] && prior[other][1] > height {
                head[other] = true;
                stack.push(other);
            }
        }
    }
    let mut keep = Vec::with_capacity(triangles.len());
    // Keep directed edges and sorted undirected keys. The first boundary
    // edge fixes the chart's start vertex and winding for the shell loft.
    let mut edges = BTreeMap::<(usize, usize), (usize, usize, u32)>::new();
    for triangle in triangles {
        let ids = triangle.map(|i| weld.id[i as usize]);
        let body = !ids.iter().all(|&i| head[i]);
        keep.push(body);
        if body {
            for (a, b) in [(ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])] {
                if head[a] && head[b] {
                    edges.entry((a.min(b), a.max(b))).or_insert((a, b, 0)).2 += 1;
                }
            }
        }
    }
    let mut next = BTreeMap::new();
    let mut start = None;
    for (_, (a, b, uses)) in edges {
        if uses == 1 {
            start.get_or_insert(a);
            ensure!(next.insert(a, b).is_none(), "branch in neck boundary");
        }
    }
    let mut vertex = start.context("missing neck boundary")?;
    let start = vertex;
    let mut ring = Vec::new();
    loop {
        ring.push(weld.representative[vertex]);
        vertex = next.remove(&vertex).context("open neck boundary")?;
        if vertex == start {
            break;
        }
    }
    ensure!(
        ring.len() > 20 && next.is_empty(),
        "ambiguous neck boundary"
    );
    charloom_io::write_json(output, &json!({"keep":keep,"collar":ring}))?;
    Ok(json!({"output":output,"collar_vertices":ring.len()}))
}
