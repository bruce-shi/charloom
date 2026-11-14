 use crate::geom::{self, V, Weld, a, v};
use anyhow::{Context, Result, ensure};
use charloom_core::{Mesh, Vec3};
use charloom_engine::surface_query::SurfaceQuery;
use faer::linalg::solvers::Solve;
use faer::sparse::{SparseColMat, Triplet};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

#[derive(Deserialize)]
struct Controls {
    positions: Vec<Vec3>,
    triangles: Vec<[u32; 3]>,
    active: Vec<bool>,
    #[serde(flatten)]
    mode: Mode,
}
#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
enum Mode {
    Fit {
        segmentation: PathBuf,
        smoothness: f64,
        data_weight: f64,
        rounds: usize,
    },
    Repair,
}

 
struct System {
    triplets: Vec<Triplet<usize, usize, f64>>,
    boundary: Vec<V>,
    ids: Vec<usize>,
}
impl System {
    fn new(rest: &[Vec3], adj: &[Vec<u32>], active: &[bool], strength: f64) -> Self {
        let ids: Vec<_> = (0..rest.len()).filter(|&i| active[i]).collect();
        let mut index = vec![usize::MAX; rest.len()];
        for (j, &i) in ids.iter().enumerate() {
            index[i] = j;
        }
        let mut triplets = Vec::new();
        let mut boundary = vec![V::zeros(); ids.len()];
        for (i, neighbours) in adj.iter().enumerate() {
            if !active[i] && !neighbours.iter().any(|&j| active[j as usize]) {
                continue;
            }
            let mut terms = vec![];
            let mut fixed = V::zeros();
            let mut term = |j: usize, w: f64| {
                if active[j] {
                    terms.push((index[j], w));
                } else {
                    fixed += v(rest[j]) * w;
                }
            };
            term(i, 1.);
            for &j in neighbours {
                term(j as usize, -1. / neighbours.len() as f64);
            }
            for &(r, x) in &terms {
                boundary[r] -= fixed * (strength * x);
                for &(c, y) in &terms {
                    if r >= c {
                        triplets.push(Triplet::new(r, c, strength * x * y));
                    }
                }
            }
        }
        Self {
            triplets,
            boundary,
            ids,
        }
    }

    fn solve(&self, positions: &mut [Vec3], targets: &[(f64, V)], step: f64) -> Result<()> {
        let n = self.ids.len();
        if n == 0 {
            return Ok(());
        }
        let mut terms = self.triplets.clone();
        for (i, (w, _)) in targets.iter().enumerate() {
            terms.push(Triplet::new(i, i, *w));
        }
        let mat = SparseColMat::<usize, f64>::try_new_from_triplets(n, n, &terms)
            .context("construct body surface matrix")?;
        let factor = mat
            .sp_cholesky(faer::Side::Lower)
            .context("body surface factorization")?;
        let rhs = faer::Mat::from_fn(n, 3, |r, c| self.boundary[r][c] + targets[r].1[c]);
        let solved = factor.solve(&rhs);
        for (row, &i) in self.ids.iter().enumerate() {
            let old = v(positions[i]);
            let new = V::new(solved[(row, 0)], solved[(row, 1)], solved[(row, 2)]);
            ensure!(new.iter().all(|v| v.is_finite()), "nonfinite body solve");
            positions[i] = a(old + (new - old) * step);
        }
        Ok(())
    }
}

fn scan_queries(path: &Path) -> Result<(SurfaceQuery, SurfaceQuery)> {
    let loaded = charloom_io::load_gltf(path)?;
    let base = crate::source::gltf_of(&loaded)?;
    let mut meshes: Vec<Mesh> = loaded.asset.meshes;
    let positions = meshes
        .iter()
        .map(|m| m.positions.clone())
        .collect::<Vec<_>>();
    let full = SurfaceQuery::new(&meshes, &positions);
    for mesh in &mut meshes {
        let primitive = base
            .nodes()
            .nth(mesh.node)
            .and_then(|n| n.mesh())
            .and_then(|m| m.primitives().nth(mesh.primitive))
            .context("segmentation primitive missing")?;
        let reader = primitive.reader(|_| Some(&loaded.document.binary));
        let skin = reader
            .read_colors(0)
            .context("segmentation requires labelled faces")?
            .into_rgb_f32()
            .map(|c| (v(c.map(f64::from)) - V::new(0.95, 0.55, 0.35)).norm_squared() < 1e-8)
            .collect::<Vec<_>>();
        ensure!(
            skin.len() == mesh.positions.len(),
            "segmentation colour count mismatch"
        );
        mesh.triangles
            .retain(|t| t.iter().all(|&i| skin[i as usize]));
    }
    ensure!(
        meshes.iter().any(|m| !m.triangles.is_empty()),
        "no observed skin in segmentation"
    );
    Ok((SurfaceQuery::new(&meshes, &positions), full))
}

pub fn run(input: &Path, output: &Path) -> Result<Value> {
    let c: Controls = super::read(input)?;
    let count = c.positions.len();
    ensure!(
        count > 0
            && c.active.len() == count
            && c.triangles.iter().flatten().all(|&i| (i as usize) < count),
        "invalid body controls"
    );
    let weld = Weld {
        id: (0..count).collect(),
        count,
        representative: (0..count).collect(),
    };
    let neighbours = geom::adjacency(&c.triangles, &weld);
    let mut x = c.positions.clone();
    let mut cohort = vec![];
    let mut history = vec![];
    match c.mode {
        Mode::Repair => {
            let system = System::new(&x, &neighbours, &c.active, 1.);
            let targets = system
                .ids
                .iter()
                .map(|&i| (0.002, v(x[i]) * 0.002))
                .collect::<Vec<_>>();
            system.solve(&mut x, &targets, 1.)?;
        }
        Mode::Fit {
            segmentation,
            smoothness,
            data_weight,
            rounds,
        } => {
            ensure!(
                smoothness >= 0. && data_weight >= 0.,
                "negative surface constraint weight"
            );
            let (skin, full) = scan_queries(&segmentation)?;
            let hits = |p| -> Result<_> {
                let s = skin
                    .closest_point(p, 100., None)
                    .context("no nearby skin surface")?;
                let f = full
                    .closest_point(p, 100., None)
                    .context("no nearby scan surface")?;
                Ok((s.distance_m < f.distance_m + 0.006, s))
            };
            let system = System::new(&x, &neighbours, &c.active, smoothness);
            let observed = system
                .ids
                .par_iter()
                .map(|&i| Ok(hits(x[i])?.0))
                .collect::<Result<Vec<_>>>()?;
            cohort.extend(
                system
                    .ids
                    .iter()
                    .zip(&observed)
                    .filter(|(_, ok)| **ok)
                    .map(|(&i, _)| i),
            );
            for round in 0..rounds {
                let normals = geom::vertex_normals(&x, &c.triangles, &weld);
                let constraints = system
                    .ids
                    .par_iter()
                    .zip(&observed)
                    .map(|(&i, &seen)| -> Result<_> {
                        let (visible, hit) = hits(x[i])?;
                        let ok = seen
                            && visible
                            && hit.distance_m < 0.12
                            && v(normals[i]).dot(&v(hit.normal)) > 0.1;
                        let w = if ok { data_weight } else { 0. };
                        let anchor = if seen { 0.3 } else { 3. };
                        Ok((
                            ok,
                            (w + anchor, v(hit.point) * w + v(c.positions[i]) * anchor),
                        ))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let (seen, targets): (Vec<_>, Vec<_>) = constraints.into_iter().unzip();
                history
                    .push(json!({"round":round,"observed":seen.iter().filter(|&&ok|ok).count()}));
                system.solve(&mut x, &targets, 0.5)?;
            }
        }
    }
    charloom_io::write_json(
        output,
        &json!({"positions":x,"cohort":cohort,"history":history}),
    )?;
    Ok(json!({"output":output,"vertices":count}))
}
