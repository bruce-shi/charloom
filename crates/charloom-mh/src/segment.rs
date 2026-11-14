//! Stage 1b — clothing segmentation and extraction.
//!
//! Each scan vertex is related to the *posed* MakeHuman body: its distance to
//! the body surface, the body region it lies over (from the rig weights) and
//! its texture colour. Skin is surface that hugs the body *and* matches the
//! skin colour sampled from the face; the face itself is always skin. The
//! remaining surface is hair (over the head) or garments, grouped into pieces
//! by body region. Explicit override boxes take precedence.
use crate::geom::{V, adjacency, percentiles, v};
use crate::prior::{Prior, Region};
use crate::scan::{Scan, build_query};
use crate::source::lab;
use charloom_core::Vec3;
use rayon::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

/// Body normals pointing down past this (under the jaw and chin) count as
/// face for the skin colour test.
const JAW_UNDERSIDE: f64 = 0.3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Label {
    Skin,
    Hair,
    Garment,
}

/// Axis-aligned box in the working frame (Y-up metres, feet at 0, facing +Z).
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct LabelOverride {
    pub label: Label,
    pub min: Vec3,
    pub max: Vec3,
}

pub struct Segmentation {
    pub vertex_label: Vec<Label>,
    pub triangle_label: Vec<Label>,
    /// Garment/hair piece of every non-skin triangle.
    pub triangle_piece: Vec<Option<usize>>,
    pub pieces: Vec<String>,
    /// Per piece: hair, layer (0 = on the skin) and tight fit, from guidance.
    pub piece_hair: Vec<bool>,
    pub piece_layer: Vec<u32>,
    pub piece_tight: Vec<bool>,
    /// Per piece: the guidance's skin-to-garment gap estimate, mm.
    pub piece_gap_mm: Vec<Option<f64>>,
    /// Per piece: a rigid part (claw, horn, tusk) bound to one bone.
    pub piece_rigid: Vec<bool>,
    /// Per scan triangle: a skin-coloured strand relabelled hair (strands.rs).
    pub strand: Vec<bool>,
    pub report: Value,
}

impl Segmentation {
    /// Scan triangle `t` belongs to a rigid piece (a prop such as a quiver
    /// or bow, a claw, a horn): it does not wrap the body, so it never bounds
    /// or shapes it.
    pub fn is_prop(&self, t: usize) -> bool {
        self.triangle_piece
            .get(t)
            .copied()
            .flatten()
            .is_some_and(|p| self.piece_rigid.get(p) == Some(&true))
    }
}

fn piece_name(label: Label, region: Region) -> &'static str {
    match (label, region) {
        (Label::Hair, _) | (_, Region::Head) => "hair",
        (_, Region::Torso | Region::Arm) => "top",
        (_, Region::Leg) => "bottom",
        (_, Region::Foot) => "shoes",
        (_, Region::Hand) => "gloves",
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs.get(xs.len() / 2).copied().unwrap_or(0.0)
}

pub fn segment(
    prior: &Prior,
    scan: &Scan,
    posed_body: &[Vec3],
    skin_delta_e: f64,
    overrides: &[LabelOverride],
    guide: Option<&crate::guide::Resolved>,
) -> Segmentation {
    let height = prior.height();
    // Vision guidance: per welded vertex (label, strong). Strong seeds
    // override geometry/colour: deep inside an LLM polygon (≥ 8 px), on a
    // stroke, matching the item's colour nearer the polygon edge, or grown
    // from the polygon over connected same-coloured surface up to the item's
    // edge (see `guide::resolve`). Weak seeds (off-colour, near a polygon
    // edge) only choose between hair and garment: an LLM face polygon drawn
    // up to the hair otherwise forces loose strands in front of the cheeks
    // to skin, and the fit then pulls the face onto them.
    let mut seed: Vec<Option<(Label, bool)>> = vec![None; scan.weld.count];
    if let Some(g) = guide {
        for (t, tri) in scan.triangles.iter().enumerate() {
            let strong = g.seed_strong[t];
            let entry = match (g.garment_seed[t], g.skin_seed[t]) {
                (Some((gi, _)), _) => Some((g.garments[gi].label, strong)),
                (None, Some(_)) => Some((Label::Skin, strong)),
                _ => None,
            };
            if let Some(e) = entry {
                for &i in tri {
                    let slot = &mut seed[scan.weld.id[i as usize]];
                    if slot.is_none_or(|(_, strong)| e.1 && !strong) {
                        *slot = Some(e);
                    }
                }
            }
        }
    }
    let apply_seeds = |labels: &mut [Label], strong_only: bool| {
        for (l, s) in labels.iter_mut().zip(&seed) {
            match s {
                Some((label, true)) => *l = *label,
                Some((label, false))
                    if !strong_only && *l != Label::Skin && *label != Label::Skin =>
                {
                    *l = *label
                }
                _ => {}
            }
        }
    };
    let body = prior.body();
    let body_query = build_query(posed_body, &body.prim.triangles);
    let count = scan.weld.count;
    // One incident triangle per welded vertex, for colour lookup.
    let mut incident = vec![usize::MAX; scan.positions.len()];
    for (t, tri) in scan.triangles.iter().enumerate() {
        for &i in tri {
            if incident[i as usize] == usize::MAX {
                incident[i as usize] = t;
            }
        }
    }
    struct Obs {
        distance: f64,
        region: Region,
        front: f64,
        /// Downward component of the body normal (under the jaw > 0).
        down: f64,
        lab: [f64; 3],
        neck: bool,
    }
    let observations: Vec<Option<Obs>> = (0..count)
        .into_par_iter()
        .map(|w| {
            let r = scan.weld.representative[w];
            let hit = body_query.closest_point(scan.positions[r], height * 0.25, None)?;
            let tri = body.prim.triangles[hit.source[1]];
            let k = (0..3).max_by(|&x, &y| hit.barycentric[x].total_cmp(&hit.barycentric[y]))?;
            let joint = prior.dominant_joint(prior.body, tri[k] as usize);
            let region = prior.region[joint].unwrap_or(Region::Torso);
            let color = scan.vertex_color(r, incident[r]).unwrap_or([0.5; 3]);
            Some(Obs {
                distance: hit.distance_m,
                region,
                front: hit.normal[2],
                down: -hit.normal[1],
                lab: lab(color),
                neck: prior.skeleton().joints[joint].name.starts_with("neck"),
            })
        })
        .collect();

    // Skin colour model from the face.
    let near = height * 0.012;
     
    let guided_skin: Vec<[f64; 3]> = observations
        .iter()
        .zip(&seed)
        .filter_map(|(o, s)| {
            o.as_ref()
                .filter(|_| matches!(s, Some((Label::Skin, true))))
                .map(|o| o.lab)
        })
        .filter(|c| c[0] > 20.0)
        .collect();
    let candidates: Vec<[f64; 3]> = if guided_skin.len() >= 200 {
        guided_skin
    } else {
        observations
            .iter()
            .flatten()
            .filter(|o| {
                o.distance < near
                    && ((o.region == Region::Head && o.front > 0.6) || o.region == Region::Hand)
            })
            .map(|o| o.lab)
            .filter(|c| c[0] > 20.0)
            .collect()
    };
    let distance = |c: &[f64; 3], m: &[f64; 3]| {
        ((c[0] - m[0]).powi(2) * 0.5 + (c[1] - m[1]).powi(2) + (c[2] - m[2]).powi(2)).sqrt()
    };
    let mut skin: [f64; 3] =
        std::array::from_fn(|k| median(candidates.iter().map(|c| c[k]).collect()));
    let mut seeds = candidates.clone();
    for radius in [25.0, 18.0, 12.0] {
        let kept: Vec<[f64; 3]> = candidates
            .iter()
            .filter(|c| distance(c, &skin) < radius)
            .copied()
            .collect();
        if kept.len() < 50 {
            break;
        }
        skin = std::array::from_fn(|k| median(kept.iter().map(|c| c[k]).collect()));
        seeds = kept;
    }
    let de = |c: &[f64; 3]| distance(c, &skin);
    // Face seeds include brows, lips and eyes; only the typical spread of the
    // seed colours may widen the configured threshold, and only slightly.
    let spread = median(seeds.iter().map(de).collect());
    let threshold = skin_delta_e.max((1.5 * spread).min(skin_delta_e * 1.4));

    let mut labels: Vec<Label> = observations
        .iter()
        .map(|o| match o {
            None => Label::Garment,
             
            Some(o)
                if o.region == Region::Head
                    && (o.front > 0.5 || o.down > JAW_UNDERSIDE)
                    && o.distance < height * 0.04
                    && de(&o.lab) < threshold * 1.8
                    && o.lab[0] > skin[0] * 0.55 =>
            {
                Label::Skin
            }
            // Palms are lighter than the face; allow a wider tolerance on bare hands.
            Some(o)
                if o.region == Region::Hand
                    && o.distance < height * 0.02
                    && de(&o.lab) < threshold * 2.0 =>
            {
                Label::Skin
            }
            Some(o)
                if o.distance < height * 0.025 && de(&o.lab) < threshold && !seeds.is_empty() =>
            {
                Label::Skin
            }
            Some(o) if o.region == Region::Head => Label::Hair,
            Some(_) => Label::Garment,
        })
        .collect();
    // Hair continues below the skull onto the nape and, when long, the
    // shoulders: non-skin surface over the neck that matches the head hair
    // colour is hair, not a collar.
    let hair_samples: Vec<[f64; 3]> = observations
        .iter()
        .zip(&labels)
        .filter_map(|(o, l)| o.as_ref().filter(|_| *l == Label::Hair).map(|o| o.lab))
        .collect();
    let mut nape_hair = 0usize;
    if hair_samples.len() > 100 {
        let hair: [f64; 3] =
            std::array::from_fn(|k| median(hair_samples.iter().map(|c| c[k]).collect()));
        let de_hair = |c: &[f64; 3]| {
            ((c[0] - hair[0]).powi(2) * 0.5 + (c[1] - hair[1]).powi(2) + (c[2] - hair[2]).powi(2))
                .sqrt()
        };
        for (o, l) in observations.iter().zip(labels.iter_mut()) {
            if let Some(o) = o
                && *l == Label::Garment
                && o.neck
                && de_hair(&o.lab) < threshold
                && de_hair(&o.lab) < de(&o.lab)
            {
                *l = Label::Hair;
                nape_hair += 1;
            }
        }
    }
    apply_seeds(&mut labels, false);
    // Majority filter on the welded surface removes speckle at texture noise.
    let adj = adjacency(&scan.triangles, &scan.weld);
    for _ in 0..3 {
        labels = (0..count)
            .into_par_iter()
            .map(|w| {
                let mut votes = [0usize; 3];
                votes[labels[w] as usize] += 2;
                for &n in &adj[w] {
                    votes[labels[n as usize] as usize] += 1;
                }
                let best = (0..3).max_by_key(|&k| votes[k]).unwrap();
                [Label::Skin, Label::Hair, Label::Garment][best]
            })
            .collect();
    }
    apply_seeds(&mut labels, true);
    let islands = absorb_small_islands(&mut labels, &adj, (count / 400).max(50));
    apply_seeds(&mut labels, true);
    let mut overridden = 0usize;
    for (w, label) in labels.iter_mut().enumerate() {
        let p = scan.positions[scan.weld.representative[w]];
        if let Some(o) = overrides
            .iter()
            .rev()
            .find(|o| (0..3).all(|k| p[k] >= o.min[k] && p[k] <= o.max[k]))
        {
            *label = o.label;
            overridden += 1;
        }
    }
    let mut vertex_label: Vec<Label> = scan.weld.id.iter().map(|&w| labels[w]).collect();

    let mut pieces: Vec<String> = guide.map_or(vec![], |g| {
        g.garments.iter().map(|x| x.name.clone()).collect()
    });
    let mut piece_hair: Vec<bool> = guide.map_or(vec![], |g| {
        g.garments.iter().map(|x| x.label == Label::Hair).collect()
    });
    let mut piece_layer: Vec<u32> =
        guide.map_or(vec![], |g| g.garments.iter().map(|x| x.layer).collect());
    let mut piece_tight: Vec<bool> =
        guide.map_or(vec![], |g| g.garments.iter().map(|x| x.tight).collect());
    let mut piece_rigid: Vec<bool> =
        guide.map_or(vec![], |g| g.garments.iter().map(|x| x.rigid).collect());
    let mut piece_gap_mm: Vec<Option<f64>> =
        guide.map_or(vec![], |g| g.garments.iter().map(|x| x.gap_mm).collect());
    let mut triangle_piece = vec![None; scan.triangles.len()];
    let mut triangle_label: Vec<Label> = scan
        .triangles
        .iter()
        .map(|t| {
            let l = t.map(|i| vertex_label[i as usize]);
            if l[0] == l[1] || l[0] == l[2] {
                l[0]
            } else if l[1] == l[2] {
                l[1]
            } else {
                Label::Garment
            }
        })
        .collect();
    // Skin-coloured hair strands (strands.rs): thin layers near the head that
    // hang from the hair or float as loose shells become hair; the guided
    // floods below then give them the hair piece they connect to.
    let ears = ear_zone(prior, posed_body, &body_query, scan, height);
    let strands = find_strands(
        scan,
        &observations
            .iter()
            .map(|o| o.as_ref().map(|o| (o.region, o.lab)))
            .collect::<Vec<_>>(),
        StrandContext {
            height,
            skin,
            threshold,
            ears: &ears,
        },
        guide,
        &mut triangle_label,
        &mut vertex_label,
    );
    // Guided pieces: seeded triangles keep the LLM's garment; the rest of
    // each connected non-skin surface is flooded from the nearest seed.
    let mut flooded = 0usize;
    let (mut leftover_skin, mut leftover_joined) = (0usize, 0usize);
    let mut hidden_skin = 0usize;
    let mut strand_joined = 0usize;
    let mut large_leftovers: Vec<Value> = vec![];
    let tri_lab: Vec<[f64; 3]> = scan
        .triangles
        .par_iter()
        .map(|t| {
            let mut c = [0.0; 3];
            for &i in t {
                if let Some(o) = &observations[scan.weld.id[i as usize]] {
                    for (acc, l) in c.iter_mut().zip(o.lab) {
                        *acc += l / 3.0;
                    }
                }
            }
            c
        })
        .collect();
    if let Some(g) = guide {
        let mut start = vec![0u32; scan.weld.count + 1];
        for tri in &scan.triangles {
            for &i in tri {
                start[scan.weld.id[i as usize] + 1] += 1;
            }
        }
        for k in 0..scan.weld.count {
            start[k + 1] += start[k];
        }
        let mut fill = start.clone();
        let mut incident = vec![0u32; start[scan.weld.count] as usize];
        for (t, tri) in scan.triangles.iter().enumerate() {
            for &i in tri {
                let w = scan.weld.id[i as usize];
                incident[fill[w] as usize] = t as u32;
                fill[w] += 1;
            }
        }
         
        let mut cost = vec![f64::INFINITY; scan.triangles.len()];
        let mut heap = std::collections::BinaryHeap::new();
        // Only the drawn polygons/strokes start the floods: grown seeds fix
        // the skin/garment/hair label up to the item's edge, but which of two
        // same-coloured neighbouring garments (navy sash, navy trousers) owns
        // the surface between them stays with the colour-aware flood.
        for (t, s) in g.garment_seed.iter().enumerate() {
            if let Some((gi, _)) = s
                && !g.seed_grown[t]
                && triangle_label[t] != Label::Skin
            {
                triangle_piece[t] = Some(*gi);
                cost[t] = 0.0;
                heap.push((std::cmp::Reverse(Ordered(0.0)), t));
            }
        }
        while let Some((std::cmp::Reverse(Ordered(c)), t)) = heap.pop() {
            if c > cost[t] {
                continue;
            }
            let gi = triangle_piece[t];
            for &i in &scan.triangles[t] {
                let w = scan.weld.id[i as usize];
                for &u in &incident[start[w] as usize..start[w + 1] as usize] {
                    let u = u as usize;
                    if triangle_label[u] == Label::Skin {
                        continue;
                    }
                    let change = distance(&tri_lab[t], &tri_lab[u]);
                     
                    if change > 15.0 && gi.is_some_and(|p| g.garments[p].rigid) {
                        continue;
                    }
                     
                    let other = gi.is_some_and(|p| {
                        (g.garments[p].label == Label::Hair) != (triangle_label[u] == Label::Hair)
                    });
                    let step = 1.0 + 0.2 * change + if other { LABEL_CROSSING_COST } else { 0.0 };
                    if c + step < cost[u] {
                        if triangle_piece[u].is_none() {
                            flooded += 1;
                        }
                        cost[u] = c + step;
                        triangle_piece[u] = gi;
                        heap.push((std::cmp::Reverse(Ordered(c + step)), u));
                    }
                }
            }
        }
        // Leftover non-skin surface the floods did not reach (thin bare skin
        // at hems, fingertips, ear rims; small separate shells): skin when its
        // colour matches the skin model, else the nearest named garment, not
        // an extra piece named after the body region.
        let centroid = |t: usize| {
            scan.triangles[t]
                .iter()
                .fold(V::zeros(), |s, &i| s + v(scan.positions[i as usize]))
                / 3.0
        };
        let assigned: Vec<(V, usize)> = triangle_piece
            .iter()
            .enumerate()
            .step_by(4)
            .filter_map(|(t, p)| p.map(|gi| (centroid(t), gi)))
            .collect();
        // Welded vertices on skin triangles (before the leftovers).
        let mut on_skin = vec![false; scan.weld.count];
        for (t, tri) in scan.triangles.iter().enumerate() {
            if triangle_label[t] == Label::Skin {
                tri.iter()
                    .for_each(|&i| on_skin[scan.weld.id[i as usize]] = true);
            }
        }
        let c0 = |component: &[usize]| {
            component.iter().fold(V::zeros(), |s, &t| s + centroid(t)) / component.len() as f64
        };
        let mut seen = vec![false; scan.triangles.len()];
        for t0 in 0..scan.triangles.len() {
            if seen[t0] || triangle_label[t0] == Label::Skin || triangle_piece[t0].is_some() {
                continue;
            }
            let mut component = vec![t0];
            seen[t0] = true;
            let mut k = 0;
            while k < component.len() {
                let t = component[k];
                k += 1;
                for &i in &scan.triangles[t] {
                    let w = scan.weld.id[i as usize];
                    for &u in &incident[start[w] as usize..start[w + 1] as usize] {
                        let u = u as usize;
                        if !seen[u]
                            && triangle_label[u] != Label::Skin
                            && triangle_piece[u].is_none()
                        {
                            seen[u] = true;
                            component.push(u);
                        }
                    }
                }
            }
            let mean: [f64; 3] = std::array::from_fn(|j| {
                component.iter().map(|&t| tri_lab[t][j]).sum::<f64>() / component.len() as f64
            });
            // A large patch embedded in skin that no view shows (a thigh
            // behind a tail) or that lies on the body (fur off the face
            // colour between body parts): body surface, not cloth to hand to
            // the nearest garment. Every garment is marked, so a large
            // unmarked patch touching skin is not one.
            let seen_count = component.iter().filter(|&&t| g.seen[t]).count();
            let touches_skin = component.iter().any(|&t| {
                scan.triangles[t]
                    .iter()
                    .any(|&i| on_skin[scan.weld.id[i as usize]])
            });
            let on_body = median(
                component
                    .iter()
                    .filter_map(|&t| {
                        observations[scan.weld.id[scan.triangles[t][0] as usize]]
                            .as_ref()
                            .map(|o| o.distance)
                    })
                    .collect(),
            );
            let hidden = component.len() >= 200
                && touches_skin
                && (seen_count * 20 < component.len() || on_body < height * 0.02);
            if hidden {
                hidden_skin += component.len();
            }
            if component.len() >= 200 {
                large_leftovers.push(json!({"triangles": component.len(), "seen_triangles": seen_count, "touches_skin": touches_skin, "skin_delta_e": de(&mean), "median_body_distance_m": on_body, "to_skin": hidden}));
            }
            // A loose strand shell (strands.rs) joins the nearest hair piece
            // although it is skin-coloured.
            let strand_count = component.iter().filter(|&&t| strands.strand[t]).count();
            if strand_count * 2 > component.len()
                && let Some(&(_, gi)) = assigned
                    .iter()
                    .filter(|(_, gi)| {
                        g.garments[*gi].label == Label::Hair && !g.garments[*gi].rigid
                    })
                    .min_by(|a, b| {
                        (a.0 - c0(&component))
                            .norm_squared()
                            .total_cmp(&(b.0 - c0(&component)).norm_squared())
                    })
            {
                strand_joined += component.len();
                for &t in &component {
                    triangle_piece[t] = Some(gi);
                }
                continue;
            }
            if hidden || de(&mean) < threshold * 1.6 {
                leftover_skin += component.len();
                for &t in &component {
                    triangle_label[t] = Label::Skin;
                    for &i in &scan.triangles[t] {
                        vertex_label[i as usize] = Label::Skin;
                    }
                }
                continue;
            }
            let c =
                component.iter().fold(V::zeros(), |s, &t| s + centroid(t)) / component.len() as f64;
            if let Some(&(_, gi)) = assigned.iter().min_by(|a, b| {
                (a.0 - c)
                    .norm_squared()
                    .total_cmp(&(b.0 - c).norm_squared())
            }) {
                leftover_joined += component.len();
                for &t in &component {
                    triangle_piece[t] = Some(gi);
                }
            }
        }
        for (t, p) in triangle_piece.iter().enumerate() {
            if let Some(gi) = p {
                triangle_label[t] = g.garments[*gi].label;
            }
        }
    }
    for (t, tri) in scan.triangles.iter().enumerate() {
        if triangle_label[t] == Label::Skin || triangle_piece[t].is_some() {
            continue;
        }
        let regions: Vec<Region> = tri
            .iter()
            .filter_map(|&i| {
                observations[scan.weld.id[i as usize]]
                    .as_ref()
                    .map(|o| o.region)
            })
            .collect();
        let region = regions
            .iter()
            .max_by_key(|r| regions.iter().filter(|x| x == r).count())
            .copied()
            .unwrap_or(Region::Torso);
        let name = piece_name(triangle_label[t], region);
        let index = pieces.iter().position(|p| p == name).unwrap_or_else(|| {
            pieces.push(name.to_string());
            piece_hair.push(name == "hair");
            piece_layer.push(0);
            piece_tight.push(false);
            piece_rigid.push(false);
            piece_gap_mm.push(None);
            pieces.len() - 1
        });
        triangle_piece[t] = Some(index);
    }
    let neckline = refine_neckline(
        scan,
        &observations
            .iter()
            .map(|o| o.as_ref().is_some_and(|o| o.neck))
            .collect::<Vec<_>>(),
        &tri_lab,
        skin,
        height,
        guide,
        &piece_hair,
        &piece_rigid,
        &mut triangle_label,
        &mut triangle_piece,
        &mut vertex_label,
    );
    let garment_islands = if let Some(g) = guide {
        let ordinary: Vec<_> = (0..pieces.len()).map(|p| !piece_hair[p] && !piece_rigid[p]).collect();
        let skin: Vec<_> = triangle_label.iter().map(|l| *l == Label::Skin).collect();
        let marked: Vec<_> = (0..scan.triangles.len()).map(|t| {
            g.garment_seed[t].is_some_and(|(p, _)| triangle_piece[t] == Some(p))
                && g.seed_strong[t] && !g.seed_grown[t]
        }).collect();
        let (changes, report) = crate::piece_islands::refine(&crate::piece_islands::Input {
            positions: &scan.positions, triangles: &scan.triangles,
            weld_id: &scan.weld.id, weld_count: scan.weld.count,
            piece: &triangle_piece, skin: &skin, ordinary_cloth: &ordinary, marked: &marked,
        });
        let mut touched = vec![false; scan.weld.count];
        for (t, p) in changes {
            triangle_piece[t] = p;
            triangle_label[t] = match p {
                None => Label::Skin,
                Some(p) if piece_hair[p] => Label::Hair,
                Some(_) => Label::Garment,
            };
            for &i in &scan.triangles[t] {
                touched[scan.weld.id[i as usize]] = true;
            }
        }
        // Rebuild the vertex view of the same partition. A boundary vertex
        // stays non-skin while any incident garment/hair triangle uses it.
        let mut around = vec![[false; 3]; scan.weld.count];
        for (t, tri) in scan.triangles.iter().enumerate() {
            for &i in tri {
                around[scan.weld.id[i as usize]][triangle_label[t] as usize] = true;
            }
        }
        for (i, label) in vertex_label.iter_mut().enumerate() {
            if !touched[scan.weld.id[i]] {
                continue;
            }
            *label = match around[scan.weld.id[i]] {
                [_, _, true] => Label::Garment,
                [_, true, false] => Label::Hair,
                _ => Label::Skin,
            };
        }
        report
    } else {
        Value::Null
    };
    let count_of = |l: Label| triangle_label.iter().filter(|x| **x == l).count();
    let piece_counts: Vec<Value> = pieces
        .iter()
        .enumerate()
        .map(|(i, n)| json!({"piece":n,"triangles":triangle_piece.iter().filter(|p| **p == Some(i)).count()}))
        .collect();
    let distances: Vec<f64> = observations.iter().flatten().map(|o| o.distance).collect();
    let report = json!({
        "skin_lab_median": skin,
        "skin_seed_vertices": seeds.len(),
        "skin_candidate_vertices": candidates.len(),
        "skin_delta_e_threshold": threshold,
        "triangles": {"skin":count_of(Label::Skin),"hair":count_of(Label::Hair),"garment":count_of(Label::Garment)},
        "pieces": piece_counts,
        "override_vertices": overridden,
        "absorbed_island_vertices": islands,
        "nape_hair_vertices": nape_hair,
        "guided": guide.is_some(),
        "guided_seed_vertices": seed.iter().filter(|s| s.is_some()).count(),
        "guided_strong_seed_vertices": seed.iter().filter(|s| matches!(s, Some((_, true)))).count(),
        "guided_grown_seed_triangles": guide.map_or(0, |g| g.seed_grown.iter().filter(|x| **x).count()),
        "flooded_triangles": flooded,
        "neckline": neckline,
        "garment_islands": garment_islands,
        "leftover_triangles": {"to_skin": leftover_skin, "hidden_to_skin": hidden_skin, "joined_nearest_garment": leftover_joined, "strands_joined_nearest_hair": strand_joined, "large": large_leftovers},
        "strands": &strands.report,
        "scan_to_posed_body_distance_m": percentiles(distances),
    });
    Segmentation {
        vertex_label,
        triangle_label,
        triangle_piece,
        pieces,
        piece_hair,
        piece_layer,
        piece_tight,
        piece_gap_mm,
        piece_rigid,
        strand: strands.strand,
        report,
    }
}

pub struct Strands {
    /// Per scan triangle: skin relabelled hair as a strand.
    pub strand: Vec<bool>,
    pub report: Value,
}

struct StrandContext<'a> {
    height: f64,
    /// Skin colour model (CIELAB median) and its ΔE threshold.
    skin: [f64; 3],
    threshold: f64,
    /// Per scan triangle: within reach of MakeHuman's pinna ([`ear_zone`]).
    ears: &'a [bool],
}

/// Scan triangles near MakeHuman's ears on the posed body. MakeHuman's rig
/// has no ear bone; the pinnae are its thin head shells on the sides of the
/// head (a ray into the surface leaves it within ~1.2 cm), as in
/// `fit::ears`. Scan surface within ~2.5 cm of them is the ear zone.
fn ear_zone(
    prior: &Prior,
    posed_body: &[Vec3],
    body_query: &charloom_engine::surface_query::SurfaceQuery,
    scan: &Scan,
    height: f64,
) -> Vec<bool> {
    let unit = height / 1.75;
    let body = prior.body();
    let weld = crate::geom::Weld::new(posed_body, 1e-7);
    let normals = crate::geom::vertex_normals(posed_body, &body.prim.triangles, &weld);
    let head: Vec<usize> = (0..posed_body.len())
        .filter(|&i| prior.region[prior.dominant_joint(prior.body, i)] == Some(Region::Head))
        .collect();
    if head.is_empty() {
        return vec![false; scan.triangles.len()];
    }
    let cx = head.iter().map(|&i| posed_body[i][0]).sum::<f64>() / head.len() as f64;
    let pinna: Vec<V> = head
        .par_iter()
        .filter(|&&i| (posed_body[i][0] - cx).abs() > 0.045 * unit)
        .filter(|&&i| {
            let n = v(normals[i]);
            let p = v(posed_body[i]) - n * 2e-5;
            body_query
                .outward_ray_distance(crate::geom::a(p), crate::geom::a(-n), 0.012 * unit)
                .is_some()
        })
        .map(|&i| v(posed_body[i]))
        .collect();
    if pinna.is_empty() {
        return vec![false; scan.triangles.len()];
    }
    let (lo, hi) = pinna.iter().fold(
        (V::repeat(f64::INFINITY), V::repeat(f64::NEG_INFINITY)),
        |(l, h), p| (l.inf(p), h.sup(p)),
    );
    let reach = 0.025 * unit;
    scan.triangles
        .par_iter()
        .map(|t| {
            let c = t
                .iter()
                .fold(V::zeros(), |s, &i| s + v(scan.positions[i as usize]))
                / 3.0;
            (0..3).all(|k| c[k] > lo[k] - reach && c[k] < hi[k] + reach)
                && pinna.iter().any(|p| (p - c).norm_squared() < reach * reach)
        })
        .collect()
}

/// Skin-coloured hair strands (see [`crate::strands`]): skin triangles near
/// the head, from the chin down to about the shoulders, are probed; accepted
/// strands are relabelled hair. Skin strokes of the guidance are kept. Then
/// the reverse on the ears: hair-labelled ear surface (a shaded rim darker
/// than the skin model) that is continuous with the ear skin, not hanging
/// from the hair, is skin again.
fn find_strands(
    scan: &Scan,
    observations: &[Option<(Region, [f64; 3])>],
    context: StrandContext,
    guide: Option<&crate::guide::Resolved>,
    triangle_label: &mut [Label],
    vertex_label: &mut [Label],
) -> Strands {
    let StrandContext {
        height,
        skin: skin_lab,
        threshold,
        ears,
    } = context;
    let n = scan.triangles.len();
    let skin: Vec<bool> = triangle_label.iter().map(|l| *l == Label::Skin).collect();
    // Lowest head skin (≈ the chin and jaw); strands hang a little below it.
    let mut head_y: Vec<f64> = (0..scan.weld.count)
        .filter(|&w| matches!(observations[w], Some((Region::Head, _))))
        .map(|w| scan.positions[scan.weld.representative[w]][1])
        .collect();
    head_y.sort_by(f64::total_cmp);
    let Some(&chin) = head_y.get(head_y.len() / 50) else {
        return Strands {
            strand: vec![false; n],
            report: json!({"applied": false, "reason": "no head"}),
        };
    };
    let floor = chin - height * 0.08;
    let in_area = |t: usize| {
        scan.triangles[t].iter().all(|&i| {
            observations[scan.weld.id[i as usize]].is_some()
                && scan.positions[i as usize][1] > floor
        })
    };
    let candidate: Vec<bool> = (0..n).map(|t| skin[t] && in_area(t)).collect();
    let hair: Vec<bool> = triangle_label.iter().map(|l| *l == Label::Hair).collect();
    let locked: Vec<bool> = match guide {
        Some(g) => (0..n)
            .map(|t| g.seed_stroke[t] && g.skin_seed[t].is_some() && g.garment_seed[t].is_none())
            .collect(),
        None => vec![false; n],
    };
    let incidence = crate::strands::Incidence::new(scan);
    let detection = crate::strands::detect(
        &crate::strands::Input {
            scan,
            candidate: &candidate,
            skin: &skin,
            hair: &hair,
            locked: &locked,
            guard: ears,
            params: crate::strands::Params::for_height(height),
        },
        &incidence,
    );
    for (t, s) in detection.strand.iter().enumerate() {
        if *s {
            triangle_label[t] = Label::Hair;
        }
    }
    // Ears: hair-labelled, skin-ish (nearer the skin colour than the hair's)
    // surface in the ear zone that the guidance did not mark hair, grouped
    // into connected patches; a patch touching more skin than hair is ear.
    let de = |c: &[f64; 3], m: &[f64; 3]| {
        ((c[0] - m[0]).powi(2) * 0.5 + (c[1] - m[1]).powi(2) + (c[2] - m[2]).powi(2)).sqrt()
    };
    let tri_lab = |t: usize| -> Option<[f64; 3]> {
        let mut c = [0.0; 3];
        for &i in &scan.triangles[t] {
            let (_, l) = observations[scan.weld.id[i as usize]]?;
            for k in 0..3 {
                c[k] += l[k] / 3.0;
            }
        }
        Some(c)
    };
    let hair_samples: Vec<[f64; 3]> = (0..n)
        .filter(|&t| hair[t] && in_area(t))
        .step_by(7)
        .filter_map(tri_lab)
        .collect();
    let mut ear_skin = 0usize;
    let mut ear_patches: Vec<Value> = vec![];
    if hair_samples.len() > 50 {
        let hair_lab: [f64; 3] =
            std::array::from_fn(|k| median(hair_samples.iter().map(|c| c[k]).collect()));
        let marked_hair = |t: usize| {
            guide.is_some_and(|g| {
                g.seed_strong[t]
                    && g.garment_seed[t].is_some_and(|(gi, _)| g.garments[gi].label == Label::Hair)
            })
        };
        let earish: Vec<bool> = (0..n)
            .map(|t| {
                ears[t]
                    && triangle_label[t] == Label::Hair
                    && !detection.strand[t]
                    && !marked_hair(t)
                    && tri_lab(t).is_some_and(|c| {
                        de(&c, &skin_lab) < threshold * 2.2 && de(&c, &skin_lab) < de(&c, &hair_lab)
                    })
            })
            .collect();
        let mut seen = vec![false; n];
        for t0 in 0..n {
            if !earish[t0] || seen[t0] {
                continue;
            }
            seen[t0] = true;
            let mut patch = vec![t0];
            let mut k = 0;
            while k < patch.len() {
                let t = patch[k];
                k += 1;
                for u in incidence.around(scan, t) {
                    if earish[u] && !seen[u] {
                        seen[u] = true;
                        patch.push(u);
                    }
                }
            }
            let mut contact: Vec<usize> = patch
                .iter()
                .flat_map(|&t| incidence.around(scan, t))
                .filter(|&u| !earish[u])
                .collect();
            contact.sort_unstable();
            contact.dedup();
            let on_skin = contact
                .iter()
                .filter(|&&u| triangle_label[u] == Label::Skin)
                .count();
            let on_hair = contact
                .iter()
                .filter(|&&u| triangle_label[u] == Label::Hair)
                .count();
            let to_skin = on_skin > on_hair;
            if patch.len() >= 20 {
                ear_patches.push(json!({"triangles": patch.len(), "skin_contact": on_skin, "hair_contact": on_hair, "to_skin": to_skin}));
            }
            if to_skin {
                ear_skin += patch.len();
                for &t in &patch {
                    triangle_label[t] = Label::Skin;
                    for &i in &scan.triangles[t] {
                        vertex_label[i as usize] = Label::Skin;
                    }
                }
            }
        }
    }
    // A strand vertex stays skin where a skin triangle still uses it.
    let mut on_skin = vec![false; scan.weld.count];
    for (t, tri) in scan.triangles.iter().enumerate() {
        if triangle_label[t] == Label::Skin {
            tri.iter()
                .for_each(|&i| on_skin[scan.weld.id[i as usize]] = true);
        }
    }
    let mut vertices = 0usize;
    for (t, tri) in scan.triangles.iter().enumerate() {
        if detection.strand[t] {
            for &i in tri {
                if !on_skin[scan.weld.id[i as usize]] && vertex_label[i as usize] != Label::Hair {
                    vertex_label[i as usize] = Label::Hair;
                    vertices += 1;
                }
            }
        }
    }
    let mut report = crate::strands::report(&detection);
    report["applied"] = json!(true);
    report["relabelled_vertices"] = json!(vertices);
    report["candidate_triangles"] = json!(candidate.iter().filter(|x| **x).count());
    report["locked_triangles"] = json!(locked.iter().filter(|x| **x).count());
    report["ear_zone_triangles"] = json!(ears.iter().filter(|x| **x).count());
    report["ear_hair_to_skin_triangles"] = json!(ear_skin);
    report["ear_patches"] = json!(ear_patches);
    Strands {
        strand: detection.strand,
        report,
    }
}

/// Snaps the skin/garment border around the neck to the collar edge (see
/// [`crate::neckline`]) and relabels the moved triangles and their vertices.
#[allow(clippy::too_many_arguments)]
fn refine_neckline(
    scan: &Scan,
    near_neck: &[bool],
    tri_lab: &[[f64; 3]],
    skin_colour: [f64; 3],
    height: f64,
    guide: Option<&crate::guide::Resolved>,
    piece_hair: &[bool],
    piece_rigid: &[bool],
    triangle_label: &mut [Label],
    triangle_piece: &mut [Option<usize>],
    vertex_label: &mut [Label],
) -> Value {
    let n = scan.triangles.len();
    let cloth = |p: usize| !piece_hair[p] && !piece_rigid.get(p).copied().unwrap_or(false);
    let eligible: Vec<bool> = (0..n)
        .map(|t| match triangle_label[t] {
            Label::Skin => true,
            Label::Garment => triangle_piece[t].is_some_and(cloth),
            Label::Hair => false,
        })
        .collect();
    let skin: Vec<bool> = triangle_label.iter().map(|l| *l == Label::Skin).collect();
    let mut sums = vec![([0.0f64; 3], 0usize); piece_hair.len()];
    for (t, p) in triangle_piece.iter().enumerate().step_by(3) {
        if let Some(p) = p {
            let s = &mut sums[*p];
            for (acc, c) in s.0.iter_mut().zip(tri_lab[t]) {
                *acc += c;
            }
            s.1 += 1;
        }
    }
    let piece_colour: Vec<[f64; 3]> = sums
        .iter()
        .map(|(c, k)| c.map(|x| x / (*k).max(1) as f64))
        .collect();
    let locked = guide.map_or(vec![false; n], |g| g.seed_stroke.clone());
    let (changes, report) = crate::neckline::refine(&crate::neckline::Band {
        positions: &scan.positions,
        triangles: &scan.triangles,
        weld_id: &scan.weld.id,
        weld_count: scan.weld.count,
        lab: tri_lab,
        eligible: &eligible,
        locked: &locked,
        skin: &skin,
        piece: triangle_piece,
        near_neck,
        skin_colour,
        piece_colour: &piece_colour,
        width: height * NECKLINE_BAND,
    });
    let mut touched = vec![false; scan.weld.count];
    for &(t, p) in &changes {
        triangle_piece[t] = p;
        triangle_label[t] = if p.is_some() {
            Label::Garment
        } else {
            Label::Skin
        };
        for &i in &scan.triangles[t] {
            touched[scan.weld.id[i as usize]] = true;
        }
    }
    // A moved vertex is skin only when every triangle around it is.
    let mut around = vec![[false; 3]; scan.weld.count];
    for (t, tri) in scan.triangles.iter().enumerate() {
        for &i in tri {
            let w = scan.weld.id[i as usize];
            if touched[w] {
                around[w][triangle_label[t] as usize] = true;
            }
        }
    }
    for (i, l) in vertex_label.iter_mut().enumerate() {
        let w = scan.weld.id[i];
        if touched[w] {
            *l = match around[w] {
                [_, _, true] => Label::Garment,
                [_, true, false] => Label::Hair,
                _ => Label::Skin,
            };
        }
    }
    report
}

/// Guided piece flood: the cost of one step between hair and garment
/// labels, against about 1 per triangle along a garment's own surface.
const LABEL_CROSSING_COST: f64 = 1.0e4;

/// Neckline band half-width, as a fraction of the character's height.
const NECKLINE_BAND: f64 = 0.012;

/// Relabels connected label islands smaller than `minimum` vertices to the
/// most common label on their boundary (texture speckle, seams, stitching).
fn absorb_small_islands(labels: &mut [Label], adj: &[Vec<u32>], minimum: usize) -> usize {
    let n = labels.len();
    let mut seen = vec![false; n];
    let mut changed = 0;
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let label = labels[start];
        let mut component = vec![start];
        seen[start] = true;
        let mut border = [0usize; 3];
        let mut k = 0;
        while k < component.len() {
            let i = component[k];
            k += 1;
            for &j in &adj[i] {
                let j = j as usize;
                if labels[j] != label {
                    border[labels[j] as usize] += 1;
                } else if !seen[j] {
                    seen[j] = true;
                    component.push(j);
                }
            }
        }
        // Real skin regions (face, limbs, hands) are large and connected;
        // small skin-coloured islands are accessories (pins, beads, buckles).
        let limit = if label == Label::Skin {
            minimum * 4
        } else {
            minimum
        };
        if component.len() < limit && border.iter().sum::<usize>() > 0 {
            let best = (0..3).max_by_key(|&b| border[b]).unwrap();
            let to = [Label::Skin, Label::Hair, Label::Garment][best];
            for i in component {
                labels[i] = to;
                changed += 1;
            }
        }
    }
    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn small_islands_take_their_border_label_and_large_regions_stay() {
        // A path graph: 30 garment, 3 skin speckles, 30 garment, 40 skin.
        let mut labels: Vec<Label> = std::iter::repeat_n(Label::Garment, 30)
            .chain(std::iter::repeat_n(Label::Skin, 3))
            .chain(std::iter::repeat_n(Label::Garment, 30))
            .chain(std::iter::repeat_n(Label::Skin, 40))
            .collect();
        let n = labels.len();
        let adj: Vec<Vec<u32>> = (0..n)
            .map(|i| {
                let mut r = vec![];
                if i > 0 {
                    r.push(i as u32 - 1);
                }
                if i + 1 < n {
                    r.push(i as u32 + 1);
                }
                r
            })
            .collect();
        let changed = absorb_small_islands(&mut labels, &adj, 5);
        assert_eq!(changed, 3);
        assert!(labels[..63].iter().all(|l| *l == Label::Garment));
        assert!(labels[63..].iter().all(|l| *l == Label::Skin));
    }
}

/// Total order for non-NaN f64 heap keys.
#[derive(PartialEq, Clone, Copy)]
struct Ordered(f64);
impl Eq for Ordered {}
impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}
