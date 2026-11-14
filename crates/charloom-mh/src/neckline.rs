//! Neckline refinement: the skin/garment border around the neck snapped to
//! the collar's edge.
//!
//! Seeds, floods and the colour rule leave the border where competing fronts
//! met: a stair-stepped line a centimetre or two off the collar, skin strips
//! along a cream lapel, collar fringe on the neck. The collar itself is
//! usually a sharp crease and colour edge on the scan. Inside a narrow band
//! around the border (only the border loops that reach the neck) the labels
//! are re-chosen by a minimum cut on the triangle graph: cutting between two
//! triangles costs their shared edge length, discounted where the surface
//! changes from skin-like to cloth-like colour and, less, where it folds;
//! each triangle pays for the label its colour fits worse (the skin colour
//! against the nearest garment's). The band's rim keeps its labels, so the
//! border moves at most `width`.
use crate::geom::{V, v};
use charloom_core::Vec3;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::VecDeque;

/// Per-triangle inputs of [`refine`]. `skin[t]` and `piece[t]` are the
/// current labels; only `eligible` triangles (skin and ordinary garments,
/// not hair or rigid parts) take part, `locked` ones (stroke seeds) keep
/// their label.
pub struct Band<'a> {
    pub positions: &'a [Vec3],
    pub triangles: &'a [[u32; 3]],
    pub weld_id: &'a [usize],
    pub weld_count: usize,
    pub lab: &'a [[f64; 3]],
    pub eligible: &'a [bool],
    pub locked: &'a [bool],
    pub skin: &'a [bool],
    pub piece: &'a [Option<usize>],
    /// Per welded vertex: over the neck (dominant joint `neck*`).
    pub near_neck: &'a [bool],
    pub skin_colour: [f64; 3],
    /// Median colour per piece.
    pub piece_colour: &'a [[f64; 3]],
    /// Band half-width (m): how far the border may move.
    pub width: f64,
}

/// Step in skin-likeness (0 = the garment's colour, 1 = the skin's) at
/// which a cut gets cheap; surface fold (1 − cos of the dihedral angle) at
/// which a fold discount sets in and its largest share; the floor of a
/// cut's cost; the weight of the colour term and of the current label
/// against the cut length.
const SKIN_STEP: f64 = 0.3;
const FOLD_EDGE: f64 = 0.08;
const FOLD_DISCOUNT: f64 = 0.6;
const CUT_FLOOR: f64 = 0.05;
const COLOUR_WEIGHT: f64 = 2.0;
const KEEP_WEIGHT: f64 = 0.1;
/// ΔE from the nearer of the two colours at which the colour term has
/// fallen to 1/e.
const COLOUR_FIT: f64 = 15.0;

/// CIELAB distance with lightness at half weight, as in `segment`.
fn lab_distance(c: &[f64; 3], m: &[f64; 3]) -> f64 {
    ((c[0] - m[0]).powi(2) * 0.5 + (c[1] - m[1]).powi(2) + (c[2] - m[2]).powi(2)).sqrt()
}

/// New labels for the band: (triangle, new piece — `None` = skin) for every
/// triangle whose skin/garment label changed, plus a report.
pub fn refine(b: &Band) -> (Vec<(usize, Option<usize>)>, Value) {
    let n = b.triangles.len();
    let w = |t: usize, k: usize| b.weld_id[b.triangles[t][k] as usize];
    // Triangles around each welded vertex (CSR).
    let mut start = vec![0u32; b.weld_count + 1];
    for t in 0..n {
        for k in 0..3 {
            start[w(t, k) + 1] += 1;
        }
    }
    for k in 0..b.weld_count {
        start[k + 1] += start[k];
    }
    let mut fill = start.clone();
    let mut incident = vec![0u32; start[b.weld_count] as usize];
    for t in 0..n {
        for k in 0..3 {
            let x = w(t, k);
            incident[fill[x] as usize] = t as u32;
            fill[x] += 1;
        }
    }
    let around = |x: usize| &incident[start[x] as usize..start[x + 1] as usize];
    // Border vertices: touched by both an eligible skin and garment triangle.
    let border: Vec<usize> = (0..b.weld_count)
        .into_par_iter()
        .filter(|&x| {
            let mut kinds = [false; 2];
            for &t in around(x) {
                let t = t as usize;
                if b.eligible[t] {
                    kinds[b.skin[t] as usize] = true;
                }
            }
            kinds[0] && kinds[1]
        })
        .collect();
    // Surface distance from the border over eligible triangles (Dijkstra on
    // the welded vertices), up to `width`.
    let mut point = vec![V::zeros(); b.weld_count];
    for t in 0..n {
        for k in 0..3 {
            point[w(t, k)] = v(b.positions[b.triangles[t][k] as usize]);
        }
    }
    let mut dist = vec![f64::INFINITY; b.weld_count];
    let mut heap = std::collections::BinaryHeap::new();
    for &x in &border {
        dist[x] = 0.0;
        heap.push((std::cmp::Reverse(Key(0.0)), x));
    }
    while let Some((std::cmp::Reverse(Key(d)), x)) = heap.pop() {
        if d > dist[x] {
            continue;
        }
        for &t in around(x) {
            let t = t as usize;
            if !b.eligible[t] {
                continue;
            }
            for k in 0..3 {
                let y = w(t, k);
                let next = d + (point[y] - point[x]).norm();
                if next < dist[y] && next <= b.width {
                    dist[y] = next;
                    heap.push((std::cmp::Reverse(Key(next)), y));
                }
            }
        }
    }
    let in_band: Vec<bool> = (0..n)
        .map(|t| b.eligible[t] && !b.locked[t] && (0..3).all(|k| dist[w(t, k)] <= b.width))
        .collect();
    // Band components; keep those whose border reaches the neck.
    let mut component = vec![usize::MAX; n];
    let mut keep_component: Vec<bool> = vec![];
    for t0 in 0..n {
        if !in_band[t0] || component[t0] != usize::MAX {
            continue;
        }
        let c = keep_component.len();
        let mut neck = false;
        let mut queue = vec![t0];
        component[t0] = c;
        while let Some(t) = queue.pop() {
            for k in 0..3 {
                let x = w(t, k);
                neck |= dist[x] == 0.0 && b.near_neck[x];
                for &u in around(x) {
                    let u = u as usize;
                    if in_band[u] && component[u] == usize::MAX {
                        component[u] = c;
                        queue.push(u);
                    }
                }
            }
        }
        keep_component.push(neck);
    }
    let band: Vec<usize> = (0..n)
        .filter(|&t| in_band[t] && keep_component[component[t]])
        .collect();
    if band.is_empty() {
        return (vec![], json!({"band_triangles": 0}));
    }
    let mut node = vec![u32::MAX; n];
    for (i, &t) in band.iter().enumerate() {
        node[t] = i as u32;
    }
    // Garment pieces reaching each band triangle (BFS from garment triangles).
    let mut nearest: Vec<Option<usize>> = vec![None; band.len()];
    let mut queue: VecDeque<usize> = VecDeque::new();
    for (i, &t) in band.iter().enumerate() {
        if !b.skin[t] {
            nearest[i] = b.piece[t];
            queue.push_back(i);
        }
    }
    for (i, &t) in band.iter().enumerate() {
        if b.skin[t] && nearest[i].is_none() {
            // Skin band triangles next to an outside garment triangle.
            'outer: for k in 0..3 {
                for &u in around(w(t, k)) {
                    let u = u as usize;
                    if node[u] == u32::MAX && b.eligible[u] && !b.skin[u] && b.piece[u].is_some() {
                        nearest[i] = b.piece[u];
                        queue.push_back(i);
                        break 'outer;
                    }
                }
            }
        }
    }
    while let Some(i) = queue.pop_front() {
        let t = band[i];
        for k in 0..3 {
            for &u in around(w(t, k)) {
                let j = node[u as usize];
                if j != u32::MAX && nearest[j as usize].is_none() {
                    nearest[j as usize] = nearest[i];
                    queue.push_back(j as usize);
                }
            }
        }
    }
    // The garment a triangle would belong to: of the pieces along its band
    // component, the one nearest in colour. The nearest piece in steps can
    // be the wrong layer: a cream collar between the neck and a blue robe is
    // closer to the skin than to the robe, but not to the collar.
    let mut component_pieces: Vec<Vec<usize>> = vec![vec![]; keep_component.len()];
    for (i, &t) in band.iter().enumerate() {
        if let Some(p) = nearest[i] {
            let list = &mut component_pieces[component[t]];
            if !list.contains(&p) {
                list.push(p);
            }
        }
    }
    for (i, &t) in band.iter().enumerate() {
        if let Some(p) = component_pieces[component[t]]
            .iter()
            .copied()
            .min_by(|&p, &q| {
                lab_distance(&b.lab[t], &b.piece_colour[p])
                    .total_cmp(&lab_distance(&b.lab[t], &b.piece_colour[q]))
            })
        {
            nearest[i] = Some(p);
        }
    }
    let normal = |t: usize| -> V {
        let [p0, p1, p2] = b.triangles[t].map(|i| v(b.positions[i as usize]));
        (p1 - p0).cross(&(p2 - p0))
    };
    let mut edge_sum = 0.0;
    for &t in &band {
        let [p0, p1, p2] = b.triangles[t].map(|i| v(b.positions[i as usize]));
        edge_sum += (p1 - p0).norm() + (p2 - p1).norm() + (p0 - p2).norm();
    }
    let unit = (edge_sum / (3 * band.len()) as f64).max(1e-9);
    // Skin-likeness of a triangle against a garment colour: a cut is cheap
    // where it changes (skin against cloth), not at any colour step (a
    // cream collar against a blue robe is cloth on both sides).
    let skinness = |t: usize, p: Option<usize>| -> f64 {
        let Some(p) = p else { return 1.0 };
        let ds = lab_distance(&b.lab[t], &b.skin_colour);
        let dg = lab_distance(&b.lab[t], &b.piece_colour[p]);
        (dg + 2.0) / (ds + dg + 4.0)
    };
    // Graph: band triangles + source (garment) + sink (skin).
    let (source, sink) = (band.len(), band.len() + 1);
    let mut graph = Flow::new(band.len() + 2);
    let mut cut_cheap = 0usize;
    for (i, &t) in band.iter().enumerate() {
        let nt = normal(t);
        for k in 0..3 {
            let (x, y) = (w(t, k), w(t, (k + 1) % 3));
            let length = (point[x] - point[y]).norm() / unit;
            for &u in around(x) {
                let u = u as usize;
                if u == t || !b.eligible[u] || !(0..3).any(|j| w(u, j) == y) {
                    continue;
                }
                let j = node[u];
                if j != u32::MAX && (j as usize) < i {
                    continue;
                }
                let hint = if j == u32::MAX && !b.skin[u] {
                    b.piece[u]
                } else {
                    nearest[i]
                };
                let step = (skinness(t, hint) - skinness(u, hint)).abs();
                let nu = normal(u);
                let cos = nt.dot(&nu) / (nt.norm() * nu.norm()).max(1e-30);
                let fold = (1.0 - cos).max(0.0);
                let ease = (-(step / SKIN_STEP).powi(2)).exp()
                    * (1.0 - FOLD_DISCOUNT * (1.0 - (-fold / FOLD_EDGE).exp()));
                let weight = length * (CUT_FLOOR + (1.0 - CUT_FLOOR) * ease);
                if ease < 0.5 {
                    cut_cheap += 1;
                }
                if j == u32::MAX {
                    // Rim: the outside neighbour keeps its label.
                    if b.skin[u] {
                        graph.add(i, sink, weight, 0.0);
                    } else {
                        graph.add(source, i, weight, 0.0);
                    }
                } else {
                    graph.add(i, j as usize, weight, weight);
                }
            }
        }
        // Colour: the skin colour against the nearest garment's, plus a
        // little for the current label.
        // The colour counts only as far as one of the two explains it: deep
        // shade under a beard fits neither skin nor cloth.
        if let Some(p) = nearest[i] {
            let area = normal(t).norm() * 0.5 / (unit * unit);
            let keep = if b.skin[t] { -KEEP_WEIGHT } else { KEEP_WEIGHT };
            let fit = lab_distance(&b.lab[t], &b.skin_colour)
                .min(lab_distance(&b.lab[t], &b.piece_colour[p]));
            let confidence = (-fit / COLOUR_FIT).exp();
            let r =
                ((1.0 - 2.0 * skinness(t, nearest[i])) * COLOUR_WEIGHT * confidence + keep) * area;
            if r > 0.0 {
                graph.add(source, i, r, 0.0);
            } else {
                graph.add(i, sink, -r, 0.0);
            }
        }
    }
    graph.max_flow(source, sink);
    let garment_side = graph.reachable(source);
    let mut changes = vec![];
    let (mut to_skin, mut to_garment) = (0usize, 0usize);
    for (i, &t) in band.iter().enumerate() {
        let garment = garment_side[i];
        if garment == !b.skin[t] {
            continue;
        }
        if garment {
            if let Some(p) = nearest[i] {
                changes.push((t, Some(p)));
                to_garment += 1;
            }
        } else {
            changes.push((t, None));
            to_skin += 1;
        }
    }
    let report = json!({
        "band_triangles": band.len(),
        "band_width_m": b.width,
        "border_vertices": border.len(),
        "edges_on_colour_or_fold": cut_cheap,
        "to_skin_triangles": to_skin,
        "to_garment_triangles": to_garment,
    });
    (changes, report)
}

/// Total order for non-NaN f64 heap keys.
#[derive(PartialEq, Clone, Copy)]
struct Key(f64);
impl Eq for Key {}
impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Key {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Max-flow / min-cut (Dinic) on a small graph with real capacities.
struct Flow {
    head: Vec<Vec<usize>>,
    to: Vec<usize>,
    cap: Vec<f64>,
}

impl Flow {
    fn new(nodes: usize) -> Flow {
        Flow {
            head: vec![vec![]; nodes],
            to: vec![],
            cap: vec![],
        }
    }
         fn add(&mut self, a: usize, b: usize, forward: f64, backward: f64) {
        self.head[a].push(self.to.len());
        self.to.push(b);
        self.cap.push(forward);
        self.head[b].push(self.to.len());
        self.to.push(a);
        self.cap.push(backward);
    }
    fn levels(&self, s: usize, t: usize) -> Option<Vec<u32>> {
        let mut level = vec![u32::MAX; self.head.len()];
        level[s] = 0;
        let mut queue = VecDeque::from([s]);
        while let Some(x) = queue.pop_front() {
            for &e in &self.head[x] {
                let y = self.to[e];
                if self.cap[e] > 1e-12 && level[y] == u32::MAX {
                    level[y] = level[x] + 1;
                    queue.push_back(y);
                }
            }
        }
        (level[t] != u32::MAX).then_some(level)
    }
    fn max_flow(&mut self, s: usize, t: usize) -> f64 {
        let mut total = 0.0;
        while let Some(level) = self.levels(s, t) {
            let mut next = vec![0usize; self.head.len()];
            loop {
                // Iterative DFS for one augmenting path in the level graph.
                let mut path: Vec<usize> = vec![];
                let mut x = s;
                let found = loop {
                    if x == t {
                        break true;
                    }
                    let mut advanced = false;
                    while next[x] < self.head[x].len() {
                        let e = self.head[x][next[x]];
                        let y = self.to[e];
                        if self.cap[e] > 1e-12 && level[y] == level[x] + 1 {
                            path.push(e);
                            x = y;
                            advanced = true;
                            break;
                        }
                        next[x] += 1;
                    }
                    if !advanced {
                        if x == s {
                            break false;
                        }
                        // Dead end: retreat and skip this arc.
                        let e = path.pop().expect("path to x");
                        x = self.to[e ^ 1];
                        next[x] += 1;
                    }
                };
                if !found {
                    break;
                }
                let push = path
                    .iter()
                    .map(|&e| self.cap[e])
                    .fold(f64::INFINITY, f64::min);
                for &e in &path {
                    self.cap[e] -= push;
                    self.cap[e ^ 1] += push;
                }
                total += push;
            }
        }
        total
    }
    /// Nodes reachable from `s` in the residual graph (the source side).
    fn reachable(&self, s: usize) -> Vec<bool> {
        let mut seen = vec![false; self.head.len()];
        seen[s] = true;
        let mut queue = vec![s];
        while let Some(x) = queue.pop() {
            for &e in &self.head[x] {
                let y = self.to[e];
                if self.cap[e] > 1e-12 && !seen[y] {
                    seen[y] = true;
                    queue.push(y);
                }
            }
        }
        seen
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_cut_separates_at_the_cheap_edge() {
         
        let mut f = Flow::new(5);
        f.add(3, 0, 10.0, 0.0);
        f.add(0, 1, 5.0, 5.0);
        f.add(1, 2, 1.0, 1.0);
        f.add(2, 4, 10.0, 0.0);
        assert!((f.max_flow(3, 4) - 1.0).abs() < 1e-9);
        let side = f.reachable(3);
        assert_eq!(&side[..3], &[true, true, false]);
    }

    /// A strip of quads along x, 12 columns of two triangles: column colours
    /// and pieces from `colour(i)`, current labels skin up to column
    /// `border` or `border + 1` by row (stair-stepped). Returns the refined
    /// skin flag per column for every triangle.
    fn refine_strip(
        colour: impl Fn(usize) -> ([f64; 3], usize),
        pieces: &[[f64; 3]],
        border: usize,
    ) -> Vec<(usize, bool)> {
        let (nx, ny) = (12usize, 6usize);
        let mut positions = vec![];
        for j in 0..=ny {
            for i in 0..=nx {
                positions.push([i as f64 * 0.01, j as f64 * 0.01, 0.0]);
            }
        }
        let id = |i: usize, j: usize| (j * (nx + 1) + i) as u32;
        let mut triangles = vec![];
        let mut column = vec![];
        for j in 0..ny {
            for i in 0..nx {
                triangles.push([id(i, j), id(i + 1, j), id(i + 1, j + 1)]);
                triangles.push([id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
                column.extend([(i, j), (i, j)]);
            }
        }
        let n = triangles.len();
        let lab: Vec<[f64; 3]> = column.iter().map(|&(i, _)| colour(i).0).collect();
        let skin: Vec<bool> = column.iter().map(|&(i, j)| i < border + j % 2).collect();
        let piece: Vec<Option<usize>> = column
            .iter()
            .zip(&skin)
            .map(|(&(i, _), s)| (!s).then(|| colour(i).1))
            .collect();
        let weld_id: Vec<usize> = (0..positions.len()).collect();
        let band = Band {
            positions: &positions,
            triangles: &triangles,
            weld_id: &weld_id,
            weld_count: positions.len(),
            lab: &lab,
            eligible: &vec![true; n],
            locked: &vec![false; n],
            skin: &skin,
            piece: &piece,
            near_neck: &vec![true; positions.len()],
            skin_colour: SKIN,
            piece_colour: pieces,
            width: 0.035,
        };
        let (changes, _) = refine(&band);
        let mut refined = skin.clone();
        for (t, p) in changes {
            refined[t] = p.is_none();
        }
        column.iter().map(|&(i, _)| i).zip(refined).collect()
    }

    const SKIN: [f64; 3] = [70.0, 20.0, 30.0];
    const CREAM: [f64; 3] = [85.0, 5.0, 15.0];
    const BLUE: [f64; 3] = [35.0, 5.0, -35.0];

    /// The stair-stepped border moves onto the colour edge at column 6 in a
    /// straight line.
    #[test]
    fn border_snaps_to_the_colour_edge() {
        let cut = refine_strip(|i| (if i < 6 { SKIN } else { CREAM }, 0), &[CREAM], 4);
        for (i, skin) in cut {
            assert_eq!(skin, i < 6, "column {i}");
        }
    }

    /// Skin | cream collar | blue robe: the stronger colour step between
    /// the two garments does not pull the border off the collar's edge.
    #[test]
    fn border_stays_at_the_skin_edge_not_the_garment_edge() {
        let colour = |i: usize| match i {
            0..6 => (SKIN, 0),
            6..8 => (CREAM, 0),
            _ => (BLUE, 1),
        };
        let cut = refine_strip(colour, &[CREAM, BLUE], 4);
        for (i, skin) in cut {
            assert_eq!(skin, i < 6, "column {i}");
        }
    }

    /// The collar labelled skin up to the robe: the cream next to the blue
    /// robe is judged against the collar's colour, not the robe's, and
    /// goes back to the collar.
    #[test]
    fn collar_wrongly_skin_next_to_another_garment_returns() {
        let colour = |i: usize| match i {
            0..6 => (SKIN, 0),
            6..9 => (CREAM, 0),
            _ => (BLUE, 1),
        };
        let cut = refine_strip(colour, &[CREAM, BLUE], 8);
        for (i, skin) in cut {
            assert_eq!(skin, i < 6, "column {i}");
        }
    }
}
