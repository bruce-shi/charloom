 use crate::geom::{V, a, face_normal, v};
use crate::scan::Scan;
use crate::source::lab;
use image::RgbaImage;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;

/// Per-triangle layer probe along the face normal (metres; `INFINITY` =
/// nothing within reach).
#[derive(Debug, Clone, Copy)]
pub struct Probe {
    /// Distance into the surface to where the ray leaves it again (0 for an
    /// open sheet).
    pub thickness: f64,
    /// Air gap to the next scan surface facing the ray behind the layer
    /// (along −normal after leaving it) or in front of it (along +normal).
    pub gap: f64,
}

impl Probe {
    const NONE: Probe = Probe {
        thickness: f64::INFINITY,
        gap: f64::INFINITY,
    };
}

/// Thresholds for a character `height` metres tall.
#[derive(Debug, Clone, Copy)]
pub struct Params {
    /// Thicker than this is solid body (a cheek, the nose), not a strand.
    pub thin_m: f64,
    /// A facing surface within this distance of the layer makes it a layer.
    pub reach_m: f64,
    /// A gap above this is "floating".
    pub float_m: f64,
    /// Strand triangles this close (along the region) to the skin they touch
    /// stay skin unless they are nearer to the hair.
    pub near_m: f64,
    /// A loose shell shorter than this is not judged a strand: small shells
    /// near the face are teeth, lashes or eye parts.
    pub detached_m: f64,
}

impl Params {
    pub fn for_height(height: f64) -> Self {
        let unit = height / 1.75;
        Self {
            thin_m: 0.008 * unit,
            reach_m: 0.04 * unit,
            float_m: 0.0015 * unit,
            near_m: 0.004 * unit,
            detached_m: 0.015 * unit,
        }
    }
}

/// Probes one triangle (see [`Probe`]).
pub fn probe(scan: &Scan, t: usize, params: &Params) -> Probe {
    let tri = scan.triangles[t];
    let Some(n) = face_normal(&scan.positions, &tri).try_normalize(1e-18) else {
        return Probe::NONE;
    };
    let c = tri
        .iter()
        .fold(V::zeros(), |s, &i| s + v(scan.positions[i as usize]))
        / 3.0;
    let eps = 2e-5;
    let reach = params.reach_m + params.thin_m;
    let inward = a(c - n * eps);
    // The exit is looked for much further: a closed body always has one (a
    // face ray leaves through the back of the head), an open sheet none.
    let exit = scan
        .query
        .outward_ray_distance(inward, a(-n), params.reach_m * 8.0);
    let behind = scan
        .first_facing_hit(inward, a(-n), reach)
        .map(|h| h.distance);
    let (thickness, behind_gap) = match (exit, behind) {
        // Left the layer, then met a surface facing it (the skin behind).
        (Some(x), Some(d)) if d > x => (x, d - x),
        // A facing surface before any exit: an open sheet in front of it.
        (_, Some(d)) => (0.0, d),
        (Some(x), None) => (x, f64::INFINITY),
        // An open sheet with nothing behind it.
        (None, None) => (0.0, f64::INFINITY),
    };
    if thickness >= params.thin_m {
        return Probe {
            thickness,
            gap: f64::INFINITY,
        };
    }
    // The strand's side facing the skin looks at it along +normal.
    let front = scan
        .first_facing_hit(a(c + n * eps), a(n), params.reach_m)
        .map_or(f64::INFINITY, |h| h.distance);
    Probe {
        thickness,
        gap: behind_gap.min(front),
    }
}

/// Labels around the triangles [`detect`] looks at, per scan triangle.
pub struct Input<'a> {
    pub scan: &'a Scan,
    /// Skin triangles that may be strands (skin near the head).
    pub candidate: &'a [bool],
    /// Skin triangles (candidates included).
    pub skin: &'a [bool],
    /// Hair triangles: what a strand hangs from.
    pub hair: &'a [bool],
         pub locked: &'a [bool],
    /// Near an ear: a strand must reach further from the skin it touches.
    pub guard: &'a [bool],
    pub params: Params,
}

/// One connected thin-layer region of candidate skin.
#[derive(Debug, Clone)]
pub struct Region {
    pub triangles: Vec<usize>,
    /// Neighbouring triangles (sharing a welded vertex) outside the region.
    pub hair_contact: usize,
    pub skin_contact: usize,
    pub locked: usize,
    /// Extents along the first two principal axes.
    pub length_m: f64,
    pub width_m: f64,
    /// Fraction of the region with an air gap above [`Params::float_m`].
    pub floating: f64,
    pub median_gap_m: f64,
    pub median_thickness_m: f64,
    pub centroid: [f64; 3],
    pub verdict: Verdict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// Touches hair and little skin: the strand hangs from the hair.
    HangsFromHair,
    /// Touches nothing and is long and thin or floating: a loose strand.
    DetachedShell,
    /// Continuous with the skin surface (ear rim, nostril, lid): skin.
    ContinuousWithSkin,
    /// Small or compact without hair contact: left alone.
    Compact,
         ConfirmedSkin,
}

impl Verdict {
    pub fn is_hair(self) -> bool {
        matches!(self, Verdict::HangsFromHair | Verdict::DetachedShell)
    }
    pub fn name(self) -> &'static str {
        match self {
            Verdict::HangsFromHair => "hangs_from_hair",
            Verdict::DetachedShell => "detached_shell",
            Verdict::ContinuousWithSkin => "continuous_with_skin",
            Verdict::Compact => "compact",
            Verdict::ConfirmedSkin => "confirmed_skin",
        }
    }
}

/// The region rule, on its measurements alone (see the module docs).
pub fn verdict(r: &Region, params: &Params) -> Verdict {
    let elongation = r.length_m / r.width_m.max(1e-6);
    if r.locked > 0 {
        Verdict::ConfirmedSkin
    } else if r.hair_contact > 0 && r.skin_contact <= r.hair_contact * 2 && r.triangles.len() >= 3 {
        Verdict::HangsFromHair
    } else if r.hair_contact == 0
        && r.skin_contact == 0
        && r.triangles.len() >= 6
        && elongation >= 3.0
        && r.floating >= 0.5
        && r.length_m >= params.detached_m
    {
        Verdict::DetachedShell
    } else if r.skin_contact > 0 {
        Verdict::ContinuousWithSkin
    } else {
        Verdict::Compact
    }
}

/// Fewer connected solid skin triangles than this are not skin a strand
/// continues into (see [`detect`]).
const SOLID_MIN: usize = 50;

/// Triangles around every welded vertex (CSR).
pub struct Incidence {
    start: Vec<u32>,
    list: Vec<u32>,
}

impl Incidence {
    pub fn new(scan: &Scan) -> Self {
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
        let mut list = vec![0u32; start[scan.weld.count] as usize];
        for (t, tri) in scan.triangles.iter().enumerate() {
            for &i in tri {
                let w = scan.weld.id[i as usize];
                list[fill[w] as usize] = t as u32;
                fill[w] += 1;
            }
        }
        Self { start, list }
    }

    /// Triangles sharing a welded vertex with `t` (`t` included).
    pub fn around<'s>(&'s self, scan: &'s Scan, t: usize) -> impl Iterator<Item = usize> + 's {
        scan.triangles[t].iter().flat_map(move |&i| {
            let w = scan.weld.id[i as usize];
            self.list[self.start[w] as usize..self.start[w + 1] as usize]
                .iter()
                .map(|&u| u as usize)
        })
    }
}

pub struct Detection {
         pub strand: Vec<bool>,
    pub regions: Vec<Region>,
    /// Triangles added to accepted regions by growing over thin skin.
    pub grown: usize,
    /// Triangles of accepted regions kept as skin: next to the skin they
    /// touch and nearer to it than to the hair.
    pub split_off: usize,
    pub probed: usize,
    pub layer_triangles: usize,
}

/// Finds skin-labelled strands (see the module docs).
pub fn detect(input: &Input, incidence: &Incidence) -> Detection {
    let scan = input.scan;
    let n = scan.triangles.len();
    let params = input.params;
    let probes: Vec<Option<Probe>> = (0..n)
        .into_par_iter()
        .map(|t| input.candidate[t].then(|| probe(scan, t, &params)))
        .collect();
    let is_layer = |p: &Option<Probe>| {
        p.is_some_and(|p| p.thickness < params.thin_m && p.gap <= params.reach_m)
    };
    let layer: Vec<bool> = probes.iter().map(is_layer).collect();
    // Skin a region continues into: solid surface (thick, or not probed),
    // not the thin skin of the same strand (its sides seen edge-on).
    // Small solid patches (a strand's end cap, probed along its length) are
    // part of the strand too: only large connected solid skin counts.
    let mut solid: Vec<bool> = (0..n)
        .map(|t| input.skin[t] && !probes[t].is_some_and(|p| p.thickness < params.thin_m))
        .collect();
    let mut seen = vec![false; n];
    for t0 in 0..n {
        if !solid[t0] || seen[t0] {
            continue;
        }
        seen[t0] = true;
        let mut patch = vec![t0];
        let mut k = 0;
        while k < patch.len() && patch.len() < SOLID_MIN {
            let t = patch[k];
            k += 1;
            for u in incidence.around(scan, t) {
                if solid[u] && !seen[u] {
                    seen[u] = true;
                    patch.push(u);
                }
            }
        }
        if patch.len() < SOLID_MIN {
            for t in patch {
                solid[t] = false;
            }
        }
    }
    let mut region_of = vec![usize::MAX; n];
    let mut regions = vec![];
    for t0 in 0..n {
        if !layer[t0] || region_of[t0] != usize::MAX {
            continue;
        }
        let id = regions.len();
        region_of[t0] = id;
        let mut members = vec![t0];
        let mut k = 0;
        while k < members.len() {
            let t = members[k];
            k += 1;
            for u in incidence.around(scan, t) {
                if layer[u] && region_of[u] == usize::MAX {
                    region_of[u] = id;
                    members.push(u);
                }
            }
        }
        regions.push(members);
    }
    let measured: Vec<Region> = regions
        .into_par_iter()
        .enumerate()
        .map(|(id, members)| {
            let mut contact: Vec<usize> = members
                .iter()
                .flat_map(|&t| incidence.around(scan, t))
                .filter(|&u| region_of[u] != id)
                .collect();
            contact.sort_unstable();
            contact.dedup();
            let hair_contact = contact.iter().filter(|&&u| input.hair[u]).count();
            let skin_contact = contact.iter().filter(|&&u| solid[u]).count();
            let mut region = Region {
                hair_contact,
                skin_contact,
                locked: members.iter().filter(|&&t| input.locked[t]).count(),
                centroid: [0.0; 3],
                length_m: 0.0,
                width_m: 0.0,
                floating: 0.0,
                median_gap_m: 0.0,
                median_thickness_m: 0.0,
                verdict: Verdict::Compact,
                triangles: members,
            };
            measure(scan, &probes, &params, &mut region);
            region.verdict = verdict(&region, &params);
            region
        })
        .collect();
    // Within an accepted region only the part nearer the hair than the skin
    // it touches, or well away from that skin, is strand: a strand fused to
    // an ear rim keeps the rim (a narrow band along the ear skin) as skin.
    let centroid = |t: usize| {
        scan.triangles[t]
            .iter()
            .fold(V::zeros(), |s, &i| s + v(scan.positions[i as usize]))
            / 3.0
    };
    let mut strand = vec![false; n];
    let mut split_off = 0usize;
    for (id, r) in measured.iter().enumerate() {
        if !r.verdict.is_hair() {
            continue;
        }
        let touches = |t: usize, other: &[bool]| {
            incidence
                .around(scan, t)
                .any(|u| region_of[u] != id && other[u])
        };
        let from = |other: &[bool]| -> BTreeMap<usize, f64> {
            let mut dist: BTreeMap<usize, f64> = BTreeMap::new();
            let mut heap = std::collections::BinaryHeap::new();
            for &t in &r.triangles {
                if touches(t, other) {
                    dist.insert(t, 0.0);
                    heap.push((std::cmp::Reverse(Key(0.0)), t));
                }
            }
            while let Some((std::cmp::Reverse(Key(d)), t)) = heap.pop() {
                if dist.get(&t).is_some_and(|&x| d > x) {
                    continue;
                }
                let c = centroid(t);
                for u in incidence.around(scan, t) {
                    if region_of[u] != id {
                        continue;
                    }
                    let e = d + (centroid(u) - c).norm();
                    if dist.get(&u).is_none_or(|&x| e < x) {
                        dist.insert(u, e);
                        heap.push((std::cmp::Reverse(Key(e)), u));
                    }
                }
            }
            dist
        };
        let to_hair = from(input.hair);
        let to_skin = from(&solid);
        for &t in &r.triangles {
            let ds = to_skin.get(&t).copied().unwrap_or(f64::INFINITY);
            let dh = to_hair.get(&t).copied().unwrap_or(f64::INFINITY);
            let near = if input.guard[t] {
                params.near_m * 2.0
            } else {
                params.near_m
            };
            if ds > near || dh < ds {
                strand[t] = true;
            } else {
                split_off += 1;
            }
        }
    }
    // A strand's edges (sides seen edge-on) may not probe as a layer: grow
    // over thin candidate skin that touches no solid (thick) skin.
    let mut grown = 0;
    let mut front: Vec<usize> = (0..n).filter(|&t| strand[t]).collect();
    for _ in 0..4 {
        let mut next = vec![];
        for &t in &front {
            for u in incidence.around(scan, t) {
                if !strand[u]
                    && input.candidate[u]
                    && !input.locked[u]
                    && !input.guard[u]
                    && probes[u].is_some_and(|p| p.thickness < params.thin_m)
                    && !incidence.around(scan, u).any(|x| solid[x])
                {
                    strand[u] = true;
                    grown += 1;
                    next.push(u);
                }
            }
        }
        if next.is_empty() {
            break;
        }
        front = next;
    }
    Detection {
        strand,
        regions: measured,
        grown,
        split_off,
        probed: probes.iter().filter(|p| p.is_some()).count(),
        layer_triangles: layer.iter().filter(|x| **x).count(),
    }
}

/// Total order for non-NaN heap keys.
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

fn median(mut xs: Vec<f64>) -> f64 {
    xs.retain(|x| x.is_finite());
    xs.sort_by(f64::total_cmp);
    xs.get(xs.len() / 2).copied().unwrap_or(f64::INFINITY)
}

fn measure(scan: &Scan, probes: &[Option<Probe>], params: &Params, r: &mut Region) {
    let mut points: Vec<V> = r
        .triangles
        .iter()
        .flat_map(|&t| scan.triangles[t])
        .map(|i| scan.weld.id[i as usize])
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|w| v(scan.positions[scan.weld.representative[w]]))
        .collect();
    if points.is_empty() {
        return;
    }
    let c = points.iter().fold(V::zeros(), |s, p| s + p) / points.len() as f64;
    for p in &mut points {
        *p -= c;
    }
    let cov = points
        .iter()
        .fold(nalgebra::Matrix3::zeros(), |s, p| s + p * p.transpose());
    let eig = nalgebra::SymmetricEigen::new(cov);
    let mut order = [0usize, 1, 2];
    order.sort_by(|&x, &y| eig.eigenvalues[y].total_cmp(&eig.eigenvalues[x]));
    let extent = |k: usize| {
        let axis = eig.eigenvectors.column(order[k]).into_owned();
        let (lo, hi) = points
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| {
                let x = p.dot(&axis);
                (l.min(x), h.max(x))
            });
        hi - lo
    };
    r.centroid = [c.x, c.y, c.z];
    r.length_m = extent(0);
    r.width_m = extent(1);
    let ps: Vec<Probe> = r.triangles.iter().filter_map(|&t| probes[t]).collect();
    r.floating =
        ps.iter().filter(|p| p.gap > params.float_m).count() as f64 / ps.len().max(1) as f64;
    r.median_gap_m = median(ps.iter().map(|p| p.gap).collect());
    r.median_thickness_m = median(ps.iter().map(|p| p.thickness).collect());
}

/// Report of a detection: totals and the larger regions with their cues.
pub fn report(d: &Detection) -> Value {
    let mut by_verdict: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
    for r in &d.regions {
        let e = by_verdict.entry(r.verdict.name()).or_default();
        e.0 += 1;
        e.1 += r.triangles.len();
    }
    let mut large: Vec<&Region> = d
        .regions
        .iter()
        .filter(|r| r.triangles.len() >= 20)
        .collect();
    large.sort_by_key(|r| std::cmp::Reverse(r.triangles.len()));
    json!({
        "relabelled_triangles": d.strand.iter().filter(|x| **x).count(),
        "grown_triangles": d.grown,
        "split_off_triangles": d.split_off,
        "probed_triangles": d.probed,
        "layer_triangles": d.layer_triangles,
        "regions_by_verdict": by_verdict.iter().map(|(k, (r, t))| (k.to_string(), json!({"regions": r, "triangles": t}))).collect::<serde_json::Map<_, _>>(),
        "regions": large.iter().take(40).map(|r| json!({
            "verdict": r.verdict.name(),
            "triangles": r.triangles.len(),
            "hair_contact": r.hair_contact,
            "skin_contact": r.skin_contact,
            "locked": r.locked,
            "length_mm": r.length_m * 1000.0,
            "width_mm": r.width_m * 1000.0,
            "floating": r.floating,
            "median_gap_mm": r.median_gap_m * 1000.0,
            "median_thickness_mm": r.median_thickness_m * 1000.0,
            "centroid": r.centroid,
        })).collect::<Vec<_>>(),
    })
}

/// Recolours, in copies of the scan's images (`images`, by scan image index,
/// made on first use), the skin-coloured texels of the given hair triangles: each takes the colour of the nearest hair-coloured
/// texel area on the hair surface (vertex colours diffused over the welded
/// hair mesh), keeping its own relative luminance. `target[t]`: recolour
/// triangle `t`; `hair[t]`: a hair triangle whose colour can be the source.
/// Colours are skin-like within `skin_threshold` (CIELAB, lightness at half
/// weight) of `skin_lab` and nearer to it than to the hair colour. On
/// `pale[t]` triangles (strands) washed-out grey or white paint, much lighter
/// than the hair, is recoloured too.
pub fn recolour(
    scan: &Scan,
    target: &[bool],
    hair: &[bool],
    pale: &[bool],
    skin_lab: [f64; 3],
    skin_threshold: f64,
    images: &mut BTreeMap<usize, RgbaImage>,
) -> Value {
    let de = |c: &[f64; 3], m: &[f64; 3]| {
        ((c[0] - m[0]).powi(2) * 0.5 + (c[1] - m[1]).powi(2) + (c[2] - m[2]).powi(2)).sqrt()
    };
    let n = scan.triangles.len();
    // Per welded hair vertex: its colour, and whether it is hair-coloured.
    let count = scan.weld.count;
    let mut colour: Vec<Option<[f64; 3]>> = vec![None; count];
    for t in (0..n).filter(|&t| hair[t]) {
        for (k, &i) in scan.triangles[t].iter().enumerate() {
            let w = scan.weld.id[i as usize];
            if colour[w].is_none() {
                let mut b = [0.0; 3];
                b[k] = 1.0;
                colour[w] = scan.color_at(t, b);
            }
        }
    }
    let labs: Vec<Option<[f64; 3]>> = colour.iter().map(|c| c.map(lab)).collect();
    let hair_samples: Vec<[f64; 3]> = labs
        .iter()
        .flatten()
        .filter(|c| de(c, &skin_lab) > skin_threshold * 1.5)
        .copied()
        .collect();
    if hair_samples.len() < 50 || !target.iter().any(|x| *x) {
        return json!({"applied": false, "reason": "no hair-coloured hair or nothing to recolour", "hair_coloured_vertices": hair_samples.len()});
    }
    let hair_lab: [f64; 3] =
        std::array::from_fn(|k| median(hair_samples.iter().map(|c| c[k]).collect()));
     
    let hair_light = {
        let mut l: Vec<f64> = hair_samples.iter().map(|c| c[0]).collect();
        l.sort_by(f64::total_cmp);
        l[l.len() * 3 / 4]
    };
    // Blond hair, a wooden hairpin, a cream ribbon: the piece's own colour is
    // too near the skin to tell paint from it.
    let all: Vec<&[f64; 3]> = labs.iter().flatten().collect();
    let piece_lab: [f64; 3] = std::array::from_fn(|k| median(all.iter().map(|c| c[k]).collect()));
    if de(&piece_lab, &skin_lab) < skin_threshold * 2.2 {
        return json!({"applied": false, "reason": "the piece itself is nearly skin-coloured", "piece_lab": piece_lab});
    }
    // Skin hue and chroma: a white or grey ribbon is not skin paint, however
    // close its lightness.
    let skin_chroma = skin_lab[1].hypot(skin_lab[2]);
    let skin_hue = skin_lab[2].atan2(skin_lab[1]);
    let skin_like = |c: &[f64; 3]| {
        let chroma = c[1].hypot(c[2]);
        let hue = (c[2].atan2(c[1]) - skin_hue + std::f64::consts::PI)
            .rem_euclid(std::f64::consts::TAU)
            - std::f64::consts::PI;
        if chroma < 0.4 * skin_chroma || hue.abs() > 0.45 {
            return 0.0;
        }
        let ds = de(c, &skin_lab);
        // 1 within 1.4× the threshold, 0 from 2.2× (pale, washed-out strand
        // paint too), and only when clearly nearer the skin than the hair.
        let s = ((skin_threshold * 2.2 - ds) / (skin_threshold * 0.8)).clamp(0.0, 1.0);
        if ds < 0.8 * de(c, &hair_lab) {
            s * s * (3.0 - 2.0 * s)
        } else {
            0.0
        }
    };
     
    let span = skin_lab[0] - hair_lab[0];
    let pale_like = |c: &[f64; 3]| {
        if span < 15.0 {
            return 0.0;
        }
        let t = ((c[0] - hair_lab[0] - 0.5 * span) / (0.25 * span)).clamp(0.0, 1.0);
        t * t * (3.0 - 2.0 * t)
    };
    // Hair-coloured source vertices, diffused (multi-source BFS over the
    // welded hair surface) to every hair vertex: the nearest hair colour.
    // Only colours nearer the hair than the skin and no lighter than its
    // upper quartile: grey-white or light-brown strand paint as a source
    // spread itself along the strands.
    let mut adjacency: Vec<Vec<u32>> = vec![vec![]; count];
    for t in (0..n).filter(|&t| hair[t]) {
        let w = scan.triangles[t].map(|i| scan.weld.id[i as usize] as u32);
        for (x, y) in [(w[0], w[1]), (w[1], w[2]), (w[2], w[0])] {
            adjacency[x as usize].push(y);
            adjacency[y as usize].push(x);
        }
    }
    let mut source: Vec<Option<[f64; 3]>> = labs
        .iter()
        .map(|c| {
            c.filter(|c| {
                skin_like(c) == 0.0
                    && de(c, &skin_lab) > skin_threshold
                    && de(c, &hair_lab) < de(c, &skin_lab)
                    && c[0] <= hair_light + 5.0
            })
        })
        .collect();
    let mut queue: std::collections::VecDeque<usize> =
        (0..count).filter(|&w| source[w].is_some()).collect();
    while let Some(w) = queue.pop_front() {
        let c = source[w];
        for &u in &adjacency[w] {
            let u = u as usize;
            if source[u].is_none() {
                source[u] = c;
                queue.push_back(u);
            }
        }
    }
    // Rasterise target triangles per image; one texel pass each.
    let mut per_image: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for t in (0..n).filter(|&t| target[t]) {
        if let Some(i) = scan.triangle_material[t]
            .and_then(|m| scan.materials.get(m))
            .and_then(|m| m.base_color)
        {
            per_image.entry(i).or_default().push(t);
        }
    }
    let mut texels = 0usize;
    let mut changed = 0usize;
    for (image, triangles) in per_image {
        let Some(src) = scan.images.get(image) else {
            continue;
        };
        let img = images.entry(image).or_insert_with(|| src.decoded.clone());
        let (w, h) = (img.width() as i64, img.height() as i64);
        let mut done = vec![false; (w * h) as usize];
        for &t in &triangles {
            let tri = scan.triangles[t];
            let uv = tri.map(|i| {
                let p = scan.uvs[i as usize];
                [p[0] as f64 * w as f64, p[1] as f64 * h as f64]
            });
            let targets: [Option<[f64; 3]>; 3] = tri.map(|i| source[scan.weld.id[i as usize]]);
            let area = (uv[1][0] - uv[0][0]) * (uv[2][1] - uv[0][1])
                - (uv[2][0] - uv[0][0]) * (uv[1][1] - uv[0][1]);
            if area.abs() < 1e-12 {
                continue;
            }
            // One texel of padding: bilinear filtering reads the neighbours.
            let lo = |k: usize| {
                (uv.iter()
                    .map(|p| p[k])
                    .fold(f64::INFINITY, f64::min)
                    .floor() as i64
                    - 1)
                .max(0)
            };
            let hi = |k: usize, m: i64| {
                (uv.iter()
                    .map(|p| p[k])
                    .fold(f64::NEG_INFINITY, f64::max)
                    .ceil() as i64
                    + 1)
                .min(m - 1)
            };
            let pad = 1.0 / area.abs().sqrt().max(1.0) * 1.5;
            for y in lo(1)..=hi(1, h) {
                for x in lo(0)..=hi(0, w) {
                    let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                    let l1 = ((uv[2][0] - px) * (uv[0][1] - py)
                        - (uv[0][0] - px) * (uv[2][1] - py))
                        / area;
                    let l2 = ((uv[0][0] - px) * (uv[1][1] - py)
                        - (uv[1][0] - px) * (uv[0][1] - py))
                        / area;
                    let l0 = 1.0 - l1 - l2;
                    if l0 < -pad || l1 < -pad || l2 < -pad {
                        continue;
                    }
                    let k = (y * w + x) as usize;
                    if done[k] {
                        continue;
                    }
                    done[k] = true;
                    texels += 1;
                    let p = img.get_pixel(x as u32, y as u32);
                    let rgb = [0, 1, 2].map(|j| p[j] as f64 / 255.0);
                    let c = lab(rgb);
                    let s = if pale[t] {
                        skin_like(&c).max(pale_like(&c))
                    } else {
                        skin_like(&c)
                    };
                    if s <= 0.0 {
                        continue;
                    }
                    // Target hair colour: barycentric mix of the vertex sources.
                    let b = [l0, l1, l2].map(|x| x.clamp(0.0, 1.0));
                    let (mut acc, mut wsum) = ([0.0; 3], 0.0);
                    for (j, tc) in targets.iter().enumerate() {
                        if let Some(tc) = tc {
                            for q in 0..3 {
                                acc[q] += tc[q] * b[j];
                            }
                            wsum += b[j];
                        }
                    }
                    let target_lab = if wsum > 1e-6 {
                        acc.map(|x| x / wsum)
                    } else {
                        hair_lab
                    };
                    // Keep the texel's lightness relative to the skin (dark
                    // strand edges stay darker, highlights lighter).
                    let ratio = (c[0] / skin_lab[0].max(1.0)).clamp(0.5, 1.5);
                    let mut out = lab_to_rgb([
                        (target_lab[0] * ratio).clamp(0.0, 100.0),
                        target_lab[1],
                        target_lab[2],
                    ]);
                    for q in 0..3 {
                        out[q] = rgb[q] * (1.0 - s) + out[q] * s;
                    }
                    img.put_pixel(
                        x as u32,
                        y as u32,
                        image::Rgba([
                            (out[0] * 255.0).round() as u8,
                            (out[1] * 255.0).round() as u8,
                            (out[2] * 255.0).round() as u8,
                            p[3],
                        ]),
                    );
                    changed += 1;
                }
            }
        }
    }
    json!({
        "applied": true,
        "hair_lab_median": hair_lab,
        "target_triangles": target.iter().filter(|x| **x).count(),
        "texels_examined": texels,
        "texels_recoloured": changed,
    })
}

/// Hair recolouring for a segmentation: per non-rigid hair piece, its thin
/// (strand-like) triangles, the relabelled strands and the hair surface away
/// from the skin border are recoloured from the piece's own hair colour (the
/// hairline itself keeps the skin's edge colour). Returns the recoloured image copies (by scan
/// image index) and the report.
pub fn recolour_hair(
    scan: &Scan,
    seg: &crate::segment::Segmentation,
    height: f64,
) -> (BTreeMap<usize, RgbaImage>, Value) {
    let mut images = BTreeMap::new();
    let skin_lab: Option<[f64; 3]> =
        serde_json::from_value(seg.report["skin_lab_median"].clone()).ok();
    let threshold = seg.report["skin_delta_e_threshold"].as_f64();
    let (Some(skin_lab), Some(threshold)) = (skin_lab, threshold) else {
        return (
            images,
            json!({"applied": false, "reason": "no skin colour model"}),
        );
    };
    let params = Params::for_height(height);
    let n = scan.triangles.len();
    let relabelled: Vec<bool> = seg.strand.clone();
    // Hair surface two rings or more away from the skin: away from the
    // hairline, where skin-coloured texels are the skin's own edge.
    let incidence = Incidence::new(scan);
    let mut near_skin: Vec<bool> = seg
        .triangle_label
        .iter()
        .map(|l| *l == crate::segment::Label::Skin)
        .collect();
    for _ in 0..2 {
        near_skin = (0..n)
            .into_par_iter()
            .map(|t| near_skin[t] || incidence.around(scan, t).any(|u| near_skin[u]))
            .collect();
    }
    let mut pieces = serde_json::Map::new();
    for (p, name) in seg.pieces.iter().enumerate() {
        if !seg.piece_hair[p] || seg.piece_rigid.get(p).copied().unwrap_or(false) {
            continue;
        }
        let member: Vec<bool> = seg.triangle_piece.iter().map(|x| *x == Some(p)).collect();
        if !member.iter().any(|x| *x) {
            continue;
        }
        let target: Vec<bool> = (0..n)
            .into_par_iter()
            .map(|t| {
                member[t]
                    && (relabelled[t]
                        || !near_skin[t]
                        || probe(scan, t, &params).thickness < params.thin_m)
            })
            .collect();
        // Pale paint is recoloured on all of them (strands and the hair away
        // from the hairline), not at the hairline itself.
        let mut r = recolour(
            scan,
            &target,
            &member,
            &target,
            skin_lab,
            threshold,
            &mut images,
        );
        r["strand_triangles"] = json!(target.iter().filter(|x| **x).count());
        pieces.insert(name.clone(), r);
    }
    let total: u64 = pieces
        .values()
        .filter_map(|r| r["texels_recoloured"].as_u64())
        .sum();
    let report = json!({"applied": total > 0, "texels_recoloured": total, "images": images.keys().collect::<Vec<_>>(), "pieces": pieces});
    (images, report)
}

 pub fn add_hair_materials(
    b: &mut crate::export::Builder,
    scan: &Scan,
    encoded: &BTreeMap<usize, (Vec<u8>, String)>,
    imported: &[usize],
) -> BTreeMap<usize, usize> {
    let mut texture_of: BTreeMap<usize, usize> = BTreeMap::new();
    let mut out = BTreeMap::new();
    for (m, material) in scan.materials.iter().enumerate() {
        let Some(i) = material.base_color else {
            continue;
        };
        let Some((bytes, mime)) = encoded.get(&i) else {
            continue;
        };
        let texture = *texture_of.entry(i).or_insert_with(|| {
            let image = b.add_image(bytes, mime);
            b.add_texture(image)
        });
        let mut json = b.doc.json["materials"][imported[m]].clone();
        json["pbrMetallicRoughness"]["baseColorTexture"]["index"] = json!(texture);
        let name = json["name"].as_str().unwrap_or("material").to_string();
        json["name"] = json!(format!("{name}_hair"));
        out.insert(m, b.add_material(json));
    }
    out
}

/// Encodes recoloured copies like their source images (JPEG stays JPEG).
pub fn encode_images(
    scan: &Scan,
    images: &BTreeMap<usize, RgbaImage>,
) -> anyhow::Result<BTreeMap<usize, (Vec<u8>, String)>> {
    images
        .par_iter()
        .map(|(&i, img)| {
            let mime = scan.images[i].mime.clone();
            let mut bytes = vec![];
            if mime == "image/jpeg" {
                let rgb = image::DynamicImage::ImageRgba8(img.clone()).to_rgb8();
                image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 92)
                    .encode_image(&rgb)?;
            } else {
                img.write_to(
                    &mut std::io::Cursor::new(&mut bytes),
                    image::ImageFormat::Png,
                )?;
                return Ok((i, (bytes, "image/png".to_string())));
            }
            Ok((i, (bytes, mime)))
        })
        .collect()
}

 pub fn lab_to_rgb(c: [f64; 3]) -> [f64; 3] {
    let fy = (c[0] + 16.0) / 116.0;
    let fx = fy + c[1] / 500.0;
    let fz = fy - c[2] / 200.0;
    let inv = |f: f64| {
        if f > 6.0 / 29.0 {
            f * f * f
        } else {
            3.0 * (6.0f64 / 29.0).powi(2) * (f - 4.0 / 29.0)
        }
    };
    let (x, y, z) = (inv(fx) * 0.95047, inv(fy), inv(fz) * 1.08883);
    let r = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let b = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    [r, g, b].map(|l| crate::source::linear_to_srgb(l.clamp(0.0, 1.0)).clamp(0.0, 1.0))
}

/// What `mh-guide-check` shows about skin-coloured strands, per scan triangle:
/// 1 = a strand the pipeline will make hair, 2 = a thin floating layer it
/// keeps as skin (continuous with the skin: an ear rim, or a strand fused
/// into the face).
pub struct GuideFlags {
    pub flag: Vec<u8>,
    /// Flagged regions (index = region id in `region_of`).
    pub regions: Vec<Region>,
    pub region_of: Vec<u32>,
}

/// Strand detection on guidance alone (no body yet): skin is the surface near
/// the head that is not marked hair or garment and matches the colour of the
/// marked skin; hair is what the hair items mark.
pub fn guide_flags(scan: &Scan, resolved: &crate::guide::Resolved) -> GuideFlags {
    let n = scan.triangles.len();
    let height = scan.height();
    let colour = |t: usize| scan.color_at(t, [1.0 / 3.0; 3]).map(lab);
    let marked_skin: Vec<[f64; 3]> = (0..n)
        .filter(|&t| {
            resolved.seed_strong[t]
                && resolved.skin_seed[t].is_some()
                && resolved.garment_seed[t].is_none()
        })
        .step_by(3)
        .filter_map(colour)
        .collect();
    let empty = GuideFlags {
        flag: vec![0; n],
        regions: vec![],
        region_of: vec![u32::MAX; n],
    };
    if marked_skin.len() < 50 {
        return empty;
    }
    let skin_lab: [f64; 3] =
        std::array::from_fn(|k| median(marked_skin.iter().map(|c| c[k]).collect()));
    let de = |c: &[f64; 3]| {
        ((c[0] - skin_lab[0]).powi(2) * 0.5
            + (c[1] - skin_lab[1]).powi(2)
            + (c[2] - skin_lab[2]).powi(2))
        .sqrt()
    };
    // Head area: from below the mouth (≈ chin, plus hanging strands) up.
    let (lo, hi) = scan
        .positions
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), p| {
            (l.min(p[1]), h.max(p[1]))
        });
    let mouth = ["mouth_l", "mouth_r"]
        .iter()
        .filter_map(|k| resolved.landmarks.get(*k))
        .map(|p| p[1])
        .fold(f64::INFINITY, f64::min);
    let floor = if mouth.is_finite() {
        mouth - height * 0.1
    } else {
        hi - (hi - lo) * 0.22
    };
    let hair: Vec<bool> = (0..n)
        .map(|t| {
            resolved.garment_seed[t]
                .is_some_and(|(g, _)| resolved.garments[g].label == crate::segment::Label::Hair)
        })
        .collect();
    let skin: Vec<bool> = (0..n)
        .map(|t| {
            resolved.garment_seed[t].is_none()
                && scan.triangles[t]
                    .iter()
                    .all(|&i| scan.positions[i as usize][1] > floor)
                && colour(t).is_some_and(|c| de(&c) < 14.0 * 1.6)
        })
        .collect();
    let locked: Vec<bool> = (0..n)
        .map(|t| {
            resolved.seed_stroke[t]
                && resolved.skin_seed[t].is_some()
                && resolved.garment_seed[t].is_none()
        })
        .collect();
    let incidence = Incidence::new(scan);
    let d = detect(
        &Input {
            scan,
            candidate: &skin,
            skin: &skin,
            hair: &hair,
            locked: &locked,
            guard: &vec![false; n],
            params: Params::for_height(height),
        },
        &incidence,
    );
    let mut flag = vec![0u8; n];
    let mut region_of = vec![u32::MAX; n];
    let mut regions = vec![];
    for r in d.regions {
        let elongation = r.length_m / r.width_m.max(1e-9);
        let suspicious = !r.verdict.is_hair()
            && r.verdict != Verdict::ConfirmedSkin
            && r.floating >= 0.5
            && elongation >= 3.0
            && r.triangles.len() >= 20;
        if !(r.verdict.is_hair() && r.triangles.len() >= 10) && !suspicious {
            continue;
        }
        let id = regions.len() as u32;
        for &t in &r.triangles {
            flag[t] = if r.verdict.is_hair() { 1 } else { 2 };
            region_of[t] = id;
        }
        regions.push(r);
    }
    // Grown strand triangles count as the strand.
    for (f, s) in flag.iter_mut().zip(&d.strand) {
        if *s && *f == 0 {
            *f = 1;
        }
    }
    GuideFlags {
        flag,
        regions,
        region_of,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scan::RawScan;

    /// Closed box (outward normals) with every face a `steps`² grid, minus
    /// the faces named in `open` (0..6 = −x, +x, −y, +y, −z, +z).
    fn add_box(
        positions: &mut Vec<[f64; 3]>,
        triangles: &mut Vec<[u32; 3]>,
        lo: [f64; 3],
        hi: [f64; 3],
        open: &[usize],
        steps: [usize; 3],
    ) -> std::ops::Range<usize> {
        let first = triangles.len();
        let at = |a: usize, k: usize| lo[a] + (hi[a] - lo[a]) * k as f64 / steps[a] as f64;
        for face in 0..6 {
            if open.contains(&face) {
                continue;
            }
            let axis = face / 2;
            let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
            let side = if face % 2 == 1 { steps[axis] } else { 0 };
            let base = positions.len() as u32;
            let (su, sw) = (steps[u], steps[w]);
            for i in 0..=su {
                for j in 0..=sw {
                    let mut p = [0.0; 3];
                    p[axis] = at(axis, side);
                    p[u] = at(u, i);
                    p[w] = at(w, j);
                    positions.push(p);
                }
            }
            let id = |i: usize, j: usize| base + (i * (sw + 1) + j) as u32;
            for i in 0..su {
                for j in 0..sw {
                    let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
                    // u × w points along +axis: keep for the + face.
                    if face % 2 == 1 {
                        triangles.extend([[a, b, c], [a, c, d]]);
                    } else {
                        triangles.extend([[a, c, b], [a, d, c]]);
                    }
                }
            }
        }
        first..triangles.len()
    }

    struct Scene {
        scan: Scan,
        slab: std::ops::Range<usize>,
        strand: std::ops::Range<usize>,
        hair: std::ops::Range<usize>,
        ear: std::ops::Range<usize>,
        ear_root: std::ops::Range<usize>,
    }

    /// A face (thick slab facing +z), a 3 mm strand floating 5 mm in front
    /// of it that hangs from a hair block, and an ear-like flap of the same
    /// size continuous with a skin block.
    fn scene() -> Scene {
        let (mut p, mut t) = (vec![], vec![]);
        let four = [4, 4, 4];
        let slab = add_box(
            &mut p,
            &mut t,
            [-0.1, 1.3, -0.1],
            [0.1, 1.6, 0.0],
            &[],
            four,
        );
        let strand = add_box(
            &mut p,
            &mut t,
            [-0.003, 1.45, 0.005],
            [0.003, 1.55, 0.008],
            &[3],
            four,
        );
        let hair = add_box(
            &mut p,
            &mut t,
            [-0.003, 1.55, 0.005],
            [0.003, 1.6, 0.008],
            &[2],
            four,
        );
        let ear = add_box(
            &mut p,
            &mut t,
            [0.05, 1.45, 0.005],
            [0.056, 1.55, 0.008],
            &[3],
            four,
        );
        // The ear's root: solid skin 4.5 cm deep, its lattice continuing the
        // ear's (0.75 mm in z) so they share vertices.
        let ear_root = add_box(
            &mut p,
            &mut t,
            [0.05, 1.55, 0.005],
            [0.056, 1.6, 0.05],
            &[],
            [4, 4, 60],
        );
        let n = t.len();
        let scan = RawScan {
            positions: p,
            uvs: vec![[0.0; 2]; 0],
            triangle_material: vec![None; n],
            triangles: t,
            images: vec![],
            materials: vec![],
            textures: vec![],
            samplers: vec![],
            source_hash: String::new(),
        };
        let count = scan.positions.len();
        let scan = RawScan {
            uvs: vec![[0.0; 2]; count],
            ..scan
        }
        .into_scan(nalgebra::Matrix4::identity());
        Scene {
            scan,
            slab,
            strand,
            hair,
            ear,
            ear_root,
        }
    }

    #[test]
    fn probe_measures_strand_thickness_and_gap() {
        let s = scene();
        let params = Params::for_height(1.75);
        // The strand's +z face: 3 mm thick, 5 mm in front of the slab.
        let front = s
            .strand
            .clone()
            .find(|&t| {
                face_normal(&s.scan.positions, &s.scan.triangles[t])
                    .normalize()
                    .z
                    > 0.5
            })
            .unwrap();
        let p = probe(&s.scan, front, &params);
        assert!((p.thickness - 0.003).abs() < 1e-4, "{p:?}");
        assert!((p.gap - 0.005).abs() < 1e-4, "{p:?}");
        // Its −z face looks at the slab along its normal.
        let back = s
            .strand
            .clone()
            .find(|&t| {
                face_normal(&s.scan.positions, &s.scan.triangles[t])
                    .normalize()
                    .z
                    < -0.5
            })
            .unwrap();
        let p = probe(&s.scan, back, &params);
        assert!(p.thickness < 0.004 && (p.gap - 0.005).abs() < 1e-4, "{p:?}");
        // The slab's face is solid.
        let face = s
            .slab
            .clone()
            .find(|&t| {
                face_normal(&s.scan.positions, &s.scan.triangles[t])
                    .normalize()
                    .z
                    > 0.5
            })
            .unwrap();
        assert!(probe(&s.scan, face, &params).thickness >= params.thin_m);
    }

    #[test]
    fn strands_hanging_from_hair_become_hair_and_ear_like_flaps_stay_skin() {
        let s = scene();
        let n = s.scan.triangles.len();
        let mut hair = vec![false; n];
        s.hair.clone().for_each(|t| hair[t] = true);
        let skin: Vec<bool> = (0..n).map(|t| !hair[t]).collect();
        let candidate: Vec<bool> = (0..n)
            .map(|t| s.strand.contains(&t) || s.ear.contains(&t) || s.slab.contains(&t))
            .collect();
        let none = vec![false; n];
        let input = Input {
            scan: &s.scan,
            candidate: &candidate,
            skin: &skin,
            hair: &hair,
            locked: &none,
            guard: &none,
            params: Params::for_height(1.75),
        };
        let incidence = Incidence::new(&s.scan);
        let d = detect(&input, &incidence);
        let facing = |t: usize| {
            face_normal(&s.scan.positions, &s.scan.triangles[t])
                .normalize()
                .z
                .abs()
                > 0.5
        };
        assert_eq!(s.strand.clone().filter(|&t| facing(t)).count(), 64);
        assert!(s.strand.clone().filter(|&t| facing(t)).all(|t| d.strand[t]));
        assert!(s.ear.clone().all(|t| !d.strand[t]));
        assert!(
            s.slab
                .clone()
                .chain(s.ear_root.clone())
                .all(|t| !d.strand[t])
        );
        assert!(
            d.regions
                .iter()
                .any(|r| r.verdict == Verdict::ContinuousWithSkin
                    && r.triangles.iter().all(|t| s.ear.contains(t)))
        );
         
        let locked: Vec<bool> = (0..n).map(|t| s.strand.contains(&t)).collect();
        let d = detect(
            &Input {
                locked: &locked,
                ..input
            },
            &incidence,
        );
        assert!(d.strand.iter().all(|x| !x));
        assert!(
            d.regions
                .iter()
                .any(|r| r.verdict == Verdict::ConfirmedSkin)
        );
    }

    #[test]
    fn verdict_needs_hair_contact_or_a_loose_elongated_shell() {
        let p = Params::for_height(1.75);
        let region = |hair_contact, skin_contact, length_m, width_m, floating| Region {
            triangles: (0..40).collect(),
            hair_contact,
            skin_contact,
            locked: 0,
            length_m,
            width_m,
            floating,
            median_gap_m: 0.005,
            median_thickness_m: 0.003,
            centroid: [0.0; 3],
            verdict: Verdict::Compact,
        };
        assert_eq!(
            verdict(&region(30, 10, 0.05, 0.005, 1.0), &p),
            Verdict::HangsFromHair
        );
        // An ear rim: a little hair on top, skin all along.
        assert_eq!(
            verdict(&region(20, 200, 0.05, 0.02, 1.0), &p),
            Verdict::ContinuousWithSkin
        );
        assert_eq!(
            verdict(&region(0, 0, 0.06, 0.004, 0.8), &p),
            Verdict::DetachedShell
        );
        // Short or not floating: teeth, lashes, eye parts.
        assert_eq!(
            verdict(&region(0, 0, 0.01, 0.002, 1.0), &p),
            Verdict::Compact
        );
        assert_eq!(
            verdict(&region(0, 0, 0.06, 0.004, 0.2), &p),
            Verdict::Compact
        );
        assert_eq!(
            verdict(&region(0, 0, 0.01, 0.008, 0.0), &p),
            Verdict::Compact
        );
    }

    #[test]
    fn lab_round_trips() {
        for rgb in [[0.8, 0.6, 0.5], [0.1, 0.05, 0.02], [0.95, 0.9, 0.85]] {
            let back = lab_to_rgb(lab(rgb));
            for k in 0..3 {
                assert!((back[k] - rgb[k]).abs() < 2e-3, "{rgb:?} → {back:?}");
            }
        }
    }
}
