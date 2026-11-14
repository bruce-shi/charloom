 use crate::geom::{V, Weld, a, v, vertex_normals};
use crate::scan::build_query;
use anyhow::{Context, Result, bail, ensure};
use charloom_core::{Influence, Vec3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// `Config::underwear`: `"auto"` (by sex: female sport bra + briefs, male
/// trunks), `"none"`, or a list of asset ids from [`CATALOG`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Underwear {
    Mode(UnderwearMode),
    Assets(Vec<String>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum UnderwearMode {
    Auto,
    None,
}

impl Default for Underwear {
    fn default() -> Self {
        Underwear::Mode(UnderwearMode::Auto)
    }
}

/// Assets known to carry a usable licence (checked again from each file's
/// header): id and MakeHuman community asset pack.
pub const CATALOG: &[(&str, &str)] = &[
    ("punkduck_sport_bra", "underwear02"),
    ("punkduck_sport_briefs", "underwear02"),
    ("mindfront_male_swimming_trunks_03", "pants03"),
    ("wolgade_female_top_01", "underwear01"),
    ("wolgade_female_panties_01", "underwear01"),
    ("punkduck_bandeau_bikini_bra", "underwear02"),
    ("punkduck_bandeau_bikini_slip", "underwear02"),
];
pub const FEMALE: &[&str] = &["punkduck_sport_bra", "punkduck_sport_briefs"];
pub const MALE: &[&str] = &["mindfront_male_swimming_trunks_03"];
const LICENCES: &[(&str, &str)] = &[
    ("CC0", "https://creativecommons.org/publicdomain/zero/1.0/"),
    ("CC BY 3.0", "https://creativecommons.org/licenses/by/3.0/"),
    ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
];
/// Vertices of the hm08 body group (helpers follow).
pub const HM08_BODY: usize = 13380;
/// MakeHuman units (decimetres) to metres: MPFB's scale factor.
const MH_TO_M: f64 = 0.1;

/// One garment vertex of an MHCLO file.
#[derive(Debug, Clone, PartialEq)]
pub struct Ref {
    pub verts: [usize; 3],
    pub weights: [f64; 3],
    /// MakeHuman units, MakeHuman axes (Y up, +Z front: the glTF axes).
    pub offset: [f64; 3],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Mhclo {
    pub name: String,
    pub author: String,
    pub license: String,
    pub obj_file: String,
    pub material: Option<String>,
    /// Per axis (x, y, z): reference vertex pair and their base-mesh distance.
    pub scale: [Option<(usize, usize, f64)>; 3],
    pub refs: Vec<Ref>,
    pub delete_verts: usize,
}

 pub fn parse_mhclo(text: &str) -> Result<Mhclo> {
    let mut m = Mhclo {
        name: String::new(),
        author: String::new(),
        license: String::new(),
        obj_file: String::new(),
        material: None,
        scale: [None; 3],
        refs: vec![],
        delete_verts: 0,
    };
    #[derive(PartialEq)]
    enum Block {
        Header,
        Verts,
        Delete,
    }
    let mut block = Block::Header;
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(c) = line.strip_prefix('#') {
            let c = c.trim();
            if let Some(x) = c.strip_prefix("author ") {
                m.author = x.trim().into();
            } else if let Some(x) = c.strip_prefix("license ") {
                m.license = x.trim().into();
            }
            continue;
        }
        let words: Vec<&str> = line.split_whitespace().collect();
        let numeric = words[0].parse::<f64>().is_ok();
        if block == Block::Verts && numeric {
            let f: Vec<f64> = words
                .iter()
                .map(|w| w.parse::<f64>())
                .collect::<std::result::Result<_, _>>()
                .with_context(|| format!("mhclo line {}: {line}", n + 1))?;
            let idx = |x: f64| -> Result<usize> {
                ensure!(
                    x >= 0.0 && x.fract() == 0.0,
                    "mhclo line {}: bad vertex",
                    n + 1
                );
                Ok(x as usize)
            };
            m.refs.push(match f.len() {
                1 => Ref {
                    verts: [idx(f[0])?; 3],
                    weights: [1.0, 0.0, 0.0],
                    offset: [0.0; 3],
                },
                9 => Ref {
                    verts: [idx(f[0])?, idx(f[1])?, idx(f[2])?],
                    weights: [f[3], f[4], f[5]],
                    offset: [f[6], f[7], f[8]],
                },
                k => bail!("mhclo line {}: {k} numbers", n + 1),
            });
            continue;
        }
        if block == Block::Delete && numeric {
            m.delete_verts += words.len();
            continue;
        }
        match words[0] {
            "verts" => block = Block::Verts,
            "delete_verts" => block = Block::Delete,
            "name" => m.name = words[1..].join(" "),
            "obj_file" => m.obj_file = words[1..].join(" "),
            "material" => m.material = Some(words[1..].join(" ")),
            k @ ("x_scale" | "y_scale" | "z_scale") => {
                ensure!(words.len() == 4, "mhclo line {}: {line}", n + 1);
                let axis = (k.as_bytes()[0] - b'x') as usize;
                m.scale[axis] = Some((words[1].parse()?, words[2].parse()?, words[3].parse()?));
            }
            _ => block = Block::Header,
        }
    }
    ensure!(!m.refs.is_empty(), "mhclo has no vertices");
    ensure!(!m.obj_file.is_empty(), "mhclo has no obj_file");
    Ok(m)
}

/// Garment positions (metres) on a body given by hm08 vertex positions.
pub fn place(m: &Mhclo, body: &dyn Fn(usize) -> Option<V>) -> Result<Vec<V>> {
    let get = |i: usize| body(i).with_context(|| format!("hm08 vertex {i} not available"));
    let mut scale = [MH_TO_M; 3];
    for (k, s) in m.scale.iter().enumerate() {
        if let Some((p, q, den)) = *s {
            ensure!(den > 0.0, "zero scale reference");
            scale[k] = (get(p)?[k] - get(q)?[k]).abs() / den;
        }
    }
    m.refs
        .iter()
        .map(|r| {
            let mut p = V::zeros();
            for k in 0..3 {
                if r.weights[k] != 0.0 {
                    p += get(r.verts[k])? * r.weights[k];
                }
            }
            Ok(p + V::new(
                r.offset[0] * scale[0],
                r.offset[1] * scale[1],
                r.offset[2] * scale[2],
            ))
        })
        .collect()
}

/// A face corner: vertex and texture coordinate indices (0-based).
pub type Corner = (usize, Option<usize>);

/// A Wavefront OBJ: positions (file units) and faces as `(v, vt)` corners.
pub struct Obj {
    pub positions: Vec<V>,
    pub uvs: Vec<[f64; 2]>,
         pub faces: Vec<(String, Vec<Corner>)>,
}

pub fn parse_obj(text: &str) -> Result<Obj> {
    let mut o = Obj {
        positions: vec![],
        uvs: vec![],
        faces: vec![],
    };
    let mut group = String::new();
    for line in text.lines() {
        let w: Vec<&str> = line.split_whitespace().collect();
        match w.first().copied() {
            Some("v") => o
                .positions
                .push(V::new(w[1].parse()?, w[2].parse()?, w[3].parse()?)),
            Some("vt") => o.uvs.push([w[1].parse()?, w[2].parse()?]),
            Some("g") => group = w.get(1).copied().unwrap_or("").into(),
            Some("f") => {
                let corners = w[1..]
                    .iter()
                    .map(|c| {
                        let mut it = c.split('/');
                        let vi: usize = it.next().context("face corner")?.parse()?;
                        let ti = it.next().filter(|s| !s.is_empty()).map(str::parse::<usize>);
                        Ok((vi - 1, ti.transpose()?.map(|t| t - 1)))
                    })
                    .collect::<Result<Vec<_>>>()?;
                ensure!(corners.len() >= 3, "face with {} corners", corners.len());
                o.faces.push((group.clone(), corners));
            }
            _ => {}
        }
    }
    Ok(o)
}

/// Render mesh of an OBJ: one vertex per `(v, vt)` pair (glTF UVs, V down),
/// fan-triangulated faces, and the OBJ vertex of every render vertex.
pub fn render_mesh(o: &Obj) -> (Vec<usize>, Vec<[f32; 2]>, Vec<[u32; 3]>) {
    let mut key: HashMap<(usize, Option<usize>), u32> = HashMap::new();
    let (mut source, mut uvs, mut triangles) = (vec![], vec![], vec![]);
    for (_, f) in &o.faces {
        let ids: Vec<u32> = f
            .iter()
            .map(|&c| {
                *key.entry(c).or_insert_with(|| {
                    source.push(c.0);
                    let t = c.1.and_then(|t| o.uvs.get(t)).copied().unwrap_or([0.0; 2]);
                    uvs.push([t[0] as f32, (1.0 - t[1]) as f32]);
                    source.len() as u32 - 1
                })
            })
            .collect();
        for k in 1..ids.len() - 1 {
            triangles.push([ids[0], ids[k], ids[k + 1]]);
        }
    }
    (source, uvs, triangles)
}

/// The prior body's glTF vertices as hm08 vertices.
pub struct Hm08 {
         pub to_prior: Vec<Option<usize>>,
    /// Base mesh, metres (all vertices, helpers included).
    base: Vec<V>,
    base_triangles: Vec<[u32; 3]>,
    pub report: Value,
}

impl Hm08 {
    /// `uvs`/`triangles`: the prior body's glTF UVs (V down) and triangles.
    pub fn new(base: &Obj, uvs: &[[f32; 2]], triangles: &[[u32; 3]]) -> Result<Self> {
         
        let mut vt_vertex: HashMap<usize, usize> = HashMap::new();
        let mut edges: HashSet<(usize, usize)> = HashSet::new();
        let mut base_triangles = vec![];
        for (_, f) in base.faces.iter().filter(|(g, _)| g == "body") {
            for &(vi, ti) in f {
                if let Some(t) = ti {
                    vt_vertex.insert(t, vi);
                }
            }
            // Every pair in a face (a quad's diagonals included).
            for i in 0..f.len() {
                for j in 0..f.len() {
                    if i != j {
                        edges.insert((f[i].0, f[j].0));
                    }
                }
            }
            for k in 1..f.len() - 1 {
                base_triangles.push([f[0].0 as u32, f[k].0 as u32, f[k + 1].0 as u32]);
            }
        }
        ensure!(
            !vt_vertex.is_empty(),
            "base mesh has no body group with UVs"
        );
        const CELL: f64 = 1e-4;
        let cell = |u: f64, w: f64| ((u / CELL).floor() as i64, (w / CELL).floor() as i64);
        let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for &t in vt_vertex.keys() {
            let [u, w] = base.uvs[t];
            grid.entry(cell(u, w)).or_default().push(t);
        }
        let mut to_hm = vec![None; uvs.len()];
        let mut missed = 0;
        for (i, uv) in uvs.iter().enumerate() {
            let (u, w) = (uv[0] as f64, 1.0 - uv[1] as f64);
            let (cu, cw) = cell(u, w);
            let mut best: Option<(f64, usize)> = None;
            for du in -1..=1 {
                for dw in -1..=1 {
                    for &t in grid.get(&(cu + du, cw + dw)).into_iter().flatten() {
                        let d = (base.uvs[t][0] - u).hypot(base.uvs[t][1] - w);
                        if d < 2e-4 && best.is_none_or(|b| d < b.0) {
                            best = Some((d, t));
                        }
                    }
                }
            }
            match best {
                Some((_, t)) => to_hm[i] = Some(vt_vertex[&t]),
                None => missed += 1,
            }
        }
        // The glTF triangles must be hm08 faces.
        let bad = triangles
            .iter()
            .filter(|t| {
                let h: Vec<Option<usize>> = t.iter().map(|&i| to_hm[i as usize]).collect();
                match (h[0], h[1], h[2]) {
                    (Some(x), Some(y), Some(z)) => {
                        !(edges.contains(&(x, y)) && edges.contains(&(y, z)))
                    }
                    _ => true,
                }
            })
            .count();
        ensure!(
            missed * 1000 <= uvs.len() && bad * 1000 <= triangles.len(),
            "prior body does not match MakeHuman's base mesh: {missed} unmatched vertices, {bad} non-hm08 triangles"
        );
        let mut to_prior = vec![None; HM08_BODY];
        for (i, h) in to_hm.iter().enumerate() {
            if let Some(h) = *h
                && h < HM08_BODY
                && to_prior[h].is_none()
            {
                to_prior[h] = Some(i);
            }
        }
        let covered = to_prior.iter().filter(|x| x.is_some()).count();
        Ok(Self {
            to_prior,
            base: base.positions.iter().map(|p| p * MH_TO_M).collect(),
            base_triangles,
            report: json!({"prior_vertices": uvs.len(), "unmatched": missed, "non_hm08_triangles": bad, "hm08_body_vertices_covered": covered}),
        })
    }

    /// hm08 positions on a body given in prior glTF vertex order. Helpers
    /// in `wanted` are bound to the base body surface (closest point plus
    /// offset) and evaluated on the body.
    pub fn lookup(&self, body: &[Vec3], wanted: &HashSet<usize>) -> HashMap<usize, V> {
        let mut out: HashMap<usize, V> = HashMap::new();
        for (h, p) in self.to_prior.iter().enumerate() {
            if let Some(p) = p {
                out.insert(h, v(body[*p]));
            }
        }
        let helpers: Vec<usize> = wanted
            .iter()
            .copied()
            .filter(|&h| h >= HM08_BODY && h < self.base.len())
            .collect();
        if helpers.is_empty() {
            return out;
        }
        let base: Vec<Vec3> = self.base.iter().map(|p| a(*p)).collect();
        let query = build_query(&base, &self.base_triangles);
        for h in helpers {
            let Some(hit) = query.closest_point(base[h], 1.0, None) else {
                continue;
            };
            let t = self.base_triangles[hit.source[1]];
            let on = t
                .iter()
                .zip(hit.barycentric)
                .try_fold(V::zeros(), |s, (&i, w)| {
                    out.get(&(i as usize)).map(|p| s + p * w)
                });
            if let Some(on) = on {
                out.insert(h, on + (self.base[h] - v(hit.point)));
            }
        }
        out
    }
}

/// A loaded, licence-checked asset.
pub struct Asset {
    pub id: String,
    pub pack: String,
    pub mhclo: Mhclo,
    pub obj: Obj,
    pub diffuse: Option<(Vec<u8>, String)>,
    pub normal: Option<(Vec<u8>, String)>,
    pub sha256: Vec<(String, String)>,
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

impl Asset {
    pub fn source_page(&self) -> String {
        format!(
            "https://static.makehumancommunity.org/assets/assetpacks/{}.html",
            self.pack
        )
    }

    pub fn load(root: &Path, id: &str) -> Result<Self> {
        let pack = CATALOG
            .iter()
            .find(|c| c.0 == id)
            .with_context(|| {
                format!(
                    "unknown underwear asset {id}; known: {:?}",
                    CATALOG.iter().map(|c| c.0).collect::<Vec<_>>()
                )
            })?
            .1;
        let dir = root.join(pack).join("clothes").join(id);
        let read = |name: &str| {
            std::fs::read(dir.join(name))
                .with_context(|| format!("read {}", dir.join(name).display()))
        };
        let mut sha256 = vec![];
        let file = format!("{id}.mhclo");
        let bytes = read(&file)?;
        sha256.push((file, sha(&bytes)));
        let mhclo = parse_mhclo(&String::from_utf8_lossy(&bytes))?;
        ensure!(
            LICENCES.iter().any(|l| l.0 == mhclo.license),
            "{id}: licence {:?} is not one of {:?}",
            mhclo.license,
            LICENCES.iter().map(|l| l.0).collect::<Vec<_>>()
        );
        let bytes = read(&mhclo.obj_file)?;
        sha256.push((mhclo.obj_file.clone(), sha(&bytes)));
        let obj = parse_obj(&String::from_utf8_lossy(&bytes))?;
        ensure!(
            obj.positions.len() == mhclo.refs.len(),
            "{id}: {} obj vertices, {} mhclo vertices",
            obj.positions.len(),
            mhclo.refs.len()
        );
        let (mut diffuse, mut normal) = (None, None);
        if let Some(mat) = &mhclo.material {
            let bytes = read(mat)?;
            sha256.push((mat.clone(), sha(&bytes)));
            for line in String::from_utf8_lossy(&bytes).lines() {
                let w: Vec<&str> = line.split_whitespace().collect();
                let slot = match w.first().copied() {
                    Some("diffuseTexture") => &mut diffuse,
                    Some("normalmapTexture") => &mut normal,
                    _ => continue,
                };
                let Some(name) = w.get(1) else { continue };
                let name = Path::new(name)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(name);
                let bytes = read(name)?;
                sha256.push((name.to_string(), sha(&bytes)));
                let mime = if name.to_lowercase().ends_with(".png") {
                    "image/png"
                } else {
                    "image/jpeg"
                };
                *slot = Some((bytes, mime.to_string()));
            }
        }
         
        if let Ok(p) = std::fs::read(root.join("provenance.json"))
            && let Ok(p) = serde_json::from_slice::<Value>(&p)
            && let Some(row) = p["assets"]
                .as_array()
                .and_then(|r| r.iter().find(|x| x["asset"] == id))
        {
            for (name, hash) in &sha256 {
                if let Some(want) = row["files_sha256"][name].as_str() {
                    ensure!(want == hash, "{id}: {name} differs from provenance.json");
                }
            }
        }
        Ok(Self {
            id: id.into(),
            pack: pack.into(),
            mhclo,
            obj,
            diffuse,
            normal,
            sha256,
        })
    }
}

/// Where the asset packs and MakeHuman's base mesh live: `explicit`, else
/// `$CHARLOOM_UNDERWEAR_ASSETS`, else `out/tools/mpfb/underwear-candidates`
/// in an ancestor of `near` (the prior; the checkout holding it).
pub fn find_assets(explicit: Option<&Path>, near: &Path) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p.to_path_buf());
    }
    if let Some(p) = std::env::var_os("CHARLOOM_UNDERWEAR_ASSETS") {
        return Some(PathBuf::from(p));
    }
    let near = std::fs::canonicalize(near).unwrap_or_else(|_| near.to_path_buf());
    near.ancestors().find_map(|d| {
        [
            "out/tools/mpfb/underwear-candidates",
            "tools/mpfb/underwear-candidates",
        ]
        .iter()
        .map(|r| d.join(r))
        .find(|p| p.is_dir())
    })
}

/// MakeHuman's `base.obj`: in the assets folder, else in the MPFB source
/// unpacked next to it (`out/tools/mpfb/source/mpfb2-*/src/mpfb/data/3dobjs`).
pub fn find_base_mesh(assets: &Path) -> Option<PathBuf> {
    let direct = assets.join("base.obj");
    if direct.is_file() {
        return Some(direct);
    }
    let source = assets.parent()?.join("source");
    let mut found: Vec<PathBuf> = std::fs::read_dir(source)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path().join("src/mpfb/data/3dobjs/base.obj"))
        .filter(|p| p.is_file())
        .collect();
    found.sort();
    found.pop()
}

/// Asset ids for `choice`; `sex` from the guidance or the prior.
pub fn choose(choice: &Underwear, sex: &str) -> Vec<String> {
    match choice {
        Underwear::Mode(UnderwearMode::None) => vec![],
        Underwear::Mode(UnderwearMode::Auto) => if sex == "female" { FEMALE } else { MALE }
            .iter()
            .map(|s| s.to_string())
            .collect(),
        Underwear::Assets(list) => list.clone(),
    }
}

/// The prior's sex: `macro_parameters.gender` of `preparation.json` next
/// to it (0 female, 1 male).
pub fn prior_sex(prior: &Path) -> Option<&'static str> {
    let p = prior.parent()?.join("preparation.json");
    let r: Value = serde_json::from_slice(&std::fs::read(p).ok()?).ok()?;
    let g = r["macro_parameters"]["gender"].as_f64()?;
    Some(if g < 0.5 { "female" } else { "male" })
}

/// One fitted garment on the body, ready to export.
pub struct Piece {
    pub name: String,
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub uvs: Vec<[f32; 2]>,
    pub triangles: Vec<[u32; 3]>,
    /// Compact joints, at most four per vertex.
    pub influences: Vec<Vec<Influence>>,
         pub targets: Vec<(String, Vec<Vec3>)>,
    pub asset: usize,
}

/// The body the underwear is fitted to (prior glTF vertex order).
pub struct BodyInput<'a> {
    pub positions: &'a [Vec3],
    pub triangles: &'a [[u32; 3]],
    /// Compact joints.
    pub influences: &'a [Vec<Influence>],
    /// Size-morph displacements of the body, by name.
    pub size_targets: &'a [(String, Vec<Vec3>)],
    pub height: f64,
}

/// Weighted sum of influence rows, four strongest kept, normalised.
fn blend<'a>(rows: impl Iterator<Item = (&'a [Influence], f64)>) -> Vec<Influence> {
    let mut acc: HashMap<usize, f64> = HashMap::new();
    for (row, w) in rows.filter(|r| r.1 > 0.0) {
        for i in row {
            *acc.entry(i.joint).or_default() += i.weight * w;
        }
    }
    let mut rows: Vec<Influence> = acc
        .into_iter()
        .filter(|x| x.1 > 0.0)
        .map(|(joint, weight)| Influence { joint, weight })
        .collect();
    rows.sort_by(|x, y| y.weight.total_cmp(&x.weight).then(x.joint.cmp(&y.joint)));
    rows.truncate(4);
    let s: f64 = rows.iter().map(|i| i.weight).sum();
    rows.iter_mut().for_each(|i| i.weight /= s);
    rows
}

/// Garment weights: the body's weights at the closest skin point (the
/// fabric moves with the skin right under it, so a tight garment does not
/// open up to the skin in motion); beyond `reach`, the blend of the MHCLO
/// reference vertices' weights (MakeHuman's proxy rule).
fn garment_weights(
    positions: &[V],
    refs: &[&Ref],
    hm: &Hm08,
    body: &BodyInput,
    reach: f64,
) -> Vec<Vec<Influence>> {
    let query = build_query(body.positions, body.triangles);
    positions
        .iter()
        .zip(refs)
        .map(|(p, r)| {
            let surface = query.closest_point(a(*p), reach, None).map(|hit| {
                let t = body.triangles[hit.source[1]];
                blend((0..3).map(|k| {
                    (
                        body.influences[t[k] as usize].as_slice(),
                        hit.barycentric[k],
                    )
                }))
            });
            match surface {
                Some(w) if !w.is_empty() => w,
                _ => blend((0..3).filter_map(|k| {
                    let row = hm.to_prior.get(r.verts[k]).copied().flatten()?;
                    Some((body.influences.get(row)?.as_slice(), r.weights[k]))
                })),
            }
        })
        .collect()
}

 fn lift(d: f64, c: f64) -> f64 {
    if d >= 2.0 * c {
        d
    } else {
        c + c * ((d - 2.0 * c) / c).exp()
    }
}

/// Distance from each garment point inward (against the garment's own
/// vertex normal) to the body, within `reach`: how far the fabric hangs
/// off it. Closest points missed a band hanging under a breast (the breast's
/// underside is near; the ribs it should hug are not).
fn inward_gaps(points: &[V], triangles: &[[u32; 3]], body: &[Vec3], body_triangles: &[[u32; 3]], reach: f64) -> Vec<Option<f64>> {
    let query = build_query(body, body_triangles);
    let pos: Vec<Vec3> = points.iter().map(|p| a(*p)).collect();
    let normals = vertex_normals(&pos, triangles, &Weld::new(&pos, 1e-7));
    points
        .iter()
        .zip(&normals)
        .map(|(p, n)| {
            let d = -v(*n);
            if d.norm() < 1e-9 {
                return None;
            }
            // From just outside, so a point a hair inside still finds it.
            let start = p - d * 0.002;
            query.first_ray_distance(a(start), a(d), reach).map(|t| t - 0.002)
        })
        .collect()
}

/// Floating fabric drawn back to the body. The MHCLO offsets are tuned to
/// MakeHuman's own body; on a fitted body with fuller or lower breasts the
/// sport bra's band hung off the ribs under them. A vertex that hangs
/// further off the fitted body (inward along its normal) than it did off
/// the unfitted prior (`designed`) by more than SLACK is pulled in to that
/// gap; the pulls are smoothed over the garment so its pattern holds.
/// Returns the pulls.
const SLACK: f64 = 0.003;
fn conform_pass(
    garment: &mut [V],
    triangles: &[[u32; 3]],
    designed: &[Option<f64>],
    body: &[Vec3],
    body_triangles: &[[u32; 3]],
    reach: f64,
) -> Vec<V> {
    let now = inward_gaps(garment, triangles, body, body_triangles, reach);
    let pos: Vec<Vec3> = garment.iter().map(|p| a(*p)).collect();
    let gnormals = vertex_normals(&pos, triangles, &Weld::new(&pos, 1e-7));
    let need: Vec<V> = now
        .iter()
        .zip(designed)
        .zip(&gnormals)
        .map(|((n, d), nrm)| match (n, d) {
            (Some(g), Some(g0)) if *g > g0.max(0.0) + SLACK => -v(*nrm) * (g - g0.max(0.0) - SLACK * 0.5),
            _ => V::zeros(),
        })
        .collect();
    let weld = Weld::new(&pos, 1e-7);
    let adjacency = crate::geom::adjacency(triangles, &weld);
    let mut pull: Vec<V> = vec![V::zeros(); weld.count];
    for (i, n) in need.iter().enumerate() {
        if n.norm() > pull[weld.id[i]].norm() {
            pull[weld.id[i]] = *n;
        }
    }
    // Smooth: the mean of a vertex and its neighbours, a few times.
    for _ in 0..6 {
        pull = (0..weld.count)
            .map(|w| {
                let nb = &adjacency[w];
                let s = nb.iter().fold(pull[w], |s, &j| s + pull[j as usize]);
                s / (nb.len() + 1) as f64
            })
            .collect();
    }
    let out: Vec<V> = weld.id.iter().map(|&w| pull[w]).collect();
    for (p, d) in garment.iter_mut().zip(&out) {
        *p += d;
    }
    out
}

/// Closest-point companion of [`conform_pass`]: a vertex whose nearest body
/// point is further than on the unfitted prior (`designed`, also nearest
/// points) by more than SLACK is drawn towards it. The inward rays of
/// `inward_gaps` miss a rolled waistband's sides (their normals run along
/// the hip), so the briefs' band stood off the hips like wings.
fn conform_closest(
    garment: &mut [V],
    triangles: &[[u32; 3]],
    designed: &[Option<f64>],
    body: &[Vec3],
    body_triangles: &[[u32; 3]],
    reach: f64,
) -> Vec<V> {
    let query = build_query(body, body_triangles);
    let need: Vec<V> = garment
        .iter()
        .zip(designed)
        .map(|(p, d0)| {
            let (Some(h), Some(d0)) = (query.closest_point(a(*p), reach, None), d0) else {
                return V::zeros();
            };
            let to = v(h.point) - p;
            let g = to.norm();
            if g > d0.max(0.0) + SLACK {
                to / g * (g - d0.max(0.0) - SLACK * 0.5)
            } else {
                V::zeros()
            }
        })
        .collect();
    let pos: Vec<Vec3> = garment.iter().map(|p| a(*p)).collect();
    let weld = Weld::new(&pos, 1e-7);
    let adjacency = crate::geom::adjacency(triangles, &weld);
    let mut pull: Vec<V> = vec![V::zeros(); weld.count];
    for (i, n) in need.iter().enumerate() {
        if n.norm() > pull[weld.id[i]].norm() {
            pull[weld.id[i]] = *n;
        }
    }
    for _ in 0..4 {
        pull = (0..weld.count)
            .map(|w| {
                let nb = &adjacency[w];
                let s = nb.iter().fold(pull[w], |s, &j| s + pull[j as usize]);
                s / (nb.len() + 1) as f64
            })
            .collect();
    }
    let out: Vec<V> = weld.id.iter().map(|&w| pull[w]).collect();
    for (p, d) in garment.iter_mut().zip(&out) {
        *p += d;
    }
    out
}

/// Pushes garment vertices out to `clearance` above the body along the
/// body's interpolated normal (within `reach`; [`lift`]), then spreads the
/// push a little over the garment so the fabric stays smooth. Returns the
/// pushes.
fn clearance_pass(
    garment: &mut [V],
    triangles: &[[u32; 3]],
    body: &[Vec3],
    body_triangles: &[[u32; 3]],
    clearance: f64,
    reach: f64,
) -> Vec<V> {
    let query = build_query(body, body_triangles);
    let normals = vertex_normals(body, body_triangles, &Weld::new(body, 1e-7));
    let need: Vec<V> = garment
        .iter()
        .map(|p| {
            let Some(hit) = query.closest_point(a(*p), reach, None) else {
                return V::zeros();
            };
            let t = body_triangles[hit.source[1]];
            let n = (0..3)
                .fold(V::zeros(), |s, j| {
                    s + v(normals[t[j] as usize]) * hit.barycentric[j]
                })
                .try_normalize(1e-12)
                .unwrap_or_else(|| v(hit.normal));
            let d = (p - v(hit.point)).dot(&n);
            n * (lift(d, clearance) - d)
        })
        .collect();
    // Spread: a vertex next to pushed ones moves at least half their mean push.
    let pos: Vec<Vec3> = garment.iter().map(|p| a(*p)).collect();
    let weld = Weld::new(&pos, 1e-7);
    let adjacency = crate::geom::adjacency(triangles, &weld);
    let mut push: Vec<V> = (0..weld.count)
        .map(|w| need[weld.representative[w]])
        .collect();
    for (i, n) in need.iter().enumerate() {
        if n.norm() > push[weld.id[i]].norm() {
            push[weld.id[i]] = *n;
        }
    }
    for _ in 0..2 {
        let next: Vec<V> = (0..weld.count)
            .map(|w| {
                let nb = &adjacency[w];
                if nb.is_empty() {
                    return push[w];
                }
                let mean =
                    nb.iter().fold(V::zeros(), |s, &j| s + push[j as usize]) / nb.len() as f64;
                if push[w].norm() >= mean.norm() * 0.5 {
                    push[w]
                } else {
                    mean * 0.5
                }
            })
            .collect();
        push = next;
    }
    let out: Vec<V> = weld.id.iter().map(|&w| push[w]).collect();
    for (p, d) in garment.iter_mut().zip(&out) {
        *p += d;
    }
    out
}

 fn tighten_pass(
    garment: &mut [V],
    triangles: &[[u32; 3]],
    body: &[Vec3],
    body_triangles: &[[u32; 3]],
    designed: &[Option<f64>],
    max_gap: f64,
    reach: f64,
) -> Vec<V> {
    let query = build_query(body, body_triangles);
    let need: Vec<V> = garment
        .iter()
        .zip(designed)
        .map(|(p, d0)| {
            let Some(h) = query.closest_point(a(*p), reach, None) else {
                return V::zeros();
            };
            let to = v(h.point) - p;
            let g = to.norm();
            // Loose by design (the male trunks stand 2-5 cm off the
            // prior): only what exceeds the designed gap is pulled, or the
            // drawstring crumples.
            let limit = max_gap.max(d0.unwrap_or(0.0));
            if g > limit {
                to / g * ((g - limit) * 0.9)
            } else {
                V::zeros()
            }
        })
        .collect();
    let pos: Vec<Vec3> = garment.iter().map(|p| a(*p)).collect();
    let weld = Weld::new(&pos, 1e-7);
    let adjacency = crate::geom::adjacency(triangles, &weld);
    let mut pull: Vec<V> = vec![V::zeros(); weld.count];
    for (i, n) in need.iter().enumerate() {
        if n.norm() > pull[weld.id[i]].norm() {
            pull[weld.id[i]] = *n;
        }
    }
    // Smooth, keeping each vertex's own pull as a floor in size.
    for _ in 0..3 {
        pull = (0..weld.count)
            .map(|w| {
                let nb = &adjacency[w];
                let mean = nb.iter().fold(pull[w], |s, &j| s + pull[j as usize]) / (nb.len() + 1) as f64;
                if pull[w].norm() > mean.norm() { pull[w] } else { mean }
            })
            .collect();
    }
    let out: Vec<V> = weld.id.iter().map(|&w| pull[w]).collect();
    for (p, d) in garment.iter_mut().zip(&out) {
        *p += d;
    }
    out
}

/// Fits `assets` to the body: positions, weights, size morphs.
pub fn fit(
    assets: &[Asset],
    hm: &Hm08,
    body: &BodyInput,
    prior_rest: &[Vec3],
    clearance: f64,
) -> Result<(Vec<Piece>, Value)> {
    let mut pieces = vec![];
    let mut rows = vec![];
    for (ai, asset) in assets.iter().enumerate() {
        let m = &asset.mhclo;
        let mut wanted: HashSet<usize> = m.refs.iter().flat_map(|r| r.verts).collect();
        for (p, q, _) in m.scale.iter().flatten() {
            wanted.extend([*p, *q]);
        }
        let at = |positions: &[Vec3]| -> Result<Vec<V>> {
            let table = hm.lookup(positions, &wanted);
            place(m, &|i| table.get(&i).copied())
        };
        let (source, uvs, triangles) = render_mesh(&asset.obj);
        let render = |p: &[V]| -> Vec<V> { source.iter().map(|&s| p[s]).collect() };
        // Diagnostic: how far the garment sits from the unfitted prior.
        let on_prior = render(&at(prior_rest)?);
        let mut positions = render(&at(body.positions)?);
        let reach = body.height * 0.05;
        let prior_query = build_query(prior_rest, body.triangles);
        let prior_gap: Vec<f64> = on_prior
            .iter()
            .filter_map(|p| prior_query.closest_point(a(*p), reach, None))
            .map(|h| h.distance_m)
            .collect();
        let designed = inward_gaps(&on_prior, &triangles, prior_rest, body.triangles, reach);
        let pull = conform_pass(&mut positions, &triangles, &designed, body.positions, body.triangles, reach);
        let pulled: Vec<f64> = pull.iter().map(|d| d.norm()).filter(|d| *d > 1e-4).collect();
        let designed_closest: Vec<Option<f64>> = on_prior.iter().map(|p| prior_query.closest_point(a(*p), reach, None).map(|h| h.distance_m)).collect();
        let pull2 = conform_closest(&mut positions, &triangles, &designed_closest, body.positions, body.triangles, reach);
        let pulled_closest: Vec<f64> = pull2.iter().map(|d| d.norm()).filter(|d| *d > 1e-4).collect();
        let push = clearance_pass(
            &mut positions,
            &triangles,
            body.positions,
            body.triangles,
            clearance,
            reach,
        );
        let pushed: Vec<f64> = push.iter().map(|d| d.norm()).filter(|d| *d > 0.0).collect();
        let max_gap = std::env::var("MH_UW_MAX_GAP").ok().and_then(|v| v.parse().ok()).unwrap_or(clearance + 0.002);
        let tight = tighten_pass(&mut positions, &triangles, body.positions, body.triangles, &designed_closest, max_gap, reach);
        let tightened: Vec<f64> = tight.iter().map(|d| d.norm()).filter(|d| *d > 1e-4).collect();
        let refs: Vec<&Ref> = source.iter().map(|&s| &m.refs[s]).collect();
        let influences = garment_weights(&positions, &refs, hm, body, reach);
        ensure!(
            influences.iter().all(|r| !r.is_empty()),
            "{}: unweighted underwear vertex",
            asset.id
        );
        // Size morphs: the binding evaluated on the morphed body, the
        // clearance push kept.
        let base = render(&at(body.positions)?);
        let mut targets = vec![];
        for (name, delta) in body.size_targets {
            let moved: Vec<Vec3> = body
                .positions
                .iter()
                .zip(delta)
                .map(|(p, d)| [p[0] + d[0], p[1] + d[1], p[2] + d[2]])
                .collect();
            let d: Vec<Vec3> = render(&at(&moved)?)
                .iter()
                .zip(&base)
                .map(|(x, y)| a(x - y))
                .collect();
            if d.iter().any(|x| v(*x).norm() > 3e-4) {
                targets.push((name.clone(), d));
            }
        }
        let positions: Vec<Vec3> = positions.iter().map(|p| a(*p)).collect();
        let normals = vertex_normals(&positions, &triangles, &Weld::new(&positions, 1e-7));
        rows.push(json!({
            "asset": asset.id,
            "name": m.name,
            "author": m.author,
            "license": m.license,
            "source": asset.source_page(),
            "sha256": asset.sha256.iter().map(|(n, h)| (n.clone(), json!(h))).collect::<serde_json::Map<_, _>>(),
            "vertices": positions.len(),
            "triangles": triangles.len(),
            "helper_references": wanted.iter().filter(|&&h| h >= HM08_BODY).count(),
            "body_vertices_masked_by_asset_not_cut": m.delete_verts,
            "gap_to_unfitted_prior_m": crate::geom::percentiles(prior_gap),
            "clearance_m": clearance,
            "conformed_vertices": pulled.len(),
            "conform_pull_m": crate::geom::percentiles(pulled),
            "conform_closest_pull_m": crate::geom::percentiles(pulled_closest),
            "pushed_vertices": pushed.len(),
            "push_m": crate::geom::percentiles(pushed),
            "max_gap_m": max_gap,
            "tighten_pull_m": crate::geom::percentiles(tightened),
            "size_targets": targets.iter().map(|t| t.0.clone()).collect::<Vec<_>>(),
        }));
        pieces.push(Piece {
            name: format!("underwear_{}", asset.id),
            positions,
            normals,
            uvs,
            triangles,
            influences,
            targets,
            asset: ai,
        });
    }
    Ok((pieces, json!(rows)))
}

/// `ATTRIBUTION.md` for the assets (None when all are CC0).
pub fn attribution(assets: &[Asset]) -> Option<String> {
    if assets.iter().all(|x| x.mhclo.license == "CC0") {
        return None;
    }
    let mut s = String::from(
        "# Underwear assets\n\nThe naked body (`body.glb`) wears these MakeHuman community assets.\n\
         Changes: fitted to this body (MHCLO binding evaluated on the fitted MakeHuman body),\n\
         outward skin clearance, weights from the body, opaque material, body-size morph targets.\n\n",
    );
    for x in assets {
        let url = LICENCES
            .iter()
            .find(|l| l.0 == x.mhclo.license)
            .map_or("", |l| l.1);
        s += &format!(
            "- `{}` (\"{}\") by {}. {} ({url}). Source: {}.\n",
            x.id,
            x.mhclo.name,
            x.mhclo.author,
            x.mhclo.license,
            x.source_page()
        );
        for (n, h) in &x.sha256 {
            s += &format!("  - `{n}` sha256 `{h}`\n");
        }
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MHCLO: &str = "# Exported from MakeClothes (TM)\n# author someone\n# license CC BY 3.0\n\
        uuid x\nbasemesh hm08\nname test-briefs\nobj_file t.obj\nx_scale 0 1 2.0\ny_scale 0 2 4.0\n\
        z_scale 0 3 1.0\nz_depth 50\nmaterial t.mhmat\nverts 0\n 0 1 2 0.5 0.25 0.25 1.0 0.0 -2.0\n\
        3\ndelete_verts\n1 2\n";

    #[test]
    fn parses_header_scales_and_references() {
        let m = parse_mhclo(MHCLO).unwrap();
        assert_eq!(m.author, "someone");
        assert_eq!(m.license, "CC BY 3.0");
        assert_eq!(m.name, "test-briefs");
        assert_eq!(m.obj_file, "t.obj");
        assert_eq!(m.material.as_deref(), Some("t.mhmat"));
        assert_eq!(
            m.scale,
            [Some((0, 1, 2.0)), Some((0, 2, 4.0)), Some((0, 3, 1.0))]
        );
        assert_eq!(m.refs.len(), 2);
        assert_eq!(m.refs[0].verts, [0, 1, 2]);
        assert_eq!(m.refs[0].offset, [1.0, 0.0, -2.0]);
        assert_eq!(m.refs[1].verts, [3, 3, 3]);
        assert_eq!(m.refs[1].weights, [1.0, 0.0, 0.0]);
        assert_eq!(m.delete_verts, 2);
    }

    #[test]
    fn places_on_the_body_with_reference_scales() {
        let m = parse_mhclo(MHCLO).unwrap();
         
        let body = [
            V::new(0.0, 0.0, 0.0),
            V::new(0.4, 0.0, 0.0),
            V::new(0.0, 0.8, 0.0),
            V::new(0.0, 0.0, 0.3),
        ];
        let p = place(&m, &|i| body.get(i).copied()).unwrap();
        let want0 =
            body[0] * 0.5 + body[1] * 0.25 + body[2] * 0.25 + V::new(1.0 * 0.2, 0.0, -2.0 * 0.3);
        assert!((p[0] - want0).norm() < 1e-12, "{}", p[0]);
        assert!((p[1] - body[3]).norm() < 1e-12);
        assert!(place(&m, &|i| body.get(i).filter(|_| i != 2).copied()).is_err());
    }

    /// Two quads of a base mesh; the "glTF" body splits vertex 1 along a UV
    /// seam and is ordered differently.
    #[test]
    fn maps_gltf_vertices_to_hm08_by_uv() {
        let obj = parse_obj(
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 0\nv 2 1 0\n\
             vt 0.1 0.1\nvt 0.2 0.1\nvt 0.2 0.2\nvt 0.1 0.2\nvt 0.6 0.1\nvt 0.7 0.1\nvt 0.7 0.2\nvt 0.6 0.2\n\
             g body\nf 1/1 2/2 3/3 4/4\nf 2/5 5/6 6/7 3/8\n",
        )
        .unwrap();
        let (source, uvs, tris) = render_mesh(&obj);
        assert_eq!(source.len(), 8);
        assert_eq!(tris.len(), 4);
        // Reverse the glTF vertex order.
        let n = uvs.len();
        let rev_uvs: Vec<[f32; 2]> = uvs.iter().rev().copied().collect();
        let rev_tris: Vec<[u32; 3]> = tris
            .iter()
            .map(|t| t.map(|i| (n - 1 - i as usize) as u32))
            .collect();
        let hm = Hm08::new(&obj, &rev_uvs, &rev_tris).unwrap();
        for (g, &s) in source.iter().enumerate() {
            let p = hm.to_prior[s].unwrap();
            assert_eq!(source[n - 1 - p], s, "glTF {g}");
        }
         
        let mut wrong = rev_tris.clone();
        wrong[0] = [
            (n - 1) as u32, // obj vertex 0
            (n - 1 - 5) as u32,
            (n - 1 - 6) as u32,
        ];
        let bad = Hm08::new(&obj, &rev_uvs, &wrong);
        assert!(bad.is_err());
    }

    #[test]
    fn lift_keeps_layer_order_and_clearance() {
        let c = 0.004;
        let ds = [-0.01, -0.002, 0.0, 0.002, 0.004, 0.006, 0.0079, 0.008, 0.02];
        let lifted: Vec<f64> = ds.iter().map(|&d| lift(d, c)).collect();
        assert!(lifted.iter().all(|&x| x >= c));
        assert!(lifted.windows(2).all(|w| w[0] < w[1]), "{lifted:?}");
        assert_eq!(lift(0.008, c), 0.008);
        assert_eq!(lift(0.02, c), 0.02);
        assert!((lift(0.0079, c) - 0.0079).abs() < 1e-5);
    }

    #[test]
    fn choice_by_sex_and_explicit_lists() {
        assert_eq!(choose(&Underwear::default(), "female"), FEMALE);
        assert_eq!(choose(&Underwear::default(), "unknown"), MALE);
        assert!(choose(&Underwear::Mode(UnderwearMode::None), "female").is_empty());
        let c: Underwear = serde_json::from_value(json!(["wolgade_female_top_01"])).unwrap();
        assert_eq!(choose(&c, "male"), vec!["wolgade_female_top_01"]);
        let c: Underwear = serde_json::from_value(json!("none")).unwrap();
        assert_eq!(c, Underwear::Mode(UnderwearMode::None));
    }

         #[test]
    fn real_assets_sit_on_the_base_mesh() {
        let here = Path::new(env!("CARGO_MANIFEST_DIR"));
        let main = here.ancestors().find(|d| d.join("out/tools/mpfb").is_dir());
        let Some(root) = find_assets(None, main.unwrap_or(here)) else {
            eprintln!("underwear assets not downloaded; skipped");
            return;
        };
        let Some(base_path) = find_base_mesh(&root) else {
            eprintln!("base.obj not found; skipped");
            return;
        };
        let base = parse_obj(&std::fs::read_to_string(base_path).unwrap()).unwrap();
        let points: Vec<Vec3> = base.positions.iter().map(|p| a(p * MH_TO_M)).collect();
        let triangles: Vec<[u32; 3]> = base
            .faces
            .iter()
            .filter(|f| f.0 == "body")
            .flat_map(|(_, f)| {
                (1..f.len() - 1).map(|k| [f[0].0 as u32, f[k].0 as u32, f[k + 1].0 as u32])
            })
            .collect();
        let query = build_query(&points, &triangles);
        for id in FEMALE.iter().chain(MALE) {
            let asset = Asset::load(&root, id).unwrap();
            let p = place(&asset.mhclo, &|i| points.get(i).map(|p| v(*p))).unwrap();
            let mut d: Vec<f64> = p
                .iter()
                .map(|x| query.closest_point(a(*x), 1.0, None).unwrap().distance_m)
                .collect();
            d.sort_by(f64::total_cmp);
            let (median, p95) = (d[d.len() / 2], d[d.len() * 95 / 100]);
            assert!(
                median < 0.012 && p95 < 0.03,
                "{id}: median {median} p95 {p95} m"
            );
        }
    }
}
