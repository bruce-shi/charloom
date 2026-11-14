//! Read-only, matched-topology head QA for saved local transfer runs.
//! Uses the production visibility and outward hair crossing predicates.
use crate::{
    fit::hidden_head,
    geom::{V, Weld, a, percentiles, proper_rotation, v},
    prior::{Prior, Region},
    scan::{RawScan, build_query},
    source::{gltf_of, read_primitives},
};
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::Path};

fn read_json(path: &Path) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn mesh(path: &Path) -> Result<(Vec<[f64; 3]>, Vec<[u32; 3]>)> {
    let loaded = charloom_io::load_gltf(path)?;
    let positions = loaded.asset.world_meshes()?;
    let prims = read_primitives(&loaded, &positions)?;
    let p = prims.first().context("missing mesh")?;
    Ok((p.positions.clone(), p.triangles.clone()))
}

fn stage(path: &Path) -> Result<Vec<V>> {
    let bytes = std::fs::read(path)?;
    ensure!(bytes.len() % 12 == 0, "invalid stage dump");
    Ok(bytes.chunks_exact(12).map(|b| V::from_fn(|i, _|
        f32::from_le_bytes(b[i*4..i*4+4].try_into().unwrap()) as f64)).collect())
}

/// Reuse one historical anatomical/visibility mask for all compared tags.
/// Shrinking a result therefore cannot make a failing vertex leave the audit.
pub fn audit(run: &Path, mask_run: &Path, mask_stages: &Path, stages: &Path) -> Result<Value> {
    let report = read_json(&run.join("report.json"))?;
    let prior = Prior::load(Path::new(report["prior"].as_str().context("prior")?))?;
    let (positions, triangles) = mesh(&run.join("2-body-rig/fit-diagnostic.glb"))?;
    let (canonical, canonical_tri) = mesh(&mask_run.join("2-body-rig/fit-diagnostic.glb"))?;
    ensure!(triangles == canonical_tri && positions.len() == canonical.len(), "matched topology required");
    let body = prior.body();
    ensure!(body.prim.triangles == triangles, "prior topology differs");
    let weld = Weld::new(&body.bind, 1e-7);
    let canonical_query = build_query(&canonical, &triangles);
    let body_query = build_query(&positions, &triangles);
    let segment_path = mask_run.join("1-segment/segmentation.glb");
    let scan = RawScan::load(&segment_path)?.into_scan(nalgebra::Matrix4::identity());
    let loaded = charloom_io::load_gltf(&segment_path)?;
    let gltf = gltf_of(&loaded)?;
    let mut labels = vec![];
    for m in &loaded.asset.meshes {
        let p = gltf.document.nodes().nth(m.node).and_then(|n| n.mesh())
            .and_then(|m0| m0.primitives().nth(m.primitive)).context("segmentation primitive")?;
        let reader = p.reader(|_| Some(loaded.document.binary.as_slice()));
        let colors: Vec<_> = reader.read_colors(0).context("segmentation colors")?.into_rgba_f32().collect();
        labels.extend(colors.iter().map(|c| {
            let near = |r: f32, g: f32, b: f32| (c[0]-r).abs()+(c[1]-g).abs()+(c[2]-b).abs()<0.01;
            if near(0.95,0.55,0.35) { 1u8 }
            else if near(0.45,0.2,0.6) || near(0.95,0.35,0.75) { 2 } else { 0 }
        }));
    }
    ensure!(labels.len() == scan.positions.len(), "segmentation indexing");
    let head: Vec<_> = (0..positions.len()).map(|i|
        prior.region[prior.dominant_joint(prior.body,i)] == Some(Region::Head)).collect();
    // Neck influence is split across three bones. A transition vertex can
    // have `head` as its single largest weight while still being neck skin.
    // Use the anatomical support, as graft::uncovered does, to exclude neck
    // from the skull-only outline gate.
    let neck: Vec<_> = body.influences.iter().map(|row|row.iter().filter(|w|
        prior.skeleton().joints[w.joint].name.to_lowercase().starts_with("neck"))
        .map(|w|w.weight).sum::<f64>()>1e-6).collect();
    let scalp: Vec<_> = (0..positions.len()).map(|i|
        prior.skeleton().joints[prior.dominant_joint(prior.body,i)].name == "head").collect();
    let ids: Vec<_> = weld.representative.iter().copied().filter(|&i| head[i]).collect();
    let centre = ids.iter().fold(V::zeros(), |s,&i|s+v(canonical[i])) / ids.len().max(1) as f64;
    let radius = ids.iter().map(|&i|(v(canonical[i])-centre).norm()).fold(0.0,f64::max)*1.6;
    let seen = hidden_head::seen_from_outside(&scan, centre, radius);
    let visible_skin = |t: usize| scan.triangles[t].iter().filter(|&&i| labels[i as usize]==1).count()>=2
        && scan.triangles[t].iter().any(|&i|seen[i as usize]);
    let flags = stage(&mask_stages.join("hidden-head-flags.f32"))?;
    let fixed = stage(&mask_stages.join("hidden-head-fixed-reasons.f32"))?;
    let height = report["stage4_garments"]["grafts"]["gap_m"].as_f64().context("height")? / 0.002;
    let visible: Vec<_> = (0..positions.len()).map(|i| (head[i]||neck[i]) && flags[i].y<0.5
        && scan.query.closest_point(canonical[i],0.012*height,None).is_some_and(|h|visible_skin(h.source[1]))).collect();
    let hidden: Vec<_> = (0..positions.len()).map(|i| head[i]&&scalp[i]&&!neck[i]&&!visible[i]
        &&flags[i].x<0.5&&flags[i].y<0.5).collect();
    let hidden_ids: Vec<_> = weld.representative.iter().copied().filter(|&i|hidden[i]).collect();
    let hair_tri: Vec<_> = scan.triangles.iter().filter(|t|
        t.iter().filter(|&&i|labels[i as usize]==2).count()>=2).copied().collect();
    let hair = build_query(&scan.positions,&hair_tri);
    let mut crossings=0; let mut poke=vec![]; let mut shortfall=vec![];
    for &i in &hidden_ids {
        let delta=v(positions[i])-centre;
        if let Some(direction)=delta.try_normalize(1e-12)
            && let Some(distance)=hair.outward_ray_distance(a(centre),a(direction),height*0.5) {
            crossings+=1;
            let excess=delta.norm()-distance;
            if excess>1e-6 {poke.push((i,excess*1000.0));}
            if excess+0.0018>1e-6 {shortfall.push((i,(excess+0.0018)*1000.0));}
        }
    }
    let skin_tri: Vec<_> = scan.triangles.iter().enumerate().filter(|(t,_)|visible_skin(*t)).map(|(_,t)|*t).collect();
    let skin = build_query(&scan.positions,&skin_tri);
    let mut distance: BTreeMap<&str,Vec<f64>> = BTreeMap::new();
    for &i in &weld.representative {
        if !visible[i] {continue;}
        let key=if neck[i]{"neck"}else if fixed[i].x>0.5{"ears"}else{"face"};
        if let Some(h)=skin.closest_point(positions[i],height*0.2,None) {
            distance.entry(key).or_default().push(h.distance_m*1000.0);
            distance.entry("all_visible").or_default().push(h.distance_m*1000.0);
        }
    }
    let mut reverse: BTreeMap<&str,Vec<f64>> = BTreeMap::new();
    for &i in &scan.weld.representative {
        if labels[i]!=1 || !seen[i] {continue;}
        let Some(h)=canonical_query.closest_point(scan.positions[i],height*0.1,None)else{continue};
        let t=triangles[h.source[1]];
        let k=(0..3).max_by(|&i,&j|h.barycentric[i].total_cmp(&h.barycentric[j])).unwrap();
        let j=t[k] as usize;
        if !(head[j]||neck[j]) || flags[j].y>0.5 {continue;}
        let key=if neck[j]{"neck"}else if fixed[j].x>0.5{"ears"}else{"face"};
        if let Some(h)=body_query.closest_point(scan.positions[i],height*0.2,None) {
            reverse.entry(key).or_default().push(h.distance_m*1000.0);
            reverse.entry("all_visible").or_default().push(h.distance_m*1000.0);
        }
    }
    let stats=|rows: BTreeMap<&str,Vec<f64>>| -> Value {
        Value::Object(rows.into_iter().map(|(k,v)|(k.to_string(),percentiles(v))).collect())
    };
    let mut stage_rows=vec![];
    // Identical reference/rotations for all tags, independently of report metrics.
    let reference=stage(&mask_stages.join("1-shape.f32"))?;
    let mut neighbours=vec![vec![];weld.count];
    for t in &triangles {
        for k in 0..3 {let i=weld.id[t[k]as usize];let j=weld.id[t[(k+1)%3]as usize];
            neighbours[i].push(j);neighbours[j].push(i);}
    }
    for row in &mut neighbours {row.sort_unstable();row.dedup();}
    let mut previous=vec![false;triangles.len()];
    for label in ["0-pose","1-shape","2-outlines","head-held-start","3-fine","sculpt-before","hidden-head-before","hidden-head-after","fit-output","post-eye-clear","post-mouth","post-rig","export"] {
        let file=stages.join(format!("{label}.f32"));
        let x=if label=="export"{positions.iter().map(|p|v(*p)).collect()}else if file.exists(){stage(&file)?}else{continue};
        let frames: Vec<_>=weld.representative.iter().enumerate().map(|(w,&i)| {
            let mut m=nalgebra::Matrix3::zeros();
            for &j in &neighbours[w] {let j=weld.representative[j];m+=(x[i]-x[j])*(reference[i]-reference[j]).transpose();}
            proper_rotation(m)
        }).collect();
        let mut by_region:BTreeMap<String,usize>=BTreeMap::new();let mut flipped=vec![false;triangles.len()];
        let mut added:BTreeMap<String,usize>=BTreeMap::new();let mut removed=0;
        for (k,t) in triangles.iter().enumerate() {
            let [i,j,l]=t.map(|i|i as usize);
            let normal=(x[j]-x[i]).cross(&(x[l]-x[i]));
            let r=(reference[j]-reference[i]).cross(&(reference[l]-reference[i]));
            let carried=(frames[weld.id[i]]+frames[weld.id[j]]+frames[weld.id[l]])*r/3.0;
            flipped[k]=normal.norm()>1e-12&&carried.norm()>1e-12&&normal.dot(&carried)<0.0;
            let joint=&prior.skeleton().joints[prior.dominant_joint(prior.body,i)].name;
            let key=if head[i] {if flags[i].y>0.5{"Head cavity".to_string()}else if hidden[i]{"Hidden scalp".to_string()}else{format!("Head {joint}")}}
                else if neck[i]{"Neck".to_string()}else{format!("{:?}",prior.region[prior.dominant_joint(prior.body,i)])};
            if flipped[k] {*by_region.entry(key.clone()).or_default()+=1;}
            if flipped[k]&&!previous[k] {*added.entry(key).or_default()+=1;}
            if !flipped[k]&&previous[k] {removed+=1;}
        }
        stage_rows.push(json!({"stage":label,"inverted":by_region.values().sum::<usize>(),"by_region":by_region,"new_since_previous":added,"resolved_since_previous":removed}));
        previous=flipped;
    }
    Ok(json!({"mask_run":mask_run,"mask_note":"Fixed baseline topology, production seen_from_outside; Head scalp only, excluding neck, ears, eyes, cavities and landmarks",
        "hidden_scalp_ids":hidden_ids,"hidden_scalp_mask":hidden,
        "visible_skin_mask":visible,"hair":{"hidden_scalp_vertices":hidden_ids.len(),"outward_crossings":crossings,
            "poke_vertices":poke.len(),"poke_pct":100.0*poke.len()as f64/hidden_ids.len().max(1)as f64,"poke_details_mm":poke,
            "clearance_mm":1.8,"clearance_shortfall_vertices":shortfall.len(),"clearance_shortfall_details_mm":shortfall},
        "scan_distance_mm":{"body_to_visible_skin":stats(distance),"visible_skin_to_body":stats(reverse)},
        "stages":{"note":"Recomputed local-rotation metric against the SAME x114 1-shape reference for each tag; diagnostic only. Acceptance uses unchanged production inverted_triangles.","rows":stage_rows}}))
}
