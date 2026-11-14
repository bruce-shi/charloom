//! Whole-head, skin-anchored registration of the existing GPT head channel.
//! All work stays in the calibrated cameras; held-out images never enter here.
use super::silhouette;
use crate::{
    geom::{V, a, v},
    guide::SuitContour,
    scan::Scan,
    segment::{Label, Segmentation},
};
use serde_json::{Value, json};

/// x,y -> (a*x-b*y+tx, b*x+a*y+ty), in the original camera's pixels.
type Similarity = [f64; 4];
fn point(p: [f64; 2], t: Similarity) -> [f64; 2] {
    [
        t[0] * p[0] - t[1] * p[1] + t[2],
        t[1] * p[0] + t[0] * p[1] + t[3],
    ]
}
fn inverse(p: [f64; 2], t: Similarity) -> [f64; 2] {
    let den = t[0] * t[0] + t[1] * t[1];
    let (x, y) = (p[0] - t[2], p[1] - t[3]);
    [(t[0] * x + t[1] * y) / den, (-t[1] * x + t[0] * y) / den]
}
fn transformed(c: &SuitContour, t: Similarity) -> SuitContour {
    SuitContour {
        camera: c.camera.clone(),
        curves: c
            .curves
            .iter()
            .map(|q| q.iter().map(|&p| point(p, t)).collect())
            .collect(),
        weights: c.weights.clone(),
        rotation_limits_deg: c.rotation_limits_deg,
        scale_tied_to: c.scale_tied_to.clone(),
        skin_outline_anchors: c.skin_outline_anchors.iter().map(|&(p,q)| (point(p,t),q)).collect(),
        skin_anchors: c
            .skin_anchors
            .iter()
            .map(|&(p, q)| (point(p, t), q))
            .collect(),
    }
}
fn fit_similarity(pairs: &[([f64; 2], [f64; 2], f64)], limits: [f64; 2], fixed_scale: Option<f64>) -> Similarity {
    let w = pairs.iter().map(|p| p.2).sum::<f64>().max(1e-20);
    let mut cp = [0.; 2];
    let mut cq = [0.; 2];
    for &(p, q, wi) in pairs {
        for k in 0..2 {
            cp[k] += p[k] * wi / w;
            cq[k] += q[k] * wi / w;
        }
    }
    let (mut dot, mut cross, mut den) = (0., 0., 0.);
    for &(p, q, wi) in pairs {
        let p = [p[0] - cp[0], p[1] - cp[1]];
        let q = [q[0] - cq[0], q[1] - cq[1]];
        dot += wi * (p[0] * q[0] + p[1] * q[1]);
        cross += wi * (p[0] * q[1] - p[1] * q[0]);
        den += wi * (p[0] * p[0] + p[1] * p[1]);
    }
    if den < 1e-12 {
        return [1., 0., 0., 0.];
    }
    let scale = fixed_scale.unwrap_or_else(|| (dot.hypot(cross) / den).clamp(0.8, 1.25));
    let angle = cross
        .atan2(dot)
        .clamp(limits[0].to_radians(), limits[1].to_radians());
    let (aa, bb) = (scale * angle.cos(), scale * angle.sin());
    [
        aa,
        bb,
        cq[0] - aa * cp[0] + bb * cp[1],
        cq[1] - bb * cp[0] - aa * cp[1],
    ]
}
fn percentile(mut x: Vec<f64>, f: f64) -> f64 {
    x.sort_by(f64::total_cmp);
    x.get(((x.len().saturating_sub(1)) as f64 * f) as usize)
        .copied()
        .unwrap_or(0.)
}
fn cardinal(c: &SuitContour) -> f64 {
    if c.camera.forward[0].abs() > 0.92 {
        1.0
    } else if c.camera.forward[2].abs() > 0.92 {
        0.8
    } else {
        0.5
    }
}

pub(super) struct Registration {
    original: Vec<SuitContour>,
    pub contours: Vec<SuitContour>,
    transforms: Vec<Similarity>,
    trust: Vec<f64>,
    pub rounds: Vec<Value>,
    pub converged: bool,
    skin_support: Vec<Option<Vec<([f64; 2], [f64; 2])>>>,
}
impl Registration {
    pub fn new(contours: &[SuitContour]) -> Self {
        Self {
            original: contours.to_vec(),
            contours: contours.to_vec(),
            transforms: vec![[1., 0., 0., 0.]; contours.len()],
            trust: contours.iter().map(cardinal).collect(),
            rounds: vec![],
            converged: false,
            skin_support: vec![None; contours.len()],
        }
    }
    pub fn update(
        &mut self,
        x: &[V],
        triangles: &[[u32; 3]],
        region: &[bool],
        scan: &Scan,
        seg: &Segmentation,
    ) {
        if self.converged || self.rounds.len() >= 3 {
            return;
        }
        let sx: Vec<_> = scan.positions.iter().copied().map(v).collect();
        let skin: Vec<_> = seg.vertex_label.iter().map(|l| *l == Label::Skin).collect();
        let mut rows = vec![];
        let mut max_scale_change: f64 = 0.;
        for k in 0..self.contours.len() {
            let initial = self.transforms[k];
            let mut t = initial;
            // Rasterize the complete scan: hair still occludes the visible skin anchors.
            let view = silhouette::View::head(&self.original[k], x, region);
            let (anchors, _, _) =
                silhouette::rendered_constraints(&[view], &sx, &scan.triangles, &skin, 0.012);
            let camera = &self.original[k].camera;
            let uv = |p: V| {
                let (u, v, _) = camera.project(a(p));
                [u, v]
            };
            let mut sampled_skin_pairs: Vec<_> = anchors
                .iter()
                .filter(|c| {
                    c.vertices
                        .iter()
                        .zip(c.bary)
                        .filter(|(i, _)| skin[**i as usize])
                        .map(|(_, b)| b)
                        .sum::<f64>()
                        > 0.99
                })
                .map(|c| {
                    let q = c
                        .vertices
                        .iter()
                        .zip(c.bary)
                        .fold(V::zeros(), |s, (&i, b)| s + sx[i as usize] * b);
                    (uv(q + c.delta), uv(q))
                })
                .collect();
            sampled_skin_pairs.extend(self.original[k].skin_outline_anchors.iter().copied());
            // Freeze scan-outline support at first registration. A retained
            // transform must not silently violate slack after resampling.
            let skin_pairs = self.skin_support[k].get_or_insert(sampled_skin_pairs);
            let skin_p95 = |t: Similarity| {
                percentile(
                    skin_pairs
                        .iter()
                        .map(|&(p, q)| {
                            let p = point(p, t);
                            (p[0] - q[0]).hypot(p[1] - q[1]) * camera.meters_per_pixel * 1000.
                        })
                        .collect(),
                    0.95,
                )
            };
            let before = skin_p95([1., 0., 0., 0.]);
            let textures = &self.original[k].skin_anchors;
            let texture_p95 = |t: Similarity| {
                percentile(
                    textures
                        .iter()
                        .map(|&(p, q)| {
                            let p = point(p, t);
                            (p[0] - q[0]).hypot(p[1] - q[1]) * camera.meters_per_pixel * 1000.
                        })
                        .collect(),
                    0.95,
                )
            };
            let texture_before = texture_p95([1., 0., 0., 0.]);
            let mut last_rms = 0.;
            let mut used = 0;
            for _ in 0..8 {
                let c = transformed(&self.original[k], t);
                let view = silhouette::View::head(&c, x, region);
                let (whole, _, _) =
                    silhouette::rendered_constraints(&[view], x, triangles, region, 0.060);
                if whole.is_empty() {
                    break;
                }
                used = whole.len();
                last_rms = (whole.iter().map(|c| c.delta.norm_squared()).sum::<f64>()
                    / used as f64)
                    .sqrt()
                    * 1000.;
                let mut pairs: Vec<_> = whole
                    .iter()
                    .map(|c| {
                        let q = c
                            .vertices
                            .iter()
                            .zip(c.bary)
                            .fold(V::zeros(), |s, (&i, b)| s + x[i as usize] * b);
                        (inverse(uv(q + c.delta), t), uv(q), 1. / used as f64)
                    })
                    .collect();
                // Each group is normalized by sample count. A large skull outline
                // cannot drown the face, ear or neck outlines in raster density.
                pairs.extend(
                    skin_pairs
                        .iter()
                        .map(|&(p, q)| (p, q, 20. / skin_pairs.len().max(1) as f64)),
                );
                pairs.extend(
                    textures
                        .iter()
                        .map(|&(p, q)| (p, q, 20. / textures.len().max(1) as f64)),
                );
                let base: Vec<_> = self.original[k].curves.iter().flatten().copied().collect();
                pairs.extend(
                    base.iter()
                        .step_by(4)
                        .map(|&p| (p, p, 0.02 / (base.len() / 4).max(1) as f64)),
                );
                let locked_scale = self.original[k].scale_tied_to.as_ref().map(|_| 1.0);
                let proposal = fit_similarity(&pairs, self.original[k].rotation_limits_deg, locked_scale);
                let mut accepted = t;
                // Feasible backtracking: retain the already registered visible-skin
                // outline to 0.3 mm, even if a whole-head match wants to trade it away.
                for alpha in [0.5, 0.25, 0.125, 0.0625, 0.03125] {
                    let mut trial: Similarity = std::array::from_fn(|j| t[j] + alpha * (proposal[j] - t[j]));
                    if let Some(scale) = locked_scale {
                        // Linear interpolation of rotations shrinks a matrix;
                        // normalize it to preserve the same-sheet scale exactly.
                        let norm = trial[0].hypot(trial[1]);
                        trial[0] *= scale / norm; trial[1] *= scale / norm;
                    }
                    if (skin_pairs.is_empty() || skin_p95(trial) <= before + 0.3)
                        && (textures.is_empty() || texture_p95(trial) <= texture_before + 0.3)
                    {
                        accepted = trial;
                        break;
                    }
                }
                let change = (accepted[0] - t[0]).hypot(accepted[1] - t[1]);
                t = accepted;
                if change < 0.0001 {
                    break;
                }
            }
            let scale = t[0].hypot(t[1]);
            let old_scale = initial[0].hypot(initial[1]);
            max_scale_change = max_scale_change.max((scale / old_scale - 1.).abs());
            self.transforms[k] = t;
            self.contours[k] = transformed(&self.original[k], t);
            let residual = last_rms.max(skin_p95(t)).max(texture_p95(t));
            self.trust[k] = cardinal(&self.original[k]) / (1. + (residual / 3.).powi(2));
            rows.push(json!({"view":camera.id,"scale":scale,"scale_tied_to":self.original[k].scale_tied_to,"skin_support":"frozen_first_round","precalibrated_skin_outline_points":self.original[k].skin_outline_anchors.len(),"rotation_deg":t[1].atan2(t[0]).to_degrees(),
                "matrix_original_px":[[t[0],-t[1],t[2]],[t[1],t[0],t[3]]],
                "incremental_rotation_limits_deg":self.original[k].rotation_limits_deg,"skin_outline_points":skin_pairs.len(),"visible_skin_texture_points":textures.len(),"texture_p95_before_mm":texture_before,"texture_p95_after_mm":texture_p95(t),"skin_p95_before_mm":before,"skin_p95_after_mm":skin_p95(t),
                "whole_head_rms_mm":last_rms,"whole_head_samples":used,"cardinal_weight":cardinal(&self.original[k]),"trust_weight":self.trust[k]}));
        }
        self.converged = max_scale_change < 0.003;
        self.rounds.push(json!({"round":self.rounds.len()+1,"max_relative_scale_change":max_scale_change,"stop_below":0.003,"views":rows}));
    }
    pub fn weighted_views(
        &self,
        x: &[V],
        triangles: &[[u32; 3]],
        region: &[bool],
        centre: V,
        hair: &charloom_engine::surface_query::SurfaceQuery,
        reach: f64,
    ) -> (Vec<silhouette::View>, Value) {
        let mut views = vec![];
        let mut records = vec![];
        for (k, c) in self.contours.iter().enumerate() {
            let mut view = silhouette::View::head(c, x, region);
            let rejected = view.hair_arbiter(
                x,
                triangles,
                region,
                centre,
                hair,
                reach,
                0.0018,
                self.trust[k],
            );
            records.push(json!({"view":c.camera.id,"zero_weight_outside_hair":rejected,"trust_weight":self.trust[k],"effective_evidence":view.evidence()}));
            views.push(view);
        }
        (views, json!(records))
    }
    pub fn relative_strength(&self) -> f64 {
        self.trust.iter().sum::<f64>() / self.original.iter().map(cardinal).sum::<f64>().max(1e-12)
    }
    pub fn report(&self) -> Value {
        let views:Vec<_>=self.contours.iter().enumerate().map(|(k,c)|json!({"view":c.camera.id,"camera":c.camera,"curves":c.curves,"weights":c.weights,"trust_weight":self.trust[k],"transform":self.transforms[k]})).collect();
        let crowns: Vec<_> = self
            .contours
            .iter()
            .filter_map(|c| {
                c.curves
                    .iter()
                    .flatten()
                    .map(|p| {
                        c.camera.center[1]
                            + (c.camera.height as f64 / 2. - p[1]) * c.camera.meters_per_pixel
                    })
                    .max_by(f64::total_cmp)
            })
            .collect();
        let spread = if crowns.len() > 1 {
            (crowns.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - crowns.iter().copied().fold(f64::INFINITY, f64::min))
                * 1000.
        } else {
            0.
        };
        json!({"method":"rendered_whole_head_similarity_strong_visible_scan_skin","rounds":self.rounds,"converged":self.converged,"maximum_rounds":3,
            "rotation_limit_deg":5,"skin_p95_slack_mm":0.3,"views":views,"cross_view_spread_mm":spread,"relative_gate_mm":3f64.max(spread/2.),
            "spread_definition":"range of registered training crown world heights before hair rejection; same invariant as x116 source_consistency","held_out_member":"e3 (never loaded by fit)"})
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn faceless_back_keeps_same_sheet_profile_scale() {
        let source = [[-40., -60.], [40., -60.], [-25., 60.], [25., 60.]];
        let enlarged = [1.12 * 0.03f64.cos(), 1.12 * 0.03f64.sin(), 4., -2.];
        let pairs: Vec<_> = source.iter().map(|&p| (p, point(p, enlarged), 1.)).collect();
        let t = fit_similarity(&pairs, [-5., 5.], Some(1.));
        assert!((t[0].hypot(t[1]) - 1.).abs() < 1e-12);
        assert!((t[1].atan2(t[0]) - 0.03).abs() < 1e-12);
        assert!((t[2] - 4.).abs() < 1e-12);
    }
    #[test]
    fn similarity_recovers_scale_translation_and_caps_total_rotation() {
        let points = [[0., 0.], [3., 0.], [0., 2.], [3., 2.]];
        let t = [1.03 * 0.04f64.cos(), 1.03 * 0.04f64.sin(), 2., -4.];
        let pairs: Vec<_> = points.iter().map(|&p| (p, point(p, t), 1.)).collect();
        let recovered = fit_similarity(&pairs, [-5., 5.], None);
        for k in 0..4 {
            assert!((recovered[k] - t[k]).abs() < 1e-10);
        }
        let large = [0.3f64.cos(), 0.3f64.sin(), 0., 0.];
        let pairs: Vec<_> = points.iter().map(|&p| (p, point(p, large), 1.)).collect();
        let result = fit_similarity(&pairs, [-5., 5.], None);
        assert!((result[1].atan2(result[0]).to_degrees() - 5.).abs() < 1e-10);
    }
    #[test]
    fn normalized_skin_terms_prevent_dense_skull_from_trading_face_alignment() {
        let skin = [[0., 0.], [0., 2.], [2., 0.], [2., 2.]];
        let mut pairs: Vec<_> = skin.iter().map(|&p| (p, p, 20. / 4.)).collect();
        for i in 0..1000 {
            let p = skin[i % 4];
            pairs.push((p, [p[0] + 1., p[1]], 1. / 1000.));
        }
        let t = fit_similarity(&pairs, [-5., 5.], None);
        assert!(skin.iter().all(|&p| {
            let q = point(p, t);
            (q[0] - p[0]).hypot(q[1] - p[1]) < 0.05
        }));
        let mut sparse: Vec<_> = skin.iter().map(|&p| (p, p, 20. / 4.)).collect();
        sparse.extend(skin.iter().map(|&p| (p, [p[0] + 1., p[1]], 1. / 4.)));
        let other = fit_similarity(&sparse, [-5., 5.], None);
        for k in 0..4 {
            assert!((t[k] - other[k]).abs() < 1e-10);
        }
    }
}
