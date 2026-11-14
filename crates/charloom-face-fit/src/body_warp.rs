//! Portable proper-similarity deformation shared by native reconstruction and browser authoring.
use anyhow::{Result, ensure};
use charloom_core::{Vec3, a3, v3};
use nalgebra::{Matrix3, Vector3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Landmark {
    pub id: String,
    pub prior: Vec3,
    pub target: Vec3,
    #[serde(default = "one")]
    pub strength: f64,
}
fn one() -> f64 {
    1.
}
pub struct Warp {
    controls: Vec<Landmark>,
    floor_squared: f64,
}

impl Warp {
    pub fn fit(controls: &[Landmark], radius: f64, regularization: f64) -> Result<Self> {
        ensure!(
            controls.len() >= 4 && controls.len() <= 512,
            "body_landmark_count"
        );
        // Similarity fits have a proper rotation and positive scale. They do
        // not extrapolate an unconstrained polynomial through the thin medial
        // landmark cloud, which can fold an entire chest/waist inside-out.
        Ok(Self {
            controls: controls.to_vec(),
            floor_squared: radius.powi(2) * (0.015 + regularization),
        })
    }

    pub fn point(&self, point: Vec3) -> Vec3 {
        let point = v3(point);
        let weights: Vec<_> = self
            .controls
            .iter()
            .map(|c| {
                c.strength / ((point - v3(c.prior)).norm_squared() + self.floor_squared).powi(2)
            })
            .collect();
        let sum = weights.iter().sum::<f64>();
        let p = self
            .controls
            .iter()
            .zip(&weights)
            .map(|(c, w)| v3(c.prior) * *w)
            .sum::<Vector3<f64>>()
            / sum;
        let q = self
            .controls
            .iter()
            .zip(&weights)
            .map(|(c, w)| v3(c.target) * *w)
            .sum::<Vector3<f64>>()
            / sum;
        let covariance = self
            .controls
            .iter()
            .zip(&weights)
            .fold(Matrix3::zeros(), |s, (c, w)| {
                s + (v3(c.target) - q) * (v3(c.prior) - p).transpose() * *w
            });
        let rotation = proper_rotation(covariance);
        let denominator = self
            .controls
            .iter()
            .zip(&weights)
            .map(|(c, w)| (v3(c.prior) - p).norm_squared() * *w)
            .sum::<f64>();
        let numerator = self
            .controls
            .iter()
            .zip(&weights)
            .map(|(c, w)| (v3(c.target) - q).dot(&(rotation * (v3(c.prior) - p))) * *w)
            .sum::<f64>();
        let scale = (numerator / denominator.max(1e-20)).clamp(0.5, 2.0);
        a3(q + rotation * (point - p) * scale)
    }

    pub fn jacobian(&self, point: Vec3) -> Matrix3<f64> {
        let h = 0.0001;
        let mut result = Matrix3::zeros();
        for k in 0..3 {
            let mut a = point;
            let mut b = point;
            a[k] += h;
            b[k] -= h;
            result.set_column(k, &((v3(self.point(a)) - v3(self.point(b))) / (2. * h)));
        }
        result
    }
}

pub fn proper_rotation(m: Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let mut correction = Matrix3::identity();
    correction[(2, 2)] = (u * vt).determinant().signum();
    u * correction * vt
}
