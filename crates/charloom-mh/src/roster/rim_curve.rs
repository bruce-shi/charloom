use anyhow::{Result, ensure};
use charloom_core::Vec3;

// Shape-preserving piecewise cubic interpolation (PCHIP), independently on
// height and depth. The end slopes use the one-sided three-point estimate.
fn tangents(x: &[f64], y: &[f64]) -> Vec<f64> {
    let h: Vec<_> = x.windows(2).map(|p| p[1] - p[0]).collect();
    let d: Vec<_> = y
        .windows(2)
        .zip(&h)
        .map(|(p, h)| (p[1] - p[0]) / h)
        .collect();
    if d.len() == 1 {
        return vec![d[0]; 2];
    }
    let end = |i: usize, j: usize| {
        let m = ((2. * h[i] + h[j]) * d[i] - h[i] * d[j]) / (h[i] + h[j]);
        if m.signum() != d[i].signum() || d[i] == 0. {
            0.
        } else if d[i].signum() != d[j].signum() && m.abs() > 3. * d[i].abs() {
            3. * d[i]
        } else {
            m
        }
    };
    let mut slopes = vec![end(0, 1)];
    for i in 1..d.len() {
        let (left, right) = (d[i - 1], d[i]);
        slopes.push(if left * right <= 0. {
            0.
        } else {
            let (w1, w2) = (2. * h[i] + h[i - 1], h[i] + 2. * h[i - 1]);
            (w1 + w2) / (w1 / left + w2 / right)
        });
    }
    slopes.push(end(d.len() - 1, d.len() - 2));
    slopes
}

pub(super) struct RimCurve<'a> {
    points: &'a [Vec3],
    pub x: Vec<f64>,
    slopes: [Vec<f64>; 2],
}

impl<'a> RimCurve<'a> {
    pub fn new(points: &'a [Vec3]) -> Result<Self> {
        let x: Vec<_> = points.iter().map(|p| p[0]).collect();
        ensure!(
            points.iter().flatten().all(|v| v.is_finite())
                && x.len() >= 2
                && x.windows(2).all(|p| p[0] < p[1]),
            "invalid ordered lip rim"
        );
        let slopes = [1, 2].map(|k| tangents(&x, &points.iter().map(|p| p[k]).collect::<Vec<_>>()));
        Ok(Self { points, x, slopes })
    }

    pub fn at(&self, xx: f64) -> Vec3 {
        let x = &self.x;
        let trace = self.points;
        let slopes = &self.slopes;
        let k = x
            .partition_point(|p| *p <= xx)
            .saturating_sub(1)
            .min(x.len() - 2);
        let h = x[k + 1] - x[k];
        let t = (xx - x[k]) / h;
        let mut target = [xx, 0., 0.];
        for axis in 1..3 {
            target[axis] = (1. + 2. * t) * (1. - t).powi(2) * trace[k][axis]
                + t * (1. - t).powi(2) * h * slopes[axis - 1][k]
                + t * t * (3. - 2. * t) * trace[k + 1][axis]
                + t * t * (t - 1.) * h * slopes[axis - 1][k + 1];
        }
        target
    }
}
