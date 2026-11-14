//! Bind the reconstructed face surface before GLB assembly. Image-chart heads
//! use calibrated pixels; volumetric heads measure the same features in metres.
use super::read;
use anyhow::{Context, Result, ensure};
use charloom_core::Vec3;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::Path;

#[derive(Deserialize)]
struct Arc {
    x: Vec<f64>,
    upper: Vec<f64>,
    lower: Vec<f64>,
    #[serde(default)]
    depth: Vec<f64>,
}

impl Arc {
    fn validate(&self, volume: bool) -> Result<()> {
        ensure!(
            self.x.len() >= 2 && self.x.windows(2).all(|v| v[0] < v[1]),
            "invalid arc abscissae"
        );
        ensure!(
            self.upper.len() == self.x.len() && self.lower.len() == self.x.len(),
            "incomplete arc"
        );
        ensure!(
            !volume || self.depth.len() == self.x.len(),
            "missing arc depth"
        );
        Ok(())
    }
    fn width(&self) -> f64 {
        self.x[self.x.len() - 1] - self.x[0]
    }
    fn at(&self, x: f64) -> (f64, f64, f64, f64) {
        let x = x.clamp(self.x[0], self.x[self.x.len() - 1]);
        let next = self
            .x
            .partition_point(|v| *v <= x)
            .clamp(1, self.x.len() - 1);
        let t = (x - self.x[next - 1]) / (self.x[next] - self.x[next - 1]);
        let lerp = |v: &[f64]| v[next - 1] + t * (v[next] - v[next - 1]);
        (
            x,
            lerp(&self.upper),
            lerp(&self.lower),
            if self.depth.is_empty() {
                0.
            } else {
                lerp(&self.depth)
            },
        )
    }
}

#[derive(Deserialize)]
struct Eye {
    arc: Arc,
    center: Vec3,
    radius: f64,
}

impl Eye {
    fn front(&self, x: f64, y: f64) -> (f64, bool) {
        let disc =
            self.radius.powi(2) - (x - self.center[0]).powi(2) - (y - self.center[1]).powi(2);
        (self.center[2] + disc.max(0.).sqrt(), disc > 0.)
    }
}

#[derive(Deserialize)]
struct Chart {
    pixels: Vec<[f64; 2]>,
    metres_per_pixel: f64,
    // The chart uses the monotone cubic traced lip rather than the linear
    // interpolation of its sampled opening. Residuals are in image rows.
    mouth_row: Vec<f64>,
}

#[derive(Deserialize)]
struct Surface {
    positions: Vec<Vec3>,
    joints: Vec<String>,
    eyes: [Eye; 2],
    mouth: Arc,
    chart: Option<Chart>,
}

#[derive(Serialize)]
struct Binding {
    joints: Vec<[usize; 4]>,
    weights: Vec<[f64; 4]>,
    blink: Vec<Vec3>,
}

pub(super) fn smooth(x: f64) -> f64 {
    let x = x.clamp(0., 1.);
    x * x * (3. - 2. * x)
}

fn bind(surface: Surface) -> Result<Binding> {
    let chart = surface.chart.as_ref();
    let volume = chart.is_none();
    let count = surface.positions.len();
    surface.mouth.validate(volume)?;
    for eye in &surface.eyes {
        eye.arc.validate(volume)?;
        ensure!(eye.radius > 0., "invalid eye radius");
    }
    if let Some(c) = chart {
        ensure!(
            c.metres_per_pixel > 0. && c.pixels.len() == count && c.mouth_row.len() == count,
            "invalid registered chart"
        );
    }
    let bone = |name: &str| {
        surface
            .joints
            .iter()
            .position(|n| n == name)
            .with_context(|| format!("missing face joint {name}"))
    };
    let head = bone("head")?;
    let jaw = bone("jaw")?;
    let middle = [bone("oris05")?, bone("oris01")?];
    let outer = [
        [bone("oris03.L")?, bone("oris03.R")?],
        [bone("oris07.L")?, bone("oris07.R")?],
    ];
    let mut out = Binding {
        joints: Vec::with_capacity(count),
        weights: Vec::with_capacity(count),
        blink: Vec::with_capacity(count),
    };
    let mouth = &surface.mouth;
    let mid = mouth.x.iter().sum::<f64>() / mouth.x.len() as f64;
    for (i, &[vx, vy, vz]) in surface.positions.iter().enumerate() {
        let [x, y] = chart.map_or([vx, vy], |c| c.pixels[i]);
        let mut dy = 0.;
        let mut dz = 0.;
        for eye in &surface.eyes {
            let (xx, a, b, z) = eye.arc.at(x);
            let upper = if volume {
                y >= (a + b) * 0.5
            } else {
                y < (a + b) * 0.5
            };
            let d = (y - if upper { a } else { b }).abs();
            let width = eye.arc.width();
            if let Some(c) = chart {
                let influence = smooth(1. - d / (width * 0.46)) * smooth(1. - (x - xx).abs() / 4.);
                dy -= c.metres_per_pixel * (b - a) * if upper { 0.77 } else { -0.23 } * influence;
                dz += (eye.front(vx, vy + dy).0 + 0.002 - vz) * influence;
            } else if vz > z - 0.042 && (x - xx).abs() < width * 0.19 {
                let influence =
                    smooth(1. - d / (width * 0.6)) * smooth(1. - (x - xx).abs() / (width * 0.18));
                dy += (a - b) * if upper { -0.75 } else { 0.25 } * influence;
                let (front, inside) = eye.front(vx, vy + dy);
                if inside && d < width * 0.8 {
                    dz = dz.max(front + 0.003 - vz);
                }
            }
        }
        if chart.is_some() {
            for eye in &surface.eyes {
                let (front, inside) = eye.front(vx, vy + dy);
                if inside {
                    dz = dz.max(front + 0.0025 - vz);
                }
            }
        }
        let radius = mouth.width() / 2.;
        let u = (x - mid).abs() / radius;
        let (upper, lip, mandible) = if let Some(c) = chart {
            let distance = y - c.mouth_row[i];
            let lip = smooth((1.12 - u) / 0.16) * smooth(1. - distance.abs() / (radius * 0.55));
            let jaw = if distance > 0. {
                (1. - lip)
                    * smooth(1. - u / 1.35)
                    * smooth(1. - distance.abs() / (radius * 1.15))
                    * 0.55
            } else {
                0.
            };
            (distance < 0., lip, jaw)
        } else {
            let (xx, a, b, z) = mouth.at(x);
            let distance = y - (a + b) * 0.5;
            let width = mouth.width();
            let active = vz > z - 0.024 && (x - xx).abs() < width * 0.19;
            let fade = if active {
                smooth((1.09 - u) / 0.19) * (1. - ((z - vz - 0.004) / 0.020).clamp(0., 1.))
            } else {
                0.
            };
            let lip = fade * smooth(1. - distance.abs() / (width * 0.24));
            let jaw = if distance < 0. {
                (1. - lip) * fade * smooth(1. - distance.abs() / (width * 0.62)) * 0.66
            } else {
                0.
            };
            (distance >= 0., lip, jaw)
        };
        let rim = usize::from(!upper);
        let side = usize::from(x <= mid);
        let lateral = u.min(1.);
        out.joints.push([head, middle[rim], outer[rim][side], jaw]);
        out.weights.push([
            1. - lip - mandible,
            lip * (1. - lateral),
            lip * lateral,
            mandible,
        ]);
        out.blink.push([0., dy, dz]);
    }
    Ok(out)
}

pub fn bind_face(input: &Path, out: &Path) -> Result<Value> {
    let result = bind(read(input)?)?;
    charloom_io::write_json(out, &result)?;
    Ok(json!({"vertices":result.joints.len(),"output":out}))
}
