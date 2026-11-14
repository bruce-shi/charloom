use crate::{CoreError, Quat, Skeleton, Transform, a4, unit_quat};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Property {
    Translation,
    Rotation,
    Scale,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Interpolation {
    Step,
    Linear,
    CubicSpline,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Track {
    pub joint: usize,
    pub property: Property,
    pub interpolation: Interpolation,
    pub times: Vec<f64>,
    /// Cubic spline stores [in tangent, value, out tangent] for every key.
    pub values: Vec<[f64; 4]>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MotionClip {
    pub name: String,
    pub tracks: Vec<Track>,
    #[serde(default)]
    pub morph_tracks: Vec<MorphTrack>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MorphTrack {
    pub node: usize,
    pub interpolation: Interpolation,
    pub times: Vec<f64>,
    /// Per-key weight vectors; cubic keys store in tangent, value, out tangent.
    pub values: Vec<Vec<f64>>,
}

impl MorphTrack {
    pub fn validate(&self) -> Result<(), CoreError> {
        let stride = if self.interpolation == Interpolation::CubicSpline {
            3
        } else {
            1
        };
        if self.times.is_empty()
            || self.times.len() * stride != self.values.len()
            || self.values[0].is_empty()
            || !self.times.iter().all(|t| t.is_finite() && *t >= 0.0)
            || self.times.windows(2).any(|w| w[1] <= w[0])
            || self
                .values
                .iter()
                .any(|v| v.len() != self.values[0].len() || !v.iter().all(|x| x.is_finite()))
        {
            return Err(CoreError::InvalidAnimation("invalid morph track".into()));
        }
        Ok(())
    }
    pub fn sample(&self, time: f64) -> Result<Vec<f64>, CoreError> {
        if !time.is_finite() || self.times.is_empty() {
            return Err(CoreError::InvalidAnimation("invalid morph sample".into()));
        }
        let stride = if self.interpolation == Interpolation::CubicSpline {
            3
        } else {
            1
        };
        let offset = usize::from(stride == 3);
        let key = |i: usize| &self.values[i * stride + offset];
        let upper = self.times.partition_point(|t| *t <= time);
        if upper == 0 {
            return Ok(key(0).clone());
        }
        if upper == self.times.len() {
            return Ok(key(upper - 1).clone());
        }
        let left = upper - 1;
        let dt = self.times[upper] - self.times[left];
        let t = (time - self.times[left]) / dt;
        let a = key(left);
        let b = key(upper);
        Ok((0..a.len())
            .map(|i| match self.interpolation {
                Interpolation::Step => a[i],
                Interpolation::Linear => a[i] + (b[i] - a[i]) * t,
                Interpolation::CubicSpline => {
                    let t2 = t * t;
                    let t3 = t2 * t;
                    (2.0 * t3 - 3.0 * t2 + 1.0) * a[i]
                        + (t3 - 2.0 * t2 + t) * dt * self.values[left * 3 + 2][i]
                        + (-2.0 * t3 + 3.0 * t2) * b[i]
                        + (t3 - t2) * dt * self.values[upper * 3][i]
                }
            })
            .collect())
    }
    pub fn merge_time_collisions(&mut self, tolerance: f64) -> Result<(usize, f64), CoreError> {
        let stride = if self.interpolation == Interpolation::CubicSpline {
            3
        } else {
            1
        };
        if self.values.len() != self.times.len() * stride || self.values.is_empty() {
            return Err(CoreError::InvalidAnimation(
                "invalid morph track shape".into(),
            ));
        }
        let mut times = Vec::new();
        let mut values: Vec<Vec<f64>> = Vec::new();
        let mut count = 0;
        let mut max_delta = 0.0_f64;
        for (i, &time) in self.times.iter().enumerate() {
            let current = &self.values[i * stride..(i + 1) * stride];
            if times.last() == Some(&time) {
                let previous = &values[values.len() - stride..];
                if previous
                    .iter()
                    .zip(current)
                    .any(|(a, b)| a.len() != b.len())
                {
                    return Err(CoreError::InvalidAnimation(
                        "inconsistent morph dimensions".into(),
                    ));
                }
                let delta = previous
                    .iter()
                    .flatten()
                    .zip(current.iter().flatten())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                if !delta.is_finite() || delta > tolerance {
                    return Err(CoreError::InvalidAnimation(format!(
                        "conflicting morph key at {time}"
                    )));
                }
                max_delta = max_delta.max(delta);
                count += 1;
                values.truncate(values.len() - stride);
                values.extend_from_slice(current);
            } else {
                times.push(time);
                values.extend_from_slice(current);
            }
        }
        self.times = times;
        self.values = values;
        self.validate()?;
        Ok((count, max_delta))
    }
}

impl Track {
    /// Some exporters collapse nearby double-precision times to the same f32.
    /// Merge only near-equivalent values (and tangents), retain the last sample.
    /// The caller must report this repair; genuinely conflicting keys are errors.
    pub fn merge_time_collisions(&mut self, tolerance: f64) -> Result<(usize, f64), CoreError> {
        let stride = if self.interpolation == Interpolation::CubicSpline {
            3
        } else {
            1
        };
        if self.times.len() * stride != self.values.len() {
            return Err(CoreError::InvalidAnimation("inconsistent track".into()));
        }
        let mut times = Vec::new();
        let mut values: Vec<[f64; 4]> = Vec::new();
        let mut removed = 0;
        let mut max_delta = 0.0_f64;
        for (i, &time) in self.times.iter().enumerate() {
            let current = &self.values[i * stride..(i + 1) * stride];
            if times.last() == Some(&time) {
                let previous = &values[values.len() - stride..];
                let delta = previous
                    .iter()
                    .flatten()
                    .zip(current.iter().flatten())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);
                if !delta.is_finite() || delta > tolerance {
                    return Err(CoreError::InvalidAnimation(format!(
                        "conflicting duplicate time {time}: value difference {delta}, tolerance {tolerance}"
                    )));
                }
                max_delta = max_delta.max(delta);
                values.truncate(values.len() - stride);
                values.extend_from_slice(current);
                removed += 1;
            } else {
                times.push(time);
                values.extend_from_slice(current);
            }
        }
        self.times = times;
        self.values = values;
        self.validate()?;
        Ok((removed, max_delta))
    }
    pub fn validate(&self) -> Result<(), CoreError> {
        let multiplier = if self.interpolation == Interpolation::CubicSpline {
            3
        } else {
            1
        };
        if self.times.is_empty() || self.values.len() != self.times.len() * multiplier {
            return Err(CoreError::InvalidAnimation(
                "empty or inconsistent track".into(),
            ));
        }
        if !self.times.iter().all(|t| t.is_finite() && *t >= 0.0)
            || self.times.windows(2).any(|w| w[1] <= w[0])
            || !self.values.iter().flatten().all(|v| v.is_finite())
        {
            return Err(CoreError::InvalidAnimation(
                "non-finite values or unordered key times".into(),
            ));
        }
        if self.property == Property::Rotation {
            for i in 0..self.times.len() {
                let q = self.key_value(i);
                if q.iter().map(|v| v * v).sum::<f64>() < 1e-20 {
                    return Err(CoreError::InvalidAnimation(
                        "zero rotation quaternion".into(),
                    ));
                }
            }
        }
        Ok(())
    }
    fn key_value(&self, i: usize) -> Quat {
        self.values[if self.interpolation == Interpolation::CubicSpline {
            3 * i + 1
        } else {
            i
        }]
    }

    pub fn sample(&self, time: f64) -> Result<[f64; 4], CoreError> {
        if !time.is_finite() || self.times.is_empty() {
            return Err(CoreError::InvalidAnimation(
                "invalid sampling time or empty track".into(),
            ));
        }
        let upper = self.times.partition_point(|t| *t <= time);
        if upper == 0 {
            return Ok(self.key_value(0));
        }
        if upper == self.times.len() {
            return Ok(self.key_value(upper - 1));
        }
        let left = upper - 1;
        let a = self.key_value(left);
        let b = self.key_value(upper);
        let dt = self.times[upper] - self.times[left];
        let t = (time - self.times[left]) / dt;
        let mut value = match self.interpolation {
            Interpolation::Step => a,
            Interpolation::Linear if self.property == Property::Rotation => {
                a4(unit_quat(a).slerp(&unit_quat(b), t))
            }
            Interpolation::Linear => std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t),
            Interpolation::CubicSpline => {
                let t2 = t * t;
                let t3 = t2 * t;
                let out_tangent = self.values[3 * left + 2];
                let in_tangent = self.values[3 * upper];
                std::array::from_fn(|i| {
                    (2.0 * t3 - 3.0 * t2 + 1.0) * a[i]
                        + (t3 - 2.0 * t2 + t) * dt * out_tangent[i]
                        + (-2.0 * t3 + 3.0 * t2) * b[i]
                        + (t3 - t2) * dt * in_tangent[i]
                })
            }
        };
        if self.property == Property::Rotation {
            if value.iter().map(|x| x * x).sum::<f64>() < 1e-20 {
                return Err(CoreError::InvalidAnimation(
                    "degenerate interpolated rotation".into(),
                ));
            }
            value = a4(unit_quat(value));
        }
        Ok(value)
    }
}

impl MotionClip {
    pub fn time_range(&self) -> [f64; 2] {
        let start = self
            .tracks
            .iter()
            .filter_map(|t| t.times.first())
            .chain(self.morph_tracks.iter().filter_map(|t| t.times.first()))
            .copied()
            .fold(f64::INFINITY, f64::min);
        let end = self
            .tracks
            .iter()
            .filter_map(|t| t.times.last())
            .chain(self.morph_tracks.iter().filter_map(|t| t.times.last()))
            .copied()
            .fold(0.0, f64::max);
        [if start.is_finite() { start } else { 0.0 }, end]
    }
    pub fn validate(&self, skeleton: &Skeleton) -> Result<(), CoreError> {
        let mut targets = std::collections::HashSet::new();
        for track in &self.tracks {
            track.validate()?;
            if track.joint >= skeleton.joints.len() || skeleton.joints[track.joint].matrix.is_some()
            {
                return Err(CoreError::InvalidAnimation(
                    "invalid or matrix-based animation target".into(),
                ));
            }
            if !targets.insert((track.joint, track.property as u8)) {
                return Err(CoreError::InvalidAnimation(
                    "duplicate animation target/property".into(),
                ));
            }
        }
        let mut morph_nodes = std::collections::HashSet::new();
        for track in &self.morph_tracks {
            track.validate()?;
            if track.node >= skeleton.joints.len() || !morph_nodes.insert(track.node) {
                return Err(CoreError::InvalidAnimation(
                    "invalid or duplicate morph node".into(),
                ));
            }
        }
        Ok(())
    }
    pub fn sample(&self, skeleton: &Skeleton, time: f64) -> Result<Vec<Transform>, CoreError> {
        let mut pose = skeleton.rest_pose();
        for track in &self.tracks {
            let v = track.sample(time)?;
            let joint = pose
                .get_mut(track.joint)
                .ok_or_else(|| CoreError::InvalidAnimation("target joint out of bounds".into()))?;
            match track.property {
                Property::Translation => joint.translation = [v[0], v[1], v[2]],
                Property::Rotation => joint.rotation = v,
                Property::Scale => joint.scale = [v[0], v[1], v[2]],
            }
        }
        Ok(pose)
    }
    pub fn sample_morphs(
        &self,
        time: f64,
    ) -> Result<std::collections::HashMap<usize, Vec<f64>>, CoreError> {
        self.morph_tracks
            .iter()
            .map(|t| Ok((t.node, t.sample(time)?)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quaternion_interpolation_takes_short_arc() {
        let t = Track {
            joint: 0,
            property: Property::Rotation,
            interpolation: Interpolation::Linear,
            times: vec![0.0, 1.0],
            values: vec![[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, -1.0]],
        };
        t.validate().unwrap();
        assert!(t.sample(0.5).unwrap()[3].abs() > 0.999999);
    }
    #[test]
    fn cubic_tangents_use_key_interval_in_seconds() {
        let t = Track {
            joint: 0,
            property: Property::Translation,
            interpolation: Interpolation::CubicSpline,
            times: vec![0.0, 2.0],
            values: vec![
                [0.0; 4],
                [0.0; 4],
                [2.0, 0.0, 0.0, 0.0],
                [0.0; 4],
                [0.0; 4],
                [0.0; 4],
            ],
        };
        assert!((t.sample(1.0).unwrap()[0] - 0.5).abs() < 1e-12);
        assert_eq!(t.sample(-1.0).unwrap(), [0.0; 4]);
    }
    #[test]
    fn time_collisions_are_bounded_and_keep_the_last_value() {
        let mut t = Track {
            joint: 0,
            property: Property::Translation,
            interpolation: Interpolation::Linear,
            times: vec![0.0, 0.5, 0.5, 1.0],
            values: vec![
                [0.0; 4],
                [1.0, 0.0, 0.0, 0.0],
                [1.000001, 0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
            ],
        };
        assert_eq!(t.merge_time_collisions(1e-5).unwrap().0, 1);
        assert_eq!(t.sample(0.5).unwrap()[0], 1.000001);
        t.times = vec![0.0, 0.0, 1.0];
        assert!(t.merge_time_collisions(1e-5).is_err());
    }
}
