use nalgebra::{Matrix4, Quaternion, Translation3, UnitQuaternion, Vector3};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub type Vec3 = [f64; 3];
pub type Quat = [f64; 4];
/// Column-major, as used by glTF. `matrix[column][row]`.
pub type Mat4 = [[f64; 4]; 4];

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0; 3],
        }
    }
}

impl Transform {
    pub fn translated(translation: Vec3) -> Self {
        Self {
            translation,
            ..Self::default()
        }
    }

    pub fn matrix(self) -> Matrix4<f64> {
        Translation3::from(v3(self.translation)).to_homogeneous()
            * unit_quat(self.rotation).to_homogeneous()
            * Matrix4::new_nonuniform_scaling(&v3(self.scale))
    }

    pub fn is_valid(self) -> bool {
        self.translation
            .iter()
            .chain(&self.rotation)
            .chain(&self.scale)
            .all(|v| v.is_finite())
            && self.rotation.iter().map(|v| v * v).sum::<f64>() > 1e-20
    }

    /// Reference and grip frames must be invertible. Animated scene nodes may
    /// use zero scale to hide a rigid accessory and still support forward FK.
    pub fn is_invertible(self) -> bool {
        self.is_valid() && self.scale.iter().all(|v| v.abs() > 1e-12)
    }
}

pub fn v3(v: Vec3) -> Vector3<f64> {
    Vector3::from(v)
}
pub fn a3(v: Vector3<f64>) -> Vec3 {
    v.into()
}
pub fn unit_quat(q: Quat) -> UnitQuaternion<f64> {
    UnitQuaternion::new_normalize(Quaternion::new(q[3], q[0], q[1], q[2]))
}
pub fn a4(q: UnitQuaternion<f64>) -> Quat {
    [q.i, q.j, q.k, q.w]
}
pub fn matrix_array(m: &Matrix4<f64>) -> Mat4 {
    std::array::from_fn(|c| std::array::from_fn(|r| m[(r, c)]))
}
pub fn matrix_from(m: Mat4) -> Matrix4<f64> {
    Matrix4::from_fn(|r, c| m[c][r])
}
pub fn transform_point(m: &Matrix4<f64>, p: Vec3) -> Vec3 {
    let q = m * v3(p).push(1.0);
    [q.x / q.w, q.y / q.w, q.z / q.w]
}
pub fn point_segment_distance(p: Vec3, a: Vec3, b: Vec3) -> f64 {
    let ab = v3(b) - v3(a);
    let t = if ab.norm_squared() > 1e-20 {
        ((v3(p) - v3(a)).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (v3(p) - (v3(a) + t * ab)).norm()
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
pub struct Bounds {
    pub min: Vec3,
    pub max: Vec3,
}
impl Bounds {
    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Option<Self> {
        let mut it = points.into_iter();
        let first = it.next()?;
        let mut b = Self {
            min: first,
            max: first,
        };
        for p in it {
            for (axis, value) in p.into_iter().enumerate() {
                b.min[axis] = b.min[axis].min(value);
                b.max[axis] = b.max[axis].max(value);
            }
        }
        Some(b)
    }
    pub fn extent(self) -> Vec3 {
        a3(v3(self.max) - v3(self.min))
    }
    pub fn center(self) -> Vec3 {
        a3((v3(self.max) + v3(self.min)) * 0.5)
    }
    pub fn diagonal(self) -> f64 {
        (v3(self.max) - v3(self.min)).norm()
    }
}
