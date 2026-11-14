use crate::{Bounds, Mat4, MotionClip, Transform, Vec3, matrix_from, transform_point};
use nalgebra::Matrix4;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, thiserror::Error)]
pub enum CoreError {
    #[error("invalid skeleton: {0}")]
    InvalidSkeleton(String),
    #[error("invalid animation: {0}")]
    InvalidAnimation(String),
    #[error("invalid asset: {0}")]
    InvalidAsset(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Joint {
    pub id: String,
    pub name: String,
    pub parent: Option<usize>,
    pub rest: Transform,
    /// Preserves a static affine node exactly, including shear. Animated nodes use TRS.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix: Option<Mat4>,
    #[serde(default)]
    pub deform: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct Skeleton {
    pub joints: Vec<Joint>,
}

impl Skeleton {
    pub fn rest_pose(&self) -> Vec<Transform> {
        self.joints.iter().map(|j| j.rest).collect()
    }

    pub fn validate(&self) -> Result<(), CoreError> {
        let mut names = std::collections::HashSet::new();
        for (i, joint) in self.joints.iter().enumerate() {
            if !names.insert(&joint.id) {
                return Err(CoreError::InvalidSkeleton(format!(
                    "duplicate joint ID {}",
                    joint.id
                )));
            }
            if !joint.rest.is_invertible() {
                return Err(CoreError::InvalidSkeleton(format!(
                    "joint {i} has invalid TRS"
                )));
            }
            if let Some(m) = joint.matrix
                && (!m.iter().flatten().all(|x| x.is_finite())
                    || matrix_from(m).determinant().abs() < 1e-15)
            {
                return Err(CoreError::InvalidSkeleton(format!(
                    "joint {i} has singular/non-finite matrix"
                )));
            }
            if joint
                .parent
                .is_some_and(|p| p >= self.joints.len() || p == i)
            {
                return Err(CoreError::InvalidSkeleton(format!(
                    "joint {i} has invalid parent"
                )));
            }
        }
        self.world_matrices(&self.rest_pose()).map(|_| ())
    }

    pub fn world_matrices(&self, pose: &[Transform]) -> Result<Vec<Matrix4<f64>>, CoreError> {
        if pose.len() != self.joints.len() {
            return Err(CoreError::InvalidSkeleton(
                "pose length does not match skeleton".into(),
            ));
        }
        let n = self.joints.len();
        let mut state = vec![0u8; n];
        let mut output = vec![Matrix4::identity(); n];
        // Iterative ancestor walk also handles files whose node order is not topological.
        for start in 0..n {
            if state[start] == 2 {
                continue;
            }
            let mut stack = Vec::new();
            let mut at = Some(start);
            while let Some(i) = at {
                if i >= n {
                    return Err(CoreError::InvalidSkeleton(
                        "parent index out of bounds".into(),
                    ));
                }
                if state[i] == 2 {
                    break;
                }
                if state[i] == 1 {
                    return Err(CoreError::InvalidSkeleton("parent cycle".into()));
                }
                state[i] = 1;
                stack.push(i);
                at = self.joints[i].parent;
            }
            while let Some(i) = stack.pop() {
                if !pose[i].is_valid() {
                    return Err(CoreError::InvalidSkeleton(format!(
                        "non-finite/degenerate pose at joint {i}"
                    )));
                }
                let local = match self.joints[i].matrix {
                    Some(m) if pose[i] == self.joints[i].rest => matrix_from(m),
                    Some(_) => {
                        return Err(CoreError::InvalidSkeleton(format!(
                            "matrix node {i} cannot be animated as TRS"
                        )));
                    }
                    None => pose[i].matrix(),
                };
                output[i] = self.joints[i].parent.map_or(local, |p| output[p] * local);
                state[i] = 2;
            }
        }
        Ok(output)
    }

    pub fn positions(&self, pose: &[Transform]) -> Result<Vec<Vec3>, CoreError> {
        Ok(self
            .world_matrices(pose)?
            .iter()
            .map(|m| transform_point(m, [0.0; 3]))
            .collect())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema)]
pub struct Influence {
    pub joint: usize,
    pub weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Mesh {
    pub name: String,
    pub node: usize,
    pub primitive: usize,
    /// Positions in the original mesh node's local space.
    pub positions: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
    pub skin: Option<usize>,
    /// Joint indices are global skeleton node indices, not skin-local indices.
    pub influences: Vec<Vec<Influence>>,
    #[serde(default)]
    pub morph_targets: Vec<Vec<Vec3>>,
    #[serde(default)]
    pub morph_weights: Vec<f64>,
}

impl Mesh {
    fn morphed_positions(&self, weights: Option<&Vec<f64>>) -> Result<Vec<Vec3>, CoreError> {
        let weights = weights.unwrap_or(&self.morph_weights);
        if weights.len() != self.morph_targets.len() || !weights.iter().all(|v| v.is_finite()) {
            return Err(CoreError::InvalidAsset(
                "morph target/weight mismatch".into(),
            ));
        }
        let mut positions = self.positions.clone();
        for (target, &weight) in self.morph_targets.iter().zip(weights) {
            if target.len() != positions.len() {
                return Err(CoreError::InvalidAsset(
                    "morph vertex count mismatch".into(),
                ));
            }
            if weight == 0.0 {
                continue;
            }
            for (p, delta) in positions.iter_mut().zip(target) {
                for axis in 0..3 {
                    p[axis] += delta[axis] * weight;
                }
            }
        }
        Ok(positions)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Skin {
    pub name: String,
    pub joints: Vec<usize>,
    pub inverse_bind_matrices: Vec<Mat4>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Asset {
    pub name: String,
    pub source_hash: String,
    pub skeleton: Skeleton,
    pub meshes: Vec<Mesh>,
    pub skins: Vec<Skin>,
    pub clips: Vec<MotionClip>,
    pub issues: Vec<Issue>,
}

impl Asset {
    pub fn world_meshes(&self) -> Result<Vec<Vec<Vec3>>, CoreError> {
        let worlds = self.skeleton.world_matrices(&self.skeleton.rest_pose())?;
        self.meshes
            .iter()
            .map(|m| {
                let w = worlds
                    .get(m.node)
                    .ok_or_else(|| CoreError::InvalidAsset("mesh node out of bounds".into()))?;
                Ok(m.morphed_positions(None)?
                    .iter()
                    .map(|&p| transform_point(w, p))
                    .collect())
            })
            .collect()
    }
    pub fn bounds(&self) -> Result<Option<Bounds>, CoreError> {
        Ok(Bounds::from_points(
            self.world_meshes()?.into_iter().flatten(),
        ))
    }
    pub fn skinned_positions(&self, pose: &[Transform]) -> Result<Vec<Vec<Vec3>>, CoreError> {
        self.deformed_positions(pose, &std::collections::HashMap::new())
    }
    pub fn deformed_positions(
        &self,
        pose: &[Transform],
        morphs: &std::collections::HashMap<usize, Vec<f64>>,
    ) -> Result<Vec<Vec<Vec3>>, CoreError> {
        let world = self.skeleton.world_matrices(pose)?;
        self.meshes
            .iter()
            .map(|mesh| {
                let positions = mesh.morphed_positions(morphs.get(&mesh.node))?;
                let Some(skin_idx) = mesh.skin else {
                    return Ok(positions
                        .iter()
                        .map(|&p| transform_point(&world[mesh.node], p))
                        .collect());
                };
                let skin = self
                    .skins
                    .get(skin_idx)
                    .ok_or_else(|| CoreError::InvalidAsset("skin out of bounds".into()))?;
                if skin.joints.len() != skin.inverse_bind_matrices.len()
                    || mesh.influences.len() != mesh.positions.len()
                {
                    return Err(CoreError::InvalidAsset(
                        "skin/weights length mismatch".into(),
                    ));
                }
                let mut matrices = std::collections::HashMap::new();
                for (&joint, ibm) in skin.joints.iter().zip(&skin.inverse_bind_matrices) {
                    let w = world.get(joint).ok_or_else(|| {
                        CoreError::InvalidAsset("skin joint out of bounds".into())
                    })?;
                    matrices.insert(joint, w * matrix_from(*ibm));
                }
                positions
                    .iter()
                    .zip(&mesh.influences)
                    .map(|(&p, influences)| {
                        let mut sum = nalgebra::Vector3::zeros();
                        let mut total = 0.0;
                        for influence in influences {
                            let m = matrices.get(&influence.joint).ok_or_else(|| {
                                CoreError::InvalidAsset(
                                    "weight references joint outside skin".into(),
                                )
                            })?;
                            sum += crate::v3(transform_point(m, p)) * influence.weight;
                            total += influence.weight;
                        }
                        if (total - 1.0).abs() > 1e-3 {
                            return Err(CoreError::InvalidAsset(
                                "weights do not sum to one".into(),
                            ));
                        }
                        Ok(crate::a3(sum))
                    })
                    .collect()
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Info,
    Warning,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Issue {
    pub code: String,
    pub severity: Severity,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub location: Option<String>,
}
impl Issue {
    pub fn warning(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Warning,
            message: message.into(),
            location: None,
        }
    }
    pub fn error(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            severity: Severity::Error,
            message: message.into(),
            location: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Region {
    pub id: String,
    pub parent: Option<usize>,
    pub center: Vec3,
    pub radius: f64,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RoleCandidate {
    pub role: String,
    /// Heuristic score; not a calibrated probability.
    pub score: f64,
    pub evidence: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MorphologyChain {
    pub id: String,
    pub regions: Vec<usize>,
    pub length: f64,
    pub roles: Vec<RoleCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SymmetryPair {
    pub a: String,
    pub b: String,
    pub plane_normal: Vec3,
    pub plane_offset: f64,
    pub normalized_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct MorphologyGraph {
    pub regions: Vec<Region>,
    pub chains: Vec<MorphologyChain>,
    pub symmetry_pairs: Vec<SymmetryPair>,
    /// Optional identity only. Geometry never requires a species label.
    pub species_label: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RigChain {
    pub id: String,
    pub joints: Vec<usize>,
    pub role: String,
    pub source_chain: String,
    /// Authored world-space flexion axis for deformation probes. Geometry-only
    /// candidates omit this rather than inventing an anatomical palm plane.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flexion_axis_world: Option<Vec3>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct RigBlueprint {
    pub schema_version: String,
    pub source_hash: String,
    pub method: String,
    pub skeleton: Skeleton,
    pub chains: Vec<RigChain>,
    pub issues: Vec<Issue>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct AnalysisResult {
    pub schema_version: String,
    pub source_hash: String,
    pub method: String,
    pub analysis_status: String,
    pub morphology: MorphologyGraph,
    pub blueprint: RigBlueprint,
    pub metrics: serde_json::Value,
    pub issues: Vec<Issue>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{a4, v3};
    use nalgebra::UnitQuaternion;

    fn joint(id: &str, parent: Option<usize>, p: Vec3) -> Joint {
        Joint {
            id: id.into(),
            name: id.into(),
            parent,
            rest: Transform::translated(p),
            matrix: None,
            deform: true,
        }
    }

    #[test]
    fn fk_handles_unsorted_parents_and_rotated_local_axes() {
        let mut root = joint("root", None, [3.0, 2.0, 1.0]);
        root.rest.rotation = a4(UnitQuaternion::from_axis_angle(
            &nalgebra::Vector3::z_axis(),
            std::f64::consts::FRAC_PI_2,
        ));
        let skeleton = Skeleton {
            joints: vec![joint("tip", Some(1), [2.0, 0.0, 0.0]), root],
        };
        skeleton.validate().unwrap();
        let p = skeleton.positions(&skeleton.rest_pose()).unwrap();
        assert!((v3(p[0]) - v3([3.0, 4.0, 1.0])).norm() < 1e-10);
    }

    #[test]
    fn forward_pose_allows_hidden_zero_scale_accessories() {
        let s = Skeleton {
            joints: vec![
                joint("body", None, [1.0, 2.0, 3.0]),
                joint("arrow", Some(0), [0.0, 1.0, 0.0]),
                joint("arrow_tip", Some(1), [0.0, 0.0, 2.0]),
            ],
        };
        let mut pose = s.rest_pose();
        pose[1].scale = [0.0; 3];
        let positions = s.positions(&pose).unwrap();
        assert_eq!(positions[0], [1.0, 2.0, 3.0]);
        assert_eq!(positions[1], positions[2]);
        assert!(pose[1].is_valid());
        assert!(!pose[1].is_invertible());
        pose[1].scale[0] = f64::NAN;
        assert!(s.positions(&pose).is_err());
    }

    #[test]
    fn cycles_and_invalid_rotations_are_rejected() {
        let mut skeleton = Skeleton {
            joints: vec![joint("a", Some(1), [0.0; 3]), joint("b", Some(0), [0.0; 3])],
        };
        assert!(skeleton.validate().is_err());
        skeleton.joints[0].parent = None;
        skeleton.joints[1].rest.rotation = [0.0; 4];
        assert!(skeleton.validate().is_err());
    }
}
