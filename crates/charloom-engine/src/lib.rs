//! Motion, kinematics and skinned-asset tools: retargeting, IK pose solving,
//! contact/prop constraints, animation recipes and hand-skin correctives.
pub mod articulation;
pub mod attachment;
pub mod collision;
pub mod ground;
pub mod hand_corrective;
mod hand_partition;
pub mod hand_shape;
pub mod hand_surface;
pub mod motion;
pub mod pipeline;
pub mod recipe;
mod relative_grip;
pub mod rigid_region;
pub mod skin_repair;
pub mod solve;
pub mod surface_contacts;
pub mod surface_query;
pub mod surface_region;
pub mod surface_topology;
pub mod validate;
