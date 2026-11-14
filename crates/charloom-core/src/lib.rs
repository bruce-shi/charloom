//! Engine-independent asset, morphology, rig, animation, and diagnostic contracts.
//! Right-handed coordinates, meters, seconds, column vectors, XYZW quaternions.

pub mod anatomy;
pub mod animation;
pub mod math;
pub mod model;

pub use animation::*;
pub use math::*;
pub use model::*;

pub const SCHEMA_VERSION: &str = "0.1";
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
