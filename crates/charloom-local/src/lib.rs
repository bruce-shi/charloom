//! Local project authority and the common CLI/MCP/Web protocol.
pub mod client;
pub mod model;
pub mod server;
pub mod store;
pub mod worker;
pub use model::*;
pub const API: &str = "/api/local/v1";
