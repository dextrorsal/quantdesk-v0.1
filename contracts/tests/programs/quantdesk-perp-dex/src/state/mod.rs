// State module - aggregates all account state definitions
// This module organizes account structs by domain for better maintainability

pub mod market;
pub mod position;
pub mod order;
pub mod collateral;
pub mod user_account;
pub mod protocol;

// Re-export all state structs for easy access
pub use market::*;
pub use position::*;
pub use order::*;
pub use collateral::*;
pub use user_account::*;
pub use protocol::*;