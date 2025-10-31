// State module - aggregates all account state definitions
// This module organizes account structs by domain for better maintainability

pub mod market;
pub mod protocol;
pub mod advanced;
pub mod remaining;
pub mod order;
pub mod position;
pub mod user_account;
pub mod price_cache;
pub mod collateral;

// Re-export all state structs for easy access
pub use market::*;
pub use protocol::*;
pub use advanced::*;
pub use remaining::*;
pub use order::*;
pub use position::*;
pub use user_account::*;
pub use price_cache::*;
pub use collateral::*;
