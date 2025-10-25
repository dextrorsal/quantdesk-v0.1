// Instructions module - aggregates all instruction handlers
// This module organizes instruction handlers by domain for better maintainability

pub mod market_management;
pub mod position_management;
pub mod order_management;
pub mod collateral_management;
pub mod token_operations;
pub mod user_account_management;
pub mod insurance_protocol;
pub mod oracle_management;
pub mod admin_functions;
pub mod advanced_orders;
pub mod emergency_functions;
pub mod cross_collateral;

// Re-export all instruction handlers for easy access
pub use market_management::*;
pub use position_management::*;
pub use order_management::*;
pub use collateral_management::*;
pub use token_operations::*;
pub use user_account_management::*;
pub use insurance_protocol::*;
pub use oracle_management::*;
pub use admin_functions::*;
pub use advanced_orders::*;
pub use emergency_functions::*;
pub use cross_collateral::*;