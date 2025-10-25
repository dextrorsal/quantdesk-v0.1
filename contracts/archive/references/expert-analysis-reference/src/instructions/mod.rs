// Instructions module - aggregates all instruction handlers
// This module organizes instruction handlers by domain for better maintainability

pub mod market_management;
pub mod position_management;
pub mod order_management;
pub mod admin_functions;
pub mod collateral_management;
pub mod user_account_management;
pub mod insurance_oracle_management;
pub mod vault_management;
pub mod cross_program;
pub mod keeper_management;
pub mod advanced_orders;
pub mod remaining_contexts;
pub mod security_management;

// Re-export all instruction handlers for easy access
pub use market_management::*;
pub use position_management::*;
pub use order_management::*;
pub use admin_functions::*;
pub use collateral_management::*;
pub use user_account_management::*;
pub use insurance_oracle_management::*;
pub use vault_management::*;
pub use cross_program::*;
pub use keeper_management::*;
pub use advanced_orders::*;
pub use remaining_contexts::*;
pub use security_management::*;
