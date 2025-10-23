use anchor_lang::prelude::*;

/// Market Management Module
/// Handles market data, pricing, and market-specific operations
/// 
/// NOTE: Market struct has been moved to state/market.rs for better organization

// Re-export Market from state module for backward compatibility
pub use crate::state::market::Market;