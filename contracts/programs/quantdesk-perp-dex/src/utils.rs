use anchor_lang::prelude::*;
use crate::ErrorCode;

/// Utility functions and constants for QuantDesk Perpetual DEX
/// This module contains helper functions that were previously in lib.rs

/// Validate price data
pub fn validate_price(price: u64) -> Result<()> {
    require!(price > 0, ErrorCode::InvalidPrice);
    Ok(())
}

/// Constants
pub const MAX_POSITIONS_PER_USER: u8 = 10;
pub const MAX_LEVERAGE: u8 = 100;
pub const MIN_POSITION_SIZE: u64 = 1000; // 0.001 tokens
pub const MAX_POSITION_SIZE: u64 = 1000000000; // 1000 tokens
pub const FUNDING_INTERVAL: i64 = 3600; // 1 hour
pub const PRICE_STALENESS_THRESHOLD: i64 = 300; // 5 minutes
pub const BASIS_POINTS_DIVISOR: u64 = 10000;
