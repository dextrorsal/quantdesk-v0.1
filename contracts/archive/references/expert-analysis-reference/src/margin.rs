//! Margin calculation and validation logic
//! This module contains all margin-related calculations and validations
//! to reduce complexity in the main instruction functions.

use anchor_lang::prelude::*;
use crate::state::position::PositionSide;
use crate::ErrorCode;

/// Validate leverage amount
pub fn validate_leverage(leverage: u16, max_leverage: u8) -> Result<()> {
    require!(leverage > 0 && leverage <= max_leverage as u16, ErrorCode::InvalidLeverage);
    Ok(())
}

/// Validate position size
pub fn validate_position_size(size: u64) -> Result<()> {
    require!(size > 0, ErrorCode::InvalidSize);
    require!(size <= 1000000000, ErrorCode::PositionTooLarge); // Example max size
    Ok(())
}

/// Calculate margin requirement for a position
pub fn calculate_margin_requirement(position_value: u64, leverage: u16) -> u64 {
    position_value.checked_div(leverage as u64).unwrap_or(0)
}

/// Calculate liquidation price for a position
pub fn calculate_liquidation_price(
    entry_price: u64,
    side: PositionSide,
    leverage: u16,
    maintenance_margin_ratio: u16,
) -> u64 {
    match side {
        PositionSide::Long => {
            // For long positions: liquidation_price = entry_price * (1 - maintenance_margin_ratio / leverage)
            let ratio = maintenance_margin_ratio as u64 * 10000 / leverage as u64;
            entry_price.checked_mul(10000 - ratio).unwrap().checked_div(10000).unwrap()
        },
        PositionSide::Short => {
            // For short positions: liquidation_price = entry_price * (1 + maintenance_margin_ratio / leverage)
            let ratio = maintenance_margin_ratio as u64 * 10000 / leverage as u64;
            entry_price.checked_mul(10000 + ratio).unwrap().checked_div(10000).unwrap()
        },
    }
}

/// Calculate P&L for a position
pub fn calculate_pnl(
    entry_price: u64,
    current_price: u64,
    size: u64,
    side: PositionSide,
) -> i64 {
    match side {
        PositionSide::Long => {
            ((current_price as i128 - entry_price as i128) * size as i128 / 1000000) as i64
        },
        PositionSide::Short => {
            ((entry_price as i128 - current_price as i128) * size as i128 / 1000000) as i64
        },
    }
}

/// Check if a position is liquidatable
pub fn is_position_liquidatable(
    margin: u64,
    unrealized_pnl: i64,
    position_value: u64,
    maintenance_margin_ratio: u16,
) -> bool {
    let equity = margin as i128 + unrealized_pnl as i128;
    let required_margin = (position_value as i128 * maintenance_margin_ratio as i128) / 10000;
    equity < required_margin
}

/// Calculate funding payment for a position
pub fn calculate_funding_payment(
    size: u64,
    funding_rate: i64,
    time_since_last_funding: i64,
) -> i64 {
    let numerator = size as i128 * funding_rate as i128 * time_since_last_funding as i128;
    let denominator = 1000000i128 * 86400i128;
    (numerator / denominator) as i64
}

/// Constants for margin calculations
pub const MAX_LEVERAGE: u8 = 100;
pub const MIN_LEVERAGE: u8 = 1;
pub const MAINTENANCE_MARGIN_RATIO: u16 = 500; // 5%
pub const INITIAL_MARGIN_RATIO: u16 = 1000; // 10%
pub const LIQUIDATION_THRESHOLD: u16 = 800; // 8%
