use anchor_lang::prelude::*;
use crate::collateral::CollateralType;

/// Collateral State Module
/// Contains account structures for collateral management

#[account]
pub struct CollateralConfig {
    pub asset_type: CollateralType,      // Type of collateral asset
    pub initial_asset_weight: u16,       // Initial asset weight (basis points)
    pub maintenance_asset_weight: u16,  // Maintenance asset weight (basis points)
    pub initial_liability_weight: u16,   // Initial liability weight (basis points)
    pub maintenance_liability_weight: u16, // Maintenance liability weight (basis points)
    pub imf_factor: u16,                 // IMF factor (basis points)
    pub max_collateral_amount: u64,       // Maximum collateral amount
    pub oracle_price_feed: Pubkey,       // Oracle price feed for this asset
    pub is_active: bool,                 // Whether this collateral type is active
    pub bump: u8,                       // PDA bump
}

impl CollateralConfig {
    pub const INIT_SPACE: usize = 1 + 2 + 2 + 2 + 2 + 2 + 8 + 32 + 1 + 1;
}