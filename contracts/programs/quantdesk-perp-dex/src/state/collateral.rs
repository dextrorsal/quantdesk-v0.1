use anchor_lang::prelude::*;
use crate::collateral::CollateralType;

/// Collateral State Module
/// Contains account structures for collateral management

/// User collateral account - tracks collateral deposited by individual users
#[account]
pub struct CollateralAccount {
    pub user: Pubkey,           // User who owns the collateral
    pub asset_type: CollateralType, // Type of collateral asset
    pub amount: u64,             // Amount of collateral
    
    // Asset weight configuration (Drift-style)
    pub initial_asset_weight: u16,      // e.g., 8000 = 80%
    pub maintenance_asset_weight: u16,  // e.g., 9000 = 90%
    pub initial_liability_weight: u16,  // e.g., 12000 = 120%
    pub maintenance_liability_weight: u16, // e.g., 11000 = 110%
    
    // Price tracking
    pub value_usd: u64,         // USD value of collateral
    pub last_price: u64,         // Last oracle price
    pub last_updated: i64,      // Last price update timestamp
    pub is_active: bool,         // Whether this collateral is active
    pub bump: u8,              // PDA bump
}

impl CollateralAccount {
    pub const INIT_SPACE: usize = 32 + 1 + 8 + 2 + 2 + 2 + 2 + 8 + 8 + 8 + 1 + 1;
    
    /// Calculate weighted collateral value for initial margin
    pub fn calculate_initial_margin_contribution(&self) -> u64 {
        self.value_usd.checked_mul(self.initial_asset_weight as u64).unwrap().checked_div(10000).unwrap()
    }
    
    /// Calculate weighted collateral value for maintenance margin
    pub fn calculate_maintenance_margin_contribution(&self) -> u64 {
        self.value_usd.checked_mul(self.maintenance_asset_weight as u64).unwrap().checked_div(10000).unwrap()
    }
    
    /// Calculate maintenance margin requirement (alias for compatibility)
    pub fn get_maintenance_margin(&self) -> u64 {
        self.calculate_maintenance_margin_contribution()
    }
}

/// Protocol collateral configuration - defines collateral parameters per asset type
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