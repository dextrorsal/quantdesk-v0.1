use anchor_lang::prelude::*;

/// Collateral Management Module
/// Handles collateral accounts and cross-collateralization

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum CollateralType {
    SOL,
    USDC,
    BTC,
    ETH,
    USDT,
    AVAX,
    MATIC,
    ARB,
    OP,
    DOGE,
    ADA,
    DOT,
    LINK,
}

impl std::fmt::Display for CollateralType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CollateralType::SOL => write!(f, "SOL"),
            CollateralType::USDC => write!(f, "USDC"),
            CollateralType::BTC => write!(f, "BTC"),
            CollateralType::ETH => write!(f, "ETH"),
            CollateralType::USDT => write!(f, "USDT"),
            CollateralType::AVAX => write!(f, "AVAX"),
            CollateralType::MATIC => write!(f, "MATIC"),
            CollateralType::ARB => write!(f, "ARB"),
            CollateralType::OP => write!(f, "OP"),
            CollateralType::DOGE => write!(f, "DOGE"),
            CollateralType::ADA => write!(f, "ADA"),
            CollateralType::DOT => write!(f, "DOT"),
            CollateralType::LINK => write!(f, "LINK"),
        }
    }
}

#[account]
pub struct CollateralAccount {
    pub user: Pubkey,           // User who owns the collateral
    pub asset_type: CollateralType, // Type of collateral asset
    pub amount: u64,             // Amount of collateral
    
    // NEW: Asset weight configuration (Drift-style)
    pub initial_asset_weight: u16,      // e.g., 8000 = 80%
    pub maintenance_asset_weight: u16,  // e.g., 9000 = 90%
    pub initial_liability_weight: u16,  // e.g., 12000 = 120%
    pub maintenance_liability_weight: u16, // e.g., 11000 = 110%
    
    // Price tracking
    pub value_usd: u64,         // USD value of collateral
    pub last_price: u64,         // NEW: Last oracle price
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
