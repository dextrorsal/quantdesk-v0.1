/// Position State Module
/// Contains all position related structures

use anchor_lang::prelude::*;

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum PositionSide {
    Long,
    Short,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum PositionStatus {
    Open,
    Closed,
    Liquidated,
}

#[account]
pub struct Position {
    pub user_account: Pubkey,       // Reference to UserAccount
    pub user: Pubkey,               // Alias for user_account (for compatibility)
    pub market: Pubkey,             // Market being traded
    pub position_index: u16,        // Position number for this user
    
    pub side: PositionSide,         // Long or Short
    pub status: PositionStatus,     // Open, Closed, Liquidated
    
    // Position details
    pub size: u64,                  // Position size in base units
    pub entry_price: u64,           // Entry price
    pub current_price: u64,         // Current mark price
    pub liquidation_price: u64,     // Liquidation price
    
    // PnL tracking
    pub unrealized_pnl: i64,        // Can be negative
    pub realized_pnl: i64,          // Settled PnL
    
    // Funding tracking
    pub funding_rate: i64,          // Current funding rate
    pub last_funding_update: i64,   // Last funding payment timestamp
    pub total_funding_paid: i64,    // Total funding for this position
    
    // Margin tracking
    pub initial_margin: u64,        // Initial margin requirement
    pub maintenance_margin: u64,    // Maintenance margin requirement
    pub leverage: u16,              // Leverage used (basis points)
    pub margin: u64,                // Current margin amount
    pub total_collateral_value: u64, // Total collateral backing position
    
    // Collateral accounts (fixed-size array instead of Vec)
    pub collateral_accounts: [Pubkey; 5], // Up to 5 collateral accounts
    
    // Timestamps
    pub opened_at: i64,
    pub closed_at: i64,
    pub last_updated: i64,
    pub created_at: i64,            // Position creation timestamp
    
    pub bump: u8,
}

impl Position {
    pub const INIT_SPACE: usize = 32 + 32 + 32 + 2 + 1 + 1 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 8 + 2 + 8 + 8 + (32 * 5) + 8 + 8 + 8 + 8 + 1;
    
    /// Calculate profit/loss for this position
    pub fn calculate_pnl(&self) -> i64 {
        let price_diff = self.current_price as i64 - self.entry_price as i64;
        match self.side {
            PositionSide::Long => (price_diff * self.size as i64) / 1_000_000,
            PositionSide::Short => (-price_diff * self.size as i64) / 1_000_000,
        }
    }
    
    /// Check if position is liquidatable based on current price
    pub fn is_liquidatable(&self) -> bool {
        match self.side {
            PositionSide::Long => self.current_price <= self.liquidation_price,
            PositionSide::Short => self.current_price >= self.liquidation_price,
        }
    }
}
