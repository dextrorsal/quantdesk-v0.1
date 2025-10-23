/// User Account State Module
/// Contains all user account related structures

use anchor_lang::prelude::*;

#[account]
pub struct UserAccount {
    pub authority: Pubkey,        // User's wallet address
    pub account_index: u16,       // Account number (for sub-accounts)
    
    // ENHANCED: Collateral tracking
    pub total_collateral: u64,    // Total collateral deposited
    
    // ENHANCED: Position/order tracking with limits
    pub total_positions: u16,     // Number of open positions
    pub total_orders: u16,        // Number of active orders
    pub max_positions: u16,       // NEW: Position limit (25-50)
    
    // ENHANCED: Margin requirements (Drift-style)
    pub initial_margin_requirement: u64,  // NEW: Initial margin requirement
    pub maintenance_margin_requirement: u64,  // NEW: Maintenance margin requirement
    pub available_margin: u64,    // NEW: Available margin for new positions
    
    // ENHANCED: Risk management
    pub account_health: u16,      // Account health (0-10000, where 10000 = 100%)
    pub liquidation_price: u64,   // Liquidation price threshold
    pub liquidation_threshold: u16,  // NEW: Basis points (e.g., 2000 = 20%)
    pub max_leverage: u16,        // NEW: Max leverage (e.g., 1000 = 10x)
    
    // NEW: Funding rate tracking
    pub total_funding_paid: i64,  // NEW: Can be negative
    pub total_funding_received: i64,  // NEW
    
    // NEW: Fee tracking
    pub total_fees_paid: u64,     // NEW
    pub total_rebates_earned: u64,  // NEW
    
    // Existing timestamps
    pub created_at: i64,          // Account creation timestamp
    pub last_activity: i64,       // Last activity timestamp
    pub is_active: bool,          // Whether account is active
    pub bump: u8,                // PDA bump seed
}

impl UserAccount {
    // authority(32) + account_index(2) + total_collateral(8) + total_positions(2) + total_orders(2) 
    // + max_positions(2) + initial_margin(8) + maintenance_margin(8) + available_margin(8)
    // + account_health(2) + liquidation_price(8) + liquidation_threshold(2) + max_leverage(2)
    // + total_funding_paid(8) + total_funding_received(8) + total_fees_paid(8) + total_rebates_earned(8)
    // + created_at(8) + last_activity(8) + is_active(1) + bump(1)
    pub const INIT_SPACE: usize = 32 + 2 + 8 + 2 + 2 + 2 + 8 + 8 + 8 + 2 + 8 + 2 + 2 + 8 + 8 + 8 + 8 + 8 + 8 + 1 + 1; 
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub enum UserAction {
    CreatePosition,
    ClosePosition,
    AddCollateral,
    RemoveCollateral,
    PlaceOrder,
    CancelOrder,
    UpdateLeverage,
    LiquidatePosition,
    UpdateAccount,
}
