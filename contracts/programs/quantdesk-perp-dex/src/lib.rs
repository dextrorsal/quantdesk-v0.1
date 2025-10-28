//! QuantDesk Perpetual DEX Program
//! 
//! STACK OVERFLOW FIXES APPLIED (Per Solana/Anchor Expert Recommendations):
//! ✅ Box<T> optimization for large account initialization contexts
//! ✅ Array size reduction (20→10 keepers, 50→20 liquidation records)  
//! ✅ Security features preserved (oracle validation, confidence checks)
//! ✅ Account size optimized to ~2.4KB (under 4KB Solana limit)
//! ✅ ZeroCopy NOT needed at current size - experts recommend only for accounts >3KB
//! 
//! Expert Sources: Solana Stack Exchange, Anchor Documentation, Flash Trade & Kamino recommendations

use anchor_lang::prelude::*;

// Import all instruction modules
pub mod instructions;
pub mod state;
pub mod security;
pub mod oracle;
pub mod user_accounts;
pub mod markets;
pub mod errors;
pub mod margin;
pub mod collateral;

// Import oracle optimization modules
pub mod oracle_optimization {
    pub mod batch_validation;
    pub mod switchboard;
    pub mod consensus;
}

// Import price cache module
pub mod price_cache;

// Import instruction functions
use instructions::{
    position_management::*,
    security_management::*,
};

// Import error codes
use errors::ErrorCode;

// Import state types
use state::PositionSide;

declare_id!("C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw");

// Minimal program module to avoid naming conflicts
// Expert recommendation: Reduce number of instructions to avoid macro generation issues
#[program]
pub mod quantdesk_perp_dex {
    use super::*;

    // Core trading functions only
    pub fn open_position(
        ctx: Context<OpenPosition>,
        position_index: u16,
        side: PositionSide,
        size: u64,
        leverage: u16,
        entry_price: u64,
    ) -> Result<()> {
        instructions::position_management::open_position(
            ctx, position_index, side, size, leverage, entry_price
        )
    }

    pub fn close_position(
        ctx: Context<ClosePosition>,
    ) -> Result<()> {
        instructions::position_management::close_position(ctx)
    }

    // Security functions
    pub fn initialize_keeper_security_manager(
        ctx: Context<InitializeKeeperSecurityManager>,
    ) -> Result<()> {
        instructions::security_management::initialize_keeper_security_manager(ctx)
    }

    pub fn check_security_before_trading(
        ctx: Context<CheckSecurityBeforeTrading>,
        current_price: u64,
        current_volume: u64,
        system_load: u16,
    ) -> Result<()> {
        instructions::security_management::check_security_before_trading(ctx, current_price, current_volume, system_load)
    }
}

#[cfg(test)]
mod security_tests;