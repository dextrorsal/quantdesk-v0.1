//! Protocol state definitions
//! This module contains protocol-level account structures

use anchor_lang::prelude::*;

// KeeperNetwork moved to state/advanced.rs to eliminate duplication

#[account]
pub struct ProgramState {
    pub authority: Pubkey,
    pub is_paused: bool,
    pub insurance_fund: Pubkey,
    pub fee_collector: Pubkey,
    pub oracle_manager: Pubkey,
    pub bump: u8,
}

impl ProgramState {
    pub const INIT_SPACE: usize = 32 + 1 + 32 + 32 + 32 + 1;
}

#[account]
pub struct InsuranceFund {
    pub total_funds: u64,                      // Total insurance funds
    pub utilized_funds: u64,                    // Currently utilized funds
    pub max_utilization: u16,                  // Max utilization percentage
    pub bump: u8,                              // PDA bump
}

impl InsuranceFund {
    pub const INIT_SPACE: usize = 8 + 8 + 8 + 2 + 1;
}

#[account]
pub struct FeeCollector {
    pub maker_fee_rate: u16,                   // Maker fee rate in basis points
    pub taker_fee_rate: u16,                   // Taker fee rate in basis points
    pub funding_rate_cap: i64,                 // Funding rate cap
    pub funding_rate_floor: i64,               // Funding rate floor
    pub trading_fees_collected: u64,           // Total trading fees collected
    pub funding_fees_collected: u64,           // Total funding fees collected
    pub bump: u8,                              // PDA bump
}

impl FeeCollector {
    pub const INIT_SPACE: usize = 2 + 2 + 8 + 8 + 8 + 8 + 1;
}