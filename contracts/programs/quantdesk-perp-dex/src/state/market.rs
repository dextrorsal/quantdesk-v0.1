use anchor_lang::prelude::*;
use crate::ErrorCode;

/// Market state definition
/// Handles market data, pricing, and market-specific operations

#[account]
pub struct Market {
    pub base_asset: String,           // e.g., "BTC"
    pub quote_asset: String,           // e.g., "USDT"
    pub base_reserve: u64,             // vAMM base reserve
    pub quote_reserve: u64,            // vAMM quote reserve
    pub k: u64,                        // vAMM constant product
    pub funding_rate: i64,             // Funding rate in basis points
    pub last_funding_time: i64,        // Last funding settlement time
    pub funding_interval: i64,         // Funding interval in seconds
    pub authority: Pubkey,             // Market authority
    pub max_leverage: u8,              // Maximum allowed leverage
    pub initial_margin_ratio: u16,     // Initial margin ratio in basis points
    pub maintenance_margin_ratio: u16, // Maintenance margin ratio in basis points
    pub is_active: bool,               // Whether market is active
    pub last_oracle_price: u64,        // Last oracle price
    pub last_oracle_update: i64,       // Last oracle update timestamp
    pub bump: u8,                     // PDA bump
}

impl Market {
    pub const INIT_SPACE: usize = 4 + 32 + 4 + 32 + 8 + 8 + 8 + 8 + 8 + 32 + 1 + 2 + 2 + 1 + 8 + 8 + 1;
    
    /// Get current price from oracle with staleness check
    pub fn get_oracle_price(&self) -> Result<u64> {
        // Check if oracle price is recent (within 5 minutes)
        let current_time = Clock::get()?.unix_timestamp;
        require!(
            current_time - self.last_oracle_update <= 300, // 5 minutes
            ErrorCode::PriceStale
        );
        
        Ok(self.last_oracle_price)
    }

    /// Calculate premium index based on market conditions
    pub fn calculate_premium_index(&self) -> Result<i64> {
        // Calculate premium index based on market conditions
        let current_price = self.get_oracle_price()? as i128;
        let oracle_price = self.last_oracle_price as i128;
        
        // Premium index = (mark_price - oracle_price) / oracle_price * 10000
        let premium = ((current_price - oracle_price) * 10000) / oracle_price;
        
        // Clamp premium to reasonable bounds
        Ok(premium.clamp(-10000, 10000) as i64) // ±100%
    }

    /// Calculate funding rate based on premium index
    pub fn calculate_funding_rate(&self, premium_index: i64) -> Result<i64> {
        // Funding rate = premium_index + clamp(interest_rate, -0.05%, +0.05%)
        let interest_rate = 100; // 1% base interest rate in basis points
        let clamped_interest = premium_index.clamp(-500, 500); // Clamp to ±0.05%
        
        Ok(premium_index + clamped_interest + interest_rate)
    }
    
    /// Get current price from oracle (alias for compatibility)
    pub fn get_current_price(&self) -> Result<u64> {
        self.get_oracle_price()
    }
    
    /// Update oracle price
    pub fn update_oracle_price(&mut self, new_price: u64, timestamp: i64) -> Result<()> {
        self.last_oracle_price = new_price;
        self.last_oracle_update = timestamp;
        Ok(())
    }
}
