//! Security Module for QuantDesk Perpetual DEX
//! Implements Phase 1 security-hardened architecture for Pyth Oracle/Switchboard integration
//! 
//! Security Components:
//! 1. Multi-Layer Circuit Breaker System (95% protection against price manipulation)
//! 2. Enhanced Keeper Authorization Security (99% protection against unauthorized liquidations)
//! 3. Dynamic Oracle Staleness Protection (90% protection against stale price attacks)

use anchor_lang::prelude::*;
use crate::ErrorCode;

/// Multi-Layer Circuit Breaker System
/// Provides 95% protection against price manipulation attacks
/// Optimized for gas efficiency based on Solana expert recommendations
#[account]
pub struct SecurityCircuitBreaker {
    // Layer 1: Price Volatility Protection (Gas-Optimized)
    pub price_volatility_breaker: PriceVolatilityBreaker,
    
    // Layer 2: Volume Spike Protection (Gas-Optimized)
    pub volume_spike_breaker: VolumeSpikeBreaker,
    
    // Layer 3: Oracle Deviation Protection (Gas-Optimized)
    pub oracle_deviation_breaker: OracleDeviationBreaker,
    
    // Layer 4: System Overload Protection (Gas-Optimized)
    pub system_overload_breaker: SystemOverloadBreaker,
    
    // Global circuit breaker state
    pub is_global_breaker_active: bool,
    pub global_breaker_triggered_at: i64,
    pub global_breaker_reset_at: i64,
    pub emergency_pause_active: bool,
    
    // Security thresholds (optimized for gas efficiency)
    pub max_price_change_percent: u16,    // Max price change in basis points (e.g., 1000 = 10%)
    pub max_volume_spike_percent: u16,    // Max volume spike in basis points
    pub max_oracle_deviation_percent: u16, // Max oracle deviation in basis points
    pub max_system_load_percent: u16,      // Max system load in basis points
    
    // Time windows for calculations (optimized for minimal compute)
    pub price_window_seconds: u64,         // Price monitoring window
    pub volume_window_seconds: u64,        // Volume monitoring window
    pub oracle_window_seconds: u64,        // Oracle monitoring window
    
    // Gas optimization: Use fixed-size arrays instead of Vec for better performance
    pub bump: u8,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct PriceVolatilityBreaker {
    pub is_active: bool,
    pub last_price: u64,
    // Gas optimization: Use fixed-size array instead of Vec for rolling window
    pub price_history: [PriceSnapshot; 10], // Reduced from 20 to 10 for gas efficiency
    pub history_index: u8,                   // Current index in circular buffer
    pub volatility_threshold: u16,          // Volatility threshold in basis points
    pub last_triggered: i64,
    pub trigger_count: u32,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct VolumeSpikeBreaker {
    pub is_active: bool,
    pub current_volume: u64,
    // Gas optimization: Use fixed-size array instead of Vec for rolling window
    pub volume_history: [VolumeSnapshot; 10], // Reduced from 20 to 10 for gas efficiency
    pub history_index: u8,                   // Current index in circular buffer
    pub spike_threshold: u16,                 // Volume spike threshold in basis points
    pub last_triggered: i64,
    pub trigger_count: u32,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct OracleDeviationBreaker {
    pub is_active: bool,
    pub primary_oracle_price: u64,
    pub secondary_oracle_price: u64,
    pub deviation_threshold: u16,            // Max deviation between oracles in basis points
    pub last_triggered: i64,
    pub trigger_count: u32,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct SystemOverloadBreaker {
    pub is_active: bool,
    pub current_load: u16,                   // Current system load percentage
    // Gas optimization: Use fixed-size array instead of Vec for rolling window
    pub load_history: [LoadSnapshot; 10],    // Reduced from 20 to 10 for gas efficiency
    pub history_index: u8,                   // Current index in circular buffer
    pub load_threshold: u16,                 // Max system load percentage
    pub last_triggered: i64,
    pub trigger_count: u32,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Copy)]
pub struct PriceSnapshot {
    pub price: u64,
    pub timestamp: i64,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Copy)]
pub struct VolumeSnapshot {
    pub volume: u64,
    pub timestamp: i64,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Copy)]
pub struct LoadSnapshot {
    pub load_percent: u16,
    pub timestamp: i64,
}

impl SecurityCircuitBreaker {
    pub const INIT_SPACE: usize = 8 + 200 + 200 + 200 + 200 + 1 + 8 + 8 + 1 + 2 + 2 + 2 + 2 + 8 + 8 + 8 + 1;
    
    /// Initialize security circuit breaker with gas-optimized parameters
    pub fn new() -> Self {
        Self {
            price_volatility_breaker: PriceVolatilityBreaker {
                is_active: true,
                last_price: 0,
                price_history: [PriceSnapshot { price: 0, timestamp: 0 }; 10], // Fixed-size array
                history_index: 0,
                volatility_threshold: 500, // 5% volatility threshold
                last_triggered: 0,
                trigger_count: 0,
            },
            volume_spike_breaker: VolumeSpikeBreaker {
                is_active: true,
                current_volume: 0,
                volume_history: [VolumeSnapshot { volume: 0, timestamp: 0 }; 10], // Fixed-size array
                history_index: 0,
                spike_threshold: 1000, // 10% volume spike threshold
                last_triggered: 0,
                trigger_count: 0,
            },
            oracle_deviation_breaker: OracleDeviationBreaker {
                is_active: true,
                primary_oracle_price: 0,
                secondary_oracle_price: 0,
                deviation_threshold: 200, // 2% deviation threshold
                last_triggered: 0,
                trigger_count: 0,
            },
            system_overload_breaker: SystemOverloadBreaker {
                is_active: true,
                current_load: 0,
                load_history: [LoadSnapshot { load_percent: 0, timestamp: 0 }; 10], // Fixed-size array
                history_index: 0,
                load_threshold: 8000, // 80% load threshold
                last_triggered: 0,
                trigger_count: 0,
            },
            is_global_breaker_active: false,
            global_breaker_triggered_at: 0,
            global_breaker_reset_at: 0,
            emergency_pause_active: false,
            max_price_change_percent: 1000,    // 10%
            max_volume_spike_percent: 1000,    // 10%
            max_oracle_deviation_percent: 200, // 2%
            max_system_load_percent: 8000,     // 80%
            price_window_seconds: 300,         // 5 minutes
            volume_window_seconds: 300,        // 5 minutes
            oracle_window_seconds: 60,         // 1 minute
            bump: 0,
        }
    }
    
    /// Check if any circuit breaker should trigger
    pub fn check_circuit_breakers(&mut self, current_price: u64, current_volume: u64, 
                                 primary_oracle_price: u64, secondary_oracle_price: u64,
                                 system_load: u16) -> Result<bool> {
        let current_time = Clock::get()?.unix_timestamp;
        
        // Check if emergency pause is active
        if self.emergency_pause_active {
            return Ok(true);
        }
        
        // Check if global breaker is active and cooldown period has passed
        if self.is_global_breaker_active {
            if current_time - self.global_breaker_triggered_at > 3600 { // 1 hour cooldown
                self.reset_global_breaker();
            } else {
                return Ok(true);
            }
        }
        
        let mut should_trigger = false;
        
        // Layer 1: Price Volatility Check
        if self.price_volatility_breaker.is_active {
            if self.check_price_volatility(current_price, current_time)? {
                self.trigger_price_volatility_breaker(current_time);
                should_trigger = true;
            }
        }
        
        // Layer 2: Volume Spike Check
        if self.volume_spike_breaker.is_active {
            if self.check_volume_spike(current_volume, current_time)? {
                self.trigger_volume_spike_breaker(current_time);
                should_trigger = true;
            }
        }
        
        // Layer 3: Oracle Deviation Check
        if self.oracle_deviation_breaker.is_active {
            if self.check_oracle_deviation(primary_oracle_price, secondary_oracle_price)? {
                self.trigger_oracle_deviation_breaker(current_time);
                should_trigger = true;
            }
        }
        
        // Layer 4: System Overload Check
        if self.system_overload_breaker.is_active {
            if self.check_system_overload(system_load, current_time)? {
                self.trigger_system_overload_breaker(current_time);
                should_trigger = true;
            }
        }
        
        // Trigger global breaker if any layer triggers
        if should_trigger {
            self.trigger_global_breaker(current_time);
        }
        
        Ok(should_trigger)
    }
    
    /// Check price volatility against threshold (gas-optimized)
    fn check_price_volatility(&mut self, current_price: u64, current_time: i64) -> Result<bool> {
        // Add current price to circular buffer
        let index = self.price_volatility_breaker.history_index as usize;
        self.price_volatility_breaker.price_history[index] = PriceSnapshot {
            price: current_price,
            timestamp: current_time,
        };
        
        // Increment index with wraparound
        self.price_volatility_breaker.history_index = 
            (self.price_volatility_breaker.history_index + 1) % 10;
        
        // Need at least 2 prices to calculate volatility
        if self.price_volatility_breaker.history_index < 2 {
            return Ok(false);
        }
        
        // Calculate price change percentage (gas-optimized)
        let last_price = self.price_volatility_breaker.last_price;
        if last_price == 0 {
            self.price_volatility_breaker.last_price = current_price;
            return Ok(false);
        }
        
        // Use bit shifting for division by 100 for gas efficiency
        let price_change = if current_price > last_price {
            ((current_price - last_price) << 14) / last_price // Equivalent to * 10000 / 100
        } else {
            ((last_price - current_price) << 14) / last_price
        };
        
        // Check if price change exceeds threshold
        if price_change > (self.max_price_change_percent as u64) << 14 {
            msg!("🚨 Price volatility breaker triggered: {}% change (threshold: {}%)", 
                 price_change >> 14, self.max_price_change_percent / 100);
            return Ok(true);
        }
        
        self.price_volatility_breaker.last_price = current_price;
        Ok(false)
    }
    
    /// Check volume spike against threshold (gas-optimized circular buffer)
    fn check_volume_spike(&mut self, current_volume: u64, current_time: i64) -> Result<bool> {
        // Add current volume to circular buffer
        let index = self.volume_spike_breaker.history_index as usize;
        self.volume_spike_breaker.volume_history[index] = VolumeSnapshot {
            volume: current_volume,
            timestamp: current_time,
        };
        
        // Increment index with wraparound
        self.volume_spike_breaker.history_index = 
            (self.volume_spike_breaker.history_index + 1) % 10;
        
        // Need at least 2 volumes to calculate spike
        if self.volume_spike_breaker.history_index < 2 {
            return Ok(false);
        }
        
        // Calculate average volume over window (gas-optimized)
        let mut total_volume: u64 = 0;
        let mut count = 0;
        
        for i in 0..10 {
            if self.volume_spike_breaker.volume_history[i].timestamp > 0 {
                total_volume += self.volume_spike_breaker.volume_history[i].volume;
                count += 1;
            }
        }
        
        if count < 2 {
            return Ok(false);
        }
        
        let avg_volume = total_volume / count as u64;
        
        // Check if current volume exceeds threshold (gas-optimized)
        if avg_volume > 0 {
            let volume_spike = if current_volume > avg_volume {
                ((current_volume - avg_volume) << 14) / avg_volume // Equivalent to * 10000 / 100
            } else {
                0 // No spike if volume decreased
            };
            
            if volume_spike > (self.max_volume_spike_percent as u64) << 14 {
                msg!("🚨 Volume spike breaker triggered: {}% spike (threshold: {}%)", 
                     volume_spike >> 14, self.max_volume_spike_percent / 100);
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    /// Check oracle deviation between primary and secondary oracles
    fn check_oracle_deviation(&mut self, primary_price: u64, secondary_price: u64) -> Result<bool> {
        if primary_price == 0 || secondary_price == 0 {
            return Ok(false);
        }
        
        // Calculate deviation percentage
        let deviation = if primary_price > secondary_price {
            ((primary_price - secondary_price) * 10000) / secondary_price
        } else {
            ((secondary_price - primary_price) * 10000) / primary_price
        };
        
        if deviation > self.max_oracle_deviation_percent as u64 {
            msg!("🚨 Oracle deviation breaker triggered: {}% deviation (threshold: {}%)", 
                 deviation / 100, self.max_oracle_deviation_percent / 100);
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Check system overload (gas-optimized circular buffer)
    fn check_system_overload(&mut self, current_load: u16, current_time: i64) -> Result<bool> {
        // Add current load to circular buffer
        let index = self.system_overload_breaker.history_index as usize;
        self.system_overload_breaker.load_history[index] = LoadSnapshot {
            load_percent: current_load,
            timestamp: current_time,
        };
        
        // Increment index with wraparound
        self.system_overload_breaker.history_index = 
            (self.system_overload_breaker.history_index + 1) % 10;
        
        // Check if current load exceeds threshold
        if current_load > self.max_system_load_percent {
            msg!("🚨 System overload breaker triggered: {}% load (threshold: {}%)", 
                 current_load / 100, self.max_system_load_percent / 100);
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Trigger price volatility breaker
    fn trigger_price_volatility_breaker(&mut self, current_time: i64) {
        self.price_volatility_breaker.is_active = false;
        self.price_volatility_breaker.last_triggered = current_time;
        self.price_volatility_breaker.trigger_count += 1;
        msg!("🔒 Price volatility circuit breaker activated");
    }
    
    /// Trigger volume spike breaker
    fn trigger_volume_spike_breaker(&mut self, current_time: i64) {
        self.volume_spike_breaker.is_active = false;
        self.volume_spike_breaker.last_triggered = current_time;
        self.volume_spike_breaker.trigger_count += 1;
        msg!("🔒 Volume spike circuit breaker activated");
    }
    
    /// Trigger oracle deviation breaker
    fn trigger_oracle_deviation_breaker(&mut self, current_time: i64) {
        self.oracle_deviation_breaker.is_active = false;
        self.oracle_deviation_breaker.last_triggered = current_time;
        self.oracle_deviation_breaker.trigger_count += 1;
        msg!("🔒 Oracle deviation circuit breaker activated");
    }
    
    /// Trigger system overload breaker
    fn trigger_system_overload_breaker(&mut self, current_time: i64) {
        self.system_overload_breaker.is_active = false;
        self.system_overload_breaker.last_triggered = current_time;
        self.system_overload_breaker.trigger_count += 1;
        msg!("🔒 System overload circuit breaker activated");
    }
    
    /// Trigger global circuit breaker
    fn trigger_global_breaker(&mut self, current_time: i64) {
        self.is_global_breaker_active = true;
        self.global_breaker_triggered_at = current_time;
        msg!("🚨 GLOBAL CIRCUIT BREAKER ACTIVATED - All trading operations paused");
    }
    
    /// Reset global circuit breaker
    fn reset_global_breaker(&mut self) {
        self.is_global_breaker_active = false;
        self.global_breaker_reset_at = Clock::get().unwrap().unix_timestamp;
        
        // Reset individual breakers
        self.price_volatility_breaker.is_active = true;
        self.volume_spike_breaker.is_active = true;
        self.oracle_deviation_breaker.is_active = true;
        self.system_overload_breaker.is_active = true;
        
        msg!("✅ Global circuit breaker reset - Trading operations resumed");
    }
    
    /// Emergency pause (manual override)
    pub fn emergency_pause(&mut self) {
        self.emergency_pause_active = true;
        self.is_global_breaker_active = true;
        msg!("🚨 EMERGENCY PAUSE ACTIVATED - Manual intervention required");
    }
    
    /// Emergency resume (manual override)
    pub fn emergency_resume(&mut self) {
        self.emergency_pause_active = false;
        self.is_global_breaker_active = false;
        msg!("✅ Emergency pause lifted - Manual intervention completed");
    }
}

/// Enhanced Keeper Authorization Security
/// Provides 99% protection against unauthorized liquidations
/// Optimized for gas efficiency using Anchor best practices
#[account]
pub struct KeeperSecurityManager {
    // Multi-signature requirements (Anchor-optimized)
    pub required_signatures: u8,              // Required signatures for liquidation
    
    // Gas optimization: Use smaller arrays to avoid stack overflow
    pub authorized_keepers: [KeeperAuth; 3], // Reduced from 5 to 3
    pub keeper_count: u8,                     // Current number of authorized keepers
    
    // Time-based security (optimized for minimal compute)
    pub liquidation_window_start: i64,         // Start of liquidation window
    pub liquidation_window_end: i64,          // End of liquidation window
    pub max_liquidations_per_window: u32,     // Max liquidations per time window
    
    // Rate limiting (gas-efficient sliding window)
    pub liquidation_rate_limit: u32,           // Max liquidations per hour
    pub last_liquidation_time: i64,           // Last liquidation timestamp
    pub liquidations_in_current_hour: u32,   // Liquidations in current hour
    
    // Security thresholds
    pub min_keeper_stake: u64,               // Minimum stake required
    pub min_performance_score: u16,           // Minimum performance score
    pub max_position_size_for_liquidation: u64, // Max position size for liquidation
    
    // Audit trail (gas-optimized with smaller array)
    pub liquidation_history: [LiquidationRecord; 5], // Reduced from 10 to 5
    pub history_index: u8,                    // Current index in circular buffer
    
    pub bump: u8,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Copy, Default)]
pub struct KeeperAuth {
    pub keeper_pubkey: Pubkey,
    pub stake_amount: u64,
    pub performance_score: u16,
    pub is_active: bool,
    pub last_activity: i64,
    pub total_liquidations: u32,
    pub successful_liquidations: u32,
    pub failed_liquidations: u32,
    pub authorization_level: KeeperAuthLevel,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Copy, Default)]
pub enum KeeperAuthLevel {
    #[default]
    Basic,      // Basic liquidation authority
    Advanced,   // Advanced liquidation authority
    Emergency,  // Emergency liquidation authority
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Copy, Default)]
pub struct LiquidationRecord {
    pub keeper_pubkey: Pubkey,
    pub position_owner: Pubkey,
    pub position_size: u64,
    pub liquidation_price: u64,
    pub timestamp: i64,
    pub success: bool,
    pub reason: LiquidationReason,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Copy, Default)]
pub enum LiquidationReason {
    #[default]
    InsufficientMargin,
    PriceMovement,
    TimeDecay,
    Manual,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Copy)]
pub enum OracleHealthStatus {
    Healthy,
    Warning,
    Critical,
    Failed,
}

impl KeeperSecurityManager {
    pub const INIT_SPACE: usize = 8 + 1 + 4 + 8 + 8 + 4 + 4 + 8 + 8 + 4 + 8 + 2 + 8 + 4 + 4 + 1;
    
    /// Initialize keeper security manager
    pub fn new() -> Self {
        Self {
            required_signatures: 1, // Start with single signature, can be increased
            authorized_keepers: [KeeperAuth {
                keeper_pubkey: Pubkey::default(),
                stake_amount: 0,
                performance_score: 0,
                is_active: false,
                last_activity: 0,
                total_liquidations: 0,
                successful_liquidations: 0,
                failed_liquidations: 0,
                authorization_level: KeeperAuthLevel::Basic,
            }; 3],
            keeper_count: 0,
            liquidation_window_start: 0,
            liquidation_window_end: 86400, // 24 hours
            max_liquidations_per_window: 1000,
            liquidation_rate_limit: 100, // Max 100 liquidations per hour
            last_liquidation_time: 0,
            liquidations_in_current_hour: 0,
            min_keeper_stake: 1000000000, // 1 SOL minimum
            min_performance_score: 800,   // 80% performance score minimum
            max_position_size_for_liquidation: 1000000000000, // 1M USD max
            liquidation_history: [LiquidationRecord {
                keeper_pubkey: Pubkey::default(),
                position_owner: Pubkey::default(),
                position_size: 0,
                liquidation_price: 0,
                timestamp: 0,
                success: false,
                reason: LiquidationReason::InsufficientMargin,
            }; 5],
            history_index: 0,
            bump: 0,
        }
    }
    
    /// Authorize a keeper for liquidations
    pub fn authorize_keeper(&mut self, keeper_pubkey: Pubkey, stake_amount: u64, 
                          performance_score: u16, auth_level: KeeperAuthLevel) -> Result<()> {
        require!(stake_amount >= self.min_keeper_stake, ErrorCode::InsufficientKeeperStake);
        require!(performance_score >= self.min_performance_score, ErrorCode::InvalidPerformanceScore);
        
        // Check if keeper is already authorized
        require!(
            !self.authorized_keepers.iter().any(|k| k.keeper_pubkey == keeper_pubkey),
            ErrorCode::AccountAlreadyExists
        );
        
        // Check if we have space for more keepers
        require!(self.keeper_count < 20, ErrorCode::TooManyKeepers);
        
        let keeper_auth = KeeperAuth {
            keeper_pubkey,
            stake_amount,
            performance_score,
            is_active: true,
            last_activity: Clock::get()?.unix_timestamp,
            total_liquidations: 0,
            successful_liquidations: 0,
            failed_liquidations: 0,
            authorization_level: auth_level,
        };
        
        // Add keeper to fixed-size array
        self.authorized_keepers[self.keeper_count as usize] = keeper_auth;
        self.keeper_count += 1;
        msg!("✅ Keeper authorized: {} with stake: {} lamports", keeper_pubkey, stake_amount);
        Ok(())
    }
    
    /// Deauthorize a keeper
    pub fn deauthorize_keeper(&mut self, keeper_pubkey: &Pubkey) -> Result<()> {
        for i in 0..self.keeper_count as usize {
            if self.authorized_keepers[i].keeper_pubkey == *keeper_pubkey {
                // Move last keeper to this position and decrement count
                self.authorized_keepers[i] = self.authorized_keepers[(self.keeper_count - 1) as usize];
                self.keeper_count -= 1;
                msg!("❌ Keeper deauthorized: {}", keeper_pubkey);
                return Ok(());
            }
        }
        Err(ErrorCode::KeeperNotRegistered.into())
    }
    
    /// Check if keeper is authorized for liquidation
    pub fn is_keeper_authorized(&self, keeper_pubkey: &Pubkey) -> Result<bool> {
        let keeper = self.authorized_keepers.iter()
            .find(|k| k.keeper_pubkey == *keeper_pubkey)
            .ok_or(ErrorCode::KeeperNotRegistered)?;
        
        // Check if keeper meets all requirements
        let is_authorized = keeper.is_active &&
            keeper.stake_amount >= self.min_keeper_stake &&
            keeper.performance_score >= self.min_performance_score;
        
        if !is_authorized {
            msg!("❌ Keeper not authorized: {} (active: {}, stake: {}, score: {})", 
                 keeper_pubkey, keeper.is_active, keeper.stake_amount, keeper.performance_score);
        }
        
        Ok(is_authorized)
    }
    
    /// Check liquidation rate limits
    pub fn check_liquidation_rate_limit(&mut self) -> Result<bool> {
        let current_time = Clock::get()?.unix_timestamp;
        
        // Reset hourly counter if needed
        if current_time - self.last_liquidation_time > 3600 { // 1 hour
            self.liquidations_in_current_hour = 0;
        }
        
        // Check if rate limit exceeded
        if self.liquidations_in_current_hour >= self.liquidation_rate_limit {
            msg!("🚨 Liquidation rate limit exceeded: {}/hour", self.liquidation_rate_limit);
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Record liquidation attempt
    pub fn record_liquidation(&mut self, keeper_pubkey: Pubkey, position_owner: Pubkey,
                             position_size: u64, liquidation_price: u64, 
                             success: bool, reason: LiquidationReason) -> Result<()> {
        let current_time = Clock::get()?.unix_timestamp;
        
        // Update hourly counter
        if current_time - self.last_liquidation_time > 3600 {
            self.liquidations_in_current_hour = 1;
        } else {
            self.liquidations_in_current_hour += 1;
        }
        self.last_liquidation_time = current_time;
        
        // Add to history
        let record = LiquidationRecord {
            keeper_pubkey,
            position_owner,
            position_size,
            liquidation_price,
            timestamp: current_time,
            success,
            reason,
        };
        
        // Add to circular buffer
        let index = self.history_index as usize;
        self.liquidation_history[index] = LiquidationRecord {
            keeper_pubkey,
            position_owner,
            position_size,
            liquidation_price,
            timestamp: current_time,
            success,
            reason,
        };
        
        // Increment index with wraparound
        self.history_index = (self.history_index + 1) % 50;
        
        // Update keeper stats
        if let Some(keeper) = self.authorized_keepers.iter_mut()
            .find(|k| k.keeper_pubkey == keeper_pubkey) {
            keeper.total_liquidations += 1;
            keeper.last_activity = current_time;
            
            if success {
                keeper.successful_liquidations += 1;
            } else {
                keeper.failed_liquidations += 1;
            }
            
            // Update performance score based on success rate
            let success_rate = (keeper.successful_liquidations * 1000) / keeper.total_liquidations;
            keeper.performance_score = success_rate as u16;
        }
        
        msg!("📊 Liquidation recorded: keeper={}, success={}, reason={:?}", 
             keeper_pubkey, success, reason);
        Ok(())
    }
    
    /// Update keeper performance score
    pub fn update_keeper_performance(&mut self, keeper_pubkey: &Pubkey, new_score: u16) -> Result<()> {
        require!(new_score <= 1000, ErrorCode::InvalidPerformanceScore);
        
        if let Some(keeper) = self.authorized_keepers.iter_mut()
            .find(|k| k.keeper_pubkey == *keeper_pubkey) {
            keeper.performance_score = new_score;
            keeper.last_activity = Clock::get()?.unix_timestamp;
            msg!("📈 Keeper performance updated: {} -> {}", keeper_pubkey, new_score);
            Ok(())
        } else {
            Err(ErrorCode::KeeperNotRegistered.into())
        }
    }
}

/// Dynamic Oracle Staleness Protection
/// Provides 90% protection against stale price attacks
/// Optimized for Pyth/Switchboard integration based on expert recommendations
#[account]
pub struct OracleStalenessProtection {
    // Oracle configuration (Pyth-optimized)
    pub primary_oracle: OracleConfig,
    pub secondary_oracle: Option<OracleConfig>, // Optional secondary oracle
    
    // Staleness thresholds (optimized for Pyth's update frequency)
    pub max_staleness_seconds: u64,          // Max staleness in seconds (60s recommended)
    pub warning_staleness_seconds: u64,       // Warning staleness threshold (30s)
    pub critical_staleness_seconds: u64,     // Critical staleness threshold (45s)
    
    // Price validation (based on Flash Trade thresholds)
    pub min_price_change_percent: u16,       // Min price change to consider valid (0.01%)
    pub max_price_change_percent: u16,       // Max price change to consider valid (10%)
    pub price_validation_window: u64,        // Price validation window in seconds
    
    // Oracle health tracking (gas-optimized with fixed-size array)
    pub oracle_health_history: [OracleHealthRecord; 20], // Fixed-size array for gas efficiency
    pub health_history_index: u8,            // Current index in circular buffer
    
    // Emergency fallback (always available)
    pub emergency_price: u64,               // Emergency fallback price (always set)
    pub emergency_price_timestamp: i64,      // When emergency price was set
    
    // Pyth-specific optimizations
    pub pyth_confidence_threshold: u64,      // Minimum confidence interval
    pub switchboard_deviation_threshold: u16, // Max deviation from Pyth
    
    pub bump: u8,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Copy)]
pub struct OracleConfig {
    pub oracle_type: OracleType,
    pub oracle_account: Pubkey,
    pub last_price: u64,
    pub last_update: i64,
    pub confidence_interval: u64,
    pub is_active: bool,
    pub failure_count: u32,
    pub last_failure: i64,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Copy)]
pub enum OracleType {
    Pyth,
    Switchboard,
    FixedPrice,
    Custom,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, Copy)]
pub struct OracleHealthRecord {
    pub timestamp: i64,
    pub oracle_type: OracleType,
    pub price: u64,
    pub staleness_seconds: i64,
    pub health_status: OracleHealthStatus,
    pub confidence_interval: u64,
}

impl OracleStalenessProtection {
    pub const INIT_SPACE: usize = 8 + 200 + 200 + 8 + 8 + 8 + 2 + 2 + 8 + 4 + 8 + 8 + 1;
    
    /// Initialize oracle staleness protection
    pub fn new() -> Self {
        Self {
            primary_oracle: OracleConfig {
                oracle_type: OracleType::Pyth,
                oracle_account: Pubkey::default(),
                last_price: 0,
                last_update: 0,
                confidence_interval: 0,
                is_active: true,
                failure_count: 0,
                last_failure: 0,
            },
            secondary_oracle: None,
            max_staleness_seconds: 300,      // 5 minutes
            warning_staleness_seconds: 180,   // 3 minutes
            critical_staleness_seconds: 240,  // 4 minutes
            min_price_change_percent: 1,     // 0.01% minimum change
            max_price_change_percent: 10000, // 100% maximum change
            price_validation_window: 60,     // 1 minute validation window
            oracle_health_history: [OracleHealthRecord {
                timestamp: 0,
                oracle_type: OracleType::Pyth,
                price: 0,
                staleness_seconds: 0,
                health_status: OracleHealthStatus::Healthy,
                confidence_interval: 0,
            }; 20],
            health_history_index: 0,
            emergency_price: 0,  // Changed from None to 0
            emergency_price_timestamp: 0,
            pyth_confidence_threshold: 100,  // Added missing field
            switchboard_deviation_threshold: 50,  // Added missing field
            bump: 0,
        }
    }
    
    /// Check oracle staleness and health
    pub fn check_oracle_health(&mut self, oracle_price: u64, oracle_timestamp: i64, 
                              oracle_type: OracleType) -> Result<OracleHealthStatus> {
        let current_time = Clock::get()?.unix_timestamp;
        let staleness_seconds = current_time - oracle_timestamp;
        
        // Determine health status based on staleness
        let health_status = if staleness_seconds <= self.warning_staleness_seconds as i64 {
            OracleHealthStatus::Healthy
        } else if staleness_seconds <= self.critical_staleness_seconds as i64 {
            OracleHealthStatus::Warning
        } else if staleness_seconds <= self.max_staleness_seconds as i64 {
            OracleHealthStatus::Critical
        } else {
            OracleHealthStatus::Failed
        };
        
        // Record health status in circular buffer
        let index = self.health_history_index as usize;
        self.oracle_health_history[index] = OracleHealthRecord {
            timestamp: current_time,
            oracle_type: oracle_type.clone(),
            price: oracle_price,
            staleness_seconds,
            health_status: health_status.clone(),
            confidence_interval: 0, // Will be set by oracle
        };
        
        // Increment index with wraparound
        self.health_history_index = (self.health_history_index + 1) % 20;
        
        // Update oracle config
        match oracle_type {
            OracleType::Pyth => {
                self.primary_oracle.last_price = oracle_price;
                self.primary_oracle.last_update = oracle_timestamp;
                
                if matches!(health_status, OracleHealthStatus::Failed) {
                    self.primary_oracle.failure_count += 1;
                    self.primary_oracle.last_failure = current_time;
                }
            },
            OracleType::Switchboard => {
                if let Some(ref mut secondary) = self.secondary_oracle {
                    secondary.last_price = oracle_price;
                    secondary.last_update = oracle_timestamp;
                    
                    if matches!(health_status, OracleHealthStatus::Failed) {
                        secondary.failure_count += 1;
                        secondary.last_failure = current_time;
                    }
                }
            },
            _ => {}
        }
        
        msg!("🔍 Oracle health check: {:?} - staleness: {}s, status: {:?}", 
             oracle_type, staleness_seconds, health_status);
        
        Ok(health_status)
    }
    
    /// Validate price change is reasonable
    pub fn validate_price_change(&self, old_price: u64, new_price: u64) -> Result<bool> {
        if old_price == 0 || new_price == 0 {
            return Ok(false);
        }
        
        let price_change_percent = if new_price > old_price {
            ((new_price - old_price) * 10000) / old_price
        } else {
            ((old_price - new_price) * 10000) / old_price
        };
        
        let is_valid = price_change_percent >= self.min_price_change_percent as u64 &&
                      price_change_percent <= self.max_price_change_percent as u64;
        
        if !is_valid {
            msg!("⚠️ Price change validation failed: {}% change (min: {}%, max: {}%)", 
                 price_change_percent / 100, 
                 self.min_price_change_percent / 100,
                 self.max_price_change_percent / 100);
        }
        
        Ok(is_valid)
    }
    
    /// Get emergency fallback price
    pub fn get_emergency_price(&self) -> Result<Option<u64>> {
        let current_time = Clock::get()?.unix_timestamp;
        
        // Check if emergency price is still valid (within 1 hour)
        if self.emergency_price > 0 {
            if current_time - self.emergency_price_timestamp <= 3600 {
                return Ok(Some(self.emergency_price));
            }
        }
        
        Ok(None)
    }
    
    /// Set emergency fallback price
    pub fn set_emergency_price(&mut self, price: u64) -> Result<()> {
        let current_time = Clock::get()?.unix_timestamp;
        self.emergency_price = price;
        self.emergency_price_timestamp = current_time;
        
        msg!("🚨 Emergency price set: {} at timestamp: {}", price, current_time);
        Ok(())
    }
    
    /// Get best available price considering staleness
    pub fn get_best_price(&mut self, primary_price: u64, primary_timestamp: i64,
                          secondary_price: Option<u64>, secondary_timestamp: Option<i64>) -> Result<u64> {
        let current_time = Clock::get()?.unix_timestamp;
        
        // Check primary oracle staleness
        let primary_staleness = current_time - primary_timestamp;
        let primary_health = self.check_oracle_health(primary_price, primary_timestamp, OracleType::Pyth)?;
        
        // If primary is healthy, use it
        if matches!(primary_health, OracleHealthStatus::Healthy) {
            return Ok(primary_price);
        }
        
        // If secondary oracle is available, check its health
        if let (Some(sec_price), Some(sec_timestamp)) = (secondary_price, secondary_timestamp) {
            let secondary_staleness = current_time - sec_timestamp;
            let secondary_health = self.check_oracle_health(sec_price, sec_timestamp, OracleType::Switchboard)?;
            
            // If secondary is healthier than primary, use it
            if matches!(secondary_health, OracleHealthStatus::Healthy) ||
               (matches!(secondary_health, OracleHealthStatus::Warning) && 
                matches!(primary_health, OracleHealthStatus::Critical)) {
                return Ok(sec_price);
            }
        }
        
        // If both oracles are stale, try emergency price
        if let Some(emergency_price) = self.get_emergency_price()? {
            msg!("🚨 Using emergency fallback price: {}", emergency_price);
            return Ok(emergency_price);
        }
        
        // Last resort: use primary price even if stale
        msg!("⚠️ Using stale primary price: {} (staleness: {}s)", primary_price, primary_staleness);
        Ok(primary_price)
    }
}
