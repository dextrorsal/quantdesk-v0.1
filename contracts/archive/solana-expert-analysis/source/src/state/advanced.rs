use anchor_lang::prelude::*;
use crate::collateral::CollateralType;

/// Advanced Features State Module
/// Contains account structures for advanced features like keeper network, circuit breakers, etc.

#[account]
pub struct KeeperNetwork {
    pub total_stake: u64,                    // Total stake in the network
    pub keepers: Vec<KeeperInfo>,            // List of registered keepers
    pub liquidation_rewards_pool: u64,       // Pool for liquidation rewards
    pub min_stake_requirement: u64,          // Minimum stake to become keeper
    pub performance_threshold: u16,          // Minimum performance score
    pub bump: u8,                           // PDA bump
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Eq, Debug)]
pub struct KeeperInfo {
    pub keeper_pubkey: Pubkey,              // Keeper's public key
    pub stake_amount: u64,                  // Amount staked
    pub performance_score: u16,              // Performance score (0-1000)
    pub is_active: bool,                    // Whether keeper is active
    pub total_liquidations: u32,            // Total liquidations performed
    pub total_rewards_earned: u64,          // Total rewards earned
    pub last_activity: i64,                 // Last activity timestamp
}

impl KeeperNetwork {
    pub const INIT_SPACE: usize = 8 + 8 + 4 + (32 + 8 + 2 + 1 + 4 + 8 + 8) * 100 + 8 + 8 + 2 + 1; // Space for up to 100 keepers
    
    /// Check if a keeper is authorized to perform liquidations
    pub fn is_authorized_keeper(&self, keeper_pubkey: &Pubkey) -> bool {
        self.keepers.iter().any(|k| 
            k.keeper_pubkey == *keeper_pubkey && 
            k.is_active && 
            k.performance_score >= self.performance_threshold &&
            k.stake_amount >= self.min_stake_requirement
        )
    }
    
    /// Increment liquidation count for a keeper
    pub fn increment_liquidations(&mut self, keeper_pubkey: &Pubkey) -> Result<()> {
        if let Some(keeper) = self.keepers.iter_mut().find(|k| k.keeper_pubkey == *keeper_pubkey) {
            keeper.total_liquidations += 1;
            keeper.last_activity = Clock::get()?.unix_timestamp;
            Ok(())
        } else {
            Err(crate::ErrorCode::KeeperNotRegistered.into())
        }
    }
    
    /// Add a new keeper to the network
    pub fn add_keeper(&mut self, keeper_info: KeeperInfo) -> Result<()> {
        require!(
            !self.keepers.iter().any(|k| k.keeper_pubkey == keeper_info.keeper_pubkey),
            crate::ErrorCode::AccountAlreadyExists
        );
        
        self.keepers.push(keeper_info);
        Ok(())
    }
    
    /// Remove a keeper from the network
    pub fn remove_keeper(&mut self, keeper_pubkey: &Pubkey) -> Result<()> {
        if let Some(index) = self.keepers.iter().position(|k| k.keeper_pubkey == *keeper_pubkey) {
            self.keepers.remove(index);
            Ok(())
        } else {
            Err(crate::ErrorCode::KeeperNotRegistered.into())
        }
    }
    
    /// Update keeper performance score
    pub fn update_keeper_performance(&mut self, keeper_pubkey: &Pubkey, performance_score: u16) -> Result<()> {
        require!(performance_score <= 1000, crate::ErrorCode::InvalidPerformanceScore);
        
        if let Some(keeper) = self.keepers.iter_mut().find(|k| k.keeper_pubkey == *keeper_pubkey) {
            keeper.performance_score = performance_score;
            keeper.last_activity = Clock::get()?.unix_timestamp;
            Ok(())
        } else {
            Err(crate::ErrorCode::KeeperNotRegistered.into())
        }
    }
}

#[account]
pub struct CircuitBreaker {
    pub is_triggered: bool,                 // Whether circuit breaker is active
    pub trigger_time: i64,                  // When it was triggered
    pub reset_time: i64,                    // When it was reset
    pub breaker_type: CircuitBreakerType,   // Type of circuit breaker
    pub triggered_by: Pubkey,               // Who triggered it
    pub reset_by: Pubkey,                   // Who reset it
    pub price_change_threshold: u16,        // Price change threshold (basis points)
    pub volume_threshold: u64,              // Volume threshold
    pub time_window: u64,                   // Time window in seconds
    pub bump: u8,                          // PDA bump
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Debug)]
pub enum CircuitBreakerType {
    PriceVolatility,    // Triggered by extreme price movements
    VolumeSpike,        // Triggered by unusual volume
    SystemOverload,     // Triggered by system performance issues
    EmergencyStop,      // Manual emergency stop
}

impl CircuitBreaker {
    pub const INIT_SPACE: usize = 1 + 8 + 8 + 1 + 32 + 32 + 2 + 8 + 8 + 1;
}

#[account]
pub struct JitProvider {
    pub provider_pubkey: Pubkey,           // JIT provider's public key
    pub available_liquidity: u64,          // Available liquidity
    pub fee_rate: u16,                      // Fee rate in basis points
    pub total_volume: u64,                  // Total volume provided
    pub total_fees_earned: u64,             // Total fees earned
    pub min_order_size: u64,                 // Minimum order size
    pub max_order_size: u64,                // Maximum order size
    pub last_update: i64,                   // Last update timestamp
    pub is_active: bool,                    // Whether provider is active
    pub bump: u8,                          // PDA bump
}

impl JitProvider {
    pub const INIT_SPACE: usize = 32 + 8 + 2 + 8 + 8 + 8 + 8 + 8 + 1 + 1;
}

#[account]
pub struct MarketMakerVault {
    pub vault_pubkey: Pubkey,              // Vault's public key
    pub strategy: MarketMakingStrategy,     // Market making strategy
    pub capital_allocation: u64,            // Capital allocated to vault
    pub performance_fee: u16,               // Performance fee in basis points
    pub total_volume: u64,                  // Total volume traded
    pub total_pnl: i64,                     // Total P&L
    pub is_active: bool,                    // Whether vault is active
    pub created_at: i64,                    // Creation timestamp
    pub bump: u8,                          // PDA bump
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub enum MarketMakingStrategy {
    GridTrading,        // Grid trading strategy
    MeanReversion,      // Mean reversion strategy
    Arbitrage,          // Arbitrage strategy
    LiquidityProvision, // Pure liquidity provision
}

impl MarketMakerVault {
    pub const INIT_SPACE: usize = 32 + 1 + 8 + 2 + 8 + 8 + 1 + 8 + 1;
}

#[account]
pub struct PointsSystem {
    pub user_points: Vec<UserPoints>,       // User points mapping
    pub trading_multiplier: u16,            // Trading activity multiplier
    pub referral_bonus: u16,                // Referral bonus multiplier
    pub staking_multiplier: u16,            // Staking multiplier
    pub total_points_distributed: u64,      // Total points distributed
    pub bump: u8,                          // PDA bump
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct UserPoints {
    pub user_pubkey: Pubkey,               // User's public key
    pub total_points: u64,                 // Total points earned
    pub trading_points: u64,               // Points from trading
    pub referral_points: u64,              // Points from referrals
    pub staking_points: u64,               // Points from staking
    pub last_updated: i64,                 // Last update timestamp
}

impl PointsSystem {
    pub const INIT_SPACE: usize = 4 + (32 + 8 + 8 + 8 + 8 + 8) * 1000 + 2 + 2 + 2 + 8 + 1; // Space for up to 1000 users
}

// ===== CROSS-COLLATERALIZATION SYSTEM =====

#[account]
pub struct CrossCollateralAccount {
    pub user: Pubkey,                    // User who owns the cross-collateral account
    pub total_collateral_value: u64,     // Total USD value of all collateral
    pub total_borrowed_value: u64,       // Total USD value borrowed against collateral
    pub collateral_assets: Vec<CollateralAsset>, // List of collateral assets
    pub initial_asset_weight: u16,       // Initial asset weight (basis points)
    pub maintenance_asset_weight: u16,    // Maintenance asset weight (basis points)
    pub initial_liability_weight: u16,   // Initial liability weight (basis points)
    pub maintenance_liability_weight: u16, // Maintenance liability weight (basis points)
    pub imf_factor: u16,                 // IMF (Initial Margin Factor) in basis points
    pub last_health_check: i64,          // Last health check timestamp
    pub is_active: bool,                 // Whether account is active
    pub bump: u8,                       // PDA bump
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct CollateralAsset {
    pub asset_type: CollateralType,      // Type of collateral asset
    pub amount: u64,                     // Amount of asset
    pub value_usd: u64,                  // USD value of asset
    pub asset_weight: u16,               // Asset weight (basis points)
    pub liability_weight: u16,           // Liability weight (basis points)
    pub last_price_update: i64,          // Last price update timestamp
}

impl CrossCollateralAccount {
    pub const INIT_SPACE: usize = 32 + 8 + 8 + 4 + (1 + 8 + 8 + 2 + 2 + 8) * 10 + 2 + 2 + 2 + 2 + 2 + 8 + 1 + 1; // Space for up to 10 collateral assets
}
