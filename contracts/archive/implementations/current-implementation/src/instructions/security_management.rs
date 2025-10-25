//! Security Management Instructions
//! Implements security-hardened architecture for QuantDesk Perpetual DEX

use anchor_lang::prelude::*;
use crate::security::*;
use crate::ErrorCode;
use crate::oracle::OracleType;

/// Initialize Security Circuit Breaker Context
#[derive(Accounts)]
pub struct InitializeSecurityCircuitBreaker<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + SecurityCircuitBreaker::INIT_SPACE,
        seeds = [b"security_circuit_breaker"],
        bump
    )]
    pub security_circuit_breaker: Account<'info, SecurityCircuitBreaker>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

/// Initialize Keeper Security Manager Context
#[derive(Accounts)]
pub struct InitializeKeeperSecurityManager<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + KeeperSecurityManager::INIT_SPACE,
        seeds = [b"keeper_security_manager"],
        bump
    )]
    pub keeper_security_manager: Box<Account<'info, KeeperSecurityManager>>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

/// Initialize Oracle Staleness Protection Context
#[derive(Accounts)]
pub struct InitializeOracleStalenessProtection<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + OracleStalenessProtection::INIT_SPACE,
        seeds = [b"oracle_staleness_protection"],
        bump
    )]
    pub oracle_staleness_protection: Account<'info, OracleStalenessProtection>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

/// Update Security Parameters Context
#[derive(Accounts)]
pub struct UpdateSecurityParameters<'info> {
    #[account(mut)]
    pub security_circuit_breaker: Account<'info, SecurityCircuitBreaker>,
    
    #[account(mut)]
    pub keeper_security_manager: Box<Account<'info, KeeperSecurityManager>>,
    
    #[account(mut)]
    pub oracle_staleness_protection: Account<'info, OracleStalenessProtection>,
    
    pub authority: Signer<'info>,
}

/// Authorize Keeper Context
#[derive(Accounts)]
pub struct AuthorizeKeeper<'info> {
    #[account(mut)]
    pub keeper_security_manager: Box<Account<'info, KeeperSecurityManager>>,
    
    pub authority: Signer<'info>,
}

/// Emergency Pause Context
#[derive(Accounts)]
pub struct EmergencyPause<'info> {
    #[account(mut)]
    pub security_circuit_breaker: Account<'info, SecurityCircuitBreaker>,
    
    pub authority: Signer<'info>,
}

/// Emergency Resume Context
#[derive(Accounts)]
pub struct EmergencyResume<'info> {
    #[account(mut)]
    pub security_circuit_breaker: Account<'info, SecurityCircuitBreaker>,
    
    pub authority: Signer<'info>,
}

/// Check Security Before Trading Context
#[derive(Accounts)]
pub struct CheckSecurityBeforeTrading<'info> {
    #[account(mut)]
    pub security_circuit_breaker: Account<'info, SecurityCircuitBreaker>,
    
    #[account(mut)]
    pub keeper_security_manager: Box<Account<'info, KeeperSecurityManager>>,
    
    #[account(mut)]
    pub oracle_staleness_protection: Account<'info, OracleStalenessProtection>,
    
    /// CHECK: Pyth oracle price feed
    pub pyth_price_feed: AccountInfo<'info>,
    
    /// CHECK: Switchboard oracle price feed (optional)
    pub switchboard_price_feed: Option<AccountInfo<'info>>,
}

/// Initialize Security Circuit Breaker
pub fn initialize_security_circuit_breaker(
    ctx: Context<InitializeSecurityCircuitBreaker>,
) -> Result<()> {
    let security_circuit_breaker = &mut ctx.accounts.security_circuit_breaker;
    **security_circuit_breaker = SecurityCircuitBreaker::new();
    security_circuit_breaker.bump = ctx.bumps.security_circuit_breaker;
    
    msg!("🔒 Security Circuit Breaker initialized");
    Ok(())
}

/// Initialize Keeper Security Manager
pub fn initialize_keeper_security_manager(
    ctx: Context<InitializeKeeperSecurityManager>,
) -> Result<()> {
    let mut keeper_security_manager = &mut ctx.accounts.keeper_security_manager;
    keeper_security_manager.required_signatures = 2;
    keeper_security_manager.authorized_keepers = [KeeperAuth::default(); 3];
    keeper_security_manager.keeper_count = 0;
    keeper_security_manager.liquidation_window_start = 0;
    keeper_security_manager.liquidation_window_end = 0;
    keeper_security_manager.max_liquidations_per_window = 10;
    keeper_security_manager.liquidation_rate_limit = 5;
    keeper_security_manager.last_liquidation_time = 0;
    keeper_security_manager.liquidations_in_current_hour = 0;
    keeper_security_manager.min_keeper_stake = 1000000; // 0.001 SOL
    keeper_security_manager.min_performance_score = 80;
    keeper_security_manager.max_position_size_for_liquidation = 1000000000; // 1 SOL
    keeper_security_manager.liquidation_history = [LiquidationRecord::default(); 5];
    keeper_security_manager.history_index = 0;
    keeper_security_manager.bump = ctx.bumps.keeper_security_manager;
    
    msg!("🛡️ Keeper Security Manager initialized");
    Ok(())
}

/// Initialize Oracle Staleness Protection
pub fn initialize_oracle_staleness_protection(
    ctx: Context<InitializeOracleStalenessProtection>,
) -> Result<()> {
    let oracle_staleness_protection = &mut ctx.accounts.oracle_staleness_protection;
    **oracle_staleness_protection = OracleStalenessProtection::new();
    oracle_staleness_protection.bump = ctx.bumps.oracle_staleness_protection;
    
    msg!("🔍 Oracle Staleness Protection initialized");
    Ok(())
}

/// Update Security Parameters
pub fn update_security_parameters(
    mut ctx: Context<UpdateSecurityParameters>,
    max_price_change_percent: u16,
    max_volume_spike_percent: u16,
    max_oracle_deviation_percent: u16,
    max_system_load_percent: u16,
    max_staleness_seconds: u64,
    liquidation_rate_limit: u32,
) -> Result<()> {
    // Box the context to reduce stack usage
    let mut boxed_ctx = Box::new(ctx);
    
    // Validate parameters first
    validate_security_parameters(
        max_price_change_percent,
        max_volume_spike_percent,
        max_oracle_deviation_percent,
        max_system_load_percent,
        max_staleness_seconds,
        liquidation_rate_limit,
    )?;
    
    // Update circuit breaker parameters
    update_circuit_breaker_params(&mut boxed_ctx, max_price_change_percent, max_volume_spike_percent, max_oracle_deviation_percent, max_system_load_percent)?;
    
    // Update oracle and keeper parameters
    update_oracle_and_keeper_params(&mut boxed_ctx, max_staleness_seconds, liquidation_rate_limit)?;
    
    msg!("🔧 Security parameters updated: price_change={}%, volume_spike={}%, oracle_deviation={}%, system_load={}%, staleness={}s, liquidation_rate={}/hour", 
         max_price_change_percent / 100, max_volume_spike_percent / 100, max_oracle_deviation_percent / 100, 
         max_system_load_percent / 100, max_staleness_seconds, liquidation_rate_limit);
    
    Ok(())
}

/// Validate security parameters
fn validate_security_parameters(
    max_price_change_percent: u16,
    max_volume_spike_percent: u16,
    max_oracle_deviation_percent: u16,
    max_system_load_percent: u16,
    max_staleness_seconds: u64,
    liquidation_rate_limit: u32,
) -> Result<()> {
    require!(max_price_change_percent > 0 && max_price_change_percent <= 10000, ErrorCode::InvalidSecurityParameters);
    require!(max_volume_spike_percent > 0 && max_volume_spike_percent <= 10000, ErrorCode::InvalidSecurityParameters);
    require!(max_oracle_deviation_percent > 0 && max_oracle_deviation_percent <= 1000, ErrorCode::InvalidSecurityParameters);
    require!(max_system_load_percent > 0 && max_system_load_percent <= 10000, ErrorCode::InvalidSecurityParameters);
    require!(max_staleness_seconds > 0 && max_staleness_seconds <= 3600, ErrorCode::InvalidSecurityParameters);
    require!(liquidation_rate_limit > 0 && liquidation_rate_limit <= 10000, ErrorCode::InvalidSecurityParameters);
    Ok(())
}

/// Update circuit breaker parameters
fn update_circuit_breaker_params(
    ctx: &mut Context<UpdateSecurityParameters>,
    max_price_change_percent: u16,
    max_volume_spike_percent: u16,
    max_oracle_deviation_percent: u16,
    max_system_load_percent: u16,
) -> Result<()> {
    let security_circuit_breaker = &mut ctx.accounts.security_circuit_breaker;
    
    security_circuit_breaker.max_price_change_percent = max_price_change_percent;
    security_circuit_breaker.max_volume_spike_percent = max_volume_spike_percent;
    security_circuit_breaker.max_oracle_deviation_percent = max_oracle_deviation_percent;
    security_circuit_breaker.max_system_load_percent = max_system_load_percent;
    
    Ok(())
}

/// Update oracle and keeper parameters
fn update_oracle_and_keeper_params(
    ctx: &mut Context<UpdateSecurityParameters>,
    max_staleness_seconds: u64,
    liquidation_rate_limit: u32,
) -> Result<()> {
    let oracle_staleness_protection = &mut ctx.accounts.oracle_staleness_protection;
    let keeper_security_manager = &mut ctx.accounts.keeper_security_manager;
    
    oracle_staleness_protection.max_staleness_seconds = max_staleness_seconds;
    keeper_security_manager.liquidation_rate_limit = liquidation_rate_limit;
    
    Ok(())
}

/// Authorize Keeper for Liquidations
pub fn authorize_keeper(
    ctx: Context<AuthorizeKeeper>,
    keeper_pubkey: Pubkey,
    stake_amount: u64,
    performance_score: u16,
    auth_level: KeeperAuthLevel,
) -> Result<()> {
    let mut keeper_security_manager = &mut ctx.accounts.keeper_security_manager;
    
    keeper_security_manager.authorize_keeper(keeper_pubkey, stake_amount, performance_score, auth_level)?;
    
    msg!("✅ Keeper authorized: {} with stake: {} lamports, score: {}, level: {:?}", 
         keeper_pubkey, stake_amount, performance_score, auth_level);
    
    Ok(())
}

/// Deauthorize Keeper
pub fn deauthorize_keeper(
    ctx: Context<AuthorizeKeeper>,
    keeper_pubkey: Pubkey,
) -> Result<()> {
    let mut keeper_security_manager = &mut ctx.accounts.keeper_security_manager;
    
    keeper_security_manager.deauthorize_keeper(&keeper_pubkey)?;
    
    msg!("❌ Keeper deauthorized: {}", keeper_pubkey);
    
    Ok(())
}

/// Emergency Pause
pub fn emergency_pause(
    ctx: Context<EmergencyPause>,
) -> Result<()> {
    let security_circuit_breaker = &mut ctx.accounts.security_circuit_breaker;
    
    security_circuit_breaker.emergency_pause();
    
    msg!("🚨 EMERGENCY PAUSE ACTIVATED by authority: {}", ctx.accounts.authority.key());
    
    Ok(())
}

/// Emergency Resume
pub fn emergency_resume(
    ctx: Context<EmergencyResume>,
) -> Result<()> {
    let security_circuit_breaker = &mut ctx.accounts.security_circuit_breaker;
    
    security_circuit_breaker.emergency_resume();
    
    msg!("✅ EMERGENCY PAUSE LIFTED by authority: {}", ctx.accounts.authority.key());
    
    Ok(())
}

/// Check Security Before Trading Operations
pub fn check_security_before_trading(
    mut ctx: Context<CheckSecurityBeforeTrading>,
    current_price: u64,
    current_volume: u64,
    system_load: u16,
) -> Result<()> {
    // Box the context to reduce stack usage
    let mut boxed_ctx = Box::new(ctx);
    
    // Get oracle prices
    let oracle_prices = get_oracle_prices(&boxed_ctx)?;
    
    // Check circuit breakers
    check_circuit_breakers(&mut boxed_ctx, current_price, current_volume, oracle_prices.0, oracle_prices.1, system_load)?;
    
    // Check oracle staleness
    check_oracle_staleness(&mut boxed_ctx, oracle_prices.0)?;
    
    msg!("✅ Security checks passed - trading operations allowed");
    
    Ok(())
}

/// Get oracle prices from feeds
fn get_oracle_prices(ctx: &Context<CheckSecurityBeforeTrading>) -> Result<(u64, Option<u64>)> {
    let primary_oracle_price = get_oracle_price(&ctx.accounts.pyth_price_feed)?;
    let secondary_oracle_price = if let Some(ref switchboard_feed) = ctx.accounts.switchboard_price_feed {
        get_oracle_price(switchboard_feed).ok()
    } else {
        None
    };
    
    Ok((primary_oracle_price, secondary_oracle_price))
}

/// Check circuit breakers
fn check_circuit_breakers(
    ctx: &mut Context<CheckSecurityBeforeTrading>,
    current_price: u64,
    current_volume: u64,
    primary_oracle_price: u64,
    secondary_oracle_price: Option<u64>,
    system_load: u16,
) -> Result<()> {
    let security_circuit_breaker = &mut ctx.accounts.security_circuit_breaker;
    
    let should_trigger = security_circuit_breaker.check_circuit_breakers(
        current_price,
        current_volume,
        primary_oracle_price,
        secondary_oracle_price.unwrap_or(0),
        system_load,
    )?;
    
    if should_trigger {
        return Err(ErrorCode::CircuitBreakerActive.into());
    }
    
    Ok(())
}

/// Check oracle staleness
fn check_oracle_staleness(
    ctx: &mut Context<CheckSecurityBeforeTrading>,
    primary_oracle_price: u64,
) -> Result<()> {
    let oracle_staleness_protection = &mut ctx.accounts.oracle_staleness_protection;
    
    let current_time = Clock::get()?.unix_timestamp;
    let primary_health = oracle_staleness_protection.check_oracle_health(
        primary_oracle_price,
        current_time,
        crate::security::OracleType::Pyth,
    )?;
    
    if matches!(primary_health, OracleHealthStatus::Failed) {
        return Err(ErrorCode::OracleStalenessProtectionTriggered.into());
    }
    
    Ok(())
}

/// Check Keeper Authorization for Liquidation
pub fn check_keeper_authorization(
    mut ctx: Context<CheckSecurityBeforeTrading>,
    keeper_pubkey: Pubkey,
) -> Result<()> {
    // Box the context to reduce stack usage
    let mut boxed_ctx = Box::new(ctx);
    
    let keeper_security_manager = &mut boxed_ctx.accounts.keeper_security_manager;
    
    // Check if keeper is authorized
    let is_authorized = keeper_security_manager.is_keeper_authorized(&keeper_pubkey)?;
    require!(is_authorized, ErrorCode::KeeperAuthorizationFailed);
    
    // Check liquidation rate limits
    let rate_limit_ok = keeper_security_manager.check_liquidation_rate_limit()?;
    require!(rate_limit_ok, ErrorCode::LiquidationRateLimitExceeded);
    
    msg!("✅ Keeper authorization check passed: {}", keeper_pubkey);
    
    Ok(())
}

/// Record Liquidation Attempt
pub fn record_liquidation_attempt(
    mut ctx: Context<CheckSecurityBeforeTrading>,
    keeper_pubkey: Pubkey,
    position_owner: Pubkey,
    position_size: u64,
    liquidation_price: u64,
    success: bool,
    reason: LiquidationReason,
) -> Result<()> {
    // Box the context to reduce stack usage
    let mut boxed_ctx = Box::new(ctx);
    
    let keeper_security_manager = &mut boxed_ctx.accounts.keeper_security_manager;
    
    keeper_security_manager.record_liquidation(
        keeper_pubkey,
        position_owner,
        position_size,
        liquidation_price,
        success,
        reason,
    )?;
    
    msg!("📊 Liquidation attempt recorded: keeper={}, success={}, reason={:?}", 
         keeper_pubkey, success, reason);
    
    Ok(())
}

/// Set Emergency Price
pub fn set_emergency_price(
    mut ctx: Context<CheckSecurityBeforeTrading>,
    emergency_price: u64,
) -> Result<()> {
    // Box the context to reduce stack usage
    let mut boxed_ctx = Box::new(ctx);
    
    let oracle_staleness_protection = &mut boxed_ctx.accounts.oracle_staleness_protection;
    
    oracle_staleness_protection.set_emergency_price(emergency_price)?;
    
    msg!("🚨 Emergency price set: {}", emergency_price);
    
    Ok(())
}

/// Helper function to get oracle price
fn get_oracle_price(price_feed: &AccountInfo) -> Result<u64> {
    // This is a simplified version - in practice, you'd use the actual oracle integration
    // For now, return a placeholder price
    Ok(1000000) // $1000.00 in 6 decimal format
}
