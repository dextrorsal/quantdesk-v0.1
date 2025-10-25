//! Admin functions and emergency controls
//! This module contains all administrative functions for protocol management, emergency controls, and system maintenance.

use anchor_lang::prelude::*;
use crate::{
    state::advanced::CircuitBreakerType,
    ErrorCode,
};

/// Update risk management parameters
pub fn update_risk_parameters(
    ctx: Context<UpdateRiskParameters>,
    max_position_size: u64,
    max_leverage: u8,
    liquidation_threshold: u16,
) -> Result<()> {
    require!(max_position_size > 0, ErrorCode::InvalidInsuranceFundOperation);
    require!(max_leverage >= 1 && max_leverage <= 100, ErrorCode::InvalidInsuranceFundOperation);
    require!(liquidation_threshold > 0 && liquidation_threshold <= 10000, ErrorCode::InvalidInsuranceFundOperation);
    
    let insurance_fund = &mut ctx.accounts.insurance_fund;
    insurance_fund.max_utilization = liquidation_threshold;
    
    msg!("Risk parameters updated: max_position={}, max_leverage={}x, threshold={}", 
         max_position_size, max_leverage, liquidation_threshold);
    Ok(())
}

/// Pause all program operations
pub fn pause_program(ctx: Context<PauseProgram>) -> Result<()> {
    require!(!ctx.accounts.program_state.is_paused, ErrorCode::ProgramPaused);
    
    let program_state = &mut ctx.accounts.program_state;
    program_state.is_paused = true;
    
    msg!("Program paused by authority");
    Ok(())
}

/// Resume program operations
pub fn resume_program(ctx: Context<ResumeProgram>) -> Result<()> {
    require!(ctx.accounts.program_state.is_paused, ErrorCode::ProgramPaused);
    
    let program_state = &mut ctx.accounts.program_state;
    program_state.is_paused = false;
    
    msg!("Program resumed by authority");
    Ok(())
}

/// Emergency withdrawal (only when paused)
pub fn emergency_withdraw(
    ctx: Context<EmergencyWithdraw>,
    amount: u64,
) -> Result<()> {
    require!(ctx.accounts.program_state.is_paused, ErrorCode::ProgramPaused);
    require!(amount > 0, ErrorCode::InvalidInsuranceFundOperation);
    
    msg!("Emergency withdrawal of {} executed", amount);
    Ok(())
}

/// Update trading fee rates
pub fn update_trading_fees(
    ctx: Context<UpdateTradingFees>,
    maker_fee_rate: u16,
    taker_fee_rate: u16,
) -> Result<()> {
    require!(maker_fee_rate <= 1000, ErrorCode::InvalidFeeParameters); // Max 10%
    require!(taker_fee_rate <= 1000, ErrorCode::InvalidFeeParameters); // Max 10%
    require!(maker_fee_rate <= taker_fee_rate, ErrorCode::InvalidFeeParameters); // Maker fee <= taker fee
    
    let fee_collector = &mut ctx.accounts.fee_collector;
    fee_collector.maker_fee_rate = maker_fee_rate;
    fee_collector.taker_fee_rate = taker_fee_rate;
    
    msg!("Trading fees updated: maker={}bps, taker={}bps", maker_fee_rate, taker_fee_rate);
    Ok(())
}

/// Update funding fee parameters
pub fn update_funding_fees(
    ctx: Context<UpdateFundingFees>,
    funding_rate_cap: i64,
    funding_rate_floor: i64,
) -> Result<()> {
    require!(funding_rate_cap >= 0, ErrorCode::InvalidFeeParameters);
    require!(funding_rate_floor <= 0, ErrorCode::InvalidFeeParameters);
    require!(funding_rate_cap <= 10000, ErrorCode::InvalidFeeParameters); // Max 100%
    require!(funding_rate_floor >= -10000, ErrorCode::InvalidFeeParameters); // Min -100%
    
    let fee_collector = &mut ctx.accounts.fee_collector;
    fee_collector.funding_rate_cap = funding_rate_cap;
    fee_collector.funding_rate_floor = funding_rate_floor;
    
    msg!("Funding fees updated: cap={}bps, floor={}bps", funding_rate_cap, funding_rate_floor);
    Ok(())
}

/// Collect accumulated fees
pub fn collect_fees(ctx: Context<CollectFees>) -> Result<()> {
    let fee_collector = &mut ctx.accounts.fee_collector;
    let total_fees = fee_collector.trading_fees_collected + fee_collector.funding_fees_collected;
    
    msg!("Collected {} total fees", total_fees);
    Ok(())
}

/// Emergency oracle price override
pub fn emergency_oracle_override(
    ctx: Context<EmergencyOracleOverride>,
    price: u64,
) -> Result<()> {
    require!(price > 0, ErrorCode::InvalidOracleWeight);
    
    let market = &mut ctx.accounts.market;
    market.last_oracle_price = price;
    market.last_oracle_update = Clock::get()?.unix_timestamp;
    
    msg!("Emergency oracle override: price={}", price);
    Ok(())
}

/// Update Pyth price feed
pub fn update_pyth_price(
    ctx: Context<UpdatePythPrice>,
    price_feed: Pubkey,
) -> Result<()> {
    let market = &mut ctx.accounts.market;
    market.last_oracle_price = 50000; // Placeholder price
    market.last_oracle_update = Clock::get()?.unix_timestamp;
    
    msg!("Updated Pyth price feed: {}", price_feed);
    Ok(())
}

/// Update program authority
pub fn update_program_authority(
    ctx: Context<UpdateProgramAuthority>,
    new_authority: Pubkey,
) -> Result<()> {
    require!(new_authority != Pubkey::default(), ErrorCode::UnauthorizedAdminOperation);
    
    let program_state = &mut ctx.accounts.program_state;
    program_state.authority = new_authority;
    
    msg!("Program authority updated to {}", new_authority);
    Ok(())
}

/// Update market parameters
pub fn update_market_parameters(
    ctx: Context<UpdateMarketParameters>,
    max_leverage: Option<u8>,
    initial_margin_ratio: Option<u16>,
    maintenance_margin_ratio: Option<u16>,
    funding_rate_cap: Option<i64>,
) -> Result<()> {
    let market = &mut ctx.accounts.market;
    
    if let Some(leverage) = max_leverage {
        require!(leverage >= 1 && leverage <= 100, ErrorCode::InvalidMarketParameters);
        market.max_leverage = leverage;
    }
    
    if let Some(ratio) = initial_margin_ratio {
        require!(ratio > 0 && ratio <= 10000, ErrorCode::InvalidMarketParameters);
        market.initial_margin_ratio = ratio;
    }
    
    if let Some(ratio) = maintenance_margin_ratio {
        require!(ratio > 0 && ratio <= 10000, ErrorCode::InvalidMarketParameters);
        market.maintenance_margin_ratio = ratio;
    }
    
    if let Some(cap) = funding_rate_cap {
        require!(cap >= 0 && cap <= 10000, ErrorCode::InvalidMarketParameters);
        market.funding_rate = cap;
    }
    
    msg!("Market parameters updated");
    Ok(())
}

/// Trigger circuit breaker for emergency situations
pub fn trigger_circuit_breaker(
    ctx: Context<TriggerCircuitBreaker>,
    breaker_type: CircuitBreakerType,
) -> Result<()> {
    let circuit_breaker = &mut ctx.accounts.circuit_breaker;
    
    circuit_breaker.is_triggered = true;
    circuit_breaker.trigger_time = Clock::get()?.unix_timestamp;
    circuit_breaker.breaker_type = breaker_type.clone();
    circuit_breaker.triggered_by = ctx.accounts.authority.key();
    
    msg!("Circuit breaker triggered: {:?} at {}", breaker_type, circuit_breaker.trigger_time);
    Ok(())
}

/// Reset circuit breaker after emergency is resolved
pub fn reset_circuit_breaker(
    ctx: Context<ResetCircuitBreaker>,
) -> Result<()> {
    let circuit_breaker = &mut ctx.accounts.circuit_breaker;
    
    circuit_breaker.is_triggered = false;
    circuit_breaker.reset_time = Clock::get()?.unix_timestamp;
    circuit_breaker.reset_by = ctx.accounts.authority.key();
    
    msg!("Circuit breaker reset at {}", circuit_breaker.reset_time);
    Ok(())
}

/// Context structs for admin functions

#[derive(Accounts)]
pub struct UpdateRiskParameters<'info> {
    #[account(mut)]
    pub insurance_fund: Account<'info, crate::state::protocol::InsuranceFund>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct PauseProgram<'info> {
    #[account(mut)]
    pub program_state: Account<'info, crate::state::protocol::ProgramState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct ResumeProgram<'info> {
    #[account(mut)]
    pub program_state: Account<'info, crate::state::protocol::ProgramState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct EmergencyWithdraw<'info> {
    #[account(mut)]
    pub program_state: Account<'info, crate::state::protocol::ProgramState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct UpdateTradingFees<'info> {
    #[account(mut)]
    pub fee_collector: Account<'info, crate::state::protocol::FeeCollector>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct UpdateFundingFees<'info> {
    #[account(mut)]
    pub fee_collector: Account<'info, crate::state::protocol::FeeCollector>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct CollectFees<'info> {
    #[account(mut)]
    pub fee_collector: Account<'info, crate::state::protocol::FeeCollector>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct EmergencyOracleOverride<'info> {
    #[account(mut)]
    pub market: Account<'info, crate::state::market::Market>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct UpdatePythPrice<'info> {
    #[account(mut)]
    pub market: Account<'info, crate::state::market::Market>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct UpdateProgramAuthority<'info> {
    #[account(mut)]
    pub program_state: Account<'info, crate::state::protocol::ProgramState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct UpdateMarketParameters<'info> {
    #[account(mut)]
    pub market: Account<'info, crate::state::market::Market>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct TriggerCircuitBreaker<'info> {
    #[account(mut)]
    pub circuit_breaker: Account<'info, crate::state::advanced::CircuitBreaker>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct ResetCircuitBreaker<'info> {
    #[account(mut)]
    pub circuit_breaker: Account<'info, crate::state::advanced::CircuitBreaker>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}