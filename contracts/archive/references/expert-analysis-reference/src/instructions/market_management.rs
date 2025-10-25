use anchor_lang::prelude::*;
use crate::state::market::Market;
use crate::ErrorCode;

/// Market Management Module
/// Handles market initialization and oracle price updates

/// Initialize a new market with oracle integration
pub fn initialize_market(
    ctx: Context<InitializeMarket>,
    base_asset: String,
    quote_asset: String,
    initial_price: u64,
    max_leverage: u8,
    initial_margin_ratio: u16, // In basis points (e.g., 500 = 5%)
    maintenance_margin_ratio: u16, // In basis points (e.g., 300 = 3%)
) -> Result<()> {
    let market = &mut ctx.accounts.market;
    
    // Validate parameters
    require!(max_leverage >= 1 && max_leverage <= 100, ErrorCode::InvalidMaxLeverage);
    require!(initial_margin_ratio > maintenance_margin_ratio, ErrorCode::InvalidMarginRatio);
    require!(initial_margin_ratio <= 10000, ErrorCode::InvalidMarginRatio); // Max 100%
    
    market.base_asset = base_asset;
    market.quote_asset = quote_asset;
    market.base_reserve = 1000000; // Initial liquidity
    market.quote_reserve = initial_price * 1000000;
    market.funding_rate = 0;
    market.last_funding_time = Clock::get()?.unix_timestamp;
    market.funding_interval = 3600; // 1 hour funding interval
    market.authority = ctx.accounts.authority.key();
    market.max_leverage = max_leverage;
    market.initial_margin_ratio = initial_margin_ratio;
    market.maintenance_margin_ratio = maintenance_margin_ratio;
    market.is_active = true;
    market.bump = ctx.bumps.market;
    
    msg!("Market initialized: {}/{} with max leverage {}x", 
         market.base_asset, market.quote_asset, max_leverage);
    Ok(())
}

/// Update oracle price (called by keeper bots)
pub fn update_oracle_price(ctx: Context<UpdateOraclePrice>, new_price: u64) -> Result<()> {
    let market = &mut ctx.accounts.market;
    
    // Validate price data
    require!(new_price > 0, ErrorCode::InvalidPrice);
    
    // Update market with oracle price
    market.last_oracle_price = new_price;
    market.last_oracle_update = Clock::get()?.unix_timestamp;
    
    msg!("Oracle price updated: {} for {}/{}", 
         new_price, market.base_asset, market.quote_asset);
    Ok(())
}

/// Settle funding for all positions in a market
pub fn settle_funding(ctx: Context<SettleFunding>) -> Result<()> {
    let market = &mut ctx.accounts.market;
    let current_time = Clock::get()?.unix_timestamp;
    
    // Check if it's time for funding settlement
    require!(
        current_time - market.last_funding_time >= market.funding_interval,
        ErrorCode::FundingNotDue
    );

    // Calculate new funding rate based on premium index
    let premium_index = market.calculate_premium_index()?;
    let new_funding_rate = market.calculate_funding_rate(premium_index)?;
    
    market.funding_rate = new_funding_rate;
    market.last_funding_time = current_time;
    
    msg!("Funding settled: rate = {} bps", new_funding_rate);
    Ok(())
}

/// Update market parameters
pub fn update_market_parameters(
    ctx: Context<UpdateMarketRiskParameters>,
    max_leverage: Option<u8>,
    initial_margin_ratio: Option<u16>,
    maintenance_margin_ratio: Option<u16>,
) -> Result<()> {
    let market = &mut ctx.accounts.market;
    
    if let Some(leverage) = max_leverage {
        market.max_leverage = leverage;
    }
    if let Some(ratio) = initial_margin_ratio {
        market.initial_margin_ratio = ratio;
    }
    if let Some(ratio) = maintenance_margin_ratio {
        market.maintenance_margin_ratio = ratio;
    }
    
    msg!("Market parameters updated");
    Ok(())
}

#[derive(Accounts)]
#[instruction(base_asset: String, quote_asset: String)]
pub struct InitializeMarket<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + Market::INIT_SPACE,
        seeds = [b"market", base_asset.as_bytes(), quote_asset.as_bytes()],
        bump
    )]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct UpdateOraclePrice<'info> {
    #[account(
        mut,
        constraint = market.authority == authority.key()
    )]
    pub market: Account<'info, Market>,
    
    /// CHECK: This is the Pyth price feed account
    pub price_feed: AccountInfo<'info>,
    
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct SettleFunding<'info> {
    #[account(
        mut,
        constraint = market.authority == authority.key()
    )]
    pub market: Account<'info, Market>,
    
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct UpdateMarketRiskParameters<'info> {
    #[account(
        mut,
        constraint = market.authority == authority.key()
    )]
    pub market: Account<'info, Market>,
    
    pub authority: Signer<'info>,
}