//! Position management instructions
//! This module contains all position-related instruction handlers and their context structs.

use anchor_lang::prelude::*;
use crate::{
    state::position::{Position, PositionSide, PositionStatus},
    state::Market,
    state::UserAccount,
    ErrorCode,
    margin::*,
};

/// Initialize a new position with all required fields
fn initialize_new_position(
    position: &mut Position,
    user_account: &UserAccount,
    market: &Pubkey,
    position_index: u16,
    side: PositionSide,
    size: u64,
    entry_price: u64,
    required_margin: u64,
    leverage: u16,
    bump: u8,
) -> Result<()> {
    let clock = Clock::get()?;

    position.user_account = user_account.authority;
    position.user = user_account.authority; // Set both fields
    position.market = *market;
    position.position_index = position_index;
    position.side = side;
    position.status = PositionStatus::Open;
    position.size = size;
    position.entry_price = entry_price;
    position.current_price = entry_price;
    position.liquidation_price = calculate_liquidation_price(
        entry_price,
        side,
        leverage,
        MAINTENANCE_MARGIN_RATIO,
    );
    position.unrealized_pnl = 0;
    position.realized_pnl = 0;
    position.funding_rate = 0;
    position.last_funding_update = clock.unix_timestamp;
    position.total_funding_paid = 0;
    position.initial_margin = required_margin;
    position.maintenance_margin = required_margin.checked_mul(8).unwrap().checked_div(10).unwrap(); // 80% of initial
    position.leverage = leverage;
    position.opened_at = clock.unix_timestamp;
    position.closed_at = 0;
    position.last_updated = clock.unix_timestamp;
    position.bump = bump;

    Ok(())
}

/// Open a new position with enhanced security
pub fn open_position(
    ctx: Context<OpenPosition>,
    position_index: u16,
    side: PositionSide,
    size: u64,
    leverage: u16,
    entry_price: u64,
) -> Result<()> {
    let position = &mut ctx.accounts.position;
    let user_account = &mut ctx.accounts.user_account;
    let market = &ctx.accounts.market;
    
    // Validate leverage
    validate_leverage(leverage, user_account.max_leverage as u8)?;
    
    // Validate position size
    validate_position_size(size)?;
    
    // Calculate margin requirement
    let position_value = size.checked_mul(entry_price).unwrap().checked_div(1_000_000).unwrap();
    let required_margin = calculate_margin_requirement(position_value, leverage);
    
    require!(user_account.available_margin >= required_margin, ErrorCode::InsufficientCollateral);
    
    // Initialize position
    initialize_new_position(
        position,
        user_account,
        &market.key(),
        position_index,
        side,
        size,
        entry_price,
        required_margin,
        leverage,
        ctx.bumps.position,
    )?;
    
    // Update user account
    user_account.add_position()?;
    
    // Extract values before mutable borrow
    let total_collateral = user_account.total_collateral;
    let available_margin = user_account.available_margin;
    let used_margin = total_collateral.checked_sub(available_margin).unwrap();
    let new_used_margin = used_margin.checked_add(required_margin).unwrap();
    
    // Now call with extracted values
    user_account.update_available_margin(total_collateral, new_used_margin)?;
    
    msg!("Position opened: {:?} {} at {}", side, size, entry_price);
    Ok(())
}

/// Close a position with P&L calculation
pub fn close_position(ctx: Context<ClosePosition>) -> Result<()> {
    let position = &mut ctx.accounts.position;
    let market = &mut ctx.accounts.market;
    
    require!(position.size > 0, ErrorCode::PositionAlreadyClosed);
    
    // Calculate P&L using oracle price
    let current_price = market.get_oracle_price()?;
    let pnl = calculate_pnl(position.entry_price, current_price, position.size, position.side);

    // Update vAMM reserves (reverse the position)
    let position_value = (position.size * current_price) / 1000000;
    match position.side {
        PositionSide::Long => {
            market.base_reserve -= position.size;
            market.quote_reserve += position_value;
        },
        PositionSide::Short => {
            market.base_reserve += position.size;
            market.quote_reserve -= position_value;
        },
    }

    // Calculate total return
    let total_return = if pnl >= 0 {
        position.margin + pnl as u64
    } else {
        position.margin.saturating_sub((-pnl) as u64)
    };
    
    msg!("Position closed: PnL = {}, Total return = {}", pnl, total_return);
    
    // Mark position as closed
    position.size = 0;
    position.unrealized_pnl = pnl;
    
    Ok(())
}

/// Liquidate a position (called by keeper bots)
pub fn liquidate_position(ctx: Context<LiquidatePosition>) -> Result<()> {
    let position = &mut ctx.accounts.position;
    let market = &mut ctx.accounts.market;
    
    require!(position.size > 0, ErrorCode::PositionAlreadyClosed);
    
    // Calculate health factor using oracle price
    let current_price = market.get_oracle_price()?;
    let unrealized_pnl = calculate_pnl(position.entry_price, current_price, position.size, position.side);
    
    let equity = position.margin as i128 + unrealized_pnl as i128;
    let position_value = (position.size * current_price) / 1000000;
    let health_factor = (equity * 10000) / position_value as i128;
    
    require!(health_factor < market.maintenance_margin_ratio as i128, ErrorCode::PositionHealthy);
    
    // Execute liquidation
    msg!("Liquidating position: Health factor = {}%", health_factor / 100);
    
    // Transfer collateral to vault (simplified)
    // In production, this would involve proper token transfers
    
    // Mark position as liquidated
    position.size = 0;
    position.unrealized_pnl = unrealized_pnl;
    
    Ok(())
}

/// Open a position with cross-collateral support
pub fn open_position_cross_collateral(
    ctx: Context<OpenPositionCrossCollateral>,
    market_index: u16,
    size: u64,
    side: PositionSide,
    leverage: u8,
    collateral_accounts: Vec<Pubkey>,
) -> Result<()> {
    let market = &ctx.accounts.market;

    // Security validations
    require!(market.is_active, ErrorCode::MarketInactive);
    validate_leverage(leverage as u16, market.max_leverage)?;
    validate_position_size(size)?;
    require!(!collateral_accounts.is_empty(), ErrorCode::NoCollateralProvided);

    let position = &mut ctx.accounts.position;

    // Calculate required margin using oracle price
    let oracle_price = market.get_oracle_price()?;
    let position_value = size.checked_mul(oracle_price).unwrap().checked_div(1_000_000).unwrap();
    let required_margin = calculate_margin_requirement(position_value, leverage as u16);

    // Initialize position
    initialize_new_position(
        position,
        &ctx.accounts.user_account,
        &market.key(),
        market_index,
        side,
        size,
        oracle_price,
        required_margin,
        leverage as u16,
        ctx.bumps.position,
    )?;

    msg!("Cross-collateral position opened: {:?} {} at {}", side, size, oracle_price);
    Ok(())
}

/// Liquidate a cross-collateral position
pub fn liquidate_position_cross_collateral(ctx: Context<LiquidatePositionCrossCollateral>) -> Result<()> {
    let position = &mut ctx.accounts.position;
    let market = &mut ctx.accounts.market;
    
    require!(position.size > 0, ErrorCode::PositionAlreadyClosed);
    
    // Calculate health factor using oracle price
    let current_price = market.get_oracle_price()?;
    let unrealized_pnl = calculate_pnl(position.entry_price, current_price, position.size, position.side);
    
    let equity = position.margin as i128 + unrealized_pnl as i128;
    let position_value = (position.size * current_price) / 1000000;
    let health_factor = (equity * 10000) / position_value as i128;
    
    require!(health_factor < market.maintenance_margin_ratio as i128, ErrorCode::PositionHealthy);
    
    // Execute liquidation
    msg!("Liquidating cross-collateral position: Health factor = {}%", health_factor / 100);
    
    // Mark position as liquidated
    position.size = 0;
    position.unrealized_pnl = unrealized_pnl;
    
    Ok(())
}

/// Liquidate a position via keeper network
pub fn liquidate_position_keeper(
    ctx: Context<LiquidatePositionKeeper>,
    position_id: u64,
) -> Result<()> {
    let keeper_network = &mut ctx.accounts.keeper_network;
    let position = &mut ctx.accounts.position;
    let market = &mut ctx.accounts.market;
    
    require!(position.size > 0, ErrorCode::PositionAlreadyClosed);
    
    // Verify keeper authorization
    require!(keeper_network.is_authorized_keeper(&ctx.accounts.keeper.key()), ErrorCode::UnauthorizedKeeper);
    
    // Calculate health factor using oracle price
    let current_price = market.get_oracle_price()?;
    let unrealized_pnl = calculate_pnl(position.entry_price, current_price, position.size, position.side);
    
    let equity = position.margin as i128 + unrealized_pnl as i128;
    let position_value = (position.size * current_price) / 1000000;
    let health_factor = (equity * 10000) / position_value as i128;
    
    require!(health_factor < market.maintenance_margin_ratio as i128, ErrorCode::PositionHealthy);
    
    // Execute liquidation
    msg!("Keeper liquidating position {}: Health factor = {}%", position_id, health_factor / 100);
    
    // Update keeper network stats
    keeper_network.increment_liquidations(&ctx.accounts.keeper.key())?;
    
    // Mark position as liquidated
    position.size = 0;
    position.unrealized_pnl = unrealized_pnl;
    
    Ok(())
}

/// Context structs for position management instructions

#[derive(Accounts)]
#[instruction(position_index: u16)]
pub struct OpenPosition<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + Position::INIT_SPACE,
        seeds = [
            b"position",
            user_account.key().as_ref(),
            &position_index.to_le_bytes()
        ],
        bump
    )]
    pub position: Account<'info, Position>,

    #[account(
        mut,
        constraint = user_account.authority == user.key()
    )]
    pub user_account: Account<'info, UserAccount>,

    #[account(mut)]
    pub market: Account<'info, Market>,

    #[account(mut)]
    pub user: Signer<'info>,

    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ClosePosition<'info> {
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(
        mut,
        constraint = position.user == user.key()
    )]
    pub position: Account<'info, Position>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct LiquidatePosition<'info> {
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub position: Account<'info, Position>,
    
    #[account(mut)]
    pub liquidator: Signer<'info>,
    
    #[account(mut)]
    /// CHECK: This is the liquidation vault account
    pub vault: AccountInfo<'info>,
}

#[derive(Accounts)]
#[instruction(market_index: u16)]
pub struct OpenPositionCrossCollateral<'info> {
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(
        init,
        payer = user,
        space = 8 + Position::INIT_SPACE,
        seeds = [b"position", user.key().as_ref(), market.key().as_ref()],
        bump
    )]
    pub position: Account<'info, Position>,
    
    #[account(
        mut,
        constraint = user_account.authority == user.key()
    )]
    pub user_account: Account<'info, UserAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct LiquidatePositionCrossCollateral<'info> {
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub position: Account<'info, Position>,
    
    #[account(mut)]
    pub liquidator: Signer<'info>,
    
    #[account(mut)]
    /// CHECK: This is the liquidation vault account
    pub vault: AccountInfo<'info>,
}

#[derive(Accounts)]
#[instruction(position_id: u64)]
pub struct LiquidatePositionKeeper<'info> {
    #[account(mut)]
    pub keeper_network: Account<'info, crate::state::KeeperNetwork>,
    
    #[account(mut)]
    pub position: Account<'info, Position>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub keeper: Signer<'info>,
    
    #[account(mut)]
    /// CHECK: This is the liquidation vault account
    pub vault: AccountInfo<'info>,
}