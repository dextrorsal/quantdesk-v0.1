use anchor_lang::prelude::*;
use crate::state::UserAccount;

/// User Account Management Instructions Module
/// This module handles user account creation, updates, and management

/// Create User Account Context
#[derive(Accounts)]
#[instruction(account_index: u16)]
pub struct CreateUserAccount<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + UserAccount::INIT_SPACE,
        seeds = [b"user_account", authority.key().as_ref(), &account_index.to_le_bytes()],
        bump
    )]
    pub user_account: Account<'info, UserAccount>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

/// Update User Account Context
#[derive(Accounts)]
pub struct UpdateUserAccount<'info> {
    #[account(
        mut,
        constraint = user_account.authority == authority.key()
    )]
    pub user_account: Account<'info, UserAccount>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

/// Close User Account Context
#[derive(Accounts)]
pub struct CloseUserAccount<'info> {
    #[account(
        mut,
        close = authority,
        constraint = user_account.authority == authority.key(),
        constraint = user_account.total_positions == 0,
        constraint = user_account.total_orders == 0
    )]
    pub user_account: Account<'info, UserAccount>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

/// User action types for permission checking
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub enum UserAction {
    Deposit,
    Withdraw,
    Trade,
    CreatePosition,
    ClosePosition,
}

/// Create a new user account
pub fn create_user_account(
    ctx: Context<CreateUserAccount>,
    account_index: u16,
) -> Result<()> {
    let user_account = &mut ctx.accounts.user_account;
    
    // Initialize the user account
    user_account.initialize(
        ctx.accounts.authority.key(),
        account_index,
        ctx.bumps.user_account,
    )?;
    
    msg!("User account created: {} for user {}", 
         account_index, ctx.accounts.authority.key());
    Ok(())
}

/// Update user account (called when positions/orders change)
pub fn update_user_account(
    ctx: Context<UpdateUserAccount>,
    total_collateral: Option<u64>,
    total_positions: Option<u16>,
    total_orders: Option<u16>,
    account_health: Option<u16>,
    liquidation_price: Option<u64>,
) -> Result<()> {
    let user_account = &mut ctx.accounts.user_account;
    
    // Update fields if provided
    if let Some(collateral) = total_collateral {
        user_account.total_collateral = collateral;
    }
    
    if let Some(positions) = total_positions {
        user_account.total_positions = positions;
    }
    
    if let Some(orders) = total_orders {
        user_account.total_orders = orders;
    }
    
    if let Some(health) = account_health {
        user_account.update_account_health(health)?;
    }
    
    if let Some(price) = liquidation_price {
        user_account.update_liquidation_price(price)?;
    }
    
    // Always update activity timestamp
    user_account.update_activity()?;
    
    msg!("User account updated: {}", ctx.accounts.user_account.key());
    Ok(())
}

/// Close user account (only if no positions/orders)
pub fn close_user_account(ctx: Context<CloseUserAccount>) -> Result<()> {
    let user_account = &mut ctx.accounts.user_account;
    
    // Deactivate the account
    user_account.deactivate()?;
    
    msg!("User account closed: {}", ctx.accounts.user_account.key());
    Ok(())
}

/// Check if user can perform specific actions
pub fn check_user_permissions(
    ctx: Context<UpdateUserAccount>,
    action: UserAction,
) -> Result<()> {
    let user_account = &ctx.accounts.user_account;
    
    match action {
        UserAction::Deposit => {
            require!(user_account.can_deposit(), crate::errors::ErrorCode::AccountInactive);
        },
        UserAction::Withdraw => {
            require!(user_account.can_withdraw(), crate::errors::ErrorCode::AccountInactive);
        },
        UserAction::Trade => {
            require!(user_account.can_trade(), crate::errors::ErrorCode::AccountInactive);
        },
        UserAction::CreatePosition => {
            require!(user_account.can_trade(), crate::errors::ErrorCode::AccountInactive);
        },
        UserAction::ClosePosition => {
            require!(user_account.is_active, crate::errors::ErrorCode::AccountInactive);
        },
    }
    
    Ok(())
}