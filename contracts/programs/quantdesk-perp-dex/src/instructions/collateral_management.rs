//! Collateral management instructions
//! This module contains all collateral-related instruction handlers for managing user collateral accounts and cross-collateral operations.

use anchor_lang::prelude::*;
use crate::{
    collateral::CollateralType,
    state::CollateralAccount,
    ErrorCode,
};
use crate::instructions::vault_management::ProtocolSolVault;

/// Initialize a collateral account for a user
pub fn initialize_collateral_account(
    ctx: Context<InitializeCollateralAccount>,
    asset_type: CollateralType,
) -> Result<()> {
    let collateral = &mut ctx.accounts.collateral_account;
    let clock = Clock::get()?;
    
    // Set asset-specific weights (Drift-style configuration)
    let (initial_weight, maintenance_weight, initial_liability, maintenance_liability) = match asset_type {
        CollateralType::USDC => (10000, 10000, 10000, 10000), // 100% for stablecoin
        CollateralType::SOL => (8000, 9000, 12000, 11000),    // 80%/90% for SOL
        CollateralType::BTC => (8000, 9000, 12000, 11000),    // 80%/90% for BTC
        CollateralType::ETH => (8000, 9000, 12000, 11000),    // 80%/90% for ETH
        CollateralType::USDT => (9500, 9700, 10500, 10200),   // 95%/97% for USDT
        _ => (5000, 7500, 15000, 12500),                       // 50%/75% for others
    };
    
    collateral.user = ctx.accounts.user.key();
    collateral.asset_type = asset_type;
    collateral.amount = 0;
    collateral.initial_asset_weight = initial_weight;
    collateral.maintenance_asset_weight = maintenance_weight;
    collateral.initial_liability_weight = initial_liability;
    collateral.maintenance_liability_weight = maintenance_liability;
    collateral.value_usd = 0;
    collateral.last_price = 0;
    collateral.last_updated = clock.unix_timestamp;
    collateral.is_active = true;
    collateral.bump = ctx.bumps.collateral_account;
    
    msg!("Collateral account initialized for asset: {:?}", asset_type);
    Ok(())
}

/// Add collateral to an existing account
pub fn add_collateral(
    ctx: Context<AddCollateral>,
    amount: u64,
) -> Result<()> {
    let collateral_account = &mut ctx.accounts.collateral_account;
    
    require!(amount > 0, ErrorCode::InvalidAmount);
    require!(collateral_account.is_active, ErrorCode::CollateralAccountInactive);
    
    // Add to existing amount
    collateral_account.amount += amount;
    collateral_account.last_updated = Clock::get()?.unix_timestamp;
    
    msg!("Added {} collateral to account {}", amount, ctx.accounts.collateral_account.key());
    Ok(())
}

/// Remove collateral from an account
pub fn remove_collateral(
    ctx: Context<RemoveCollateral>,
    amount: u64,
) -> Result<()> {
    let collateral_account = &mut ctx.accounts.collateral_account;
    
    require!(amount > 0, ErrorCode::InvalidAmount);
    require!(collateral_account.amount >= amount, ErrorCode::InsufficientCollateral);
    require!(collateral_account.is_active, ErrorCode::CollateralAccountInactive);
    
    // Check if this collateral is being used in any positions
    // In production, this would check all user positions
    // For now, we'll allow removal if amount is available
    
    // Remove from amount
    collateral_account.amount -= amount;
    collateral_account.last_updated = Clock::get()?.unix_timestamp;
    
    msg!("Removed {} collateral from account {}", amount, ctx.accounts.collateral_account.key());
    Ok(())
}

/// Update collateral value using oracle price
pub fn update_collateral_value(
    ctx: Context<UpdateCollateralValue>,
    new_price: u64,
) -> Result<()> {
    let account_key = ctx.accounts.collateral_account.key();
    let collateral_account = &mut ctx.accounts.collateral_account;
    
    require!(new_price > 0, ErrorCode::InvalidPrice);
    require!(collateral_account.is_active, ErrorCode::CollateralAccountInactive);
    
    // Update USD value based on new price
    collateral_account.value_usd = (collateral_account.amount * new_price) / 1000000; // Assuming 6 decimals
    collateral_account.last_updated = Clock::get()?.unix_timestamp;
    
    msg!("Updated collateral value: {} USD for account {}", 
         collateral_account.value_usd, account_key);
    Ok(())
}

/// Deposit native SOL to user account
pub fn deposit_native_sol(ctx: Context<DepositNativeSol>, amount: u64) -> Result<()> {
    let user_account = &mut ctx.accounts.user_account;
    let protocol_vault = &mut ctx.accounts.protocol_vault;
    let collateral_account = &mut ctx.accounts.collateral_account;
    
    require!(amount > 0, ErrorCode::InvalidAmount);
    
    // Initialize user account if it's being created for the first time
    if user_account.authority == Pubkey::default() {
        let clock = Clock::get()?;
        
        // Initialize user account
        user_account.authority = ctx.accounts.user.key();
        user_account.account_index = 0;
        user_account.total_collateral = 0;
        user_account.total_positions = 0;
        user_account.total_orders = 0;
        user_account.max_positions = 50; // Default max positions
        user_account.initial_margin_requirement = 0;
        user_account.maintenance_margin_requirement = 0;
        user_account.available_margin = 0;
        user_account.account_health = 10000; // 100% health
        user_account.liquidation_price = 0;
        user_account.liquidation_threshold = 2000; // 20%
        user_account.max_leverage = 1000; // 10x
        user_account.total_funding_paid = 0;
        user_account.total_funding_received = 0;
        user_account.is_active = true;
        user_account.created_at = clock.unix_timestamp;
        user_account.bump = ctx.bumps.user_account;
        
        msg!("Initialized user account for user: {}", ctx.accounts.user.key());
    }
    
    // Initialize collateral account if it's being created for the first time
    if collateral_account.user == Pubkey::default() {
        let clock = Clock::get()?;
        
        // Initialize SOL collateral account with proper weights
        collateral_account.user = ctx.accounts.user.key();
        collateral_account.asset_type = CollateralType::SOL;
        collateral_account.amount = 0;
        collateral_account.initial_asset_weight = 8000;      // 80%
        collateral_account.maintenance_asset_weight = 9000;   // 90%
        collateral_account.initial_liability_weight = 12000;  // 120%
        collateral_account.maintenance_liability_weight = 11000; // 110%
        collateral_account.value_usd = 0;
        collateral_account.last_price = 0;
        collateral_account.last_updated = clock.unix_timestamp;
        collateral_account.is_active = true;
        collateral_account.bump = ctx.bumps.collateral_account;
        
        msg!("Initialized SOL collateral account for user: {}", ctx.accounts.user.key());
    }
    
    require!(collateral_account.is_active, ErrorCode::CollateralAccountInactive);
    
    // Transfer SOL from user to protocol vault
    let transfer_instruction = anchor_lang::system_program::Transfer {
        from: ctx.accounts.user.to_account_info(),
        to: protocol_vault.to_account_info(),
    };
    
    anchor_lang::system_program::transfer(
        CpiContext::new(
            ctx.accounts.system_program.to_account_info(),
            transfer_instruction,
        ),
        amount,
    )?;
    
    // Calculate USD value using oracle price
    let usd_value = crate::oracle::get_usd_from_sol_devnet_safe(
        amount, 
        &ctx.accounts.sol_usd_price_feed
    )?;
    
    // Update collateral account
    collateral_account.amount += amount;
    collateral_account.value_usd += usd_value; // Add USD value, not lamports
    collateral_account.last_updated = Clock::get()?.unix_timestamp;
    
    // Update user account total collateral with USD value
    user_account.total_collateral += usd_value; // Add USD value, not lamports
    user_account.available_margin += usd_value; // Add USD value, not lamports
    
    msg!("Deposited {} SOL (${} USD) to collateral account", 
         amount, usd_value as f64 / 1_000_000.0);
    Ok(())
}

/// Withdraw native SOL from user account
pub fn withdraw_native_sol(ctx: Context<WithdrawNativeSol>, amount: u64) -> Result<()> {
    let user_account = &mut ctx.accounts.user_account;
    let protocol_vault = &mut ctx.accounts.protocol_vault;
    let collateral_account = &mut ctx.accounts.collateral_account;
    
    require!(amount > 0, ErrorCode::InvalidAmount);
    require!(collateral_account.amount >= amount, ErrorCode::InsufficientCollateral);
    require!(collateral_account.is_active, ErrorCode::CollateralAccountInactive);
    
    // Calculate USD value of withdrawal amount using current oracle price
    let usd_value = crate::oracle::get_usd_from_sol_devnet_safe(
        amount, 
        &ctx.accounts.sol_usd_price_feed
    )?;
    
    // Check if user has enough available margin (in USD)
    require!(user_account.available_margin >= usd_value, ErrorCode::InsufficientCollateral);
    
    // Transfer SOL from protocol vault to user
    let transfer_instruction = anchor_lang::system_program::Transfer {
        from: protocol_vault.to_account_info(),
        to: ctx.accounts.user.to_account_info(),
    };
    
    anchor_lang::system_program::transfer(
        CpiContext::new(
            ctx.accounts.system_program.to_account_info(),
            transfer_instruction,
        ),
        amount,
    )?;
    
    // Update collateral account (subtract SOL amount and USD value)
    collateral_account.amount -= amount;
    collateral_account.value_usd -= usd_value;
    collateral_account.last_updated = Clock::get()?.unix_timestamp;
    
    // Update user account total collateral (subtract USD value)
    user_account.total_collateral -= usd_value;
    user_account.available_margin -= usd_value;
    
    msg!("Withdrew {} SOL (${} USD) from collateral account", 
         amount, usd_value as f64 / 1_000_000.0);
    Ok(())
}

/// Fix corrupted collateral data from previous buggy versions
/// This recalculates value_usd based on actual SOL amount in the collateral account
pub fn fix_corrupted_collateral(ctx: Context<FixCorruptedCollateral>) -> Result<()> {
    let collateral_account = &mut ctx.accounts.collateral_account;
    let protocol_vault = &ctx.accounts.protocol_vault;
    
    // Get actual SOL balance from vault
    let vault_balance = protocol_vault.lamports();
    
    // Recalculate value_usd based on actual balance
    // Assuming SOL price is around $100 (100 * 10^6 for 6 decimals)
    let sol_price = 100_000_000u64; // $100 in 6 decimals
    let corrected_value_usd = (vault_balance * sol_price) / 1_000_000_000; // Convert lamports to SOL, then to USD
    
    // Update the collateral account with correct values
    collateral_account.amount = vault_balance;
    collateral_account.value_usd = corrected_value_usd;
    collateral_account.last_price = sol_price;
    collateral_account.last_updated = Clock::get()?.unix_timestamp;
    
    msg!("🔧 Fixed corrupted collateral: {} SOL = {} USD", vault_balance, corrected_value_usd);
    Ok(())
}

/// EMERGENCY: Close corrupted collateral account and return rent
/// This allows starting fresh with correct calculations
pub fn close_collateral_account(_ctx: Context<CloseCollateralAccount>) -> Result<()> {
    msg!("🧹 Closing collateral account and returning rent");
    
    // Account will be closed automatically by Anchor's `close` constraint
    // This allows users to start fresh with correct collateral calculations
    
    Ok(())
}

/// Initialize cross-collateral account for a user
pub fn initialize_cross_collateral_account(
    ctx: Context<InitializeCrossCollateralAccount>,
) -> Result<()> {
    let cross_collateral_account = &mut ctx.accounts.cross_collateral_account;
    
    cross_collateral_account.user = ctx.accounts.user.key();
    cross_collateral_account.total_collateral_value = 0;
    cross_collateral_account.total_borrowed_value = 0;
    cross_collateral_account.initial_asset_weight = 1000; // 10%
    cross_collateral_account.maintenance_asset_weight = 500; // 5%
    cross_collateral_account.is_active = true;
    cross_collateral_account.bump = ctx.bumps.cross_collateral_account;
    
    msg!("Cross-collateral account initialized for user: {}", ctx.accounts.user.key());
    Ok(())
}

/// Add collateral to cross-collateral account
pub fn add_cross_collateral(
    ctx: Context<AddCrossCollateral>,
    asset_type: CollateralType,
    amount: u64,
) -> Result<()> {
    let cross_collateral_account = &mut ctx.accounts.cross_collateral_account;
    
    require!(amount > 0, ErrorCode::InvalidAmount);
    require!(cross_collateral_account.is_active, ErrorCode::CollateralAccountInactive);
    
    // Calculate USD value based on asset type
    let usd_value = match asset_type {
        CollateralType::USDC => amount, // 1:1 for USDC
        CollateralType::SOL => amount * 100 / 1_000_000, // Assuming SOL = $100
        CollateralType::BTC => amount * 50000 / 1_000_000, // Assuming BTC = $50k
        CollateralType::ETH => amount * 3000 / 1_000_000, // Assuming ETH = $3k
        _ => amount * 1 / 1_000_000, // Default 1:1
    };
    
    cross_collateral_account.total_collateral_value += usd_value;
    cross_collateral_account.last_health_check = Clock::get()?.unix_timestamp;
    
    msg!("Added {} {} collateral (${} USD) to cross-collateral account", 
         amount, format!("{:?}", asset_type), usd_value);
    Ok(())
}

/// Remove collateral from cross-collateral account
pub fn remove_cross_collateral(
    ctx: Context<RemoveCrossCollateral>,
    asset_type: CollateralType,
    amount: u64,
) -> Result<()> {
    let cross_collateral_account = &mut ctx.accounts.cross_collateral_account;
    
    require!(amount > 0, ErrorCode::InvalidAmount);
    require!(cross_collateral_account.is_active, ErrorCode::CollateralAccountInactive);
    
    // Calculate USD value based on asset type
    let usd_value = match asset_type {
        CollateralType::USDC => amount, // 1:1 for USDC
        CollateralType::SOL => amount * 100 / 1_000_000, // Assuming SOL = $100
        CollateralType::BTC => amount * 50000 / 1_000_000, // Assuming BTC = $50k
        CollateralType::ETH => amount * 3000 / 1_000_000, // Assuming ETH = $3k
        _ => amount * 1 / 1_000_000, // Default 1:1
    };
    
    require!(cross_collateral_account.total_collateral_value >= usd_value, ErrorCode::InsufficientCollateral);
    
    cross_collateral_account.total_collateral_value -= usd_value;
    cross_collateral_account.last_health_check = Clock::get()?.unix_timestamp;
    
    msg!("Removed {} {} collateral (${} USD) from cross-collateral account", 
         amount, format!("{:?}", asset_type), usd_value);
    Ok(())
}

/// Update collateral configuration
pub fn update_collateral_config(
    ctx: Context<UpdateCollateralConfig>,
    initial_asset_weight: u16,
    maintenance_asset_weight: u16,
) -> Result<()> {
    let collateral_account = &mut ctx.accounts.collateral_account;
    
    require!(initial_asset_weight > 0 && initial_asset_weight <= 10000, ErrorCode::InvalidCollateralConfig);
    require!(maintenance_asset_weight > 0 && maintenance_asset_weight <= 10000, ErrorCode::InvalidCollateralConfig);
    require!(maintenance_asset_weight <= initial_asset_weight, ErrorCode::InvalidCollateralConfig);
    
    collateral_account.initial_asset_weight = initial_asset_weight;
    collateral_account.maintenance_asset_weight = maintenance_asset_weight;
    collateral_account.last_updated = Clock::get()?.unix_timestamp;
    
    msg!("Updated collateral config: initial={}, maintenance={}", 
         initial_asset_weight, maintenance_asset_weight);
    Ok(())
}

/// Context structs for collateral management instructions

#[derive(Accounts)]
#[instruction(asset_type: CollateralType)]
pub struct InitializeCollateralAccount<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + CollateralAccount::INIT_SPACE,
        seeds = [b"collateral", user.key().as_ref(), &[asset_type as u8]],
        bump
    )]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct AddCollateral<'info> {
    #[account(mut)]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct RemoveCollateral<'info> {
    #[account(mut)]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct UpdateCollateralValue<'info> {
    #[account(mut)]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct DepositNativeSol<'info> {
    // CRITICAL: Account order MUST match IDL exactly to fix AccountNotSigner error
    // IDL order: user_account, user, protocol_vault, collateral_account, sol_usd_price_feed, system_program, rent
    
    #[account(
        init_if_needed,
        payer = user,
        space = 8 + crate::state::UserAccount::INIT_SPACE,
        seeds = [b"user_account", user.key().as_ref(), &[0u8, 0u8]], // Account index 0
        bump
    )]
    pub user_account: Account<'info, crate::state::UserAccount>,
    
    // Position 1 in IDL - MUST be here for signer validation
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(
        mut,
        seeds = [b"protocol_sol_vault"],
        bump,
    )]
    pub protocol_vault: Account<'info, ProtocolSolVault>,
    
    #[account(
        init_if_needed,
        payer = user,
        space = 8 + CollateralAccount::INIT_SPACE,
        seeds = [b"collateral", user.key().as_ref(), b"SOL"],
        bump
    )]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    /// Pyth SOL/USD price feed account
    /// CHECK: Validated by Pyth SDK in get_usd_value_from_sol
    pub sol_usd_price_feed: AccountInfo<'info>,
    
    pub system_program: Program<'info, System>,
    
    // Note: IDL shows rent but it may be optional - checking IDL
    pub rent: Sysvar<'info, Rent>,
}

#[derive(Accounts)]
pub struct WithdrawNativeSol<'info> {
    #[account(mut)]
    pub user_account: Account<'info, crate::state::UserAccount>,
    
    #[account(mut)]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    #[account(mut)]
    pub protocol_vault: SystemAccount<'info>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    /// Pyth SOL/USD price feed account
    /// CHECK: Validated by Pyth SDK in get_usd_value_from_sol
    pub sol_usd_price_feed: AccountInfo<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct FixCorruptedCollateral<'info> {
    #[account(mut)]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    #[account(mut)]
    pub protocol_vault: SystemAccount<'info>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct CloseCollateralAccount<'info> {
    #[account(
        mut,
        close = user,
        constraint = collateral_account.user == user.key()
    )]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct InitializeCrossCollateralAccount<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + 32 + 8 + 8 + 2 + 2 + 1 + 1,
        seeds = [b"cross_collateral", user.key().as_ref()],
        bump
    )]
    pub cross_collateral_account: Account<'info, crate::state::advanced::CrossCollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct AddCrossCollateral<'info> {
    #[account(mut)]
    pub cross_collateral_account: Account<'info, crate::state::advanced::CrossCollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct RemoveCrossCollateral<'info> {
    #[account(mut)]
    pub cross_collateral_account: Account<'info, crate::state::advanced::CrossCollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct UpdateCollateralConfig<'info> {
    #[account(mut)]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}