use anchor_lang::prelude::*;
use anchor_lang::system_program::System;
use anchor_spl::token::{self, Token, TokenAccount, Mint, Transfer};
use anchor_spl::associated_token::AssociatedToken;

/// Token Operations Module
/// Following Solana Cookbook patterns for professional token management
/// https://solanacookbook.com/references/programs.html#token-program
/// 
/// Note: TokenVault and ProtocolSolVault structures moved to instructions/vault_management.rs
/// to eliminate duplication and improve organization.
    pub bump: u8,                 // PDA bump
}

impl ProtocolSolVault {
    pub const INIT_SPACE: usize = 8 + 8 + 1 + 1;
}

/// Initialize Token Vault Context
#[derive(Accounts)]
#[instruction(mint_address: Pubkey)]
pub struct InitializeTokenVault<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + TokenVault::INIT_SPACE,
        seeds = [b"vault", mint_address.as_ref()],
        bump
    )]
    pub vault: Account<'info, TokenVault>,
    
    #[account(
        init,
        payer = authority,
        associated_token::mint = mint,
        associated_token::authority = vault,
    )]
    pub vault_token_account: Account<'info, TokenAccount>,
    
    /// CHECK: This is the mint we're creating a vault for
    pub mint: Account<'info, Mint>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
}

/// Deposit Tokens Context
#[derive(Accounts)]
pub struct DepositTokens<'info> {
    #[account(
        mut,
        constraint = vault.is_active @ TokenError::VaultInactive
    )]
    pub vault: Account<'info, TokenVault>,
    
    #[account(
        mut,
        associated_token::mint = vault.mint,
        associated_token::authority = user,
    )]
    pub user_token_account: Account<'info, TokenAccount>,
    
    #[account(
        mut,
        associated_token::mint = vault.mint,
        associated_token::authority = vault,
    )]
    pub vault_token_account: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
}

/// Withdraw Tokens Context
#[derive(Accounts)]
pub struct WithdrawTokens<'info> {
    #[account(
        mut,
        constraint = vault.is_active @ TokenError::VaultInactive
    )]
    pub vault: Account<'info, TokenVault>,
    
    #[account(
        mut,
        associated_token::mint = vault.mint,
        associated_token::authority = vault,
    )]
    pub vault_token_account: Account<'info, TokenAccount>,
    
    #[account(
        mut,
        associated_token::mint = vault.mint,
        associated_token::authority = user,
    )]
    pub user_token_account: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
}

/// Create User Token Account Context
#[derive(Accounts)]
pub struct CreateUserTokenAccount<'info> {
    #[account(
        init,
        payer = user,
        associated_token::mint = mint,
        associated_token::authority = user,
    )]
    pub user_token_account: Account<'info, TokenAccount>,
    
    /// CHECK: This is the mint we're creating an account for
    pub mint: Account<'info, Mint>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    pub token_program: Program<'info, Token>,
    pub associated_token_program: Program<'info, AssociatedToken>,
    pub system_program: Program<'info, System>,
}

/// Initialize Protocol SOL Vault Context
#[derive(Accounts)]
pub struct InitializeProtocolSolVault<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + ProtocolSolVault::INIT_SPACE,
        seeds = [b"protocol_sol_vault"],
        bump
    )]
    pub protocol_vault: Account<'info, ProtocolSolVault>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

/// Withdraw Native SOL Context
#[derive(Accounts)]
pub struct WithdrawNativeSol<'info> {
    #[account(
        mut,
        seeds = [b"user_account", user.key().as_ref(), &[0u8, 0u8]], // Account index 0
        bump = user_account.bump,
    )]
    pub user_account: Account<'info, crate::user_accounts::UserAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    /// Protocol SOL vault PDA - this will hold the deposited SOL
    #[account(
        mut,
        seeds = [b"protocol_sol_vault"],
        bump,
    )]
    pub protocol_vault: Account<'info, crate::ProtocolSolVault>,
    
    /// SOL collateral account for the user - must exist for withdrawal
    #[account(
        mut,
        seeds = [b"collateral", user.key().as_ref(), b"SOL"],
        bump = collateral_account.bump,
    )]
    pub collateral_account: Account<'info, crate::CollateralAccount>,
    
    /// Pyth SOL/USD price feed account
    /// CHECK: Validated by Pyth SDK in get_sol_from_usd_value
    pub sol_usd_price_feed: AccountInfo<'info>,
    
    pub system_program: Program<'info, System>,
}

/// Close corrupted collateral account (EMERGENCY)
#[derive(Accounts)]
pub struct FixCorruptedCollateral<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    
    /// SOL collateral account to fix
    #[account(
        mut,
        seeds = [b"collateral", user.key().as_ref(), b"SOL"],
        bump = collateral_account.bump,
    )]
    pub collateral_account: Account<'info, crate::CollateralAccount>,
    
    /// Protocol SOL vault (to check actual balance)
    #[account(
        seeds = [b"protocol_sol_vault"],
        bump = protocol_vault.bump,
    )]
    pub protocol_vault: Account<'info, crate::ProtocolSolVault>,
}

#[derive(Accounts)]
pub struct CloseCollateralAccount<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    
    /// SOL collateral account - will be closed and rent returned
    #[account(
        mut,
        close = user,
        seeds = [b"collateral", user.key().as_ref(), b"SOL"],
        bump = collateral_account.bump,
    )]
    pub collateral_account: Account<'info, crate::CollateralAccount>,
}

/// Deposit Native SOL Context
#[derive(Accounts)]
pub struct DepositNativeSol<'info> {
    #[account(
        mut,
        seeds = [b"user_account", user.key().as_ref(), &[0u8, 0u8]], // Account index 0
        bump = user_account.bump,
    )]
    pub user_account: Account<'info, crate::user_accounts::UserAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    /// Protocol SOL vault PDA - this will hold the deposited SOL
    #[account(
        mut,
        seeds = [b"protocol_sol_vault"],
        bump = protocol_vault.bump,
    )]
    pub protocol_vault: Account<'info, crate::ProtocolSolVault>,
    
    /// SOL collateral account for the user - will be initialized if needed
    #[account(
        init_if_needed,
        payer = user,
        space = 8 + crate::CollateralAccount::INIT_SPACE,
        seeds = [b"collateral", user.key().as_ref(), b"SOL"],
        bump
    )]
    pub collateral_account: Account<'info, crate::CollateralAccount>,
    
    /// Pyth SOL/USD price feed account
    /// CHECK: Validated by Pyth SDK in get_usd_value_from_sol
    pub sol_usd_price_feed: AccountInfo<'info>,
    
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,
}

/// Token-specific error codes
#[error_code]
pub enum TokenError {
    #[msg("Invalid token amount")]
    InvalidAmount,
    #[msg("Vault is inactive")]
    VaultInactive,
    #[msg("Insufficient vault balance")]
    InsufficientVaultBalance,
    #[msg("Unauthorized token authority")]
    UnauthorizedTokenAuthority,
    #[msg("Unauthorized token user")]
    UnauthorizedTokenUser,
    #[msg("Token account not found")]
    TokenAccountNotFound,
    #[msg("Invalid mint address")]
    InvalidMintAddress,
}

/// Initialize a token vault for a specific asset
/// This creates a PDA that holds tokens for the protocol
pub fn initialize_token_vault(
    ctx: Context<InitializeTokenVault>,
    mint_address: Pubkey,
) -> Result<()> {
    let vault = &mut ctx.accounts.vault;
    
    // Initialize vault with mint address
    vault.mint = mint_address;
    vault.authority = ctx.accounts.authority.key();
    vault.total_deposits = 0;
    vault.total_withdrawals = 0;
    vault.is_active = true;
    vault.bump = ctx.bumps.vault;
    
    msg!("Token vault initialized for mint: {}", mint_address);
    Ok(())
}

/// Deposit tokens into the protocol vault
/// Following Solana Cookbook CPI patterns
pub fn deposit_tokens(
    ctx: Context<DepositTokens>,
    amount: u64,
) -> Result<()> {
    let vault = &mut ctx.accounts.vault;
    
    // Validate deposit amount
    require!(amount > 0, TokenError::InvalidAmount);
    require!(vault.is_active, TokenError::VaultInactive);
    
    // Transfer tokens from user to vault using CPI
    let cpi_accounts = Transfer {
        from: ctx.accounts.user_token_account.to_account_info(),
        to: ctx.accounts.vault_token_account.to_account_info(),
        authority: ctx.accounts.user.to_account_info(),
    };
    
    let cpi_program = ctx.accounts.token_program.to_account_info();
    let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
    
    token::transfer(cpi_ctx, amount)?;
    
    // Update vault statistics
    vault.total_deposits += amount;
    
    msg!("Deposited {} tokens to vault", amount);
    Ok(())
}

/// Withdraw tokens from the protocol vault
/// Following Solana Cookbook CPI patterns
pub fn withdraw_tokens(
    ctx: Context<WithdrawTokens>,
    amount: u64,
) -> Result<()> {
    let vault = &mut ctx.accounts.vault;
    
    // Validate withdrawal amount
    require!(amount > 0, TokenError::InvalidAmount);
    require!(vault.is_active, TokenError::VaultInactive);
    
    // Check vault balance
    let vault_balance = ctx.accounts.vault_token_account.amount;
    require!(vault_balance >= amount, TokenError::InsufficientVaultBalance);
    
    // Transfer tokens from vault to user using CPI
    let seeds = &[
        b"vault",
        vault.mint.as_ref(),
        &[vault.bump],
    ];
    let signer = &[&seeds[..]];
    
    let cpi_accounts = Transfer {
        from: ctx.accounts.vault_token_account.to_account_info(),
        to: ctx.accounts.user_token_account.to_account_info(),
        authority: vault.to_account_info(),
    };
    
    let cpi_program = ctx.accounts.token_program.to_account_info();
    let cpi_ctx = CpiContext::new_with_signer(cpi_program, cpi_accounts, signer);
    
    token::transfer(cpi_ctx, amount)?;
    
    // Update vault statistics
    vault.total_withdrawals += amount;
    
    msg!("Withdrew {} tokens from vault", amount);
    Ok(())
}

/// Create user token account if needed
/// Following Solana Cookbook ATA patterns
pub fn create_user_token_account(
    ctx: Context<CreateUserTokenAccount>,
) -> Result<()> {
    // The account is created automatically by the init_if_needed constraint
    // This function can be used for additional validation or setup
    
    msg!("User token account created or verified for mint: {}", ctx.accounts.mint.key());
    Ok(())
}

/// Initialize protocol SOL vault
/// This creates a PDA that holds native SOL for the protocol
pub fn initialize_protocol_sol_vault(ctx: Context<InitializeProtocolSolVault>) -> Result<()> {
    let vault = &mut ctx.accounts.protocol_vault;
    
    // Initialize vault with zero balance
    vault.total_deposits = 0;
    vault.total_withdrawals = 0;
    vault.is_active = true;
    vault.bump = ctx.bumps.protocol_vault;
    
    msg!("Protocol SOL vault initialized");
    Ok(())
}