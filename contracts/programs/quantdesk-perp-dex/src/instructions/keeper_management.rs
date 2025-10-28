use anchor_lang::prelude::*;
use crate::state::{KeeperNetwork, KeeperInfo};

/// Keeper Management Module
/// Handles keeper registration and performance tracking for the decentralized keeper network

/// Register Keeper Context
#[derive(Accounts)]
pub struct RegisterKeeper<'info> {
    #[account(
        init,
        payer = keeper,
        space = 8 + KeeperNetwork::INIT_SPACE,
        seeds = [b"keeper_network"],
        bump
    )]
    pub keeper_network: Account<'info, KeeperNetwork>,
    
    #[account(mut)]
    pub keeper: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

/// Update Keeper Performance Context
#[derive(Accounts)]
pub struct UpdateKeeperPerformance<'info> {
    #[account(mut)]
    pub keeper_network: Account<'info, KeeperNetwork>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

/// Keeper Management Error Codes
#[error_code]
pub enum KeeperError {
    #[msg("Insufficient keeper stake")]
    InsufficientKeeperStake,
    #[msg("Invalid performance score")]
    InvalidPerformanceScore,
    #[msg("Keeper not registered")]
    KeeperNotRegistered,
    #[msg("Keeper already registered")]
    KeeperAlreadyRegistered,
    #[msg("Unauthorized keeper operation")]
    UnauthorizedKeeperOperation,
}

/// Register a new keeper in the network
pub fn register_keeper(
    ctx: Context<RegisterKeeper>,
    stake_amount: u64,
) -> Result<()> {
    require!(stake_amount >= 1000000000, KeeperError::InsufficientKeeperStake); // Min 1 SOL
    
    let keeper_network = &mut ctx.accounts.keeper_network;
    let keeper_info = KeeperInfo {
        keeper_pubkey: ctx.accounts.keeper.key(),
        stake_amount,
        performance_score: 1000, // Start with perfect score
        is_active: true,
        total_liquidations: 0,
        total_rewards_earned: 0,
        last_activity: Clock::get()?.unix_timestamp,
    };
    
    // Check if keeper is already registered
    require!(
        !keeper_network.keepers.iter().any(|k| k.keeper_pubkey == keeper_info.keeper_pubkey),
        KeeperError::KeeperAlreadyRegistered
    );
    
    keeper_network.keepers.push(keeper_info);
    keeper_network.total_stake += stake_amount;
    keeper_network.bump = ctx.bumps.keeper_network;
    
    msg!("Keeper registered with stake: {} lamports", stake_amount);
    Ok(())
}

/// Update keeper performance score
pub fn update_keeper_performance(
    ctx: Context<UpdateKeeperPerformance>,
    keeper_pubkey: Pubkey,
    performance_score: u16,
) -> Result<()> {
    require!(performance_score <= 1000, KeeperError::InvalidPerformanceScore);
    
    let keeper_network = &mut ctx.accounts.keeper_network;
    let keeper_index = keeper_network.keepers.iter()
        .position(|k| k.keeper_pubkey == keeper_pubkey)
        .ok_or(KeeperError::KeeperNotRegistered)?;
    
    keeper_network.keepers[keeper_index].performance_score = performance_score;
    keeper_network.keepers[keeper_index].last_activity = Clock::get()?.unix_timestamp;
    
    msg!("Keeper performance updated: {} -> {}", keeper_pubkey, performance_score);
    Ok(())
}
