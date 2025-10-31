use anchor_lang::prelude::*;
use crate::{
    state::Market,
    state::{
        protocol::ProgramState,
        advanced::{CircuitBreaker, JitProvider, MarketMakerVault, PointsSystem},
    },
};

/// Remaining Context Structures Module
/// Contains context structures that don't fit into other instruction modules

// Admin Context Structures
#[derive(Accounts)]
pub struct UpdateWhitelist<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

// ===== ADVANCED FEATURE CONTEXT STRUCTURES =====

#[derive(Accounts)]
pub struct TriggerCircuitBreaker<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + CircuitBreaker::INIT_SPACE,
        seeds = [b"circuit_breaker"],
        bump
    )]
    pub circuit_breaker: Account<'info, CircuitBreaker>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ResetCircuitBreaker<'info> {
    #[account(mut)]
    pub circuit_breaker: Account<'info, CircuitBreaker>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct ProvideJitLiquidity<'info> {
    #[account(
        init,
        payer = provider,
        space = 8 + JitProvider::INIT_SPACE,
        seeds = [b"jit_provider", provider.key().as_ref()],
        bump
    )]
    pub jit_provider: Account<'info, JitProvider>,
    
    #[account(mut)]
    pub provider: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ExecuteJitOrder<'info> {
    #[account(mut)]
    pub jit_provider: Account<'info, JitProvider>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct CreateMarketMakerVault<'info> {
    #[account(
        init,
        payer = creator,
        space = 8 + MarketMakerVault::INIT_SPACE,
        seeds = [b"mm_vault", creator.key().as_ref()],
        bump
    )]
    pub vault: Account<'info, MarketMakerVault>,
    
    #[account(mut)]
    pub creator: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct InitializePointsSystem<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + PointsSystem::INIT_SPACE,
        seeds = [b"points_system"],
        bump
    )]
    pub points_system: Account<'info, PointsSystem>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct UpdateUserPoints<'info> {
    #[account(mut)]
    pub points_system: Account<'info, PointsSystem>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}
