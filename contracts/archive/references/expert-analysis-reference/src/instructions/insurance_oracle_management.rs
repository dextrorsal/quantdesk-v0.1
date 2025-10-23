use anchor_lang::prelude::*;

/// Insurance & Oracle Management Instructions Module
/// This module handles insurance fund operations and oracle feed management
/// Following Solana/Anchor best practices for modular architecture

/// Insurance Fund Account Structure
#[account]
pub struct InsuranceFund {
    pub total_deposits: u64,
    pub total_withdrawals: u64,
    pub utilization_rate: u16,
    pub max_utilization: u16,
    pub is_active: bool,
    pub bump: u8,
}

impl InsuranceFund {
    pub const INIT_SPACE: usize = 8 + 8 + 2 + 2 + 1 + 1;
}

/// Fee Collector Account Structure
#[account]
pub struct FeeCollector {
    pub trading_fees_collected: u64,
    pub funding_fees_collected: u64,
    pub maker_fee_rate: u16,
    pub taker_fee_rate: u16,
    pub funding_rate_cap: i64,
    pub funding_rate_floor: i64,
    pub bump: u8,
}

impl FeeCollector {
    pub const INIT_SPACE: usize = 8 + 8 + 2 + 2 + 8 + 8 + 1;
}

/// Oracle Feed Type Enum
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq, Copy)]
pub enum OracleFeedType {
    Pyth,
    Switchboard,
    Chainlink,
    Custom,
}

/// Oracle Feed Structure
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq)]
pub struct OracleFeed {
    pub feed_type: OracleFeedType,
    pub feed_account: Pubkey,
    pub weight: u8,
    pub is_active: bool,
}

/// Oracle Manager Account Structure
#[account]
pub struct OracleManager {
    pub feeds: Vec<OracleFeed>,
    pub weights: Vec<u8>,
    pub total_weight: u8,
    pub is_active: bool,
    pub bump: u8,
}

impl OracleManager {
    pub const INIT_SPACE: usize = 4 + 32 + 4 + 8 + 2 + 8 + 1; // Vec<OracleFeed> + Vec<u8> + other fields
}

/// Initialize Insurance Fund Context
#[derive(Accounts)]
pub struct InitializeInsuranceFund<'info> {
    #[account(
        init,
        payer = authority,
        space = 8 + InsuranceFund::INIT_SPACE,
        seeds = [b"insurance_fund"],
        bump
    )]
    pub insurance_fund: Account<'info, InsuranceFund>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

/// Deposit Insurance Fund Context
#[derive(Accounts)]
pub struct DepositInsuranceFund<'info> {
    #[account(
        mut,
        constraint = insurance_fund.is_active @ InsuranceOracleError::InvalidInsuranceFundOperation
    )]
    pub insurance_fund: Account<'info, InsuranceFund>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

/// Withdraw Insurance Fund Context
#[derive(Accounts)]
pub struct WithdrawInsuranceFund<'info> {
    #[account(
        mut,
        constraint = insurance_fund.is_active @ InsuranceOracleError::InvalidInsuranceFundOperation
    )]
    pub insurance_fund: Account<'info, InsuranceFund>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

/// Distribute Fees Context
#[derive(Accounts)]
pub struct DistributeFees<'info> {
    #[account(mut)]
    pub fee_collector: Account<'info, FeeCollector>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

/// Add Oracle Feed Context
#[derive(Accounts)]
pub struct AddOracleFeed<'info> {
    #[account(mut)]
    pub oracle_manager: Account<'info, OracleManager>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

/// Remove Oracle Feed Context
#[derive(Accounts)]
pub struct RemoveOracleFeed<'info> {
    #[account(mut)]
    pub oracle_manager: Account<'info, OracleManager>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

/// Update Oracle Weights Context
#[derive(Accounts)]
pub struct UpdateOracleWeights<'info> {
    #[account(mut)]
    pub oracle_manager: Account<'info, OracleManager>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

/// Initialize insurance fund with initial deposit
pub fn initialize_insurance_fund(
    ctx: Context<InitializeInsuranceFund>,
    initial_deposit: u64,
) -> Result<()> {
    require!(initial_deposit > 0, InsuranceOracleError::InvalidInsuranceFundOperation);
    
    let insurance_fund = &mut ctx.accounts.insurance_fund;
    insurance_fund.total_deposits = initial_deposit;
    insurance_fund.total_withdrawals = 0;
    insurance_fund.utilization_rate = 0;
    insurance_fund.max_utilization = 8000; // 80% max utilization
    insurance_fund.is_active = true;
    insurance_fund.bump = ctx.bumps.insurance_fund;
    
    msg!("Insurance fund initialized with {} deposit", initial_deposit);
    Ok(())
}

/// Deposit funds to insurance pool
pub fn deposit_insurance_fund(
    ctx: Context<DepositInsuranceFund>,
    amount: u64,
) -> Result<()> {
    require!(amount > 0, InsuranceOracleError::InvalidInsuranceFundOperation);
    require!(ctx.accounts.insurance_fund.is_active, InsuranceOracleError::InvalidInsuranceFundOperation);
    
    let insurance_fund = &mut ctx.accounts.insurance_fund;
    insurance_fund.total_deposits += amount;
    
    msg!("Deposited {} to insurance fund", amount);
    Ok(())
}

/// Withdraw funds from insurance pool (admin only)
pub fn withdraw_insurance_fund(
    ctx: Context<WithdrawInsuranceFund>,
    amount: u64,
) -> Result<()> {
    require!(amount > 0, InsuranceOracleError::InvalidInsuranceFundOperation);
    require!(ctx.accounts.insurance_fund.is_active, InsuranceOracleError::InvalidInsuranceFundOperation);
    
    let insurance_fund = &mut ctx.accounts.insurance_fund;
    let available_balance = insurance_fund.total_deposits - insurance_fund.total_withdrawals;
    require!(available_balance >= amount, InsuranceOracleError::InsufficientInsuranceFundBalance);
    
    let current_utilization = (insurance_fund.total_withdrawals + amount) * 10000 / insurance_fund.total_deposits;
    require!(current_utilization < insurance_fund.max_utilization.into(), InsuranceOracleError::InvalidInsuranceFundOperation);
    
    insurance_fund.total_withdrawals += amount;
    
    msg!("Withdrew {} from insurance fund", amount);
    Ok(())
}

/// Distribute fees to stakeholders
pub fn distribute_fees(
    ctx: Context<DistributeFees>,
    amount: u64,
) -> Result<()> {
    require!(amount > 0, InsuranceOracleError::InvalidFeeParameters);
    
    let fee_collector = &mut ctx.accounts.fee_collector;
    let total_fees = fee_collector.trading_fees_collected + fee_collector.funding_fees_collected;
    require!(amount <= total_fees, InsuranceOracleError::InvalidFeeParameters);
    
    msg!("Distributed {} fees to stakeholders", amount);
    Ok(())
}

/// Add new oracle feed
pub fn add_oracle_feed(
    ctx: Context<AddOracleFeed>,
    feed_type: OracleFeedType,
    weight: u8,
) -> Result<()> {
    require!(weight > 0 && weight <= 100, InsuranceOracleError::InvalidOracleWeight);
    
    let oracle_manager = &mut ctx.accounts.oracle_manager;
    let new_feed = OracleFeed {
        feed_type,
        feed_account: Pubkey::default(), // Will be set by caller
        weight,
        is_active: true,
    };
    
    oracle_manager.feeds.push(new_feed);
    oracle_manager.weights.push(weight);
    
    msg!("Added oracle feed: {:?} with weight {}", feed_type, weight);
    Ok(())
}

/// Remove oracle feed
pub fn remove_oracle_feed(
    ctx: Context<RemoveOracleFeed>,
    feed_index: u8,
) -> Result<()> {
    require!(feed_index < ctx.accounts.oracle_manager.feeds.len() as u8, InsuranceOracleError::OracleFeedNotFound);
    
    let oracle_manager = &mut ctx.accounts.oracle_manager;
    oracle_manager.feeds.remove(feed_index as usize);
    oracle_manager.weights.remove(feed_index as usize);
    
    msg!("Removed oracle feed at index {}", feed_index);
    Ok(())
}

/// Update oracle feed weights
pub fn update_oracle_weights(
    ctx: Context<UpdateOracleWeights>,
    weights: Vec<u8>,
) -> Result<()> {
    require!(weights.len() == ctx.accounts.oracle_manager.feeds.len(), InsuranceOracleError::InvalidOracleWeight);
    
    let total_weight: u8 = weights.iter().sum();
    require!(total_weight == 100, InsuranceOracleError::InvalidOracleWeight);
    
    for weight in &weights {
        require!(*weight > 0 && *weight <= 100, InsuranceOracleError::InvalidOracleWeight);
    }
    
    let oracle_manager = &mut ctx.accounts.oracle_manager;
    oracle_manager.weights = weights;
    
    msg!("Updated oracle weights");
    Ok(())
}

#[error_code]
pub enum InsuranceOracleError {
    #[msg("Invalid insurance fund operation")]
    InvalidInsuranceFundOperation,
    
    #[msg("Insufficient insurance fund balance")]
    InsufficientInsuranceFundBalance,
    
    #[msg("Invalid fee parameters")]
    InvalidFeeParameters,
    
    #[msg("Invalid oracle weight")]
    InvalidOracleWeight,
    
    #[msg("Oracle feed not found")]
    OracleFeedNotFound,
}
