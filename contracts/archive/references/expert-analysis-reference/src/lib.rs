use anchor_lang::prelude::*;

mod user_accounts;
mod pda_utils;
mod markets;
mod collateral;
mod oracle;
mod utils;
mod margin;
mod events;
mod errors;
mod instructions;
mod state;
mod security;
mod security_tests;

use user_accounts::*;
use collateral::CollateralType;
use errors::ErrorCode;
// State structures moved to state/ modules
use state::position::{PositionSide, PositionStatus};
use state::order::{OrderType, OrderStatus, TimeInForce};
use instructions::user_account_management::UserAction;

use instructions::{
    market_management::*,
    position_management::*,
    order_management::*,
    collateral_management::*,
    user_account_management::*,
    insurance_oracle_management::*,
    vault_management::*,
    cross_program::*,
    keeper_management::*,
    advanced_orders::*,
    remaining_contexts::*,
    security_management::*,
};

declare_id!("C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw");

#[program]
pub mod quantdesk_perp_dex {
    use super::*;

    pub fn initialize_market(
        ctx: Context<InitializeMarket>,
        base_asset: String,
        quote_asset: String,
        initial_price: u64,
        max_leverage: u8,
        initial_margin_ratio: u16,
        maintenance_margin_ratio: u16,
    ) -> Result<()> {
        instructions::market_management::initialize_market(
            ctx, base_asset, quote_asset, initial_price, max_leverage, 
            initial_margin_ratio, maintenance_margin_ratio
        )
    }

    pub fn update_oracle_price(ctx: Context<UpdateOraclePrice>, new_price: u64) -> Result<()> {
        instructions::market_management::update_oracle_price(ctx, new_price)
    }

    pub fn settle_funding(ctx: Context<SettleFunding>) -> Result<()> {
        instructions::market_management::settle_funding(ctx)
    }

    // Position Management - CRITICAL Trading Functions
    pub fn open_position(
        ctx: Context<OpenPosition>,
        position_index: u16,
        side: PositionSide,
        size: u64,
        leverage: u16,
        entry_price: u64,
    ) -> Result<()> {
        instructions::position_management::open_position(
            ctx, position_index, side, size, leverage, entry_price
        )
    }

    pub fn close_position(ctx: Context<ClosePosition>) -> Result<()> {
        instructions::position_management::close_position(ctx)
    }

    // Order Management - CRITICAL Trading Functions
    pub fn place_order(
        ctx: Context<PlaceOrder>,
        order_type: OrderType,
        side: PositionSide,
        size: u64,
        price: u64,
        stop_price: u64,
        trailing_distance: u64,
        leverage: u8,
        expires_at: i64,
        hidden_size: u64,
        display_size: u64,
        time_in_force: TimeInForce,
        target_price: u64,
        twap_duration: u64,
        twap_interval: u64,
    ) -> Result<()> {
        instructions::order_management::place_order(
            ctx, order_type, side, size, price, stop_price, trailing_distance,
            leverage, expires_at, hidden_size, display_size, time_in_force,
            target_price, twap_duration, twap_interval
        )
    }

    pub fn cancel_order(ctx: Context<CancelOrder>) -> Result<()> {
        instructions::order_management::cancel_order(ctx)
    }

    // Position Management - IMPORTANT Functions
    pub fn liquidate_position(ctx: Context<LiquidatePosition>) -> Result<()> {
        instructions::position_management::liquidate_position(ctx)
    }

    pub fn open_position_cross_collateral(
        ctx: Context<OpenPositionCrossCollateral>,
        market_index: u16,
        size: u64,
        side: PositionSide,
        leverage: u8,
        collateral_accounts: Vec<Pubkey>,
    ) -> Result<()> {
        instructions::position_management::open_position_cross_collateral(
            ctx, market_index, size, side, leverage, collateral_accounts
        )
    }

    pub fn liquidate_position_cross_collateral(ctx: Context<LiquidatePositionCrossCollateral>) -> Result<()> {
        instructions::position_management::liquidate_position_cross_collateral(ctx)
    }

    pub fn liquidate_position_keeper(
        ctx: Context<LiquidatePositionKeeper>,
        position_id: u64,
    ) -> Result<()> {
        instructions::position_management::liquidate_position_keeper(ctx, position_id)
    }

    // Order Management - IMPORTANT Functions
    pub fn execute_conditional_order(ctx: Context<ExecuteConditionalOrder>) -> Result<()> {
        instructions::order_management::execute_conditional_order(ctx)
    }

    // Order Management - ADVANCED Functions (Key ones only)
    pub fn place_oco_order(
        ctx: Context<PlaceOcoOrder>,
        size: u64,
        entry_price: u64,
        stop_loss_price: u64,
        take_profit_price: u64,
    ) -> Result<()> {
        instructions::order_management::place_oco_order(
            ctx, size, entry_price, stop_loss_price, take_profit_price
        )
    }

    pub fn place_iceberg_order(
        ctx: Context<PlaceIcebergOrder>, 
        total_size: u64,
        display_size: u64,
        price: u64,
        side: PositionSide,
        leverage: u8,
    ) -> Result<()> {
        instructions::order_management::place_iceberg_order(
            ctx, total_size, display_size, price, side, leverage
        )
    }

    pub fn place_twap_order(
        ctx: Context<PlaceTwapOrder>,
        total_size: u64,
        duration_seconds: u64,
        interval_seconds: u64,
        price: u64,
        side: PositionSide,
    ) -> Result<()> {
        instructions::order_management::place_twap_order(
            ctx, total_size, duration_seconds, interval_seconds, price, side
        )
    }

    // Collateral Management - IMPORTANT Functions
    pub fn initialize_collateral_account(
        ctx: Context<InitializeCollateralAccount>,
        asset_type: CollateralType,
    ) -> Result<()> {
        instructions::collateral_management::initialize_collateral_account(ctx, asset_type)
    }

    pub fn add_collateral(
        ctx: Context<AddCollateral>,
        amount: u64,
    ) -> Result<()> {
        instructions::collateral_management::add_collateral(ctx, amount)
    }

    pub fn remove_collateral(
        ctx: Context<RemoveCollateral>,
        amount: u64,
    ) -> Result<()> {
        instructions::collateral_management::remove_collateral(ctx, amount)
    }

    pub fn deposit_native_sol(ctx: Context<DepositNativeSol>, amount: u64) -> Result<()> {
        instructions::collateral_management::deposit_native_sol(ctx, amount)
    }

    pub fn withdraw_native_sol(ctx: Context<WithdrawNativeSol>, amount: u64) -> Result<()> {
        instructions::collateral_management::withdraw_native_sol(ctx, amount)
    }

    // Token Operations
    
    pub fn initialize_token_vault(ctx: Context<InitializeTokenVault>, mint_address: Pubkey) -> Result<()> {
        instructions::vault_management::initialize_token_vault(ctx, mint_address)
    }
    
    pub fn deposit_tokens(ctx: Context<DepositTokens>, amount: u64) -> Result<()> {
        instructions::vault_management::deposit_tokens(ctx, amount)
    }
    
    pub fn withdraw_tokens(ctx: Context<WithdrawTokens>, amount: u64) -> Result<()> {
        instructions::vault_management::withdraw_tokens(ctx, amount)
    }
    
    pub fn create_user_token_account(ctx: Context<CreateUserTokenAccount>) -> Result<()> {
        instructions::vault_management::create_user_token_account(ctx)
    }

    pub fn initialize_protocol_sol_vault(ctx: Context<InitializeProtocolSolVault>) -> Result<()> {
        instructions::vault_management::initialize_protocol_sol_vault(ctx)
    }

    // User Account Management

    pub fn create_user_account(ctx: Context<CreateUserAccount>, account_index: u16) -> Result<()> {
        instructions::user_account_management::create_user_account(ctx, account_index)
    }

    pub fn update_user_account(
        ctx: Context<UpdateUserAccount>,
        total_collateral: Option<u64>,
        total_positions: Option<u16>,
        total_orders: Option<u16>,
        account_health: Option<u16>,
        liquidation_price: Option<u64>,
    ) -> Result<()> {
        instructions::user_account_management::update_user_account(
            ctx, total_collateral, total_positions, total_orders, account_health, liquidation_price
        )
    }

    pub fn close_user_account(ctx: Context<CloseUserAccount>) -> Result<()> {
        instructions::user_account_management::close_user_account(ctx)
    }

    pub fn check_user_permissions(ctx: Context<UpdateUserAccount>, action: UserAction) -> Result<()> {
        instructions::user_account_management::check_user_permissions(ctx, action)
    }

    // Insurance Fund Management
    
    pub fn initialize_insurance_fund(ctx: Context<InitializeInsuranceFund>, initial_deposit: u64) -> Result<()> {
        instructions::insurance_oracle_management::initialize_insurance_fund(ctx, initial_deposit)
    }
    
    pub fn deposit_insurance_fund(ctx: Context<DepositInsuranceFund>, amount: u64) -> Result<()> {
        instructions::insurance_oracle_management::deposit_insurance_fund(ctx, amount)
    }
    
    pub fn withdraw_insurance_fund(ctx: Context<WithdrawInsuranceFund>, amount: u64) -> Result<()> {
        instructions::insurance_oracle_management::withdraw_insurance_fund(ctx, amount)
    }
    
    pub fn distribute_fees(ctx: Context<DistributeFees>, amount: u64) -> Result<()> {
        instructions::insurance_oracle_management::distribute_fees(ctx, amount)
    }

    pub fn add_oracle_feed(ctx: Context<AddOracleFeed>, feed_type: OracleFeedType, weight: u8) -> Result<()> {
        instructions::insurance_oracle_management::add_oracle_feed(ctx, feed_type, weight)
    }
    
    pub fn remove_oracle_feed(ctx: Context<RemoveOracleFeed>, feed_index: u8) -> Result<()> {
        instructions::insurance_oracle_management::remove_oracle_feed(ctx, feed_index)
    }
    
    pub fn update_oracle_weights(ctx: Context<UpdateOracleWeights>, weights: Vec<u8>) -> Result<()> {
        instructions::insurance_oracle_management::update_oracle_weights(ctx, weights)
    }
    
    pub fn update_whitelist(_ctx: Context<UpdateWhitelist>, user: Pubkey, is_whitelisted: bool) -> Result<()> {
        msg!("Whitelist updated for user {}: {}", user, is_whitelisted);
        Ok(())
    }
    
    pub fn update_market_parameters(
        ctx: Context<UpdateMarketRiskParameters>,
        max_leverage: Option<u8>,
        initial_margin_ratio: Option<u16>,
        maintenance_margin_ratio: Option<u16>,
    ) -> Result<()> {
        instructions::market_management::update_market_parameters(
            ctx, max_leverage, initial_margin_ratio, maintenance_margin_ratio
        )
    }

    // Cross-Program Integration
    
    pub fn jupiter_swap(ctx: Context<JupiterSwap>, input_amount: u64, minimum_output_amount: u64) -> Result<()> {
        instructions::cross_program::jupiter_swap(ctx, input_amount, minimum_output_amount)
    }

    // Keeper Network
    
    pub fn register_keeper(ctx: Context<RegisterKeeper>, stake_amount: u64) -> Result<()> {
        instructions::keeper_management::register_keeper(ctx, stake_amount)
    }

    // JIT Liquidity
    
    pub fn provide_jit_liquidity(ctx: Context<ProvideJitLiquidity>, amount: u64, fee_rate: u16) -> Result<()> {
        require!(fee_rate <= 1000, ErrorCode::InvalidFeeRate);
        let jit_provider = &mut ctx.accounts.jit_provider;
        jit_provider.available_liquidity += amount;
        jit_provider.fee_rate = fee_rate;
        jit_provider.last_update = Clock::get()?.unix_timestamp;
        msg!("JIT liquidity provided: {} with fee rate {}bps", amount, fee_rate);
        Ok(())
    }

    pub fn execute_twap_chunk(ctx: Context<ExecuteTwapChunk>, chunk_size: u64) -> Result<()> {
        instructions::advanced_orders::execute_twap_chunk(ctx, chunk_size)
    }

    pub fn execute_iceberg_chunk(ctx: Context<ExecuteIcebergChunk>, chunk_size: u64) -> Result<()> {
        instructions::advanced_orders::execute_iceberg_chunk(ctx, chunk_size)
    }

    // Cross-Collateralization System
    
    pub fn initialize_cross_collateral_account(ctx: Context<InitializeCrossCollateralAccount>) -> Result<()> {
        instructions::collateral_management::initialize_cross_collateral_account(ctx)
    }

    pub fn add_cross_collateral(ctx: Context<AddCrossCollateral>, asset_type: CollateralType, amount: u64) -> Result<()> {
        instructions::collateral_management::add_cross_collateral(ctx, asset_type, amount)
    }

    pub fn remove_cross_collateral(ctx: Context<RemoveCrossCollateral>, asset_type: CollateralType, amount: u64) -> Result<()> {
        instructions::collateral_management::remove_cross_collateral(ctx, asset_type, amount)
    }

    pub fn update_collateral_config(
        ctx: Context<UpdateCollateralConfig>,
        initial_asset_weight: u16,
        maintenance_asset_weight: u16,
        _initial_liability_weight: u16,
        _maintenance_liability_weight: u16,
        _imf_factor: u16,
        _max_collateral_amount: u64,
    ) -> Result<()> {
        instructions::collateral_management::update_collateral_config(
            ctx, initial_asset_weight, maintenance_asset_weight
        )
    }

    // ===== SECURITY MANAGEMENT INSTRUCTIONS =====
    
    /// Initialize Security Circuit Breaker System
    pub fn initialize_security_circuit_breaker(ctx: Context<InitializeSecurityCircuitBreaker>) -> Result<()> {
        instructions::security_management::initialize_security_circuit_breaker(ctx)
    }

    /// Initialize Keeper Security Manager
    pub fn initialize_keeper_security_manager(ctx: Context<InitializeKeeperSecurityManager>) -> Result<()> {
        instructions::security_management::initialize_keeper_security_manager(ctx)
    }

    /// Initialize Oracle Staleness Protection
    pub fn initialize_oracle_staleness_protection(ctx: Context<InitializeOracleStalenessProtection>) -> Result<()> {
        instructions::security_management::initialize_oracle_staleness_protection(ctx)
    }

    /// Update Security Parameters
    pub fn update_security_parameters(
        ctx: Context<UpdateSecurityParameters>,
        max_price_change_percent: u16,
        max_volume_spike_percent: u16,
        max_oracle_deviation_percent: u16,
        max_system_load_percent: u16,
        max_staleness_seconds: u64,
        liquidation_rate_limit: u32,
    ) -> Result<()> {
        instructions::security_management::update_security_parameters(
            ctx, max_price_change_percent, max_volume_spike_percent, max_oracle_deviation_percent,
            max_system_load_percent, max_staleness_seconds, liquidation_rate_limit
        )
    }

    /// Authorize Keeper for Liquidations
    pub fn authorize_keeper(
        ctx: Context<AuthorizeKeeper>,
        keeper_pubkey: Pubkey,
        stake_amount: u64,
        performance_score: u16,
        auth_level: crate::security::KeeperAuthLevel,
    ) -> Result<()> {
        instructions::security_management::authorize_keeper(ctx, keeper_pubkey, stake_amount, performance_score, auth_level)
    }

    /// Deauthorize Keeper
    pub fn deauthorize_keeper(ctx: Context<AuthorizeKeeper>, keeper_pubkey: Pubkey) -> Result<()> {
        instructions::security_management::deauthorize_keeper(ctx, keeper_pubkey)
    }

    /// Emergency Pause
    pub fn emergency_pause(ctx: Context<EmergencyPause>) -> Result<()> {
        instructions::security_management::emergency_pause(ctx)
    }

    /// Emergency Resume
    pub fn emergency_resume(ctx: Context<EmergencyResume>) -> Result<()> {
        instructions::security_management::emergency_resume(ctx)
    }

    /// Check Security Before Trading Operations
    pub fn check_security_before_trading(
        ctx: Context<CheckSecurityBeforeTrading>,
        current_price: u64,
        current_volume: u64,
        system_load: u16,
    ) -> Result<()> {
        instructions::security_management::check_security_before_trading(ctx, current_price, current_volume, system_load)
    }

    /// Check Keeper Authorization for Liquidation
    pub fn check_keeper_authorization(ctx: Context<CheckSecurityBeforeTrading>, keeper_pubkey: Pubkey) -> Result<()> {
        instructions::security_management::check_keeper_authorization(ctx, keeper_pubkey)
    }

    /// Record Liquidation Attempt
    pub fn record_liquidation_attempt(
        ctx: Context<CheckSecurityBeforeTrading>,
        keeper_pubkey: Pubkey,
        position_owner: Pubkey,
        position_size: u64,
        liquidation_price: u64,
        success: bool,
        reason: crate::security::LiquidationReason,
    ) -> Result<()> {
        instructions::security_management::record_liquidation_attempt(
            ctx, keeper_pubkey, position_owner, position_size, liquidation_price, success, reason
        )
    }

    /// Set Emergency Price
    pub fn set_emergency_price(ctx: Context<CheckSecurityBeforeTrading>, emergency_price: u64) -> Result<()> {
        instructions::security_management::set_emergency_price(ctx, emergency_price)
    }
}
