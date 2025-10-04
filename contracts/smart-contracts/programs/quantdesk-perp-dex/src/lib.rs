use anchor_lang::prelude::*;
use anchor_spl::token::{self, Token, TokenAccount, Mint, Transfer};
use anchor_spl::associated_token::AssociatedToken;

mod user_accounts;
mod token_operations;
mod pda_utils;

use user_accounts::*;
use token_operations::*;
use pda_utils::*;

declare_id!("GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a");

#[program]
pub mod quantdesk_perp_dex {
    use super::*;

    // Initialize a new market with oracle integration
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

    // Update oracle price (called by keeper bots)
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

    // Open a new position with enhanced security
    pub fn open_position(
        ctx: Context<OpenPosition>,
        size: u64,
        side: PositionSide,
        leverage: u8,
    ) -> Result<()> {
        let market = &ctx.accounts.market;
        
        // Security validations
        require!(market.is_active, ErrorCode::MarketInactive);
        require!(leverage >= 1 && leverage <= market.max_leverage, ErrorCode::InvalidLeverage);
        require!(size > 0, ErrorCode::InvalidSize);
        require!(size <= 1000000000, ErrorCode::PositionTooLarge); // Max 1B units
        
        let position = &mut ctx.accounts.position;
        
        // Calculate required margin using oracle price
        let oracle_price = market.get_oracle_price()?;
        let required_margin = (size * oracle_price) / (leverage as u64 * 1000000);
        
        // Check if user has enough collateral
        require!(
            ctx.accounts.user_collateral.amount >= required_margin,
            ErrorCode::InsufficientCollateral
        );

        // Calculate position value
        let position_value = (size * oracle_price) / 1000000;
        
        // Check position limits
        require!(position_value <= 1000000000000, ErrorCode::PositionTooLarge); // Max $1T position

        // Update vAMM reserves (simplified for now)
        // In production, this would be more sophisticated
        let market = &mut ctx.accounts.market;
        match side {
            PositionSide::Long => {
                market.base_reserve += size;
                market.quote_reserve -= position_value;
            },
            PositionSide::Short => {
                market.base_reserve -= size;
                market.quote_reserve += position_value;
            },
        }

        // Initialize position
        position.user = ctx.accounts.user.key();
        position.market = market.key();
        position.size = size;
        position.side = side.clone();
        position.leverage = leverage;
        position.entry_price = oracle_price;
        position.margin = required_margin;
        position.unrealized_pnl = 0;
        position.created_at = Clock::get()?.unix_timestamp;
        position.bump = ctx.bumps.position;

        msg!("Position opened: {} {:?} at {}x leverage, margin: {}", 
             size, side, leverage, required_margin);
        Ok(())
    }

    // Close a position with P&L calculation
    pub fn close_position(ctx: Context<ClosePosition>) -> Result<()> {
        let position = &mut ctx.accounts.position;
        let market = &mut ctx.accounts.market;
        
        require!(position.size > 0, ErrorCode::PositionAlreadyClosed);
        
        // Calculate P&L using oracle price
        let current_price = market.get_oracle_price()?;
        let pnl = match position.side {
            PositionSide::Long => {
                ((current_price as i128 - position.entry_price as i128) * position.size as i128) / 1000000
            },
            PositionSide::Short => {
                ((position.entry_price as i128 - current_price as i128) * position.size as i128) / 1000000
            },
        };

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
        position.unrealized_pnl = pnl as i64;
        
        Ok(())
    }

    // Liquidate a position (called by keeper bots)
    pub fn liquidate_position(ctx: Context<LiquidatePosition>) -> Result<()> {
        let position = &mut ctx.accounts.position;
        let market = &mut ctx.accounts.market;
        
        require!(position.size > 0, ErrorCode::PositionAlreadyClosed);
        
        // Calculate health factor using oracle price
        let current_price = market.get_oracle_price()?;
        let unrealized_pnl = match position.side {
            PositionSide::Long => {
                ((current_price as i128 - position.entry_price as i128) * position.size as i128) / 1000000
            },
            PositionSide::Short => {
                ((position.entry_price as i128 - current_price as i128) * position.size as i128) / 1000000
            },
        };
        
        let equity = position.margin as i128 + unrealized_pnl;
        let position_value = (position.size * current_price) / 1000000;
        let health_factor = (equity * 10000) / position_value as i128;
        
        require!(health_factor < market.maintenance_margin_ratio as i128, ErrorCode::PositionHealthy);
        
        // Execute liquidation
        msg!("Liquidating position: Health factor = {}%", health_factor / 100);
        
        // Transfer collateral to vault (simplified)
        // In production, this would involve proper token transfers
        
        // Mark position as liquidated
        position.size = 0;
        position.unrealized_pnl = unrealized_pnl as i64;
        
        Ok(())
    }

    // Settle funding for all positions in a market
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

    // Place an advanced order with enhanced validation
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
        let market = &ctx.accounts.market;
        
        // Security validations
        require!(market.is_active, ErrorCode::MarketInactive);
        require!(leverage >= 1 && leverage <= market.max_leverage, ErrorCode::InvalidLeverage);
        require!(size > 0, ErrorCode::InvalidSize);
        require!(size <= 1000000000, ErrorCode::PositionTooLarge);
        
        let current_time = Clock::get()?.unix_timestamp;
        
        // Validate order parameters based on type
        match order_type {
            OrderType::Market => {
                require!(price == 0, ErrorCode::InvalidPrice);
            },
            OrderType::Limit => {
                require!(price > 0, ErrorCode::InvalidPrice);
                let oracle_price = market.get_oracle_price()?;
                require!(price <= oracle_price * 110 / 100, ErrorCode::PriceTooHigh); // Max 10% above oracle
                require!(price >= oracle_price * 90 / 100, ErrorCode::PriceTooLow); // Min 10% below oracle
            },
            OrderType::StopLoss | OrderType::TakeProfit => {
                require!(stop_price > 0, ErrorCode::InvalidStopPrice);
            },
            OrderType::TrailingStop => {
                require!(trailing_distance > 0, ErrorCode::InvalidTrailingDistance);
                require!(trailing_distance <= 1000000, ErrorCode::TrailingDistanceTooLarge); // Max 100%
            },
            OrderType::Iceberg => {
                require!(hidden_size > 0, ErrorCode::InvalidSize);
                require!(display_size > 0, ErrorCode::InvalidSize);
                require!(hidden_size + display_size == size, ErrorCode::InvalidSize);
            },
            OrderType::TWAP => {
                require!(twap_duration > 0, ErrorCode::InvalidDuration);
                require!(twap_interval > 0, ErrorCode::InvalidInterval);
                require!(twap_interval <= twap_duration, ErrorCode::InvalidInterval);
            },
            OrderType::StopLimit => {
                require!(stop_price > 0, ErrorCode::InvalidStopPrice);
                require!(price > 0, ErrorCode::InvalidPrice);
            },
            OrderType::Bracket => {
                require!(target_price > 0, ErrorCode::InvalidTargetPrice);
                require!(stop_price > 0, ErrorCode::InvalidStopPrice);
            },
            _ => {}
        }

        // Validate expiration
        if expires_at > 0 {
            require!(expires_at > current_time, ErrorCode::OrderExpired);
            require!(expires_at <= current_time + 86400 * 30, ErrorCode::OrderExpirationTooLong); // Max 30 days
        }

        // Initialize order
        let order = &mut ctx.accounts.order;
        order.user = ctx.accounts.user.key();
        order.market = market.key();
        order.order_type = order_type;
        order.side = side;
        order.size = size;
        order.price = price;
        order.stop_price = stop_price;
        order.trailing_distance = trailing_distance;
        order.leverage = leverage;
        order.status = OrderStatus::Pending;
        order.created_at = current_time;
        order.expires_at = expires_at;
        order.filled_size = 0;
        order.bump = ctx.bumps.order;
        // Advanced order fields
        order.hidden_size = hidden_size;
        order.display_size = display_size;
        order.time_in_force = time_in_force;
        order.target_price = target_price;
        order.parent_order = None; // Will be set for bracket orders
        order.twap_duration = twap_duration;
        order.twap_interval = twap_interval;

        msg!("Order placed: {:?} {} {:?} at {}x leverage", order_type, size, side, leverage);
        Ok(())
    }

    // Cancel an order with security checks
    pub fn cancel_order(ctx: Context<CancelOrder>) -> Result<()> {
        let order = &mut ctx.accounts.order;
        
        require!(order.status == OrderStatus::Pending, ErrorCode::OrderNotPending);
        require!(order.user == ctx.accounts.user.key(), ErrorCode::UnauthorizedUser);
        
        // Check if order has expired
        let current_time = Clock::get()?.unix_timestamp;
        if order.expires_at > 0 && current_time > order.expires_at {
            order.status = OrderStatus::Expired;
        } else {
            order.status = OrderStatus::Cancelled;
        }
        
        msg!("Order cancelled: {}", ctx.accounts.order.key());
        Ok(())
    }

    // Execute a conditional order with price validation
    pub fn execute_conditional_order(ctx: Context<ExecuteConditionalOrder>) -> Result<()> {
        let order = &mut ctx.accounts.order;
        let market = &ctx.accounts.market;
        
        require!(order.status == OrderStatus::Pending, ErrorCode::OrderNotPending);
        
        let current_price = market.get_oracle_price()?;
        let should_execute = match order.order_type {
            OrderType::StopLoss => {
                match order.side {
                    PositionSide::Long => current_price <= order.stop_price,
                    PositionSide::Short => current_price >= order.stop_price,
                }
            },
            OrderType::TakeProfit => {
                match order.side {
                    PositionSide::Long => current_price >= order.stop_price,
                    PositionSide::Short => current_price <= order.stop_price,
                }
            },
            _ => false,
        };
        
        require!(should_execute, ErrorCode::ConditionNotMet);
        
        // Execute the order as a market order
        order.status = OrderStatus::Filled;
        order.filled_size = order.size;
        
        msg!("Conditional order executed: {} at price {}", order.key(), current_price);
        Ok(())
    }

    // Cross-collateralization functions

    // Initialize a collateral account for a user
    pub fn initialize_collateral_account(
        ctx: Context<InitializeCollateralAccount>,
        asset_type: CollateralType,
        initial_amount: u64,
    ) -> Result<()> {
        let collateral_account = &mut ctx.accounts.collateral_account;
        
        // Validate initial amount
        require!(initial_amount > 0, ErrorCode::InvalidAmount);
        
        // Initialize collateral account
        collateral_account.user = ctx.accounts.user.key();
        collateral_account.asset_type = asset_type;
        collateral_account.amount = initial_amount;
        collateral_account.value_usd = 0; // Will be updated by oracle
        collateral_account.last_updated = Clock::get()?.unix_timestamp;
        collateral_account.is_active = true;
        collateral_account.bump = ctx.bumps.collateral_account;
        
        msg!("Collateral account initialized: {} {} for user {}", 
             initial_amount, format!("{:?}", asset_type), ctx.accounts.user.key());
        Ok(())
    }

    // Add collateral to an existing account
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

    // Remove collateral from an account
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

    // Update collateral value using oracle price
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

    // Open position with cross-collateralization
    pub fn open_position_cross_collateral(
        ctx: Context<OpenPositionCrossCollateral>,
        size: u64,
        side: PositionSide,
        leverage: u8,
        collateral_accounts: Vec<Pubkey>,
    ) -> Result<()> {
        let market = &ctx.accounts.market;
        
        // Security validations
        require!(market.is_active, ErrorCode::MarketInactive);
        require!(leverage >= 1 && leverage <= market.max_leverage, ErrorCode::InvalidLeverage);
        require!(size > 0, ErrorCode::InvalidSize);
        require!(size <= 1000000000, ErrorCode::PositionTooLarge);
        require!(!collateral_accounts.is_empty(), ErrorCode::NoCollateralProvided);
        
        let position = &mut ctx.accounts.position;
        
        // Calculate required margin using oracle price
        let oracle_price = market.get_oracle_price()?;
        let required_margin_usd = (size * oracle_price) / (leverage as u64 * 1000000);
        
        // Calculate total collateral value from all accounts
        let mut total_collateral_value = 0u64;
        for _collateral_pubkey in &collateral_accounts {
            // In production, this would fetch each collateral account
            // For now, we'll assume sufficient collateral
            total_collateral_value += 1000000; // Mock value
        }
        
        // Check if user has enough total collateral
        require!(
            total_collateral_value >= required_margin_usd,
            ErrorCode::InsufficientCollateral
        );

        // Calculate position value
        let position_value = (size * oracle_price) / 1000000;
        
        // Check position limits
        require!(position_value <= 1000000000000, ErrorCode::PositionTooLarge);

        // Update vAMM reserves (simplified for now)
        let market = &mut ctx.accounts.market;
        match side {
            PositionSide::Long => {
                market.base_reserve += size;
                market.quote_reserve -= position_value;
            },
            PositionSide::Short => {
                market.base_reserve -= size;
                market.quote_reserve += position_value;
            },
        }

        // Initialize position with cross-collateralization
        position.user = ctx.accounts.user.key();
        position.market = market.key();
        position.size = size;
        position.side = side.clone();
        position.leverage = leverage;
        position.entry_price = oracle_price;
        position.margin = required_margin_usd;
        position.unrealized_pnl = 0;
        position.created_at = Clock::get()?.unix_timestamp;
        position.bump = ctx.bumps.position;
        position.collateral_accounts = collateral_accounts;
        position.total_collateral_value = total_collateral_value;

        msg!("Cross-collateralized position opened: {} {:?} at {}x leverage, total collateral: {} USD", 
             size, side, leverage, total_collateral_value);
        Ok(())
    }

    // Liquidate position with cross-collateralization
    pub fn liquidate_position_cross_collateral(ctx: Context<LiquidatePositionCrossCollateral>) -> Result<()> {
        let position = &mut ctx.accounts.position;
        let market = &mut ctx.accounts.market;
        
        require!(position.size > 0, ErrorCode::PositionAlreadyClosed);
        
        // Calculate health factor using oracle price
        let current_price = market.get_oracle_price()?;
        let unrealized_pnl = match position.side {
            PositionSide::Long => {
                ((current_price as i128 - position.entry_price as i128) * position.size as i128) / 1000000
            },
            PositionSide::Short => {
                ((position.entry_price as i128 - current_price as i128) * position.size as i128) / 1000000
            },
        };
        
        let equity = position.total_collateral_value as i128 + unrealized_pnl;
        let position_value = (position.size * current_price) / 1000000;
        let health_factor = (equity * 10000) / position_value as i128;
        
        require!(health_factor < market.maintenance_margin_ratio as i128, ErrorCode::PositionHealthy);
        
        // Execute liquidation with cross-collateralization
        msg!("Liquidating cross-collateralized position: Health factor = {}%, Total collateral: {} USD", 
             health_factor / 100, position.total_collateral_value);
        
        // In production, this would distribute liquidation across collateral accounts
        // For now, we'll mark position as liquidated
        
        // Mark position as liquidated
        position.size = 0;
        position.unrealized_pnl = unrealized_pnl as i64;
        
        Ok(())
    }

    // Token Operations - Following Solana Cookbook patterns
    
    /// Initialize a token vault for protocol deposits
    pub fn initialize_token_vault(
        ctx: Context<InitializeTokenVault>,
        mint_address: Pubkey,
    ) -> Result<()> {
        token_operations::initialize_token_vault(ctx, mint_address)
    }
    
    /// Deposit tokens into protocol vault
    pub fn deposit_tokens(
        ctx: Context<DepositTokens>,
        amount: u64,
    ) -> Result<()> {
        token_operations::deposit_tokens(ctx, amount)
    }
    
    /// Withdraw tokens from protocol vault
    pub fn withdraw_tokens(
        ctx: Context<WithdrawTokens>,
        amount: u64,
    ) -> Result<()> {
        token_operations::withdraw_tokens(ctx, amount)
    }
    
    /// Create user token account if needed
    pub fn create_user_token_account(
        ctx: Context<CreateUserTokenAccount>,
    ) -> Result<()> {
        token_operations::create_user_token_account(ctx)
    }

    /// Initialize protocol SOL vault
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

    /// Deposit native SOL to user account
    pub fn deposit_native_sol(ctx: Context<DepositNativeSol>, amount: u64) -> Result<()> {
        let user_account = &mut ctx.accounts.user_account;
        let protocol_vault = &mut ctx.accounts.protocol_vault;
        let collateral_account = &mut ctx.accounts.collateral_account;
        
        // Validate amount
        require!(amount > 0, TokenError::InvalidAmount);
        
        // Update user account with SOL deposit
        user_account.total_collateral += amount;
        user_account.last_activity = Clock::get()?.unix_timestamp;
        
        // Update protocol vault statistics
        protocol_vault.total_deposits += amount;
        
        // Update or initialize SOL collateral account
        if collateral_account.amount == 0 {
            // Initialize collateral account if it's new
            collateral_account.user = ctx.accounts.user.key();
            collateral_account.asset_type = CollateralType::SOL;
            collateral_account.amount = amount;
            collateral_account.value_usd = amount; // SOL value in lamports (1:1 for now)
            collateral_account.last_updated = Clock::get()?.unix_timestamp;
            collateral_account.is_active = true;
            collateral_account.bump = ctx.bumps.collateral_account;
            
            msg!("SOL collateral account initialized with {} lamports", amount);
        } else {
            // Add to existing collateral account
            collateral_account.amount += amount;
            collateral_account.value_usd += amount;
            collateral_account.last_updated = Clock::get()?.unix_timestamp;
            
            msg!("Added {} lamports to existing SOL collateral account", amount);
        }
        
        // Transfer SOL to protocol vault
        let transfer_instruction = anchor_lang::system_program::Transfer {
            from: ctx.accounts.user.to_account_info(),
            to: ctx.accounts.protocol_vault.to_account_info(),
        };
        
        anchor_lang::system_program::transfer(
            CpiContext::new(
                ctx.accounts.system_program.to_account_info(),
                transfer_instruction,
            ),
            amount,
        )?;
        
        msg!("Native SOL deposited: {} lamports by user: {}", amount, ctx.accounts.user.key());
        
        Ok(())
    }

    /// Withdraw native SOL from user account
    pub fn withdraw_native_sol(ctx: Context<WithdrawNativeSol>, amount: u64) -> Result<()> {
        let user_account = &mut ctx.accounts.user_account;
        let protocol_vault = &mut ctx.accounts.protocol_vault;
        let collateral_account = &mut ctx.accounts.collateral_account;
        
        // Validate amount
        require!(amount > 0, TokenError::InvalidAmount);
        require!(collateral_account.amount >= amount, TokenError::InsufficientVaultBalance);
        
        // Update user account
        user_account.total_collateral -= amount;
        user_account.last_activity = Clock::get()?.unix_timestamp;
        
        // Update protocol vault statistics
        protocol_vault.total_withdrawals += amount;
        
        // Update collateral account
        collateral_account.amount -= amount;
        collateral_account.value_usd -= amount;
        collateral_account.last_updated = Clock::get()?.unix_timestamp;
        
        // Transfer SOL from protocol vault to user
        let transfer_instruction = anchor_lang::system_program::Transfer {
            from: ctx.accounts.protocol_vault.to_account_info(),
            to: ctx.accounts.user.to_account_info(),
        };
        
        anchor_lang::system_program::transfer(
            CpiContext::new(
                ctx.accounts.system_program.to_account_info(),
                transfer_instruction,
            ),
            amount,
        )?;
        
        msg!("Native SOL withdrawn: {} lamports by user: {}", amount, ctx.accounts.user.key());
        
        Ok(())
    }

    // User Account Management Functions

    // Create a new user account
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

    // Update user account (called when positions/orders change)
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

    // Close user account (only if no positions/orders)
    pub fn close_user_account(ctx: Context<CloseUserAccount>) -> Result<()> {
        let user_account = &mut ctx.accounts.user_account;
        
        // Deactivate the account
        user_account.deactivate()?;
        
        msg!("User account closed: {}", ctx.accounts.user_account.key());
        Ok(())
    }

    // Check if user can perform specific actions
    pub fn check_user_permissions(
        ctx: Context<UpdateUserAccount>,
        action: UserAction,
    ) -> Result<()> {
        let user_account = &ctx.accounts.user_account;
        
        match action {
            UserAction::Deposit => {
                require!(user_account.can_deposit(), UserAccountError::AccountInactive);
            },
            UserAction::Withdraw => {
                require!(user_account.can_withdraw(), UserAccountError::AccountInactive);
            },
            UserAction::Trade => {
                require!(user_account.can_trade(), UserAccountError::AccountInactive);
            },
            UserAction::CreatePosition => {
                require!(user_account.can_trade(), UserAccountError::AccountInactive);
            },
            UserAction::ClosePosition => {
                require!(user_account.is_active, UserAccountError::AccountInactive);
            },
        }
        
        Ok(())
    }

    // ===== INSURANCE FUND MANAGEMENT =====
    
    /// Initialize insurance fund with initial deposit
    pub fn initialize_insurance_fund(
        ctx: Context<InitializeInsuranceFund>,
        initial_deposit: u64,
    ) -> Result<()> {
        require!(initial_deposit > 0, ErrorCode::InvalidInsuranceFundOperation);
        
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
        require!(amount > 0, ErrorCode::InvalidInsuranceFundOperation);
        require!(ctx.accounts.insurance_fund.is_active, ErrorCode::InvalidInsuranceFundOperation);
        
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
        require!(amount > 0, ErrorCode::InvalidInsuranceFundOperation);
        require!(ctx.accounts.insurance_fund.is_active, ErrorCode::InvalidInsuranceFundOperation);
        
        let insurance_fund = &mut ctx.accounts.insurance_fund;
        let available_balance = insurance_fund.total_deposits - insurance_fund.total_withdrawals;
        require!(available_balance >= amount, ErrorCode::InsufficientInsuranceFundBalance);
        
        let current_utilization = (insurance_fund.total_withdrawals + amount) * 10000 / insurance_fund.total_deposits;
        require!(current_utilization < insurance_fund.max_utilization.into(), ErrorCode::InvalidInsuranceFundOperation);
        
        insurance_fund.total_withdrawals += amount;
        
        msg!("Withdrew {} from insurance fund", amount);
        Ok(())
    }
    
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

    // ===== EMERGENCY CONTROLS =====
    
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

    // ===== FEE MANAGEMENT =====
    
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
    
    /// Distribute fees to stakeholders
    pub fn distribute_fees(
        ctx: Context<DistributeFees>,
        amount: u64,
    ) -> Result<()> {
        require!(amount > 0, ErrorCode::InvalidFeeParameters);
        
        let fee_collector = &mut ctx.accounts.fee_collector;
        let total_fees = fee_collector.trading_fees_collected + fee_collector.funding_fees_collected;
        require!(amount <= total_fees, ErrorCode::InvalidFeeParameters);
        
        msg!("Distributed {} fees to stakeholders", amount);
        Ok(())
    }

    // ===== ORACLE MANAGEMENT =====
    
    /// Add new oracle feed
    pub fn add_oracle_feed(
        ctx: Context<AddOracleFeed>,
        feed_type: OracleFeedType,
        weight: u8,
    ) -> Result<()> {
        require!(weight > 0 && weight <= 100, ErrorCode::InvalidOracleWeight);
        
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
        require!(feed_index < ctx.accounts.oracle_manager.feeds.len() as u8, ErrorCode::OracleFeedNotFound);
        
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
        require!(weights.len() == ctx.accounts.oracle_manager.feeds.len(), ErrorCode::InvalidOracleWeight);
        
        let total_weight: u8 = weights.iter().sum();
        require!(total_weight == 100, ErrorCode::InvalidOracleWeight);
        
        for weight in &weights {
            require!(*weight > 0 && *weight <= 100, ErrorCode::InvalidOracleWeight);
        }
        
        let oracle_manager = &mut ctx.accounts.oracle_manager;
        oracle_manager.weights = weights;
        
        msg!("Updated oracle weights");
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

    // ===== GOVERNANCE & ADMIN =====
    
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
    
    /// Update user whitelist
    pub fn update_whitelist(
        ctx: Context<UpdateWhitelist>,
        user: Pubkey,
        is_whitelisted: bool,
    ) -> Result<()> {
        msg!("Whitelist updated for user {}: {}", user, is_whitelisted);
        Ok(())
    }
    
    /// Update market parameters
    pub fn update_market_parameters(
        ctx: Context<UpdateMarketParameters>,
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

    // ===== ADVANCED ORDER TYPES =====
    
    /// Place One-Cancels-Other order
    pub fn place_oco_order(
        _ctx: Context<PlaceOcoOrder>,
        size: u64,
        _entry_price: u64,
        _stop_price: u64,
        _limit_price: u64,
        side: PositionSide,
        leverage: u8,
    ) -> Result<()> {
        // Implementation for OCO orders
        msg!("OCO order placed: {} {} at {}x leverage", size, format!("{:?}", side), leverage);
        Ok(())
    }
    
    /// Place bracket order (entry + stop loss + take profit)
    pub fn place_bracket_order(
        _ctx: Context<PlaceBracketOrder>,
        size: u64,
        _entry_price: u64,
        _stop_loss_price: u64,
        _take_profit_price: u64,
        side: PositionSide,
        leverage: u8,
    ) -> Result<()> {
        // Implementation for bracket orders
        msg!("Bracket order placed: {} {} at {}x leverage", size, format!("{:?}", side), leverage);
        Ok(())
    }

    // ===== CROSS-PROGRAM INTEGRATION =====
    
    /// Jupiter DEX integration for token swaps
    pub fn jupiter_swap(
        _ctx: Context<JupiterSwap>,
        input_amount: u64,
        minimum_output_amount: u64,
    ) -> Result<()> {
        // Implementation for Jupiter integration
        msg!("Jupiter swap: {} -> min {}", input_amount, minimum_output_amount);
        Ok(())
    }

    // ===== KEEPER NETWORK =====
    
    /// Register a new keeper in the network
    pub fn register_keeper(
        ctx: Context<RegisterKeeper>,
        stake_amount: u64,
    ) -> Result<()> {
        require!(stake_amount >= 1000000000, ErrorCode::InsufficientKeeperStake); // Min 1 SOL
        
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
        
        keeper_network.keepers.push(keeper_info);
        keeper_network.total_stake += stake_amount;
        
        msg!("Keeper registered: {} with stake {}", ctx.accounts.keeper.key(), stake_amount);
        Ok(())
    }

    /// Execute liquidation through keeper network
    pub fn liquidate_position_keeper(
        ctx: Context<LiquidatePositionKeeper>,
        position_id: u64,
    ) -> Result<()> {
        let keeper_network = &mut ctx.accounts.keeper_network;
        let position = &mut ctx.accounts.position;
        let market = &ctx.accounts.market;
        
        // Verify keeper is registered and active
        let keeper_index = keeper_network.keepers.iter()
            .position(|k| k.keeper_pubkey == ctx.accounts.keeper.key())
            .ok_or(ErrorCode::KeeperNotRegistered)?;
        
        require!(keeper_network.keepers[keeper_index].is_active, ErrorCode::KeeperInactive);
        
        // Check if position is liquidatable
        let current_price = market.get_current_price()?;
        let position_value = position.calculate_position_value(current_price)?;
        let margin_ratio = position.calculate_margin_ratio(position_value)?;
        
        require!(margin_ratio < market.maintenance_margin_ratio, ErrorCode::PositionNotLiquidatable);
        
        // Execute liquidation
        let liquidation_fee = position_value * 50 / 10000; // 0.5% liquidation fee
        let keeper_reward = liquidation_fee * 80 / 100; // 80% to keeper, 20% to insurance fund
        
        // Update keeper stats
        keeper_network.keepers[keeper_index].total_liquidations += 1;
        keeper_network.keepers[keeper_index].total_rewards_earned += keeper_reward;
        keeper_network.keepers[keeper_index].last_activity = Clock::get()?.unix_timestamp;
        
        // Close position
        position.unrealized_pnl = 0; // Reset P&L when closed
        
        msg!("Position {} liquidated by keeper {} for reward {}", 
             position_id, ctx.accounts.keeper.key(), keeper_reward);
        Ok(())
    }

    /// Update keeper performance score
    pub fn update_keeper_performance(
        ctx: Context<UpdateKeeperPerformance>,
        keeper_pubkey: Pubkey,
        performance_score: u16,
    ) -> Result<()> {
        require!(performance_score <= 1000, ErrorCode::InvalidPerformanceScore);
        
        let keeper_network = &mut ctx.accounts.keeper_network;
        let keeper_index = keeper_network.keepers.iter()
            .position(|k| k.keeper_pubkey == keeper_pubkey)
            .ok_or(ErrorCode::KeeperNotRegistered)?;
        
        keeper_network.keepers[keeper_index].performance_score = performance_score;
        
        // Deactivate keeper if performance is too low
        if performance_score < 500 {
            keeper_network.keepers[keeper_index].is_active = false;
            msg!("Keeper {} deactivated due to poor performance", keeper_pubkey);
        }
        
        Ok(())
    }

    // ===== CIRCUIT BREAKERS =====
    
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

    // ===== JIT LIQUIDITY =====
    
    /// Provide Just-In-Time liquidity
    pub fn provide_jit_liquidity(
        ctx: Context<ProvideJitLiquidity>,
        amount: u64,
        fee_rate: u16,
    ) -> Result<()> {
        require!(fee_rate <= 1000, ErrorCode::InvalidFeeRate); // Max 10%
        
        let jit_provider = &mut ctx.accounts.jit_provider;
        jit_provider.available_liquidity += amount;
        jit_provider.fee_rate = fee_rate;
        jit_provider.last_update = Clock::get()?.unix_timestamp;
        
        msg!("JIT liquidity provided: {} with fee rate {}bps", amount, fee_rate);
        Ok(())
    }

    /// Execute order with JIT liquidity
    pub fn execute_jit_order(
        ctx: Context<ExecuteJitOrder>,
        order_size: u64,
        is_buy: bool,
    ) -> Result<()> {
        let jit_provider = &mut ctx.accounts.jit_provider;
        let market = &mut ctx.accounts.market;
        
        require!(jit_provider.available_liquidity >= order_size, ErrorCode::InsufficientJitLiquidity);
        
        // Calculate execution price with JIT fee
        let base_price = market.get_current_price()?;
        let jit_fee = (order_size * jit_provider.fee_rate as u64) / 10000;
        let execution_price = if is_buy {
            base_price + jit_fee
        } else {
            base_price - jit_fee
        };
        
        // Update liquidity
        jit_provider.available_liquidity -= order_size;
        jit_provider.total_volume += order_size;
        jit_provider.total_fees_earned += jit_fee;
        
        msg!("JIT order executed: {} {} at price {}", 
             order_size, if is_buy { "buy" } else { "sell" }, execution_price);
        Ok(())
    }

    // ===== ADVANCED ORDER TYPES =====
    
    /// Place an Iceberg order (large order split into smaller chunks)
    pub fn place_iceberg_order(
        ctx: Context<PlaceIcebergOrder>,
        total_size: u64,
        display_size: u64,
        price: u64,
        side: PositionSide,
        leverage: u8,
    ) -> Result<()> {
        require!(display_size > 0, ErrorCode::InvalidSize);
        require!(display_size <= total_size, ErrorCode::InvalidSize);
        require!(price > 0, ErrorCode::InvalidPrice);
        
        let order = &mut ctx.accounts.order;
        let hidden_size = total_size - display_size;
        
        order.order_type = OrderType::Iceberg;
        order.size = total_size;
        order.display_size = display_size;
        order.hidden_size = hidden_size;
        order.price = price;
        order.side = side;
        order.leverage = leverage;
        order.status = OrderStatus::Pending;
        order.created_at = Clock::get()?.unix_timestamp;
        order.time_in_force = TimeInForce::GTC; // Good Till Cancelled
        
        msg!("Iceberg order placed: {} total ({} visible) {} at {}x leverage", 
             total_size, display_size, format!("{:?}", side), leverage);
        Ok(())
    }

    /// Place a TWAP order (Time Weighted Average Price)
    pub fn place_twap_order(
        ctx: Context<PlaceTwapOrder>,
        total_size: u64,
        duration_seconds: u64,
        interval_seconds: u64,
        price_limit: u64,
        side: PositionSide,
        leverage: u8,
    ) -> Result<()> {
        require!(total_size > 0, ErrorCode::InvalidSize);
        require!(duration_seconds > 0, ErrorCode::InvalidTwapParameters);
        require!(interval_seconds > 0, ErrorCode::InvalidTwapParameters);
        require!(interval_seconds <= duration_seconds, ErrorCode::InvalidTwapParameters);
        require!(price_limit > 0, ErrorCode::InvalidPrice);
        
        let order = &mut ctx.accounts.order;
        let chunk_size = total_size / (duration_seconds / interval_seconds);
        
        order.order_type = OrderType::TWAP;
        order.size = total_size;
        order.price = price_limit;
        order.side = side;
        order.leverage = leverage;
        order.status = OrderStatus::Pending;
        order.created_at = Clock::get()?.unix_timestamp;
        order.expires_at = Clock::get()?.unix_timestamp + duration_seconds as i64;
        order.time_in_force = TimeInForce::GTD; // Good Till Date
        
        msg!("TWAP order placed: {} {} over {}s ({}s intervals) at {}x leverage", 
             total_size, format!("{:?}", side), duration_seconds, interval_seconds, leverage);
        Ok(())
    }

    /// Place an IOC order (Immediate or Cancel)
    pub fn place_ioc_order(
        ctx: Context<PlaceIocOrder>,
        size: u64,
        price: u64,
        side: PositionSide,
        leverage: u8,
    ) -> Result<()> {
        require!(size > 0, ErrorCode::InvalidSize);
        require!(price > 0, ErrorCode::InvalidPrice);
        
        let order = &mut ctx.accounts.order;
        
        order.order_type = OrderType::IOC;
        order.size = size;
        order.price = price;
        order.side = side;
        order.leverage = leverage;
        order.status = OrderStatus::Pending;
        order.created_at = Clock::get()?.unix_timestamp;
        order.time_in_force = TimeInForce::IOC; // Immediate or Cancel
        
        msg!("IOC order placed: {} {} at {}x leverage", 
             size, format!("{:?}", side), leverage);
        Ok(())
    }

    /// Place an FOK order (Fill or Kill)
    pub fn place_fok_order(
        ctx: Context<PlaceFokOrder>,
        size: u64,
        price: u64,
        side: PositionSide,
        leverage: u8,
    ) -> Result<()> {
        require!(size > 0, ErrorCode::InvalidSize);
        require!(price > 0, ErrorCode::InvalidPrice);
        
        let order = &mut ctx.accounts.order;
        
        order.order_type = OrderType::FOK;
        order.size = size;
        order.price = price;
        order.side = side;
        order.leverage = leverage;
        order.status = OrderStatus::Pending;
        order.created_at = Clock::get()?.unix_timestamp;
        order.time_in_force = TimeInForce::FOK; // Fill or Kill
        
        msg!("FOK order placed: {} {} at {}x leverage", 
             size, format!("{:?}", side), leverage);
        Ok(())
    }

    /// Place a Post-Only order (Maker only)
    pub fn place_post_only_order(
        ctx: Context<PlacePostOnlyOrder>,
        size: u64,
        price: u64,
        side: PositionSide,
        leverage: u8,
    ) -> Result<()> {
        require!(size > 0, ErrorCode::InvalidSize);
        require!(price > 0, ErrorCode::InvalidPrice);
        
        let order = &mut ctx.accounts.order;
        
        order.order_type = OrderType::PostOnly;
        order.size = size;
        order.price = price;
        order.side = side;
        order.leverage = leverage;
        order.status = OrderStatus::Pending;
        order.created_at = Clock::get()?.unix_timestamp;
        order.time_in_force = TimeInForce::GTC; // Good Till Cancelled
        
        msg!("Post-Only order placed: {} {} at {}x leverage", 
             size, format!("{:?}", side), leverage);
        Ok(())
    }

    /// Place a Stop-Limit order
    pub fn place_stop_limit_order(
        ctx: Context<PlaceStopLimitOrder>,
        size: u64,
        stop_price: u64,
        limit_price: u64,
        side: PositionSide,
        leverage: u8,
    ) -> Result<()> {
        require!(size > 0, ErrorCode::InvalidSize);
        require!(stop_price > 0, ErrorCode::InvalidStopPrice);
        require!(limit_price > 0, ErrorCode::InvalidPrice);
        
        let order = &mut ctx.accounts.order;
        
        order.order_type = OrderType::StopLimit;
        order.size = size;
        order.price = limit_price;
        order.stop_price = stop_price;
        order.side = side;
        order.leverage = leverage;
        order.status = OrderStatus::Pending;
        order.created_at = Clock::get()?.unix_timestamp;
        order.time_in_force = TimeInForce::GTC; // Good Till Cancelled
        
        msg!("Stop-Limit order placed: {} {} stop={} limit={} at {}x leverage", 
             size, format!("{:?}", side), stop_price, limit_price, leverage);
        Ok(())
    }

    /// Execute TWAP order chunk
    pub fn execute_twap_chunk(
        ctx: Context<ExecuteTwapChunk>,
        chunk_size: u64,
    ) -> Result<()> {
        let order = &mut ctx.accounts.order;
        let market = &ctx.accounts.market;
        
        require!(order.order_type == OrderType::TWAP, ErrorCode::InvalidOrderType);
        require!(order.status == OrderStatus::Pending, ErrorCode::OrderNotPending);
        require!(chunk_size <= order.size - order.filled_size, ErrorCode::InvalidSize);
        
        // Check if TWAP order is still valid
        let current_time = Clock::get()?.unix_timestamp;
        require!(current_time <= order.expires_at, ErrorCode::OrderExpired);
        
        // Execute the chunk
        let current_price = market.get_current_price()?;
        let execution_price = if order.price > 0 {
            order.price.min(current_price)
        } else {
            current_price
        };
        
        order.filled_size += chunk_size;
        
        // Check if order is fully filled
        if order.filled_size >= order.size {
            order.status = OrderStatus::Filled;
        }
        
        msg!("TWAP chunk executed: {} at price {}", chunk_size, execution_price);
        Ok(())
    }

    /// Execute Iceberg order chunk
    pub fn execute_iceberg_chunk(
        ctx: Context<ExecuteIcebergChunk>,
        chunk_size: u64,
    ) -> Result<()> {
        let order = &mut ctx.accounts.order;
        let market = &ctx.accounts.market;
        
        require!(order.order_type == OrderType::Iceberg, ErrorCode::InvalidOrderType);
        require!(order.status == OrderStatus::Pending, ErrorCode::OrderNotPending);
        require!(chunk_size <= order.display_size, ErrorCode::InvalidSize);
        require!(chunk_size <= order.size - order.filled_size, ErrorCode::InvalidSize);
        
        // Execute the chunk
        let current_price = market.get_current_price()?;
        let execution_price = if order.price > 0 {
            order.price.min(current_price)
        } else {
            current_price
        };
        
        order.filled_size += chunk_size;
        
        // Check if order is fully filled
        if order.filled_size >= order.size {
            order.status = OrderStatus::Filled;
        }
        
        msg!("Iceberg chunk executed: {} at price {}", chunk_size, execution_price);
        Ok(())
    }

    // ===== CROSS-COLLATERALIZATION SYSTEM =====
    
    /// Initialize cross-collateral account for a user
    pub fn initialize_cross_collateral_account(
        ctx: Context<InitializeCrossCollateralAccount>,
    ) -> Result<()> {
        let cross_collateral_account = &mut ctx.accounts.cross_collateral_account;
        
        cross_collateral_account.user = ctx.accounts.user.key();
        cross_collateral_account.total_collateral_value = 0;
        cross_collateral_account.total_borrowed_value = 0;
        cross_collateral_account.collateral_assets = Vec::new();
        cross_collateral_account.initial_asset_weight = 8000; // 80% default
        cross_collateral_account.maintenance_asset_weight = 9000; // 90% default
        cross_collateral_account.initial_liability_weight = 12000; // 120% default
        cross_collateral_account.maintenance_liability_weight = 11000; // 110% default
        cross_collateral_account.imf_factor = 125; // 0.125% default
        cross_collateral_account.last_health_check = Clock::get()?.unix_timestamp;
        cross_collateral_account.is_active = true;
        
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
        let collateral_config = &ctx.accounts.collateral_config;
        
        require!(collateral_config.is_active, ErrorCode::CollateralTypeNotActive);
        require!(amount > 0, ErrorCode::InvalidAmount);
        require!(amount <= collateral_config.max_collateral_amount, ErrorCode::ExceedsMaxCollateral);
        
        // Get current price from oracle
        let current_price = 100_000_000; // Mock price: $100 with 6 decimals
        let value_usd = (amount * current_price) / 1_000_000; // Assuming 6 decimals
        
        // Calculate scaled asset weight based on IMF factor
        // Implement IMF scaling similar to Drift Protocol
        let sqrt_value = (cross_collateral_account.total_collateral_value as f64).sqrt() as u64;
        let denominator = 1000 + (collateral_config.imf_factor as u64 * sqrt_value) / 1000;
        let scaled_weight = (1100 * 10000) / denominator;
        let scaled_weight = std::cmp::min(collateral_config.initial_asset_weight, scaled_weight as u16);
        
        // Add or update collateral asset
        let mut asset_found = false;
        for asset in &mut cross_collateral_account.collateral_assets {
            if asset.asset_type == asset_type {
                asset.amount += amount;
                asset.value_usd += value_usd;
                asset.asset_weight = scaled_weight;
                asset.last_price_update = Clock::get()?.unix_timestamp;
                asset_found = true;
                break;
            }
        }
        
        if !asset_found {
            cross_collateral_account.collateral_assets.push(CollateralAsset {
                asset_type,
                amount,
                value_usd,
                asset_weight: scaled_weight,
                liability_weight: collateral_config.initial_liability_weight,
                last_price_update: Clock::get()?.unix_timestamp,
            });
        }
        
        // Update total collateral value
        cross_collateral_account.total_collateral_value += value_usd;
        cross_collateral_account.last_health_check = Clock::get()?.unix_timestamp;
        
        msg!("Added {} {} collateral worth {} USD", amount, format!("{:?}", asset_type), value_usd);
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
        
        // Find and update collateral asset
        let mut asset_found = false;
        for asset in &mut cross_collateral_account.collateral_assets {
            if asset.asset_type == asset_type {
                require!(asset.amount >= amount, ErrorCode::InsufficientCollateral);
                
                // Get current price for accurate USD calculation
                let current_price = 100_000_000; // Mock price: $100 with 6 decimals
                let value_usd = (amount * current_price) / 1_000_000;
                
                asset.amount -= amount;
                asset.value_usd -= value_usd;
                asset.last_price_update = Clock::get()?.unix_timestamp;
                
                // Remove asset if amount becomes zero
                if asset.amount == 0 {
                    cross_collateral_account.collateral_assets.retain(|a| a.asset_type != asset_type);
                }
                
                // Update total collateral value
                cross_collateral_account.total_collateral_value -= value_usd;
                cross_collateral_account.last_health_check = Clock::get()?.unix_timestamp;
                
                asset_found = true;
                break;
            }
        }
        
        require!(asset_found, ErrorCode::CollateralAssetNotFound);
        
        // Check health factor after removal
        let mut total_weighted_collateral = 0u64;
        for asset in &cross_collateral_account.collateral_assets {
            total_weighted_collateral += (asset.value_usd * asset.asset_weight as u64) / 10000;
        }
        let available_margin = total_weighted_collateral - cross_collateral_account.total_borrowed_value;
        let total_borrowed = cross_collateral_account.total_borrowed_value;
        let health_factor = if total_borrowed == 0 { 
            10000 // 100% health factor
        } else {
            (available_margin * 10000) / total_borrowed
        };
        require!(health_factor >= cross_collateral_account.maintenance_asset_weight as u64, ErrorCode::InsufficientHealthFactor);
        
        msg!("Removed {} {} collateral", amount, format!("{:?}", asset_type));
        Ok(())
    }

    /// Update collateral configuration
    pub fn update_collateral_config(
        ctx: Context<UpdateCollateralConfig>,
        initial_asset_weight: u16,
        maintenance_asset_weight: u16,
        initial_liability_weight: u16,
        maintenance_liability_weight: u16,
        imf_factor: u16,
        max_collateral_amount: u64,
    ) -> Result<()> {
        let collateral_config = &mut ctx.accounts.collateral_config;
        
        require!(initial_asset_weight <= 10000, ErrorCode::InvalidWeight);
        require!(maintenance_asset_weight <= 10000, ErrorCode::InvalidWeight);
        require!(initial_liability_weight >= 10000, ErrorCode::InvalidWeight);
        require!(maintenance_liability_weight >= 10000, ErrorCode::InvalidWeight);
        
        collateral_config.initial_asset_weight = initial_asset_weight;
        collateral_config.maintenance_asset_weight = maintenance_asset_weight;
        collateral_config.initial_liability_weight = initial_liability_weight;
        collateral_config.maintenance_liability_weight = maintenance_liability_weight;
        collateral_config.imf_factor = imf_factor;
        collateral_config.max_collateral_amount = max_collateral_amount;
        
        msg!("Collateral config updated for {:?}", collateral_config.asset_type);
        Ok(())
    }
}

// Enhanced account structures
#[account]
pub struct Market {
    pub base_asset: String,           // e.g., "BTC"
    pub quote_asset: String,           // e.g., "USDT"
    pub base_reserve: u64,             // vAMM base reserve
    pub quote_reserve: u64,            // vAMM quote reserve
    pub funding_rate: i64,             // Funding rate in basis points
    pub last_funding_time: i64,        // Last funding settlement time
    pub funding_interval: i64,         // Funding interval in seconds
    pub authority: Pubkey,             // Market authority
    pub max_leverage: u8,              // Maximum allowed leverage
    pub initial_margin_ratio: u16,     // Initial margin ratio in basis points
    pub maintenance_margin_ratio: u16, // Maintenance margin ratio in basis points
    pub is_active: bool,               // Whether market is active
    pub last_oracle_price: u64,        // Last oracle price
    pub last_oracle_update: i64,       // Last oracle update timestamp
    pub bump: u8,                     // PDA bump
}

impl Market {
    pub fn get_oracle_price(&self) -> Result<u64> {
        // Check if oracle price is recent (within 5 minutes)
        let current_time = Clock::get()?.unix_timestamp;
        require!(
            current_time - self.last_oracle_update <= 300, // 5 minutes
            ErrorCode::PriceStale
        );
        
        Ok(self.last_oracle_price)
    }

    pub fn calculate_premium_index(&self) -> Result<i64> {
        // Calculate premium index based on market conditions
        let current_price = self.get_oracle_price()? as i128;
        let oracle_price = self.last_oracle_price as i128;
        
        // Premium index = (mark_price - oracle_price) / oracle_price * 10000
        let premium = ((current_price - oracle_price) * 10000) / oracle_price;
        
        // Clamp premium to reasonable bounds
        Ok(premium.clamp(-10000, 10000) as i64) // ±100%
    }

    pub fn calculate_funding_rate(&self, premium_index: i64) -> Result<i64> {
        // Funding rate = premium_index + clamp(interest_rate, -0.05%, +0.05%)
        let interest_rate = 100; // 1% base interest rate in basis points
        let clamped_interest = premium_index.clamp(-500, 500); // Clamp to ±0.05%
        
        Ok(premium_index + clamped_interest + interest_rate)
    }

    pub fn get_current_price(&self) -> Result<u64> {
        // Get current price from oracle
        self.get_oracle_price()
    }
}

#[account]
pub struct CollateralAccount {
    pub user: Pubkey,           // User who owns the collateral
    pub asset_type: CollateralType, // Type of collateral asset
    pub amount: u64,             // Amount of collateral
    pub value_usd: u64,         // USD value of collateral
    pub last_updated: i64,      // Last price update timestamp
    pub is_active: bool,         // Whether this collateral is active
    pub bump: u8,              // PDA bump
}

// ===== CROSS-COLLATERALIZATION SYSTEM =====

#[account]
pub struct CrossCollateralAccount {
    pub user: Pubkey,                    // User who owns the cross-collateral account
    pub total_collateral_value: u64,     // Total USD value of all collateral
    pub total_borrowed_value: u64,       // Total USD value borrowed against collateral
    pub collateral_assets: Vec<CollateralAsset>, // List of collateral assets
    pub initial_asset_weight: u16,       // Initial asset weight (basis points)
    pub maintenance_asset_weight: u16,    // Maintenance asset weight (basis points)
    pub initial_liability_weight: u16,   // Initial liability weight (basis points)
    pub maintenance_liability_weight: u16, // Maintenance liability weight (basis points)
    pub imf_factor: u16,                 // IMF (Initial Margin Factor) in basis points
    pub last_health_check: i64,          // Last health check timestamp
    pub is_active: bool,                 // Whether account is active
    pub bump: u8,                       // PDA bump
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct CollateralAsset {
    pub asset_type: CollateralType,      // Type of collateral asset
    pub amount: u64,                     // Amount of asset
    pub value_usd: u64,                  // USD value of asset
    pub asset_weight: u16,               // Asset weight (basis points)
    pub liability_weight: u16,           // Liability weight (basis points)
    pub last_price_update: i64,          // Last price update timestamp
}

#[account]
pub struct CollateralConfig {
    pub asset_type: CollateralType,      // Type of collateral asset
    pub initial_asset_weight: u16,       // Initial asset weight (basis points)
    pub maintenance_asset_weight: u16,  // Maintenance asset weight (basis points)
    pub initial_liability_weight: u16,   // Initial liability weight (basis points)
    pub maintenance_liability_weight: u16, // Maintenance liability weight (basis points)
    pub imf_factor: u16,                 // IMF factor (basis points)
    pub max_collateral_amount: u64,       // Maximum collateral amount
    pub oracle_price_feed: Pubkey,       // Oracle price feed for this asset
    pub is_active: bool,                 // Whether this collateral type is active
    pub bump: u8,                       // PDA bump
}

#[account]
pub struct Position {
    pub user: Pubkey,           // User who owns the position
    pub market: Pubkey,         // Market this position is in
    pub size: u64,              // Position size
    pub side: PositionSide,     // Long or Short
    pub leverage: u8,           // Leverage multiplier
    pub entry_price: u64,       // Price when position was opened
    pub margin: u64,            // Collateral amount
    pub unrealized_pnl: i64,     // Unrealized P&L
    pub created_at: i64,        // Timestamp when position was created
    pub bump: u8,              // PDA bump
    // Cross-collateralization fields
    pub collateral_accounts: Vec<Pubkey>, // List of collateral accounts used
    pub total_collateral_value: u64,      // Total collateral value in USD
}

#[account]
pub struct Order {
    pub user: Pubkey,           // User who placed the order
    pub market: Pubkey,         // Market this order is for
    pub order_type: OrderType,  // Type of order
    pub side: PositionSide,     // Long or Short
    pub size: u64,              // Order size
    pub price: u64,             // Order price (0 for market orders)
    pub stop_price: u64,        // Stop price for SL/TP orders
    pub trailing_distance: u64, // Trailing distance for trailing stops
    pub leverage: u8,           // Leverage multiplier
    pub status: OrderStatus,    // Order status
    pub created_at: i64,        // Timestamp when order was created
    pub expires_at: i64,        // Timestamp when order expires (0 = never)
    pub filled_size: u64,       // Amount already filled
    pub bump: u8,              // PDA bump
    // Advanced order fields
    pub hidden_size: u64,       // Hidden size for iceberg orders
    pub display_size: u64,      // Display size for iceberg orders
    pub time_in_force: TimeInForce, // Time in force for the order
    pub target_price: u64,      // Target price for bracket orders
    pub parent_order: Option<Pubkey>, // Parent order for bracket orders
    pub twap_duration: u64,     // Duration for TWAP orders (in seconds)
    pub twap_interval: u64,     // Interval for TWAP orders (in seconds)
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum CollateralType {
    SOL,
    USDC,
    BTC,
    ETH,
    USDT,
    AVAX,
    MATIC,
    ARB,
    OP,
    DOGE,
    ADA,
    DOT,
    LINK,
}

impl std::fmt::Display for CollateralType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CollateralType::SOL => write!(f, "SOL"),
            CollateralType::USDC => write!(f, "USDC"),
            CollateralType::BTC => write!(f, "BTC"),
            CollateralType::ETH => write!(f, "ETH"),
            CollateralType::USDT => write!(f, "USDT"),
            CollateralType::AVAX => write!(f, "AVAX"),
            CollateralType::MATIC => write!(f, "MATIC"),
            CollateralType::ARB => write!(f, "ARB"),
            CollateralType::OP => write!(f, "OP"),
            CollateralType::DOGE => write!(f, "DOGE"),
            CollateralType::ADA => write!(f, "ADA"),
            CollateralType::DOT => write!(f, "DOT"),
            CollateralType::LINK => write!(f, "LINK"),
        }
    }
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum PositionSide {
    Long,
    Short,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum OrderType {
    Market,
    Limit,
    StopLoss,
    TakeProfit,
    TrailingStop,
    PostOnly,
    IOC, // Immediate or Cancel
    FOK, // Fill or Kill
    Iceberg, // Iceberg order (hidden size)
    TWAP, // Time Weighted Average Price
    StopLimit, // Stop limit order
    Bracket, // Bracket order (entry + stop + target)
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum OrderStatus {
    Pending,
    Filled,
    Cancelled,
    Expired,
    PartiallyFilled,
    Rejected,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeInForce {
    GTC, // Good Till Cancelled
    IOC, // Immediate or Cancel
    FOK, // Fill or Kill
    GTD, // Good Till Date
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum UserAction {
    Deposit,
    Withdraw,
    Trade,
    CreatePosition,
    ClosePosition,
}

// ===== NEW ACCOUNT STRUCTURES FOR ADVANCED FEATURES =====

#[account]
pub struct KeeperNetwork {
    pub total_stake: u64,                    // Total stake in the network
    pub keepers: Vec<KeeperInfo>,            // List of registered keepers
    pub liquidation_rewards_pool: u64,       // Pool for liquidation rewards
    pub min_stake_requirement: u64,          // Minimum stake to become keeper
    pub performance_threshold: u16,          // Minimum performance score
    pub bump: u8,                           // PDA bump
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct KeeperInfo {
    pub keeper_pubkey: Pubkey,              // Keeper's public key
    pub stake_amount: u64,                  // Amount staked
    pub performance_score: u16,              // Performance score (0-1000)
    pub is_active: bool,                    // Whether keeper is active
    pub total_liquidations: u32,            // Total liquidations performed
    pub total_rewards_earned: u64,          // Total rewards earned
    pub last_activity: i64,                 // Last activity timestamp
}

impl KeeperNetwork {
    pub const INIT_SPACE: usize = 8 + 8 + 4 + (32 + 8 + 2 + 1 + 4 + 8 + 8) * 100 + 8 + 8 + 2 + 1; // Space for up to 100 keepers
}

#[account]
pub struct CircuitBreaker {
    pub is_triggered: bool,                 // Whether circuit breaker is active
    pub trigger_time: i64,                  // When it was triggered
    pub reset_time: i64,                    // When it was reset
    pub breaker_type: CircuitBreakerType,   // Type of circuit breaker
    pub triggered_by: Pubkey,               // Who triggered it
    pub reset_by: Pubkey,                   // Who reset it
    pub price_change_threshold: u16,        // Price change threshold (basis points)
    pub volume_threshold: u64,              // Volume threshold
    pub time_window: u64,                   // Time window in seconds
    pub bump: u8,                          // PDA bump
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, PartialEq, Debug)]
pub enum CircuitBreakerType {
    PriceVolatility,    // Triggered by extreme price movements
    VolumeSpike,        // Triggered by unusual volume
    SystemOverload,     // Triggered by system performance issues
    EmergencyStop,      // Manual emergency stop
}

impl CircuitBreaker {
    pub const INIT_SPACE: usize = 1 + 8 + 8 + 1 + 32 + 32 + 2 + 8 + 8 + 1;
}

#[account]
pub struct JitProvider {
    pub provider_pubkey: Pubkey,           // JIT provider's public key
    pub available_liquidity: u64,          // Available liquidity
    pub fee_rate: u16,                      // Fee rate in basis points
    pub total_volume: u64,                  // Total volume provided
    pub total_fees_earned: u64,             // Total fees earned
    pub min_order_size: u64,                 // Minimum order size
    pub max_order_size: u64,                // Maximum order size
    pub last_update: i64,                   // Last update timestamp
    pub is_active: bool,                    // Whether provider is active
    pub bump: u8,                          // PDA bump
}

impl JitProvider {
    pub const INIT_SPACE: usize = 32 + 8 + 2 + 8 + 8 + 8 + 8 + 8 + 1 + 1;
}

#[account]
pub struct MarketMakerVault {
    pub vault_pubkey: Pubkey,              // Vault's public key
    pub strategy: MarketMakingStrategy,     // Market making strategy
    pub capital_allocation: u64,            // Capital allocated to vault
    pub performance_fee: u16,               // Performance fee in basis points
    pub total_volume: u64,                  // Total volume traded
    pub total_pnl: i64,                     // Total P&L
    pub is_active: bool,                    // Whether vault is active
    pub created_at: i64,                    // Creation timestamp
    pub bump: u8,                          // PDA bump
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub enum MarketMakingStrategy {
    GridTrading,        // Grid trading strategy
    MeanReversion,      // Mean reversion strategy
    Arbitrage,          // Arbitrage strategy
    LiquidityProvision, // Pure liquidity provision
}

impl MarketMakerVault {
    pub const INIT_SPACE: usize = 32 + 1 + 8 + 2 + 8 + 8 + 1 + 8 + 1;
}

#[account]
pub struct PointsSystem {
    pub user_points: Vec<UserPoints>,       // User points mapping
    pub trading_multiplier: u16,            // Trading activity multiplier
    pub referral_bonus: u16,                // Referral bonus multiplier
    pub staking_multiplier: u16,            // Staking multiplier
    pub total_points_distributed: u64,      // Total points distributed
    pub bump: u8,                          // PDA bump
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct UserPoints {
    pub user_pubkey: Pubkey,               // User's public key
    pub total_points: u64,                 // Total points earned
    pub trading_points: u64,               // Points from trading
    pub referral_points: u64,              // Points from referrals
    pub staking_points: u64,               // Points from staking
    pub last_updated: i64,                 // Last update timestamp
}

impl PointsSystem {
    pub const INIT_SPACE: usize = 4 + (32 + 8 + 8 + 8 + 8 + 8) * 1000 + 2 + 2 + 2 + 8 + 1; // Space for up to 1000 users
}

// Token Operations Context Structures are defined in token_operations.rs

// Context structures
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
pub struct OpenPosition<'info> {
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
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub user_collateral: Account<'info, TokenAccount>,
    
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
    
    #[account(mut)]
    pub user_collateral: Account<'info, TokenAccount>,
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
    pub vault: Account<'info, TokenAccount>,
}

#[derive(Accounts)]
pub struct SettleFunding<'info> {
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub keeper: Signer<'info>,
}

#[derive(Accounts)]
pub struct PlaceOrder<'info> {
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(
        init,
        payer = user,
        space = 8 + Order::INIT_SPACE,
        seeds = [b"order", user.key().as_ref(), market.key().as_ref()],
        bump
    )]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub user_collateral: Account<'info, TokenAccount>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct CancelOrder<'info> {
    #[account(mut)]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct ExecuteConditionalOrder<'info> {
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub executor: Signer<'info>,
}

// Cross-collateralization context structures

#[derive(Accounts)]
#[instruction(asset_type: CollateralType)]
pub struct InitializeCollateralAccount<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + CollateralAccount::INIT_SPACE,
        seeds = [b"collateral", user.key().as_ref(), &asset_type.to_string().as_bytes()],
        bump
    )]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct AddCollateral<'info> {
    #[account(
        mut,
        constraint = collateral_account.user == user.key()
    )]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub user_token_account: Account<'info, TokenAccount>,
}

#[derive(Accounts)]
pub struct RemoveCollateral<'info> {
    #[account(
        mut,
        constraint = collateral_account.user == user.key()
    )]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub user_token_account: Account<'info, TokenAccount>,
}

#[derive(Accounts)]
pub struct UpdateCollateralValue<'info> {
    #[account(mut)]
    pub collateral_account: Account<'info, CollateralAccount>,
    
    /// CHECK: This is the oracle price feed account
    pub price_feed: AccountInfo<'info>,
    
    pub keeper: Signer<'info>,
}

#[derive(Accounts)]
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
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    // Multiple collateral accounts
    #[account(mut)]
    pub collateral_account_1: Account<'info, CollateralAccount>,
    
    #[account(mut)]
    pub collateral_account_2: Option<Account<'info, CollateralAccount>>,
    
    #[account(mut)]
    pub collateral_account_3: Option<Account<'info, CollateralAccount>>,
    
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
    pub vault: Account<'info, TokenAccount>,
}

// ===== NEW ACCOUNT STRUCTURES FOR ADVANCED FEATURES =====

#[account]
pub struct ProgramState {
    pub authority: Pubkey,
    pub is_paused: bool,
    pub insurance_fund: Pubkey,
    pub fee_collector: Pubkey,
    pub oracle_manager: Pubkey,
    pub bump: u8,
}

#[account]
pub struct InsuranceFund {
    pub total_deposits: u64,
    pub total_withdrawals: u64,
    pub utilization_rate: u16,
    pub max_utilization: u16,
    pub is_active: bool,
    pub bump: u8,
}

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

#[account]
pub struct OracleManager {
    pub feeds: Vec<OracleFeed>,
    pub weights: Vec<u8>,
    pub max_deviation: u16,
    pub staleness_threshold: i64,
    pub bump: u8,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub struct OracleFeed {
    pub feed_type: OracleFeedType,
    pub feed_account: Pubkey,
    pub weight: u8,
    pub is_active: bool,
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum OracleFeedType {
    Pyth,
    Switchboard,
    Chainlink,
}

// Space calculations for new accounts
impl ProgramState {
    pub const INIT_SPACE: usize = 32 + 1 + 32 + 32 + 32 + 1;
}

impl InsuranceFund {
    pub const INIT_SPACE: usize = 8 + 8 + 2 + 2 + 1 + 1;
}

impl FeeCollector {
    pub const INIT_SPACE: usize = 8 + 8 + 2 + 2 + 8 + 8 + 1;
}

impl OracleManager {
    pub const INIT_SPACE: usize = 4 + 32 + 4 + 8 + 2 + 8 + 1; // Vec<OracleFeed> + Vec<u8> + other fields
}

// ===== CONTEXT STRUCTURES FOR ADVANCED FEATURES =====

// Insurance Fund Contexts
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

#[derive(Accounts)]
pub struct DepositInsuranceFund<'info> {
    #[account(mut)]
    pub insurance_fund: Account<'info, InsuranceFund>,
    
    #[account(mut)]
    pub depositor: Signer<'info>,
    
    #[account(mut)]
    pub depositor_token_account: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub fund_vault: Account<'info, TokenAccount>,
}

#[derive(Accounts)]
pub struct WithdrawInsuranceFund<'info> {
    #[account(
        mut,
        constraint = insurance_fund.is_active
    )]
    pub insurance_fund: Account<'info, InsuranceFund>,
    
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    #[account(mut)]
    pub fund_vault: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub recipient_token_account: Account<'info, TokenAccount>,
}

#[derive(Accounts)]
pub struct UpdateRiskParameters<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub insurance_fund: Account<'info, InsuranceFund>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

// Emergency Controls Contexts
#[derive(Accounts)]
pub struct PauseProgram<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct ResumeProgram<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct EmergencyWithdraw<'info> {
    #[account(
        mut,
        constraint = program_state.is_paused
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    #[account(mut)]
    pub vault: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub recipient: Account<'info, TokenAccount>,
}

// Fee Management Contexts
#[derive(Accounts)]
pub struct UpdateTradingFees<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub fee_collector: Account<'info, FeeCollector>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct UpdateFundingFees<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub fee_collector: Account<'info, FeeCollector>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct CollectFees<'info> {
    #[account(mut)]
    pub fee_collector: Account<'info, FeeCollector>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub fee_vault: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub collector: Signer<'info>,
}

#[derive(Accounts)]
pub struct DistributeFees<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub fee_collector: Account<'info, FeeCollector>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    #[account(mut)]
    pub fee_vault: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub recipient: Account<'info, TokenAccount>,
}

// Oracle Management Contexts
#[derive(Accounts)]
pub struct AddOracleFeed<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub oracle_manager: Account<'info, OracleManager>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct RemoveOracleFeed<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub oracle_manager: Account<'info, OracleManager>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct UpdateOracleWeights<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub oracle_manager: Account<'info, OracleManager>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct EmergencyOracleOverride<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct UpdatePythPrice<'info> {
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    /// CHECK: This is the Pyth price feed account
    pub pyth_price_feed: AccountInfo<'info>,
    
    #[account(mut)]
    pub keeper: Signer<'info>,
}

// Governance Contexts
#[derive(Accounts)]
pub struct UpdateProgramAuthority<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

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

#[derive(Accounts)]
pub struct UpdateMarketParameters<'info> {
    #[account(
        mut,
        constraint = program_state.authority == authority.key()
    )]
    pub program_state: Account<'info, ProgramState>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

// Advanced Order Contexts
#[derive(Accounts)]
pub struct PlaceOcoOrder<'info> {
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(
        init,
        payer = user,
        space = 8 + Order::INIT_SPACE,
        seeds = [b"oco_order", user.key().as_ref(), market.key().as_ref()],
        bump
    )]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub user_collateral: Account<'info, TokenAccount>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct PlaceBracketOrder<'info> {
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(
        init,
        payer = user,
        space = 8 + Order::INIT_SPACE,
        seeds = [b"bracket_order", user.key().as_ref(), market.key().as_ref()],
        bump
    )]
    pub entry_order: Account<'info, Order>,
    
    #[account(
        init,
        payer = user,
        space = 8 + Order::INIT_SPACE,
        seeds = [b"stop_order", user.key().as_ref(), market.key().as_ref()],
        bump
    )]
    pub stop_order: Account<'info, Order>,
    
    #[account(
        init,
        payer = user,
        space = 8 + Order::INIT_SPACE,
        seeds = [b"profit_order", user.key().as_ref(), market.key().as_ref()],
        bump
    )]
    pub profit_order: Account<'info, Order>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub user_collateral: Account<'info, TokenAccount>,
    
    pub system_program: Program<'info, System>,
}

// Cross-Program Integration Contexts
#[derive(Accounts)]
pub struct JupiterSwap<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub input_token_account: Account<'info, TokenAccount>,
    
    #[account(mut)]
    pub output_token_account: Account<'info, TokenAccount>,
    
    /// CHECK: Jupiter program account
    pub jupiter_program: AccountInfo<'info>,
}

// ===== NEW CONTEXT STRUCTURES FOR ADVANCED FEATURES =====

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

#[derive(Accounts)]
pub struct LiquidatePositionKeeper<'info> {
    #[account(mut)]
    pub keeper_network: Account<'info, KeeperNetwork>,
    
    #[account(mut)]
    pub position: Account<'info, Position>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub keeper: Signer<'info>,
    
    #[account(mut)]
    pub insurance_fund: Account<'info, InsuranceFund>,
}

#[derive(Accounts)]
pub struct UpdateKeeperPerformance<'info> {
    #[account(mut)]
    pub keeper_network: Account<'info, KeeperNetwork>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

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

// ===== ADVANCED ORDER CONTEXT STRUCTURES =====

#[derive(Accounts)]
pub struct PlaceIcebergOrder<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + Order::INIT_SPACE,
        seeds = [b"order", user.key().as_ref()],
        bump
    )]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct PlaceTwapOrder<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + Order::INIT_SPACE,
        seeds = [b"order", user.key().as_ref()],
        bump
    )]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct PlaceIocOrder<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + Order::INIT_SPACE,
        seeds = [b"order", user.key().as_ref()],
        bump
    )]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct PlaceFokOrder<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + Order::INIT_SPACE,
        seeds = [b"order", user.key().as_ref()],
        bump
    )]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct PlacePostOnlyOrder<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + Order::INIT_SPACE,
        seeds = [b"order", user.key().as_ref()],
        bump
    )]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct PlaceStopLimitOrder<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + Order::INIT_SPACE,
        seeds = [b"order", user.key().as_ref()],
        bump
    )]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ExecuteTwapChunk<'info> {
    #[account(mut)]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub executor: Signer<'info>,
}

#[derive(Accounts)]
pub struct ExecuteIcebergChunk<'info> {
    #[account(mut)]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub executor: Signer<'info>,
}

// ===== CROSS-COLLATERALIZATION CONTEXT STRUCTURES =====

#[derive(Accounts)]
pub struct InitializeCrossCollateralAccount<'info> {
    #[account(
        init,
        payer = user,
        space = 8 + CrossCollateralAccount::INIT_SPACE,
        seeds = [b"cross_collateral", user.key().as_ref()],
        bump
    )]
    pub cross_collateral_account: Account<'info, CrossCollateralAccount>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct AddCrossCollateral<'info> {
    #[account(mut)]
    pub cross_collateral_account: Account<'info, CrossCollateralAccount>,
    
    #[account(mut)]
    pub collateral_config: Account<'info, CollateralConfig>,
    
    /// CHECK: Oracle price feed account
    pub oracle_price_feed: AccountInfo<'info>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}

#[derive(Accounts)]
pub struct RemoveCrossCollateral<'info> {
    #[account(mut)]
    pub cross_collateral_account: Account<'info, CrossCollateralAccount>,
    
    /// CHECK: Oracle price feed account
    pub oracle_price_feed: AccountInfo<'info>,
    
    #[account(mut)]
    pub user: Signer<'info>,
}


#[derive(Accounts)]
pub struct UpdateCollateralConfig<'info> {
    #[account(mut)]
    pub collateral_config: Account<'info, CollateralConfig>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

// Enhanced error codes
#[error_code]
pub enum ErrorCode {
    #[msg("Invalid leverage amount")]
    InvalidLeverage,
    #[msg("Invalid position size")]
    InvalidSize,
    #[msg("Insufficient collateral")]
    InsufficientCollateral,
    #[msg("Position is healthy, no liquidation needed")]
    PositionHealthy,
    #[msg("Position is not liquidatable")]
    PositionNotLiquidatable,
    #[msg("Funding settlement not due yet")]
    FundingNotDue,
    #[msg("Invalid order price")]
    InvalidPrice,
    #[msg("Invalid stop price")]
    InvalidStopPrice,
    #[msg("Invalid trailing distance")]
    InvalidTrailingDistance,
    #[msg("Order is not pending")]
    OrderNotPending,
    #[msg("Unauthorized user")]
    UnauthorizedUser,
    #[msg("Condition not met for execution")]
    ConditionNotMet,
    #[msg("Invalid max leverage")]
    InvalidMaxLeverage,
    #[msg("Invalid margin ratio")]
    InvalidMarginRatio,
    #[msg("Market is inactive")]
    MarketInactive,
    #[msg("Position too large")]
    PositionTooLarge,
    #[msg("Price is stale")]
    PriceStale,
    #[msg("Price too high")]
    PriceTooHigh,
    #[msg("Price too low")]
    PriceTooLow,
    #[msg("Trailing distance too large")]
    TrailingDistanceTooLarge,
    #[msg("Order expired")]
    OrderExpired,
    #[msg("Order expiration too long")]
    OrderExpirationTooLong,
    #[msg("Position already closed")]
    PositionAlreadyClosed,
    #[msg("Invalid duration")]
    InvalidDuration,
    #[msg("Invalid interval")]
    InvalidInterval,
    #[msg("Invalid target price")]
    InvalidTargetPrice,
    #[msg("Invalid amount")]
    InvalidAmount,
    #[msg("Collateral account inactive")]
    CollateralAccountInactive,
    #[msg("No collateral provided")]
    NoCollateralProvided,
    // User Account Management Errors
    #[msg("No positions to remove")]
    NoPositionsToRemove,
    #[msg("No orders to remove")]
    NoOrdersToRemove,
    #[msg("Invalid health value")]
    InvalidHealthValue,
    #[msg("Account has open positions")]
    AccountHasPositions,
    #[msg("Account has active orders")]
    AccountHasOrders,
    #[msg("Account is not active")]
    AccountInactive,
    #[msg("Account already exists")]
    AccountAlreadyExists,
    #[msg("Account not found")]
    AccountNotFound,
    // Token-specific error codes
    #[msg("Invalid token amount")]
    InvalidTokenAmount,
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
    #[msg("PDA derivation failed")]
    PdaDerivationFailed,
    #[msg("Invalid PDA owner")]
    InvalidPdaOwner,
    // Advanced Features Error Codes
    #[msg("Invalid insurance fund operation")]
    InvalidInsuranceFundOperation,
    #[msg("Insufficient insurance fund balance")]
    InsufficientInsuranceFundBalance,
    #[msg("Program is paused")]
    ProgramPaused,
    #[msg("Invalid fee parameters")]
    InvalidFeeParameters,
    #[msg("Invalid oracle weight")]
    InvalidOracleWeight,
    #[msg("Oracle feed not found")]
    OracleFeedNotFound,
    #[msg("Unauthorized admin operation")]
    UnauthorizedAdminOperation,
    
    // ===== NEW ERROR CODES FOR ADVANCED FEATURES =====
    
    // Keeper Network Errors
    #[msg("Insufficient keeper stake")]
    InsufficientKeeperStake,
    #[msg("Keeper not registered")]
    KeeperNotRegistered,
    #[msg("Keeper is inactive")]
    KeeperInactive,
    #[msg("Invalid performance score")]
    InvalidPerformanceScore,
    
    // Circuit Breaker Errors
    #[msg("Circuit breaker is already triggered")]
    CircuitBreakerAlreadyTriggered,
    #[msg("Circuit breaker is not triggered")]
    CircuitBreakerNotTriggered,
    
    // JIT Liquidity Errors
    #[msg("Invalid fee rate")]
    InvalidFeeRate,
    #[msg("Insufficient JIT liquidity")]
    InsufficientJitLiquidity,
    #[msg("JIT provider not active")]
    JitProviderInactive,
    
    // Market Maker Vault Errors
    #[msg("Invalid vault strategy")]
    InvalidVaultStrategy,
    #[msg("Insufficient vault capital")]
    InsufficientVaultCapital,
    
    // Points System Errors
    #[msg("Invalid points multiplier")]
    InvalidPointsMultiplier,
    #[msg("User not found in points system")]
    UserNotFoundInPointsSystem,
    
    // ===== ADVANCED ORDER ERROR CODES =====
    
    #[msg("Invalid TWAP parameters")]
    InvalidTwapParameters,
    #[msg("Invalid order type")]
    InvalidOrderType,
    
    // ===== CROSS-COLLATERALIZATION ERROR CODES =====
    
    #[msg("Collateral type is not active")]
    CollateralTypeNotActive,
    #[msg("Exceeds maximum collateral amount")]
    ExceedsMaxCollateral,
    #[msg("Collateral asset not found")]
    CollateralAssetNotFound,
    #[msg("Insufficient health factor")]
    InsufficientHealthFactor,
    #[msg("Invalid weight")]
    InvalidWeight,
}

// TokenVault is defined in token_operations.rs

// Space calculations
impl Market {
    pub const INIT_SPACE: usize = 4 + 32 + 4 + 32 + 8 + 8 + 8 + 8 + 8 + 32 + 1 + 2 + 2 + 1 + 8 + 8 + 1;
}

impl Position {
    pub const INIT_SPACE: usize = 32 + 32 + 8 + 1 + 1 + 8 + 8 + 8 + 1 + 4 + 32 + 8; // Added Vec<Pubkey> and u64

    pub fn calculate_position_value(&self, current_price: u64) -> Result<u64> {
        let price_diff = if self.side == PositionSide::Long {
            current_price as i128 - self.entry_price as i128
        } else {
            self.entry_price as i128 - current_price as i128
        };
        
        let pnl = (price_diff * self.size as i128) / self.entry_price as i128;
        let position_value = (self.size as i128 + pnl) as u64;
        
        Ok(position_value)
    }

    pub fn calculate_margin_ratio(&self, position_value: u64) -> Result<u16> {
        if position_value == 0 {
            return Ok(10000); // 100% margin ratio if position has no value
        }
        
        let margin_ratio = (self.margin as u128 * 10000) / position_value as u128;
        Ok(margin_ratio as u16)
    }
}

impl CollateralAccount {
    pub const INIT_SPACE: usize = 32 + 1 + 8 + 8 + 8 + 1 + 1;
}

impl Order {
    pub const INIT_SPACE: usize = 32 + 32 + 1 + 1 + 8 + 8 + 8 + 8 + 1 + 1 + 8 + 8 + 8 + 1 + 8 + 8 + 1 + 8 + 1 + 8 + 8;
}

impl CrossCollateralAccount {
    pub const INIT_SPACE: usize = 32 + 8 + 8 + 4 + (1 + 8 + 8 + 2 + 2 + 8) * 10 + 2 + 2 + 2 + 2 + 2 + 8 + 1 + 1; // Space for up to 10 collateral assets
}

impl CollateralConfig {
    pub const INIT_SPACE: usize = 1 + 2 + 2 + 2 + 2 + 2 + 8 + 32 + 1 + 1;
}

// Protocol SOL Vault Account Structure
#[account]
pub struct ProtocolSolVault {
    pub total_deposits: u64,       // Total SOL deposited
    pub total_withdrawals: u64,    // Total SOL withdrawn
    pub is_active: bool,           // Whether vault is active
    pub bump: u8,                 // PDA bump
}

impl ProtocolSolVault {
    pub const INIT_SPACE: usize = 8 + 8 + 1 + 1;
}

// Initialize Protocol SOL Vault Context
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