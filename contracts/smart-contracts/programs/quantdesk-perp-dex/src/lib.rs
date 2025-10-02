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
}

// TokenVault is defined in token_operations.rs

// Space calculations
impl Market {
    pub const INIT_SPACE: usize = 4 + 32 + 4 + 32 + 8 + 8 + 8 + 8 + 8 + 32 + 1 + 2 + 2 + 1 + 8 + 8 + 1;
}

impl Position {
    pub const INIT_SPACE: usize = 32 + 32 + 8 + 1 + 1 + 8 + 8 + 8 + 1 + 4 + 32 + 8; // Added Vec<Pubkey> and u64
}

impl CollateralAccount {
    pub const INIT_SPACE: usize = 32 + 1 + 8 + 8 + 8 + 1 + 1;
}

impl Order {
    pub const INIT_SPACE: usize = 32 + 32 + 1 + 1 + 8 + 8 + 8 + 8 + 1 + 1 + 8 + 8 + 8 + 1 + 8 + 8 + 1 + 8 + 1 + 8 + 8;
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