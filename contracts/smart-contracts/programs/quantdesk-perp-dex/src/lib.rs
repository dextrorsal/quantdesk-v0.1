use anchor_lang::prelude::*;
use anchor_spl::token::{Token, TokenAccount, Transfer};
use pyth_sdk_solana::load_price_feed_from_account_info;
use std::ops::Div;

declare_id!("G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J");

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
    pub fn update_oracle_price(ctx: Context<UpdateOraclePrice>) -> Result<()> {
        let market = &mut ctx.accounts.market;
        
        // Load Pyth price feed
        let price_feed = load_price_feed_from_account_info(&ctx.accounts.price_feed)?;
        let current_price = price_feed.get_current_price()?;
        
        // Validate price data
        require!(current_price.conf < 1000000, ErrorCode::PriceStale); // Max 0.1% confidence
        require!(current_price.price > 0, ErrorCode::InvalidPrice);
        
        // Update market with oracle price
        market.last_oracle_price = current_price.price as u64;
        market.last_oracle_update = Clock::get()?.unix_timestamp;
        
        msg!("Oracle price updated: {} for {}/{}", 
             current_price.price, market.base_asset, market.quote_asset);
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
        position.unrealized_pnl = pnl;
        
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
        position.unrealized_pnl = unrealized_pnl;
        
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
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum OrderStatus {
    Pending,
    Filled,
    Cancelled,
    Expired,
}

// Context structures
#[derive(Accounts)]
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
}

// Space calculations
impl Market {
    pub const INIT_SPACE: usize = 4 + 32 + 4 + 32 + 8 + 8 + 8 + 8 + 8 + 32 + 1 + 2 + 2 + 1 + 8 + 8 + 1;
}

impl Position {
    pub const INIT_SPACE: usize = 32 + 32 + 8 + 1 + 1 + 8 + 8 + 8 + 1;
}

impl Order {
    pub const INIT_SPACE: usize = 32 + 32 + 1 + 1 + 8 + 8 + 8 + 8 + 1 + 1 + 8 + 8 + 8 + 1;
}