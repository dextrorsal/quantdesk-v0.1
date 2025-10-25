//! Order management instructions
//! This module contains all order-related instruction handlers and their context structs.

use anchor_lang::prelude::*;
use crate::{
    state::order::{Order, OrderType, OrderStatus, TimeInForce},
    state::position::PositionSide,
    markets::Market,
    ErrorCode,
    margin::*,
};

/// Place an advanced order with enhanced validation
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
    validate_leverage(leverage as u16, market.max_leverage)?;
    validate_position_size(size)?;
    
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

/// Cancel an order with security checks
pub fn cancel_order(ctx: Context<CancelOrder>) -> Result<()> {
    let order = &mut ctx.accounts.order;
    
    require!(order.status == OrderStatus::Pending, ErrorCode::OrderNotPending);
    require!(order.user == ctx.accounts.user.key(), ErrorCode::UnauthorizedUser);
    
    // Check if order has expired
    let current_time = Clock::get()?.unix_timestamp;
    if order.expires_at > 0 && current_time > order.expires_at {
        order.status = OrderStatus::Expired;
        return Ok(());
    }
    
    // Mark order as cancelled
    order.status = OrderStatus::Cancelled;
    
    msg!("Order cancelled: {}", ctx.accounts.order.key());
    Ok(())
}

/// Execute a conditional order with price validation
pub fn execute_conditional_order(ctx: Context<ExecuteConditionalOrder>) -> Result<()> {
    let order = &mut ctx.accounts.order;
    let market = &ctx.accounts.market;
    
    require!(order.status == OrderStatus::Pending, ErrorCode::OrderNotPending);
    
    let current_price = market.get_oracle_price()?;
    let current_time = Clock::get()?.unix_timestamp;
    
    // Check if order has expired
    if order.expires_at > 0 && current_time > order.expires_at {
        order.status = OrderStatus::Expired;
        return Ok(());
    }
    
    // Check if condition is met based on order type
    let condition_met = match order.order_type {
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
        OrderType::StopLimit => {
            match order.side {
                PositionSide::Long => current_price <= order.stop_price,
                PositionSide::Short => current_price >= order.stop_price,
            }
        },
        _ => false,
    };
    
    require!(condition_met, ErrorCode::ConditionNotMet);
    
    // Execute the order (simplified - in production this would involve position creation)
    order.status = OrderStatus::Filled;
    order.filled_size = order.size;
    
    msg!("Conditional order executed: {} at price {}", order.key(), current_price);
    Ok(())
}

/// Place One-Cancels-Other order
pub fn place_oco_order(
    _ctx: Context<PlaceOcoOrder>,
    size: u64,
    _entry_price: u64,
    _stop_loss_price: u64,
    _take_profit_price: u64,
) -> Result<()> {
    // TODO: Implement OCO order logic
    msg!("OCO order placed: size {}", size);
    Ok(())
}

/// Place bracket order (entry + stop loss + take profit)
pub fn place_bracket_order(
    _ctx: Context<PlaceBracketOrder>,
    size: u64,
    _entry_price: u64,
    _stop_loss_price: u64,
    _take_profit_price: u64,
) -> Result<()> {
    // TODO: Implement bracket order logic
    msg!("Bracket order placed: size {}", size);
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
    let base_price = market.get_oracle_price()?;
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
    
    msg!("Iceberg order placed: total {} display {} at {}", total_size, display_size, price);
    Ok(())
}

/// Place a TWAP order (Time Weighted Average Price)
pub fn place_twap_order(
    ctx: Context<PlaceTwapOrder>,
    total_size: u64,
    duration_seconds: u64,
    interval_seconds: u64,
    price: u64,
    side: PositionSide,
) -> Result<()> {
    require!(duration_seconds > 0, ErrorCode::InvalidDuration);
    require!(interval_seconds > 0, ErrorCode::InvalidInterval);
    require!(interval_seconds <= duration_seconds, ErrorCode::InvalidInterval);
    require!(price > 0, ErrorCode::InvalidPrice);
    
    let order = &mut ctx.accounts.order;
    
    order.order_type = OrderType::TWAP;
    order.size = total_size;
    order.price = price;
    order.side = side;
    order.twap_duration = duration_seconds;
    order.twap_interval = interval_seconds;
    order.status = OrderStatus::Pending;
    order.created_at = Clock::get()?.unix_timestamp;
    
    msg!("TWAP order placed: {} over {}s in {}s intervals", total_size, duration_seconds, interval_seconds);
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
    order.time_in_force = TimeInForce::IOC;
    order.status = OrderStatus::Pending;
    order.created_at = Clock::get()?.unix_timestamp;
    
    msg!("IOC order placed: {} at {}", size, price);
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
    order.time_in_force = TimeInForce::FOK;
    order.status = OrderStatus::Pending;
    order.created_at = Clock::get()?.unix_timestamp;
    
    msg!("FOK order placed: {} at {}", size, price);
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
    order.time_in_force = TimeInForce::PostOnly;
    order.status = OrderStatus::Pending;
    order.created_at = Clock::get()?.unix_timestamp;
    
    msg!("Post-only order placed: {} at {}", size, price);
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
    order.stop_price = stop_price;
    order.price = limit_price;
    order.side = side;
    order.leverage = leverage;
    order.status = OrderStatus::Pending;
    order.created_at = Clock::get()?.unix_timestamp;
    
    msg!("Stop-limit order placed: {} stop {} limit {}", size, stop_price, limit_price);
    Ok(())
}

/// Context structs for order management instructions

#[derive(Accounts)]
pub struct PlaceOrder<'info> {
    #[account(mut)]
    pub market: Account<'info, Market>,
    
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

// Advanced Order Contexts
#[derive(Accounts)]
pub struct PlaceOcoOrder<'info> {
    #[account(mut)]
    pub market: Account<'info, Market>,
    
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
        seeds = [b"order", user.key().as_ref()],
        bump
    )]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[derive(Accounts)]
pub struct ExecuteJitOrder<'info> {
    #[account(mut)]
    pub jit_provider: Account<'info, crate::state::advanced::JitProvider>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub executor: Signer<'info>,
}

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
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
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
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
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
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
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
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
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
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
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
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}