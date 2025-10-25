use anchor_lang::prelude::*;
use crate::state::order::{Order, OrderType, OrderStatus};
use crate::markets::Market;

/// Advanced Orders Module
/// Handles execution of complex order types like TWAP and Iceberg orders

/// Execute TWAP Chunk Context
#[derive(Accounts)]
pub struct ExecuteTwapChunk<'info> {
    #[account(mut)]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub executor: Signer<'info>,
}

/// Execute Iceberg Chunk Context
#[derive(Accounts)]
pub struct ExecuteIcebergChunk<'info> {
    #[account(mut)]
    pub order: Account<'info, Order>,
    
    #[account(mut)]
    pub market: Account<'info, Market>,
    
    #[account(mut)]
    pub executor: Signer<'info>,
}

/// Advanced Orders Error Codes
#[error_code]
pub enum AdvancedOrderError {
    #[msg("Invalid order type")]
    InvalidOrderType,
    #[msg("Order not pending")]
    OrderNotPending,
    #[msg("Invalid chunk size")]
    InvalidChunkSize,
    #[msg("Order already completed")]
    OrderAlreadyCompleted,
    #[msg("Insufficient order size")]
    InsufficientOrderSize,
    #[msg("Unauthorized executor")]
    UnauthorizedExecutor,
}

/// Execute a chunk of a TWAP (Time-Weighted Average Price) order
pub fn execute_twap_chunk(
    ctx: Context<ExecuteTwapChunk>,
    chunk_size: u64,
) -> Result<()> {
    let order = &mut ctx.accounts.order;
    let market = &ctx.accounts.market;
    
    require!(order.order_type == OrderType::TWAP, AdvancedOrderError::InvalidOrderType);
    require!(order.status == OrderStatus::Pending, AdvancedOrderError::OrderNotPending);
    require!(chunk_size <= order.size - order.filled_size, AdvancedOrderError::InvalidChunkSize);
    
    // Execute the chunk
    order.filled_size += chunk_size;
    
    // Check if order is complete
    if order.filled_size >= order.size {
        order.status = OrderStatus::Filled;
        msg!("TWAP order completed: {} filled", order.filled_size);
    } else {
        msg!("TWAP chunk executed: {} of {} filled", order.filled_size, order.size);
    }
    
    Ok(())
}

/// Execute a chunk of an Iceberg order
pub fn execute_iceberg_chunk(
    ctx: Context<ExecuteIcebergChunk>,
    chunk_size: u64,
) -> Result<()> {
    let order = &mut ctx.accounts.order;
    let market = &ctx.accounts.market;
    
    require!(order.order_type == OrderType::Iceberg, AdvancedOrderError::InvalidOrderType);
    require!(order.status == OrderStatus::Pending, AdvancedOrderError::OrderNotPending);
    require!(chunk_size <= order.display_size, AdvancedOrderError::InvalidChunkSize);
    require!(chunk_size <= order.size - order.filled_size, AdvancedOrderError::InvalidChunkSize);
    
    // Execute the chunk
    order.filled_size += chunk_size;
    
    // Check if order is complete
    if order.filled_size >= order.size {
        order.status = OrderStatus::Filled;
        msg!("Iceberg order completed: {} filled", order.filled_size);
    } else {
        msg!("Iceberg chunk executed: {} of {} filled", order.filled_size, order.size);
    }
    
    Ok(())
}