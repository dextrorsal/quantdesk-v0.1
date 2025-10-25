use anchor_lang::prelude::*;
use crate::state::order::{OrderType};
use crate::state::position::PositionSide;

/// Events Module
/// Contains all program events for monitoring and analytics

#[event]
pub struct LiquidationExecuted {
    pub user: Pubkey,
    pub position: Pubkey,
    pub market: Pubkey,
    pub liquidated_size: u64,
    pub liquidation_price: u64,
    pub penalty: u64,
    pub timestamp: i64,
}

#[event]
pub struct FundingRateUpdated {
    pub market: Pubkey,
    pub funding_rate: i64,
    pub timestamp: i64,
}

#[event]
pub struct OraclePriceUpdated {
    pub market: Pubkey,
    pub price: u64,
    pub confidence: u64,
    pub timestamp: i64,
}

#[event]
pub struct OrderPlaced {
    pub user: Pubkey,
    pub order: Pubkey,
    pub market: Pubkey,
    pub side: PositionSide,
    pub size: u64,
    pub price: u64,
    pub order_type: OrderType,
    pub timestamp: i64,
}

#[event]
pub struct OrderFilled {
    pub user: Pubkey,
    pub order: Pubkey,
    pub market: Pubkey,
    pub side: PositionSide,
    pub size: u64,
    pub price: u64,
    pub fee: u64,
    pub timestamp: i64,
}

#[event]
pub struct PositionOpened {
    pub user: Pubkey,
    pub position: Pubkey,
    pub market: Pubkey,
    pub side: PositionSide,
    pub size: u64,
    pub entry_price: u64,
    pub leverage: u16,
    pub timestamp: i64,
}

#[event]
pub struct PositionClosed {
    pub user: Pubkey,
    pub position: Pubkey,
    pub market: Pubkey,
    pub side: PositionSide,
    pub size: u64,
    pub exit_price: u64,
    pub pnl: i64,
    pub timestamp: i64,
}
