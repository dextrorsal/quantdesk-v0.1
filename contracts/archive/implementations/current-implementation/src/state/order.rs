/// Order State Module
/// Contains all order-related account structures

use anchor_lang::prelude::*;
use crate::state::position::PositionSide;

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
    PostOnly, // Post only (maker only)
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

impl Order {
    pub const INIT_SPACE: usize = 32 + 32 + 1 + 1 + 8 + 8 + 8 + 8 + 1 + 1 + 8 + 8 + 8 + 1 + 8 + 8 + 1 + 8 + 1 + 8 + 8;
    
    /// Check if order is executable
    pub fn is_executable(&self, current_price: u64) -> bool {
        match self.order_type {
            OrderType::Market => true,
            OrderType::Limit => {
                match self.side {
                    PositionSide::Long => current_price <= self.price,
                    PositionSide::Short => current_price >= self.price,
                }
            },
            OrderType::StopLoss => {
                match self.side {
                    PositionSide::Long => current_price <= self.stop_price,
                    PositionSide::Short => current_price >= self.stop_price,
                }
            },
            OrderType::TakeProfit => {
                match self.side {
                    PositionSide::Long => current_price >= self.stop_price,
                    PositionSide::Short => current_price <= self.stop_price,
                }
            },
            _ => false,
        }
    }
}
