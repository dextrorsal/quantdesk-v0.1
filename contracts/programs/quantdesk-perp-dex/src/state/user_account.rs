/// User Account State Module
/// Contains all user account related structures

use anchor_lang::prelude::*;

#[account]
pub struct UserAccount {
    pub authority: Pubkey,        // User's wallet address
    pub account_index: u16,       // Account number (for sub-accounts)
    
    // ENHANCED: Collateral tracking
    pub total_collateral: u64,    // Total collateral deposited
    
    // ENHANCED: Position/order tracking with limits
    pub total_positions: u16,     // Number of open positions
    pub total_orders: u16,        // Number of active orders
    pub max_positions: u16,       // NEW: Position limit (25-50)
    
    // ENHANCED: Margin requirements (Drift-style)
    pub initial_margin_requirement: u64,  // NEW: Initial margin requirement
    pub maintenance_margin_requirement: u64,  // NEW: Maintenance margin requirement
    pub available_margin: u64,    // NEW: Available margin for new positions
    
    // ENHANCED: Risk management
    pub account_health: u16,      // Account health (0-10000, where 10000 = 100%)
    pub liquidation_price: u64,   // Liquidation price threshold
    pub liquidation_threshold: u16,  // NEW: Basis points (e.g., 2000 = 20%)
    pub max_leverage: u16,        // NEW: Max leverage (e.g., 1000 = 10x)
    
    // NEW: Funding rate tracking
    pub total_funding_paid: i64,  // NEW: Can be negative
    pub total_funding_received: i64,  // NEW
    
    // NEW: Fee tracking
    pub total_fees_paid: u64,     // NEW
    pub total_rebates_earned: u64,  // NEW
    
    // Existing timestamps
    pub created_at: i64,          // Account creation timestamp
    pub last_activity: i64,       // Last activity timestamp
    pub is_active: bool,          // Whether account is active
    pub bump: u8,                // PDA bump seed
}

impl UserAccount {
    // authority(32) + account_index(2) + total_collateral(8) + total_positions(2) + total_orders(2) 
    // + max_positions(2) + initial_margin(8) + maintenance_margin(8) + available_margin(8)
    // + account_health(2) + liquidation_price(8) + liquidation_threshold(2) + max_leverage(2)
    // + total_funding_paid(8) + total_funding_received(8) + total_fees_paid(8) + total_rebates_earned(8)
    // + created_at(8) + last_activity(8) + is_active(1) + bump(1)
    pub const INIT_SPACE: usize = 32 + 2 + 8 + 2 + 2 + 2 + 8 + 8 + 8 + 2 + 8 + 2 + 2 + 8 + 8 + 8 + 8 + 8 + 8 + 1 + 1; 
    
    // Helper functions for user account management
    pub fn initialize(
        &mut self,
        authority: Pubkey,
        account_index: u16,
        bump: u8,
    ) -> Result<()> {
        self.authority = authority;
        self.account_index = account_index;
        self.total_collateral = 0;
        self.total_positions = 0;
        self.total_orders = 0;
        self.max_positions = 50; // Default to 50 positions max
        self.initial_margin_requirement = 0;
        self.maintenance_margin_requirement = 0;
        self.available_margin = 0;
        self.account_health = 10000; // 100% health initially
        self.liquidation_price = 0;
        self.liquidation_threshold = 2000; // 20% default
        self.max_leverage = 1000; // 10x default
        self.total_funding_paid = 0;
        self.total_funding_received = 0;
        self.total_fees_paid = 0;
        self.total_rebates_earned = 0;
        self.created_at = Clock::get()?.unix_timestamp;
        self.last_activity = Clock::get()?.unix_timestamp;
        self.is_active = true;
        self.bump = bump;
        
        Ok(())
    }
    
    pub fn update_activity(&mut self) -> Result<()> {
        self.last_activity = Clock::get()?.unix_timestamp;
        Ok(())
    }
    
    pub fn add_collateral(&mut self, amount: u64) -> Result<()> {
        self.total_collateral += amount;
        self.update_activity()?;
        Ok(())
    }
    
    pub fn remove_collateral(&mut self, amount: u64) -> Result<()> {
        require!(self.total_collateral >= amount, crate::errors::ErrorCode::InsufficientCollateral);
        self.total_collateral -= amount;
        self.update_activity()?;
        Ok(())
    }
    
    pub fn add_position(&mut self) -> Result<()> {
        self.total_positions += 1;
        self.update_activity()?;
        Ok(())
    }
    
    pub fn remove_position(&mut self) -> Result<()> {
        require!(self.total_positions > 0, crate::errors::ErrorCode::NoPositionsToRemove);
        self.total_positions -= 1;
        self.update_activity()?;
        Ok(())
    }
    
    pub fn add_order(&mut self) -> Result<()> {
        self.total_orders += 1;
        self.update_activity()?;
        Ok(())
    }
    
    pub fn remove_order(&mut self) -> Result<()> {
        require!(self.total_orders > 0, crate::errors::ErrorCode::NoOrdersToRemove);
        self.total_orders -= 1;
        self.update_activity()?;
        Ok(())
    }
    
    pub fn update_account_health(&mut self, health: u16) -> Result<()> {
        require!(health <= 10000, crate::errors::ErrorCode::InvalidHealthValue);
        self.account_health = health;
        self.update_activity()?;
        Ok(())
    }
    
    pub fn update_liquidation_price(&mut self, price: u64) -> Result<()> {
        self.liquidation_price = price;
        self.update_activity()?;
        Ok(())
    }
    
    pub fn deactivate(&mut self) -> Result<()> {
        require!(self.total_positions == 0, crate::errors::ErrorCode::AccountHasPositions);
        require!(self.total_orders == 0, crate::errors::ErrorCode::AccountHasOrders);
        self.is_active = false;
        self.update_activity()?;
        Ok(())
    }
    
    pub fn can_trade(&self) -> bool {
        self.is_active && self.total_collateral > 0
    }
    
    pub fn can_deposit(&self) -> bool {
        self.is_active
    }
    
    pub fn can_withdraw(&self) -> bool {
        self.is_active && self.total_collateral > 0 && self.total_positions == 0
    }
    
    pub fn is_at_risk(&self) -> bool {
        self.account_health < 5000 // Less than 50% health
    }
    
    pub fn is_liquidatable(&self) -> bool {
        self.account_health < 2000 // Less than 20% health
    }
    
    // Funding rate methods
    pub fn add_funding_payment(&mut self, amount: i64) -> Result<()> {
        if amount > 0 {
            self.total_funding_received = self.total_funding_received.checked_add(amount).unwrap();
        } else {
            self.total_funding_paid = self.total_funding_paid.checked_add((-amount) as i64).unwrap();
        }
        self.update_activity()?;
        Ok(())
    }
    
    // Fee tracking methods
    pub fn add_fee(&mut self, amount: u64) -> Result<()> {
        self.total_fees_paid = self.total_fees_paid.checked_add(amount).unwrap();
        self.update_activity()?;
        Ok(())
    }
    
    pub fn add_rebate(&mut self, amount: u64) -> Result<()> {
        self.total_rebates_earned = self.total_rebates_earned.checked_add(amount).unwrap();
        self.update_activity()?;
        Ok(())
    }
    
    // Margin calculation methods (Drift-style)
    pub fn calculate_margin_requirement(&self, position_value: u64, leverage: u16) -> u64 {
        position_value.checked_mul(10000).unwrap().checked_div(leverage as u64).unwrap()
    }
    
    pub fn update_available_margin(&mut self, collateral: u64, used_margin: u64) -> Result<()> {
        self.available_margin = collateral.checked_sub(used_margin).unwrap_or(0);
        self.update_activity()?;
        Ok(())
    }
    
    pub fn can_open_position(&self, required_margin: u64) -> bool {
        self.is_active && self.available_margin >= required_margin && self.total_positions < self.max_positions
    }
}

#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub enum UserAction {
    CreatePosition,
    ClosePosition,
    AddCollateral,
    RemoveCollateral,
    PlaceOrder,
    CancelOrder,
    UpdateLeverage,
    LiquidatePosition,
    UpdateAccount,
}
