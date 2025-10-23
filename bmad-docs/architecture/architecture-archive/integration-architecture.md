# 🏗️ **INTEGRATION ARCHITECTURE**

## **Unified Security System Integration**
```rust
/// Master security orchestrator
pub struct SecurityOrchestrator {
    pub circuit_breaker: PriceCircuitBreaker,
    pub liquidation_breaker: LiquidationCircuitBreaker,
    pub oracle_health_breaker: OracleHealthCircuitBreaker,
    pub dynamic_oracle_health: DynamicOracleHealth,
    pub multi_oracle_fallback: MultiOracleFallback,
    pub keeper_network: KeeperNetwork,
}

impl SecurityOrchestrator {
    /// Execute secure liquidation with all protections
    pub fn execute_secure_liquidation(
        &mut self,
        keeper_pubkey: &Pubkey,
        position_id: u64,
        liquidation_amount: u64,
        multi_sig_signatures: Option<&[Pubkey]>,
    ) -> Result<()> {
        let current_time = Clock::get()?.unix_timestamp;
        
        // 1. Check circuit breakers
        if self.circuit_breaker.is_triggered {
            return Err(ErrorCode::CircuitBreakerTriggered.into());
        }
        
        if self.liquidation_breaker.is_triggered {
            return Err(ErrorCode::LiquidationCircuitBreakerTriggered.into());
        }
        
        // 2. Check oracle health
        self.dynamic_oracle_health.update_oracle_health()?;
        if self.dynamic_oracle_health.fallback_oracle_active {
            // Use fallback oracle for price
            let consensus_price = self.multi_oracle_fallback.get_consensus_price()?;
            // Update market with consensus price
        }
        
        // 3. Verify keeper authorization
        if !self.keeper_network.is_authorized_keeper_secure(
            keeper_pubkey,
            liquidation_amount,
            multi_sig_signatures,
        )? {
            return Err(ErrorCode::UnauthorizedKeeper.into());
        }
        
        // 4. Execute liquidation
        self.keeper_network.execute_secure_liquidation(
            keeper_pubkey,
            liquidation_amount,
            position_id,
        )?;
        
        // 5. Update liquidation circuit breaker
        if self.liquidation_breaker.check_liquidation_rate()? {
            self.liquidation_breaker.trigger_liquidation_breaker()?;
        }
        
        Ok(())
    }
}
```

---
