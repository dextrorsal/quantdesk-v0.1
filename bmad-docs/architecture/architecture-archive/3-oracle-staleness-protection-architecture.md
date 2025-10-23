# ⏰ **3. ORACLE STALENESS PROTECTION ARCHITECTURE**

## **Problem Statement**
Current oracle staleness protection is **insufficient under load**:
- Fixed 5-minute staleness threshold
- No dynamic adjustment based on market conditions
- No fallback oracle coordination
- No load-based staleness detection

## **Architectural Solution: Dynamic Oracle Health Management**

### **Dynamic Staleness Detection**
```rust
pub struct DynamicOracleHealth {
    pub base_staleness_threshold: u64,       // Base threshold (e.g., 300 seconds)
    pub load_multiplier: f64,               // Dynamic multiplier based on load
    pub market_volatility_factor: f64,      // Volatility-based adjustment
    pub oracle_health_score: u16,           // 0-1000 health score
    pub consecutive_stale_updates: u8,      // Track consecutive stale updates
    pub fallback_oracle_active: bool,       // Fallback oracle status
    pub last_health_check: i64,
    pub health_check_interval: u64,
}

impl DynamicOracleHealth {
    /// Calculate dynamic staleness threshold based on current conditions
    pub fn calculate_dynamic_threshold(
        &self,
        current_load: f64,                  // Current system load (0.0-1.0)
        market_volatility: f64,             // Market volatility (0.0-1.0)
        oracle_response_time: u64,          // Average oracle response time in ms
    ) -> Result<u64> {
        let base_threshold = self.base_staleness_threshold as f64;
        
        // Load-based adjustment (higher load = stricter threshold)
        let load_adjustment = if current_load > 0.8 {
            0.5 // 50% of base threshold under high load
        } else if current_load > 0.6 {
            0.7 // 70% of base threshold under medium load
        } else {
            1.0 // Full threshold under normal load
        };
        
        // Volatility-based adjustment (higher volatility = stricter threshold)
        let volatility_adjustment = if market_volatility > 0.7 {
            0.6 // 60% of base threshold in high volatility
        } else if market_volatility > 0.4 {
            0.8 // 80% of base threshold in medium volatility
        } else {
            1.0 // Full threshold in low volatility
        };
        
        // Response time adjustment
        let response_adjustment = if oracle_response_time > 5000 { // > 5 seconds
            0.5 // 50% of base threshold for slow responses
        } else if oracle_response_time > 2000 { // > 2 seconds
            0.8 // 80% of base threshold for medium responses
        } else {
            1.0 // Full threshold for fast responses
        };
        
        // Calculate final threshold
        let final_threshold = base_threshold 
            * load_adjustment 
            * volatility_adjustment 
            * response_adjustment;
        
        // Ensure minimum threshold of 30 seconds
        Ok(final_threshold.max(30.0) as u64)
    }
    
    /// Check oracle staleness with dynamic threshold
    pub fn check_dynamic_staleness(
        &mut self,
        oracle_timestamp: i64,
        current_load: f64,
        market_volatility: f64,
        oracle_response_time: u64,
    ) -> Result<bool> {
        let current_time = Clock::get()?.unix_timestamp;
        let staleness = current_time - oracle_timestamp;
        
        // Calculate dynamic threshold
        let dynamic_threshold = self.calculate_dynamic_threshold(
            current_load,
            market_volatility,
            oracle_response_time,
        )?;
        
        // Check staleness
        let is_stale = staleness > dynamic_threshold as i64;
        
        if is_stale {
            self.consecutive_stale_updates += 1;
            
            // Update health score
            self.oracle_health_score = self.oracle_health_score.saturating_sub(50);
            
            // Activate fallback oracle after 3 consecutive stale updates
            if self.consecutive_stale_updates >= 3 {
                self.fallback_oracle_active = true;
                
                emit!(OracleFallbackActivated {
                    consecutive_stale_updates: self.consecutive_stale_updates,
                    health_score: self.oracle_health_score,
                    dynamic_threshold,
                });
            }
        } else {
            // Reset consecutive stale updates on successful update
            self.consecutive_stale_updates = 0;
            
            // Improve health score
            self.oracle_health_score = self.oracle_health_score.saturating_add(10);
            
            // Deactivate fallback oracle if health improves
            if self.oracle_health_score > 800 {
                self.fallback_oracle_active = false;
            }
        }
        
        self.last_health_check = current_time;
        Ok(is_stale)
    }
}
```

### **Multi-Oracle Fallback System**
```rust
pub struct MultiOracleFallback {
    pub primary_oracle: Pubkey,             // Pyth Network
    pub secondary_oracle: Pubkey,           // Switchboard
    pub tertiary_oracle: Pubkey,            // Custom oracle
    pub oracle_weights: Vec<u16>,           // Weight for each oracle (basis points)
    pub consensus_threshold: u16,           // Minimum consensus percentage
    pub last_consensus_check: i64,
    pub oracle_prices: Vec<OraclePriceData>,
}

#[derive(Clone)]
pub struct OraclePriceData {
    pub oracle_id: Pubkey,
    pub price: u64,
    pub confidence: u64,
    pub timestamp: i64,
    pub is_stale: bool,
    pub health_score: u16,
}

impl MultiOracleFallback {
    /// Get consensus price from multiple oracles
    pub fn get_consensus_price(&mut self) -> Result<u64> {
        let current_time = Clock::get()?.unix_timestamp;
        
        // Collect prices from all oracles
        let mut valid_prices = Vec::new();
        
        for (i, oracle_price) in self.oracle_prices.iter().enumerate() {
            if !oracle_price.is_stale && oracle_price.health_score > 500 {
                valid_prices.push((oracle_price.price, self.oracle_weights[i]));
            }
        }
        
        // Check if we have enough consensus
        if valid_prices.len() < 2 {
            return Err(ErrorCode::InsufficientOracleConsensus.into());
        }
        
        // Calculate weighted average
        let mut total_weight = 0u64;
        let mut weighted_sum = 0u64;
        
        for (price, weight) in valid_prices {
            weighted_sum += price * weight as u64;
            total_weight += weight as u64;
        }
        
        let consensus_price = weighted_sum / total_weight;
        
        // Validate consensus (prices within 5% of each other)
        let max_deviation = consensus_price * 5 / 100; // 5% deviation
        for (price, _) in valid_prices {
            if price > consensus_price + max_deviation || price < consensus_price - max_deviation {
                return Err(ErrorCode::OraclePriceDeviationTooHigh.into());
            }
        }
        
        self.last_consensus_check = current_time;
        Ok(consensus_price)
    }
    
    /// Update oracle health and activate fallbacks as needed
    pub fn update_oracle_health(&mut self) -> Result<()> {
        let current_time = Clock::get()?.unix_timestamp;
        
        for oracle_price in &mut self.oracle_prices {
            let staleness = current_time - oracle_price.timestamp;
            
            // Update staleness status
            oracle_price.is_stale = staleness > 300; // 5 minutes base threshold
            
            // Update health score based on staleness and confidence
            if oracle_price.is_stale {
                oracle_price.health_score = oracle_price.health_score.saturating_sub(100);
            } else {
                oracle_price.health_score = oracle_price.health_score.saturating_add(50);
            }
            
            // Cap health score
            if oracle_price.health_score > 1000 {
                oracle_price.health_score = 1000;
            }
        }
        
        Ok(())
    }
}
```

---
