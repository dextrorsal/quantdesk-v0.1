# 🔐 **2. KEEPER AUTHORIZATION SECURITY ARCHITECTURE**

## **Problem Statement**
Current keeper authorization has **critical security gaps**:
- No multi-signature requirements
- No time-based authorization limits
- No performance monitoring
- No stake slashing mechanisms
- No emergency revocation

## **Architectural Solution: Multi-Factor Keeper Security System**

### **Enhanced Keeper Authorization**
```rust
pub struct SecureKeeperInfo {
    pub keeper_pubkey: Pubkey,
    pub stake_amount: u64,
    pub performance_score: u16,
    pub is_active: bool,
    pub total_liquidations: u32,
    pub total_rewards_earned: u64,
    pub last_activity: i64,
    
    // Security enhancements
    pub authorization_expiry: i64,           // Time-based authorization
    pub max_liquidations_per_hour: u32,      // Rate limiting
    pub liquidations_this_hour: u32,
    pub hour_start_time: i64,
    pub slashing_risk_score: u16,           // 0-1000, higher = more risk
    pub emergency_revoked: bool,
    pub multi_sig_required: bool,
    pub cooldown_period: u64,               // After failed liquidation
    pub last_cooldown_start: i64,
}

impl SecureKeeperInfo {
    /// Check if keeper is authorized with enhanced security
    pub fn is_authorized_secure(&self, current_time: i64) -> Result<bool> {
        // Basic checks
        if !self.is_active || self.emergency_revoked {
            return Ok(false);
        }
        
        // Time-based authorization check
        if current_time > self.authorization_expiry {
            return Ok(false);
        }
        
        // Performance threshold check
        if self.performance_score < 800 { // 80% minimum performance
            return Ok(false);
        }
        
        // Stake requirement check
        if self.stake_amount < 10_000_000_000 { // 10 SOL minimum
            return Ok(false);
        }
        
        // Slashing risk check
        if self.slashing_risk_score > 500 { // 50% risk threshold
            return Ok(false);
        }
        
        // Cooldown period check
        if self.last_cooldown_start > 0 {
            let cooldown_elapsed = current_time - self.last_cooldown_start;
            if cooldown_elapsed < self.cooldown_period as i64 {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Update liquidation rate limiting
    pub fn update_liquidation_rate(&mut self, current_time: i64) -> Result<()> {
        // Reset hourly counter if hour has passed
        if current_time - self.hour_start_time >= 3600 { // 1 hour
            self.liquidations_this_hour = 0;
            self.hour_start_time = current_time;
        }
        
        // Check rate limit
        if self.liquidations_this_hour >= self.max_liquidations_per_hour {
            return Err(ErrorCode::KeeperRateLimitExceeded.into());
        }
        
        self.liquidations_this_hour += 1;
        Ok(())
    }
    
    /// Calculate slashing risk based on performance
    pub fn calculate_slashing_risk(&mut self) -> Result<()> {
        let base_risk = 100; // Base 10% risk
        
        // Performance penalty
        let performance_penalty = if self.performance_score < 900 {
            (900 - self.performance_score) * 2 // 2x penalty for each point below 90%
        } else {
            0
        };
        
        // Recent activity penalty
        let activity_penalty = if self.total_liquidations > 1000 {
            ((self.total_liquidations - 1000) / 100) * 10 // 10% penalty per 100 liquidations over 1000
        } else {
            0
        };
        
        self.slashing_risk_score = base_risk + performance_penalty + activity_penalty;
        
        // Cap at 1000 (100%)
        if self.slashing_risk_score > 1000 {
            self.slashing_risk_score = 1000;
        }
        
        Ok(())
    }
}
```

### **Multi-Signature Keeper Authorization**
```rust
pub struct KeeperMultiSig {
    pub keeper_pubkey: Pubkey,
    pub required_signatures: u8,
    pub authorized_signers: Vec<Pubkey>,
    pub signature_threshold: u8,
    pub emergency_override: Pubkey,
}

impl KeeperMultiSig {
    /// Verify multi-signature authorization for liquidation
    pub fn verify_multi_sig_authorization(
        &self,
        signatures: &[Pubkey],
        liquidation_amount: u64,
    ) -> Result<bool> {
        // Check signature count
        if signatures.len() < self.required_signatures as usize {
            return Ok(false);
        }
        
        // Check if all signers are authorized
        for signature in signatures {
            if !self.authorized_signers.contains(signature) {
                return Ok(false);
            }
        }
        
        // Check signature threshold for amount
        let required_sigs_for_amount = if liquidation_amount > 1_000_000_000 { // > 1 SOL
            self.signature_threshold
        } else {
            self.required_signatures
        };
        
        Ok(signatures.len() >= required_sigs_for_amount as usize)
    }
}
```

### **Enhanced Keeper Network Security**
```rust
impl KeeperNetwork {
    /// Secure keeper authorization with all security checks
    pub fn is_authorized_keeper_secure(
        &self,
        keeper_pubkey: &Pubkey,
        liquidation_amount: u64,
        multi_sig_signatures: Option<&[Pubkey]>,
    ) -> Result<bool> {
        let keeper = self.keepers.iter()
            .find(|k| k.keeper_pubkey == *keeper_pubkey)
            .ok_or(ErrorCode::KeeperNotRegistered)?;
        
        let current_time = Clock::get()?.unix_timestamp;
        
        // Basic authorization check
        if !keeper.is_authorized_secure(current_time)? {
            return Ok(false);
        }
        
        // Rate limiting check
        if keeper.liquidations_this_hour >= keeper.max_liquidations_per_hour {
            return Ok(false);
        }
        
        // Multi-signature check for large liquidations
        if liquidation_amount > 10_000_000_000 { // > 10 SOL
            if let Some(signatures) = multi_sig_signatures {
                let multi_sig = self.get_keeper_multi_sig(keeper_pubkey)?;
                if !multi_sig.verify_multi_sig_authorization(signatures, liquidation_amount)? {
                    return Ok(false);
                }
            } else {
                return Ok(false); // Multi-sig required for large liquidations
            }
        }
        
        // Stake requirement check for liquidation amount
        let required_stake = liquidation_amount * 2; // 2x liquidation amount as stake
        if keeper.stake_amount < required_stake {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    /// Secure liquidation execution with enhanced monitoring
    pub fn execute_secure_liquidation(
        &mut self,
        keeper_pubkey: &Pubkey,
        liquidation_amount: u64,
        position_id: u64,
    ) -> Result<()> {
        let keeper_index = self.keepers.iter()
            .position(|k| k.keeper_pubkey == *keeper_pubkey)
            .ok_or(ErrorCode::KeeperNotRegistered)?;
        
        let keeper = &mut self.keepers[keeper_index];
        let current_time = Clock::get()?.unix_timestamp;
        
        // Final authorization check
        if !keeper.is_authorized_secure(current_time)? {
            return Err(ErrorCode::UnauthorizedKeeper.into());
        }
        
        // Update liquidation rate
        keeper.update_liquidation_rate(current_time)?;
        
        // Increment liquidation count
        keeper.total_liquidations += 1;
        keeper.last_activity = current_time;
        
        // Update performance score based on liquidation success
        keeper.performance_score = keeper.performance_score.saturating_add(10); // Reward successful liquidation
        
        // Calculate new slashing risk
        keeper.calculate_slashing_risk()?;
        
        // Emit security event
        emit!(SecureLiquidationExecuted {
            keeper: *keeper_pubkey,
            liquidation_amount,
            position_id,
            timestamp: current_time,
            performance_score: keeper.performance_score,
            slashing_risk: keeper.slashing_risk_score,
        });
        
        Ok(())
    }
}
```

---
