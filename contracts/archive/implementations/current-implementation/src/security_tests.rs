//! Comprehensive Security Architecture Tests
//! Tests for Phase 1 security-hardened architecture components

use anchor_lang::prelude::*;
use crate::security::*;
use crate::ErrorCode;

#[cfg(test)]
mod security_tests {
    use super::*;
    use anchor_lang::prelude::*;

    /// Test Multi-Layer Circuit Breaker System
    #[test]
    fn test_circuit_breaker_price_volatility() {
        let mut circuit_breaker = SecurityCircuitBreaker::new();
        
        // Test normal price movement (should not trigger)
        let result = circuit_breaker.check_circuit_breakers(
            1000000, // $1000.00
            1000000, // volume
            1000000, // primary oracle
            1000000, // secondary oracle
            5000,    // 50% system load
        );
        assert!(result.is_ok());
        assert!(!result.unwrap());
        
        // Test extreme price movement (should trigger)
        let result = circuit_breaker.check_circuit_breakers(
            1200000, // $1200.00 (20% increase)
            1000000, // volume
            1000000, // primary oracle
            1000000, // secondary oracle
            5000,    // 50% system load
        );
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should trigger circuit breaker
    }

    #[test]
    fn test_circuit_breaker_volume_spike() {
        let mut circuit_breaker = SecurityCircuitBreaker::new();
        
        // Test normal volume (should not trigger)
        let result = circuit_breaker.check_circuit_breakers(
            1000000, // price
            1000000, // normal volume
            1000000, // primary oracle
            1000000, // secondary oracle
            5000,    // 50% system load
        );
        assert!(result.is_ok());
        assert!(!result.unwrap());
        
        // Test volume spike (should trigger)
        let result = circuit_breaker.check_circuit_breakers(
            1000000, // price
            2000000, // doubled volume
            1000000, // primary oracle
            1000000, // secondary oracle
            5000,    // 50% system load
        );
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should trigger circuit breaker
    }

    #[test]
    fn test_circuit_breaker_oracle_deviation() {
        let mut circuit_breaker = SecurityCircuitBreaker::new();
        
        // Test normal oracle prices (should not trigger)
        let result = circuit_breaker.check_circuit_breakers(
            1000000, // price
            1000000, // volume
            1000000, // primary oracle
            1000000, // secondary oracle (same price)
            5000,    // 50% system load
        );
        assert!(result.is_ok());
        assert!(!result.unwrap());
        
        // Test oracle deviation (should trigger)
        let result = circuit_breaker.check_circuit_breakers(
            1000000, // price
            1000000, // volume
            1000000, // primary oracle
            1200000, // secondary oracle (20% deviation)
            5000,    // 50% system load
        );
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should trigger circuit breaker
    }

    #[test]
    fn test_circuit_breaker_system_overload() {
        let mut circuit_breaker = SecurityCircuitBreaker::new();
        
        // Test normal system load (should not trigger)
        let result = circuit_breaker.check_circuit_breakers(
            1000000, // price
            1000000, // volume
            1000000, // primary oracle
            1000000, // secondary oracle
            5000,    // 50% system load
        );
        assert!(result.is_ok());
        assert!(!result.unwrap());
        
        // Test system overload (should trigger)
        let result = circuit_breaker.check_circuit_breakers(
            1000000, // price
            1000000, // volume
            1000000, // primary oracle
            1000000, // secondary oracle
            9000,    // 90% system load
        );
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should trigger circuit breaker
    }

    /// Test Enhanced Keeper Authorization Security
    #[test]
    fn test_keeper_authorization() {
        let mut keeper_manager = KeeperSecurityManager::new();
        
        // Test authorizing a keeper
        let keeper_pubkey = Pubkey::new_unique();
        let result = keeper_manager.authorize_keeper(
            keeper_pubkey,
            2000000000, // 2 SOL stake
            900,        // 90% performance score
            KeeperAuthLevel::Basic,
        );
        assert!(result.is_ok());
        
        // Test keeper authorization check
        let is_authorized = keeper_manager.is_keeper_authorized(&keeper_pubkey);
        assert!(is_authorized.is_ok());
        assert!(is_authorized.unwrap());
        
        // Test unauthorized keeper
        let unauthorized_keeper = Pubkey::new_unique();
        let is_authorized = keeper_manager.is_keeper_authorized(&unauthorized_keeper);
        assert!(is_authorized.is_err());
    }

    #[test]
    fn test_keeper_rate_limiting() {
        let mut keeper_manager = KeeperSecurityManager::new();
        
        // Test rate limit check (should pass initially)
        let rate_limit_ok = keeper_manager.check_liquidation_rate_limit();
        assert!(rate_limit_ok.is_ok());
        assert!(rate_limit_ok.unwrap());
        
        // Simulate multiple liquidations to exceed rate limit
        for _ in 0..keeper_manager.liquidation_rate_limit {
            let result = keeper_manager.record_liquidation(
                Pubkey::new_unique(),
                Pubkey::new_unique(),
                1000000,
                1000000,
                true,
                LiquidationReason::InsufficientMargin,
            );
            assert!(result.is_ok());
        }
        
        // Test rate limit exceeded
        let rate_limit_ok = keeper_manager.check_liquidation_rate_limit();
        assert!(rate_limit_ok.is_ok());
        assert!(!rate_limit_ok.unwrap()); // Should be rate limited
    }

    #[test]
    fn test_keeper_performance_tracking() {
        let mut keeper_manager = KeeperSecurityManager::new();
        
        let keeper_pubkey = Pubkey::new_unique();
        keeper_manager.authorize_keeper(
            keeper_pubkey,
            2000000000, // 2 SOL stake
            1000,       // 100% performance score
            KeeperAuthLevel::Basic,
        ).unwrap();
        
        // Record successful liquidation
        keeper_manager.record_liquidation(
            keeper_pubkey,
            Pubkey::new_unique(),
            1000000,
            1000000,
            true,
            LiquidationReason::InsufficientMargin,
        ).unwrap();
        
        // Record failed liquidation
        keeper_manager.record_liquidation(
            keeper_pubkey,
            Pubkey::new_unique(),
            1000000,
            1000000,
            false,
            LiquidationReason::InsufficientMargin,
        ).unwrap();
        
        // Check that performance score was updated
        let keeper = keeper_manager.authorized_keepers.iter()
            .find(|k| k.keeper_pubkey == keeper_pubkey)
            .unwrap();
        
        assert_eq!(keeper.total_liquidations, 2);
        assert_eq!(keeper.successful_liquidations, 1);
        assert_eq!(keeper.failed_liquidations, 1);
        assert_eq!(keeper.performance_score, 500); // 50% success rate
    }

    /// Test Dynamic Oracle Staleness Protection
    #[test]
    fn test_oracle_staleness_protection() {
        let mut oracle_protection = OracleStalenessProtection::new();
        
        let current_time = 1000000; // Mock timestamp
        
        // Test healthy oracle (should pass)
        let health_status = oracle_protection.check_oracle_health(
            1000000, // price
            current_time, // current timestamp
            OracleType::Pyth,
        );
        assert!(health_status.is_ok());
        assert!(matches!(health_status.unwrap(), OracleHealthStatus::Healthy));
        
        // Test stale oracle (should trigger warning)
        let stale_time = current_time - 200; // 200 seconds old
        let health_status = oracle_protection.check_oracle_health(
            1000000, // price
            stale_time, // stale timestamp
            OracleType::Pyth,
        );
        assert!(health_status.is_ok());
        assert!(matches!(health_status.unwrap(), OracleHealthStatus::Warning));
        
        // Test critically stale oracle (should trigger critical)
        let critical_time = current_time - 300; // 300 seconds old
        let health_status = oracle_protection.check_oracle_health(
            1000000, // price
            critical_time, // critically stale timestamp
            OracleType::Pyth,
        );
        assert!(health_status.is_ok());
        assert!(matches!(health_status.unwrap(), OracleHealthStatus::Critical));
    }

    #[test]
    fn test_oracle_price_validation() {
        let oracle_protection = OracleStalenessProtection::new();
        
        // Test valid price change (should pass)
        let is_valid = oracle_protection.validate_price_change(1000000, 1010000); // 1% change
        assert!(is_valid.is_ok());
        assert!(is_valid.unwrap());
        
        // Test invalid price change (should fail)
        let is_valid = oracle_protection.validate_price_change(1000000, 2000000); // 100% change
        assert!(is_valid.is_ok());
        assert!(!is_valid.unwrap());
    }

    #[test]
    fn test_oracle_emergency_fallback() {
        let mut oracle_protection = OracleStalenessProtection::new();
        
        // Set emergency price
        let emergency_price = 1500000; // $1500.00
        let result = oracle_protection.set_emergency_price(emergency_price);
        assert!(result.is_ok());
        
        // Get emergency price
        let emergency_price_result = oracle_protection.get_emergency_price();
        assert!(emergency_price_result.is_ok());
        assert_eq!(emergency_price_result.unwrap(), Some(emergency_price));
    }

    #[test]
    fn test_oracle_best_price_selection() {
        let mut oracle_protection = OracleStalenessProtection::new();
        
        let current_time = 1000000;
        
        // Test with healthy primary oracle (should use primary)
        let best_price = oracle_protection.get_best_price(
            1000000, // primary price
            current_time, // primary timestamp
            Some(1200000), // secondary price
            Some(current_time - 100), // secondary timestamp (stale)
        );
        assert!(best_price.is_ok());
        assert_eq!(best_price.unwrap(), 1000000); // Should use primary
        
        // Test with stale primary, healthy secondary (should use secondary)
        let best_price = oracle_protection.get_best_price(
            1000000, // primary price
            current_time - 400, // primary timestamp (stale)
            Some(1200000), // secondary price
            Some(current_time), // secondary timestamp (fresh)
        );
        assert!(best_price.is_ok());
        assert_eq!(best_price.unwrap(), 1200000); // Should use secondary
    }

    /// Test Integration Scenarios
    #[test]
    fn test_security_integration_scenario() {
        // Initialize all security components
        let mut circuit_breaker = SecurityCircuitBreaker::new();
        let mut keeper_manager = KeeperSecurityManager::new();
        let mut oracle_protection = OracleStalenessProtection::new();
        
        // Authorize a keeper
        let keeper_pubkey = Pubkey::new_unique();
        keeper_manager.authorize_keeper(
            keeper_pubkey,
            2000000000, // 2 SOL stake
            900,        // 90% performance score
            KeeperAuthLevel::Basic,
        ).unwrap();
        
        // Set up emergency price
        oracle_protection.set_emergency_price(1000000).unwrap();
        
        // Test normal trading scenario (should pass all checks)
        let circuit_breaker_result = circuit_breaker.check_circuit_breakers(
            1000000, // normal price
            1000000, // normal volume
            1000000, // primary oracle
            1000000, // secondary oracle
            5000,    // normal system load
        );
        assert!(circuit_breaker_result.is_ok());
        assert!(!circuit_breaker_result.unwrap()); // Should not trigger
        
        let keeper_auth_result = keeper_manager.is_keeper_authorized(&keeper_pubkey);
        assert!(keeper_auth_result.is_ok());
        assert!(keeper_auth_result.unwrap()); // Should be authorized
        
        let oracle_health = oracle_protection.check_oracle_health(
            1000000,
            1000000, // current time
            OracleType::Pyth,
        );
        assert!(oracle_health.is_ok());
        assert!(matches!(oracle_health.unwrap(), OracleHealthStatus::Healthy));
        
        // Test emergency scenario (should trigger circuit breaker)
        let circuit_breaker_result = circuit_breaker.check_circuit_breakers(
            2000000, // extreme price (100% increase)
            2000000, // extreme volume
            2000000, // primary oracle
            1000000, // secondary oracle (deviation)
            9000,    // high system load
        );
        assert!(circuit_breaker_result.is_ok());
        assert!(circuit_breaker_result.unwrap()); // Should trigger circuit breaker
    }

    /// Test Gas Efficiency
    #[test]
    fn test_gas_efficiency_circuit_breaker() {
        let mut circuit_breaker = SecurityCircuitBreaker::new();
        
        // Test multiple rapid checks (should be gas efficient)
        for i in 0..100 {
            let price = 1000000 + (i * 1000); // Gradual price increase
            let result = circuit_breaker.check_circuit_breakers(
                price,
                1000000, // volume
                1000000, // primary oracle
                1000000, // secondary oracle
                5000,    // system load
            );
            assert!(result.is_ok());
        }
        
        // Verify circuit breaker is still functional
        assert!(circuit_breaker.price_volatility_breaker.is_active);
        assert!(circuit_breaker.volume_spike_breaker.is_active);
        assert!(circuit_breaker.oracle_deviation_breaker.is_active);
        assert!(circuit_breaker.system_overload_breaker.is_active);
    }

    #[test]
    fn test_gas_efficiency_keeper_manager() {
        let mut keeper_manager = KeeperSecurityManager::new();
        
        // Test multiple keeper operations (should be gas efficient)
        for i in 0..20 {
            let keeper_pubkey = Pubkey::new_unique();
            let result = keeper_manager.authorize_keeper(
                keeper_pubkey,
                1000000000, // 1 SOL stake
                800 + (i as u16 * 10), // varying performance scores
                KeeperAuthLevel::Basic,
            );
            assert!(result.is_ok());
        }
        
        // Verify all keepers are authorized
        assert_eq!(keeper_manager.keeper_count, 20);
    }

    #[test]
    fn test_gas_efficiency_oracle_protection() {
        let mut oracle_protection = OracleStalenessProtection::new();
        
        // Test multiple oracle health checks (should be gas efficient)
        for i in 0..50 {
            let timestamp = 1000000 + (i * 10);
            let health_status = oracle_protection.check_oracle_health(
                1000000 + (i * 1000), // varying prices
                timestamp as i64, // Convert u64 to i64
                OracleType::Pyth,
            );
            assert!(health_status.is_ok());
        }
        
        // Verify oracle protection is still functional
        assert_eq!(oracle_protection.health_history_index, 50 % 20);
    }
}

/// Integration test with real-world scenarios
#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_flash_crash_protection() {
        let mut circuit_breaker = SecurityCircuitBreaker::new();
        
        // Simulate flash crash scenario
        let normal_price = 1000000; // $1000.00
        let crash_price = 500000;   // $500.00 (50% drop)
        
        // Normal price should not trigger
        let result = circuit_breaker.check_circuit_breakers(
            normal_price,
            1000000, // volume
            normal_price, // primary oracle
            normal_price, // secondary oracle
            5000,    // system load
        );
        assert!(result.is_ok());
        assert!(!result.unwrap());
        
        // Flash crash should trigger circuit breaker
        let result = circuit_breaker.check_circuit_breakers(
            crash_price,
            2000000, // increased volume
            crash_price, // primary oracle
            normal_price, // secondary oracle (deviation)
            8000,    // high system load
        );
        assert!(result.is_ok());
        assert!(result.unwrap()); // Should trigger circuit breaker
    }

    #[test]
    fn test_oracle_manipulation_protection() {
        let mut oracle_protection = OracleStalenessProtection::new();
        
        // Test oracle manipulation attempt
        let normal_price = 1000000; // $1000.00
        let manipulated_price = 2000000; // $2000.00 (100% increase)
        
        // Normal price should be valid
        let is_valid = oracle_protection.validate_price_change(normal_price, normal_price);
        assert!(is_valid.is_ok());
        assert!(is_valid.unwrap());
        
        // Manipulated price should be invalid
        let is_valid = oracle_protection.validate_price_change(normal_price, manipulated_price);
        assert!(is_valid.is_ok());
        assert!(!is_valid.unwrap());
    }

    #[test]
    fn test_keeper_attack_protection() {
        let mut keeper_manager = KeeperSecurityManager::new();
        
        // Test unauthorized keeper attempt
        let unauthorized_keeper = Pubkey::new_unique();
        let is_authorized = keeper_manager.is_keeper_authorized(&unauthorized_keeper);
        assert!(is_authorized.is_err());
        
        // Test rate limiting attack
        let authorized_keeper = Pubkey::new_unique();
        keeper_manager.authorize_keeper(
            authorized_keeper,
            1000000000, // 1 SOL stake
            900,        // 90% performance score
            KeeperAuthLevel::Basic,
        ).unwrap();
        
        // Attempt to exceed rate limit
        for _ in 0..keeper_manager.liquidation_rate_limit + 1 {
            let result = keeper_manager.record_liquidation(
                authorized_keeper,
                Pubkey::new_unique(),
                1000000,
                1000000,
                true,
                LiquidationReason::InsufficientMargin,
            );
            assert!(result.is_ok());
        }
        
        // Rate limit should be exceeded
        let rate_limit_ok = keeper_manager.check_liquidation_rate_limit();
        assert!(rate_limit_ok.is_ok());
        assert!(!rate_limit_ok.unwrap());
    }
}
