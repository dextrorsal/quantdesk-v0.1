use anchor_lang::prelude::*;

/// Errors Module
/// Contains all program error codes for better organization and maintainability

#[error_code]
pub enum ErrorCode {
    #[msg("Invalid leverage amount")]
    InvalidLeverage,
    #[msg("Invalid position size")]
    InvalidSize,
    #[msg("Insufficient collateral")]
    InsufficientCollateral,
    #[msg("Max positions reached")]
    MaxPositionsReached,
    #[msg("Position is healthy, no liquidation needed")]
    PositionHealthy,
    #[msg("Position is not liquidatable")]
    PositionNotLiquidatable,
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
    #[msg("Unauthorized keeper")]
    UnauthorizedKeeper,
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
    #[msg("Invalid duration")]
    InvalidDuration,
    #[msg("Invalid interval")]
    InvalidInterval,
    #[msg("Invalid target price")]
    InvalidTargetPrice,
    #[msg("Invalid amount")]
    InvalidAmount,
    #[msg("Collateral account inactive")]
    CollateralAccountInactive,
    #[msg("No collateral provided")]
    NoCollateralProvided,
    // User Account Management Errors
    #[msg("No positions to remove")]
    NoPositionsToRemove,
    #[msg("No orders to remove")]
    NoOrdersToRemove,
    #[msg("Invalid health value")]
    InvalidHealthValue,
    #[msg("Account has open positions")]
    AccountHasPositions,
    #[msg("Account has active orders")]
    AccountHasOrders,
    #[msg("Account is not active")]
    AccountInactive,
    #[msg("Account already exists")]
    AccountAlreadyExists,
    #[msg("Account not found")]
    AccountNotFound,
    // Token-specific error codes
    #[msg("Invalid token amount")]
    InvalidTokenAmount,
    #[msg("Vault is inactive")]
    VaultInactive,
    #[msg("Insufficient vault balance")]
    InsufficientVaultBalance,
    #[msg("Unauthorized token authority")]
    UnauthorizedTokenAuthority,
    #[msg("Unauthorized token user")]
    UnauthorizedTokenUser,
    #[msg("Token account not found")]
    TokenAccountNotFound,
    #[msg("Invalid mint address")]
    InvalidMintAddress,
    #[msg("PDA derivation failed")]
    PdaDerivationFailed,
    #[msg("Invalid PDA owner")]
    InvalidPdaOwner,
    // Advanced Features Error Codes
    #[msg("Invalid insurance fund operation")]
    InvalidInsuranceFundOperation,
    #[msg("Insufficient insurance fund balance")]
    InsufficientInsuranceFundBalance,
    #[msg("Program is paused")]
    ProgramPaused,
    #[msg("Invalid fee parameters")]
    InvalidFeeParameters,
    #[msg("Invalid market parameters")]
    InvalidMarketParameters,
    #[msg("Invalid collateral configuration")]
    InvalidCollateralConfig,
    #[msg("Invalid oracle weight")]
    InvalidOracleWeight,
    #[msg("Oracle feed not found")]
    OracleFeedNotFound,
    #[msg("Unauthorized admin operation")]
    UnauthorizedAdminOperation,
    
    // ===== NEW ERROR CODES FOR ADVANCED FEATURES =====
    
    // Keeper Network Errors
    #[msg("Insufficient keeper stake")]
    InsufficientKeeperStake,
    #[msg("Keeper not registered")]
    KeeperNotRegistered,
    #[msg("Keeper is inactive")]
    KeeperInactive,
    #[msg("Invalid performance score")]
    InvalidPerformanceScore,
    
    // Circuit Breaker Errors
    #[msg("Circuit breaker is already triggered")]
    CircuitBreakerAlreadyTriggered,
    #[msg("Circuit breaker is not triggered")]
    CircuitBreakerNotTriggered,
    
    // JIT Liquidity Errors
    #[msg("Invalid fee rate")]
    InvalidFeeRate,
    #[msg("Insufficient JIT liquidity")]
    InsufficientJitLiquidity,
    #[msg("JIT provider not active")]
    JitProviderInactive,
    
    // Market Maker Vault Errors
    #[msg("Invalid vault strategy")]
    InvalidVaultStrategy,
    #[msg("Insufficient vault capital")]
    InsufficientVaultCapital,
    
    // Points System Errors
    #[msg("Invalid points multiplier")]
    InvalidPointsMultiplier,
    #[msg("User not found in points system")]
    UserNotFoundInPointsSystem,
    
    // ===== ADVANCED ORDER ERROR CODES =====
    
    #[msg("Invalid TWAP parameters")]
    InvalidTwapParameters,
    #[msg("Invalid order type")]
    InvalidOrderType,
    
    // ===== CROSS-COLLATERALIZATION ERROR CODES =====
    
    #[msg("Collateral type is not active")]
    CollateralTypeNotActive,
    #[msg("Exceeds maximum collateral amount")]
    ExceedsMaxCollateral,
    #[msg("Collateral asset not found")]
    CollateralAssetNotFound,
    #[msg("Insufficient health factor")]
    InsufficientHealthFactor,
    #[msg("Invalid weight")]
    InvalidWeight,
    
    // ===== SECURITY MODULE ERROR CODES =====
    
    #[msg("Circuit breaker is active - trading paused")]
    CircuitBreakerActive,
    #[msg("Emergency pause is active")]
    EmergencyPauseActive,
    #[msg("Oracle staleness protection triggered")]
    OracleStalenessProtectionTriggered,
    #[msg("Price volatility exceeds threshold")]
    PriceVolatilityExceeded,
    #[msg("Volume spike exceeds threshold")]
    VolumeSpikeExceeded,
    #[msg("Oracle deviation exceeds threshold")]
    OracleDeviationExceeded,
    #[msg("System overload detected")]
    SystemOverloadDetected,
    #[msg("Keeper authorization failed")]
    KeeperAuthorizationFailed,
    #[msg("Liquidation rate limit exceeded")]
    LiquidationRateLimitExceeded,
    #[msg("Invalid security parameters")]
    InvalidSecurityParameters,
    #[msg("Security module not initialized")]
    SecurityModuleNotInitialized,
    #[msg("Oracle health check failed")]
    OracleHealthCheckFailed,
    #[msg("Emergency price not available")]
    EmergencyPriceNotAvailable,
    #[msg("Too many keepers authorized")]
    TooManyKeepers,
}
