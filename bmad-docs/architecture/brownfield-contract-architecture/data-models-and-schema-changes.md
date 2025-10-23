# Data Models and Schema Changes

## New Data Models

### Enhanced Market Configuration
**Purpose:** Extend existing market management with advanced features
**Integration:** Extends existing `Market` struct in `state/market.rs`

**Key Attributes:**
- `advanced_order_types`: `Vec<OrderType>` - Support for new order types
- `risk_parameters`: `RiskParameters` - Enhanced risk management
- `liquidity_pools`: `Vec<LiquidityPool>` - JIT liquidity integration

**Relationships:**
- **With Existing:** Extends `Market` struct, integrates with `Position` and `Order` systems
- **With New:** Connects to enhanced security and oracle systems

### Cross-Collateral Enhancement
**Purpose:** Extend existing cross-collateral system with advanced features
**Integration:** Builds upon existing `CrossCollateralAccount` in `state/advanced.rs`

**Key Attributes:**
- `dynamic_weights`: `DynamicWeightConfig` - Adaptive collateral weights
- `risk_adjustment`: `RiskAdjustmentFactor` - Dynamic risk management
- `liquidation_buffer`: `LiquidationBuffer` - Enhanced liquidation protection

**Relationships:**
- **With Existing:** Extends `CollateralAccount` and `UserAccount` systems
- **With New:** Integrates with enhanced security and oracle systems

## Schema Integration Strategy

**Database Changes Required:**
- **New Tables:** Enhanced market configurations, advanced collateral settings
- **Modified Tables:** Extend existing `Market`, `UserAccount`, `CollateralAccount` structs
- **New Indexes:** Performance optimization for new query patterns
- **Migration Strategy:** Incremental updates preserving existing data

**Backward Compatibility:**
- All existing account structures remain unchanged
- New fields added with default values
- Existing instructions maintain full compatibility

---
