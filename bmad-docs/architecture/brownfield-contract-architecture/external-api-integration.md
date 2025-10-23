# External API Integration

## Pyth Network Enhancement
- **Purpose:** Enhanced price feed integration with multi-source validation
- **Documentation:** https://pyth.network/
- **Base URL:** Pyth Network on-chain programs
- **Authentication:** On-chain program interaction
- **Integration Method:** Direct CPI calls with enhanced validation

**Key Endpoints Used:**
- `get_price()` - Enhanced price retrieval with staleness checks
- `validate_price()` - Cross-oracle price validation

**Error Handling:** Graceful fallback to emergency price mechanisms

## Switchboard Oracle Integration
- **Purpose:** Secondary oracle for price validation and redundancy
- **Documentation:** https://switchboard.xyz/
- **Base URL:** Switchboard on-chain programs
- **Authentication:** On-chain program interaction
- **Integration Method:** CPI integration with existing oracle system

**Key Endpoints Used:**
- `get_aggregated_price()` - Aggregated price from multiple sources
- `validate_price_deviation()` - Cross-oracle deviation checking

**Error Handling:** Integration with existing circuit breaker system

---
