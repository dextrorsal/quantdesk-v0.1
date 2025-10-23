# 🚨 **CRITICAL ARCHITECTURAL DECISIONS**

## **1. Circuit Breaker Thresholds**
- **Price Deviation**: 5% (500 basis points)
- **Volume Spike**: 10x normal volume
- **Liquidation Rate**: 100 liquidations per 5 minutes
- **Oracle Staleness**: Dynamic (30-300 seconds based on conditions)

## **2. Keeper Security Requirements**
- **Minimum Stake**: 10 SOL
- **Performance Threshold**: 80% (800/1000)
- **Rate Limiting**: 50 liquidations per hour
- **Multi-Sig Required**: For liquidations > 10 SOL

## **3. Oracle Health Management**
- **Base Staleness**: 5 minutes
- **Dynamic Adjustment**: Based on load and volatility
- **Fallback Activation**: After 3 consecutive stale updates
- **Consensus Requirement**: 2+ oracles with <5% deviation

---
