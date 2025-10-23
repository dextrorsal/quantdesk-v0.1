# Smart Contracts Department Architecture

## Overview
Solana-based perpetual trading DEX using Anchor 0.32.1 with multi-oracle support (Pyth Network + FixedPrice for devnet). Program ID: HsSzXVuSwiqGQvocT2zMX8zc86KQ9TYfZFZcfmmejzso

## Technology Stack (Based on Real Implementation)
- **Blockchain**: Solana (currently deployed on devnet)
- **Framework**: Anchor 0.32.1 with init-if-needed feature
- **Language**: Rust 2021 Edition
- **Oracle Integration**: pyth-sdk-solana 0.10.0
- **SPL Integration**: anchor-spl 0.32.1 for token operations
- **Build System**: Cargo with custom features (cpi, no-entrypoint, idl-build)

## Actual Contract Structure
```
contracts/programs/quantdesk-perp-dex/ (11核心模块)
├── src/
│   ├── lib.rs                    # Main program (3,430 lines) - Core perpetual DEX
│   ├── user_accounts.rs          # User account management
│   ├── token_operations.rs       # SPL token operations (11,122 lines)
│   ├── positions.rs              # Position management (7,209 lines)
│   ├── markets.rs                # Market configuration (3,187 lines)
│   ├── orders.rs                 # Order management (3,509 lines)
│   ├── collateral.rs             # Collateral system (2,875 lines)
│   ├── oracle.rs                 # Multi-oracle support (Pyth + FixedPrice)
│   └── pda_utils.rs              # PDA utility functions (2,752 lines)
├── Cargo.toml                    # Dependencies and build configuration
├── tests/                        # Anchor test framework
├── migrations/                   # Deployment scripts
└── Anchor.toml                   # Anchor configuration
```

## Core Programs

### Perpetuals Program
- **Position Management**: Opening/closing positions
- **Funding Rate**: Automated funding calculations
- **Liquidation**: Risk-based liquidation system
- **Fees**: Trading fee calculations and collection

### Oracle Integration
- **Price Feeds**: Real-time asset prices from Pyth
- **Switchboard**: Custom price feeds for exotic assets
- **Price Verification**: Anti-manipulation mechanisms
- **Circuit Breakers**: Emergency price controls

### Margin System
- **Collateral Management**: Multi-collateral support
- **Risk Calculations**: Real-time risk metrics
- **Health Monitoring**: Position health tracking
- **Insurance Fund**: Protocol insurance mechanisms

## Security Architecture
- **Multi-sig Operations**: Timelock governance
- **Price Feed Redundancy**: Multiple oracle sources
- **Emergency Controls**: Pause mechanisms
- **Audit Trails**: Comprehensive event logging

## Development Guidelines
- Secure Rust programming practices
- Comprehensive test coverage (>95%)
- Formal verification for critical functions
- Gas optimization techniques
- Upgradability patterns

## Testing Strategy
- Local testing: Anchor test framework
- Fork testing: Solana mainnet forks
- Integration tests: Cross-program interactions
- Security audits: Professional audits + internal reviews
- Testnet deployment: Extended testing period

## Deployment Workflow
1. **Local Testing**: Full validation
2. **Devnet Testing**: Public testing environment
3. **Code Audit**: Professional security audit
4. **Testnet Validation**: Extended testnet period
5. **Mainnet Deployment**: Gradual rollout with monitoring
