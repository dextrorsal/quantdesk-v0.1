# Source Tree

## Existing Project Structure
```
contracts/
├── programs/
│   └── quantdesk-perp-dex/
│       ├── src/
│       │   ├── instructions/
│       │   │   ├── market_management.rs
│       │   │   ├── position_management.rs
│       │   │   ├── order_management.rs
│       │   │   ├── collateral_management.rs
│       │   │   ├── security_management.rs
│       │   │   └── ...
│       │   ├── state/
│       │   │   ├── market.rs
│       │   │   ├── position.rs
│       │   │   ├── order.rs
│       │   │   ├── user_account.rs
│       │   │   └── ...
│       │   ├── security.rs
│       │   ├── oracle.rs
│       │   └── lib.rs
│       └── Cargo.toml
├── tests/
├── Anchor.toml
└── Cargo.toml
```

## New File Organization
```
contracts/
├── programs/
│   └── quantdesk-perp-dex/
│       ├── src/
│       │   ├── instructions/
│       │   │   ├── market_management.rs       # Existing file
│       │   │   ├── position_management.rs     # Existing file
│       │   │   ├── order_management.rs         # Existing file
│       │   │   ├── collateral_management.rs   # Existing file
│       │   │   ├── security_management.rs      # Existing file
│       │   │   ├── enhanced_orders.rs          # New: Advanced order types
│       │   │   ├── dynamic_risk.rs             # New: Dynamic risk management
│       │   │   └── oracle_enhancement.rs       # New: Enhanced oracle integration
│       │   ├── state/
│       │   │   ├── market.rs                   # Existing file
│       │   │   ├── position.rs                 # Existing file
│       │   │   ├── order.rs                    # Existing file
│       │   │   ├── user_account.rs             # Existing file
│       │   │   ├── enhanced_market.rs          # New: Enhanced market state
│       │   │   ├── dynamic_risk.rs             # New: Dynamic risk state
│       │   │   └── oracle_config.rs            # New: Oracle configuration
│       │   ├── security.rs                     # Existing file
│       │   ├── oracle.rs                       # Existing file
│       │   ├── enhanced_security.rs            # New: Enhanced security features
│       │   └── lib.rs                          # Existing file (updated)
│       └── Cargo.toml                          # Existing file
├── tests/
│   ├── enhanced_orders_test.ts                 # New: Advanced order testing
│   ├── dynamic_risk_test.ts                    # New: Risk management testing
│   └── oracle_enhancement_test.ts              # New: Oracle testing
├── Anchor.toml                                 # Existing file
└── Cargo.toml                                  # Existing file
```

## Integration Guidelines

- **File Naming:** Follow existing snake_case convention for Rust files
- **Folder Organization:** Maintain existing modular structure by domain
- **Import/Export Patterns:** Follow existing Anchor module patterns and re-exports

---
