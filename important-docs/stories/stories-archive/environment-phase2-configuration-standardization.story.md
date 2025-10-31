# Phase 2: Environment Configuration Standardization

## User Story

As a **developer**,
I want **standardized environment variable names in code**,
so that **all services use consistent configuration patterns** while maintaining backward compatibility.

## Story Context

**Phase 1 Complete:** Critical security fixes implemented successfully
**Current State:** Backward compatibility working, tests passing
**Goal:** Standardize variable names in code while keeping backward compatibility

## CRITICAL CONSTRAINT - ENVIRONMENT FILE PROTECTION

**🚨 DEVELOPERS MUST NEVER TOUCH .env FILES**

> **📋 See also:** `docs/qa/CRITICAL-NOTICE-ENVIRONMENT-FILES.md` for complete developer guidance

### Environment File Structure (5 Directories):
Each directory has **3 specific environment files**:

```
root/
├── .env                    # Main env used in codebase (NEVER TOUCH)
├── .env.example           # Example with "enter api here" instructions (CAN MODIFY)
└── .env.backup            # Backup with migration instructions (CAN MODIFY)

frontend/
├── .env                   # Main env used in codebase (NEVER TOUCH)
├── .env.example          # Example with "enter api here" instructions (CAN MODIFY)
└── .env.backup           # Backup with migration instructions (CAN MODIFY)

backend/
├── .env                   # Main env used in codebase (NEVER TOUCH)
├── .env.example          # Example with "enter api here" instructions (CAN MODIFY)
└── .env.backup           # Backup with migration instructions (CAN MODIFY)

mikey-ai/
├── .env                   # Main env used in codebase (NEVER TOUCH)
├── .env.example          # Example with "enter api here" instructions (CAN MODIFY)
└── .env.backup           # Backup with migration instructions (CAN MODIFY)

data-ingestion/
├── .env                   # Main env used in codebase (NEVER TOUCH)
├── .env.example          # Example with "enter api here" instructions (CAN MODIFY)
└── .env.backup           # Backup with migration instructions (CAN MODIFY)
```

## Acceptance Criteria

### **Functional Requirements:**

1. **Standardize Variable Names in Code**
   - Use standardized names as primary variables
   - Maintain backward compatibility with old names
   - Update all service configurations consistently

2. **Implement Configuration Loading**
   - Create centralized configuration loading utilities
   - Support environment-specific configurations (dev, staging, prod)
   - Maintain backward compatibility during transition

3. **Update Documentation**
   - Update .env.example files with standardized variable names
   - Create migration instructions in .env.backup files
   - Document the migration strategy

### **Integration Requirements:**

4. **Maintain Backward Compatibility**
   - All existing deployments continue to work
   - No breaking changes to existing functionality
   - Smooth transition path for users

5. **Service Integration**
   - All services use consistent configuration patterns
   - Deployment scripts maintain current behavior
   - No regression in existing functionality

### **Quality Requirements:**

6. **Code Quality**
   - Follow existing patterns and standards
   - Comprehensive error handling
   - Clear documentation and comments

## Technical Implementation

### **Standardized Variable Names (Target)**

```typescript
// Target standardized configuration
const STANDARDIZED_CONFIG = {
  // Solana Configuration
  SOLANA_NETWORK: 'devnet',
  SOLANA_RPC_URL: 'https://api.devnet.solana.com',
  SOLANA_WS_URL: 'wss://api.devnet.solana.com',
  SOLANA_PRIVATE_KEY: 'base58_encoded_key',
  
  // Program Configuration
  QUANTDESK_PROGRAM_ID: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
  
  // Database Configuration
  SUPABASE_URL: 'https://project.supabase.co',
  SUPABASE_ANON_KEY: 'anon_key',
  
  // Oracle Configuration
  PYTH_NETWORK_URL: 'https://hermes.pyth.network/v2/updates/price/latest',
  PYTH_PRICE_FEED_SOL: 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG',
  PYTH_PRICE_FEED_BTC: 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J',
  PYTH_PRICE_FEED_ETH: 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB'
};
```

### **Implementation Pattern**

```typescript
// Phase 2: Standardized approach with backward compatibility
const getEnvironmentConfig = () => {
  return {
    // Use standardized names as primary, fallback to old names
    solanaNetwork: process.env.SOLANA_NETWORK || 'devnet',
    rpcUrl: process.env.SOLANA_RPC_URL || process.env.RPC_URL || 'https://api.devnet.solana.com',
    programId: process.env.QUANTDESK_PROGRAM_ID || process.env.PROGRAM_ID || 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
    privateKey: process.env.SOLANA_PRIVATE_KEY || process.env.SOLANA_WALLET_KEY,
    supabaseUrl: process.env.SUPABASE_URL,
    supabaseAnonKey: process.env.SUPABASE_ANON_KEY
  };
};
```

### **Migration Documentation Pattern**

```bash
# Example .env.backup migration instructions:
# ===========================================
# MIGRATION INSTRUCTIONS FOR PHASE 2
# ===========================================
# 
# The following variables have been standardized in code.
# You can optionally update your .env files to use the new names.
# 
# OLD NAME → NEW NAME (both work during transition)
# RPC_URL → SOLANA_RPC_URL
# PROGRAM_ID → QUANTDESK_PROGRAM_ID
# SOLANA_WALLET_KEY → SOLANA_PRIVATE_KEY
# 
# To migrate (OPTIONAL - both names work):
# 1. Add new variable names to .env
# 2. Test that everything works
# 3. Remove old variable names from .env
# 
# Example migration:
# OLD: RPC_URL=https://api.devnet.solana.com
# NEW: SOLANA_RPC_URL=https://api.devnet.solana.com
```

## Definition of Done

- [ ] All services use standardized variable names as primary
- [ ] Backward compatibility maintained for all old variable names
- [ ] .env.example files updated with standardized names
- [ ] .env.backup files contain migration instructions
- [ ] All existing functionality continues to work
- [ ] Code follows existing patterns and standards
- [ ] Documentation updated with migration strategy

## Risk and Compatibility Check

**Minimal Risk Assessment:**
- **Primary Risk:** Breaking existing deployments during transition
- **Mitigation:** Maintain full backward compatibility
- **Rollback:** Revert to old variable names if needed

**Compatibility Verification:**
- [ ] No breaking changes to existing APIs
- [ ] All existing deployments continue to work
- [ ] New deployments can use standardized names
- [ ] Migration path is clear and documented

---

## Phase 2 Success Criteria

✅ **Phase 2 Complete When:**
1. All services use standardized variable names as primary
2. Backward compatibility maintained for all old names
3. Migration documentation created in .env.backup files
4. .env.example files show standardized names
5. All existing functionality continues to work
6. Ready for Phase 3 (Migration Documentation)

**Next Phase:** Phase 3 - Migration Documentation and User Guidance
