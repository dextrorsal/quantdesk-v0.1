# Phase 3: Environment Migration Documentation

## User Story

As a **system administrator**,
I want **clear migration documentation and instructions**,
so that **I can safely update my .env files to use standardized variable names**.

## Story Context

**Phase 2 Complete:** Configuration standardization implemented in code
**Current State:** Both old and new variable names work in code
**Goal:** Create comprehensive migration documentation for users

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

1. **Create Migration Instructions**
   - Detailed step-by-step migration guides in .env.backup files
   - Clear mapping of old → new variable names
   - Safety checks and validation steps

2. **Update Example Files**
   - .env.example files show standardized variable names
   - Clear instructions for new deployments
   - Examples for all required variables

3. **Create Migration Tools**
   - Validation scripts to check migration readiness
   - Backup and restore procedures
   - Rollback instructions if needed

### **Integration Requirements:**

4. **Maintain Backward Compatibility**
   - All existing deployments continue to work
   - Migration is optional and safe
   - No breaking changes during transition

5. **Service Integration**
   - All services documented for migration
   - Deployment scripts updated with new patterns
   - Clear migration timeline

### **Quality Requirements:**

6. **Documentation Quality**
   - Clear, step-by-step instructions
   - Examples for all scenarios
   - Troubleshooting guides

## Technical Implementation

### **Migration Instructions Template**

```bash
# ===========================================
# ENVIRONMENT VARIABLE MIGRATION GUIDE
# ===========================================
# 
# This guide helps you migrate from old variable names to standardized names.
# Both old and new names work during the transition period.
# 
# MIGRATION IS OPTIONAL - Your current setup will continue to work!
# 
# ===========================================
# VARIABLE NAME MAPPINGS
# ===========================================
# 
# OLD NAME → NEW NAME (both work during transition)
# RPC_URL → SOLANA_RPC_URL
# PROGRAM_ID → QUANTDESK_PROGRAM_ID
# SOLANA_WALLET_KEY → SOLANA_PRIVATE_KEY
# ANCHOR_WALLET → SOLANA_WALLET (if used)
# 
# ===========================================
# MIGRATION STEPS
# ===========================================
# 
# 1. BACKUP YOUR CURRENT .env FILE
#    cp .env .env.backup.$(date +%Y%m%d)
# 
# 2. ADD NEW VARIABLE NAMES (keep old ones for now)
#    # Add these lines to your .env file:
#    SOLANA_RPC_URL=https://api.devnet.solana.com
#    QUANTDESK_PROGRAM_ID=C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw
#    SOLANA_PRIVATE_KEY=your_base58_private_key
# 
# 3. TEST YOUR APPLICATION
#    # Start your services and verify everything works
#    npm run dev
# 
# 4. REMOVE OLD VARIABLE NAMES (optional)
#    # Once you're confident everything works, you can remove:
#    # RPC_URL=old_value
#    # PROGRAM_ID=old_value
#    # SOLANA_WALLET_KEY=old_value
# 
# 5. VERIFY MIGRATION
#    # Check that your application still works with only new names
# 
# ===========================================
# ROLLBACK INSTRUCTIONS
# ===========================================
# 
# If you need to rollback:
# 1. Restore your backup: cp .env.backup.$(date +%Y%m%d) .env
# 2. Restart your services
# 
# ===========================================
# SUPPORT
# ===========================================
# 
# If you encounter issues:
# 1. Check the troubleshooting guide below
# 2. Verify all required variables are present
# 3. Check service logs for specific error messages
```

### **Updated .env.example Template**

```bash
# ===========================================
# QUANTDESK ENVIRONMENT CONFIGURATION
# ===========================================
# 
# Copy this file to .env and fill in your actual values
# 
# ===========================================
# SOLANA CONFIGURATION
# ===========================================
SOLANA_NETWORK=devnet
SOLANA_RPC_URL=https://api.devnet.solana.com
SOLANA_WS_URL=wss://api.devnet.solana.com
SOLANA_PRIVATE_KEY=your_base58_private_key_here

# ===========================================
# PROGRAM CONFIGURATION
# ===========================================
QUANTDESK_PROGRAM_ID=C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw

# ===========================================
# DATABASE CONFIGURATION
# ===========================================
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key

# ===========================================
# ORACLE CONFIGURATION
# ===========================================
PYTH_NETWORK_URL=https://hermes.pyth.network/v2/updates/price/latest
PYTH_PRICE_FEED_SOL=H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG
PYTH_PRICE_FEED_BTC=HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J
PYTH_PRICE_FEED_ETH=JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB

# ===========================================
# SECURITY CONFIGURATION
# ===========================================
JWT_SECRET=your_jwt_secret_here
CORS_ORIGIN=http://localhost:3000

# ===========================================
# OPTIONAL CONFIGURATION
# ===========================================
# LOG_LEVEL=info
# ENABLE_METRICS=true
# REDIS_URL=redis://localhost:6379
```

### **Migration Validation Script**

```typescript
// Example validation script for migration
const validateMigration = () => {
  const requiredVars = [
    'SOLANA_RPC_URL',
    'QUANTDESK_PROGRAM_ID', 
    'SOLANA_PRIVATE_KEY',
    'SUPABASE_URL',
    'SUPABASE_ANON_KEY'
  ];
  
  const missing = requiredVars.filter(key => !process.env[key]);
  
  if (missing.length > 0) {
    console.error('❌ Missing required variables:', missing.join(', '));
    console.log('📋 See .env.backup for migration instructions');
    return false;
  }
  
  console.log('✅ All required variables present');
  return true;
};
```

## Definition of Done

- [ ] Migration instructions created in all .env.backup files
- [ ] .env.example files updated with standardized names
- [ ] Migration validation scripts created
- [ ] Rollback procedures documented
- [ ] Troubleshooting guides created
- [ ] All existing functionality continues to work
- [ ] Migration is optional and safe

## Risk and Compatibility Check

**Minimal Risk Assessment:**
- **Primary Risk:** Users making mistakes during migration
- **Mitigation:** Clear instructions, backup procedures, rollback options
- **Rollback:** Restore backup files if needed

**Compatibility Verification:**
- [ ] Migration is completely optional
- [ ] All existing deployments continue to work
- [ ] Clear rollback procedures available
- [ ] Comprehensive troubleshooting guides

---

## Phase 3 Success Criteria

✅ **Phase 3 Complete When:**
1. All .env.backup files contain detailed migration instructions
2. All .env.example files show standardized variable names
3. Migration validation scripts created
4. Rollback procedures documented
5. Troubleshooting guides created
6. Migration is optional and safe
7. Ready for Phase 4 (Cleanup)

**Next Phase:** Phase 4 - Remove Backward Compatibility (Future)
