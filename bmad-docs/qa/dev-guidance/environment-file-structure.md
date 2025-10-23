# Environment File Structure - Developer Guidance

## 🚨 CRITICAL CONSTRAINT - ENVIRONMENT FILE PROTECTION

**DEVELOPERS MUST NEVER TOUCH .env FILES**

> **📋 This is the main developer guidance document. See also:** `docs/qa/CRITICAL-NOTICE-ENVIRONMENT-FILES.md` for quick reference

## 📋 QUICK REFERENCE - What Devs CAN and CANNOT Do

### ✅ **DEVELOPERS CAN:**
- **Modify .env.backup files** - Make corrections and add migration instructions
- **Modify .env.example files** - Show users what variables they need
- **Modify code** - Handle environment variables with backward compatibility
- **Add validation** - Environment validation in code only

### ❌ **DEVELOPERS CANNOT:**
- **Modify .env files** - Main environment files are OFF LIMITS
- **Delete variables** - From .env files
- **Change values** - In .env files
- **Create new .env files** - Without user permission

### 🔄 **WORKFLOW:**
1. **Dev modifies** .env.backup and .env.example files
2. **User reviews** the changes in backup/example files
3. **User manually** updates the actual .env files
4. **Dev tests** with the updated configuration

## Environment File Structure (5 Directories)

Each directory has **3 specific environment files**:

```
root/
├── .env                    # Main env used in codebase
├── .env.example           # Example with "enter api here" instructions
└── .env.backup            # Backup of main env file

frontend/
├── .env                   # Main env used in codebase
├── .env.example          # Example with "enter api here" instructions
└── .env.backup           # Backup of main env file

backend/
├── .env                   # Main env used in codebase
├── .env.example          # Example with "enter api here" instructions
└── .env.backup           # Backup of main env file

mikey-ai/
├── .env                   # Main env used in codebase
├── .env.example          # Example with "enter api here" instructions
└── .env.backup           # Backup of main env file

data-ingestion/
├── .env                   # Main env used in codebase
├── .env.example          # Example with "enter api here" instructions
└── .env.backup           # Backup of main env file
```

## File Types Explained

### `.env` (Main Environment File)
- **Purpose**: Contains actual environment variables used by the codebase
- **Content**: Real API keys, database URLs, private keys, etc.
- **Access**: NEVER modify this file directly
- **Security**: Contains sensitive information - must be protected

### `.env.example` (Example Template)
- **Purpose**: Template showing required environment variables
- **Content**: Placeholder values like "enter_api_key_here" or instructions
- **Access**: Can be modified to show new required variables
- **Security**: No sensitive information - safe to commit to git

### `.env.backup` (Backup File)
- **Purpose**: Backup of the main .env file with migration instructions
- **Content**: Commands and instructions for environment changes
- **Access**: Can be created/modified to provide migration guidance
- **Security**: May contain sensitive information - handle carefully

## Allowed Actions (CODE ONLY)

### ✅ What Developers CAN Do:
- **Modify code** to use standardized environment variable names
- **Add environment validation** in code (not .env files)
- **Modify .env.backup files** with corrections and migration instructions
- **Modify .env.example files** to show users required variables
- **Update documentation** with migration instructions
- **Add backward compatibility** in code for old variable names
- **Implement secure loading** of environment variables in code

### ❌ What Developers CANNOT Do:
- **Modify existing .env files** directly (main environment files)
- **Delete environment variables** from .env files
- **Change environment variable values** in .env files
- **Create new .env files** without user permission
- **Rename environment variables** in .env files
- **Add new environment variables** to .env files

## Required Process for Environment Changes

### When Environment Variables Need Changes:

1. **Document the Change** in code comments
   ```typescript
   // ENVIRONMENT CHANGE REQUIRED:
   // Add SOLANA_PRIVATE_KEY to .env file
   // Remove ANCHOR_WALLET from .env file
   ```

2. **Make Corrections in .env.backup files** with migration instructions
   ```bash
   # Migration commands for .env.backup
   # Add this line to .env:
   # SOLANA_PRIVATE_KEY=your_base58_private_key_here
   
   # Remove this line from .env:
   # ANCHOR_WALLET=~/.config/solana/keys/id.json
   ```

3. **Update .env.example files** to show users what variables they need
   ```bash
   # Example for .env.example
   # Required Solana Configuration:
   SOLANA_PRIVATE_KEY=your_base58_private_key_here
   SOLANA_RPC_URL=https://api.devnet.solana.com
   QUANTDESK_PROGRAM_ID=C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw
   ```

4. **Notify the User** for manual .env file updates
   ```typescript
   // TODO: User must manually update .env file
   // See .env.backup for migration instructions
   // See .env.example for required variables
   ```

5. **User Handles** all .env file modifications manually

## Implementation Examples

### ✅ Correct Approach - Code Changes Only:

```typescript
// ✅ CORRECT: Handle both old and new variable names in code
const getEnvironmentConfig = () => {
  return {
    // Support both old and new names for backward compatibility
    solanaNetwork: process.env.SOLANA_NETWORK || 'devnet',
    rpcUrl: process.env.SOLANA_RPC_URL || process.env.RPC_URL || 'https://api.devnet.solana.com',
    programId: process.env.QUANTDESK_PROGRAM_ID || process.env.PROGRAM_ID || 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
    privateKey: process.env.SOLANA_PRIVATE_KEY || process.env.SOLANA_WALLET_KEY,
  };
};
```

### ❌ Incorrect Approach - Modifying .env Files:

```bash
# ❌ WRONG: Never modify .env files directly
# Don't do this:
echo "SOLANA_PRIVATE_KEY=new_key" >> .env
sed -i 's/ANCHOR_WALLET/SOLANA_WALLET/g' .env
```

## Security Considerations

### Private Key Handling:
- **Never log** private keys or sensitive environment variables
- **Validate format** of private keys in code
- **Use secure loading** mechanisms (not hardcoded values)
- **Support both** old and new variable names for compatibility

### Environment Validation:
- **Check required variables** at service startup
- **Validate format** of critical variables (URLs, keys, IDs)
- **Provide clear error messages** for missing/invalid variables
- **Fail securely** if critical variables are missing

## Migration Strategy

### Phase 1: Code Changes (Current)
- Implement backward compatibility in code
- Add environment validation
- Support both old and new variable names
- Document required .env changes

### Phase 2: User Migration (Future)
- User manually updates .env files
- User removes old variable names
- User adds new standardized variables
- User tests configuration

### Phase 3: Cleanup (Future)
- Remove backward compatibility code
- Standardize on new variable names only
- Update documentation

## Common Patterns

### Environment Variable Loading:
```typescript
// ✅ Standardized loading with backward compatibility
const config = {
  solanaRpcUrl: process.env.SOLANA_RPC_URL || process.env.RPC_URL,
  programId: process.env.QUANTDESK_PROGRAM_ID || process.env.PROGRAM_ID,
  privateKey: process.env.SOLANA_PRIVATE_KEY || process.env.SOLANA_WALLET_KEY,
};
```

### Environment Validation:
```typescript
// ✅ Comprehensive validation
const validateEnvironment = () => {
  const required = [
    'SOLANA_PRIVATE_KEY', 'SOLANA_WALLET_KEY', // Support both
    'QUANTDESK_PROGRAM_ID', 'PROGRAM_ID', // Support both
    'SOLANA_RPC_URL', 'RPC_URL', // Support both
  ];
  
  const missing = required.filter(key => !process.env[key]);
  if (missing.length > 0) {
    throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
  }
};
```

## Summary

- **5 directories** with environment files: root, frontend, backend, mikey-ai, data-ingestion
- **3 files per directory**: .env (main), .env.example (template), .env.backup (instructions)
- **CODE ONLY approach** - never modify .env files directly
- **User handles** all .env file modifications manually
- **Backward compatibility** implemented in code for existing deployments
- **Security first** - validate and secure all environment variable handling

**Remember: When in doubt, modify code, not .env files!**
