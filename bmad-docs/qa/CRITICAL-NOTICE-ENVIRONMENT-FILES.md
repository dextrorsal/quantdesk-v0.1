# 🚨 CRITICAL DEVELOPER NOTICE - ENVIRONMENT FILES

## ⚠️ **DEVELOPERS MUST NEVER TOUCH .env FILES**

### ✅ **WHAT DEVELOPERS CAN DO:**
- **Modify .env.backup files** - Make corrections and add migration instructions
- **Modify .env.example files** - Show users what variables they need  
- **Modify code** - Handle environment variables with backward compatibility
- **Add validation** - Environment validation in code only

### ❌ **WHAT DEVELOPERS CANNOT DO:**
- **Modify .env files** - Main environment files are OFF LIMITS
- **Delete variables** - From .env files
- **Change values** - In .env files
- **Create new .env files** - Without user permission

## 🔄 **CORRECT WORKFLOW:**

1. **Dev modifies** `.env.backup` and `.env.example` files
2. **User reviews** the changes in backup/example files  
3. **User manually** updates the actual `.env` files
4. **Dev tests** with the updated configuration

## 📁 **ENVIRONMENT FILE STRUCTURE:**

Each directory has **3 specific files**:
- `.env` - Main env used in codebase (**NEVER TOUCH**)
- `.env.example` - Example with "enter api here" instructions (**CAN MODIFY**)
- `.env.backup` - Backup with migration instructions (**CAN MODIFY**)

**5 Directories:** root, frontend, backend, mikey-ai, data-ingestion

## 🎯 **KEY POINTS:**

- **Never modify** `.env` files directly
- **Always use** `.env.backup` for corrections
- **Always update** `.env.example` for user guidance
- **Code handles** backward compatibility
- **User handles** actual .env file updates

---

**This notice appears in:**
- Story documentation
- QA gates and assessments  
- Test design documents
- Developer guidance files
- Risk assessments

**Remember: When in doubt, modify .env.backup and .env.example, NOT .env files!**
