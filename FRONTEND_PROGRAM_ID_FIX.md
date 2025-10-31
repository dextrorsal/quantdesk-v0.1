# Frontend Program ID Fix

## Critical Issue
The frontend `ProgramContext.tsx` was using a **WRONG program ID**:
- ❌ Old: `GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a`
- ✅ Fixed: `C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw`

This caused the frontend to try to connect to the wrong program, so balance queries would fail silently.

## Fix Applied
✅ Updated `frontend/src/contexts/ProgramContext.tsx` program ID to match deployed program  
✅ Updated `frontend/src/services/unifiedBalanceService.ts` import (was using old anchor package)

## Next Steps
1. **Restart frontend dev server** to pick up the change
2. **Clear browser cache** if balance still doesn't show
3. **Check browser console** for any remaining errors

## Verification
The frontend IDL already has the correct program ID (`C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw`), so now:
- ✅ ProgramContext uses correct ID
- ✅ SmartContractService uses IDL address (correct)
- ✅ All balance queries will now hit the right program

