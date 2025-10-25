# AI Development Assistant Guide

## Overview

This guide helps AI development assistants (Cursor AI, Claude, Grok, etc.) understand and work effectively with the QuantDesk codebase.

## Quick Start for AI Assistants

### 1. Understand the System
```bash
# Get live system architecture
curl http://localhost:3002/api/dev/codebase-structure

# Get market data structure
curl http://localhost:3002/api/dev/market-summary

# Get API documentation
curl http://localhost:3002/api/docs/swagger
```

### 2. Key Patterns to Follow

#### Database Access
```typescript
// ✅ CORRECT - Use databaseService
import { databaseService } from '../services/supabaseDatabase';
const users = await databaseService.select('users', '*', { is_active: true });

// ❌ WRONG - Direct Supabase calls
import { supabase } from '../services/supabaseService';
const users = await supabase.from('users').select('*');
```

#### Oracle Price Access
```typescript
// ✅ CORRECT - Use getAllPrices()
import { pythOracleService } from '../services/pythOracleService';
const prices = await pythOracleService.getAllPrices();
console.log(prices.BTC); // 0.0012116391507573001

// ❌ WRONG - Direct oracle calls
const response = await fetch('https://api.pyth.network/...');
```

#### Error Handling
```typescript
// ✅ CORRECT - Use custom error classes
import { QuantDeskError, ValidationError } from '../middleware/errorHandling';
throw new ValidationError('Invalid wallet address');

// ❌ WRONG - Generic errors
throw new Error('Invalid wallet address');
```

### 3. Common Development Tasks

#### Adding a New API Route
```typescript
// 1. Create route file: backend/src/routes/newFeature.ts
import express, { Request, Response } from 'express';
import { asyncHandler } from '../middleware/errorHandling';
import { databaseService } from '../services/supabaseDatabase';

const router = express.Router();

router.get('/new-endpoint', asyncHandler(async (req: Request, res: Response) => {
  const data = await databaseService.select('table_name', '*', {});
  res.json({ success: true, data });
}));

export default router;

// 2. Register in server.ts
import newFeatureRoutes from './routes/newFeature';
app.use('/api/new-feature', newFeatureRoutes);
```

#### Adding a New Database Service Method
```typescript
// In backend/src/services/supabaseDatabase.ts
export class SupabaseDatabaseService {
  /**
   * Get users by status
   * @param status - User status filter
   * @returns Promise<User[]>
   */
  async getUsersByStatus(status: string): Promise<User[]> {
    const { data, error } = await this.client
      .from('users')
      .select('*')
      .eq('status', status);
    
    if (error) throw new QuantDeskError(`Database error: ${error.message}`);
    return data || [];
  }
}
```

#### Adding a New Smart Contract Instruction
```rust
// In contracts/smart-contracts/programs/quantdesk-perp-dex/src/lib.rs
#[program]
pub mod quantdesk_perp_dex {
    use super::*;

    pub fn new_instruction(
        ctx: Context<NewInstructionContext>,
        param1: u64,
        param2: String,
    ) -> Result<()> {
        // Implementation
        Ok(())
    }
}

#[derive(Accounts)]
pub struct NewInstructionContext<'info> {
    // Account definitions
}
```

### 4. Testing Changes

#### Backend Testing
```bash
# Compile TypeScript
cd backend && pnpm run build

# Start development server
cd backend && pnpm run start:dev

# Test endpoints
curl http://localhost:3002/api/dev/codebase-structure
```

#### Smart Contract Testing
```bash
# Build contracts
cd contracts/smart-contracts && anchor build

# Run tests
cd contracts/smart-contracts && anchor test

# Deploy to local devnet
cd contracts/smart-contracts && anchor deploy
```

#### Frontend Testing
```bash
# Start frontend
cd frontend && pnpm run dev

# Build for production
cd frontend && pnpm run build
```

### 5. Troubleshooting Common Issues

#### Supabase Function Not Found
```
Error: Could not find the function public.execute_sql(params, sql)
```
**Solution**: This is EXPECTED for devnet. The `execute_sql` function needs to be deployed to Supabase separately.

#### Oracle Price Format
```typescript
// Oracle prices are in scientific notation - this is correct
const prices = await pythOracleService.getAllPrices();
console.log(prices.BTC); // 0.0012116391507573001 (normalized)
```

#### Package Manager Issues
```bash
# ✅ CORRECT - Always use pnpm
pnpm install
pnpm run build

# ❌ WRONG - Don't use npm
npm install
npm run build
```

#### Port Conflicts
```bash
# Check what's using port 3002
lsof -ti:3002

# Kill process if needed
lsof -ti:3002 | xargs kill -9
```

### 6. Where to Find Examples

#### Working Backend Routes
- `backend/src/routes/markets.ts` - Market data endpoints
- `backend/src/routes/oracle.ts` - Oracle price endpoints
- `backend/src/routes/users.ts` - User management

#### Working Database Queries
- `backend/src/services/supabaseDatabase.ts` - All database methods
- `backend/src/services/devnetService.ts` - Devnet-specific queries

#### Working Smart Contract Patterns
- `contracts/smart-contracts/programs/quantdesk-perp-dex/src/lib.rs` - Main program
- `contracts/smart-contracts/programs/quantdesk-perp-dex/src/user_accounts.rs` - User logic

#### Working Frontend Components
- `frontend/src/components/` - React components
- `frontend/src/pages/` - Page components

### 7. AI Assistant Best Practices

1. **Always check existing patterns** before implementing new features
2. **Use the `/api/dev/*` endpoints** to understand system state
3. **Follow the established error handling patterns**
4. **Test changes incrementally** - compile, test endpoints, then integrate
5. **Reference the architecture documentation** for system understanding
6. **Use pnpm consistently** throughout the codebase

### 8. Useful Commands for AI Assistants

```bash
# Get system status
curl http://localhost:3002/api/dev/codebase-structure | jq

# Check backend compilation
cd backend && pnpm run build

# Check smart contract compilation
cd contracts/smart-contracts && anchor build

# Get live market data
curl http://localhost:3002/api/dev/market-summary | jq

# Get API documentation
curl http://localhost:3002/api/docs/swagger | jq
```

This guide should help AI assistants work more effectively with the QuantDesk codebase and provide better suggestions for development tasks.
