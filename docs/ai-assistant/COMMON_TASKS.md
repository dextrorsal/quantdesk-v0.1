# Common Development Tasks

This guide provides step-by-step instructions for common development tasks in the QuantDesk codebase.

## Adding a New API Route

### 1. Create Route File
```typescript
// backend/src/routes/newFeature.ts
import express, { Request, Response } from 'express';
import { asyncHandler } from '../middleware/errorHandling';
import { databaseService } from '../services/supabaseDatabase';

const router = express.Router();

/**
 * GET /api/new-feature/endpoint
 * Example endpoint for new feature
 */
router.get('/endpoint', asyncHandler(async (req: Request, res: Response) => {
  try {
    // Use databaseService for database operations
    const data = await databaseService.select('table_name', '*', {});
    
    res.json({
      success: true,
      data: data,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error in new endpoint:', error);
    res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
}));

export default router;
```

### 2. Register Route in Server
```typescript
// backend/src/server.ts
import newFeatureRoutes from './routes/newFeature';

// Register the route
app.use('/api/new-feature', newFeatureRoutes);
```

### 3. Test the Route
```bash
# Start backend
cd backend && pnpm run start:dev

# Test endpoint
curl http://localhost:3002/api/new-feature/endpoint
```

## Adding a New Database Service Method

### 1. Add Method to SupabaseDatabaseService
```typescript
// backend/src/services/supabaseDatabase.ts
export class SupabaseDatabaseService {
  /**
   * Get users by specific criteria
   * @param criteria - Filter criteria
   * @returns Promise<User[]> - Array of matching users
   */
  async getUsersByCriteria(criteria: any): Promise<User[]> {
    try {
      const { data, error } = await this.getClient()
        .from('users')
        .select('*')
        .match(criteria);
      
      if (error) {
        logger.error('Database error:', error);
        throw new Error(`Database error: ${error.message}`);
      }
      
      return data || [];
    } catch (error) {
      logger.error('Error getting users by criteria:', error);
      throw error;
    }
  }
}
```

### 2. Use the Method in Routes
```typescript
// In your route file
import { databaseService } from '../services/supabaseDatabase';

router.get('/users', asyncHandler(async (req: Request, res: Response) => {
  const users = await databaseService.getUsersByCriteria({ is_active: true });
  res.json({ success: true, data: users });
}));
```

## Adding a New Smart Contract Instruction

### 1. Define the Instruction
```rust
// contracts/programs/src/lib.rs
#[program]
pub mod quantdesk_perp_dex {
    use super::*;

    pub fn new_instruction(
        ctx: Context<NewInstructionContext>,
        param1: u64,
        param2: String,
    ) -> Result<()> {
        // Get accounts
        let user_account = &mut ctx.accounts.user_account;
        let market = &ctx.accounts.market;
        
        // Validate parameters
        require!(param1 > 0, ErrorCode::InvalidParameter);
        
        // Update state
        user_account.some_field = param1;
        
        // Emit event
        emit!(NewInstructionEvent {
            user: user_account.key(),
            param1,
            param2,
        });
        
        Ok(())
    }
}
```

### 2. Define Account Context
```rust
#[derive(Accounts)]
pub struct NewInstructionContext<'info> {
    #[account(mut)]
    pub user_account: Account<'info, UserAccount>,
    
    pub market: Account<'info, Market>,
    
    pub system_program: Program<'info, System>,
}
```

### 3. Define Event Structure
```rust
#[event]
pub struct NewInstructionEvent {
    pub user: Pubkey,
    pub param1: u64,
    pub param2: String,
}
```

### 4. Test the Instruction
```bash
# Build contracts
cd contracts && anchor build

# Run tests
cd contracts && anchor test
```

## Adding a New Frontend Component

### 1. Create Component File
```typescript
// frontend/src/components/NewComponent.tsx
import React, { useState, useEffect } from 'react';

interface NewComponentProps {
  title: string;
  data?: any[];
}

export const NewComponent: React.FC<NewComponentProps> = ({ title, data = [] }) => {
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // Component initialization
  }, []);

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">{title}</h2>
      {loading ? (
        <div>Loading...</div>
      ) : (
        <div>
          {data.map((item, index) => (
            <div key={index} className="mb-2">
              {item.name}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
```

### 2. Use Component in Page
```typescript
// frontend/src/pages/NewPage.tsx
import React from 'react';
import { NewComponent } from '../components/NewComponent';

export const NewPage: React.FC = () => {
  const data = [
    { name: 'Item 1' },
    { name: 'Item 2' },
  ];

  return (
    <div className="container mx-auto p-4">
      <NewComponent title="New Feature" data={data} />
    </div>
  );
};
```

## Adding a New Database Table

### 1. Create Migration File
```sql
-- database/migrations/add_new_table.sql
CREATE TABLE IF NOT EXISTS new_table (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Add indexes
CREATE INDEX idx_new_table_name ON new_table(name);
CREATE INDEX idx_new_table_active ON new_table(is_active);

-- Add RLS policies
ALTER TABLE new_table ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view active records" ON new_table
    FOR SELECT USING (is_active = true);
```

### 2. Update TypeScript Interfaces
```typescript
// backend/src/services/supabaseDatabase.ts
export interface NewTable {
  id: string;
  name: string;
  description?: string;
  is_active: boolean;
  created_at: Date;
  updated_at: Date;
}
```

### 3. Add Service Methods
```typescript
// backend/src/services/supabaseDatabase.ts
export class SupabaseDatabaseService {
  /**
   * Get all records from new_table
   */
  async getNewTableRecords(): Promise<NewTable[]> {
    return await this.select('new_table', '*', { is_active: true });
  }

  /**
   * Create new record in new_table
   */
  async createNewTableRecord(data: Partial<NewTable>): Promise<NewTable> {
    return await this.insert('new_table', data);
  }
}
```

## Testing Changes

### Backend Testing
```bash
# Compile TypeScript
cd backend && pnpm run build

# Start development server
cd backend && pnpm run start:dev

# Test endpoints
curl http://localhost:3002/api/dev/codebase-structure
```

### Smart Contract Testing
```bash
# Build contracts
cd contracts && anchor build

# Run tests
cd contracts && anchor test

# Deploy to local devnet
cd contracts && anchor deploy
```

### Frontend Testing
```bash
# Start frontend
cd frontend && pnpm run dev

# Build for production
cd frontend && pnpm run build
```

## Common Patterns

### Error Handling
```typescript
try {
  const result = await someOperation();
  return result;
} catch (error) {
  logger.error('Operation failed:', error);
  throw new QuantDeskError(`Operation failed: ${error.message}`);
}
```

### Database Queries
```typescript
// Always use databaseService
const data = await databaseService.select('table', '*', { filter: value });

// Never use direct Supabase calls
// const data = await supabase.from('table').select('*');
```

### API Responses
```typescript
// Success response
res.json({
  success: true,
  data: result,
  timestamp: new Date().toISOString()
});

// Error response
res.status(500).json({
  success: false,
  error: 'Error message',
  message: 'Detailed error description'
});
```

This guide should help you implement common development tasks in the QuantDesk codebase efficiently and consistently.
