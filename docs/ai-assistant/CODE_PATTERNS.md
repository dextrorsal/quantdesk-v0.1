# Code Patterns and Examples

This document provides common code patterns and examples used throughout the QuantDesk codebase.

## Database Access Patterns

### ✅ Correct Database Access
```typescript
import { databaseService } from '../services/supabaseDatabase';

// Select with filters
const users = await databaseService.select('users', '*', { is_active: true });

// Insert new record
const newUser = await databaseService.insert('users', {
  wallet_address: '0x123...',
  username: 'trader1',
  is_active: true
});

// Update record
const updatedUser = await databaseService.update('users', { id: 'user-id' }, {
  last_login: new Date()
});

// Delete record
await databaseService.delete('users', { id: 'user-id' });
```

### ❌ Incorrect Database Access
```typescript
import { supabase } from '../services/supabaseService';

// Don't use direct Supabase calls
const users = await supabase.from('users').select('*');
const newUser = await supabase.from('users').insert({...});
```

## Error Handling Patterns

### ✅ Correct Error Handling
```typescript
import { QuantDeskError, ValidationError } from '../middleware/errorHandling';
import { asyncHandler } from '../middleware/errorHandling';

// In route handlers
router.get('/endpoint', asyncHandler(async (req: Request, res: Response) => {
  try {
    // Validate input
    if (!req.params.id) {
      throw new ValidationError('ID parameter is required');
    }
    
    // Business logic
    const result = await someOperation(req.params.id);
    
    res.json({
      success: true,
      data: result,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Endpoint error:', error);
    throw error; // Let asyncHandler handle the response
  }
}));

// In service methods
async someServiceMethod(id: string): Promise<any> {
  try {
    const data = await databaseService.select('table', '*', { id });
    
    if (!data.length) {
      throw new QuantDeskError(`Record with id ${id} not found`);
    }
    
    return data[0];
  } catch (error) {
    logger.error('Service method error:', error);
    throw error;
  }
}
```

### ❌ Incorrect Error Handling
```typescript
// Don't use generic errors
throw new Error('Something went wrong');

// Don't forget to handle errors
const result = await someOperation(); // Missing try-catch

// Don't return error responses directly
res.status(500).json({ error: 'Error' }); // Use asyncHandler instead
```

## API Response Patterns

### ✅ Correct API Responses
```typescript
// Success response
res.json({
  success: true,
  data: result,
  timestamp: new Date().toISOString()
});

// Error response (handled by asyncHandler)
throw new ValidationError('Invalid input');
throw new QuantDeskError('Operation failed');
```

### ❌ Incorrect API Responses
```typescript
// Don't return inconsistent response format
res.json(result); // Missing success flag
res.json({ data: result }); // Missing success flag and timestamp
```

## Oracle Price Access Patterns

### ✅ Correct Oracle Access
```typescript
import { pythOracleService } from '../services/pythOracleService';

// Get all prices
const prices = await pythOracleService.getAllPrices();
console.log(prices.BTC); // 0.0012116391507573001
console.log(prices.ETH); // 0.000043722750125000004

// Get specific price
const btcPrice = await pythOracleService.getPrice('BTC');
console.log(btcPrice); // 0.0012116391507573001
```

### ❌ Incorrect Oracle Access
```typescript
// Don't make direct API calls to Pyth
const response = await fetch('https://api.pyth.network/...');

// Don't assume price format
const price = prices.BTC * 1000; // Prices are already normalized
```

## Smart Contract Patterns

### ✅ Correct Smart Contract Structure
```rust
#[program]
pub mod quantdesk_perp_dex {
    use super::*;

    pub fn instruction_name(
        ctx: Context<InstructionContext>,
        param1: u64,
        param2: String,
    ) -> Result<()> {
        // Get accounts
        let user_account = &mut ctx.accounts.user_account;
        let market = &ctx.accounts.market;
        
        // Validate parameters
        require!(param1 > 0, ErrorCode::InvalidParameter);
        require!(!param2.is_empty(), ErrorCode::InvalidParameter);
        
        // Update state
        user_account.some_field = param1;
        
        // Emit event
        emit!(InstructionEvent {
            user: user_account.key(),
            param1,
            param2,
        });
        
        Ok(())
    }
}

#[derive(Accounts)]
pub struct InstructionContext<'info> {
    #[account(mut)]
    pub user_account: Account<'info, UserAccount>,
    
    pub market: Account<'info, Market>,
    
    pub system_program: Program<'info, System>,
}

#[event]
pub struct InstructionEvent {
    pub user: Pubkey,
    pub param1: u64,
    pub param2: String,
}
```

### ❌ Incorrect Smart Contract Patterns
```rust
// Don't forget parameter validation
pub fn instruction_name(ctx: Context<InstructionContext>, param1: u64) -> Result<()> {
    let user_account = &mut ctx.accounts.user_account;
    user_account.some_field = param1; // No validation
    Ok(())
}

// Don't forget to emit events
pub fn instruction_name(ctx: Context<InstructionContext>, param1: u64) -> Result<()> {
    let user_account = &mut ctx.accounts.user_account;
    user_account.some_field = param1;
    // Missing emit!(InstructionEvent { ... });
    Ok(())
}
```

## Frontend Component Patterns

### ✅ Correct React Component Structure
```typescript
import React, { useState, useEffect } from 'react';

interface ComponentProps {
  title: string;
  data?: any[];
  onAction?: (item: any) => void;
}

export const Component: React.FC<ComponentProps> = ({ 
  title, 
  data = [], 
  onAction 
}) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        // Fetch data
      } catch (err) {
        setError('Failed to fetch data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div className="p-4 bg-white rounded-lg shadow">
      <h2 className="text-xl font-bold mb-4">{title}</h2>
      <div className="space-y-2">
        {data.map((item, index) => (
          <div key={index} className="p-2 border rounded">
            {item.name}
            {onAction && (
              <button 
                onClick={() => onAction(item)}
                className="ml-2 px-2 py-1 bg-blue-500 text-white rounded"
              >
                Action
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
```

### ❌ Incorrect React Patterns
```typescript
// Don't forget error handling
export const Component: React.FC<ComponentProps> = ({ title, data = [] }) => {
  const [loading, setLoading] = useState(false);
  // Missing error state

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      // Missing try-catch
      // Fetch data
      setLoading(false);
    };

    fetchData();
  }, []);

  // Missing loading and error states
  return (
    <div>
      <h2>{title}</h2>
      {data.map((item, index) => (
        <div key={index}>{item.name}</div>
      ))}
    </div>
  );
};
```

## Service Method Patterns

### ✅ Correct Service Method Structure
```typescript
export class ExampleService {
  private static instance: ExampleService;
  private logger = new Logger();

  private constructor() {}

  public static getInstance(): ExampleService {
    if (!ExampleService.instance) {
      ExampleService.instance = new ExampleService();
    }
    return ExampleService.instance;
  }

  /**
   * Perform some operation
   * @param param1 - First parameter
   * @param param2 - Second parameter
   * @returns Promise<ResultType>
   */
  async performOperation(param1: string, param2: number): Promise<ResultType> {
    try {
      this.logger.info(`Performing operation with ${param1} and ${param2}`);
      
      // Validate parameters
      if (!param1 || param2 <= 0) {
        throw new ValidationError('Invalid parameters');
      }
      
      // Perform operation
      const result = await this.doSomething(param1, param2);
      
      this.logger.info('Operation completed successfully');
      return result;
    } catch (error) {
      this.logger.error('Operation failed:', error);
      throw error;
    }
  }

  private async doSomething(param1: string, param2: number): Promise<ResultType> {
    // Private implementation
    return {} as ResultType;
  }
}
```

### ❌ Incorrect Service Patterns
```typescript
// Don't forget singleton pattern
export class ExampleService {
  // Missing singleton implementation
  
  // Don't forget error handling
  async performOperation(param1: string, param2: number): Promise<ResultType> {
    // Missing try-catch
    const result = await this.doSomething(param1, param2);
    return result;
  }
  
  // Don't forget parameter validation
  async doSomething(param1: string, param2: number): Promise<ResultType> {
    // Missing validation
    return {} as ResultType;
  }
}
```

## Testing Patterns

### ✅ Correct Testing Structure
```typescript
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { databaseService } from '../services/supabaseDatabase';

describe('ExampleService', () => {
  let service: ExampleService;

  beforeEach(() => {
    service = ExampleService.getInstance();
  });

  afterEach(() => {
    // Cleanup
  });

  it('should perform operation successfully', async () => {
    // Arrange
    const param1 = 'test';
    const param2 = 100;

    // Act
    const result = await service.performOperation(param1, param2);

    // Assert
    expect(result).toBeDefined();
    expect(result.success).toBe(true);
  });

  it('should throw error for invalid parameters', async () => {
    // Arrange
    const param1 = '';
    const param2 = -1;

    // Act & Assert
    await expect(service.performOperation(param1, param2))
      .rejects
      .toThrow('Invalid parameters');
  });
});
```

### ❌ Incorrect Testing Patterns
```typescript
// Don't forget test structure
it('should work', async () => {
  const result = await service.performOperation('test', 100);
  // Missing assertions
});

// Don't forget error testing
it('should perform operation', async () => {
  const result = await service.performOperation('test', 100);
  expect(result).toBeDefined();
  // Missing error case testing
});
```

## Configuration Patterns

### ✅ Correct Configuration Structure
```typescript
// config/environment.ts
export const config = {
  database: {
    url: process.env.SUPABASE_URL || '',
    key: process.env.SUPABASE_ANON_KEY || '',
  },
  oracle: {
    pythUrl: process.env.PYTH_URL || 'https://api.pyth.network',
    wsUrl: process.env.PYTH_WS_URL || 'wss://api.pyth.network',
  },
  server: {
    port: parseInt(process.env.PORT || '3002'),
    env: process.env.NODE_ENV || 'development',
  },
} as const;
```

### ❌ Incorrect Configuration Patterns
```typescript
// Don't hardcode values
export const config = {
  database: {
    url: 'https://hardcoded-url.supabase.co',
    key: 'hardcoded-key',
  },
  // Missing environment variable handling
};
```

These patterns should be followed consistently throughout the QuantDesk codebase to ensure maintainability, consistency, and reliability.
