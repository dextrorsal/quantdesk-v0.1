# Synchronization Strategy

## Event-Driven Architecture (Recommended)

**Expert Recommendation**: Use Anchor events for all significant state changes.

**Benefits**:
- Most reliable and scalable synchronization method
- Real-time updates without polling
- Efficient data transfer
- Built-in retry mechanisms

**Implementation**:

### 1. Anchor Events Definition
```rust
#[event]
pub struct OrderFilled {
    pub market: Pubkey,
    pub position_id: u64,
    pub order_id: u64,
    pub price: u64,
    pub quantity: u64,
    pub taker_fee: u64,
    pub maker_fee: u64,
}

#[event]
pub struct PositionUpdated {
    pub position_id: u64,
    pub user: Pubkey,
    pub market: Pubkey,
    pub size: u64,
    pub entry_price: u64,
    pub unrealized_pnl: i64,
}

#[event]
pub struct LiquidationExecuted {
    pub position_id: u64,
    pub user: Pubkey,
    pub market: Pubkey,
    pub liquidated_size: u64,
    pub liquidation_price: u64,
    pub penalty_fee: u64,
}
```

### 2. Off-Chain Event Listener
```typescript
// Event listener service
import { Program } from "@coral-xyz/anchor";
import { Connection } from "@solana/web3.js";

export class EventListener {
  private program: Program;
  private connection: Connection;

  async subscribeToEvents() {
    // Order filled events
    this.program.addEventListener("OrderFilled", async (event, slot, signature) => {
      await this.handleOrderFilled(event, signature);
    });

    // Position updated events
    this.program.addEventListener("PositionUpdated", async (event, slot, signature) => {
      await this.handlePositionUpdated(event, signature);
    });

    // Liquidation events
    this.program.addEventListener("LiquidationExecuted", async (event, slot, signature) => {
      await this.handleLiquidation(event, signature);
    });
  }

  private async handleOrderFilled(event: any, signature: string) {
    // Insert into trades table
    await this.database.query(`
      INSERT INTO trades (signature, market_id, user_id, side, size, price, fees, timestamp)
      VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
    `, [signature, event.market, event.user, event.side, event.quantity, event.price, event.taker_fee]);
  }
}
```

### 3. Database Event Storage
```sql
-- Event storage tables
CREATE TABLE program_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signature VARCHAR(64) UNIQUE NOT NULL,
    slot BIGINT NOT NULL,
    event_type TEXT NOT NULL,
    event_data JSONB NOT NULL,
    processed BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for efficient querying
CREATE INDEX idx_program_events_signature ON program_events(signature);
CREATE INDEX idx_program_events_slot ON program_events(slot);
CREATE INDEX idx_program_events_type ON program_events(event_type);
```
