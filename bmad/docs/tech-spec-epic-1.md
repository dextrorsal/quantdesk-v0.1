# Epic Technical Specification: Complete Advanced Order Types

Date: 2025-10-27
Author: Dex
Epic ID: 1
Status: Draft

---

## Overview

This epic implements professional-grade advanced order types for the QuantDesk perpetual DEX platform. The current system has basic market and limit orders fully implemented, but stop-loss, take-profit, trailing stop, and bracket orders remain scaffolded. This epic delivers production-ready implementations to enable professional risk management and trading strategies for users.

The epic requires coordinated development across multiple system layers: smart contract extensions (Solana program), backend monitoring service, database schema updates, WebSocket real-time updates, and UI integration. All implementation aligns with the existing layered monolith architecture, leveraging current infrastructure without introducing new architectural patterns.

## Objectives and Scope

### In-Scope
- **Smart Contract Extensions:** Add stop-loss, take-profit, trailing stop, and bracket order instructions to existing Solana program
- **Backend Monitoring Service:** Real-time price monitoring and conditional order execution (1-second polling interval)
- **Database Schema:** Extend orders table with conditional order fields (trigger_price, trailing_percentage, parent_bracket_id)
- **WebSocket Integration:** Real-time order lifecycle updates for all advanced order types
- **UI Implementation:** Order entry forms and visual feedback for advanced order types
- **Testing:** Comprehensive unit, integration, and end-to-end testing for all order types

### Out-of-Scope
- TWAP/Iceberg orders (already implemented in smart contract)
- Order routing across multiple venues
- Institutional API endpoints
- Mobile-specific UI implementations
- Off-chain order types (all orders execute on Solana)

## System Architecture Alignment

### Services and Modules

| Service/Module | Responsibility | Owner | Inputs | Outputs |
|----------------|---------------|-------|---------|---------|
| **Smart Contract: `advanced_orders.rs`** | Conditional order execution logic | Dev | Market price, trigger conditions | Order state updates |
| **Backend: `orderExecutionMonitorService.ts`** | Real-time price monitoring and trigger evaluation | Dev | Pyth Oracle prices, user orders | Order execution triggers |
| **Backend: `advancedOrders.ts` routes** | Order management API endpoints | Dev | HTTP requests | Database updates, WebSocket broadcasts |
| **Database: `orders` table** | Extended schema for conditional fields | Dev | Insert/Update queries | Order records with conditional fields |
| **WebSocket: `websocket.ts` service** | Real-time order updates to clients | Dev | Order lifecycle events | Client notifications |
| **Frontend: Advanced Order Forms** | User interface for order creation | Dev | User input | Order API calls |

### Architecture Constraints
- **Single Solana Program:** All order logic in existing `quantdesk-perp-dex` program (C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw on devnet)
- **Backend Layered Monolith:** Extend existing `advancedOrders.ts` route with real implementation
- **Monitoring Service:** Polling-based approach (1-second interval) rather than event-driven for simplicity
- **Database:** Extend existing `orders` table with nullable columns to support new order types

---

## Detailed Design

### Services and Modules

#### 1. Smart Contract Extensions

**File:** `contracts/programs/quantdesk-perp-dex/src/instructions/advanced_orders.rs`

**New Instructions to Add:**
```rust
// Stop-loss order placement
pub fn place_stop_loss_order(ctx: Context<PlaceStopLossOrder>, ...) -> Result<()>

// Take-profit order placement  
pub fn place_take_profit_order(ctx: Context<PlaceTakeProfitOrder>, ...) -> Result<()>

// Trailing stop order placement
pub fn place_trailing_stop_order(ctx: Context<PlaceTrailingStopOrder>, ...) -> Result<()>

// Bracket order placement (single instruction for entry + stop + take-profit)
pub fn place_bracket_order(ctx: Context<PlaceBracketOrder>, ...) -> Result<()>

// Conditional order execution (called by backend monitor)
pub fn execute_conditional_order(ctx: Context<ExecuteConditionalOrder>, ...) -> Result<()>
```

**State Extensions:**
- Extend `Order` struct with optional fields: `trigger_price`, `trailing_percentage`, `parent_bracket_id`
- Add enum variants for order subtypes in `OrderType`

#### 2. Backend Monitoring Service

**New Service:** `backend/src/services/orderExecutionMonitorService.ts`

**Responsibilities:**
- Poll Pyth Oracle for current market prices every 1 second
- Query database for active conditional orders (status='pending', not expired)
- Evaluate trigger conditions for each order
- Execute orders when conditions met (call smart contract, update database)
- Broadcast execution events via WebSocket

**Key Functions:**
```typescript
async function monitorConditionalOrders(): Promise<void>
async function evaluateStopLossOrder(order: Order, currentPrice: number): Promise<boolean>
async function evaluateTakeProfitOrder(order: Order, currentPrice: number): Promise<boolean>
async function evaluateTrailingStop(order: Order, currentPrice: number): Promise<void>
async function executeOrder(orderId: string): Promise<void>
```

#### 3. Backend API Routes

**File:** `backend/src/routes/advancedOrders.ts` (currently placeholder)

**Endpoints to Implement:**
```
POST /api/advanced-orders/stop-loss
  Body: { market_id, size, direction, trigger_price, user_address }
  Response: { order_id, status }

POST /api/advanced-orders/take-profit
  Body: { market_id, size, direction, trigger_price, user_address }
  Response: { order_id, status }

POST /api/advanced-orders/trailing-stop
  Body: { market_id, size, direction, trailing_percentage, user_address }
  Response: { order_id, status }

POST /api/advanced-orders/bracket
  Body: { market_id, entry_price, stop_loss_price, take_profit_price, size, direction }
  Response: { order_ids: [entry_id, stop_id, take_id], status }

GET /api/advanced-orders/:user_address
  Response: { orders: [...] }

DELETE /api/advanced-orders/:order_id
  Response: { success: true }
```

#### 4. Database Schema Extensions

**File:** `database/migrations/XXXX_add_advanced_order_fields.sql`

**Schema Changes:**
```sql
ALTER TABLE orders ADD COLUMN IF NOT EXISTS order_subtype VARCHAR(50);
ALTER TABLE orders ADD COLUMN IF NOT EXISTS trigger_price DECIMAL(20,8);
ALTER TABLE orders ADD COLUMN IF NOT EXISTS trailing_percentage DECIMAL(5,2);
ALTER TABLE orders ADD COLUMN IF NOT EXISTS parent_bracket_id UUID REFERENCES orders(id);
ALTER TABLE orders ADD COLUMN IF NOT EXISTS tracking_price DECIMAL(20,8); -- for trailing stops

CREATE INDEX idx_orders_subtype ON orders(order_subtype) WHERE order_subtype IS NOT NULL;
CREATE INDEX idx_orders_bracket ON orders(parent_bracket_id) WHERE parent_bracket_id IS NOT NULL;
```

**Order Subtype Enum:**
- `stop_loss`
- `take_profit`
- `trailing_stop`
- `bracket_entry`
- `bracket_stop`
- `bracket_take_profit`

### Data Models and Contracts

#### Database Schema Extensions

**Orders Table Extended Fields:**
```sql
CREATE TABLE orders (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_address VARCHAR(44) NOT NULL,
  market_id VARCHAR(20) NOT NULL,
  order_type VARCHAR(20) NOT NULL,  -- limit, market, stop_loss, etc.
  order_subtype VARCHAR(50),         -- NEW: stop_loss, take_profit, trailing_stop, bracket_*
  direction VARCHAR(10) NOT NULL,    -- long, short
  size DECIMAL(20,8) NOT NULL,
  price DECIMAL(20,8),               -- for limit orders
  trigger_price DECIMAL(20,8),       -- NEW: for conditional orders
  trailing_percentage DECIMAL(5,2),  -- NEW: for trailing stops (e.g., 2.5%)
  parent_bracket_id UUID REFERENCES orders(id),  -- NEW: for bracket orders
  tracking_price DECIMAL(20,8),      -- NEW: current best price (for trailing stops)
  status VARCHAR(20) NOT NULL,       -- pending, active, filled, cancelled
  filled_size DECIMAL(20,8) DEFAULT 0,
  created_at TIMESTAMP DEFAULT now(),
  expires_at TIMESTAMP,
  -- ... existing fields
);
```

#### Smart Contract Data Structures

**Extended Order Type Enum:**
```rust
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq)]
pub enum OrderSubtype {
    StopLoss,
    TakeProfit,
    TrailingStop,
    BracketEntry,
    BracketStop,
    BracketTakeProfit,
}

#[account]
pub struct Order {
    pub user: Pubkey,
    pub market: Pubkey,
    pub order_type: OrderType,
    pub order_subtype: Option<OrderSubtype>,  // NEW
    pub direction: Direction,
    pub size: u64,
    pub price: Option<u64>,                   // for limit orders
    pub trigger_price: Option<u64>,           // NEW: for conditional orders
    pub trailing_percentage: Option<u16>,      // NEW: basis points (e.g., 250 = 2.5%)
    pub parent_bracket_id: Option<Pubkey>,    // NEW: for bracket orders
    pub tracking_price: Option<u64>,          // NEW: current best price
    pub status: OrderStatus,
    pub filled_size: u64,
    pub created_at: i64,
    pub expires_at: Option<i64>,
    // ... existing fields
}
```

### APIs and Interfaces

#### Backend API Endpoints

**Stop-Loss Order:**
```
POST /api/advanced-orders/stop-loss
Request:
{
  "market_id": "btc-perp",
  "direction": "short",
  "size": "1.0",
  "trigger_price": 44500.0,
  "user_address": "ABC123..."
}

Response:
{
  "order_id": "uuid-here",
  "status": "pending",
  "type": "stop_loss",
  "trigger_price": 44500.0,
  "created_at": "2025-10-27T..."
}
```

**Take-Profit Order:**
```
POST /api/advanced-orders/take-profit
Request:
{
  "market_id": "btc-perp",
  "direction": "long",
  "size": "0.5",
  "trigger_price": 46500.0,
  "user_address": "ABC123..."
}

Response: (similar structure to stop-loss)
```

**Trailing Stop Order:**
```
POST /api/advanced-orders/trailing-stop
Request:
{
  "market_id": "btc-perp",
  "direction": "long",
  "size": "1.0",
  "trailing_percentage": 2.5,
  "user_address": "ABC123..."
}

Response:
{
  "order_id": "uuid-here",
  "status": "pending",
  "type": "trailing_stop",
  "trailing_percentage": 2.5,
  "created_at": "2025-10-27T..."
}
```

**Bracket Order:**
```
POST /api/advanced-orders/bracket
Request:
{
  "market_id": "btc-perp",
  "entry_price": 45000.0,
  "stop_loss_price": 44500.0,
  "take_profit_price": 46500.0,
  "direction": "long",
  "size": "1.0",
  "user_address": "ABC123..."
}

Response:
{
  "order_ids": {
    "entry": "uuid-entry",
    "stop_loss": "uuid-stop",
    "take_profit": "uuid-take"
  },
  "status": "pending"
}
```

#### WebSocket Events

**Order Execution Notification:**
```json
{
  "event": "order_executed",
  "order_id": "uuid-here",
  "type": "stop_loss",
  "executed_price": 44495.0,
  "filled_size": "1.0",
  "timestamp": "2025-10-27T...",
  "user_address": "ABC123..."
}
```

**Trailing Stop Update:**
```json
{
  "event": "trailing_stop_updated",
  "order_id": "uuid-here",
  "new_trigger_price": 45150.0,
  "timestamp": "2025-10-27T..."
}
```

### Workflows and Sequencing

#### Stop-Loss Order Lifecycle

1. **User places stop-loss order** via POST `/api/advanced-orders/stop-loss`
2. **Backend validates** order parameters (trigger price below market for sell, above for buy)
3. **Backend creates database record** with status='pending'
4. **Backend stores order on-chain** (if needed for custody/transparency)
5. **Monitoring service** (every 1 second):
   - Queries pending stop-loss orders from database
   - Gets current market price from Pyth Oracle
   - Evaluates: `current_price <= trigger_price` (for sell) or `current_price >= trigger_price` (for buy)
   - If condition met: Execute order
6. **Order execution:**
   - Backend calls smart contract instruction `execute_conditional_order()`
   - Update database: status='filled', filled_size=size
   - Broadcast WebSocket event to user
7. **Position update:** Close/reduce user's existing position

#### Trailing Stop Order Lifecycle

1. **User places trailing stop** via POST `/api/advanced-orders/trailing-stop`
2. **Backend creates record** with trailing_percentage (e.g., 2.5%)
3. **Monitoring service** (every 1 second):
   - Gets current market price
   - For long positions: `trigger_price = current_price - (current_price * trailing_percentage / 100)`
   - For short positions: `trigger_price = current_price + (current_price * trailing_percentage / 100)`
   - Updates `tracking_price` in database (best price seen since order placed)
   - Adjusts trigger_price dynamically as price moves favorably
   - Sends trailing stop update event via WebSocket
   - If current_price crosses trigger_price: Execute order
4. **Order execution:** Same as stop-loss flow

#### Bracket Order Lifecycle

1. **User places bracket order** via POST `/api/advanced-orders/bracket`
2. **Backend creates 3 linked orders:**
   - Entry order (status='active')
   - Stop-loss order (status='pending', parent_bracket_id=entry_id)
   - Take-profit order (status='pending', parent_bracket_id=entry_id)
3. **Entry order executes** (when market reaches entry_price)
   - Updates status='filled'
   - Activates stop-loss and take-profit (change status to 'active')
4. **Monitoring service** monitors stop-loss and take-profit:
   - If either triggers: Execute that order
   - Other order automatically cancels
5. **Position closed:** User's position fully managed by bracket order

---

## Non-Functional Requirements

### Performance

**API Response Time:**
- POST `/api/advanced-orders/*` endpoints must respond within <500ms
- GET `/api/advanced-orders/:user_address` must respond within <500ms
- Monitoring service evaluation loop completes within <100ms per 100 orders

**Real-Time Monitoring:**
- Price polling interval: 1 second (configurable)
- Order trigger detection latency: <2 seconds from price crossing trigger
- WebSocket broadcast latency: <100ms from database update

**Database:**
- Queries for conditional orders (by status='pending') must complete within <200ms
- Batch updates for order status changes: <50ms per batch

### Security

**Authentication & Authorization:**
- All POST requests require SIWS wallet signature verification
- Users can only create orders for their own wallet address
- Order cancellation requires signature from original order creator
- Backend monitor service runs with minimal privileges (read DB, write DB, call smart contract)

**Data Integrity:**
- All order execution must be verified on-chain via smart contract
- Database state must match smart contract state (eventual consistency is acceptable, but discrepancies must be resolved)
- Order trigger evaluation must use oracle prices directly (no cached/interpolated values)
- Circuit breaker: Pause all conditional order execution if price deviation >5% from last price

**Threat Mitigation:**
- Prevent front-running: Backend monitor service evaluates all orders atomically for same market
- Prevent DOS: Rate limiting on order creation (10 orders/minute per user)
- Prevent manipulation: Order validation ensures trigger price is within ±10% of current market price
- Audit logging: All order state changes logged with timestamps and user addresses

### Reliability/Availability

**Uptime:**
- Monitoring service must be running 99.9% of the time
- Database queries must have 99.9% success rate
- Smart contract transaction success rate must be >95% (retries handle failures)

**Recovery:**
- If monitoring service crashes, restart automatically within 30 seconds
- Failed smart contract transactions: Retry up to 3 times with exponential backoff
- Missed triggers: Alert administrators if price crosses trigger but order not executed within 5 seconds
- Database connection failures: Automatic reconnection with order queuing

**Graceful Degradation:**
- If Pyth Oracle unavailable: Pause conditional order monitoring, alert users
- If WebSocket disconnected: Order still executes, notification sent when reconnected
- If smart contract call fails: Order remains 'pending', retry on next monitoring cycle

### Observability

**Logging:**
- Order creation: Log order_id, user_address, type, trigger_price, timestamp
- Order execution: Log order_id, executed_price, latency, timestamp
- Order cancellation: Log order_id, user_address, timestamp
- Monitoring service: Log every evaluation cycle (count of orders checked, execution count)

**Metrics:**
- Total active conditional orders by type (stop_loss, take_profit, trailing_stop, bracket)
- Orders executed per hour
- Average trigger latency (time from price crossing to execution)
- Failed executions count (with reason)
- Monitoring service uptime %
- Smart contract transaction success rate

**Alerting:**
- Alert if monitoring service down for >1 minute
- Alert if >10 failed executions in 5 minutes
- Alert if trigger latency >5 seconds
- Alert if Oracle price staleness >2 minutes

**Tracing:**
- Trace order lifecycle from creation to execution
- Track which orders triggered in each monitoring cycle
- Track smart contract transaction IDs for all executions

---

## Dependencies and Integrations

### External Dependencies

**Pyth Network Oracle:**
- **Purpose:** Real-time price feeds for trigger evaluation
- **Version:** Latest SDK (1.4+)
- **Endpoints:** `https://pyth-api.pyth.network` (devnet)
- **Data Format:** JSON with price, confidence, exponent fields
- **Rate Limits:** 100 requests/second

**Solana Web3.js:**
- **Purpose:** Smart contract interactions for order execution
- **Version:** 1.87.2
- **Usage:** Transaction building, signing, sending for `execute_conditional_order` instruction

**Supabase Client (PostgreSQL):**
- **Purpose:** Order storage and state management
- **Version:** 1.38.0+
- **Database:** PostgreSQL 14+
- **Extensions Required:** `uuid-ossp` for UUID generation

### Internal Service Dependencies

**Redis Client:**
- **Purpose:** Pub/Sub for WebSocket notifications
- **Status:** Enabled (from previous epic)
- **Configuration:** `REDIS_URL=redis://localhost:6379`

**WebSocket Service:**
- **Purpose:** Real-time order updates to frontend
- **File:** `backend/src/services/websocket.ts`
- **Method:** `broadcastOrderUpdate(orderId, eventData)`

**Database Service:**
- **Purpose:** Database abstraction layer
- **File:** `backend/src/services/supabaseDatabase.ts`
- **Functions:** `createOrder()`, `updateOrder()`, `getPendingOrders()`

**Logger:**
- **Purpose:** Application logging
- **File:** `backend/src/utils/logger.ts`
- **Usage:** Error logging, trace logging

---

## Acceptance Criteria (Authoritative)

### AC1: Stop-Loss Order Creation
- **Given** a user wants to protect their long BTC position
- **When** they create a stop-loss order with trigger price $44,500
- **Then** the order is saved to the database with status='pending'
- **And** the order is stored on-chain via smart contract
- **And** a confirmation is returned with order_id

### AC2: Stop-Loss Order Execution
- **Given** a pending stop-loss order exists for BTC at trigger $44,500
- **When** BTC price drops to $44,495
- **Then** the monitoring service detects the trigger within 2 seconds
- **And** the order is executed via smart contract within 5 seconds
- **And** the user's position is closed
- **And** a WebSocket notification is sent to the user

### AC3: Take-Profit Order Execution
- **Given** a pending take-profit order exists for ETH at trigger $3,200
- **When** ETH price reaches $3,200.50
- **Then** the order executes within 2 seconds
- **And** the user's position is closed
- **And** the order is marked as 'filled' in database

### AC4: Trailing Stop Order Logic
- **Given** a trailing stop order with 2.5% trailing percentage
- **When** BTC price moves from $45,000 → $46,000 → $47,000
- **Then** the trigger price updates from $43,875 → $44,700 → $45,825
- **And** WebSocket sends trailing_stop_updated events for each adjustment
- **When** price drops to $45,825
- **Then** the trailing stop order executes immediately

### AC5: Bracket Order Activation
- **Given** a bracket order with entry at $45,000, stop at $44,500, take-profit at $46,500
- **When** price crosses $45,000
- **Then** the entry order fills
- **And** stop-loss and take-profit orders change status from 'pending' to 'active'
- **And** WebSocket broadcasts bracket activation event

### AC6: Bracket Order Completion
- **Given** an active bracket order (entry filled)
- **When** price reaches stop-loss trigger before take-profit
- **Then** the stop-loss order executes
- **And** the take-profit order is automatically cancelled
- **And** the position is closed at stop-loss price
- **And** user receives execution notification for both stop and cancellation

### AC7: Order Cancellation
- **Given** a pending stop-loss order
- **When** user cancels the order before execution
- **Then** the order status changes to 'cancelled' in database
- **And** the order is removed from monitoring service queue
- **And** a cancellation confirmation is returned

### AC8: Multiple Orders Same Market
- **Given** 5 pending conditional orders for BTC
- **When** monitoring service evaluates orders
- **Then** all 5 orders are checked within <1 second
- **And** executed orders are triggered atomically (no race conditions)

### AC9: Price Deviation Protection
- **Given** monitoring service is active
- **When** Oracle price deviation >5% from last known price
- **Then** all conditional order execution is paused
- **And** administrators are alerted
- **And** order evaluation resumes when deviation <2%

### AC10: Database Consistency
- **Given** an order is executed on-chain
- **When** database update fails
- **Then** the transaction is retried up to 3 times
- **And** if still failing, order is marked 'failed' and admin is alerted
- **And** user can see order status in UI (doesn't appear 'stuck')

---

## Traceability Mapping

| AC ID | Spec Section | Component/API | Test Idea |
|-------|-------------|--------------|-----------|
| AC1 | Detailed Design: APIs | POST /api/advanced-orders/stop-loss | Unit: mock order creation, verify DB insert |
| AC1 | Detailed Design: Smart Contract | place_stop_loss_order() | Integration: call Anchor test, verify on-chain account |
| AC2 | Detailed Design: Monitoring | orderExecutionMonitorService.ts | E2E: Mock price feed, verify execution within 2s |
| AC2 | Detailed Design: Smart Contract | execute_conditional_order() | Integration: Verify position closed on-chain |
| AC2 | Detailed Design: APIs | WebSocket order_executed event | E2E: Verify notification received in test client |
| AC3 | Detailed Design: Monitoring | evaluateTakeProfitOrder() | Unit: Test price comparison logic |
| AC4 | Detailed Design: Monitoring | evaluateTrailingStop() | Unit: Test trailing price calculation |
| AC4 | Detailed Design: APIs | WebSocket trailing_stop_updated event | Integration: Verify event broadcast on update |
| AC5 | Detailed Design: APIs | POST /api/advanced-orders/bracket | E2E: Create bracket order, verify 3 orders created |
| AC5 | Detailed Design: Database | parent_bracket_id relationship | Unit: Verify foreign key constraints |
| AC6 | Detailed Design: Monitoring | Bracket order execution logic | E2E: Verify stop-loss triggers, take-profit cancels |
| AC7 | Detailed Design: APIs | DELETE /api/advanced-orders/:id | Unit: Test order status update to 'cancelled' |
| AC8 | Detailed Design: Monitoring | Batch order evaluation | Performance: Test with 100 concurrent orders |
| AC9 | Detailed Design: NFR Security | Circuit breaker logic | Unit: Mock 5%+ price deviation, verify pause |
| AC10 | Detailed Design: NFR Reliability | Database retry logic | Integration: Mock DB failure, verify retries |

---

## Risks, Assumptions, Open Questions

### Risks

**R1: Smart Contract Program Size Limit**
- **Risk:** Solana program size limit (10KB) may be exceeded with new instructions
- **Mitigation:** Optimize Rust code, use CPI for complex logic if needed, monitor program size during build
- **Impact:** High - Could prevent deployment

**R2: Price Oracle Staleness**
- **Risk:** Pyth Oracle prices may be stale, causing missed triggers or premature execution
- **Mitigation:** Implement staleness checks (reject prices >2 minutes old), alert on stale data
- **Impact:** High - Financial risk

**R3: Smart Contract Transaction Failures**
- **Risk:** Transaction may fail due to network congestion or insufficient fees
- **Mitigation:** Automatic retry with higher fee escalation, alert on persistent failures
- **Impact:** Medium - User experience degradation

**R4: Race Conditions in Monitoring Service**
- **Risk:** Multiple order executions for same trigger condition if not properly synchronized
- **Mitigation:** Use database locks or transaction isolation, evaluate orders atomically
- **Impact:** Medium - Incorrect order execution

**R5: High-Frequency Price Updates**
- **Risk:** Excessive API calls to Pyth Oracle may hit rate limits
- **Mitigation:** Batch requests, implement caching with 1-second TTL
- **Impact:** Low - Performance degradation

### Assumptions

**A1: Oracle Prices Available**
- **Assumption:** Pyth Network provides reliable price feeds for all supported markets (BTC, ETH, SOL)
- **Evidence:** Currently working in production for basic orders
- **Confidence:** High

**A2: Database Performance**
- **Assumption:** Supabase PostgreSQL can handle <200ms queries for pending order lookups
- **Evidence:** Current queries perform well
- **Confidence:** High

**A3: Monitoring Service Uptime**
- **Assumption:** Backend service runs continuously with auto-restart capability
- **Evidence:** Infrastructure already supports process management
- **Confidence:** High

**A4: Smart Contract Upgrade Path**
- **Assumption:** Can upgrade existing program without breaking basic orders
- **Evidence:** Program uses Anchor Upgrade Authority pattern
- **Confidence:** Medium - Requires careful testing

### Open Questions

**Q1: Off-Chain vs On-Chain Storage**
- **Question:** Should order state be stored on-chain or only in database?
- **Decision Required:** Smart contract must verify order existence for execution, but full data can be off-chain
- **Owner:** Dev + Architect
- **Timeline:** Need answer before smart contract implementation

**Q2: Order Expiration**
- **Question:** Should conditional orders expire if not triggered after X days?
- **Suggestion:** Yes, default 7 days, configurable by user
- **Impact:** Low - Feature enhancement

**Q3: Partial Fills for Large Orders**
- **Question:** How to handle partial fills for stop-loss orders if market liquidity is low?
- **Suggestion:** Execute remaining size across multiple blocks if needed
- **Impact:** Medium - Order execution reliability

---

## Test Strategy Summary

### Unit Tests

**Backend:**
- Test order creation API endpoints (POST requests)
- Test monitoring service evaluation logic (stop-loss, take-profit, trailing)
- Test database operations (insert, update, query)
- Test WebSocket event broadcasting
- Test circuit breaker logic (price deviation >5%)

**Smart Contract:**
- Test `place_stop_loss_order()` with various trigger prices
- Test `execute_conditional_order()` with invalid inputs
- Test trailing stop price calculation logic
- Test bracket order linking (parent_bracket_id validation)

### Integration Tests

**Backend + Database:**
- Test order creation → database insert → status='pending'
- Test order execution → database update → status='filled'
- Test order cancellation → database delete

**Backend + Smart Contract:**
- Test order execution flow: monitor → smart contract call → database update
- Test failed smart contract transaction → retry logic
- Test position closing after stop-loss execution

**Backend + WebSocket:**
- Test order execution → WebSocket notification sent
- Test trailing stop update → WebSocket event broadcast

### End-to-End Tests

**Complete User Journey:**
1. User places stop-loss order via UI
2. Monitor service detects trigger condition
3. Order executes on-chain
4. Position is closed
5. User receives WebSocket notification
6. UI updates to show order as 'filled'

**Bracket Order Journey:**
1. User places bracket order via UI
2. Entry order fills when price crosses entry point
3. Stop-loss and take-profit activate automatically
4. User receives bracket activation notification
5. Stop-loss executes before take-profit
6. User receives execution and cancellation notifications
7. UI shows all 3 orders with correct statuses

### Performance Tests

**Load Testing:**
- Create 1000 concurrent pending conditional orders
- Verify monitoring service completes evaluation within <2 seconds
- Verify database queries remain <200ms under load

**Latency Testing:**
- Measure order execution latency from trigger to on-chain confirmation
- Target: <5 seconds total

### Security Tests

**Input Validation:**
- Test invalid trigger prices (outside ±10% of current market)
- Test negative trailing percentages
- Test oversized orders (beyond position size)

**Authorization:**
- Test order creation with invalid user signature
- Test order cancellation by non-owner

**Circuit Breaker:**
- Test automatic pause when price deviation >5%
- Test resume when price deviation normalizes

---

_This technical specification serves as the implementation guide for Epic 1: Complete Advanced Order Types._
_All development must trace back to this specification, and all acceptance criteria must be validated before epic completion._

