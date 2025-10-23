# Core Workflows

## User Authentication and Trading Session Workflow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend API
    participant S as Supabase
    participant SC as Smart Contract
    participant SOL as Solana RPC
    
    U->>F: Login with wallet
    F->>B: POST /api/auth/login
    B->>S: Verify user credentials
    S-->>B: User data + JWT token
    B->>SC: Query wallet balance
    SC->>SOL: getAccountInfo
    SOL-->>SC: Account balance
    SC-->>B: Balance data
    B-->>F: Auth success + portfolio data
    F-->>U: Trading interface loaded
    
    Note over U,SOL: User authenticated and ready to trade
```

## Real-Time Market Data Flow

```mermaid
sequenceDiagram
    participant F as Frontend
    participant B as Backend API
    participant D as Data Ingestion
    participant P as Pyth Network
    participant S as Supabase
    
    F->>B: WebSocket connection
    B->>D: Subscribe to price feeds
    D->>P: Connect to price stream
    P-->>D: Real-time price updates
    D->>D: Validate staleness & confidence
    D->>S: Store price history
    D-->>B: Processed price data
    B-->>F: Price updates via WebSocket
    F->>F: Update trading interface
    
    Note over F,S: Continuous real-time data flow
```

## AI-Assisted Trading Order Workflow

```mermaid
sequenceDiagram
    participant U as User
    participant F as Frontend
    participant B as Backend API
    participant A as MIKEY AI
    participant LLM as External LLM
    participant SC as Smart Contract
    participant P as Pyth Oracle
    
    U->>F: Request trading recommendation
    F->>B: POST /api/ai/analyze
    B->>A: Market analysis request
    A->>P: Get current market data
    P-->>A: Price feeds
    A->>LLM: Analyze market conditions
    LLM-->>A: Trading recommendation
    A-->>B: Analysis + recommendation
    B-->>F: AI recommendation
    F-->>U: Display recommendation
    
    U->>F: Execute recommended trade
    F->>B: POST /api/trading/order
    B->>SC: Submit trading order
    SC->>P: Validate price
    P-->>SC: Price confirmation
    SC-->>B: Transaction result
    B-->>F: Order confirmation
    F-->>U: Show execution result
```

## Error Handling and Recovery Workflow

```mermaid
sequenceDiagram
    participant F as Frontend
    participant B as Backend API
    participant P as Pyth Network
    participant SC as Smart Contract
    participant SOL as Solana RPC
    
    F->>B: Place trading order
    B->>P: Get current price
    P-->>B: Price data (stale)
    B->>B: Detect staleness > threshold
    B-->>F: Error: Price data stale
    
    F->>B: Retry with fresh data
    B->>P: Get fresh price
    P-->>B: Current price data
    B->>SC: Execute trade
    SC->>SOL: Submit transaction
    SOL-->>SC: Transaction failed (insufficient funds)
    SC-->>B: Error: Insufficient balance
    B-->>F: Error: Insufficient funds
    F-->>F: Show error message + retry option
    
    Note over F,SOL: Comprehensive error handling with recovery paths
```

## Position Management and Risk Monitoring

```mermaid
sequenceDiagram
    participant D as Data Ingestion
    participant B as Backend API
    participant SC as Smart Contract
    participant P as Pyth Network
    participant S as Supabase
    participant F as Frontend
    
    D->>P: Monitor price feeds
    P-->>D: Price updates
    D->>B: Price change alert
    B->>SC: Check position health
    SC->>SC: Calculate P&L and margin
    SC-->>B: Position status
    
    alt Position at risk
        B->>S: Log risk event
        B->>F: Send liquidation warning
        F-->>F: Display risk alert
    else Position healthy
        B->>S: Update position data
        B->>F: Send position update
        F-->>F: Update portfolio display
    end
    
    Note over D,F: Continuous risk monitoring and position management
```
