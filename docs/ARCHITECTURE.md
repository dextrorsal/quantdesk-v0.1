# QuantDesk Architecture Documentation

## Overview

QuantDesk is a decentralized perpetual futures trading platform built on the Solana ecosystem, inspired by platforms like AsterDex, Quanto, and Drift. This document outlines the complete architecture and technology stack.

## Platform Analysis

### Reference Platforms Studied

#### 1. AsterDex (https://www.asterdex.com/en/futures/v1/BTCUSDT)
**Key Features Observed:**
- Dark theme UI with professional trading interface
- Real-time price display with 24h change indicators
- Order book with buy/sell depth visualization
- Multiple order types: Market, Limit, Stop Limit
- Leverage settings (up to 25x observed)
- Position management with P&L tracking
- Account balance and margin information
- Multi-asset trading support

#### 2. Quanto Trade (https://quanto.trade/en/markets/BTC-USD-SWAP-LIN)
**Key Features Observed:**
- Purple accent theme with clean design
- TradingView chart integration
- Order book with price/size/total columns
- Long/Short position buttons
- Leverage slider and position sizing
- Real-time funding rate display
- Portfolio management tabs

#### 3. Drift Protocol (https://app.drift.trade/BTC-PERP)
**Key Features Observed:**
- Professional trading interface
- Advanced charting with TradingView
- Order book with depth visualization
- Multiple order types and advanced options
- Position and order management
- Account health monitoring
- Real-time market data

## Technology Stack

### Layer 0: Core Infrastructure
- **Validator Clients**: Agave, Firedancer, Mithril, Sig
- **RPC & Data Platforms**: Helius, QuickNode, Alchemy, Triton One, Ankr
- **Cloud & Deployment**: AWS (Solana Blueprints), Google Cloud
- **Development Tools**: Solana CLI, Anchor CLI, Solana Playground

### Layer 1: On-Chain Development (Smart Contracts)
- **Programming Language**: Rust (primary for Solana programs)
- **Framework**: Anchor Framework
- **Libraries**: Solana Program Library (SPL)
- **Compilers**: Solang (Solidity compiler for Solana)

### Layer 2: Off-Chain & Client-Side
- **Frontend Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Wallet Integration**: @solana/wallet-adapter
- **Charts**: TradingView Charting Library
- **Real-time**: WebSocket connections

### Layer 3: Backend Services
- **Database**: PostgreSQL with Supabase
- **API Framework**: Node.js with Express
- **Real-time**: Supabase Realtime, WebSocket server
- **Authentication**: Supabase Auth + Solana wallet integration
- **Data Indexing**: Helius Enhanced APIs

### Layer 4: Application-Specific Tools
- **Treasury Management**: Squads
- **Payments**: Blinks, Token Extensions
- **Security**: Radar, Solsec, Trident, OtterSec

## System Architecture

### Frontend Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend                          │
├─────────────────────────────────────────────────────────────┤
│  Components: TradingInterface, OrderBook, PriceChart       │
│  State: Zustand stores for trading, positions, orders      │
│  Services: Solana Web3.js, WebSocket client               │
│  UI: Tailwind CSS, Lucide React icons                     │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                 Wallet Integration                         │
├─────────────────────────────────────────────────────────────┤
│  Phantom, Solflare, Sollet wallet support                 │
│  Transaction signing and account management                │
└─────────────────────────────────────────────────────────────┘
```

### Backend Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (Node.js)                     │
├─────────────────────────────────────────────────────────────┤
│  REST APIs: Market data, orders, positions                 │
│  WebSocket: Real-time updates, order book                  │
│  Authentication: Supabase Auth + Solana verification       │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Layer                                │
├─────────────────────────────────────────────────────────────┤
│  Supabase PostgreSQL: User data, positions, orders         │
│  Real-time subscriptions for live updates                  │
│  Caching layer for market data                             │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                Blockchain Layer (Solana)                   │
├─────────────────────────────────────────────────────────────┤
│  Smart Contracts: Trading, order matching, positions       │
│  RPC Endpoints: Helius/QuickNode for blockchain access     │
│  Program Library: SPL tokens, associated accounts          │
└─────────────────────────────────────────────────────────────┘
```

## Smart Contract Architecture

### Core Programs
1. **Trading Program**: Handles order placement, matching, and execution
2. **Position Program**: Manages user positions and P&L calculations
3. **Margin Program**: Handles margin requirements and liquidations
4. **Oracle Program**: Price feed integration and validation
5. **Fee Program**: Trading fee collection and distribution

### Key Data Structures
```rust
// Order structure
pub struct Order {
    pub id: u64,
    pub user: Pubkey,
    pub market: Pubkey,
    pub side: OrderSide, // Buy/Sell
    pub order_type: OrderType, // Market/Limit/Stop
    pub size: u64,
    pub price: Option<u64>,
    pub timestamp: i64,
    pub status: OrderStatus,
}

// Position structure
pub struct Position {
    pub user: Pubkey,
    pub market: Pubkey,
    pub side: PositionSide, // Long/Short
    pub size: u64,
    pub entry_price: u64,
    pub margin: u64,
    pub leverage: u8,
    pub unrealized_pnl: i64,
}
```

## Database Schema (Supabase)

### Core Tables
```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    wallet_address TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Markets table
CREATE TABLE markets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    base_asset TEXT NOT NULL,
    quote_asset TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Positions table
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    market_id UUID REFERENCES markets(id),
    side TEXT NOT NULL, -- 'long' or 'short'
    size DECIMAL NOT NULL,
    entry_price DECIMAL NOT NULL,
    current_price DECIMAL,
    margin DECIMAL NOT NULL,
    leverage INTEGER NOT NULL,
    unrealized_pnl DECIMAL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Orders table
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    market_id UUID REFERENCES markets(id),
    side TEXT NOT NULL, -- 'buy' or 'sell'
    order_type TEXT NOT NULL, -- 'market', 'limit', 'stop'
    size DECIMAL NOT NULL,
    price DECIMAL,
    status TEXT DEFAULT 'pending', -- 'pending', 'filled', 'cancelled'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Real-time Data Flow

### WebSocket Events
```typescript
// Market data updates
interface MarketDataUpdate {
  type: 'market_data';
  symbol: string;
  price: number;
  change24h: number;
  volume24h: number;
  timestamp: number;
}

// Order book updates
interface OrderBookUpdate {
  type: 'order_book';
  symbol: string;
  bids: [number, number][]; // [price, size]
  asks: [number, number][];
  timestamp: number;
}

// Position updates
interface PositionUpdate {
  type: 'position';
  userId: string;
  positionId: string;
  unrealizedPnl: number;
  marginRatio: number;
  timestamp: number;
}
```

## Security Considerations

### Smart Contract Security
- Input validation and sanitization
- Access control and authorization
- Reentrancy protection
- Integer overflow/underflow protection
- Oracle price validation

### Frontend Security
- Wallet signature verification
- Input validation and sanitization
- Secure WebSocket connections (WSS)
- XSS and CSRF protection

### Backend Security
- API rate limiting
- Input validation
- SQL injection prevention
- Secure authentication flows
- CORS configuration

## Performance Optimization

### Frontend
- Component memoization
- Virtual scrolling for large lists
- Image optimization
- Code splitting and lazy loading
- Efficient state management

### Backend
- Database indexing
- Query optimization
- Caching strategies
- Connection pooling
- Load balancing

### Smart Contracts
- Efficient data structures
- Minimal storage operations
- Parallel transaction processing
- Gas optimization

## Development Workflow

### Frontend Development
1. Component development with Storybook
2. Unit testing with Jest/Vitest
3. Integration testing with React Testing Library
4. E2E testing with Playwright

### Backend Development
1. API development with Express
2. Database migrations with Supabase
3. Smart contract development with Anchor
4. Testing with Anchor Test framework

### Deployment
1. Frontend: Vercel/Netlify
2. Backend: AWS/Google Cloud
3. Smart contracts: Solana mainnet/devnet
4. Database: Supabase managed PostgreSQL

## Monitoring and Analytics

### Key Metrics
- Trading volume and frequency
- User engagement and retention
- System performance and uptime
- Error rates and response times
- Smart contract gas usage

### Tools
- Application monitoring: Sentry, LogRocket
- Analytics: Mixpanel, Google Analytics
- Infrastructure: AWS CloudWatch, Supabase Dashboard
- Blockchain: Solana Explorer, Solscan

## Future Enhancements

### Phase 2 Features
- Advanced order types (OCO, bracket orders)
- Mobile application
- Social trading features
- Advanced charting tools
- API for third-party integrations

### Phase 3 Features
- Cross-chain trading
- Institutional features
- Advanced risk management
- Automated trading strategies
- Governance token integration
