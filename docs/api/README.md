# QuantDesk API Documentation

## 🚀 **Complete API Reference**

This document provides comprehensive API documentation for the QuantDesk perpetual DEX platform, showcasing our complete backend service architecture and integration capabilities.

## 📊 **API Overview**

```mermaid
graph TB
    subgraph "API Gateway"
        GATEWAY[Express.js Gateway<br/>Port 3002]
        MIDDLEWARE[Middleware Stack<br/>Auth, Validation, Rate Limiting]
        ROUTES[Route Handlers<br/>Business Logic]
    end
    
    subgraph "Core Services"
        TRADING[Trading Service<br/>Position Management]
        MARKET[Market Service<br/>Price Data]
        USER[User Service<br/>Account Management]
        AI[AI Service<br/>MIKEY Integration]
    end
    
    subgraph "External Integrations"
        SOLANA[Solana RPC<br/>Blockchain Interaction]
        SUPABASE[Supabase<br/>Database Operations]
        PYTH[Pyth Network<br/>Price Feeds]
        REDIS[Redis Cache<br/>Performance Layer]
    end
    
    GATEWAY --> MIDDLEWARE
    MIDDLEWARE --> ROUTES
    ROUTES --> TRADING
    ROUTES --> MARKET
    ROUTES --> USER
    ROUTES --> AI
    
    TRADING --> SOLANA
    MARKET --> PYTH
    USER --> SUPABASE
    AI --> REDIS
```

## 🔐 **Authentication**

QuantDesk uses **SIWS (Solana In-App Web3 Signing)** for authentication. This provides secure, wallet-based authentication without passwords.

### **SIWS Authentication Flow**

#### **Step 1: Get Nonce**
```http
POST /api/siws/nonce
Content-Type: application/json

{
  "walletPubkey": "YOUR_WALLET_PUBLIC_KEY"
}
```

**Response:**
```json
{
  "nonce": "random_base58_string"
}
```

#### **Step 2: Sign and Verify**
```http
POST /api/siws/verify
Content-Type: application/json

{
  "walletPubkey": "YOUR_WALLET_PUBLIC_KEY",
  "signature": "SIGNED_NONCE_BASE58",
  "nonce": "nonce_from_step_1",
  "ref": "OPTIONAL_REFERRER_PUBKEY"
}
```

**Response:**
```json
{
  "success": true
}
```

This sets an HTTP-only cookie `qd_session` that authenticates future requests.

#### **Step 3: Use Authentication**
All protected endpoints automatically read the `qd_session` cookie. No additional headers needed.

### **Logout**
```http
POST /api/siws/logout
```

Clears the session cookie.

### **Session Management**
- Sessions are stored in Redis (optional in development)
- Sessions expire after 7 days
- Users are automatically created on first sign-in

## 📈 **Market Data Endpoints**

### **Get Available Markets**
```http
GET /api/markets
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "symbol": "SOL-PERP",
      "name": "Solana Perpetual",
      "baseAsset": "SOL",
      "quoteAsset": "USDC",
      "isActive": true,
      "minSize": 0.01,
      "maxSize": 1000,
      "tickSize": 0.01,
      "stepSize": 0.01
    }
  ]
}
```

### **Get Market Data**
```http
GET /api/markets/{symbol}
```

**Parameters:**
- `symbol` (string): Market symbol (e.g., "SOL-PERP")

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "SOL-PERP",
    "price": 100.50,
    "volume": 1250000.75,
    "change24h": 0.025,
    "high24h": 102.30,
    "low24h": 98.75,
    "open24h": 98.00,
    "timestamp": "2024-01-15T10:30:00Z",
    "orderBook": {
      "bids": [
        { "price": 100.49, "size": 1.5 },
        { "price": 100.48, "size": 2.3 }
      ],
      "asks": [
        { "price": 100.51, "size": 1.2 },
        { "price": 100.52, "size": 3.1 }
      ]
    }
  }
}
```

### **Get Historical Data**
```http
GET /api/markets/{symbol}/history
```

**Parameters:**
- `symbol` (string): Market symbol
- `interval` (string): Time interval (1m, 5m, 15m, 1h, 4h, 1d)
- `limit` (number): Number of candles (max 1000)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "timestamp": "2024-01-15T10:00:00Z",
      "open": 100.00,
      "high": 101.50,
      "low": 99.75,
      "close": 100.50,
      "volume": 1250.75
    }
  ]
}
```

## 📝 **Order Management**

### **Get Orders**
```http
GET /api/orders
```

**Note:** Order management endpoints are currently in development. API returns placeholder responses.

### **Place Order**
```http
POST /api/orders
```

### **Cancel Order**
```http
DELETE /api/orders/:id
```

---

## 💼 **Trading Endpoints**

### **Get Portfolio**
```http
GET /api/portfolio
```

**Response:**
```json
{
  "success": true,
  "data": {
    "totalValue": 10000.50,
    "totalPnL": 250.75,
    "totalPositions": 3,
    "availableBalance": 5000.25,
    "marginUsed": 2000.00,
    "marginAvailable": 3000.25,
    "positions": [
      {
        "id": "pos_123",
        "market": "SOL-PERP",
        "side": "long",
        "size": 1.5,
        "entryPrice": 95.00,
        "currentPrice": 100.50,
        "pnL": 8.25,
        "leverage": 10,
        "margin": 150.00,
        "liquidationPrice": 85.50
      }
    ]
  }
}
```

### **Get Positions**
```http
GET /api/positions
```

**Query Parameters:**
- `market` (string, optional): Filter by market symbol
- `status` (string, optional): Filter by status (open, closed)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "pos_123",
      "market": "SOL-PERP",
      "side": "long",
      "size": 1.5,
      "entryPrice": 95.00,
      "currentPrice": 100.50,
      "pnL": 8.25,
      "leverage": 10,
      "margin": 150.00,
      "liquidationPrice": 85.50,
      "status": "open",
      "createdAt": "2024-01-15T09:00:00Z",
      "updatedAt": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### **Open Position**
```http
POST /api/positions
```

**Request Body:**
```json
{
  "market": "SOL-PERP",
  "side": "long",
  "size": 1.0,
  "leverage": 10,
  "entryPrice": 100.00,
  "stopLoss": 95.00,
  "takeProfit": 110.00
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "pos_124",
    "market": "SOL-PERP",
    "side": "long",
    "size": 1.0,
    "entryPrice": 100.00,
    "leverage": 10,
    "margin": 100.00,
    "status": "open",
    "createdAt": "2024-01-15T10:30:00Z"
  }
}
```

### **Close Position**
```http
DELETE /api/positions/{positionId}
```

**Parameters:**
- `positionId` (string): Position ID

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "pos_124",
    "status": "closed",
    "pnL": 5.25,
    "closedAt": "2024-01-15T11:00:00Z"
  }
}
```

## 📝 **Order Management**

### **Place Order**
```http
POST /api/orders
```

**Request Body:**
```json
{
  "market": "SOL-PERP",
  "side": "buy",
  "size": 1.0,
  "price": 99.50,
  "orderType": "limit",
  "timeInForce": "GTC"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "ord_456",
    "market": "SOL-PERP",
    "side": "buy",
    "size": 1.0,
    "price": 99.50,
    "orderType": "limit",
    "status": "pending",
    "createdAt": "2024-01-15T10:30:00Z"
  }
}
```

### **Get Orders**
```http
GET /api/orders
```

**Query Parameters:**
- `market` (string, optional): Filter by market
- `status` (string, optional): Filter by status
- `limit` (number, optional): Number of orders (max 100)

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "ord_456",
      "market": "SOL-PERP",
      "side": "buy",
      "size": 1.0,
      "price": 99.50,
      "orderType": "limit",
      "status": "filled",
      "filledSize": 1.0,
      "filledPrice": 99.50,
      "createdAt": "2024-01-15T10:30:00Z",
      "filledAt": "2024-01-15T10:31:00Z"
    }
  ]
}
```

### **Cancel Order**
```http
DELETE /api/orders/{orderId}
```

**Parameters:**
- `orderId` (string): Order ID

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "ord_456",
    "status": "cancelled",
    "cancelledAt": "2024-01-15T10:35:00Z"
  }
}
```

## 🤖 **AI Integration Endpoints**

### **Get AI Analysis**
```http
GET /api/ai/analysis/{market}
```

**Parameters:**
- `market` (string): Market symbol

**Response:**
```json
{
  "success": true,
  "data": {
    "market": "SOL-PERP",
    "sentiment": "bullish",
    "confidence": 0.85,
    "recommendation": "buy",
    "riskLevel": "medium",
    "insights": [
      "Strong upward momentum detected",
      "Volume increasing with price",
      "Support level holding at $95"
    ],
    "technicalIndicators": {
      "rsi": 65.5,
      "macd": "bullish",
      "movingAverage": "above"
    },
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### **Get Trading Signals**
```http
GET /api/ai/signals
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "market": "SOL-PERP",
      "action": "buy",
      "strength": "strong",
      "confidence": 0.90,
      "reason": "Breakout above resistance",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ]
}
```

### **Get Risk Assessment**
```http
GET /api/ai/risk-assessment
```

**Response:**
```json
{
  "success": true,
  "data": {
    "overallRisk": "medium",
    "portfolioRisk": 0.15,
    "marketRisk": 0.25,
    "recommendations": [
      "Consider reducing position sizes",
      "Set stop-loss orders",
      "Monitor market volatility"
    ],
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### **Chat with MIKEY**
```http
POST /api/ai/chat
```

**Request Body:**
```json
{
  "message": "What's your analysis of SOL-PERP?",
  "context": {
    "market": "SOL-PERP",
    "userPositions": ["pos_123"]
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "response": "Based on my analysis, SOL-PERP shows strong bullish momentum with increasing volume. The price has broken above the key resistance level at $100, suggesting potential for further upside. However, I recommend setting a stop-loss at $95 to manage risk.",
    "confidence": 0.88,
    "sources": [
      "Technical analysis",
      "Volume analysis",
      "Market sentiment"
    ],
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## 🎁 **Referrals**

### **Get Referral Summary**
```http
GET /api/referrals/summary?wallet=WALLET_PUBKEY
```

**Response:**
```json
{
  "count": 5,
  "earnings": 125.50,
  "refs": [
    {
      "id": "ref_123",
      "referee_pubkey": "ABC123...",
      "earned": 25.00,
      "created_at": "2024-01-15T10:00:00Z"
    }
  ]
}
```

### **Preview Referral Earnings**
```http
GET /api/referrals/preview?referrer=WALLET_PUBKEY
```

### **Claim Referral Earnings**
```http
POST /api/referrals/claim
Content-Type: application/json

{
  "referrer": "YOUR_WALLET_PUBKEY"
}
```

### **Activate Referral**
```http
POST /api/referrals/activate
Content-Type: application/json

{
  "referee_pubkey": "REFEREE_WALLET_PUBKEY",
  "minimum_volume": 100
}
```

---

## 💬 **Chat System**

### **Get Chat Channels**
```http
GET /api/chat/channels
```

### **Get Chat History**
```http
GET /api/chat/history?channelId=global&limit=50
```

### **Send Chat Message**
```http
POST /api/chat/send
Content-Type: application/json

{
  "channelId": "global",
  "message": "Hello from @wallet_pubkey!"
}
```

---

## 💼 **Account Management**

### **Get Trading Accounts**
```http
GET /api/accounts/trading-accounts
```

### **Create Trading Account**
```http
POST /api/accounts/trading-accounts
Content-Type: application/json

{
  "name": "My Trading Account"
}
```

### **Get Account Balances**
```http
GET /api/accounts/balances
```

### **Transfer Between Accounts**
```http
POST /api/accounts/transfer
Content-Type: application/json

{
  "fromSubAccountId": "sub_account_1",
  "toSubAccountId": "sub_account_2",
  "asset": "USDC",
  "amount": 1000.00
}
```

### **Manage Delegates**
```http
# Add delegate
POST /api/accounts/delegates

# Get delegates
GET /api/accounts/delegates

# Update delegate permissions
PUT /api/accounts/delegates/:id

# Remove delegate
DELETE /api/accounts/delegates/:id
```

---

## 👤 **User Management**

### **Get User Profile**
```http
GET /api/users/profile
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "user_789",
    "wallet_pubkey": "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU",
    "created_at": "2024-01-01T00:00:00Z",
    "last_login": "2024-01-15T10:30:00Z"
  }
}
```

## 📊 **Analytics Endpoints**

### **Get Trading History**
```http
GET /api/analytics/trading-history
```

**Query Parameters:**
- `startDate` (string): Start date (ISO format)
- `endDate` (string): End date (ISO format)
- `market` (string, optional): Filter by market

**Response:**
```json
{
  "success": true,
  "data": {
    "totalTrades": 150,
    "winningTrades": 95,
    "losingTrades": 55,
    "winRate": 0.633,
    "totalPnL": 1250.75,
    "averageWin": 25.50,
    "averageLoss": -15.25,
    "profitFactor": 1.67,
    "trades": [
      {
        "id": "trade_123",
        "market": "SOL-PERP",
        "side": "long",
        "size": 1.0,
        "entryPrice": 95.00,
        "exitPrice": 100.50,
        "pnL": 5.50,
        "timestamp": "2024-01-15T10:30:00Z"
      }
    ]
  }
}
```

### **Get Performance Metrics**
```http
GET /api/analytics/performance
```

**Response:**
```json
{
  "success": true,
  "data": {
    "totalReturn": 0.125,
    "annualizedReturn": 0.45,
    "sharpeRatio": 1.85,
    "maxDrawdown": 0.08,
    "volatility": 0.25,
    "beta": 1.2,
    "alpha": 0.05,
    "period": "30d"
  }
}
```

## 🔧 **System Endpoints**

### **Health Check**
```http
GET /api/health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2024-01-15T10:30:00Z",
    "services": {
      "database": "healthy",
      "redis": "healthy",
      "solana": "healthy",
      "pyth": "healthy"
    },
    "uptime": 86400,
    "version": "1.0.0"
  }
}
```

### **Get System Status**
```http
GET /api/status
```

**Response:**
```json
{
  "success": true,
  "data": {
    "activeUsers": 1250,
    "totalVolume": 5000000.75,
    "openPositions": 3500,
    "systemLoad": 0.45,
    "lastUpdate": "2024-01-15T10:30:00Z"
  }
}
```

## ⚠️ **Error Handling**

### **Error Response Format**
```json
{
  "success": false,
  "error": {
    "code": "INSUFFICIENT_BALANCE",
    "message": "Insufficient balance to open position",
    "details": {
      "required": 1000.00,
      "available": 500.00
    }
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### **Common Error Codes**
- `INSUFFICIENT_BALANCE`: Not enough balance for operation
- `INVALID_MARKET`: Market symbol not found
- `POSITION_SIZE_TOO_LARGE`: Position size exceeds limits
- `MARKET_CLOSED`: Market is currently closed
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `UNAUTHORIZED`: Authentication required
- `FORBIDDEN`: Insufficient permissions

## 📚 **Rate Limiting**

### **Rate Limits**
- **Public API**: 100 requests per minute
- **Trading API**: 10 requests per minute (order placement)
- **Authentication**: 5 attempts per 15 minutes
- **Admin API**: 50 requests per minute
- **Webhook API**: 20 requests per minute

### **Rate Limit Headers**
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248600
```

**Note:** Rate limits are applied per-minute, not per-hour, to prevent abuse while allowing normal usage.

## 🔒 **Security**

### **HTTPS Only**
All API endpoints require HTTPS encryption.

### **CORS Policy**
```typescript
{
  "origin": ["https://quantdesk.com", "https://app.quantdesk.com"],
  "methods": ["GET", "POST", "PUT", "DELETE"],
  "headers": ["Content-Type", "Authorization", "X-API-Key"]
}
```

### **Input Validation**
All inputs are validated and sanitized before processing.

---

**QuantDesk API Documentation: Complete API reference showcasing our comprehensive backend service architecture with trading, AI integration, and analytics capabilities.**
