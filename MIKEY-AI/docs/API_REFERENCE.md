# API Reference

## Base URL
```
Production: https://api.solanadefiai.com/v1
Staging: https://staging-api.solanadefiai.com/v1
Local: http://localhost:3000/api/v1
```

## Authentication

### API Key Authentication
All API requests require an API key in the header:

```http
Authorization: Bearer YOUR_API_KEY
```

### Rate Limiting
- **Free Tier**: 100 requests/hour
- **Pro Tier**: 1,000 requests/hour
- **Enterprise**: 10,000 requests/hour

## Market Data Endpoints

### Get Real-time Prices
```http
GET /market/prices
```

**Query Parameters:**
- `symbols` (string, required): Comma-separated list of trading pairs (e.g., "SOL/USD,ETH/USD")
- `source` (string, optional): Price source ("pyth", "switchboard", "aggregated")

**Response:**
```json
{
  "success": true,
  "data": {
    "SOL/USD": {
      "price": 95.50,
      "change_24h": 2.5,
      "volume_24h": 1250000,
      "source": "pyth",
      "timestamp": "2025-10-18T18:34:29Z"
    },
    "ETH/USD": {
      "price": 2450.75,
      "change_24h": -1.2,
      "volume_24h": 890000,
      "source": "switchboard",
      "timestamp": "2025-10-18T18:34:29Z"
    }
  }
}
```

### Get Historical Prices
```http
GET /market/prices/historical
```

**Query Parameters:**
- `symbol` (string, required): Trading pair (e.g., "SOL/USD")
- `interval` (string, required): Time interval ("1m", "5m", "15m", "1h", "4h", "1d")
- `start_time` (string, required): Start time (ISO 8601)
- `end_time` (string, required): End time (ISO 8601)

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "SOL/USD",
    "interval": "1h",
    "prices": [
      {
        "timestamp": "2025-10-18T18:34:29Z",
        "open": 94.20,
        "high": 95.80,
        "low": 93.50,
        "close": 95.50,
        "volume": 125000
      }
    ]
  }
}
```

### Get Market Sentiment
```http
GET /market/sentiment
```

**Query Parameters:**
- `symbol` (string, optional): Trading pair (default: "SOL/USD")
- `timeframe` (string, optional): Timeframe ("1h", "4h", "24h", "7d")

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "SOL/USD",
    "sentiment_score": 0.73,
    "sentiment_label": "bullish",
    "social_volume": 15420,
    "news_sentiment": 0.68,
    "twitter_sentiment": 0.76,
    "reddit_sentiment": 0.71,
    "timestamp": "2025-10-18T18:34:29Z"
  }
}
```

## Wallet Analysis Endpoints

### Track Wallet
```http
POST /wallets/track
```

**Request Body:**
```json
{
  "wallet_address": "ABC123...",
  "label": "Whale Wallet #1",
  "alerts": {
    "large_transactions": true,
    "threshold": 100000
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "wallet_id": "wallet_123",
    "address": "ABC123...",
    "label": "Whale Wallet #1",
    "status": "tracking",
    "created_at": "2025-10-18T18:34:29Z"
  }
}
```

### Get Wallet Analysis
```http
GET /wallets/{wallet_id}/analysis
```

**Query Parameters:**
- `timeframe` (string, optional): Analysis timeframe ("24h", "7d", "30d")
- `include_positions` (boolean, optional): Include position data

**Response:**
```json
{
  "success": true,
  "data": {
    "wallet_id": "wallet_123",
    "address": "ABC123...",
    "total_value": 2500000,
    "portfolio": {
      "SOL": {
        "amount": 25000,
        "value": 2387500,
        "percentage": 95.5
      },
      "USDC": {
        "amount": 112500,
        "value": 112500,
        "percentage": 4.5
      }
    },
    "recent_activity": {
      "transactions_24h": 15,
      "volume_24h": 450000,
      "largest_transaction": 150000
    },
    "positions": [
      {
        "protocol": "drift",
        "type": "long",
        "size": 1000,
        "entry_price": 94.50,
        "current_price": 95.50,
        "pnl": 1000
      }
    ]
  }
}
```

### Get Wallet Transactions
```http
GET /wallets/{wallet_id}/transactions
```

**Query Parameters:**
- `limit` (number, optional): Number of transactions (default: 50, max: 100)
- `offset` (number, optional): Pagination offset
- `type` (string, optional): Transaction type ("swap", "transfer", "liquidation")

**Response:**
```json
{
  "success": true,
  "data": {
    "transactions": [
      {
        "transaction_id": "tx_123",
        "timestamp": "2025-10-18T18:34:29Z",
        "type": "swap",
        "from_token": "SOL",
        "to_token": "USDC",
        "amount_in": 1000,
        "amount_out": 94500,
        "protocol": "jupiter",
        "gas_fee": 0.005
      }
    ],
    "pagination": {
      "total": 150,
      "limit": 50,
      "offset": 0,
      "has_more": true
    }
  }
}
```

## Trading Analysis Endpoints

### Get Liquidations
```http
GET /trading/liquidations
```

**Query Parameters:**
- `protocol` (string, optional): Protocol filter ("drift", "mango", "all")
- `timeframe` (string, optional): Timeframe ("1h", "24h", "7d")
- `min_size` (number, optional): Minimum liquidation size

**Response:**
```json
{
  "success": true,
  "data": {
    "liquidations": [
      {
        "liquidation_id": "liq_123",
        "timestamp": "2025-10-18T18:34:29Z",
        "protocol": "drift",
        "wallet_address": "ABC123...",
        "position_type": "long",
        "size": 5000,
        "collateral": 100000,
        "liquidation_price": 94.50,
        "market_price": 94.20
      }
    ],
    "summary": {
      "total_liquidations": 25,
      "total_volume": 1250000,
      "largest_liquidation": 150000
    }
  }
}
```

### Get Market Analysis
```http
GET /trading/analysis
```

**Query Parameters:**
- `symbol` (string, required): Trading pair
- `timeframe` (string, optional): Analysis timeframe ("1h", "4h", "24h")

**Response:**
```json
{
  "success": true,
  "data": {
    "symbol": "SOL/USD",
    "technical_analysis": {
      "trend": "bullish",
      "support_levels": [87.20, 89.50, 92.10],
      "resistance_levels": [95.50, 98.20, 101.50],
      "rsi": 65.4,
      "macd": {
        "macd": 1.2,
        "signal": 0.8,
        "histogram": 0.4
      }
    },
    "market_structure": {
      "regime": "bull_market",
      "volatility": "medium",
      "liquidity": "high"
    },
    "predictions": {
      "short_term": "bullish",
      "medium_term": "neutral",
      "confidence": 0.72
    }
  }
}
```

## AI Query Endpoints

### Natural Language Query
```http
POST /ai/query
```

**Request Body:**
```json
{
  "query": "What's the current sentiment for SOL and show me the top whale wallets",
  "context": {
    "symbols": ["SOL/USD"],
    "timeframe": "24h"
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "response": "Based on current data, SOL sentiment is bullish (73% positive). Here are the top whale wallets:\n\n1. Wallet ABC123: $2.5M SOL, recently opened long position\n2. Wallet XYZ789: $1.8M SOL, accumulating over past week\n3. Wallet DEF456: $1.2M SOL, active trader",
    "sources": [
      "sentiment_analysis",
      "wallet_tracking",
      "transaction_monitoring"
    ],
    "confidence": 0.85,
    "timestamp": "2025-10-18T18:34:29Z"
  }
}
```

### Get AI Analysis
```http
GET /ai/analysis
```

**Query Parameters:**
- `type` (string, required): Analysis type ("market", "wallet", "sentiment")
- `symbol` (string, optional): Trading pair
- `wallet_id` (string, optional): Wallet ID

**Response:**
```json
{
  "success": true,
  "data": {
    "analysis_type": "market",
    "symbol": "SOL/USD",
    "insights": [
      "Strong bullish momentum with 15% gain in 24h",
      "Volume spike indicates institutional interest",
      "Support level at $87.20 holding strong"
    ],
    "recommendations": [
      "Consider taking profits at $98.20 resistance",
      "Monitor for potential pullback to $92.10 support",
      "Watch for volume confirmation on next move"
    ],
    "risk_assessment": {
      "risk_level": "medium",
      "volatility": "high",
      "liquidity": "good"
    }
  }
}
```

## WebSocket Endpoints

### Real-time Market Data
```javascript
const ws = new WebSocket('wss://api.solanadefiai.com/ws/market-data');

ws.onopen = function() {
  // Subscribe to SOL/USD price updates
  ws.send(JSON.stringify({
    action: 'subscribe',
    channel: 'prices',
    symbols: ['SOL/USD']
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Price update:', data);
};
```

### Real-time Wallet Updates
```javascript
const ws = new WebSocket('wss://api.solanadefiai.com/ws/wallet-updates');

ws.onopen = function() {
  // Subscribe to wallet updates
  ws.send(JSON.stringify({
    action: 'subscribe',
    channel: 'wallet',
    wallet_id: 'wallet_123'
  }));
};
```

## Error Handling

### Error Response Format
```json
{
  "success": false,
  "error": {
    "code": "INVALID_SYMBOL",
    "message": "Invalid trading pair symbol",
    "details": {
      "symbol": "INVALID/USD",
      "valid_symbols": ["SOL/USD", "ETH/USD", "BTC/USD"]
    }
  },
  "timestamp": "2025-10-18T18:34:29Z"
}
```

### Common Error Codes
- `INVALID_API_KEY`: API key is missing or invalid
- `RATE_LIMIT_EXCEEDED`: Rate limit exceeded
- `INVALID_SYMBOL`: Invalid trading pair symbol
- `WALLET_NOT_FOUND`: Wallet address not found
- `INSUFFICIENT_DATA`: Insufficient data for analysis
- `INTERNAL_ERROR`: Internal server error

## SDK Examples

### JavaScript/TypeScript
```typescript
import { SolanaDeFiAI } from '@solanadefiai/sdk';

const client = new SolanaDeFiAI({
  apiKey: 'your_api_key',
  baseUrl: 'https://api.solanadefiai.com/v1'
});

// Get real-time prices
const prices = await client.market.getPrices(['SOL/USD', 'ETH/USD']);

// Track a wallet
const wallet = await client.wallets.track({
  address: 'ABC123...',
  label: 'Whale Wallet'
});

// Natural language query
const response = await client.ai.query(
  'What are the top performing wallets today?'
);
```

### Python
```python
from solanadefiai import SolanaDeFiAI

client = SolanaDeFiAI(api_key='your_api_key')

# Get market data
prices = client.market.get_prices(['SOL/USD', 'ETH/USD'])

# Track wallet
wallet = client.wallets.track(
    address='ABC123...',
    label='Whale Wallet'
)

# AI query
response = client.ai.query('Show me the market sentiment for SOL')
```

## Rate Limiting

### Limits by Tier
- **Free**: 100 requests/hour
- **Pro**: 1,000 requests/hour
- **Enterprise**: 10,000 requests/hour

### Rate Limit Headers
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
```

### Exceeding Limits
When rate limits are exceeded, the API returns a 429 status code:

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 3600 seconds.",
    "retry_after": 3600
  }
}
```
