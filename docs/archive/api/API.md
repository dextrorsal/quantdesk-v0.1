# QuantDesk API Documentation

Welcome to the QuantDesk API! This documentation provides comprehensive information about our REST API endpoints for trading, portfolio management, and market data.

## 🔗 **Base URL**

```
Development: https://localhost:3001
Production: https://quantdesk.app
Local: http://localhost:3002
```

## 🔐 **Authentication**

Most endpoints require authentication using JWT tokens:

```bash
Authorization: Bearer <your_jwt_token>
```

### Getting Started

1. **Register/Login** to get your JWT token
2. **Include the token** in all API requests
3. **Refresh tokens** as needed

## 📊 **Core Endpoints**

### Markets

#### Get All Markets
```http
GET /api/supabase-oracle/markets
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "d87a99b4-148a-49c2-a2ad-ca1ee17a9372",
      "symbol": "BTC-PERP",
      "base_asset": "BTC",
      "quote_asset": "USDT",
      "is_active": true,
      "max_leverage": 100,
      "tick_size": 0.01,
      "step_size": 0.001,
      "min_order_size": 0.001,
      "max_order_size": 1000000
    }
  ],
  "count": 11,
  "timestamp": 1703123456789
}
```

### Orders

#### Place Order
```http
POST /api/orders
Authorization: Bearer <token>
Content-Type: application/json

{
  "marketId": "d87a99b4-148a-49c2-a2ad-ca1ee17a9372",
  "side": "buy",
  "size": 0.1,
  "price": 45000,
  "orderType": "limit"
}
```

#### Get Orders
```http
GET /api/orders?status=open&limit=50
Authorization: Bearer <token>
```

### Positions

#### Get Positions
```http
GET /api/positions
Authorization: Bearer <token>
```

#### Open Position
```http
POST /api/positions
Authorization: Bearer <token>
Content-Type: application/json

{
  "marketId": "d87a99b4-148a-49c2-a2ad-ca1ee17a9372",
  "side": "long",
  "size": 0.1,
  "leverage": 10
}
```

### Portfolio Analytics

#### Get Portfolio Metrics
```http
GET /api/portfolio/metrics
Authorization: Bearer <token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "totalValue": 100000,
    "totalPnL": 5000,
    "totalPnLPercentage": 5.0,
    "dayPnL": 1000,
    "dayPnLPercentage": 1.0,
    "sharpeRatio": 1.2,
    "maxDrawdown": 0.15,
    "winRate": 0.65,
    "totalTrades": 150,
    "avgTradeSize": 0.5
  }
}
```

#### Get Risk Analysis
```http
GET /api/portfolio/risk
Authorization: Bearer <token>
```

### Advanced Orders

#### Place Advanced Order
```http
POST /api/advanced-orders
Authorization: Bearer <token>
Content-Type: application/json

{
  "marketId": "d87a99b4-148a-49c2-a2ad-ca1ee17a9372",
  "orderType": "stop_loss",
  "side": "sell",
  "size": 0.1,
  "triggerPrice": 40000,
  "limitPrice": 39500
}
```

### Cross-Collateral

#### Get Collateral Accounts
```http
GET /api/cross-collateral/accounts
Authorization: Bearer <token>
```

#### Add Collateral
```http
POST /api/cross-collateral/accounts
Authorization: Bearer <token>
Content-Type: application/json

{
  "collateralType": "SOL",
  "amount": 10
}
```

## 📈 **WebSocket API**

### Connection
```javascript
const ws = new WebSocket('ws://localhost:3002/ws');

ws.onopen = () => {
  console.log('Connected to QuantDesk WebSocket');
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Subscribe to Market Data
```javascript
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'market_data',
  marketId: 'd87a99b4-148a-49c2-a2ad-ca1ee17a9372'
}));
```

### Subscribe to User Updates
```javascript
ws.send(JSON.stringify({
  type: 'subscribe',
  channel: 'user_updates',
  token: 'your_jwt_token'
}));
```

## 🔧 **Error Handling**

### Error Response Format
```json
{
  "success": false,
  "error": "Error message",
  "code": "ERROR_CODE",
  "details": {
    "field": "additional error details"
  },
  "timestamp": 1703123456789
}
```

### Common Error Codes
- `MISSING_TOKEN`: Authentication required
- `INVALID_TOKEN`: Invalid or expired token
- `INSUFFICIENT_BALANCE`: Not enough balance
- `INVALID_ORDER`: Order validation failed
- `MARKET_CLOSED`: Market is not available
- `RATE_LIMITED`: Too many requests

## 📊 **Rate Limits**

| Endpoint Type | Limit | Window |
|---------------|-------|--------|
| Public Data | 100 requests | 1 minute |
| Trading | 30 requests | 1 minute |
| Authentication | 5 requests | 15 minutes |
| WebSocket | 1000 messages | 1 minute |

## 🧪 **Testing**

### Test Environment
Use our test environment for development:
```
Base URL: https://api-test.quantdesk.app
Testnet: Solana Devnet
```

### Example Requests

#### Using cURL
```bash
# Get markets
curl -X GET "https://api-dev.quantdesk.app/api/supabase-oracle/markets"

# Place order
curl -X POST "https://api-dev.quantdesk.app/api/orders" \
  -H "Authorization: Bearer your_token" \
  -H "Content-Type: application/json" \
  -d '{
    "marketId": "d87a99b4-148a-49c2-a2ad-ca1ee17a9372",
    "side": "buy",
    "size": 0.1,
    "price": 45000,
    "orderType": "limit"
  }'
```

#### Using JavaScript
```javascript
const response = await fetch('https://api-dev.quantdesk.app/api/orders', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer your_token',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    marketId: 'd87a99b4-148a-49c2-a2ad-ca1ee17a9372',
    side: 'buy',
    size: 0.1,
    price: 45000,
    orderType: 'limit'
  })
});

const data = await response.json();
console.log(data);
```

## 📚 **SDKs**

### JavaScript/TypeScript
```bash
npm install @quantdesk/sdk
```

```javascript
import { QuantDeskClient } from '@quantdesk/sdk';

const client = new QuantDeskClient({
  apiKey: 'your_api_key',
  environment: 'development'
});

// Place order
const order = await client.orders.create({
  marketId: 'd87a99b4-148a-49c2-a2ad-ca1ee17a9372',
  side: 'buy',
  size: 0.1,
  price: 45000,
  orderType: 'limit'
});
```

### Python
```bash
pip install quantdesk-sdk
```

```python
from quantdesk import QuantDeskClient

client = QuantDeskClient(api_key='your_api_key', environment='development')

# Place order
order = client.orders.create(
    market_id='d87a99b4-148a-49c2-a2ad-ca1ee17a9372',
    side='buy',
    size=0.1,
    price=45000,
    order_type='limit'
)
```

## 🔄 **Webhooks**

### Setup Webhook
```http
POST /api/webhooks
Authorization: Bearer <token>
Content-Type: application/json

{
  "url": "https://your-app.com/webhook",
  "events": ["order_filled", "position_opened", "position_closed"],
  "secret": "your_webhook_secret"
}
```

### Webhook Events
- `order_filled`: Order was executed
- `position_opened`: New position opened
- `position_closed`: Position closed
- `deposit_completed`: Deposit confirmed
- `withdrawal_completed`: Withdrawal confirmed

## 📞 **Support**

- **Documentation**: [GitBook](https://quantdesk.gitbook.io)
- **API Status**: [Status Page](https://status.quantdesk.app)
- **Support**: [Discord](https://discord.gg/quantdesk)
- **Issues**: [GitHub Issues](https://github.com/dextrorsal/quantdesk/issues)

---

*QuantDesk API - Professional trading infrastructure for the decentralized future*