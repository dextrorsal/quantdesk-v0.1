# QuantDesk API Documentation

## Overview

The QuantDesk API provides comprehensive endpoints for trading operations, market data, user management, and system monitoring.

**Base URL**: `http://localhost:3002/api`

## Authentication

Most endpoints require authentication via JWT tokens. Include the token in the Authorization header:

```
Authorization: Bearer <your-jwt-token>
```

## Endpoints

### Authentication

#### POST `/auth/authenticate`
Authenticate a user with wallet signature.

**Request Body**:
```json
{
  "publicKey": "string",
  "signature": "string",
  "message": "string"
}
```

**Response**:
```json
{
  "success": true,
  "token": "jwt-token",
  "user": {
    "id": "string",
    "publicKey": "string",
    "createdAt": "timestamp"
  }
}
```

#### GET `/auth/profile`
Get current user profile.

**Headers**: `Authorization: Bearer <token>`

**Response**:
```json
{
  "success": true,
  "user": {
    "id": "string",
    "publicKey": "string",
    "balance": "number",
    "createdAt": "timestamp"
  }
}
```

### Markets

#### GET `/markets`
Get all available markets.

**Response**:
```json
{
  "success": true,
  "data": [
    {
      "id": "string",
      "symbol": "BTC-PERP",
      "baseAsset": "BTC",
      "quoteAsset": "USDT",
      "isActive": true,
      "maxLeverage": 100,
      "tickSize": 0.01,
      "stepSize": 0.001
    }
  ]
}
```

#### GET `/markets/:id`
Get specific market details.

**Parameters**:
- `id` (string): Market ID

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "string",
    "symbol": "BTC-PERP",
    "baseAsset": "BTC",
    "quoteAsset": "USDT",
    "isActive": true,
    "maxLeverage": 100,
    "tickSize": 0.01,
    "stepSize": 0.001,
    "minOrderSize": 0.001,
    "maxOrderSize": 1000000
  }
}
```

### Orders

#### POST `/orders`
Create a new order.

**Headers**: `Authorization: Bearer <token>`

**Request Body**:
```json
{
  "marketId": "string",
  "side": "long" | "short",
  "type": "market" | "limit",
  "size": "number",
  "price": "number",
  "leverage": "number"
}
```

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "string",
    "marketId": "string",
    "userId": "string",
    "side": "long",
    "type": "market",
    "size": 1.0,
    "price": 50000,
    "leverage": 10,
    "status": "pending",
    "createdAt": "timestamp"
  }
}
```

#### GET `/orders`
Get user's orders.

**Headers**: `Authorization: Bearer <token>`

**Query Parameters**:
- `status` (optional): Filter by order status
- `marketId` (optional): Filter by market
- `limit` (optional): Number of results (default: 50)
- `offset` (optional): Pagination offset

**Response**:
```json
{
  "success": true,
  "data": [
    {
      "id": "string",
      "marketId": "string",
      "side": "long",
      "type": "market",
      "size": 1.0,
      "price": 50000,
      "leverage": 10,
      "status": "filled",
      "createdAt": "timestamp",
      "filledAt": "timestamp"
    }
  ],
  "pagination": {
    "total": 100,
    "limit": 50,
    "offset": 0
  }
}
```

#### DELETE `/orders/:id`
Cancel an order.

**Headers**: `Authorization: Bearer <token>`

**Parameters**:
- `id` (string): Order ID

**Response**:
```json
{
  "success": true,
  "message": "Order cancelled successfully"
}
```

### Positions

#### GET `/positions`
Get user's positions.

**Headers**: `Authorization: Bearer <token>`

**Response**:
```json
{
  "success": true,
  "data": [
    {
      "id": "string",
      "marketId": "string",
      "side": "long",
      "size": 1.0,
      "entryPrice": 50000,
      "markPrice": 51000,
      "leverage": 10,
      "pnl": 1000,
      "unrealizedPnl": 1000,
      "margin": 5000,
      "status": "open",
      "createdAt": "timestamp"
    }
  ]
}
```

#### POST `/positions/:id/close`
Close a position.

**Headers**: `Authorization: Bearer <token>`

**Parameters**:
- `id` (string): Position ID

**Response**:
```json
{
  "success": true,
  "data": {
    "id": "string",
    "status": "closed",
    "closedAt": "timestamp",
    "pnl": 1000
  }
}
```

### Oracle Data

#### GET `/supabase-oracle/prices`
Get latest oracle prices.

**Response**:
```json
{
  "success": true,
  "data": [
    {
      "symbol": "BTC-PERP",
      "baseAsset": "BTC",
      "quoteAsset": "USDT",
      "price": 50000.50,
      "confidence": 0.01,
      "exponent": -8,
      "timestamp": "timestamp"
    }
  ],
  "timestamp": "timestamp",
  "source": "supabase-mcp"
}
```

#### GET `/supabase-oracle/markets`
Get active markets from oracle.

**Response**:
```json
{
  "success": true,
  "data": [
    {
      "id": "string",
      "symbol": "BTC-PERP",
      "baseAsset": "BTC",
      "quoteAsset": "USDT",
      "programId": "string",
      "isActive": true,
      "maxLeverage": 100,
      "tickSize": 0.01,
      "stepSize": 0.001,
      "minOrderSize": 0.001,
      "maxOrderSize": 1000000
    }
  ],
  "count": 3,
  "timestamp": "timestamp"
}
```

### Metrics & Monitoring

#### GET `/metrics/trading`
Get current trading metrics.

**Response**:
```json
{
  "success": true,
  "data": {
    "timestamp": "timestamp",
    "totalVolume24h": 1000000,
    "activeTraders": 150,
    "totalPositions": 500,
    "totalValueLocked": 5000000,
    "averagePositionSize": 10000,
    "leverageDistribution": {
      "1x-5x": 200,
      "5x-10x": 150,
      "10x-20x": 100,
      "20x+": 50
    },
    "marketMetrics": [
      {
        "symbol": "BTC-PERP",
        "volume24h": 500000,
        "priceChange24h": 0.05,
        "openInterest": 2000000,
        "longShortRatio": 1.2
      }
    ]
  }
}
```

#### GET `/metrics/system`
Get system performance metrics.

**Response**:
```json
{
  "success": true,
  "data": {
    "timestamp": "timestamp",
    "apiResponseTime": 25.5,
    "databaseQueryTime": 15.2,
    "activeConnections": 150,
    "errorRate": 0.001,
    "memoryUsage": 256.5,
    "cpuUsage": 45.2
  }
}
```

#### GET `/grafana/metrics`
Get metrics formatted for Grafana.

**Response**:
```json
{
  "success": true,
  "data": {
    "quantdesk_trading_volume_24h": 1000000,
    "quantdesk_active_traders": 150,
    "quantdesk_total_positions": 500,
    "quantdesk_total_value_locked": 5000000,
    "quantdesk_api_response_time": 25.5,
    "quantdesk_memory_usage": 256.5,
    "quantdesk_cpu_usage": 45.2,
    "quantdesk_market_volume_by_symbol": [
      {"symbol": "BTC-PERP", "volume": 500000},
      {"symbol": "ETH-PERP", "volume": 300000}
    ],
    "quantdesk_leverage_distribution": [
      {"leverage_range": "1x-5x", "count": 200},
      {"leverage_range": "5x-10x", "count": 150}
    ]
  }
}
```

### Health Check

#### GET `/health`
Check API health status.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "timestamp",
  "uptime": 3600,
  "environment": "development",
  "version": "1.0.0"
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "success": false,
  "error": "Error type",
  "message": "Detailed error message",
  "code": "ERROR_CODE"
}
```

### Common Error Codes

- `UNAUTHORIZED`: Authentication required
- `FORBIDDEN`: Insufficient permissions
- `NOT_FOUND`: Resource not found
- `VALIDATION_ERROR`: Invalid request data
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Server error

## Rate Limiting

API requests are rate limited:
- **General endpoints**: 1000 requests per 15 minutes
- **Trading endpoints**: 60 requests per minute
- **Authentication endpoints**: 10 requests per minute

Rate limit headers are included in responses:
- `RateLimit-Limit`: Request limit
- `RateLimit-Remaining`: Remaining requests
- `RateLimit-Reset`: Reset timestamp

## WebSocket API

Real-time updates are available via WebSocket connection to `ws://localhost:3002`.

### Connection
```javascript
const ws = new WebSocket('ws://localhost:3002');
```

### Message Types

#### Price Updates
```json
{
  "type": "price_update",
  "data": {
    "symbol": "BTC-PERP",
    "price": 50000.50,
    "timestamp": "timestamp"
  }
}
```

#### Order Updates
```json
{
  "type": "order_update",
  "data": {
    "orderId": "string",
    "status": "filled",
    "filledSize": 1.0,
    "filledPrice": 50000
  }
}
```

#### Position Updates
```json
{
  "type": "position_update",
  "data": {
    "positionId": "string",
    "markPrice": 51000,
    "unrealizedPnl": 1000
  }
}
```

## SDKs

Official SDKs are available for:
- [JavaScript/TypeScript](https://github.com/quantdesk/sdk-js)
- [Python](https://github.com/quantdesk/sdk-python)
- [Rust](https://github.com/quantdesk/sdk-rust)

## Support

For API support:
- **Documentation**: [docs.quantdesk.io](https://docs.quantdesk.io)
- **Issues**: [GitHub Issues](https://github.com/quantdesk/quantdesk/issues)
- **Discord**: [Join our community](https://discord.gg/quantdesk)
