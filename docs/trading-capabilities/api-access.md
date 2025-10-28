# API Access & Integration

QuantDesk exposes REST and WebSocket APIs so partners and advanced traders can plug the terminal’s data and trading workflows into their own systems.

## Base URLs

| Environment | REST Base | WebSocket Base |
|-------------|-----------|----------------|
| Local Dev | `http://localhost:3002` | `ws://localhost:3002` |
| Staging/Prod | `https://api.quantdesk.app` | `wss://api.quantdesk.app` |

Swap in the environment that matches your deployment; all examples below use the local defaults.

## Authentication

- **REST endpoints** – Use SIWS authentication which creates an HTTP-only session cookie (`qd_session`)
- **WebSocket connections** – Socket.IO server reads the session cookie automatically; no separate token needed
- **Session duration** – 7-day expiration with automatic cleanup
- **Wallet-based** – No passwords; all signatures are client-side with your wallet

## Market Data

```http
GET /api/markets                      # List active perps
GET /api/markets/:symbol/price        # Latest price + confidence
GET /api/markets/:symbol/orderbook    # Top-of-book depth
GET /api/markets/:symbol/funding      # Funding history
```

- Symbols are normalized to `BASE-PERP` (e.g., `SOL-PERP`). Aliases like `SOL` or `SOL/USDT` are accepted and mapped internally.
- WebSocket messages mirror REST payloads using the `type` key (`price`, `orderbook`, `trades`) and a versioned envelope.

## Trading & Accounts

```http
# Account Management
GET  /api/accounts/trading-accounts        # List sub-accounts
POST /api/accounts/trading-accounts       # Create sub-account
GET  /api/accounts/balances                # Get all balances
POST /api/accounts/transfer               # Transfer between accounts

# Deposits & Withdrawals
GET  /api/deposits/balances                # Get deposit balances
POST /api/deposits/deposit                # Initiate deposit
POST /api/deposits/deposit/confirm        # Confirm deposit transaction
POST /api/deposits/withdraw               # Initiate withdrawal

# Trading
GET  /api/positions                       # Current positions + P&L (implemented)
POST /api/orders                          # Place order (placeholder)
POST /api/orders/:id/cancel              # Cancel order (placeholder)

# Referrals
GET  /api/referrals/summary               # Get referral earnings
POST /api/referrals/claim                # Claim SOL rewards
```

- Order endpoints expect canonical market IDs and size/price fields. Advanced order types (stop-loss, bracket) use dedicated payloads under `/api/advanced-orders`.
- Cross-account transfers and delegated access live under `/api/accounts/*` routes for multi-account workflows.

## Streaming Workflow (WebSocket via Socket.IO)

1. **Connect** – Open Socket.IO connection to `ws://localhost:3002` (session cookie auto-sent)
2. **Authentication** – Session cookie from SIWS login is automatically used
3. **Listen** – Subscribe to events via Socket.IO event system:
   ```javascript
   socket.on('price_update', (data) => {
     // Handle price updates
     console.log(data);
   });
   ```
4. **Reconnection** – Socket.IO handles automatic reconnection with session validation

## Rate Limits & Resilience

- **Public routes**: 100 requests/minute per IP
- **Trading routes**: 10 requests/minute per wallet
- **Auth routes**: 5 attempts per 15 minutes
- **Admin routes**: 50 requests/minute
- All endpoints return `429` with `X-RateLimit-Remaining` header when limits are exceeded
- Redis caching (optional in dev) keeps responses fast when enabled

## Tooling & Environments

- The `docs/api/QuantDesk-API-Collection.json` Postman collection bundles all endpoints above, with environments in `docs/api/QuantDesk-Complete-Environment.json`.
- Core environment variables to configure:
  - `BACKEND_BASE_URL`, `BACKEND_WS_URL`
  - `SOLANA_RPC_URL`, `PROGRAM_ID`
  - `SUPABASE_URL`, `SUPABASE_ANON_KEY`
- Optional integrations include Pyth (price feeds), Jupiter (quotes), and Mikey AI endpoints.

For deeper architectural context, pair this guide with [Live Market Data Delivery](./live-market-data.md) and [Multi-Account Control](./multi-account-control.md).
