# API Access & Integration

QuantDesk exposes REST and WebSocket APIs so partners and advanced traders can plug the terminal’s data and trading workflows into their own systems.

## Base URLs

| Environment | REST Base | WebSocket Base |
|-------------|-----------|----------------|
| Local Dev | `http://localhost:3002` | `ws://localhost:3002` |
| Staging/Prod | `https://api.quantdesk.app` | `wss://api.quantdesk.app` |

Swap in the environment that matches your deployment; all examples below use the local defaults.

## Authentication

- **REST endpoints** require a session token issued after Sign-In with Solana (SIWS). Include it as `Authorization: Bearer <token>`.
- **WebSocket connections** accept the same bearer token via query string or initial auth message.
- The auth service rotates tokens; expect to refresh them according to the session TTL (default 7 days for trusted clients).

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
GET  /api/account/state               # Wallet + account status
POST /api/accounts/trading-accounts   # Create sub-account
POST /api/deposits/deposit            # Initiate deposit flow
POST /api/orders                      # Place order (market/limit/stop)
POST /api/orders/:id/cancel           # Cancel order
GET  /api/positions                   # Current positions + P&L
```

- Order endpoints expect canonical market IDs and size/price fields. Advanced order types (stop-loss, bracket) use dedicated payloads under `/api/advanced-orders`.
- Cross-account transfers and delegated access live under `/api/accounts/*` routes for multi-account workflows.

## Streaming Workflow

1. Open a WebSocket connection: `ws://localhost:3002/ws?token=<JWT>`
2. Subscribe to channels:
   ```json
   {"type":"subscribe","channel":"price","symbol":"SOL-PERP"}
   {"type":"subscribe","channel":"orderbook","symbol":"SOL-PERP"}
   ```
3. Handle messages shaped like:
   ```json
   {
     "v":1,
     "type":"price",
     "symbol":"SOL-PERP",
     "ts":1730930000000,
     "data":{"price":55.12,"confidence":0.02,"slot":211883040}
   }
   ```
4. If the stream drops, fall back to REST polling until the socket reconnects.

## Rate Limits & Resilience

- Public market routes enforce per-IP limits and return `429` with a `Retry-After` header when exceeded.
- Authenticated trading routes apply stricter per-wallet limits.
- The backend caches price/orderbook snapshots in Redis (1–3 second TTL) to keep responses fast and consistent across clients.

## Tooling & Environments

- The `docs/api/QuantDesk-API-Collection.json` Postman collection bundles all endpoints above, with environments in `docs/api/QuantDesk-Complete-Environment.json`.
- Core environment variables to configure:
  - `BACKEND_BASE_URL`, `BACKEND_WS_URL`
  - `SOLANA_RPC_URL`, `PROGRAM_ID`
  - `SUPABASE_URL`, `SUPABASE_ANON_KEY`
- Optional integrations include Pyth (price feeds), Jupiter (quotes), and Mikey AI endpoints.

For deeper architectural context, pair this guide with [Live Market Data Delivery](./live-market-data.md) and [Multi-Account Control](./multi-account-control.md).
