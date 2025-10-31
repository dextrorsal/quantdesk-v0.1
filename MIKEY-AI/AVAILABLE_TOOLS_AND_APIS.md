# MIKEY AI - Available Tools, APIs, and Integrations

## 📊 Summary

MIKEY has **NO MCPs configured** - all integrations use direct HTTP API calls to:
1. **QuantDesk Backend** (localhost:3002) - Your main API
2. **External APIs** - Pyth, CoinGecko, Drift, Jupiter, etc.
3. **LLM Providers** - OpenAI, Google Gemini, Mistral, Cohere, XAI, Qwen

---

## 🔌 Backend API Tools (QuantDesk - localhost:3002)

### Real Data Tools (`RealDataTools.ts`)
| Tool Name | Endpoint | What It Does |
|-----------|----------|--------------|
| `get_pyth_prices` | `GET /api/oracle/prices` | Get real-time Pyth prices (BTC, ETH, SOL, etc.) |
| `get_coingecko_prices` | `GET /api/coingecko/prices` | Get CoinGecko market prices |
| `get_whale_data` | `GET /api/v1/trading/whales?threshold=100000&timeframe=24h` | Get large transactions/whale movements |
| `get_crypto_news` | `GET /api/v1/market/sentiment?symbol=SOL` | Get crypto news and sentiment |
| `get_market_analysis` | `GET /api/real-supabase-markets` | Get market data (TVL, volume, trends) |
| `get_arbitrage_opportunities` | `GET /api/arbitrage/opportunities` | Find cross-exchange arbitrage |

### Real Token Analysis Tool (`RealTokenAnalysisTool.ts`)
| Tool Name | Endpoints Called | What It Does |
|-----------|------------------|--------------|
| `analyze_token_real_data` | • `GET /api/oracle/price/:asset`<br>• `GET /api/oracle/prices`<br>• `GET /api/dev/market-summary` | **Comprehensive token analysis** using real backend data:<br>- Pyth prices<br>- Market summary (volume, OI, leverage)<br>- Calculates indicators (RSI, support/resistance)<br>- Provides trading recommendations |

### QuantDesk Protocol Tools (`QuantDeskProtocolTools.ts`)
| Tool Name | Endpoint | What It Does |
|-----------|----------|--------------|
| `check_quantdesk_portfolio` | `GET /api/dev/user-portfolio/:wallet` | Check user portfolio balances and health |
| `get_quantdesk_market_data` | `GET /api/dev/market-summary` | Get market summary (prices, volume, OI, leverage) |
| `get_live_price` | `GET /api/oracle/price/:asset` | Get single asset price from Pyth oracle |
| `place_quantdesk_trade` | `POST /api/orders` | Place a trade via QuantDesk backend |
| `get_quantdesk_protocol_health` | `GET /api/metrics`, `GET /api/status` | Check protocol health and metrics |
| `analyze_wallet_risk` | `GET /api/portfolio/:wallet`, `GET /api/positions/:wallet` | Analyze wallet risk (portfolio + positions) |

### QuantDesk Tools (`QuantDeskTools.ts`)
| Tool Name | Endpoint | What It Does |
|-----------|----------|--------------|
| `get_markets` | `GET /api/markets` | Get available trading markets |
| `get_prices` | `GET /api/prices` | Get current prices for trading pairs |
| `get_account` | `GET /api/account/:userId` | Get account information (requires auth) |
| `get_trading_data` | `GET /api/orders`, `GET /api/positions` | Get orders and positions (requires auth) |
| `check_health` | `GET /api/health` | Check QuantDesk API health |

### QuantDesk Trading Tools (`QuantDeskTradingTools.ts`)
| Tool Name | Endpoint | What It Does |
|-----------|----------|--------------|
| `get_account_state` | `GET /api/account-state/:userId` | Get account state, balances, permissions |
| `get_positions` | `GET /api/positions` | Get all open positions for user |
| `get_orders` | `GET /api/orders` | Get all orders (open, filled, cancelled) |
| `place_order` | `POST /api/orders` | Place a new order |
| `cancel_order` | `DELETE /api/orders/:orderId` | Cancel an order |
| `get_funding_rates` | `GET /api/funding-rates` | Get current funding rates |
| `get_liquidation_risk` | `GET /api/liquidation-risk/:userId` | Calculate liquidation risk |

### Supabase Tools (`SupabaseTools.ts`)
| Tool Name | What It Does |
|-----------|--------------|
| `query_supabase` | Direct Supabase database queries (if configured) |

---

## 🌐 External API Integrations

### DeFi Protocols (`EnhancedDeFiTools.ts`)
| Protocol | API Base | What It Does |
|----------|----------|--------------|
| **Drift Protocol** | `https://app.drift.trade/api/perps/market/:market` | Get Drift perpetual market data |
| **Jupiter** | `https://quote-api.jup.ag` | Token swaps, price quotes |
| **Raydium** | N/A | DEX liquidity and trading |
| **Mango Markets** | N/A | Perpetual trading platform |
| **Hyperliquid** | `https://api.hyperliquid.xyz` | Perpetual futures platform |
| **Axiom** | `https://api.axiom.xyz` | Perpetual DEX |
| **Asterdex** | `https://api.asterdex.io` | Perpetual DEX |

### Data Sources (`DataSources.ts`)
| Service | What It Provides |
|---------|------------------|
| **Drift Protocol API** | Market data, positions, orders |
| **Jupiter API** | Token prices, swap routes |
| **Hyperliquid API** | Perpetual futures data |
| **Axiom API** | Perpetual DEX data |
| **Asterdex API** | Perpetual DEX data |

### CCXT Service (`CCXTService.ts`)
| What It Does | Supported Exchanges |
|--------------|---------------------|
| CEX market data (prices, order books, funding rates, liquidations, OI) | Binance, Bybit, OKX, Kraken, Coinbase, and others |

---

## 🤖 LLM Providers (OfficialLLMRouter.ts)

MIKEY uses multiple LLM providers for generating responses:

| Provider | Model | Status |
|----------|-------|--------|
| **OpenAI** | `gpt-4o-mini` | ✅ Configured (requires `OPENAI_API_KEY`) |
| **Google Gemini** | `gemini-2.5-flash` | ✅ Configured (requires `GOOGLE_API_KEY`) |
| **Cohere** | `command-a-03-2025` | ✅ Configured (requires `COHERE_API_KEY`) |
| **Mistral AI** | `mistral-small-latest` | ✅ Configured (requires `MISTRAL_API_KEY`) |
| **XAI (Grok)** | `grok-beta` | ✅ Configured (requires `XAI_API_KEY`) |
| **Qwen (Alibaba)** | `qwen-*` | ✅ Configured (requires `QWEN_API_KEY`) |
| **Hugging Face** | Various | ✅ Configured (requires `HUGGINGFACE_API_KEY`) |

**Note:** LLM providers are used to **generate human-readable responses** from tool data, not as data sources themselves.

---

## 🎭 Demo Mock Tools (`DemoMockTools.ts`)

| Tool Name | What It Does |
|-----------|--------------|
| `analyze_token_market` | Returns **mock data** for token analysis (for demo) |
| `open_position_mock` | Returns **mock transaction** for position opening (for demo) |

**Usage:** Only used if query explicitly includes "mock" or "demo", OR for position opening requests.

---

## ❌ What's NOT Available

### MCPs (Model Context Protocol)
- **No MCPs configured** - MIKEY does not use any MCP servers
- All integrations are direct HTTP API calls
- MCPs mentioned in docs refer to "Postman MCP" (just a tool name), not actual MCP protocol

### Direct Solana RPC
- MIKEY does NOT directly call Solana RPC
- All Solana data comes through backend APIs (`/api/dev/user-portfolio/:wallet`, etc.)

---

## 🔄 Tool Decision Flow

1. **Real Data Tools** (priority) - Uses `/api/oracle/prices`, `/api/dev/market-summary`
2. **QuantDesk Protocol Tools** - Uses `/api/portfolio`, `/api/positions`, `/api/orders`
3. **Demo Mock Tools** - Only if "mock" or "demo" in query

---

## 🚀 How to Add New Tools

1. **Add to service file** (e.g., `RealDataTools.ts`)
2. **Register in `TradingAgent.ts`** → `initializeTools()` method
3. **Add routing logic** → `needsRealData()`, `processWithRealDataTools()`, etc.

---

## 📝 Example Query Flow

**Query:** "Analyze BTC market sentiment and show me TVL"

1. ✅ `needsRealData()` detects "analyze", "sentiment", "tvl"
2. ✅ Calls `RealTokenAnalysisTool.createRealTokenAnalysisTool()`
3. ✅ Fetches:
   - `GET /api/oracle/price/BTC` → Real Pyth price
   - `GET /api/dev/market-summary` → Market data
4. ✅ Formats response
5. ✅ Sends to LLM (Google/Mistral/etc.) → Generates human response

---

**Last Updated:** 2025-01-31
**Total Tools Available:** ~30+ tools across 8+ service categories

