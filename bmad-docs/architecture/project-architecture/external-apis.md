# External APIs

## Pyth Network API

- **Purpose:** Real-time price feed oracle for perpetual trading markets
- **Documentation:** https://docs.pyth.network/
- **Base URL(s):** https://hermes.pyth.network/v2/updates, Solana RPC endpoints
- **Authentication:** Public API (no authentication required)
- **Rate Limits:** No explicit rate limits, but recommends reasonable request frequency

**Key Endpoints Used:**
- `GET /v2/updates/price/{feed_id}` - Get latest price updates for specific feed
- `GET /v2/updates/price/latest` - Get latest price updates for all feeds
- `WebSocket /v2/updates/price/stream` - Real-time price feed streaming

**Integration Notes:** Primary oracle for all trading pairs, requires staleness validation, supports multiple price feeds simultaneously

## Supabase API

- **Purpose:** Database operations, user authentication, real-time subscriptions
- **Documentation:** https://supabase.com/docs
- **Base URL(s):** Project-specific Supabase URL
- **Authentication:** JWT tokens, API keys for service-to-service communication
- **Rate Limits:** Based on Supabase plan (Free tier: 50,000 requests/month)

**Key Endpoints Used:**
- `POST /auth/v1/token` - User authentication
- `GET /rest/v1/{table}` - Database queries
- `POST /rest/v1/{table}` - Database inserts
- `PATCH /rest/v1/{table}` - Database updates
- `WebSocket /realtime/v1/` - Real-time data subscriptions

**Integration Notes:** Primary database service, handles user sessions, portfolio data, trading history, real-time position updates

## External LLM APIs (OpenAI/Anthropic)

- **Purpose:** AI trading analysis and recommendations via MIKEY AI service
- **Documentation:** OpenAI: https://platform.openai.com/docs, Anthropic: https://docs.anthropic.com/
- **Base URL(s):** https://api.openai.com/v1, https://api.anthropic.com/v1
- **Authentication:** API keys for each provider
- **Rate Limits:** OpenAI: 3,000 RPM, Anthropic: 1,000 RPM (varies by model)

**Key Endpoints Used:**
- `POST /v1/chat/completions` - OpenAI chat completions
- `POST /v1/messages` - Anthropic Claude API
- `POST /v1/embeddings` - Text embeddings for market analysis

**Integration Notes:** Multi-LLM routing via LangChain, trading analysis, market sentiment, risk assessment, automated trading recommendations

## Solana RPC API

- **Purpose:** Blockchain interactions, transaction submission, account queries
- **Documentation:** https://docs.solana.com/developing/clients/jsonrpc-api
- **Base URL(s):** https://api.mainnet-beta.solana.com, https://api.devnet.solana.com
- **Authentication:** Public endpoints (rate limited), private RPC providers available
- **Rate Limits:** Public: 10 requests/second, Private providers: Higher limits

**Key Endpoints Used:**
- `POST /` - JSON-RPC calls for blockchain operations
- `getAccountInfo` - Query account data
- `sendTransaction` - Submit transactions
- `getTransaction` - Retrieve transaction details
- `getProgramAccounts` - Query program accounts

**Integration Notes:** Primary blockchain interface, transaction signing, account management, program interactions for smart contracts

## Vercel API

- **Purpose:** Deployment platform, serverless function hosting, edge network
- **Documentation:** https://vercel.com/docs
- **Base URL(s):** https://api.vercel.com
- **Authentication:** Vercel API tokens
- **Rate Limits:** 100 requests/minute for API calls

**Key Endpoints Used:**
- `GET /v1/projects` - Project management
- `POST /v1/deployments` - Deployment operations
- `GET /v1/domains` - Domain management

**Integration Notes:** Deployment platform, serverless hosting, edge network optimization, automatic scaling
