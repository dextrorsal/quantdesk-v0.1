QuantDesk Postman Guide (Envs, APIs, Requests)

1) Environments (variables)

- Core (dev)
  - BACKEND_BASE_URL = http://localhost:3002
  - BACKEND_WS_URL = ws://localhost:3002
  - SOLANA_RPC_URL = https://api.devnet.solana.com
  - SOLANA_WS_URL = wss://api.devnet.solana.com
  - PROGRAM_ID = GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a
  - IDL_HOST_URL = (optional) https://cdn.quantdesk.com/idl/quantdesk_perp_dex.json
  - SUPABASE_URL = https://<project>.supabase.co
  - SUPABASE_REST_URL = https://<project>.supabase.co/rest/v1
  - SUPABASE_ANON_KEY = (secret)
  - SUPABASE_SERVICE_ROLE_KEY = (server-only secret)
  - PYTH_BASE_URL = https://hermes.pyth.network
  - PYTH_WS_URL = wss://hermes.pyth.network/ws
  - JUPITER_TOKENS_URL = https://tokens.jup.ag/tokens
  - JUPITER_QUOTE_URL = https://quote-api.jup.ag/v6/quote
  - WALLET_PUBKEY = <your_dev_wallet>
  - GRAFANA_URL = http://localhost:3100

- RPC providers (optional / LB)
  - HELIUS_RPC_URL = https://devnet.helius-rpc.com
  - QUICKNODE_RPC_URL = https://solana-devnet.g.alchemy.com/v2/demo
  - ALCHEMY_RPC_URL = https://solana-devnet.g.alchemy.com/v2/demo
  - SYNDICA_RPC_URL = https://solana-api.syndica.io/access-token/demo
  - CHAINSTACK_RPC_URL = https://solana-devnet.core.chainstack.com/demo

- Social/notifications (optional)
  - TWITTER_API_BASE = https://api.twitter.com/2
  - TELEGRAM_BOT_BASE = https://api.telegram.org/bot<token>/
  - DISCORD_API_BASE = https://discord.com/api

- AI/ML (Mikey AI)
  - AI_BASE_URL = https://ai.quantdesk.com
  - OPENAI_API_BASE = https://api.openai.com/v1
  - ANTHROPIC_API_BASE = https://api.anthropic.com/v1
  - GOOGLE_AI_BASE = https://generativelanguage.googleapis.com

2) Backend API endpoints (Express)

- Health
  - GET {{BACKEND_BASE_URL}}/health

- Auth
  - POST {{BACKEND_BASE_URL}}/api/auth/authenticate
  - POST {{BACKEND_BASE_URL}}/api/auth/refresh
  - GET  {{BACKEND_BASE_URL}}/api/auth/profile

- Account state
  - GET {{BACKEND_BASE_URL}}/api/account/state
  - GET {{BACKEND_BASE_URL}}/api/account/summary
  - POST {{BACKEND_BASE_URL}}/api/account/can-perform
  - GET {{BACKEND_BASE_URL}}/api/account/trading-accounts
  - GET {{BACKEND_BASE_URL}}/api/account/balances
  - GET {{BACKEND_BASE_URL}}/api/account/health
  - GET {{BACKEND_BASE_URL}}/api/account/trading-accounts/:accountId/state

- Accounts (management)
  - POST {{BACKEND_BASE_URL}}/api/accounts/verify-creation
  - GET  {{BACKEND_BASE_URL}}/api/accounts/trading-accounts
  - POST {{BACKEND_BASE_URL}}/api/accounts/trading-accounts
  - PUT  {{BACKEND_BASE_URL}}/api/accounts/sub-accounts/:id
  - DELETE {{BACKEND_BASE_URL}}/api/accounts/sub-accounts/:id
  - GET  {{BACKEND_BASE_URL}}/api/accounts/delegates
  - POST {{BACKEND_BASE_URL}}/api/accounts/delegates
  - PUT  {{BACKEND_BASE_URL}}/api/accounts/delegates/:id
  - DELETE {{BACKEND_BASE_URL}}/api/accounts/delegates/:id
  - GET  {{BACKEND_BASE_URL}}/api/accounts/balances
  - POST {{BACKEND_BASE_URL}}/api/accounts/transfer

- Deposits / Withdrawals
  - GET  {{BACKEND_BASE_URL}}/api/deposits/balances
  - POST {{BACKEND_BASE_URL}}/api/deposits/deposit
  - POST {{BACKEND_BASE_URL}}/api/deposits/deposit/confirm
  - POST {{BACKEND_BASE_URL}}/api/deposits/withdraw
  - POST {{BACKEND_BASE_URL}}/api/deposits/withdraw/confirm
  - GET  {{BACKEND_BASE_URL}}/api/deposits/history

- Orders / Trades
  - GET  {{BACKEND_BASE_URL}}/api/orders
  - GET  {{BACKEND_BASE_URL}}/api/orders/:id
  - POST {{BACKEND_BASE_URL}}/api/orders
  - POST {{BACKEND_BASE_URL}}/api/orders/:id/cancel
  - GET  {{BACKEND_BASE_URL}}/api/trades

- Markets
  - GET {{BACKEND_BASE_URL}}/api/real-supabase-markets
  - GET {{BACKEND_BASE_URL}}/api/real-supabase-markets/:symbol
  - GET {{BACKEND_BASE_URL}}/api/real-supabase-markets/:symbol/price
  - GET {{BACKEND_BASE_URL}}/api/real-supabase-markets/:symbol/price-history
  - GET {{BACKEND_BASE_URL}}/api/real-supabase-markets/:symbol/funding
  - GET {{BACKEND_BASE_URL}}/api/supabase-oracle/markets
  - GET {{BACKEND_BASE_URL}}/api/supabase-oracle/prices
  - GET {{BACKEND_BASE_URL}}/api/prices (simple Pyth proxy)

- RPC health & testing
  - POST {{BACKEND_BASE_URL}}/api/rpc-testing/test
  - POST {{BACKEND_BASE_URL}}/api/rpc-testing/stress-test
  - GET  {{BACKEND_BASE_URL}}/api/rpc-testing/metrics
  - GET  {{BACKEND_BASE_URL}}/api/rpc-stats/stats
  - GET  {{BACKEND_BASE_URL}}/api/rpc-stats/health
  - GET  {{BACKEND_BASE_URL}}/api/rpc-stats/providers

- Admin
  - POST {{BACKEND_BASE_URL}}/api/admin/login
  - GET  {{BACKEND_BASE_URL}}/api/admin/verify
  - GET  {{BACKEND_BASE_URL}}/api/admin/users
  - POST {{BACKEND_BASE_URL}}/api/admin/users
  - PUT  {{BACKEND_BASE_URL}}/api/admin/users/:id
  - DELETE {{BACKEND_BASE_URL}}/api/admin/users/:id
  - GET  {{BACKEND_BASE_URL}}/api/admin/audit-logs
  - GET  {{BACKEND_BASE_URL}}/api/admin/mode
  - POST {{BACKEND_BASE_URL}}/api/admin/mode
  - GET  {{BACKEND_BASE_URL}}/api/admin/stats
  - GET  {{BACKEND_BASE_URL}}/api/admin/health
  - GET  {{BACKEND_BASE_URL}}/api/admin/metrics
  - GET  {{BACKEND_BASE_URL}}/api/admin/logs
  - POST {{BACKEND_BASE_URL}}/api/admin/emergency-stop

3) Solana JSON-RPC (cluster)

- POST {{SOLANA_RPC_URL}}
  - getLatestBlockhash
  - getBalance (params: ["{{WALLET_PUBKEY}}", {"commitment":"confirmed"}])
  - getProgramAccounts (params: ["{{PROGRAM_ID}}", {"encoding":"base64"}])
  - sendTransaction (raw tx base64)

4) Pyth Network

- REST
  - GET {{PYTH_BASE_URL}}/v2/updates/price/latest?ids[]=<PYTH_PRICE_ID>
- WebSocket
  - {{PYTH_WS_URL}}

5) Jupiter

- Tokens
  - GET {{JUPITER_TOKENS_URL}}
- Quotes
  - GET {{JUPITER_QUOTE_URL}}?inputMint=<MINT>&outputMint=<MINT>&amount=1000000&slippageBps=50

6) Supabase REST

- Base
  - {{SUPABASE_REST_URL}}
- Example
  - GET {{SUPABASE_REST_URL}}/profiles?select=*&limit=10
    - Headers: apikey={{SUPABASE_ANON_KEY}}, Authorization=Bearer {{SUPABASE_ANON_KEY}}

7) Social (optional)

- X/Twitter
  - {{TWITTER_API_BASE}}
- Telegram Bot
  - {{TELEGRAM_BOT_BASE}}sendMessage
- Discord
  - {{DISCORD_API_BASE}}

8) AI/ML (Mikey AI + providers)

- Mikey AI gateway
  - {{AI_BASE_URL}}/infer, /embed, /classify (custom)
- OpenAI / Anthropic / Google
  - {{OPENAI_API_BASE}}, {{ANTHROPIC_API_BASE}}, {{GOOGLE_AI_BASE}}

9) Tips (from Solana Cookbook)

- Use devnet/testnet/localnet for testing; airdrop SOL when needed.
- For token balances, use getParsedTokenAccountsByOwner (programId: TOKEN_PROGRAM_ID) or filter by mint.
- Native SOL vs SPL: native uses SystemProgram.transfer; tokens use SPL Token Program.

10) Postman collection structure (recommended)

- Folders: Backend, Solana RPC, Supabase, Pyth, Jupiter, Admin, RPC Tools, AI, Social
- All requests parametrize base URLs with {{VAR}} from environments.


### Base URLs and where they come from (with file refs)
- Backend API
  - Default: http://localhost:3002
  - Source: `backend/src/server.ts` (PORT env) and `backend/Dockerfile` (PORT=3002)
- Solana RPC
  - Backend: `process.env.SOLANA_RPC_URL || https://api.devnet.solana.com`
    - Source: `backend/src/services/transactionVerificationService.ts`
  - Frontend: `process.env.REACT_APP_SOLANA_RPC_URL || https://api.devnet.solana.com`
    - Source: `frontend/src/services/smartContractService.ts`, `frontend/src/services/balanceService.ts`
  - RPC Load Balancer (optional providers): `backend/src/services/rpcLoadBalancer.ts`
- Program ID
  - `process.env.QUANTDESK_PROGRAM_ID` (backend routes reference)
    - Source: `backend/src/routes/deposits.ts`, `backend/src/routes/accounts.ts`
  - Frontend uses IDL address (`contracts/smart-contracts/target/idl/quantdesk_perp_dex.json`)
- Supabase
  - Backend: `SUPABASE_URL`, `SUPABASE_ANON_KEY`, `SUPABASE_SERVICE_ROLE_KEY`
    - Source: `backend/src/services/supabaseService.ts`, `backend/src/config/environment.ts`, `env.example`
  - Frontend: `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`
    - Source: `frontend/src/utils/supabase.ts`
- Pyth Network
  - REST: `https://hermes.pyth.network`
  - WS: `wss://hermes.pyth.network/ws`
  - Source: `backend/src/services/pythOracleService.ts`
- Jupiter
  - Token list: `https://tokens.jup.ag/tokens`
  - Quote: `https://quote-api.jup.ag/v6/quote`
  - Source: used client-side in token logic (config/services)
- Backend feature endpoints (for Postman collection)
  - Accounts: `backend/src/routes/accounts.ts`, `backend/src/routes/accountState.ts`
  - Deposits/Withdrawals: `backend/src/routes/deposits.ts`
  - Orders/Trades: `backend/src/routes/orders.ts`, `backend/src/routes/trades.ts`
  - Markets: `backend/src/routes/realSupabaseMarkets.ts`, `backend/src/routes/supabaseMarkets.ts`
  - Auth: `backend/src/routes/auth.ts`
  - Admin: `backend/src/routes/admin.ts`
  - RPC test/health: `backend/src/routes/rpcTesting.ts`, `backend/src/routes/rpcStats.ts`

### Postman environments (suggested variables)
- Common
  - BACKEND_BASE_URL = http://localhost:3002
  - SOLANA_RPC_URL = https://api.devnet.solana.com
  - PROGRAM_ID = GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a
  - SUPABASE_REST_URL = https://your-project.supabase.co
  - SUPABASE_ANON_KEY = (secret)
  - PYTH_BASE_URL = https://hermes.pyth.network
  - PYTH_WS_URL = wss://hermes.pyth.network/ws
  - JUPITER_BASE_URL = https://quote-api.jup.ag
  - WALLET_PUBKEY = (your test pubkey)

### I can expand your `postman-doc.md` with this API inventory
- Crypto/on-chain: Solana RPC, Program ID, Pyth REST/WS, Jupiter tokens/quote
- Backend: Accounts, Deposits/Withdrawals, Orders/Trades, Markets, Auth, Admin, RPC stats
- Data: Supabase REST (profiles, balances)
- Social (future): X/Twitter, Telegram Bot API, Discord
- ML/AI (Mikey AI): add `AI_BASE_URL` if you have a gateway; plus OpenAI/Anthropic/Google endpoints if used

Want me to:
- create the Postman environments with these variables, and
- generate a collection with all the key requests pre-wired to `{{VAR}}`?