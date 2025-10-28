# Start QuantDesk in 5 Minutes

🚀 **Spin up the QuantDesk perpetual DEX terminal locally, connect to live data, and explore the tooling that powers your trading decisions.**

## 🎯 What You'll Have in Five Minutes

- ✅ The full QuantDesk terminal running on your machine
- ✅ Live perpetual market data streaming into the dashboard
- ✅ Wallet-based authentication with Solana wallets
- ✅ On-chain trading account ready to fund
- ✅ MIKEY insights highlighting notable market structure changes

## ⚡ Quick Setup Checklist

### 1. Clone the Repository

```bash
git clone https://github.com/dextrorsal/quantdesk.git
cd quantdesk
```

### 2. Install Project Dependencies

```bash
pnpm install
```

This script installs dependencies for the frontend, backend, and supporting services in one pass.

### 3. Configure Environment Variables

```bash
# From project root
cp env.template backend/.env
```

Inside `backend/.env` you'll find sections for:
- **Supabase credentials** – database connection for accounts, positions, and market data
- **Solana RPC** – connection to devnet/mainnet (defaults to public RPC)
- **Program IDs** – smart contract addresses for QuantDesk programs
- **JWT Secret** – session token encryption key
- **Market data caching** – low-latency market data updates (Redis optional in development)

> 💡 Tip: The platform is fully non-custodial - you connect with your wallet and maintain control of all funds.

### 4. Launch the Platform

```bash
pnpm run dev
```

This boots every service together:
- **Frontend (Port 3001)** → Trading terminal with wallet connection
- **Backend API (Port 3002)** → REST + WebSocket gateway with authentication
- **MIKEY-AI (Port 3000)** → AI trading assistant and market analysis
- **Data Ingestion (Port 3003)** → Real-time market data pipeline

### 5. Connect Your Wallet

Visit **http://localhost:3001** to access the trading workspace. 

**First-time users will see:**
1. **Wallet Connection Prompt** – Connect Phantom, Solflare, or any Solana wallet
2. **SIWS Authentication** – Sign a message to verify wallet ownership (non-custodial)
3. **Session Created** – HTTP-only cookie established (7-day expiration)
4. **Account Detection** – System checks for existing on-chain trading account

If you're new, you'll create an on-chain trading account before funding and trading.

## 🌐 What You Get Out of the Box

### 📊 Real-Time Market Surfaces
- Perpetual order books across Solana DEX venues
- Funding, open interest, and liquidity metrics at a glance
- Configurable watchlists for assets, pools, and accounts

### 🔎 MIKEY Insight Panels
- Highlights unusual funding shifts, liquidity imbalances, and velocity changes
- Surfaced explanations designed for traders—not data scientists
- Breakdowns stay high level, keeping proprietary modelling details private

### 🛠️ Execution & Risk Tools
- Structured order tickets with slippage, margin, and collateral visibility
- Status panel showing backend health, WebSocket uptime, and RPC latency
- Position overview synced with your connected wallets/exchanges

## 🧪 Explore Without Risk First

1. **Connect to Devnet** – Default Solana network is devnet (test tokens)
2. **Use Test Tokens** – Request devnet SOL from faucet for testing
3. **Create Test Account** – On-chain account creation on devnet (free)
4. **Try Features Safely** – All features work on devnet with no real funds at risk
5. **Review Audit Logs** – Confirm every API call and event is tracked for accountability

> 🔒 **Security**: Your wallet private key never leaves your browser. All signatures are client-side.

## ❓ Quick FAQ

**Do I need to code?**  
No. Everything runs from prebuilt scripts. Optional API endpoints are documented for teams who want deeper integrations.

**How do I authenticate?**  
Connect your Solana wallet (Phantom, Solflare, etc.) and sign a message. No passwords or API keys needed.

**Where do my wallet credentials live?**  
Your wallet private key never leaves your browser. The backend never stores or has access to your keys—all signatures are client-side.

**What wallets work?**  
Any Solana wallet with message signing support: Phantom, Solflare, Trust Wallet, Solong, etc.

**Can I control which services run?**  
Yes. Each service can be started individually:
- `cd frontend && pnpm run dev` - Frontend only
- `cd backend && pnpm run start:dev` - Backend only
- `cd MIKEY-AI && pnpm run dev` - AI service only
- `cd data-ingestion && pnpm start` - Data pipeline only

**What about mainnet?**  
Change `SOLANA_NETWORK=mainnet` in `backend/.env` and update `RPC_URL` to a mainnet endpoint. Production requires proper setup and funding.

## 📈 After the First Boot

### Next Steps:

1. **Explore the Terminal** – Browse available markets, check funding rates, review order books
2. **Fund Your Account** – Deposit assets to start trading (devnet test tokens available)
3. **Check Health Dashboard** – Monitor service status at backend health endpoints
4. **Inspect Data** – Check Supabase tables to see how analytics data is structured
5. **Connect MIKEY** – Interact with AI assistant for market analysis and insights
6. **Join the Community** – Access chat channels for trader discussions

### View Logs:
```bash
# Backend logs
cd backend && tail -f logs/quantdesk.log

# Frontend console
# Open browser DevTools to see client-side logs

# Check service health
curl http://localhost:3002/health
```

---

**Ready to dive deeper?**  
[**Account Lifecycle Guide**](../core-features/account-lifecycle.md) | [**Trading Capabilities**](../trading-capabilities/) | [**API Reference**](../api/)
