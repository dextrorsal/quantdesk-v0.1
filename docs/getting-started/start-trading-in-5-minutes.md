# Start QuantDesk in 5 Minutes

🚀 **Spin up the QuantDesk perpetual DEX terminal locally, connect to live data, and explore the tooling that powers your trading decisions.**

## 🎯 What You’ll Have in Five Minutes

- ✅ The full QuantDesk terminal running on your machine
- ✅ Live perpetual market data streaming into the dashboard
- ✅ Secure API connections ready for your personal exchange keys
- ✅ MIKEY insights highlighting notable market structure changes

## ⚡ Quick Setup Checklist

### 1. Clone the Repository

```bash
git clone https://github.com/dextrorsal/quantdesk.git
cd quantdesk
```

### 2. Install Project Dependencies

```bash
npm run install:all
```

This script installs dependencies for the frontend, backend, and supporting services in one pass.

### 3. Configure Environment Variables

```bash
cp .env.example .env
```

Inside `.env` you’ll find sections for:
- **Exchange connectivity** – supply read-only keys while you validate the workflow
- **Supabase credentials** – defaults work for local development; production keys stay outside the repo
- **Redis connection string** – used for low-latency market caching

> 💡 Tip: start with sandbox keys from your preferred exchange. The terminal is fully non-custodial, so funds stay in your accounts.

### 4. Launch the Platform

```bash
npm run dev
```

This boots every service together:
- **Frontend terminal** → http://localhost:3001
- **MIKEY workspace** → http://localhost:3000
- **API + WebSocket backend** → http://localhost:3002
- **Data ingestion workers** → stream prices, depth, funding, and alerts

### 5. Open the Terminal

Visit **http://localhost:3001** to access the trading workspace. You’ll see live market tiles populating as soon as data streams connect.

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

1. **Run in observation mode** – load live data without enabling order routing
2. **Use testnet keys** – validate flows against Solana devnet or exchange sandboxes
3. **Review audit logs** – confirm every API call and event is tracked for accountability
4. **Switch on trading routes** – once comfortable, enable execution for specific venues

## ❓ Quick FAQ

**Do I need to code?**  
No. Everything runs from prebuilt scripts. Optional API endpoints are documented for teams who want deeper integrations.

**Where do my keys live?**  
Securely in your `.env` or connected wallet. The backend never persists secrets—requests are signed client-side.

**Can I control which services run?**  
Yes. Each workspace (frontend, backend, ingestion) can be started individually via `npm run dev:*` scripts described in `package.json`.

**Is there a hosted option?**  
We support cloud deployments; the local setup mirrors production components so stakeholders can validate architecture before connecting to shared infrastructure.

## 📈 After the First Boot

- Tail logs in `logs/` to observe market pipelines
- Inspect Supabase tables to understand how we structure analytics data
- Connect MIKEY’s alert channel to your preferred notifications provider
- Share the terminal with stakeholders to demonstrate stability and UX polish

---

[**See Core Terminal Capabilities**](../core-features/) | [**Dive Into Trading Operations**](../trading-capabilities/) | [**Review Security & Trust**](../security-trust/)
