# Market Intelligence Pipeline

Behind every MIKEY insight sits a real-time data pipeline that ingests, scores, and correlates market signals. Here’s what’s already running and what’s being layered in next.

## Data Streams Online Today

- **Price Collector** – Pyth oracle feeds for BTC, ETH, SOL plus CoinPaprika backups, refreshed every few seconds with confidence intervals.
- **News Scraper** – CoinDesk, CoinTelegraph, and The Block headlines parsed via RSS with sentiment scoring and keyword tagging.
- **Whale Monitor** – Solana RPC watchers capturing six-figure transactions and routing them into `whales.raw` for MIKEY analysis.
- **Trench Watcher** – Birdeye/GMGN integrations surfacing brand-new Solana tokens, early liquidity, and holder concentration.
- **DeFi Analytics** – Artemis and DeFiLlama metrics (TVL, fees, user activity) streamed into `defi.raw` for protocol health context.
- **Perps Analytics** – Funding, OI, liquidation, and long/short data from Coinglass/Coinalyze to situate perp opportunities.
- **Custom Analytics** – Dune queries and CoinPaprika market stats populate broader market trend tiles.

All streams land in Redis, giving the terminal and MIKEY sub-second access to the latest events.

## Analytics Writer (Rolling Out)

- Normalizes the eight streams into cross-market correlations (whales vs. price, sentiment vs. funding, etc.).
- Generates MIKEY-ready signals (opportunity, risk, alerts) that flow into the terminal’s insight cards and alert system.
- Outputs daily rollups into Supabase tables for historical analysis and roadmap reporting.

## Smart Money Strategy Layers

- **Whale Watcher** – Focused on perp/established markets, tying large position shifts to funding anomalies and exchange flows.
- **Trench Watcher** – Targets new DeFi launches, measuring liquidity quality, holder dispersion, and early momentum.
- Priority integrations include CoinGecko, Birdeye, GMGN (live) with CryptoQuant, Arkham, and Coinglass staged for enrichment.

## What Traders See

- MIKEY highlights unusual whale activity, funding flips, or DeFi stress sourced directly from these pipelines.
- Smart Money tiles and alerts in the roadmap reflect real ingestion status, not mock data.
- Planned upgrades (alert routing, unified dashboard, ML feature tagging) will build on the same infrastructure—no replatforming required.

QuantDesk’s intelligence layer is already operational; ongoing work focuses on richer correlations and delivery channels so traders act on the right signal at the right moment.
