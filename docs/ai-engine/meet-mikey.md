# Meet MIKEY – Your QuantDesk Intelligence Layer

MIKEY is the intelligence engine that turns raw market noise into the insights you see inside QuantDesk. This guide explains what MIKEY watches, how it surfaces information, and the guardrails that keep sensitive logic private.

## 1. What MIKEY Does for You

- **Market Summary Feed** – Condenses price moves, funding changes, and liquidity shifts into plain language updates.
- **Opportunity Spotter** – Flags unusual order flow, funding skews, or cross-market dislocations that might deserve your attention.
- **Risk Monitor** – Keeps an eye on liquidation buffers, leverage levels, and collateral usage, nudging you when it’s time to rebalance.
- **Context Layer** – Merges on-chain events, protocol news, and macro headlines so you don’t miss the story behind the candles.

## 2. Data Sources (High Level)

- **Market Data** – Normalized price, depth, and funding streams from the exchanges and DEX protocols you connect.
- **On-Chain Signals** – Whale transfers, governance events, or spikes in smart contract activity.
- **News & Sentiment** – Curated feeds focused on DeFi and perpetual markets.
- **Portfolio State** – Non-custodial snapshots of your positions, margin, and cash balances for risk awareness.

These pipelines flow through the same ingestion services described in the [perp architecture overview](../trading-capabilities/how-quantdesk-powers-perps.md), so you get live context without juggling multiple dashboards.

## 3. How MIKEY Communicates

- **Insight Cards** – Highlight key takeaways in the dashboard sidebar.
- **Terminal Alerts** – Immediate notifications for funding flips, collateral stress, or protocol incidents.
- **Chat Interface** – Ask clarifying questions ("Why is SOL funding spiking?"), get a concise explanation.
- **API Hooks** – Programmatic access to summaries if you want to forward alerts to Slack, Discord, or custom tooling.

## 4. Safety & Privacy

- **No proprietary prompts or models exposed** – Everything you see is curated output, not the tuning secrets.
- **Strict input filters** – Sensitive environment variables, keys, or private chat content never enter MIKEY’s processing pipeline.
- **Role-aware responses** – MIKEY respects your account permissions; it only references markets and portfolios you can view.
- **Audit logging** – Requests and responses are logged with timestamps to support compliance reviews and debugging.

## 5. Make the Most of MIKEY

- Pin insight cards that align with your trading style (funding, liquidity, momentum, etc.).
- Use the chat interface to double-check a thesis before you size up.
- Let MIKEY watch the boring parts—set alerts for levels or metrics you usually track manually.
- Share relevant summaries with teammates directly from the dashboard when collaborating.

---

MIKEY is your market co-pilot: it keeps a constant pulse on Solana perps while you stay focused on execution. Next, explore the [terminal toolbox](../trading-capabilities/perp-terminal-toolbox.md) to see where MIKEY’s insights surface inside the workspace.
