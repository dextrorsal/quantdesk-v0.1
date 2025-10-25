# QuantDesk Terminal Toolbox

QuantDesk packs the essentials a perpetual trader needs into a single workspace. Use this guide to understand what each area of the terminal delivers and how it supports your daily workflow.

## 1. Market Surfaces

### Real-Time Perp Boards (`QM`, `FLOW`)
- **Live tiles** display price, change, funding, and open interest for core markets (BTC, ETH, SOL, and rotating pairs).
- **Depth view** shows top-of-book liquidity and laddered order book snapshots so you can size orders with confidence.
- **Funding tracker** highlights when payments flip positive/negative, signalling potential basis trades.
- **Smart money stream** (`FLOW`) surfaces whale activity alongside pricing so you can correlate positioning with tape action.

### News & Sentiment Stream (`NEWS`, `DEFI`)
- Curated crypto headlines (`NEWS`, `CN`, `CT`, `TB`) and on-chain cues surface beside the market tiles.
- DeFi dashboards (`DEFI`) highlight protocol stress or TVL spikes that could influence funding and volatility.
- Alerts flag protocol incidents, governance votes, or volatility spikes that could impact perp pricing.

## 2. Execution Desk

### Order Ticket (`ORDER`, `Ctrl + B`, `Ctrl + S`)
- Supports market, limit, stop, and bracket orders out of the box.
- Slippage controls, trigger conditions, and time-in-force options are one click away.
- Displays projected margin usage and liquidation buffer before you submit.

### Position Controls (`POSITIONS`, `Ctrl + C`)
- Snapshot of current holdings with entry price, unrealized P&L, and effective leverage.
- Close, reduce, or flip positions directly from the panel without hunting through history.
- Add take-profit/stop-loss instructions after entry if you want to tighten risk later.

## 3. Risk Dashboard

### Account Health
- Margin bar illustrates how close you are to maintenance thresholds.
- Collateral breakdown lists every asset supporting your positions.
- Funding projections show upcoming payments so you can rebalance ahead of time.

### Alerts & Safeguards (`ALERT`)
- Customizable warnings for drawdown, leverage, or unusual funding activity.
- MIKEY-generated notes explain why an alert triggered—useful for quick decisions during volatility.
- Configure alert levels directly from the command palette (`ALERT`) or inline buttons when metrics exceed thresholds.

## 4. Analytics & Performance

### Trade Journal (`PF`, `NOTE`)
- Automatic logging of entries, exits, and adjustments with timestamps.
- Tag trades by strategy or market to analyze performance later.
- Keep rich-text notes (`NOTE`) directly attached to market sessions.

### Charts & Indicators (`CHART`)
- Multi-timeframe view (1m to 1d) with popular indicators ready to apply.
- Save layouts per asset so your preferred setup loads immediately next session.

### Portfolio Metrics (`PF`)
- Daily and cumulative P&L curves.
- Risk-adjusted stats (Sharpe-style metrics) generated from your actual trading data.
- Exportable summaries for sharing or personal records.

## 5. Connectivity & Automation

### API Hooks
- REST and WebSocket endpoints mirror what the terminal uses, letting you script or automate any action.
- Rate limits and authentication details are documented for teams that want to integrate bots or analytical tools.

### Workspace Shortcuts
- Keyboard mappings let you place, cancel, or adjust orders without leaving the chart.
- Multi-monitor layouts: pin market boards, execution tickets, and analytics to separate screens.

## 6. Supportive Services

- **Onboarding walkthrough** explains each panel the first time you log in.
- **System status panel** keeps you updated on RPC latency, WebSocket health, and data freshness (`STATUS`).
- **Help links** jump directly to troubleshooting steps or live chat if you need assistance mid-session (`HELP`, `ERR`, `CHAT`).

---

Focus on the sections that match your trading style—scalpers can live in the execution desk, swing traders may lean on analytics, and everyone benefits from the risk dashboard keeping leverage in check.

Next steps: [Explore how QuantDesk powers perp markets](./how-quantdesk-powers-perps.md) or head back to the [core features](../core-features/) overview.
