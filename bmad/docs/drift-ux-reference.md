# Drift Perp Trading UI – Key Micro-Interactions and Controls

Source: Observed via live app (https://app.drift.trade/, SOL-PERP). Focus is on small but important UX details to mirror in QuantDesk Pro (/pro) and Lite (/lite).

## Core Order Panel
- Side toggle: Long / Short
- Order type tabs: Market, Limit, Others (menu)
- Size input with asset selector button (e.g., SOL)
- Max balance hint: "Max: 0.00 USD" near size
- Leverage controls:
  - Preset buttons: 0%, 25%, 50%, 75%, 100% (percentage sizing)
  - Slider control (continuous)
  - Textbox for explicit multiplier with trailing "x"
- Reduce Only checkbox
- TP/SL expander:
  - Toggle button "TP/SL" reveals two rows (TP, SL)
  - For each: price or % mode selector (button "Price %"), paired input, unit label
  - Disabled until size is valid; inline validation messages
  - Est. P&L readout per TP/SL row

## Execution Options / Costs
- Slippage control:
  - Button "Slippage (Dynamic)" opens presets
  - Textbox (e.g., "Dynamic") + presets 0.1%, 0.5%, 1%
- Liquidation Price: value or "None" until position derived
- Position: current position size
- Fees: readout; disabled state when not applicable
- Account Leverage: live value (e.g., 0x)

## Market/Chart/Orderbook Context
- Market summary:
  - Oracle price label/value alongside last/mark
  - Funding / Last 24h (APR) pair (e.g., "-28.59% / -0.33%")
  - Open Interest, 24h Volume
  - "About Market / View Details" button
- Orderbook:
  - Tabs: Orderbook / Recent Trades
  - Depth controls: price increment grouping (e.g., $0.001) and quote currency button (USD)
  - Center ticker shows last/ mid and "Spread: $X (Y%)"
- Chart (TradingView):
  - Interval (1h) toolbar, indicators, Oracle vs Fill candles toggle
  - Quick interval buttons: 1D, 5D, 1M, 3M, 6M, ALL

## Account Overview Widgets
- Net USD Value (eye toggle), Account Leverage, Health (%) with icons
- Lower tabs: Positions / Orders / Balances / Order History / Position History / Account
- "Hide other markets" toggle in tables

## Footer / System Controls
- Status: Operational
- RPC selector with latency (e.g., "RPC: Helius 1 (105ms)")
- Lite/Pro toggle switch
- Wallet auto-confirm (Phantom) toggle
- App version (e.g., v2.143.25)

## Micro-Interaction Patterns to Mirror
- Disable TP/SL until size > 0; show inline validation
- TP/SL support absolute price or %; keep unit label visible and consistent
- Leverage: presets + slider + textbox (all in sync)
- Slippage: presets + custom, default to dynamic
- Always show Oracle price beside mark/last; show Spread near orderbook midpoint
- Clear Lite/Pro toggle and status indicators

## Mapping to QuantDesk
- Pro (/pro):
  - ORDER window: add TP/SL expander with price/% modes and disabled-until-valid logic (partially done)
  - Add slippage control (dynamic + presets) and show Liq/Fees/Acct Leverage
  - OVERVIEW window: compact quick conditional form (done); consider mini TP/SL expander
- Lite (/lite → Trading):
  - Trading column: conditional form + orders list (done); add Reduce Only, Slippage presets, and leverage presets next to slider
  - Account Overview bottom: include quick TP/SL inline

## Next Steps
- Implement UI states & wiring:
  - TP/SL enablement, price/% toggle, reduce-only, slippage presets
  - Leverage presets + slider + textbox sync
  - Wire Oracle/Mark, Liq estimate, Fees
- Add Playwright UX checks for presence/enable/disable/validation

---
Generated from UX reconnaissance on Drift Trade (public market view, no wallet).
