UI Integration Plan: Pro + Lite
===============================

Goals
-----
- Insert your Pro terminal UI unchanged under `/pro`.
- Build Lite shell under `/lite` using Bitget/Hyperliquid layout with your tools.
- Share data providers, keep visuals intact.

Milestones
----------
1) Scaffolding (no visuals change)
   - Add routes `/pro` and `/lite` with lazy loading.
   - Add mode switcher (header or quick command) with `localStorage` persistence.
2) Pro Shell Insertion
   - Create `src/pro/` and drop-in your components, assets, styles.
   - Wrap with adapters to `WebSocketProvider`/`TradingProvider`.
   - Mount bottom taskbar + terminal; wire core buttons (markets, chart, trade, portfolio).
3) Lite Shell Build
   - Keep QuantDesk chrome (header/sidebar/layout) intact.
   - Mount Quantify Lite main content inside QuantDesk Lite route: all tabs, cards, heatmaps.
   - Use adapter layer to map QuantDesk stores/WS to Quantify Lite props.
   - Wire Lite tabs to mocks: market data, order book, trades, positions, orders.
4) Data Wiring
   - Market data, order book, trades from `websocketService` (mock for now).
   - Positions/orders from `tradingStore` (mock) mapped via adapter.
   - Component mapping inventory: reuse vs replace for chart, order book, trades, order form.
5) QA & Stabilization
   - Visual checklist, keyboard shortcuts, responsiveness, dark theme parity.
   - Mode switcher QA: persistence, route restore, no style leakage.
   - Compare `/lite` vs `/quantify-lite-ref` for parity on each tab.
6) Chart Library Improvement
   - Keep `lightweight-charts` with:
     - Volume, MA/EMA, RSI, MACD, VWAP
     - Drawing toolbar (trendline, ray, fib retracement)
   - Provide TradingView path if license becomes available.

Technical Steps
---------------
- Create `src/pro/index.tsx` exporting `ProApp`.
- Create `src/lite/index.tsx` exporting `LiteApp`.
- Update `App.tsx` routes with React.lazy and Suspense.
- Add `src/pro/adapters/marketAdapter.ts` to map store → Pro props.
- Add `src/pro/theme.css` and import only within Pro route.

Risks
-----
- Style leakage into global scope → contain CSS and code-split.
- Keyboard shortcut conflicts → namespace handlers per shell.
- Asset licensing → confirm before bundling external fonts/icons.

Acceptance Criteria
-------------------
- `/pro` renders identical to your current build.
- Bottom taskbar and terminal fully functional.
- `/lite` matches Bitget/Hyperliquid layout and uses your feature tabs.
- Both shells share data and work with mock WS streams today.


