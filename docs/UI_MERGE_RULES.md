QuantDesk UI Merge Rules (Pro + Lite)
=====================================

Non‑negotiables
---------------
- Preserve your existing Pro UI exactly: colors, typography, textures, icons, spacing, shadows, terminal UX, bottom taskbar, chart tabs, keyboard handling.
- No visual token substitutions in Pro. We can wrap, not restyle.
- All imported assets (fonts, svgs, images) must remain byte‑identical.
- Keep component DOM structure where hotkeys / CSS rely on element order.

Routing & Modes
---------------
- Provide two shells:
  - `/pro` → your Pro terminal (unchanged except for data wiring).
  - `/lite` → Bitget/Hyperliquid layout using your feature set and our data plumbing.
- Add a header mode switcher that persists preference to `localStorage:quantdesk_ui_mode`.
  - On app load, read `quantdesk_ui_mode` to default route/visuals; do not auto-redirect if explicit route is provided.

Code Organization
-----------------
- Place your UI under `frontend/src/pro/` and `frontend/src/lite/` to avoid collisions.
- Shared primitives in `frontend/src/ui-kit/` (buttons, badges) but Pro components should depend only if it does not alter visuals.
- Theme files:
  - Pro: `frontend/src/pro/theme.css` (imported only inside Pro routes).
  - Lite: uses existing Tailwind theme and tokens.
  - Persistence rules:
    - Do not write theme tokens from Pro into global scope.
    - When switching modes, ensure Pro theme is unmounted (no lingering CSS).

Styling Rules
-------------
- Do not merge Pro CSS into global scope. Scope via route‑based code‑split or CSS modules to avoid leaking styles.
- No Tailwind class replacement inside Pro components; if needed, add wrappers.
- Keep z-index layering from Pro exactly (terminal, taskbar, overlays).

Data & State
------------
- Use current providers (`WebSocketProvider`, `TradingProvider`, wallet provider).
- Pro components consume data via a thin adapter layer to avoid internal refactors.
- Do not mutate shared state shape; adapters map to existing store types.

Performance & Loading
---------------------
- Code-split shells: lazy‑load `/pro` and `/lite`.
- Ensure zero layout shift on first paint (fixed heights for terminal/taskbar during load).

Testing & QA
------------
- Visual regression checklist for Pro terminal and bottom taskbar.
- Smoke tests: route loads, hotkeys work, chart renders, dock buttons clickable.
- Manual cross‑browser check: Chromium, Firefox.

Security & Licensing
--------------------
- Confirm licenses for any third‑party assets/libs bundled with Pro.
- No external CDN fonts without explicit approval; bundle locally.

Migration Boundaries
--------------------
- Pro UI: insert as‑is; only add wrappers/adapters.
- Lite UI: QuantDesk chrome (header/sidebar/layout) stays; port Quantify Lite main content (all tabs, cards, heatmaps) into `/lite` without altering QuantDesk chrome.
- Add hidden reference route `/quantify-lite-ref` for raw Quantify Lite to check visual parity.


