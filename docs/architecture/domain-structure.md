# 🌐 QuantDesk Domain Structure

## **Updated Domain Layout for quantdesk.app**

```
quantdesk.app/              → Landing Page
├── /                       → Landing Page (marketing)

quantdesk.app/lite/         → QuantDesk Lite (Trading Interface)
├── /lite/trading           → Trading Interface (DEFAULT for lite)
├── /lite/overview          → Overview/Dashboard
├── /lite/portfolio         → Portfolio Management
├── /lite/markets           → Market Overview
├── /lite/staking           → Staking Interface
└── /lite/*                 → Other lite features

quantdesk.app/pro/          → QuantDesk Pro (Terminal Interface)
├── /pro                    → Pro Terminal (DEFAULT for pro)
└── /pro/*                  → Pro-specific features

quantdesk.app/admin/        → Admin Dashboard (AUTH GATED)
├── /admin                  → Admin Terminal
├── /admin/login            → Admin Login
└── /admin/logout           → Admin Logout

quantdesk.app/api/          → Backend API
├── /api/*                  → All API endpoints
└── /ws                     → WebSocket connections

quantdesk.app/docs/         → Documentation
├── /docs                   → Documentation Index
└── /docs/html/*            → HTML documentation files
```

## **Key Changes Made:**

1. **Lite starts at `/lite/trading`** - Trading interface is the default for lite
2. **Pro starts at `/pro`** - Simple pro terminal interface
3. **All other pages under `/lite/`** - Overview, portfolio, markets, staking, etc.
4. **Landing page at root** - Marketing page at `/`
5. **Admin gated at `/admin/`** - Protected admin interface
