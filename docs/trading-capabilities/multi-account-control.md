# Multi-Account Control

QuantDesk lets traders organize capital the way professional desks do: master accounts, isolated trading sub-accounts, and optional delegated access.

## Account Types

- **Master Account** – Created automatically when a wallet connects; controls collateral, sub-accounts, and permissions.
- **Trading Accounts** – Multiple isolated buckets (typically up to 10) each with their own margin, positions, and order limits.
- **Delegated Accounts** – Optional wallets allowed to trade and manage positions without withdrawal rights—ideal for teammates or managed accounts.

## Why It Matters

- **Strategy Isolation** – Run different leverage profiles or assets without cross-contaminating risk.
- **Capital Efficiency** – Transfer collateral between trading accounts instantly while keeping liquidation buffers visible.
- **Team Collaboration** – Hand trading execution to a delegate while the master account retains ultimate control.

## How It Works in the Terminal

- Switch trading accounts from the terminal header; all positions, P&L, and risk metrics update instantly for the selected account.
- Use the account management drawer to create or rename trading accounts, set delegate permissions, and monitor health indicators.
- Move collateral between accounts with on-screen transfer flows—each transfer logs into the audit trail for transparency.

## Safeguards

- Delegated accounts inherit granular permissions (deposit, trade, cancel) with withdraw disabled by default.
- Every action is logged against the master account, making reviews and audits straightforward.
- Margin checks run per trading account, keeping liquidations isolated to the strategy that triggered them.

QuantDesk’s multi-account framework gives solo traders and teams the same tooling institutional desks rely on—without sacrificing the simplicity of a wallet-based sign-in.
