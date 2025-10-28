# Account Lifecycle Overview

QuantDesk walks traders through a professional onboarding flow—from wallet connect to funded trading account—in a way that mirrors top-tier perp venues.

## 1. Connect Wallet

- **Wallet prompt** invites users to connect Phantom, Solflare, or any supported Solana wallet.
- **SIWS Authentication**: Wallet signature verification ensures secure, non-custodial access
- Feature highlights set expectations (non-custodial, low latency, risk managed).
- Session established via HTTP-only cookies with 7-day expiration
- Once connected, the app detects any existing trading accounts and proceeds accordingly.

## 2. Create Trading Account

- If a wallet has no QuantDesk account PDA, the user sees the account creation panel.
- The panel displays wallet details, network, and account features before initiating the on-chain transaction.
- Account creation uses our Solana program (PDA-based) to set up collateral, positions, and permissions in one step.

## 3. Fund the Account

- Newly created accounts show the deposit module with live balance/health readouts.
- Traders choose from supported assets (USDC, SOL, BTC, ETH, etc.), input size, and use quick percentage buttons.
- Projected collateral, APY, and risk information appear before confirming.

## 4. Start Trading

- Once funded, the full terminal unlocks: order tickets, positions panel, risk dashboard, and analytics.
- Account state stays synced via backend services—every trade, transfer, or adjustment updates the dashboard instantly.
- MIKEY continues to monitor health factors and will surface alerts if collateral drops or leverage climbs.

## Built-In Safeguards

- **Signature Verification**: SIWS authentication requires cryptographic wallet signatures
- **Transaction Verification**: All on-chain transactions verified before processing
- **Session Security**: HTTP-only cookies prevent XSS attacks
- **Rate Limiting**: 5 requests/minute for authentication endpoints
- **Audit Logging**: Complete transaction history for compliance review
- **Error Handling**: Clear loading/error states throughout lifecycle
- **Backend Validation**: All account creation and deposit flows validated server-side

This lifecycle makes onboarding smooth for newcomers while preserving the controls professional desks expect. Pair it with the [Multi-Account Control](../trading-capabilities/multi-account-control.md) and [Terminal Toolbox](../trading-capabilities/perp-terminal-toolbox.md) docs for the full experience story.
