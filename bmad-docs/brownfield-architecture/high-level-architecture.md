# High Level Architecture

## Technical Summary

QuantDesk is a sophisticated multi-service Solana DEX platform with enterprise-grade security, real-time data ingestion, AI-powered trading assistance, and professional-grade infrastructure. The platform is designed to eliminate the need for traders to manage multiple tabs and platforms.

## Actual Tech Stack (from package.json analysis)

| Category  | Technology | Version | Notes                      |
| --------- | ---------- | ------- | -------------------------- |
| Runtime   | Node.js    | 20+     | Backend services           |
| Framework | Express    | 4.18.2  | Backend API server         |
| Frontend  | React      | 18.2.0  | Vite-based frontend        |
| Database  | PostgreSQL | 13+     | Supabase with TimescaleDB  |
| Blockchain| Solana     | 1.87.0  | Web3.js integration        |
| Smart Contracts | Rust | 1.70+ | Anchor Framework           |
| AI        | LangChain  | 0.3.15  | MIKEY-AI service           |
| Oracle    | Pyth Network | 2.0.0 | Real-time price feeds      |
| Cache     | Redis      | 4.6.10  | Session and data caching   |
| Package Manager | pnpm | Latest | Used throughout project     |

## Repository Structure Reality Check

- **Type**: Monorepo with multi-service architecture
- **Package Manager**: pnpm (CRITICAL: Always use pnpm, never npm)
- **Notable**: Each service has independent package.json with shared dependencies
