# Backend Department Architecture

## Overview
Comprehensive Node.js/Express API server with 34+ specialized route handlers and 25+ services covering trading, risk management, analytics, and AI integration. Based on actual codebase analysis.

## Technology Stack (Based on Actual package.json)
- **Runtime**: Node.js 20.10.0 (engines: >=18.0.0)
- **Framework**: Express 4.18.2 + TypeScript 5.3.2
- **Database**: Supabase (PostgreSQL 8.11.3 + @supabase/supabase-js 2.58.0)
- **Caching**: Redis 4.6.10 + ioredis 5.8.0
- **Real-time**: Socket.io 4.7.4 + WebSocket 8.18.3
- **Authentication**: JWT 9.0.2 + Passport 0.7.0 + SIWS support
- **Security**: Helmet 7.1.0 + bcrypt 6.0.0 + CORS 2.8.5
- **Blockchain**: @coral-xyz/anchor 0.32.0 + @solana/web3.js 1.87.0
- **Oracle**: @pythnetwork/hermes-client 2.0.0
- **Monitoring**: Winston 3.11.0 + morgan 1.10.0

## Actual Route Architecture (34+ Route Handlers)
```
src/routes/ (34+ specialized route files)
├── Trading Core
│   ├── orders.ts                 # Order management
│   ├── trades.ts                 # Trade execution
│   ├── positions.ts              # Position tracking
│   ├── markets.ts                # Market data
│   ├── realSupabaseMarkets.ts    # Real market integration
│   ├── advancedOrders.ts         # Advanced order types
│   ├── crossCollateral.ts        # Cross-collateral trading
│   └── liquidity.ts              # Liquidity management
├── Risk & Analytics
│   ├── portfolioAnalytics.ts     # Portfolio analytics service
│   ├── advancedRiskManagement.ts # Advanced risk system
│   ├── jitLiquidity.ts           # Just-in-time liquidity
│   ├── deposits.ts               # Deposit/funding management
│   └── protocol-stats.ts        # Protocol statistics
├── AI & Development Support
│   ├── ai.ts                     # AI service endpoints
│   ├── aiAgent.ts                # AI development assistance
│   └── apiDocs.ts                # API documentation
├── User & Authentication
│   ├── auth.ts                   # Authentication (JWT + SIWS)
│   ├── siws.ts                   # Sign-in with Solana
│   ├── users.ts                  # User management
│   ├── referrals.ts              # Referral system
│   └── accounts.ts               # Account management
├── Oracle & Data
│   ├── oracle.ts                 # Oracle integration
│   ├── supabaseOracle.ts         # Supabase oracle service
│   └── metrics.ts                # System metrics
├── System & Monitoring
│   ├── admin.ts                  # Admin management
│   ├── grafanaIntegration.ts     # Grafana integration
│   ├── rpcStats.ts               # RPC statistics
│   └── accountState.ts           # Account state management
└── Development Tools
    ├── rpcTesting.ts             # RPC testing
    ├── marketManagement.ts       # Market management
    └── simpleMarkets.ts          # Simple market endpoints
```

## Service Architecture (25+ Specialized Services)
```
src/services/ (25+ service files)
├── Core Trading Services
│   ├── supabaseDatabase.ts       # Main database abstraction
│   ├── pythOracleService.ts      # Pyth oracle integration
│   ├── advancedOrderService.ts   # Advanced order handling
│   ├── matching.ts               # Order matching engine
│   └── liquidationBot.ts         # Liquidation automation
├── Risk Management Services
│   ├── advancedRiskManagementService.ts # Risk calculations
│   ├── portfolioAnalyticsService.ts     # Portfolio analytics
│   ├── crossCollateralService.ts        # Cross-collateral logic
│   └── jitLiquidityService.ts           # JIT liquidity provider
├── System & Infrastructure
│   ├── websocket.ts              # WebSocket service
│   ├── redisClient.ts            # Redis client management
│   ├── rpcLoadBalancer.ts        # RPC load balancing
│   ├── systemMonitor.ts          # System monitoring
│   └── transactionVerificationService.ts # Transaction verification
├── User & Support Services
│   ├── adminUserService.ts       # Admin user management
│   ├── referralService.ts        # Referral system
│   ├── coinGeckoService.ts       # Price data
│   └── fallbackPriceService.ts   # Fallback pricing
├── Development Support
│   ├── apiDocumentation.ts       # Auto-generated API docs
│   ├── solana.ts                 # Solana integration
│   └── grafanaMetrics.ts         # Grafana metrics
└── Utilities
    ├── funding.ts                # Funding rate calculations
    ├── orderScheduler.ts         # Order scheduling
    └── webhookService.ts         # Webhook handling
```

## Key Integrations (Based on Actual Codebase)
- **Database**: Supabase with custom supabaseDatabase abstraction layer
- **Oracle Integration**: Pyth Network via @pythnetwork/hermes-client 2.0.0
- **Solana Integration**: Anchor 0.32.0 with RPC load balancing and transaction verification
- **Caching Layer**: Redis 4.6.10 + ioredis with custom client management
- **Real-time Communication**: Socket.io 4.7.4 + WebSocket 8.18.3
- **AI Development Support**: Dedicated /api/dev/* endpoints for AI assistance
- **Authentication**: SIWS (Sign-In with Solana) + JWT + Passport strategies
- **Risk Management**: Advanced risk calculations with portfolio analytics and JIT liquidity
- **Monitoring**: Winston logging + Grafana metrics integration
- **Analytics**: Portfolio analytics with VaR, Sharpe ratio, and advanced risk metrics

## API Design Patterns
- RESTful design with consistent response format
- Versioned endpoints (/api/v1/)
- Rate limiting per user/IP
- Comprehensive error handling
- OpenAPI/Swagger documentation

## Development Guidelines
- Service layer architecture
- Dependency injection pattern
- Database abstraction layer
- Comprehensive input validation
- Structured logging with correlation IDs

## Testing Strategy (Based on Actual package.json Scripts)
- Unit Tests: Jest 29.7.0 with Jest config for TypeScript
- Integration Testing: Supertest 6.3.3 for API testing
- Coverage Reporting: Jest --coverage for code coverage metrics
- Type Checking: TypeScript 5.3.2 with type-check script
- Linting: ESLint 8.54.0 with @typescript-eslint plugins
- Development Scripts: nodemon 3.0.2 for hot reloading during development
