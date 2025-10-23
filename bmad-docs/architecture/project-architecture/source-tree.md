# Source Tree

```
quantdesk-1.0.6/
├── backend/                          # Backend API Gateway Service
│   ├── src/
│   │   ├── controllers/              # Request handlers
│   │   │   ├── authController.ts
│   │   │   ├── tradingController.ts
│   │   │   ├── portfolioController.ts
│   │   │   └── aiController.ts
│   │   ├── services/                # Business logic
│   │   │   ├── supabaseDatabase.ts
│   │   │   ├── pythOracleService.ts
│   │   │   ├── tradingService.ts
│   │   │   └── aiIntegrationService.ts
│   │   ├── middleware/              # Express middleware
│   │   │   ├── errorHandling.ts
│   │   │   ├── rateLimiting.ts
│   │   │   └── authentication.ts
│   │   ├── routes/                  # API routes
│   │   │   ├── auth.ts
│   │   │   ├── trading.ts
│   │   │   ├── portfolio.ts
│   │   │   └── market.ts
│   │   ├── utils/                    # Shared utilities
│   │   │   ├── validation.ts
│   │   │   ├── logger.ts
│   │   │   └── constants.ts
│   │   ├── types/                    # TypeScript definitions
│   │   │   ├── trading.ts
│   │   │   ├── user.ts
│   │   │   └── api.ts
│   │   └── server.ts                 # Application entry point
│   ├── package.json
│   ├── tsconfig.json
│   └── Dockerfile
│
├── frontend/                         # Frontend Trading Interface
│   ├── src/
│   │   ├── components/               # React components
│   │   │   ├── trading/
│   │   │   │   ├── TradingInterface.tsx
│   │   │   │   ├── OrderForm.tsx
│   │   │   │   └── PositionManager.tsx
│   │   │   ├── portfolio/
│   │   │   │   ├── PortfolioOverview.tsx
│   │   │   │   └── PerformanceChart.tsx
│   │   │   ├── ai/
│   │   │   │   ├── MikeyChat.tsx
│   │   │   │   └── TradingRecommendations.tsx
│   │   │   └── common/
│   │   │       ├── Header.tsx
│   │   │       └── Sidebar.tsx
│   │   ├── hooks/                    # Custom React hooks
│   │   │   ├── useWebSocket.ts
│   │   │   ├── useTradingData.ts
│   │   │   └── useAuth.ts
│   │   ├── services/                 # API clients
│   │   │   ├── apiClient.ts
│   │   │   ├── websocketService.ts
│   │   │   └── tradingService.ts
│   │   ├── utils/                    # Frontend utilities
│   │   │   ├── formatters.ts
│   │   │   ├── validators.ts
│   │   │   └── constants.ts
│   │   ├── types/                    # TypeScript definitions
│   │   │   ├── trading.ts
│   │   │   ├── user.ts
│   │   │   └── api.ts
│   │   ├── styles/                   # Styling
│   │   │   ├── globals.css
│   │   │   └── components.css
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── MIKEY-AI/                         # AI Assistant Service
│   ├── src/
│   │   ├── agents/                   # LangChain agents
│   │   │   ├── tradingAgent.ts
│   │   │   ├── marketAnalysisAgent.ts
│   │   │   └── riskAssessmentAgent.ts
│   │   ├── services/                # AI services
│   │   │   ├── llmService.ts
│   │   │   ├── marketDataService.ts
│   │   │   └── analysisService.ts
│   │   ├── tools/                   # LangChain tools
│   │   │   ├── tradingTools.ts
│   │   │   ├── marketTools.ts
│   │   │   └── portfolioTools.ts
│   │   ├── utils/                   # AI utilities
│   │   │   ├── promptTemplates.ts
│   │   │   ├── dataProcessors.ts
│   │   │   └── responseFormatters.ts
│   │   ├── types/                    # AI-specific types
│   │   │   ├── analysis.ts
│   │   │   ├── recommendations.ts
│   │   │   └── llm.ts
│   │   └── server.ts                # AI service entry point
│   ├── package.json
│   └── tsconfig.json
│
├── data-ingestion/                   # Data Ingestion Pipeline
│   ├── src/
│   │   ├── services/                # Data services
│   │   │   ├── pythService.ts
│   │   │   ├── marketDataService.ts
│   │   │   └── dataValidationService.ts
│   │   ├── processors/              # Data processors
│   │   │   ├── priceProcessor.ts
│   │   │   ├── stalenessDetector.ts
│   │   │   └── dataAggregator.ts
│   │   ├── connectors/              # External connectors
│   │   │   ├── websocketConnector.ts
│   │   │   ├── restConnector.ts
│   │   │   └── databaseConnector.ts
│   │   ├── utils/                   # Data utilities
│   │   │   ├── formatters.ts
│   │   │   ├── validators.ts
│   │   │   └── transformers.ts
│   │   ├── types/                    # Data types
│   │   │   ├── marketData.ts
│   │   │   ├── priceFeed.ts
│   │   │   └── validation.ts
│   │   └── server.ts                # Data service entry point
│   ├── package.json
│   └── tsconfig.json
│
├── contracts/                        # Smart Contracts
│   ├── programs/
│   │   └── quantdesk-perp-dex/
│   │       ├── src/
│   │       │   ├── lib.rs            # Main program
│   │       │   ├── instructions/     # Program instructions
│   │       │   │   ├── open_position.rs
│   │       │   │   ├── close_position.rs
│   │       │   │   └── liquidate_position.rs
│   │       │   ├── state/            # Program state
│   │       │   │   ├── position.rs
│   │       │   │   ├── market.rs
│   │       │   │   └── user.rs
│   │       │   ├── utils/            # Utility functions
│   │       │   │   ├── math.rs
│   │       │   │   ├── validation.rs
│   │       │   │   └── pyth.rs
│   │       │   └── errors.rs         # Custom errors
│   │       └── Cargo.toml
│   ├── tests/                       # Contract tests
│   │   ├── integration/
│   │   └── unit/
│   ├── migrations/                  # Database migrations
│   └── Anchor.toml
│
├── shared/                          # Shared utilities and types
│   ├── types/                       # Common TypeScript types
│   │   ├── trading.ts
│   │   ├── user.ts
│   │   ├── api.ts
│   │   └── blockchain.ts
│   ├── utils/                       # Shared utilities
│   │   ├── validation.ts
│   │   ├── formatters.ts
│   │   ├── constants.ts
│   │   └── logger.ts
│   ├── schemas/                     # Validation schemas
│   │   ├── tradingSchemas.ts
│   │   ├── userSchemas.ts
│   │   └── apiSchemas.ts
│   └── package.json
│
├── database/                        # Database schemas and migrations
│   ├── schema.sql                   # Main schema
│   ├── migrations/                  # Migration files
│   ├── seeds/                       # Seed data
│   └── security/                    # Security policies
│
├── scripts/                         # Development and deployment scripts
│   ├── dev/                         # Development scripts
│   ├── deploy/                      # Deployment scripts
│   └── utils/                       # Utility scripts
│
├── docs/                           # Documentation
│   ├── architecture/               # Architecture docs
│   ├── api/                        # API documentation
│   └── deployment/                 # Deployment guides
│
├── tests/                          # Integration and E2E tests
│   ├── integration/
│   ├── e2e/
│   └── fixtures/
│
├── package.json                    # Root package.json with workspaces
├── pnpm-workspace.yaml             # pnpm workspace configuration
├── docker-compose.yml              # Local development environment
├── vercel.json                     # Vercel deployment configuration
└── README.md
```
