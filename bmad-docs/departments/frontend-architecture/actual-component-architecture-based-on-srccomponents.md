# Actual Component Architecture (Based on src/components)
```
src/
├── components/ (45+ components)
│   ├── charts/                    # Chart components
│   │   ├── BorrowedChart/        # TradingView integration
│   │   │   ├── index.ts
│   │   │   └── QuantDeskTradingViewChart.tsx
│   │   ├── QuantDeskChart.tsx    # Main chart component
│   │   ├── RechartsCandleChart.tsx # Candlestick charts
│   │   ├── RechartsTVChart.tsx   # TradingView-style charts
│   │   └── SimpleChart.tsx       # Basic chart component
│   ├── trading/                   # Trading interface
│   │   ├── DexTradingInterface.tsx    # Main trading UI
│   │   ├── OrderBook.tsx              # Order book display
│   │   ├── PortfolioDashboard.tsx     # Portfolio management
│   │   ├── AccountSlideOut.tsx       # Account management
│   │   └── PositionsTable.tsx        # Positions display
│   ├── market/                    # Market data
│   │   ├── DexScreener.tsx           # Market screener
│   │   ├── DexHeatmap.tsx            # Market heatmap
│   │   ├── DexTickerTape.tsx         # Price ticker
│   │   ├── PriceDisplay.tsx          # Price display
│   │   └── RecentTrades.tsx          # Recent trades
│   ├── order/                     # Order management
│   │   ├── Orders.tsx                 # Order placement
│   │   └── WithdrawModal.tsx         # Withdrawal interface
│   ├── wallet/                    # Wallet integration
│   │   ├── WalletButton.tsx          # Wallet connection
│   │   └── DepositModal.tsx         # Deposit interface
│   ├── admin/                     # Admin interface
│   │   └── ProTerminalSettings.tsx   # Admin settings
│   └── common/                    # Reusable components
│       ├── Layout.tsx                # Page layout
│       ├── Header.tsx                # Site header
│       ├── Sidebar.tsx               # Navigation
│       ├── ThemeToggle.tsx           # Theme switching
│       └── BottomTaskbar.tsx         # Bottom navigation
├── contexts/ (7 contexts)
│   ├── AccountContext.tsx           # Account state management
│   ├── MarketContext.tsx            # Market data context
│   ├── PriceContext.tsx             # Price data context
│   ├── ProgramContext.tsx           # Solana program context
│   ├── TabContext.tsx               # Tab navigation context
│   ├── ThemeContext.tsx             # Theme management
│   └── MockWalletProvider.tsx       # Mock wallet for testing
├── stores/ (2 stores)
│   ├── PriceStore.ts                # Centralized price data management
│   └── tradingStore.ts              # Trading state store
├── services/ (19 services)
│   ├── smartContractService.ts      # Solana smart contract integration
│   ├── websocketService.ts          # WebSocket management
│   ├── tradingService.ts             # Trading operations
│   ├── apiClient.ts                  # HTTP client configuration
│   ├── balanceService.ts             # Balance management
│   ├── marketDataService.ts          # Market data handling
│   ├── portfolioService.ts           # Portfolio operations
│   └── *.ts                          # Other services
├── hooks/ (6 hooks)
│   ├── usePerformanceMonitor.ts      # Performance monitoring
│   ├── useResponsiveDesign.ts        # Responsive design utilities
│   ├── useTabState.ts               # Tab state management
│   ├── useTickerClick.ts            # Ticker interaction
│   ├── useTrading.ts                # Trading operations
│   └── useWalletAuth.ts             # Wallet authentication
├── pages/ (4 pages)
│   ├── PortfolioPage.tsx            # Portfolio view
│   ├── MarketsPage.tsx              # Markets view
│   ├── LandingPage.tsx              # Landing page
│   └── ChatPage.tsx                 # Chat interface
├── providers/ (2 providers)
│   ├── TradingProvider.tsx          # Main trading provider
│   └── WebSocketProvider.tsx       # WebSocket provider
├── lite/                            # Lite mode components
│   ├── LiteRouter.tsx               # Lite mode routing
│   ├── TradingTab.tsx               # Trading tab component
│   └── adapters/
│       └── marketAdapter.ts          # Market data adapter
├── pro/                             # Pro mode components
│   ├── index.tsx                    # Pro mode entry point
│   └── theme.css                    # Pro mode styling
├── lib/                             # External library integrations
│   ├── dydxfeed/                    # dYdX data feed
│   ├── launchableMarketFeed/        # Launchable market data
│   └── spotDatafeed/               # Spot data feed
└── utils/ (9 utilities)
    ├── accountHelpers.ts            # Account utilities
    ├── ChartManager.ts              # Chart management
    ├── constants.ts                 # Application constants
    ├── formatters.ts                # Data formatting
    ├── IndicatorManager.ts          # Technical indicators
    ├── indicators.ts                # Indicator definitions
    ├── logger.ts                    # Logging utilities
    └── supabase.ts                  # Supabase client
```
