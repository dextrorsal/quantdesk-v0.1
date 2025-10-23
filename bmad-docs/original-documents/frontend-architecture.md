# Frontend Department Architecture

## Overview
React-based trading interface with PWA capabilities, professional trading components, and Solana wallet integration. Based on actual codebase analysis with 45+ specialized components.

## Technology Stack (Based on Actual Package.json & Vite Config)
- **Framework**: React 18.2.0 + TypeScript 5.2.2
- **Build Tool**: Vite 7.1.8 with PWA plugin
- **Styling**: Tailwind CSS 3.3.6 + Bootstrap 5.3.8 + React Bootstrap 2.10.10
- **State Management**: Zustand 4.4.7 + React Context (TradingProvider, MarketProvider, etc.)
- **Real-time**: Socket.io-client 4.8.1 + WebSocket 8.18.3
- **Charts**: Lightweight-charts 4.2.3 + Recharts 2.15.4
- **Routing**: React Router DOM 6.30.1 + React Router 7.9.3
- **Solana Integration**: @solana/wallet-adapter ecosystem (base, react, react-ui, wallets)
- **UI Components**: Lucide React, React Feather, Heroicons
- **Grid Layout**: React Grid Layout 1.5.2
- **Forms**: React Hot Toast 2.4.1

## Actual Component Architecture (Based on src/components)
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

## Key Integrations (Based on Actual Code)
- **Backend API**: REST + Socket.io WebSocket endpoints (proxy through Vite)
- **Solana Wallet**: Phantom & Solflare adapters via @solana/wallet-adapter ecosystem
- **Real-time Data**: Socket.io-client + WebSocket for live updates
- **Chart Libraries**: Lightweight-charts (professional), Recharts (analytics)
- **Authentication**: SIWS (Sign-In with Solana) + JWT tokens
- **PWA**: Progressive Web App with offline capabilities and service worker
- **Environment**: Devnet deployment with localhost backend proxy

## State Management Architecture

### Zustand Stores
```typescript
// PriceStore.ts - Centralized price data management
class PriceStore {
  private prices = new Map<string, PriceData>()
  private subscribers = new Set<PriceSubscriber>()
  
  // Singleton pattern for global access
  static getInstance(): PriceStore
  
  // Subscription-based updates
  subscribe(callback: PriceSubscriber): () => void
  updatePrices(priceUpdates: PriceUpdate[]): void
}

// tradingStore.ts - Trading state management
interface TradingState {
  markets: Market[]
  selectedMarket: Market | null
  positions: Position[]
  orders: Order[]
  isLoading: boolean
  error: string | null
}
```

### React Context Providers
```typescript
// AccountContext.tsx - Account state management
interface AccountContextType {
  wallet: any
  accountState: UserAccountState | null
  collateralAccounts: CollateralAccount[]
  positions: Position[]
  orders: Order[]
  loading: boolean
  error: string | null
  
  // Actions
  fetchAccountState: () => Promise<void>
  createAccount: () => Promise<string>
  depositCollateral: (assetType: CollateralType, amount: number) => Promise<string>
  placeOrder: (market: string, orderType: OrderType, side: PositionSide, size: number, price: number, leverage: number) => Promise<string>
}
```

## Component Architecture Patterns

### Functional Components with TypeScript
```typescript
// DexTradingInterface.tsx - Main trading interface
const DexTradingInterface: React.FC = () => {
  const { markets, selectedMarket, selectMarketBySymbol } = useMarkets();
  const [selectedSymbol, setSelectedSymbol] = useState(selectedMarket?.symbol || 'BTC-PERP');
  
  const handleSymbolChange = (symbol: string) => {
    setSelectedSymbol(symbol);
    selectMarketBySymbol(symbol);
  };
  
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Component JSX */}
    </div>
  );
};
```

### Layout Component Pattern
```typescript
// Layout.tsx - Application layout wrapper
const Layout: React.FC<LayoutProps> = ({ children }) => {
  return (
    <ThemeProvider>
      <TabProvider defaultTab="trading">
        <div className="h-screen flex flex-col overflow-hidden">
          <Header />
          <main className="flex-1 overflow-auto pb-16">
            {children}
          </main>
          <BottomTaskbar />
        </div>
      </TabProvider>
    </ThemeProvider>
  )
}
```

## API Integration Architecture

### HTTP Client Configuration
```typescript
// apiClient.ts - Axios configuration
export const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:3002',
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for authentication
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})
```

### WebSocket Service Architecture
```typescript
// websocketService.ts - Real-time data streaming
export interface WebSocketMessage {
  type: 'market_data' | 'order_book' | 'trade' | 'position_update' | 'order_update'
  channel: string
  data: any
  timestamp: number
}

class WebSocketService {
  private ws: WebSocket | null = null
  private subscribers = new Map<string, Set<Function>>()
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  
  // Connection management
  connect(url: string): Promise<void>
  disconnect(): void
  subscribe(channel: string, callback: Function): () => void
  send(message: WebSocketMessage): void
}
```

## Routing Architecture

### React Router Implementation
```typescript
// App.tsx - Main routing configuration
function AppRoutes() {
  return (
    <ConnectionProvider endpoint={endpoint}>
      <WalletProvider wallets={wallets} autoConnect>
        <WalletModalProvider>
          <MarketProvider>
            <PriceProvider websocketUrl="ws://localhost:3002/" fallbackApiUrl="/api/prices">
              <TradingProvider>
                <AccountProvider>
                  <Routes>
                    <Route path="/" element={<LandingPage />} />
                    <Route path="/lite" element={<Layout><LiteRouter /></Layout>} />
                    <Route path="/pro" element={<React.Suspense fallback={<div>Loading Pro…</div>}>
                      {React.createElement(React.lazy(() => import('./pro/index')))}
                    </React.Suspense>} />
                    <Route path="/trading" element={<Layout><TradingTab /></Layout>} />
                    <Route path="/portfolio" element={<Layout><PortfolioPage /></Layout>} />
                    <Route path="/markets" element={<Layout><MarketsPage /></Layout>} />
                    <Route path="*" element={<LandingPage />} />
                  </Routes>
                </AccountProvider>
              </TradingProvider>
            </PriceProvider>
          </MarketProvider>
        </WalletModalProvider>
      </WalletProvider>
    </ConnectionProvider>
  )
}
```

### Lazy Loading Implementation
```typescript
// Lazy loading for performance optimization
<Route path="/pro" element={
  <React.Suspense fallback={<div className="p-6 text-white">Loading Pro…</div>}>
    {React.createElement(React.lazy(() => import('./pro/index')))}
  </React.Suspense>
} />
```

## Styling System Architecture

### Tailwind CSS Configuration
```typescript
// tailwind.config.js - Comprehensive theme system
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      screens: {
        'xs': '475px',
        'sm': '640px',
        'md': '768px',
        'lg': '1024px',
        'xl': '1280px',
        '2xl': '1536px',
        '3xl': '1920px',
        '4xl': '2560px',
        // Custom breakpoints for trading interfaces
        'mobile': {'max': '768px'},
        'tablet': {'min': '769px', 'max': '1024px'},
        'desktop': {'min': '1025px', 'max': '1920px'},
        'large-desktop': {'min': '1921px', 'max': '2560px'},
        'ultra-wide': {'min': '2561px'},
      },
      colors: {
        // Lite Theme Colors (Blue)
        'lite-primary': {
          50: '#eff6ff',
          100: '#dbeafe',
          200: '#bfdbfe',
          300: '#93c5fd',
          400: '#60a5fa',
          500: '#3b82f6',  // Main brand blue
          600: '#2563eb',
          700: '#1d4ed8',
          800: '#1e40af',
          900: '#1e3a8a',
        },
        // Pro Theme Colors (Orange)
        'pro-primary': {
          50: '#fff7ed',
          100: '#ffedd5',
          200: '#fed7aa',
          300: '#fdba74',
          400: '#fb923c',
          500: '#f97316',  // Main brand orange
          600: '#ea580c',
          700: '#c2410c',
          800: '#9a3412',
          900: '#7c2d12',
        },
        // Universal CSS Variables
        primary: {
          50: 'var(--primary-50)',
          100: 'var(--primary-100)',
          // ... dynamic theme colors
        },
      },
    },
  },
}
```

### Theme System Implementation
```typescript
// ThemeContext.tsx - Dynamic theme management
interface ThemeContextType {
  theme: 'lite' | 'pro'
  setTheme: (theme: 'lite' | 'pro') => void
  colors: ThemeColors
}

// CSS Variables for dynamic theming
:root {
  --primary-50: #eff6ff;
  --primary-500: #3b82f6;
  --bg-primary: #000000;
  --text-primary: #ffffff;
}
```

## Testing Architecture

### Vitest Configuration
```typescript
// vitest.config.ts - Testing framework setup
export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{js,mjs,cjs,ts,mts,cts,jsx,tsx}'],
    exclude: ['node_modules', 'dist', '.idea', '.git', '.cache'],
  },
})
```

### Test Setup and Mocks
```typescript
// test/setup.ts - Test environment configuration
import '@testing-library/jest-dom'

// Mock environment variables
Object.defineProperty(import.meta, 'env', {
  value: {
    VITE_SOLANA_RPC_URL: 'https://api.devnet.solana.com',
    VITE_SUPABASE_URL: 'https://test.supabase.co',
    VITE_SUPABASE_ANON_KEY: 'test-key',
  },
})

// Mock WebSocket
global.WebSocket = class WebSocket {
  constructor() {}
  close() {}
  send() {}
  addEventListener() {}
  removeEventListener() {}
} as any

// Mock ResizeObserver
global.ResizeObserver = vi.fn().mockImplementation(() => ({
  observe: vi.fn(),
  unobserve: vi.fn(),
  disconnect: vi.fn(),
}))
```

### Testing Patterns
```typescript
// Component testing example
describe('PortfolioDashboard', () => {
  it('renders portfolio data correctly', () => {
    render(<PortfolioDashboard />)
    expect(screen.getByText('Portfolio')).toBeInTheDocument()
  })
  
  it('handles loading state', () => {
    render(<PortfolioDashboard loading={true} />)
    expect(screen.getByText('Loading...')).toBeInTheDocument()
  })
})
```

## Environment Configuration

### Vite Configuration
```typescript
// vite.config.ts - Build and development configuration
export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
        runtimeCaching: [{
          urlPattern: /^https:\/\/api\.quantdesk\.app\/.*/i,
          handler: 'NetworkFirst',
          options: {
            cacheName: 'api-cache',
            expiration: { maxEntries: 10, maxAgeSeconds: 60 * 60 * 24 }
          }
        }]
      },
      manifest: {
        name: 'QuantDesk Trading Platform',
        short_name: 'QuantDesk',
        description: 'Professional crypto trading platform',
        theme_color: '#3b82f6',
        background_color: '#000000',
        display: 'standalone',
      }
    })
  ],
  resolve: {
    alias: { '@': path.resolve(__dirname, './src') },
  },
  define: {
    global: 'globalThis',
    'process.env': 'import.meta.env',
    // Environment variables
    'import.meta.env.VITE_API_URL': JSON.stringify(process.env.VITE_API_URL || 'http://localhost:3002'),
    'import.meta.env.VITE_WS_URL': JSON.stringify(process.env.VITE_WS_URL || 'ws://localhost:3002'),
    'import.meta.env.VITE_SOLANA_RPC_URL': JSON.stringify(process.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com'),
  },
  server: {
    port: 3001,
    host: true,
    proxy: {
      '/api': {
        target: 'http://localhost:3002',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false, // Disable sourcemaps in production for security
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          solana: ['@solana/web3.js', '@solana/wallet-adapter-base'],
          charts: ['lightweight-charts', 'recharts'],
        },
      },
    },
  },
})
```

### TypeScript Configuration
```json
// tsconfig.json - TypeScript compiler options
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": { "@/*": ["./src/*"] }
  },
  "include": ["src", "../archive/index-stock-backup.tsx"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

## Security Architecture

### Authentication Patterns
```typescript
// Wallet-based authentication
const { connected, publicKey, wallet } = useWallet();

// JWT token management
const token = localStorage.getItem('auth_token');
if (token) {
  config.headers.Authorization = `Bearer ${token}`;
}

// SIWS (Sign-In with Solana) integration
const signMessage = async (message: string) => {
  if (!wallet || !publicKey) throw new Error('Wallet not connected');
  const signature = await wallet.signMessage(new TextEncoder().encode(message));
  return signature;
};
```

### Data Validation Strategies
```typescript
// Input validation patterns
interface OrderFormData {
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop';
  size: number;
  price?: number;
  leverage: number;
}

const validateOrderForm = (data: OrderFormData): ValidationResult => {
  const errors: string[] = [];
  
  if (!data.symbol || data.symbol.length < 3) {
    errors.push('Invalid symbol');
  }
  
  if (data.size <= 0) {
    errors.push('Size must be greater than 0');
  }
  
  if (data.leverage < 1 || data.leverage > 20) {
    errors.push('Leverage must be between 1 and 20');
  }
  
  return { isValid: errors.length === 0, errors };
};
```

### XSS Prevention Measures
```typescript
// Safe HTML rendering
import DOMPurify from 'dompurify';

const SafeHTML: React.FC<{ content: string }> = ({ content }) => {
  const sanitizedContent = DOMPurify.sanitize(content);
  return <div dangerouslySetInnerHTML={{ __html: sanitizedContent }} />;
};

// Input sanitization
const sanitizeInput = (input: string): string => {
  return input.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '');
};
```

## Performance Optimization Architecture

### Bundle Optimization Strategies
```typescript
// Manual chunk splitting for optimal loading
rollupOptions: {
  output: {
    manualChunks: {
      vendor: ['react', 'react-dom'],
      solana: ['@solana/web3.js', '@solana/wallet-adapter-base'],
      charts: ['lightweight-charts', 'recharts'],
    },
  },
}

// Dynamic imports for code splitting
const ProMode = React.lazy(() => import('./pro/index'));
const TradingTab = React.lazy(() => import('./lite/TradingTab'));
```

### Lazy Loading Implementation
```typescript
// Route-based lazy loading
<Route path="/pro" element={
  <React.Suspense fallback={<div className="p-6 text-white">Loading Pro…</div>}>
    {React.createElement(React.lazy(() => import('./pro/index')))}
  </React.Suspense>
} />

// Component-based lazy loading
const LazyChart = React.lazy(() => import('./components/DexChart'));
```

### Caching Strategies
```typescript
// Service Worker caching
runtimeCaching: [{
  urlPattern: /^https:\/\/api\.quantdesk\.app\/.*/i,
  handler: 'NetworkFirst',
  options: {
    cacheName: 'api-cache',
    expiration: {
      maxEntries: 10,
      maxAgeSeconds: 60 * 60 * 24 // 24 hours
    }
  }
}]

// Memory caching in stores
class PriceStore {
  private prices = new Map<string, PriceData>();
  private cache = new Map<string, { data: any; timestamp: number }>();
  
  getCachedData(key: string, ttl: number = 300000): any {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < ttl) {
      return cached.data;
    }
    return null;
  }
}
```

## Deployment Architecture

### Vercel Deployment Configuration
```json
// vercel.json - Deployment settings
{
  "buildCommand": "cd frontend && pnpm run build",
  "outputDirectory": "frontend/dist",
  "installCommand": "pnpm install",
  "framework": "vite",
  "rewrites": [
    {
      "source": "/api/(.*)",
      "destination": "https://api.quantdesk.app/api/$1"
    }
  ],
  "headers": [
    {
      "source": "/(.*)",
      "headers": [
        {
          "key": "X-Content-Type-Options",
          "value": "nosniff"
        },
        {
          "key": "X-Frame-Options",
          "value": "DENY"
        },
        {
          "key": "X-XSS-Protection",
          "value": "1; mode=block"
        }
      ]
    }
  ]
}
```

### Build Process
```bash
# Development
pnpm run dev          # Start development server on port 3001
pnpm run build        # Build for production
pnpm run preview      # Preview production build
pnpm run type-check   # TypeScript type checking
pnpm run lint         # ESLint code analysis
pnpm run test         # Run Vitest tests
pnpm run test:e2e     # Run Playwright E2E tests
```

### Environment-Specific Configurations
```typescript
// Environment variables
VITE_API_URL=http://localhost:3002          # Development
VITE_API_URL=https://api.quantdesk.app     # Production
VITE_WS_URL=ws://localhost:3002            # Development WebSocket
VITE_WS_URL=wss://api.quantdesk.app        # Production WebSocket
VITE_SOLANA_RPC_URL=https://api.devnet.solana.com  # Solana RPC
VITE_SOLANA_NETWORK=devnet                 # Network environment
VITE_DEBUG=true                            # Debug mode
```

## Development Guidelines

### Component Development Standards
- **Functional Components**: Use functional components with TypeScript
- **Props Interfaces**: Define clear prop interfaces for type safety
- **Error Boundaries**: Implement error boundaries for error handling
- **Loading States**: Always handle loading states for async operations
- **Accessibility**: Follow WCAG 2.1 AA guidelines

### State Management Best Practices
- **Zustand for Global State**: Use Zustand for complex global state
- **React Context for Cross-Component**: Use Context for component tree state
- **Custom Hooks for Reusable Logic**: Extract reusable logic into custom hooks
- **Local State for Component-Specific**: Use useState for component-specific data

### Service Layer Patterns
- **Singleton Pattern**: Use singleton pattern for services
- **Async/Await**: Use async/await for API calls
- **Error Handling**: Implement comprehensive error handling with try-catch
- **TypeScript Interfaces**: Define interfaces for API responses

### Testing Standards
- **Test File Location**: `*.test.tsx` alongside components
- **E2E Tests**: `*.e2e.test.ts` in components directory
- **Service Tests**: `*.test.ts` in services directory
- **Arrange-Act-Assert**: Follow AAA testing pattern
- **Mock External Dependencies**: Mock API calls, WebSocket, Solana
- **Test User Interactions**: Test clicks, form submissions, navigation
- **Test Error States**: Test error handling and edge cases
- **Coverage Requirements**: Aim for 80% code coverage

## Integration Points

### Backend Integration
- **API Gateway**: REST + Socket.io WebSocket endpoints (proxy through Vite)
- **Authentication**: JWT + SIWS (Sign-In with Solana) integration
- **Real-time Data**: WebSocket connection for live updates
- **Error Handling**: Comprehensive error handling and retry logic

### Solana Integration
- **Wallet Connection**: Phantom & Solflare adapters via @solana/wallet-adapter ecosystem
- **Smart Contract Interaction**: Direct interaction via Anchor framework
- **RPC Connection**: Connection to Solana devnet/mainnet
- **Transaction Management**: Transaction building, signing, and monitoring

### External Services
- **Pyth Network**: Price feeds integration
- **Supabase**: Database operations and real-time subscriptions
- **Vercel**: Deployment and hosting
- **TradingView**: Chart integration and market data

## Performance Monitoring

### Performance Metrics
- **Bundle Size**: Monitor bundle size and chunk optimization
- **Load Time**: Track initial load time and time to interactive
- **Runtime Performance**: Monitor component render times
- **Memory Usage**: Track memory usage and potential leaks

### Optimization Strategies
- **Code Splitting**: Implement route-based and component-based code splitting
- **Lazy Loading**: Use React.lazy for component lazy loading
- **Memoization**: Use React.memo and useMemo for expensive computations
- **Virtual Scrolling**: Implement virtual scrolling for large lists
- **Image Optimization**: Optimize images and use appropriate formats

This comprehensive frontend architecture documentation provides a complete overview of the QuantDesk frontend system, based on actual codebase analysis and verified against the real implementation.