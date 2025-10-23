# QuantDesk Frontend Architecture Document

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|---------|
| 2025-10-19 | 1.0 | Initial frontend architecture document | Winston (Architect) |

## Frontend Tech Stack

### Technology Stack Table

| Category | Technology | Version | Purpose | Rationale |
|----------|------------|---------|---------|------------|
| Framework | React | 18.2.0 | Core UI framework | Industry standard, excellent ecosystem, strong TypeScript support |
| UI Library | React Bootstrap | 2.10.10 | Component library | Consistent design system, accessibility features |
| State Management | Zustand | 4.4.7 | Global state | Lightweight, TypeScript-first, simple API |
| Routing | React Router | 6.30.1 | Client-side routing | Industry standard, excellent TypeScript support |
| Build Tool | Vite | 7.1.8 | Build system | Fast HMR, optimized builds, modern tooling |
| Styling | Tailwind CSS | 3.3.6 | Utility-first CSS | Rapid development, consistent design system |
| Testing | Vitest | 3.2.4 | Unit testing | Fast, Jest-compatible, Vite integration |
| Component Library | Custom + Heroicons | 1.0.6 | Icon system | Consistent iconography, tree-shaking |
| Form Handling | React Hook Form | N/A | Form management | Performance optimized, validation support |
| Animation | React Spring | 9.7.5 | Animations | Physics-based animations, smooth transitions |
| Dev Tools | ESLint + TypeScript | Latest | Code quality | Type safety, code standards enforcement |
| Charts | Lightweight Charts | 4.2.3 | Financial charts | High-performance, trading-focused |
| WebSocket | Socket.io Client | 4.8.1 | Real-time data | Reliable real-time communication |
| PWA | Vite PWA Plugin | 1.0.3 | Progressive Web App | Offline support, app-like experience |

## Project Structure

```
frontend/
├── public/                     # Static assets
│   ├── logos/                 # Token logos and icons
│   ├── tradingview/           # TradingView custom styles
│   └── *.png                  # App icons and banners
├── src/
│   ├── components/            # Reusable UI components
│   │   ├── charts/           # Chart components
│   │   ├── Header.tsx         # Main navigation
│   │   ├── Layout.tsx         # App layout wrapper
│   │   ├── Sidebar.tsx        # Navigation sidebar
│   │   └── *.tsx             # Other components
│   ├── contexts/             # React contexts
│   │   ├── AccountContext.tsx # Account state management
│   │   ├── MarketContext.tsx  # Market data context
│   │   ├── PriceContext.tsx   # Price data context
│   │   ├── ProgramContext.tsx # Solana program context
│   │   ├── TabContext.tsx     # Tab navigation context
│   │   └── ThemeContext.tsx   # Theme management
│   ├── hooks/                # Custom React hooks
│   ├── lib/                  # External library integrations
│   │   ├── dydxfeed/         # dYdX data feed
│   │   ├── launchableMarketFeed/ # Launchable market data
│   │   └── spotDatafeed/     # Spot market data
│   ├── lite/                 # Lite mode components
│   ├── pages/                # Page components
│   ├── pro/                  # Pro mode components
│   ├── providers/            # Context providers
│   ├── services/             # Business logic services
│   │   ├── smartContractService.ts # Solana smart contract integration
│   │   ├── websocketService.ts    # WebSocket management
│   │   ├── tradingService.ts      # Trading operations
│   │   └── *.ts             # Other services
│   ├── stores/               # Zustand stores
│   │   ├── PriceStore.ts     # Price data store
│   │   └── tradingStore.ts   # Trading state store
│   ├── types/                # TypeScript type definitions
│   ├── utils/                # Utility functions
│   ├── App.tsx               # Main app component
│   ├── main.tsx              # App entry point
│   └── index.css             # Global styles
├── package.json              # Dependencies and scripts
├── vite.config.ts           # Vite configuration
├── tailwind.config.js       # Tailwind configuration
├── tsconfig.json            # TypeScript configuration
└── vercel.json              # Deployment configuration
```

## Component Standards

### Component Template

```typescript
import React, { useState, useEffect, useCallback } from 'react';
import { useAccount } from '../contexts/AccountContext';
import { usePrice } from '../contexts/PriceContext';

interface ComponentProps {
  title: string;
  onAction?: (data: any) => void;
  className?: string;
  children?: React.ReactNode;
}

const Component: React.FC<ComponentProps> = ({ 
  title, 
  onAction, 
  className = '', 
  children 
}) => {
  const { accountState, loading } = useAccount();
  const { prices } = usePrice();
  const [localState, setLocalState] = useState<string>('');

  const handleAction = useCallback((data: any) => {
    if (onAction) {
      onAction(data);
    }
  }, [onAction]);

  useEffect(() => {
    // Side effects here
  }, []);

  if (loading) {
    return <div className="animate-pulse">Loading...</div>;
  }

  return (
    <div className={`component-container ${className}`}>
      <h2 className="text-xl font-semibold">{title}</h2>
      {children}
    </div>
  );
};

export default Component;
```

### Naming Conventions

- **Components**: PascalCase (e.g., `TradingInterface`, `PortfolioDashboard`)
- **Files**: PascalCase for components, camelCase for utilities (e.g., `TradingInterface.tsx`, `formatPrice.ts`)
- **Hooks**: camelCase with `use` prefix (e.g., `useAccount`, `useTrading`)
- **Services**: camelCase with `Service` suffix (e.g., `smartContractService`, `websocketService`)
- **Stores**: camelCase with `Store` suffix (e.g., `priceStore`, `tradingStore`)
- **Types**: PascalCase with descriptive names (e.g., `UserAccountState`, `TradingOrder`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `API_ENDPOINTS`, `DEFAULT_CONFIG`)

## State Management

### Store Structure

```
src/stores/
├── PriceStore.ts          # Centralized price data management
├── tradingStore.ts        # Trading state and operations
└── index.ts              # Store exports and initialization
```

### State Management Template

```typescript
import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

interface StateType {
  // State properties
  data: DataType[];
  loading: boolean;
  error: string | null;
  
  // Actions
  setData: (data: DataType[]) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  reset: () => void;
}

export const useStore = create<StateType>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    data: [],
    loading: false,
    error: null,

    // Actions
    setData: (data) => set({ data }),
    setLoading: (loading) => set({ loading }),
    setError: (error) => set({ error }),
    reset: () => set({ data: [], loading: false, error: null }),
  }))
);
```

## API Integration

### Service Template

```typescript
import axios, { AxiosInstance, AxiosResponse } from 'axios';

interface ApiResponse<T> {
  success: boolean;
  data: T;
  error?: string;
}

class ApiService {
  private client: AxiosInstance;

  constructor(baseURL: string) {
    this.client = axios.create({
      baseURL,
      timeout: 10000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add auth token if available
        const token = localStorage.getItem('auth_token');
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response: AxiosResponse) => response,
      (error) => {
        if (error.response?.status === 401) {
          // Handle unauthorized access
          localStorage.removeItem('auth_token');
          window.location.href = '/login';
        }
        return Promise.reject(error);
      }
    );
  }

  async get<T>(url: string): Promise<ApiResponse<T>> {
    try {
      const response = await this.client.get(url);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  async post<T>(url: string, data: any): Promise<ApiResponse<T>> {
    try {
      const response = await this.client.post(url, data);
      return response.data;
    } catch (error) {
      throw this.handleError(error);
    }
  }

  private handleError(error: any): Error {
    if (error.response) {
      return new Error(error.response.data.error || 'API request failed');
    }
    return new Error('Network error');
  }
}

export default ApiService;
```

### API Client Configuration

```typescript
// apiClient.ts
import ApiService from './ApiService';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3002';

export const apiClient = new ApiService(API_BASE_URL);

// Specific API endpoints
export const tradingApi = {
  getPositions: () => apiClient.get('/api/trading/positions'),
  placeOrder: (orderData: any) => apiClient.post('/api/trading/orders', orderData),
  getOrders: () => apiClient.get('/api/trading/orders'),
};

export const marketApi = {
  getMarkets: () => apiClient.get('/api/markets'),
  getMarketData: (symbol: string) => apiClient.get(`/api/markets/${symbol}`),
};
```

## Routing

### Route Configuration

```typescript
// App.tsx routing configuration
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Suspense, lazy } from 'react';

// Lazy load components for better performance
const TradingInterface = lazy(() => import('./components/TradingInterface'));
const PortfolioPage = lazy(() => import('./pages/PortfolioPage'));
const MarketsPage = lazy(() => import('./pages/MarketsPage'));

function AppRoutes() {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/" element={<LandingPage />} />
      
      {/* Protected routes */}
      <Route path="/lite" element={
        <Layout>
          <Suspense fallback={<div>Loading...</div>}>
            <LiteRouter />
          </Suspense>
        </Layout>
      } />
      
      <Route path="/pro" element={
        <Layout>
          <Suspense fallback={<div>Loading Pro...</div>}>
            <TradingInterface />
          </Suspense>
        </Layout>
      } />
      
      <Route path="/trading/:symbol" element={
        <Layout>
          <Suspense fallback={<div>Loading Trading...</div>}>
            <TradingInterface />
          </Suspense>
        </Layout>
      } />
      
      <Route path="/portfolio" element={
        <Layout>
          <PortfolioPage />
        </Layout>
      } />
      
      <Route path="/markets" element={
        <Layout>
          <MarketsPage />
        </Layout>
      } />
      
      {/* Admin routes */}
      <Route path="/admin/*" element={<AdminRedirect />} />
      
      {/* Catch-all route */}
      <Route path="*" element={<LandingPage />} />
    </Routes>
  );
}
```

## Styling Guidelines

### Styling Approach

QuantDesk uses **Tailwind CSS** as the primary styling solution with a custom design system:

- **Utility-first approach**: Rapid development with consistent spacing, colors, and typography
- **Custom theme system**: Dual theme support (Lite/Pro) with CSS custom properties
- **Responsive design**: Mobile-first approach with custom breakpoints
- **Component composition**: Reusable utility classes for common patterns

### Global Theme Variables

```css
/* index.css - Global theme system */
:root {
  /* Lite Theme Colors (Blue) */
  --lite-primary-50: #eff6ff;
  --lite-primary-100: #dbeafe;
  --lite-primary-200: #bfdbfe;
  --lite-primary-300: #93c5fd;
  --lite-primary-400: #60a5fa;
  --lite-primary-500: #3b82f6;
  --lite-primary-600: #2563eb;
  --lite-primary-700: #1d4ed8;
  --lite-primary-800: #1e40af;
  --lite-primary-900: #1e3a8a;

  /* Pro Theme Colors (Orange) */
  --pro-primary-50: #fff7ed;
  --pro-primary-100: #ffedd5;
  --pro-primary-200: #fed7aa;
  --pro-primary-300: #fdba74;
  --pro-primary-400: #fb923c;
  --pro-primary-500: #f97316;
  --pro-primary-600: #ea580c;
  --pro-primary-700: #c2410c;
  --pro-primary-800: #9a3412;
  --pro-primary-900: #7c2d12;

  /* Universal Colors */
  --primary-50: var(--lite-primary-50);
  --primary-100: var(--lite-primary-100);
  --primary-200: var(--lite-primary-200);
  --primary-300: var(--lite-primary-300);
  --primary-400: var(--lite-primary-400);
  --primary-500: var(--lite-primary-500);
  --primary-600: var(--lite-primary-600);
  --primary-700: var(--lite-primary-700);
  --primary-800: var(--lite-primary-800);
  --primary-900: var(--lite-primary-900);

  /* Background Colors */
  --bg-primary: #000000;
  --bg-secondary: #111111;
  --bg-tertiary: #1a1a1a;
  
  /* Text Colors */
  --text-primary: #ffffff;
  --text-secondary: #a0a0a0;
  --text-muted: #666666;

  /* Success/Danger/Warning */
  --success-500: #22c55e;
  --danger-500: #ef4444;
  --warning-500: #f59e0b;

  /* Spacing */
  --spacing-xs: 0.25rem;
  --spacing-sm: 0.5rem;
  --spacing-md: 1rem;
  --spacing-lg: 1.5rem;
  --spacing-xl: 2rem;

  /* Typography */
  --font-family-sans: 'Inter', system-ui, sans-serif;
  --font-family-mono: 'JetBrains Mono', monospace;

  /* Shadows */
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
  --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
  --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
}

/* Pro theme override */
.pro-theme {
  --primary-50: var(--pro-primary-50);
  --primary-100: var(--pro-primary-100);
  --primary-200: var(--pro-primary-200);
  --primary-300: var(--pro-primary-300);
  --primary-400: var(--pro-primary-400);
  --primary-500: var(--pro-primary-500);
  --primary-600: var(--pro-primary-600);
  --primary-700: var(--pro-primary-700);
  --primary-800: var(--pro-primary-800);
  --primary-900: var(--pro-primary-900);
}
```

## Testing Requirements

### Component Test Template

```typescript
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { AccountProvider } from '../contexts/AccountContext';
import { PriceProvider } from '../contexts/PriceContext';
import Component from './Component';

// Mock external dependencies
vi.mock('../services/apiClient', () => ({
  apiClient: {
    get: vi.fn(),
    post: vi.fn(),
  },
}));

const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <AccountProvider>
    <PriceProvider websocketUrl="ws://localhost:3002" fallbackApiUrl="/api/prices">
      {children}
    </PriceProvider>
  </AccountProvider>
);

describe('Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders with title', () => {
    render(
      <MockProviders>
        <Component title="Test Component" />
      </MockProviders>
    );
    
    expect(screen.getByText('Test Component')).toBeInTheDocument();
  });

  it('handles user interaction', async () => {
    const mockAction = vi.fn();
    
    render(
      <MockProviders>
        <Component title="Test Component" onAction={mockAction} />
      </MockProviders>
    );
    
    const button = screen.getByRole('button');
    fireEvent.click(button);
    
    await waitFor(() => {
      expect(mockAction).toHaveBeenCalled();
    });
  });

  it('displays loading state', () => {
    // Mock loading state
    render(
      <MockProviders>
        <Component title="Test Component" />
      </MockProviders>
    );
    
    expect(screen.getByText('Loading...')).toBeInTheDocument();
  });
});
```

### Testing Best Practices

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions with contexts and services
3. **E2E Tests**: Test critical user flows (using Playwright)
4. **Coverage Goals**: Aim for 80% code coverage
5. **Test Structure**: Arrange-Act-Assert pattern
6. **Mock External Dependencies**: API calls, routing, state management
7. **Test User Interactions**: Click events, form submissions, navigation
8. **Test Error States**: Network failures, validation errors, edge cases
9. **Test Accessibility**: Screen reader compatibility, keyboard navigation
10. **Test Performance**: Component rendering speed, memory usage

## Environment Configuration

### Required Environment Variables

```bash
# API Configuration
VITE_API_URL=http://localhost:3002
VITE_WS_URL=ws://localhost:3002

# Solana Configuration
VITE_SOLANA_RPC_URL=https://api.devnet.solana.com
VITE_SOLANA_NETWORK=devnet

# Development
VITE_DEBUG=true

# Production
VITE_APP_VERSION=1.0.0
VITE_BUILD_TIMESTAMP=2024-12-19T00:00:00Z
```

### Environment-Specific Configurations

- **Development**: Hot reloading, debug logging, devnet Solana
- **Staging**: Production-like environment, testnet Solana
- **Production**: Optimized builds, mainnet Solana, error tracking

## Frontend Developer Standards

### Critical Coding Rules

1. **TypeScript First**: All components must be fully typed
2. **Component Composition**: Prefer composition over inheritance
3. **Error Boundaries**: Wrap components in error boundaries
4. **Performance**: Use React.memo, useMemo, useCallback appropriately
5. **Accessibility**: All interactive elements must be accessible
6. **Responsive Design**: Mobile-first approach required
7. **State Management**: Use Zustand for global state, local state for component-specific data
8. **API Integration**: Always handle loading and error states
9. **Security**: Sanitize user inputs, validate data
10. **Testing**: Write tests for all new components and features

### Quick Reference

#### Common Commands
```bash
# Development
pnpm run dev              # Start development server
pnpm run build            # Build for production
pnpm run preview          # Preview production build

# Testing
pnpm run test             # Run unit tests
pnpm run test:run         # Run tests once
pnpm run test:e2e         # Run E2E tests

# Code Quality
pnpm run lint             # Run ESLint
pnpm run lint:fix         # Fix ESLint issues
pnpm run type-check       # TypeScript type checking
```

#### Key Import Patterns
```typescript
// React imports
import React, { useState, useEffect, useCallback } from 'react';

// Context imports
import { useAccount } from '../contexts/AccountContext';
import { usePrice } from '../contexts/PriceContext';

// Service imports
import { tradingApi } from '../services/apiClient';
import SmartContractService from '../services/smartContractService';

// Store imports
import { useTradingStore } from '../stores/tradingStore';
import PriceStore from '../stores/PriceStore';

// Component imports
import Layout from '../components/Layout';
import Header from '../components/Header';
```

#### File Naming Conventions
- **Components**: `ComponentName.tsx`
- **Pages**: `PageName.tsx`
- **Services**: `serviceName.ts`
- **Stores**: `storeName.ts`
- **Types**: `types.ts` or `ComponentName.types.ts`
- **Tests**: `ComponentName.test.tsx`

#### Project-Specific Patterns
- **Context Pattern**: Use React contexts for cross-component state
- **Service Pattern**: Centralized business logic in services
- **Store Pattern**: Zustand stores for global state
- **Hook Pattern**: Custom hooks for reusable logic
- **Component Pattern**: Functional components with TypeScript
- **Error Handling**: Try-catch blocks with user-friendly error messages
- **Loading States**: Always show loading indicators for async operations
- **Theme System**: Use CSS custom properties for theming
