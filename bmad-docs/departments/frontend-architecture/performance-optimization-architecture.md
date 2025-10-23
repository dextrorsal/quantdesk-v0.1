# Performance Optimization Architecture

## Bundle Optimization Strategies
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

## Lazy Loading Implementation
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

## Caching Strategies
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
