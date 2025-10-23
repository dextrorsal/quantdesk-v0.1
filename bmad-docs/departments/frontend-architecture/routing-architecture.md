# Routing Architecture

## React Router Implementation
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

## Lazy Loading Implementation
```typescript
// Lazy loading for performance optimization
<Route path="/pro" element={
  <React.Suspense fallback={<div className="p-6 text-white">Loading Pro…</div>}>
    {React.createElement(React.lazy(() => import('./pro/index')))}
  </React.Suspense>
} />
```
