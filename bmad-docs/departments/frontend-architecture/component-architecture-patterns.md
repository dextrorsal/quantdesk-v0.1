# Component Architecture Patterns

## Functional Components with TypeScript
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

## Layout Component Pattern
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
