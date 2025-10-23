# State Management Architecture

## Zustand Stores
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

## React Context Providers
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
