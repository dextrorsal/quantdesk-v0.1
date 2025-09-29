import { create } from 'zustand'
import { subscribeWithSelector } from 'zustand/middleware'

export interface Market {
  symbol: string
  name: string
  price: number
  change24h: number
  volume24h: number
  high24h: number
  low24h: number
}

export interface Position {
  id: string
  symbol: string
  side: 'long' | 'short'
  size: number
  entryPrice: number
  currentPrice: number
  pnl: number
  pnlPercentage: number
  margin: number
  leverage: number
}

export interface Order {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stop'
  size: number
  price?: number
  status: 'pending' | 'filled' | 'cancelled'
  timestamp: number
}

interface TradingState {
  // Markets
  markets: Market[]
  selectedMarket: Market | null
  
  // Positions
  positions: Position[]
  
  // Orders
  orders: Order[]
  
  // UI State
  isLoading: boolean
  error: string | null
  
  // Actions
  setMarkets: (markets: Market[]) => void
  setSelectedMarket: (market: Market | null) => void
  addPosition: (position: Position) => void
  updatePosition: (id: string, updates: Partial<Position>) => void
  removePosition: (id: string) => void
  addOrder: (order: Order) => void
  updateOrder: (id: string, updates: Partial<Order>) => void
  removeOrder: (id: string) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

export const useTradingStore = create<TradingState>()(
  subscribeWithSelector((set, _get) => ({
    // Initial state
    markets: [],
    selectedMarket: null,
    positions: [],
    orders: [],
    isLoading: false,
    error: null,

    // Actions
    setMarkets: (markets) => set({ markets }),
    setSelectedMarket: (market) => set({ selectedMarket: market }),
    
    addPosition: (position) => 
      set((state) => ({ positions: [...state.positions, position] })),
    
    updatePosition: (id, updates) =>
      set((state) => ({
        positions: state.positions.map((pos) =>
          pos.id === id ? { ...pos, ...updates } : pos
        ),
      })),
    
    removePosition: (id) =>
      set((state) => ({
        positions: state.positions.filter((pos) => pos.id !== id),
      })),
    
    addOrder: (order) =>
      set((state) => ({ orders: [...state.orders, order] })),
    
    updateOrder: (id, updates) =>
      set((state) => ({
        orders: state.orders.map((order) =>
          order.id === id ? { ...order, ...updates } : order
        ),
      })),
    
    removeOrder: (id) =>
      set((state) => ({
        orders: state.orders.filter((order) => order.id !== id),
      })),
    
    setLoading: (isLoading) => set({ isLoading }),
    setError: (error) => set({ error }),
  }))
)
