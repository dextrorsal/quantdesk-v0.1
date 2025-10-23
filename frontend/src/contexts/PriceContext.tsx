/**
 * Price Context - React Context for accessing price data throughout the app
 * Provides centralized access to real-time price information
 */

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import PriceStore, { PriceData } from '../stores/PriceStore'
import { WebSocketService } from '../services/websocketService'

interface PriceContextType {
  prices: Map<string, PriceData>
  isConnected: boolean
  lastUpdate: number
  connectionStatus: 'connecting' | 'connected' | 'disconnected'
  getPrice: (symbol: string) => PriceData | undefined
  subscribeToSymbol: (symbol: string) => void
  unsubscribeFromSymbol: (symbol: string) => void
}

const PriceContext = createContext<PriceContextType | undefined>(undefined)

interface PriceProviderProps {
  children: ReactNode
  websocketUrl?: string
  fallbackApiUrl?: string
}

export const PriceProvider: React.FC<PriceProviderProps> = ({ 
  children, 
  websocketUrl = 'ws://localhost:3002/',
  fallbackApiUrl = '/api/prices'
}) => {
  const [prices, setPrices] = useState<Map<string, PriceData>>(new Map())
  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState(0)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected')
  
  const priceStore = PriceStore.getInstance()
  const webSocketService = new WebSocketService(websocketUrl, true) // Make WebSocket optional

  useEffect(() => {
    // Subscribe to price updates
    const unsubscribe = priceStore.subscribe((newPrices) => {
      setPrices(newPrices)
      setIsConnected(priceStore.getConnectionStatus())
      setLastUpdate(priceStore.getLastUpdate())
      setConnectionStatus(priceStore.getConnectionStatus() ? 'connected' : 'disconnected')
    })

    // Initialize connection - Try WebSocket first, fallback to API polling
    const initializeConnection = async () => {
      try {
        setConnectionStatus('connecting')
        console.log('🔄 Attempting WebSocket connection for real-time price updates...')
        
        // Try WebSocket connection (now optional)
        await webSocketService.connect()
        
        // Check if WebSocket is actually connected
        if (webSocketService.getConnectionStatus()) {
          setConnectionStatus('connected')
          console.log('✅ WebSocket connected successfully')
          
          // Subscribe to market data updates
          webSocketService.subscribe('market_data_update', (data) => {
            if (data?.data) {
              const marketData = data.data
              priceStore.updatePrice(marketData.symbol, {
                symbol: marketData.symbol,
                price: marketData.price,
                change: marketData.change24h,
                changePercent: marketData.changePercent24h,
                timestamp: marketData.lastUpdate,
                confidence: marketData.confidence
              })
            }
          })
          
          // Subscribe to order book updates
          webSocketService.subscribe('order_book_update', (data) => {
            // Handle order book updates if needed
            console.log('📊 Order book update:', data)
          })
          
          // Subscribe to trade updates
          webSocketService.subscribe('trade_update', (data) => {
            // Handle trade updates if needed
            console.log('💰 Trade update:', data)
          })
          
        } else {
          // WebSocket failed but didn't throw error (optional mode)
          console.log('📡 WebSocket unavailable, using API polling fallback')
          setConnectionStatus('disconnected')
          priceStore.startFallbackPolling(fallbackApiUrl, 5000)
        }
        
      } catch (error) {
        console.warn('⚠️ WebSocket connection failed, falling back to API polling:', error)
        setConnectionStatus('disconnected')
        priceStore.startFallbackPolling(fallbackApiUrl, 5000)
      }
    }

    initializeConnection()

    // Cleanup on unmount
    return () => {
      unsubscribe()
      webSocketService.disconnect()
      priceStore.stopFallbackPolling()
    }
  }, [websocketUrl, fallbackApiUrl])

  // Update connection status from WebSocket service
  useEffect(() => {
    const interval = setInterval(() => {
      const wsInfo = webSocketService.getConnectionInfo()
      if (wsInfo.connected) {
        setConnectionStatus('connected')
      } else if (wsInfo.failed) {
        setConnectionStatus('disconnected')
      } else {
        setConnectionStatus('connecting')
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [])

  const getPrice = (symbol: string): PriceData | undefined => {
    return priceStore.getPrice(symbol)
  }

  const subscribeToSymbol = (symbol: string) => {
    webSocketService.subscribeToSymbol(symbol)
  }

  const unsubscribeFromSymbol = (symbol: string) => {
    webSocketService.unsubscribeFromSymbol(symbol)
  }

  const contextValue: PriceContextType = {
    prices,
    isConnected,
    lastUpdate,
    connectionStatus,
    getPrice,
    subscribeToSymbol,
    unsubscribeFromSymbol
  }

  return (
    <PriceContext.Provider value={contextValue}>
      {children}
    </PriceContext.Provider>
  )
}

/**
 * Hook to use price context
 */
export const usePrice = (): PriceContextType => {
  const context = useContext(PriceContext)
  if (context === undefined) {
    throw new Error('usePrice must be used within a PriceProvider')
  }
  return context
}

/**
 * Hook to get price for a specific symbol
 */
export const usePriceData = (symbol: string): PriceData | undefined => {
  const { getPrice } = usePrice()
  return getPrice(symbol)
}

/**
 * Hook to get multiple prices
 */
export const usePrices = (symbols: string[]): Map<string, PriceData> => {
  const { prices } = usePrice()
  const filteredPrices = new Map<string, PriceData>()
  
  symbols.forEach(symbol => {
    const price = prices.get(symbol)
    if (price) {
      filteredPrices.set(symbol, price)
    }
  })
  
  return filteredPrices
}

export default PriceContext
