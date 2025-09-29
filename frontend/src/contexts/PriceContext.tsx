/**
 * Price Context - React Context for accessing price data throughout the app
 * Provides centralized access to real-time price information
 */

import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react'
import PriceStore, { PriceData } from '../stores/PriceStore'
import WebSocketService from '../services/WebSocketService'

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
  websocketUrl = 'http://localhost:3002',
  fallbackApiUrl = '/api/prices'
}) => {
  const [prices, setPrices] = useState<Map<string, PriceData>>(new Map())
  const [isConnected, setIsConnected] = useState(false)
  const [lastUpdate, setLastUpdate] = useState(0)
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected')
  
  const priceStore = PriceStore.getInstance()
  const webSocketService = new WebSocketService({ url: websocketUrl })

  useEffect(() => {
    // Subscribe to price updates
    const unsubscribe = priceStore.subscribe((newPrices) => {
      setPrices(newPrices)
      setIsConnected(priceStore.getConnectionStatus())
      setLastUpdate(priceStore.getLastUpdate())
      setConnectionStatus(priceStore.getConnectionStatus() ? 'connected' : 'disconnected')
    })

    // Initialize connection - Skip WebSocket for now
    const initializeConnection = async () => {
      try {
        setConnectionStatus('connecting')
        console.log('⚠️ WebSocket disabled, using API polling only')
        
        // Skip WebSocket, go straight to API polling
        setConnectionStatus('disconnected')
        
        // Use API polling only
        priceStore.startFallbackPolling(fallbackApiUrl, 5000) // 5 second polling
        
      } catch (error) {
        console.error('API polling failed:', error)
        setConnectionStatus('disconnected')
      }
    }

    initializeConnection()

    // Cleanup on unmount
    return () => {
      unsubscribe()
      // webSocketService.disconnect() // Disabled
      priceStore.stopFallbackPolling()
    }
  }, [websocketUrl, fallbackApiUrl])

  // Update connection status from WebSocket service
  useEffect(() => {
    const interval = setInterval(() => {
      const wsStatus = webSocketService.getConnectionStatus()
      setConnectionStatus(wsStatus)
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
