// Simplified WebSocket Provider for QuantDesk
// Provides WebSocket context without causing re-render issues

import React, { createContext, useContext, useEffect, useState, ReactNode, useMemo } from 'react'

interface WebSocketContextType {
  isConnected: boolean
  reconnectAttempts: number
  subscribe: (channel: string, callback: (data: any) => void) => () => void
  unsubscribe: (channel: string, callback: (data: any) => void) => void
  marketData: Map<string, any>
  orderBookData: Map<string, any>
  tradeData: Map<string, any[]>
  positionData: Map<string, any[]>
  orderData: Map<string, any[]>
}

const WebSocketContext = createContext<WebSocketContextType | undefined>(undefined)

interface WebSocketProviderProps {
  children: ReactNode
}

export const WebSocketProvider: React.FC<WebSocketProviderProps> = ({ children }) => {
  const [isConnected, setIsConnected] = useState(false)
  const [reconnectAttempts, setReconnectAttempts] = useState(0)
  const [marketData, setMarketData] = useState<Map<string, any>>(new Map())
  const [orderBookData, setOrderBookData] = useState<Map<string, any>>(new Map())
  const [tradeData, setTradeData] = useState<Map<string, any[]>>(new Map())
  const [positionData, setPositionData] = useState<Map<string, any[]>>(new Map())
  const [orderData, setOrderData] = useState<Map<string, any[]>>(new Map())

  // Simplified WebSocket connection - no constant polling
  useEffect(() => {
    // Mock connection for now - no actual WebSocket connection
    setIsConnected(true)
    setReconnectAttempts(0)
    
    // Cleanup
    return () => {
      setIsConnected(false)
    }
  }, [])

  // Memoized subscribe function
  const subscribe = useMemo(() => {
    return (channel: string, callback: (data: any) => void) => {
      // Mock subscription - no actual WebSocket
      console.log(`Subscribed to ${channel}`)
      return () => {
        console.log(`Unsubscribed from ${channel}`)
      }
    }
  }, [])

  // Memoized unsubscribe function
  const unsubscribe = useMemo(() => {
    return (channel: string, callback: (data: any) => void) => {
      console.log(`Unsubscribed from ${channel}`)
    }
  }, [])

  // Memoized context value to prevent unnecessary re-renders
  const value: WebSocketContextType = useMemo(() => ({
    isConnected,
    reconnectAttempts,
    subscribe,
    unsubscribe,
    marketData,
    orderBookData,
    tradeData,
    positionData,
    orderData,
  }), [isConnected, reconnectAttempts, subscribe, unsubscribe, marketData, orderBookData, tradeData, positionData, orderData])

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  )
}

export const useWebSocket = (): WebSocketContextType => {
  const context = useContext(WebSocketContext)
  if (context === undefined) {
    throw new Error('useWebSocket must be used within a WebSocketProvider')
  }
  return context
}

export default WebSocketProvider