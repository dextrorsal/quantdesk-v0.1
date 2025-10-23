// Simplified WebSocket Provider for QuantDesk
// Provides WebSocket context without causing re-render issues

import React, { createContext, useContext, useEffect, useState, ReactNode, useMemo } from 'react'
import { io, Socket } from 'socket.io-client'

interface WebSocketContextType {
  isConnected: boolean
  reconnectAttempts: number
  subscribe: (channel: string, callback: (data: any) => void) => () => void
  unsubscribe: (channel: string, callback: (data: any) => void) => void
  subscribeToPortfolio: (userId: string, callback: (data: any) => void) => () => void
  marketData: Map<string, any>
  orderBookData: Map<string, any>
  tradeData: Map<string, any[]>
  positionData: Map<string, any[]>
  orderData: Map<string, any[]>
  portfolioData: Map<string, any>
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
  const [portfolioData, setPortfolioData] = useState<Map<string, any>>(new Map())

  // Socket.IO connection for live data
  useEffect(() => {
    let socket: Socket | null = null
    
    const connectSocket = () => {
      try {
        socket = io('http://localhost:3002', {
          transports: ['websocket'],
          timeout: 10000,
          reconnection: true,
          reconnectionAttempts: 5,
          reconnectionDelay: 3000
        })
        
        socket.on('connect', () => {
          console.log('✅ Connected to QuantDesk Socket.IO')
          setIsConnected(true)
          setReconnectAttempts(0)
          
          // Subscribe to market data
          socket?.emit('subscribe_market_data')
        })
        
        socket.on('market_data_update', (message) => {
          if (message.data?.symbol) {
            setMarketData(prev => {
              const newData = new Map(prev)
              newData.set(message.data.symbol, {
                price: message.data.price,
                change: message.data.change24h || 0,
                changePercent: message.data.change24h ? (message.data.change24h / message.data.price) * 100 : 0,
                volume: message.data.volume24h || 0,
                timestamp: message.timestamp
              })
              return newData
            })
          }
        })
        
        socket.on('market_update', (message) => {
          if (message.data?.symbol) {
            setMarketData(prev => {
              const newData = new Map(prev)
              newData.set(message.data.symbol, {
                price: message.data.price,
                change: message.data.change24h || 0,
                changePercent: message.data.change24h ? (message.data.change24h / message.data.price) * 100 : 0,
                volume: message.data.volume24h || 0,
                timestamp: message.timestamp
              })
              return newData
            })
          }
        })
        
        socket.on('trade_update', (message) => {
          if (message.data?.symbol) {
            setTradeData(prev => {
              const newData = new Map(prev)
              const trades = newData.get(message.data.symbol) || []
              trades.unshift(message.data)
              // Keep only last 100 trades
              if (trades.length > 100) trades.splice(100)
              newData.set(message.data.symbol, trades)
              return newData
            })
          }
        })
        
        socket.on('order_book_update', (message) => {
          if (message.data?.symbol) {
            setOrderBookData(prev => {
              const newData = new Map(prev)
              newData.set(message.data.symbol, message.data)
              return newData
            })
          }
        })
        
        // Add portfolio event handler
        socket.on('portfolio_update', (message) => {
          if (message.data?.userId) {
            setPortfolioData(prev => {
              const newData = new Map(prev)
              newData.set(message.data.userId, message.data)
              return newData
            })
          }
        })
        
        // 🚀 NEW: Add order status update handler
        socket.on('order_update', (message) => {
          console.log('📋 Order status update:', message.data)
          // Dispatch custom event for order updates
          window.dispatchEvent(new CustomEvent('orderStatusUpdate', {
            detail: message.data
          }))
        })
        
        // 🚀 NEW: Add position update handler
        socket.on('position_update', (message) => {
          console.log('📈 Position update:', message.data)
          // Dispatch custom event for position updates
          window.dispatchEvent(new CustomEvent('positionStatusUpdate', {
            detail: message.data
          }))
        })
        
        // 🚀 NEW: Add portfolio update handler
        socket.on('portfolio_update', (message) => {
          console.log('💰 Portfolio update:', message.data)
          // Dispatch custom event for portfolio updates
          window.dispatchEvent(new CustomEvent('portfolioStatusUpdate', {
            detail: message.data
          }))
        })
        
        socket.on('disconnect', () => {
          // console.log('Socket.IO disconnected')
          setIsConnected(false)
        })
        
        socket.on('connect_error', (error) => {
          // console.error('Socket.IO connection error:', error)
          setIsConnected(false)
        })
        
      } catch (error) {
        console.error('Failed to create Socket.IO connection:', error)
        setIsConnected(false)
      }
    }
    
    connectSocket()
    
    return () => {
      if (socket) {
        socket.disconnect()
      }
    }
  }, [reconnectAttempts])

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

  // Add portfolio subscription method
  const subscribeToPortfolio = useMemo(() => {
    return (userId: string, callback: (data: any) => void) => {
      // This would emit to the socket when connected
      console.log(`Subscribed to portfolio updates for user ${userId}`)
      return () => {
        console.log(`Unsubscribed from portfolio updates for user ${userId}`)
      }
    }
  }, [])

  // Memoized context value to prevent unnecessary re-renders
  const value: WebSocketContextType = useMemo(() => ({
    isConnected,
    reconnectAttempts,
    subscribe,
    unsubscribe,
    subscribeToPortfolio,
    marketData,
    orderBookData,
    tradeData,
    positionData,
    orderData,
    portfolioData,
  }), [isConnected, reconnectAttempts, subscribe, unsubscribe, subscribeToPortfolio, marketData, orderBookData, tradeData, positionData, orderData, portfolioData])

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