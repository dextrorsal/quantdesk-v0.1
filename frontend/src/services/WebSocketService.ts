/**
 * WebSocket Service for Real-time Price Updates
 * Connects to backend WebSocket and manages real-time price data
 */

import { io, Socket } from 'socket.io-client'
import PriceStore, { PriceUpdate } from '../stores/PriceStore'

interface WebSocketConfig {
  url: string
  reconnectAttempts?: number
  reconnectDelay?: number
}

class WebSocketService {
  private socket: Socket | null = null
  private priceStore: PriceStore
  private config: WebSocketConfig
  private isConnecting = false
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectDelay = 1000

  constructor(config: WebSocketConfig) {
    this.config = {
      reconnectAttempts: 5,
      reconnectDelay: 1000,
      ...config
    }
    this.priceStore = PriceStore.getInstance()
    this.maxReconnectAttempts = this.config.reconnectAttempts || 5
    this.reconnectDelay = this.config.reconnectDelay || 1000
  }

  /**
   * Connect to WebSocket server
   */
  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      if (this.socket?.connected) {
        resolve()
        return
      }

      if (this.isConnecting) {
        // Instead of rejecting, wait for the current connection to complete
        console.log('⏳ Connection already in progress, waiting...')
        const checkConnection = () => {
          if (this.socket?.connected) {
            resolve()
          } else if (!this.isConnecting) {
            // If connection failed, try again
            this.connect().then(resolve).catch(reject)
          } else {
            // Still connecting, check again in 100ms
            setTimeout(checkConnection, 100)
          }
        }
        checkConnection()
        return
      }

      this.isConnecting = true
      console.log(`🔌 Connecting to WebSocket: ${this.config.url}`)

      try {
        this.socket = io(this.config.url, {
          transports: ['polling', 'websocket'],
          timeout: 10000,
          forceNew: true,
          autoConnect: true,
          reconnection: true,
          reconnectionAttempts: 5,
          reconnectionDelay: 1000
        })

        this.setupEventHandlers()

        this.socket.on('connect', () => {
          console.log('✅ WebSocket connected successfully')
          this.isConnecting = false
          this.reconnectAttempts = 0
          this.priceStore.setConnectionStatus(true)
          resolve()
        })

        this.socket.on('connect_error', (error) => {
          console.error('❌ WebSocket connection error:', error)
          this.isConnecting = false
          this.priceStore.setConnectionStatus(false)
          reject(error)
        })

        this.socket.on('disconnect', (reason) => {
          console.log(`🔌 WebSocket disconnected: ${reason}`)
          this.priceStore.setConnectionStatus(false)
          this.handleReconnect()
        })

      } catch (error) {
        this.isConnecting = false
        reject(error)
      }
    })
  }

  /**
   * Setup WebSocket event handlers
   */
  private setupEventHandlers() {
    if (!this.socket) return

    // Handle price updates
    this.socket.on('price_update', (data: any) => {
      try {
        const priceUpdates: PriceUpdate[] = Array.isArray(data) ? data : [data]
        this.priceStore.updatePrices(priceUpdates)
        console.log(`💰 Received ${priceUpdates.length} price updates`)
      } catch (error) {
        console.error('Error processing price update:', error)
      }
    })

    // Handle market data updates
    this.socket.on('market_data', (data: any) => {
      try {
        if (data.prices && Array.isArray(data.prices)) {
          this.priceStore.updatePrices(data.prices)
          console.log(`📊 Received market data: ${data.prices.length} prices`)
        }
      } catch (error) {
        console.error('Error processing market data:', error)
      }
    })

    // Handle connection status
    this.socket.on('connect', () => {
      console.log('🟢 WebSocket connected')
      this.priceStore.setConnectionStatus(true)
    })

    this.socket.on('disconnect', () => {
      console.log('🔴 WebSocket disconnected')
      this.priceStore.setConnectionStatus(false)
    })

    // Subscribe to market data on connection
    this.socket.on('connect', () => {
      this.subscribeToMarketData()
    })
  }

  /**
   * Subscribe to market data
   */
  private subscribeToMarketData() {
    if (!this.socket?.connected) return

    console.log('📡 Subscribing to market data...')
    this.socket.emit('subscribe_market_data')
  }

  /**
   * Handle reconnection logic
   */
  private handleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error(`❌ Max reconnection attempts (${this.maxReconnectAttempts}) reached`)
      return
    }

    this.reconnectAttempts++
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1) // Exponential backoff

    console.log(`🔄 Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`)

    setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnection failed:', error)
      })
    }, delay)
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect() {
    if (this.socket) {
      console.log('🔌 Disconnecting WebSocket...')
      this.socket.disconnect()
      this.socket = null
    }
    this.priceStore.setConnectionStatus(false)
  }

  /**
   * Check if connected
   */
  isConnected(): boolean {
    return this.socket?.connected || false
  }

  /**
   * Get connection status
   */
  getConnectionStatus(): 'connecting' | 'connected' | 'disconnected' {
    if (this.isConnecting) return 'connecting'
    if (this.socket?.connected) return 'connected'
    return 'disconnected'
  }

  /**
   * Send custom message
   */
  send(event: string, data?: any) {
    if (this.socket?.connected) {
      this.socket.emit(event, data)
    } else {
      console.warn('Cannot send message: WebSocket not connected')
    }
  }

  /**
   * Subscribe to specific symbol
   */
  subscribeToSymbol(symbol: string) {
    this.send('subscribe_symbol', { symbol })
  }

  /**
   * Unsubscribe from specific symbol
   */
  unsubscribeFromSymbol(symbol: string) {
    this.send('unsubscribe_symbol', { symbol })
  }
}

export default WebSocketService
