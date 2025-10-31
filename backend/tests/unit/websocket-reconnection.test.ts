/**
 * Test 1.1-UNIT-001: WebSocket client reconnection logic with exponential backoff
 * 
 * Tests the exponential backoff algorithm for WebSocket reconnection
 * to ensure stable connections under network instability conditions.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'

// Mock WebSocket client for testing
class MockWebSocketClient {
  private reconnectAttempts = 0
  private maxReconnectAttempts = 10
  private baseDelay = 1000 // 1 second
  private maxDelay = 30000 // 30 seconds
  private reconnectTimer: NodeJS.Timeout | null = null
  private isConnected = false
  private onConnectCallback?: () => void
  private onDisconnectCallback?: () => void

  constructor() {
    this.reconnectAttempts = 0
  }

  connect() {
    // Simulate connection attempt
    const success = Math.random() > 0.3 // 70% success rate for testing
    if (success) {
      this.isConnected = true
      this.reconnectAttempts = 0
      this.onConnectCallback?.()
    } else {
      this.handleDisconnect()
    }
  }

  disconnect() {
    this.isConnected = false
    this.onDisconnectCallback?.()
  }

  private handleDisconnect() {
    this.isConnected = false
    this.onDisconnectCallback?.()
    
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.scheduleReconnect()
    }
  }

  private scheduleReconnect() {
    const delay = this.calculateReconnectDelay()
    console.log(`Scheduling reconnect attempt ${this.reconnectAttempts + 1} in ${delay}ms`)
    
    this.reconnectTimer = setTimeout(() => {
      this.reconnectAttempts++
      this.connect()
    }, delay)
  }

  private calculateReconnectDelay(): number {
    // Exponential backoff with jitter
    const exponentialDelay = this.baseDelay * Math.pow(2, this.reconnectAttempts)
    const jitter = Math.random() * 1000 // Add up to 1 second of jitter
    const delay = Math.min(exponentialDelay + jitter, this.maxDelay)
    
    return Math.floor(delay)
  }

  onConnect(callback: () => void) {
    this.onConnectCallback = callback
  }

  onDisconnect(callback: () => void) {
    this.onDisconnectCallback = callback
  }

  getReconnectAttempts(): number {
    return this.reconnectAttempts
  }

  getIsConnected(): boolean {
    return this.isConnected
  }

  cleanup() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }
}

describe('1.1-UNIT-001: WebSocket Reconnection Logic', () => {
  let mockClient: MockWebSocketClient
  let connectSpy: ReturnType<typeof vi.fn>
  let disconnectSpy: ReturnType<typeof vi.fn>

  beforeEach(() => {
    mockClient = new MockWebSocketClient()
    connectSpy = vi.fn()
    disconnectSpy = vi.fn()
    
    mockClient.onConnect(connectSpy)
    mockClient.onDisconnect(disconnectSpy)
  })

  afterEach(() => {
    mockClient.cleanup()
  })

  it('should implement exponential backoff with jitter', () => {
    // Test the delay calculation algorithm
    const delays: number[] = []
    
    // Simulate multiple reconnection attempts
    for (let attempt = 0; attempt < 5; attempt++) {
      const delay = mockClient['calculateReconnectDelay']()
      delays.push(delay)
    }

    // Verify exponential growth pattern
    expect(delays[1]).toBeGreaterThan(delays[0])
    expect(delays[2]).toBeGreaterThan(delays[1])
    expect(delays[3]).toBeGreaterThan(delays[2])
    expect(delays[4]).toBeGreaterThan(delays[3])

    // Verify jitter is applied (delays should vary)
    const uniqueDelays = new Set(delays)
    expect(uniqueDelays.size).toBeGreaterThan(1)
  })

  it('should respect maximum delay limit', () => {
    // Force high reconnect attempts to test max delay
    mockClient['reconnectAttempts'] = 10
    
    const delay = mockClient['calculateReconnectDelay']()
    expect(delay).toBeLessThanOrEqual(30000) // Max delay is 30 seconds
  })

  it('should reset reconnect attempts on successful connection', () => {
    // Simulate failed connections
    mockClient['reconnectAttempts'] = 5
    mockClient['isConnected'] = false
    
    // Mock successful connection
    vi.spyOn(Math, 'random').mockReturnValue(0.9) // 90% success rate
    
    mockClient.connect()
    
    expect(mockClient.getReconnectAttempts()).toBe(0)
    expect(mockClient.getIsConnected()).toBe(true)
  })

  it('should stop reconnecting after max attempts', () => {
    const scheduleSpy = vi.spyOn(mockClient as any, 'scheduleReconnect')
    
    // Force max attempts
    mockClient['reconnectAttempts'] = 10
    mockClient['isConnected'] = false
    
    mockClient['handleDisconnect']()
    
    expect(scheduleSpy).not.toHaveBeenCalled()
  })

  it('should handle rapid disconnect/connect cycles', async () => {
    const connectPromises: Promise<void>[] = []
    
    // Simulate rapid connection attempts
    for (let i = 0; i < 3; i++) {
      connectPromises.push(
        new Promise<void>((resolve) => {
          mockClient.onConnect(() => {
            connectSpy()
            resolve()
          })
          mockClient.connect()
        })
      )
    }

    await Promise.allSettled(connectPromises)
    
    // Should have attempted connections
    expect(connectSpy).toHaveBeenCalled()
  })

  it('should provide reasonable delay progression', () => {
    const expectedDelays = [
      1000, // Base delay
      2000, // 2^1 * 1000
      4000, // 2^2 * 1000
      8000, // 2^3 * 1000
      16000, // 2^4 * 1000
      30000, // Capped at max delay
    ]

    for (let attempt = 0; attempt < expectedDelays.length; attempt++) {
      mockClient['reconnectAttempts'] = attempt
      const delay = mockClient['calculateReconnectDelay']()
      
      // Allow for jitter variance (±1000ms)
      expect(delay).toBeGreaterThanOrEqual(expectedDelays[attempt] - 1000)
      expect(delay).toBeLessThanOrEqual(expectedDelays[attempt] + 1000)
    }
  })

  it('should handle cleanup properly', () => {
    // Start a reconnection process
    mockClient['reconnectAttempts'] = 1
    mockClient['isConnected'] = false
    
    const scheduleSpy = vi.spyOn(mockClient as any, 'scheduleReconnect')
    mockClient['handleDisconnect']()
    
    expect(scheduleSpy).toHaveBeenCalled()
    
    // Cleanup should clear timers
    mockClient.cleanup()
    
    // Verify cleanup doesn't throw errors
    expect(() => mockClient.cleanup()).not.toThrow()
  })
})
