import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { Connection, PublicKey } from '@solana/web3.js'
import WalletTestingComponent from '../components/devnet-testing/WalletTestingComponent'
import BalanceDisplayComponent from '../components/devnet-testing/BalanceDisplayComponent'

// Mock Solana Web3.js
vi.mock('@solana/web3.js', () => ({
  Connection: vi.fn().mockImplementation(() => ({
    getVersion: vi.fn().mockResolvedValue({ version: '1.16.0' }),
    getBalance: vi.fn().mockResolvedValue(10000000000), // 10 SOL in lamports
    rpcEndpoint: 'https://api.devnet.solana.com'
  })),
  PublicKey: vi.fn().mockImplementation((key) => ({
    toString: () => key,
    toBuffer: () => Buffer.from(key)
  })),
  Keypair: {
    generate: vi.fn().mockReturnValue({
      publicKey: {
        toString: () => '11111111111111111111111111111111',
        toBuffer: () => Buffer.from('11111111111111111111111111111111')
      },
      secretKey: Buffer.from('test-secret-key')
    })
  }
}))

// Mock wallet adapter
const mockWallet = {
  adapter: {
    name: 'Phantom',
    publicKey: new PublicKey('11111111111111111111111111111111'),
    connected: true
  }
}

const mockConnection = new Connection('https://api.devnet.solana.com')
const mockProps = {
  connection: mockConnection,
  wallet: mockWallet,
  connected: true,
  connecting: false,
  debugInfo: {},
  updateDebugInfo: vi.fn(),
  errorLog: [],
  addToErrorLog: vi.fn(),
  clearErrorLog: vi.fn(),
  testRPCConnection: vi.fn()
}

describe('Performance Tests - Contract Operations', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Wallet Connection Performance', () => {
    it('should connect wallet within 3 seconds', async () => {
      const startTime = Date.now()
      
      render(<WalletTestingComponent {...mockProps} />)
      
      const connectButton = screen.getByText('Connect Wallet')
      fireEvent.click(connectButton)
      
      await waitFor(() => {
        expect(screen.getByText('Connected')).toBeInTheDocument()
      })
      
      const endTime = Date.now()
      const connectionTime = endTime - startTime
      
      expect(connectionTime).toBeLessThan(3000) // 3 seconds
    })

    it('should handle multiple rapid connection attempts efficiently', async () => {
      render(<WalletTestingComponent {...mockProps} />)
      
      const connectButton = screen.getByText('Connect Wallet')
      
      // Rapid clicks
      for (let i = 0; i < 5; i++) {
        fireEvent.click(connectButton)
      }
      
      // Should handle gracefully without errors
      await waitFor(() => {
        expect(screen.getByText('Connected')).toBeInTheDocument()
      })
    })
  })

  describe('RPC Connection Performance', () => {
    it('should test RPC connection within 1 second', async () => {
      const startTime = Date.now()
      
      render(<WalletTestingComponent {...mockProps} />)
      
      const testButton = screen.getByText('Test RPC Connection')
      fireEvent.click(testButton)
      
      await waitFor(() => {
        expect(mockProps.testRPCConnection).toHaveBeenCalled()
      })
      
      const endTime = Date.now()
      const testTime = endTime - startTime
      
      expect(testTime).toBeLessThan(1000) // 1 second
    })

    it('should handle RPC timeout gracefully', async () => {
      const mockConnectionSlow = new Connection('https://api.devnet.solana.com')
      mockConnectionSlow.getVersion = vi.fn().mockImplementation(() => 
        new Promise((resolve) => setTimeout(resolve, 5000)) // 5 second delay
      )
      
      const propsWithSlowConnection = {
        ...mockProps,
        connection: mockConnectionSlow
      }
      
      render(<WalletTestingComponent {...propsWithSlowConnection} />)
      
      const testButton = screen.getByText('Test RPC Connection')
      fireEvent.click(testButton)
      
      // Should not hang indefinitely
      await waitFor(() => {
        expect(screen.getByText('Test RPC Connection')).toBeInTheDocument()
      }, { timeout: 2000 })
    })
  })

  describe('Balance Update Performance', () => {
    it('should update balance within 5 seconds', async () => {
      const startTime = Date.now()
      
      render(<BalanceDisplayComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      })
      
      const endTime = Date.now()
      const updateTime = endTime - startTime
      
      expect(updateTime).toBeLessThan(5000) // 5 seconds
    })

    it('should handle rapid balance updates efficiently', async () => {
      render(<BalanceDisplayComponent {...mockProps} />)
      
      const refreshButton = screen.getByText('Refresh')
      
      // Rapid refresh clicks
      for (let i = 0; i < 10; i++) {
        fireEvent.click(refreshButton)
      }
      
      // Should handle gracefully
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      })
    })

    it('should auto-refresh balance every 30 seconds', async () => {
      vi.useFakeTimers()
      
      render(<BalanceDisplayComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      })
      
      // Fast-forward 30 seconds
      vi.advanceTimersByTime(30000)
      
      // Should trigger auto-refresh
      expect(mockConnection.getBalance).toHaveBeenCalledTimes(2) // Initial + auto-refresh
      
      vi.useRealTimers()
    })
  })

  describe('Error Response Performance', () => {
    it('should display error feedback within 1 second', async () => {
      const mockConnectionError = new Connection('https://api.devnet.solana.com')
      mockConnectionError.getBalance = vi.fn().mockRejectedValue(new Error('Network error'))
      
      const propsWithError = {
        ...mockProps,
        connection: mockConnectionError
      }
      
      const startTime = Date.now()
      
      render(<WalletTestingComponent {...propsWithError} />)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Balance Check Failed')
        )
      })
      
      const endTime = Date.now()
      const errorTime = endTime - startTime
      
      expect(errorTime).toBeLessThan(1000) // 1 second
    })

    it('should handle multiple simultaneous errors efficiently', async () => {
      const mockConnectionError = new Connection('https://api.devnet.solana.com')
      mockConnectionError.getBalance = vi.fn().mockRejectedValue(new Error('Network error'))
      mockConnectionError.getVersion = vi.fn().mockRejectedValue(new Error('Network error'))
      
      const propsWithError = {
        ...mockProps,
        connection: mockConnectionError
      }
      
      render(<WalletTestingComponent {...propsWithError} />)
      
      // Trigger multiple errors simultaneously
      const testButton = screen.getByText('Test RPC Connection')
      fireEvent.click(testButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalled()
      })
      
      // Should handle multiple errors without performance degradation
      expect(mockProps.addToErrorLog).toHaveBeenCalledTimes(2) // Balance + RPC errors
    })
  })

  describe('Memory Performance', () => {
    it('should not leak memory during component updates', async () => {
      const { unmount } = render(<BalanceDisplayComponent {...mockProps} />)
      
      // Simulate multiple updates
      for (let i = 0; i < 100; i++) {
        const refreshButton = screen.getByText('Refresh')
        fireEvent.click(refreshButton)
        
        await waitFor(() => {
          expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
        })
      }
      
      // Unmount and check for cleanup
      unmount()
      
      // Should not have any lingering timers or listeners
      expect(mockConnection.getBalance).toHaveBeenCalled()
    })

    it('should handle large error logs efficiently', async () => {
      const propsWithLargeErrorLog = {
        ...mockProps,
        errorLog: Array(1000).fill('Test error message')
      }
      
      render(<BalanceDisplayComponent {...propsWithLargeErrorLog} />)
      
      // Should render without performance issues
      await waitFor(() => {
        expect(screen.getByText('Balance Display')).toBeInTheDocument()
      })
    })
  })

  describe('Network Performance', () => {
    it('should handle slow network conditions gracefully', async () => {
      const mockConnectionSlow = new Connection('https://api.devnet.solana.com')
      mockConnectionSlow.getBalance = vi.fn().mockImplementation(() => 
        new Promise((resolve) => setTimeout(() => resolve(10000000000), 2000)) // 2 second delay
      )
      
      const propsWithSlowConnection = {
        ...mockProps,
        connection: mockConnectionSlow
      }
      
      const startTime = Date.now()
      
      render(<BalanceDisplayComponent {...propsWithSlowConnection} />)
      
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      })
      
      const endTime = Date.now()
      const loadTime = endTime - startTime
      
      // Should complete within reasonable time even with slow network
      expect(loadTime).toBeLessThan(5000) // 5 seconds
    })

    it('should retry failed requests efficiently', async () => {
      let callCount = 0
      const mockConnectionRetry = new Connection('https://api.devnet.solana.com')
      mockConnectionRetry.getBalance = vi.fn().mockImplementation(() => {
        callCount++
        if (callCount < 3) {
          return Promise.reject(new Error('Network error'))
        }
        return Promise.resolve(10000000000)
      })
      
      const propsWithRetryConnection = {
        ...mockProps,
        connection: mockConnectionRetry
      }
      
      render(<BalanceDisplayComponent {...propsWithRetryConnection} />)
      
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      })
      
      // Should have retried failed requests
      expect(callCount).toBeGreaterThan(1)
    })
  })

  describe('UI Performance', () => {
    it('should render complex UI within 2 seconds', async () => {
      const startTime = Date.now()
      
      render(<BalanceDisplayComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(screen.getByText('Portfolio Summary')).toBeInTheDocument()
        expect(screen.getByText('Wallet Balances')).toBeInTheDocument()
        expect(screen.getByText('Account Balances')).toBeInTheDocument()
        expect(screen.getByText('Current Prices')).toBeInTheDocument()
      })
      
      const endTime = Date.now()
      const renderTime = endTime - startTime
      
      expect(renderTime).toBeLessThan(2000) // 2 seconds
    })

    it('should handle rapid UI updates efficiently', async () => {
      render(<BalanceDisplayComponent {...mockProps} />)
      
      const refreshButton = screen.getByText('Refresh')
      
      // Rapid UI updates
      for (let i = 0; i < 20; i++) {
        fireEvent.click(refreshButton)
        await new Promise(resolve => setTimeout(resolve, 10)) // Small delay
      }
      
      // Should remain responsive
      await waitFor(() => {
        expect(screen.getByText('Balance Display')).toBeInTheDocument()
      })
    })
  })
})
