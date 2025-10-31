import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { Connection, PublicKey } from '@solana/web3.js'
import WalletTestingComponent from '../components/devnet-testing/WalletTestingComponent'
import AccountTestingComponent from '../components/devnet-testing/AccountTestingComponent'
import DepositTestingComponent from '../components/devnet-testing/DepositTestingComponent'

// Mock Solana Web3.js
vi.mock('@solana/web3.js', () => ({
  Connection: vi.fn().mockImplementation(() => ({
    getVersion: vi.fn().mockResolvedValue({ version: '1.16.0' }),
    getBalance: vi.fn().mockResolvedValue(10000000000), // 10 SOL in lamports
    getAccountInfo: vi.fn().mockResolvedValue({
      owner: new PublicKey('11111111111111111111111111111111'),
      lamports: 5000000000, // 5 SOL
      data: Buffer.from('test data'),
      executable: false,
      rentEpoch: 0
    }),
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
    connected: true,
    connect: vi.fn(),
    disconnect: vi.fn()
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

describe('Error Scenario Testing - Contract Interactions', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Network Error Scenarios', () => {
    it('should handle RPC connection failure', async () => {
      const mockConnectionError = new Connection('https://api.devnet.solana.com')
      mockConnectionError.getVersion = vi.fn().mockRejectedValue(new Error('RPC connection failed'))
      
      const propsWithError = {
        ...mockProps,
        connection: mockConnectionError
      }
      
      render(<WalletTestingComponent {...propsWithError} />)
      
      const testButton = screen.getByText('Test RPC Connection')
      fireEvent.click(testButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('RPC Connection Failed')
        )
      })
    })

    it('should handle network timeout', async () => {
      const mockConnectionTimeout = new Connection('https://api.devnet.solana.com')
      mockConnectionTimeout.getBalance = vi.fn().mockRejectedValue(new Error('Request timeout'))
      
      const propsWithTimeout = {
        ...mockProps,
        connection: mockConnectionTimeout
      }
      
      render(<WalletTestingComponent {...propsWithTimeout} />)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Balance Check Failed')
        )
      })
    })

    it('should handle rate limit exceeded', async () => {
      const mockConnectionRateLimit = new Connection('https://api.devnet.solana.com')
      mockConnectionRateLimit.getAccountInfo = vi.fn().mockRejectedValue(new Error('Rate limit exceeded'))
      
      const propsWithRateLimit = {
        ...mockProps,
        connection: mockConnectionRateLimit
      }
      
      render(<AccountTestingComponent {...propsWithRateLimit} />)
      
      const checkButton = screen.getByText('Check Account')
      fireEvent.click(checkButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Account Check Failed')
        )
      })
    })

    it('should handle network unavailable', async () => {
      const mockConnectionUnavailable = new Connection('https://api.devnet.solana.com')
      mockConnectionUnavailable.getVersion = vi.fn().mockRejectedValue(new Error('Network unavailable'))
      
      const propsWithUnavailable = {
        ...mockProps,
        connection: mockConnectionUnavailable
      }
      
      render(<WalletTestingComponent {...propsWithUnavailable} />)
      
      const testButton = screen.getByText('Test RPC Connection')
      fireEvent.click(testButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('RPC Connection Failed')
        )
      })
    })
  })

  describe('Wallet Error Scenarios', () => {
    it('should handle wallet not connected', async () => {
      const propsNotConnected = {
        ...mockProps,
        connected: false,
        wallet: null
      }
      
      render(<DepositTestingComponent {...propsNotConnected} />)
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '1' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      expect(mockProps.addToErrorLog).toHaveBeenCalledWith('Wallet not connected')
    })

    it('should handle user rejected transaction', async () => {
      const mockWalletReject = {
        adapter: {
          ...mockWallet.adapter,
          connect: vi.fn().mockRejectedValue(new Error('User rejected'))
        }
      }
      
      const propsWithReject = {
        ...mockProps,
        wallet: mockWalletReject
      }
      
      render(<WalletTestingComponent {...propsWithReject} />)
      
      const connectButton = screen.getByText('Connect Wallet')
      fireEvent.click(connectButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Wallet Connect Failed')
        )
      })
    })

    it('should handle insufficient funds for fees', async () => {
      const mockConnectionLowBalance = new Connection('https://api.devnet.solana.com')
      mockConnectionLowBalance.getBalance = vi.fn().mockResolvedValue(1000000) // Very low balance
      
      const propsWithLowBalance = {
        ...mockProps,
        connection: mockConnectionLowBalance
      }
      
      render(<DepositTestingComponent {...propsWithLowBalance} />)
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '1' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          'Amount exceeds wallet balance'
        )
      })
    })

    it('should handle wallet locked', async () => {
      const mockWalletLocked = {
        adapter: {
          ...mockWallet.adapter,
          connect: vi.fn().mockRejectedValue(new Error('Wallet is locked'))
        }
      }
      
      const propsWithLocked = {
        ...mockProps,
        wallet: mockWalletLocked
      }
      
      render(<WalletTestingComponent {...propsWithLocked} />)
      
      const connectButton = screen.getByText('Connect Wallet')
      fireEvent.click(connectButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Wallet Connect Failed')
        )
      })
    })
  })

  describe('Program Error Scenarios', () => {
    it('should handle insufficient collateral', async () => {
      const mockConnectionInsufficient = new Connection('https://api.devnet.solana.com')
      mockConnectionInsufficient.getAccountInfo = vi.fn().mockResolvedValue({
        owner: new PublicKey('11111111111111111111111111111111'),
        lamports: 1000000, // Very low balance
        data: Buffer.from('test data'),
        executable: false,
        rentEpoch: 0
      })
      
      const propsWithInsufficient = {
        ...mockProps,
        connection: mockConnectionInsufficient
      }
      
      render(<DepositTestingComponent {...propsWithInsufficient} />)
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '1' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          'Amount exceeds wallet balance'
        )
      })
    })

    it('should handle account not found', async () => {
      const mockConnectionNotFound = new Connection('https://api.devnet.solana.com')
      mockConnectionNotFound.getAccountInfo = vi.fn().mockResolvedValue(null)
      
      const propsWithNotFound = {
        ...mockProps,
        connection: mockConnectionNotFound
      }
      
      render(<AccountTestingComponent {...propsWithNotFound} />)
      
      const checkButton = screen.getByText('Check Account')
      fireEvent.click(checkButton)
      
      await waitFor(() => {
        expect(screen.getByText('Not Found')).toBeInTheDocument()
      })
    })

    it('should handle market closed', async () => {
      const mockConnectionMarketClosed = new Connection('https://api.devnet.solana.com')
      mockConnectionMarketClosed.getAccountInfo = vi.fn().mockRejectedValue(new Error('Market is closed'))
      
      const propsWithMarketClosed = {
        ...mockProps,
        connection: mockConnectionMarketClosed
      }
      
      render(<AccountTestingComponent {...propsWithMarketClosed} />)
      
      const checkButton = screen.getByText('Check Account')
      fireEvent.click(checkButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Account Check Failed')
        )
      })
    })

    it('should handle oracle error', async () => {
      const mockConnectionOracleError = new Connection('https://api.devnet.solana.com')
      mockConnectionOracleError.getAccountInfo = vi.fn().mockRejectedValue(new Error('Price oracle error'))
      
      const propsWithOracleError = {
        ...mockProps,
        connection: mockConnectionOracleError
      }
      
      render(<AccountTestingComponent {...propsWithOracleError} />)
      
      const checkButton = screen.getByText('Check Account')
      fireEvent.click(checkButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Account Check Failed')
        )
      })
    })
  })

  describe('User Error Scenarios', () => {
    it('should handle invalid amount input', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: 'invalid' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          'Amount must be greater than 0'
        )
      })
    })

    it('should handle amount too small', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '0.000001' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          'Amount must be greater than 0'
        )
      })
    })

    it('should handle amount too large', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '1000000' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          'Amount exceeds wallet balance'
        )
      })
    })

    it('should handle invalid token selection', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      const tokenSelector = screen.getByDisplayValue('SOL')
      fireEvent.change(tokenSelector, { target: { value: 'INVALID' } })
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '1' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      // Should still process with default validation
      await waitFor(() => {
        expect(screen.getByText('Deposit Testing')).toBeInTheDocument()
      })
    })
  })

  describe('Recovery Scenarios', () => {
    it('should recover from network error with retry', async () => {
      let callCount = 0
      const mockConnectionRetry = new Connection('https://api.devnet.solana.com')
      mockConnectionRetry.getBalance = vi.fn().mockImplementation(() => {
        callCount++
        if (callCount === 1) {
          return Promise.reject(new Error('Network error'))
        }
        return Promise.resolve(10000000000)
      })
      
      const propsWithRetry = {
        ...mockProps,
        connection: mockConnectionRetry
      }
      
      render(<WalletTestingComponent {...propsWithRetry} />)
      
      // First call should fail
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Balance Check Failed')
        )
      })
      
      // Retry should succeed
      const testButton = screen.getByText('Test RPC Connection')
      fireEvent.click(testButton)
      
      await waitFor(() => {
        expect(mockProps.testRPCConnection).toHaveBeenCalled()
      })
    })

    it('should recover from wallet disconnection', async () => {
      const propsDisconnected = {
        ...mockProps,
        connected: false
      }
      
      render(<WalletTestingComponent {...propsDisconnected} />)
      
      // Should show connect button
      expect(screen.getByText('Connect Wallet')).toBeInTheDocument()
      
      // Reconnect
      const connectButton = screen.getByText('Connect Wallet')
      fireEvent.click(connectButton)
      
      // Should attempt reconnection
      expect(mockWallet.adapter.connect).toHaveBeenCalled()
    })

    it('should recover from account initialization failure', async () => {
      const mockConnectionNotFound = new Connection('https://api.devnet.solana.com')
      mockConnectionNotFound.getAccountInfo = vi.fn().mockResolvedValue(null)
      
      const propsWithNotFound = {
        ...mockProps,
        connection: mockConnectionNotFound
      }
      
      render(<AccountTestingComponent {...propsWithNotFound} />)
      
      await waitFor(() => {
        expect(screen.getByText('Not Found')).toBeInTheDocument()
      })
      
      // Should show initialize button
      const initializeButton = screen.getByText('Initialize Account')
      expect(initializeButton).toBeInTheDocument()
      
      // Should be able to retry initialization
      fireEvent.click(initializeButton)
      
      await waitFor(() => {
        expect(screen.getByText('Initializing...')).toBeInTheDocument()
      })
    })
  })

  describe('Error Logging and Display', () => {
    it('should log all error types correctly', async () => {
      const mockConnectionError = new Connection('https://api.devnet.solana.com')
      mockConnectionError.getVersion = vi.fn().mockRejectedValue(new Error('RPC error'))
      mockConnectionError.getBalance = vi.fn().mockRejectedValue(new Error('Balance error'))
      mockConnectionError.getAccountInfo = vi.fn().mockRejectedValue(new Error('Account error'))
      
      const propsWithErrors = {
        ...mockProps,
        connection: mockConnectionError
      }
      
      render(<WalletTestingComponent {...propsWithErrors} />)
      
      // Trigger multiple errors
      const testButton = screen.getByText('Test RPC Connection')
      fireEvent.click(testButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('RPC Connection Failed')
        )
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Balance Check Failed')
        )
      })
    })

    it('should maintain error log across component updates', async () => {
      const propsWithErrorLog = {
        ...mockProps,
        errorLog: ['Test error 1', 'Test error 2']
      }
      
      render(<WalletTestingComponent {...propsWithErrorLog} />)
      
      // Add more errors
      const testButton = screen.getByText('Test RPC Connection')
      fireEvent.click(testButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalled()
      })
      
      // Error log should be maintained
      expect(mockProps.errorLog).toHaveLength(2)
    })
  })
})
