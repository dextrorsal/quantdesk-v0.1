import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { Connection, PublicKey } from '@solana/web3.js'
import AccountTestingComponent from './AccountTestingComponent'

// Mock Solana Web3.js
vi.mock('@solana/web3.js', () => ({
  Connection: vi.fn().mockImplementation(() => ({
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

describe('AccountTestingComponent', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Rendering', () => {
    it('should render account management testing interface', () => {
      render(<AccountTestingComponent {...mockProps} />)
      
      expect(screen.getByText('Account Management Testing')).toBeInTheDocument()
      expect(screen.getByText('Account Status')).toBeInTheDocument()
      expect(screen.getByText('Account Actions')).toBeInTheDocument()
    })

    it('should display account status correctly', async () => {
      render(<AccountTestingComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(screen.getByText('Found')).toBeInTheDocument()
        expect(screen.getByText('5.0000 SOL')).toBeInTheDocument()
      })
    })
  })

  describe('Account Detection', () => {
    it('should check account existence on wallet connection', async () => {
      render(<AccountTestingComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(mockConnection.getAccountInfo).toHaveBeenCalled()
        expect(screen.getByText('Found')).toBeInTheDocument()
      })
    })

    it('should handle account not found scenario', async () => {
      const mockConnectionNotFound = new Connection('https://api.devnet.solana.com')
      mockConnectionNotFound.getAccountInfo = vi.fn().mockResolvedValue(null)
      
      const propsWithNotFound = {
        ...mockProps,
        connection: mockConnectionNotFound
      }
      
      render(<AccountTestingComponent {...propsWithNotFound} />)
      
      await waitFor(() => {
        expect(screen.getByText('Not Found')).toBeInTheDocument()
        expect(screen.getByText('Initialize Account')).toBeInTheDocument()
      })
    })

    it('should handle account check errors', async () => {
      const mockConnectionError = new Connection('https://api.devnet.solana.com')
      mockConnectionError.getAccountInfo = vi.fn().mockRejectedValue(new Error('RPC Error'))
      
      const propsWithError = {
        ...mockProps,
        connection: mockConnectionError
      }
      
      render(<AccountTestingComponent {...propsWithError} />)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Account Check Failed')
        )
      })
    })
  })

  describe('Account Initialization', () => {
    it('should show initialize button when account not found', async () => {
      const mockConnectionNotFound = new Connection('https://api.devnet.solana.com')
      mockConnectionNotFound.getAccountInfo = vi.fn().mockResolvedValue(null)
      
      const propsWithNotFound = {
        ...mockProps,
        connection: mockConnectionNotFound
      }
      
      render(<AccountTestingComponent {...propsWithNotFound} />)
      
      await waitFor(() => {
        const initializeButton = screen.getByText('Initialize Account')
        expect(initializeButton).toBeInTheDocument()
        expect(initializeButton).not.toBeDisabled()
      })
    })

    it('should initialize account when button is clicked', async () => {
      const mockConnectionNotFound = new Connection('https://api.devnet.solana.com')
      mockConnectionNotFound.getAccountInfo = vi.fn().mockResolvedValue(null)
      
      const propsWithNotFound = {
        ...mockProps,
        connection: mockConnectionNotFound
      }
      
      render(<AccountTestingComponent {...propsWithNotFound} />)
      
      await waitFor(() => {
        const initializeButton = screen.getByText('Initialize Account')
        fireEvent.click(initializeButton)
      })
      
      await waitFor(() => {
        expect(screen.getByText('Initializing...')).toBeInTheDocument()
      })
      
      // Wait for initialization to complete
      await waitFor(() => {
        expect(screen.getByText('Found')).toBeInTheDocument()
      }, { timeout: 3000 })
    })

    it('should handle initialization errors', async () => {
      const mockConnectionNotFound = new Connection('https://api.devnet.solana.com')
      mockConnectionNotFound.getAccountInfo = vi.fn().mockResolvedValue(null)
      
      const propsWithNotFound = {
        ...mockProps,
        connection: mockConnectionNotFound
      }
      
      // Mock initialization to fail
      vi.spyOn(global, 'setTimeout').mockImplementation((callback) => {
        callback()
        return 1 as any
      })
      
      render(<AccountTestingComponent {...propsWithNotFound} />)
      
      await waitFor(() => {
        const initializeButton = screen.getByText('Initialize Account')
        fireEvent.click(initializeButton)
      })
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Account Initialization Failed')
        )
      })
    })
  })

  describe('Account Information Display', () => {
    it('should display account information when account exists', async () => {
      render(<AccountTestingComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(screen.getByText('Account Information')).toBeInTheDocument()
        expect(screen.getByText('11111111111111111111111111111111')).toBeInTheDocument()
        expect(screen.getByText('5000000000')).toBeInTheDocument()
        expect(screen.getByText('No')).toBeInTheDocument() // executable
      })
    })

    it('should show view on explorer button when account exists', async () => {
      render(<AccountTestingComponent {...mockProps} />)
      
      await waitFor(() => {
        const explorerButton = screen.getByText('View on Explorer')
        expect(explorerButton).toBeInTheDocument()
      })
    })
  })

  describe('Error Handling', () => {
    it('should handle wallet not connected scenario', () => {
      const propsNotConnected = {
        ...mockProps,
        connected: false,
        wallet: null
      }
      
      render(<AccountTestingComponent {...propsNotConnected} />)
      
      expect(mockProps.addToErrorLog).toHaveBeenCalledWith('No wallet connected')
    })

    it('should handle RPC connection errors', async () => {
      const mockConnectionError = new Connection('https://api.devnet.solana.com')
      mockConnectionError.getAccountInfo = vi.fn().mockRejectedValue(new Error('Network error'))
      
      const propsWithError = {
        ...mockProps,
        connection: mockConnectionError
      }
      
      render(<AccountTestingComponent {...propsWithError} />)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Account Check Failed')
        )
      })
    })
  })

  describe('Status Indicators', () => {
    it('should show correct status colors for found account', async () => {
      render(<AccountTestingComponent {...mockProps} />)
      
      await waitFor(() => {
        const foundStatus = screen.getByText('Found')
        expect(foundStatus).toHaveClass('text-green-400')
      })
    })

    it('should show correct status colors for not found account', async () => {
      const mockConnectionNotFound = new Connection('https://api.devnet.solana.com')
      mockConnectionNotFound.getAccountInfo = vi.fn().mockResolvedValue(null)
      
      const propsWithNotFound = {
        ...mockProps,
        connection: mockConnectionNotFound
      }
      
      render(<AccountTestingComponent {...propsWithNotFound} />)
      
      await waitFor(() => {
        const notFoundStatus = screen.getByText('Not Found')
        expect(notFoundStatus).toHaveClass('text-yellow-400')
      })
    })
  })
})
