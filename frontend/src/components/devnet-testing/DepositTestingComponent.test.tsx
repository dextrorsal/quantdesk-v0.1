import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { Connection, PublicKey, Transaction, SystemProgram } from '@solana/web3.js'
import DepositTestingComponent from './DepositTestingComponent'

// Mock wallet adapter
const mockWallet = {
  adapter: {
    name: 'Phantom',
    publicKey: {
      toString: () => '11111111111111111111111111111111',
      toBuffer: () => Buffer.from('11111111111111111111111111111111')
    },
    connected: true
  }
}

const mockSignTransaction = vi.fn().mockResolvedValue({
  serialize: vi.fn().mockReturnValue(Buffer.from('signed-transaction'))
})

// Mock useWallet hook
vi.mock('@solana/wallet-adapter-react', () => ({
  useWallet: () => ({
    connect: vi.fn(),
    disconnect: vi.fn(),
    signTransaction: mockSignTransaction,
    publicKey: new PublicKey('11111111111111111111111111111111'),
    connected: true,
    connecting: false
  })
}))

// Mock connection.getBalance to return a proper balance
vi.mock('@solana/web3.js', () => ({
  Connection: vi.fn().mockImplementation(() => ({
    getBalance: vi.fn().mockResolvedValue(10000000000), // 10 SOL in lamports
    getVersion: vi.fn().mockResolvedValue({ 'feature-set': 3604001754 }),
    sendRawTransaction: vi.fn().mockResolvedValue('test-signature-123'),
    confirmTransaction: vi.fn().mockResolvedValue({ value: { err: null } }),
    rpcEndpoint: 'https://api.devnet.solana.com'
  })),
  PublicKey: vi.fn().mockImplementation((key) => ({
    toString: () => key,
    toBuffer: () => Buffer.from(key)
  })),
  Transaction: vi.fn().mockImplementation(() => ({
    add: vi.fn(),
    serialize: vi.fn().mockReturnValue(Buffer.from('serialized-transaction'))
  })),
  SystemProgram: {
    transfer: vi.fn().mockReturnValue({})
  },
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

const mockConnection = {
  getBalance: vi.fn().mockResolvedValue(10000000000), // 10 SOL in lamports
  getVersion: vi.fn().mockResolvedValue({ 'feature-set': 3604001754 }),
  sendRawTransaction: vi.fn().mockResolvedValue('test-signature-123'),
  confirmTransaction: vi.fn().mockResolvedValue({ value: { err: null } }),
  rpcEndpoint: 'https://api.devnet.solana.com'
}
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

describe('DepositTestingComponent', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Rendering', () => {
    it('should render deposit testing interface', () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      expect(screen.getByText('Deposit Testing')).toBeInTheDocument()
      expect(screen.getByText('Transaction Status')).toBeInTheDocument()
      expect(screen.getByText('Deposit Form')).toBeInTheDocument()
      expect(screen.getByText('Balance Preview')).toBeInTheDocument()
    })

    it('should display wallet balance correctly', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      }, { timeout: 3000 })
    })
  })

  describe('Form Validation', () => {
    it('should validate amount input', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '15' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          'Amount exceeds wallet balance'
        )
      })
    })

    it('should validate zero amount', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '0' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          'Amount must be greater than 0'
        )
      })
    })

    it('should validate negative amount', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '-1' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          'Amount must be greater than 0'
        )
      })
    })
  })

  describe('Percentage Buttons', () => {
    it('should set amount to 25% of balance', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      }, { timeout: 3000 })
      
      const percentageButton = screen.getByText('25%')
      fireEvent.click(percentageButton)
      
      const amountInput = screen.getByPlaceholderText('Enter amount') as HTMLInputElement
      expect(amountInput.value).toBe('2.5')
    })

    it('should set amount to 50% of balance', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      }, { timeout: 3000 })
      
      const percentageButton = screen.getByText('50%')
      fireEvent.click(percentageButton)
      
      const amountInput = screen.getByPlaceholderText('Enter amount') as HTMLInputElement
      expect(amountInput.value).toBe('5')
    })

    it('should set amount to 100% of balance', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      }, { timeout: 3000 })
      
      const percentageButton = screen.getByText('100%')
      fireEvent.click(percentageButton)
      
      const amountInput = screen.getByPlaceholderText('Enter amount') as HTMLInputElement
      expect(amountInput.value).toBe('10')
    })
  })

  describe('Transaction Flow', () => {
    it('should execute successful deposit transaction', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      }, { timeout: 3000 })
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '1' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      // Check transaction status progression
      await waitFor(() => {
        expect(screen.getByText('Preparing transaction...')).toBeInTheDocument()
      })
      
      await waitFor(() => {
        expect(screen.getByText('Please sign transaction in wallet...')).toBeInTheDocument()
      })
      
      await waitFor(() => {
        expect(screen.getByText('Submitting transaction...')).toBeInTheDocument()
      })
      
      await waitFor(() => {
        expect(screen.getByText('Waiting for confirmation...')).toBeInTheDocument()
      })
      
      await waitFor(() => {
        expect(screen.getByText('Deposit successful!')).toBeInTheDocument()
      })
    })

    it('should handle transaction failure', async () => {
      const mockConnectionError = new Connection('https://api.devnet.solana.com')
      mockConnectionError.sendRawTransaction = vi.fn().mockRejectedValue(new Error('Transaction failed'))
      
      const propsWithError = {
        ...mockProps,
        connection: mockConnectionError
      }
      
      render(<DepositTestingComponent {...propsWithError} />)
      
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      }, { timeout: 3000 })
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '1' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      await waitFor(() => {
        expect(screen.getByText('Transaction failed')).toBeInTheDocument()
      })
      
      expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
        expect.stringContaining('Deposit Failed')
      )
    })

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
  })

  describe('Balance Preview', () => {
    it('should show correct balance preview', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      }, { timeout: 3000 })
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '2' } })
      
      expect(screen.getByText('2 SOL')).toBeInTheDocument()
      expect(screen.getByText('8.0000 SOL')).toBeInTheDocument() // New wallet balance
    })
  })

  describe('Token Selection', () => {
    it('should allow token selection', () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      const tokenSelector = screen.getByDisplayValue('SOL')
      expect(tokenSelector).toBeInTheDocument()
      
      fireEvent.change(tokenSelector, { target: { value: 'USDC' } })
      expect(tokenSelector).toHaveValue('USDC')
    })
  })

  describe('Error Handling', () => {
    it('should handle balance fetch errors', async () => {
      const mockConnectionError = new Connection('https://api.devnet.solana.com')
      mockConnectionError.getBalance = vi.fn().mockRejectedValue(new Error('Network error'))
      
      const propsWithError = {
        ...mockProps,
        connection: mockConnectionError
      }
      
      render(<DepositTestingComponent {...propsWithError} />)
      
      await waitFor(() => {
        expect(mockProps.addToErrorLog).toHaveBeenCalledWith(
          expect.stringContaining('Balance Check Failed')
        )
      })
    })
  })

  describe('Transaction Status Display', () => {
    it('should show transaction signature when available', async () => {
      render(<DepositTestingComponent {...mockProps} />)
      
      await waitFor(() => {
        expect(screen.getByText('10.0000 SOL')).toBeInTheDocument()
      }, { timeout: 3000 })
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '1' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      await waitFor(() => {
        expect(screen.getByText('test-signature-123')).toBeInTheDocument()
        expect(screen.getByText('View on Explorer')).toBeInTheDocument()
      })
    })
  })
})
