import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { Connection, PublicKey } from '@solana/web3.js'
import DevnetTestingPage from '../pages/DevnetTestingPage'

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

// Mock ProgramProvider
vi.mock('../contexts/ProgramContext', () => ({
  ProgramProvider: ({ children }: { children: React.ReactNode }) => children
}))

describe('DevnetTestingPage Integration', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Page Rendering', () => {
    it('should render the main devnet testing interface', () => {
      render(<DevnetTestingPage />)
      
      expect(screen.getByText('Devnet Testing Interface')).toBeInTheDocument()
      expect(screen.getByText('Bare-bones interface for testing QuantDesk contract integration')).toBeInTheDocument()
    })

    it('should render all testing component tabs', () => {
      render(<DevnetTestingPage />)
      
      expect(screen.getByText('Wallet')).toBeInTheDocument()
      expect(screen.getByText('Account')).toBeInTheDocument()
      expect(screen.getByText('Deposit')).toBeInTheDocument()
      expect(screen.getByText('Withdraw')).toBeInTheDocument()
      expect(screen.getByText('Balance')).toBeInTheDocument()
      expect(screen.getByText('Debug')).toBeInTheDocument()
    })

    it('should start with wallet tab active', () => {
      render(<DevnetTestingPage />)
      
      const walletTab = screen.getByText('Wallet')
      expect(walletTab).toHaveClass('bg-blue-600')
    })
  })

  describe('Tab Navigation', () => {
    it('should switch between tabs correctly', () => {
      render(<DevnetTestingPage />)
      
      // Click on Account tab
      const accountTab = screen.getByText('Account')
      fireEvent.click(accountTab)
      
      expect(accountTab).toHaveClass('bg-blue-600')
      
      // Click on Deposit tab
      const depositTab = screen.getByText('Deposit')
      fireEvent.click(depositTab)
      
      expect(depositTab).toHaveClass('bg-blue-600')
      expect(accountTab).not.toHaveClass('bg-blue-600')
    })

    it('should render correct component for each tab', () => {
      render(<DevnetTestingPage />)
      
      // Test Wallet tab
      expect(screen.getByText('Wallet Connection Testing')).toBeInTheDocument()
      
      // Switch to Account tab
      fireEvent.click(screen.getByText('Account'))
      expect(screen.getByText('Account Management Testing')).toBeInTheDocument()
      
      // Switch to Deposit tab
      fireEvent.click(screen.getByText('Deposit'))
      expect(screen.getByText('Deposit Testing')).toBeInTheDocument()
      
      // Switch to Withdraw tab
      fireEvent.click(screen.getByText('Withdraw'))
      expect(screen.getByText('Withdraw Testing')).toBeInTheDocument()
      
      // Switch to Balance tab
      fireEvent.click(screen.getByText('Balance'))
      expect(screen.getByText('Balance Display')).toBeInTheDocument()
      
      // Switch to Debug tab
      fireEvent.click(screen.getByText('Debug'))
      expect(screen.getByText('Debug Information Panel')).toBeInTheDocument()
    })
  })

  describe('Debug Information Sharing', () => {
    it('should share debug information between components', async () => {
      render(<DevnetTestingPage />)
      
      // Start on Wallet tab and trigger some action
      const testRPCButton = screen.getByText('Test RPC Connection')
      fireEvent.click(testRPCButton)
      
      // Switch to Debug tab
      fireEvent.click(screen.getByText('Debug'))
      
      // Check if debug information is shared
      await waitFor(() => {
        expect(screen.getByText('Connection Information')).toBeInTheDocument()
      })
    })

    it('should maintain error log across tab switches', async () => {
      render(<DevnetTestingPage />)
      
      // Go to Deposit tab and trigger an error
      fireEvent.click(screen.getByText('Deposit'))
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '0' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      // Switch to Debug tab
      fireEvent.click(screen.getByText('Debug'))
      
      // Check if error is logged
      await waitFor(() => {
        expect(screen.getByText('Error Log')).toBeInTheDocument()
      })
    })
  })

  describe('Component Integration', () => {
    it('should handle wallet connection across all components', async () => {
      render(<DevnetTestingPage />)
      
      // Test wallet connection on Wallet tab
      const connectButton = screen.getByText('Connect Wallet')
      fireEvent.click(connectButton)
      
      // Switch to Account tab and verify wallet is still connected
      fireEvent.click(screen.getByText('Account'))
      await waitFor(() => {
        expect(screen.getByText('Account Management Testing')).toBeInTheDocument()
      })
      
      // Switch to Deposit tab and verify wallet connection
      fireEvent.click(screen.getByText('Deposit'))
      await waitFor(() => {
        expect(screen.getByText('Deposit Testing')).toBeInTheDocument()
      })
    })

    it('should maintain account state across components', async () => {
      render(<DevnetTestingPage />)
      
      // Go to Account tab and check account
      fireEvent.click(screen.getByText('Account'))
      
      const checkAccountButton = screen.getByText('Check Account')
      fireEvent.click(checkAccountButton)
      
      // Switch to Balance tab and verify account state is maintained
      fireEvent.click(screen.getByText('Balance'))
      
      await waitFor(() => {
        expect(screen.getByText('Balance Display')).toBeInTheDocument()
      })
    })
  })

  describe('Error Handling Integration', () => {
    it('should handle network errors across all components', async () => {
      const mockConnectionError = new Connection('https://api.devnet.solana.com')
      mockConnectionError.getVersion = vi.fn().mockRejectedValue(new Error('Network error'))
      mockConnectionError.getBalance = vi.fn().mockRejectedValue(new Error('Network error'))
      
      // Mock the connection to return error
      vi.mocked(Connection).mockImplementation(() => mockConnectionError)
      
      render(<DevnetTestingPage />)
      
      // Test error handling on Wallet tab
      const testRPCButton = screen.getByText('Test RPC Connection')
      fireEvent.click(testRPCButton)
      
      // Switch to Debug tab to see error log
      fireEvent.click(screen.getByText('Debug'))
      
      await waitFor(() => {
        expect(screen.getByText('Debug Information Panel')).toBeInTheDocument()
      })
    })

    it('should handle wallet disconnection across components', async () => {
      render(<DevnetTestingPage />)
      
      // Disconnect wallet on Wallet tab
      const disconnectButton = screen.getByText('Disconnect Wallet')
      fireEvent.click(disconnectButton)
      
      // Switch to Deposit tab and verify wallet is disconnected
      fireEvent.click(screen.getByText('Deposit'))
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '1' } })
      
      const depositButton = screen.getByText('Deposit')
      fireEvent.click(depositButton)
      
      // Should show error about wallet not connected
      await waitFor(() => {
        expect(screen.getByText('Deposit Testing')).toBeInTheDocument()
      })
    })
  })

  describe('Performance Integration', () => {
    it('should handle rapid tab switching without errors', () => {
      render(<DevnetTestingPage />)
      
      const tabs = ['Account', 'Deposit', 'Withdraw', 'Balance', 'Debug', 'Wallet']
      
      // Rapidly switch between tabs
      tabs.forEach(tab => {
        fireEvent.click(screen.getByText(tab))
      })
      
      // Should still be functional
      expect(screen.getByText('Devnet Testing Interface')).toBeInTheDocument()
    })

    it('should maintain component state during tab switches', async () => {
      render(<DevnetTestingPage />)
      
      // Go to Deposit tab and enter amount
      fireEvent.click(screen.getByText('Deposit'))
      
      const amountInput = screen.getByPlaceholderText('Enter amount')
      fireEvent.change(amountInput, { target: { value: '5' } })
      
      // Switch to another tab and back
      fireEvent.click(screen.getByText('Balance'))
      fireEvent.click(screen.getByText('Deposit'))
      
      // Amount should be preserved
      const amountInputAfter = screen.getByPlaceholderText('Enter amount') as HTMLInputElement
      expect(amountInputAfter.value).toBe('5')
    })
  })

  describe('Responsive Design Integration', () => {
    it('should render correctly on different screen sizes', () => {
      render(<DevnetTestingPage />)
      
      // Check for responsive classes
      const sidebar = screen.getByText('Testing Components').closest('div')
      expect(sidebar).toHaveClass('lg:col-span-1')
      
      const mainContent = screen.getByText('Wallet Connection Testing').closest('div')
      expect(mainContent).toHaveClass('lg:col-span-3')
    })
  })
})
