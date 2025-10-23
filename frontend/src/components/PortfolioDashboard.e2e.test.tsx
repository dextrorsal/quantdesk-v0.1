import { describe, it, expect, beforeEach, afterEach } from 'vitest'
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react'
import { BrowserRouter } from 'react-router-dom'
import { WalletProvider } from '@solana/wallet-adapter-react'
import { PhantomWalletAdapter } from '@solana/wallet-adapter-phantom'
import { WalletModalProvider } from '@solana/wallet-adapter-react-ui'
import { PortfolioDashboard } from '../components/PortfolioDashboard'
import { WebSocketProvider } from '../providers/WebSocketProvider'
import { portfolioService } from '../services/portfolioService'
import React from 'react'

// Mock the WebSocket provider
const MockWebSocketProvider = ({ children }: { children: React.ReactNode }) => {
  const [isConnected, setIsConnected] = React.useState(true)
  const [portfolioData, setPortfolioData] = React.useState<Map<string, any>>(new Map())

  const subscribeToPortfolio = React.useCallback((userId: string, callback: (data: any) => void) => {
    // Mock subscription
    const mockData = {
      summary: {
        totalEquity: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 75,
      },
      riskMetrics: {
        totalTrades: 25,
        winRate: 68,
        avgTradeSize: 1000,
        maxDrawdown: 5.2,
      },
      performanceMetrics: {
        dailyReturn: 2.5,
        weeklyReturn: 8.3,
        monthlyReturn: 15.7,
        yearlyReturn: 45.2,
      },
    }
    
    // Simulate real-time updates
    setTimeout(() => {
      callback(mockData)
    }, 100)

    return () => {} // Mock unsubscribe
  }, [])

  return (
    <WebSocketProvider>
      {children}
    </WebSocketProvider>
  )
}

// Mock wallet provider
const MockWalletProvider = ({ children }: { children: React.ReactNode }) => {
  const mockWallet = {
    publicKey: {
      toString: () => 'test-wallet-address-123',
    },
    connected: true,
    connecting: false,
    disconnect: () => {},
    connect: () => {},
  }

  return (
    <WalletProvider wallets={[new PhantomWalletAdapter()]} autoConnect>
      <WalletModalProvider>
        {children}
      </WalletModalProvider>
    </WalletProvider>
  )
}

// Mock portfolio service
vi.mock('../services/portfolioService', () => ({
  portfolioService: {
    instance: {
      getPortfolioSummary: vi.fn(() => ({
        totalEquity: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 75,
      })),
      getRiskMetrics: vi.fn(() => ({
        totalTrades: 25,
        winRate: 68,
        avgTradeSize: 1000,
        maxDrawdown: 5.2,
      })),
      getPerformanceMetrics: vi.fn(() => ({
        dailyReturn: 2.5,
        weeklyReturn: 8.3,
        monthlyReturn: 15.7,
        yearlyReturn: 45.2,
      })),
    },
  },
}))

// Mock WebSocket provider
vi.mock('../providers/WebSocketProvider', () => ({
  useWebSocket: vi.fn(() => ({
    subscribeToPortfolio: vi.fn((userId: string, callback: (data: any) => void) => {
      // Mock subscription with delayed callback
      setTimeout(() => {
        callback({
          summary: {
            totalEquity: 10000,
            totalUnrealizedPnl: 500,
            totalRealizedPnl: 200,
            marginRatio: 75,
          },
          riskMetrics: {
            totalTrades: 25,
            winRate: 68,
            avgTradeSize: 1000,
            maxDrawdown: 5.2,
          },
          performanceMetrics: {
            dailyReturn: 2.5,
            weeklyReturn: 8.3,
            monthlyReturn: 15.7,
            yearlyReturn: 45.2,
          },
        })
      }, 100)
      return () => {} // Mock unsubscribe
    }),
    isConnected: true,
    reconnectAttempts: 0,
  })),
  WebSocketProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

// Mock wallet hook
vi.mock('@solana/wallet-adapter-react', () => ({
  useWallet: vi.fn(() => ({
    publicKey: {
      toString: () => 'test-wallet-address-123',
    },
    connected: true,
    connecting: false,
    disconnect: vi.fn(),
    connect: vi.fn(),
  })),
  WalletProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  WalletModalProvider: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}))

// Mock react-spring
vi.mock('react-spring', () => ({
  useSpring: vi.fn((config) => ({
    number: {
      to: vi.fn((formatter) => formatter(config.number || 0)),
    },
  })),
  animated: {
    div: 'div',
  },
}))

describe('PortfolioDashboard E2E Tests', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('Component Rendering', () => {
    it('should render loading state initially', () => {
      // Mock empty data to trigger loading state
      vi.mocked(portfolioService.instance.getPortfolioSummary).mockReturnValue(null as any)
      vi.mocked(portfolioService.instance.getRiskMetrics).mockReturnValue(null as any)
      vi.mocked(portfolioService.instance.getPerformanceMetrics).mockReturnValue(null as any)

      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      expect(screen.getByText('Loading portfolio data...')).toBeInTheDocument()
    })

    it('should render portfolio dashboard with all sections', async () => {
      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        // Check portfolio summary cards
        expect(screen.getByText('Total Equity')).toBeInTheDocument()
        expect(screen.getByText('Unrealized P&L')).toBeInTheDocument()
        expect(screen.getByText('Realized P&L')).toBeInTheDocument()
        expect(screen.getByText('Margin Ratio')).toBeInTheDocument()

        // Check risk metrics section
        expect(screen.getByText('Risk Metrics')).toBeInTheDocument()
        expect(screen.getByText('Total Trades')).toBeInTheDocument()
        expect(screen.getByText('Win Rate')).toBeInTheDocument()
        expect(screen.getByText('Avg Trade Size')).toBeInTheDocument()
        expect(screen.getByText('Max Drawdown')).toBeInTheDocument()

        // Check performance metrics section
        expect(screen.getByText('Performance Metrics')).toBeInTheDocument()
        expect(screen.getByText('Daily Return')).toBeInTheDocument()
        expect(screen.getByText('Weekly Return')).toBeInTheDocument()
        expect(screen.getByText('Monthly Return')).toBeInTheDocument()
        expect(screen.getByText('Yearly Return')).toBeInTheDocument()
      })
    })

    it('should display correct portfolio values', async () => {
      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        // Check portfolio values are displayed
        expect(screen.getByText('$10,000.00')).toBeInTheDocument() // Total Equity
        expect(screen.getByText('$500.00')).toBeInTheDocument() // Unrealized P&L
        expect(screen.getByText('$200.00')).toBeInTheDocument() // Realized P&L
        expect(screen.getByText('75.0%')).toBeInTheDocument() // Margin Ratio
      })
    })

    it('should display correct risk metrics', async () => {
      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        // Check risk metrics values
        expect(screen.getByText('25')).toBeInTheDocument() // Total Trades
        expect(screen.getByText('68%')).toBeInTheDocument() // Win Rate
        expect(screen.getByText('$1,000')).toBeInTheDocument() // Avg Trade Size
        expect(screen.getByText('5.2%')).toBeInTheDocument() // Max Drawdown
      })
    })

    it('should display correct performance metrics', async () => {
      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        // Check performance metrics values
        expect(screen.getByText('+2.50%')).toBeInTheDocument() // Daily Return
        expect(screen.getByText('+8.30%')).toBeInTheDocument() // Weekly Return
        expect(screen.getByText('+15.70%')).toBeInTheDocument() // Monthly Return
        expect(screen.getByText('+45.20%')).toBeInTheDocument() // Yearly Return
      })
    })
  })

  describe('WebSocket Integration', () => {
    it('should subscribe to portfolio updates when wallet is connected', async () => {
      const mockSubscribeToPortfolio = vi.fn()
      vi.mocked(require('@solana/wallet-adapter-react').useWallet).mockReturnValue({
        publicKey: { toString: () => 'test-wallet-address-123' },
        connected: true,
      })
      vi.mocked(require('../providers/WebSocketProvider').useWebSocket).mockReturnValue({
        subscribeToPortfolio: mockSubscribeToPortfolio,
        isConnected: true,
        reconnectAttempts: 0,
      })

      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        expect(mockSubscribeToPortfolio).toHaveBeenCalledWith(
          'test-wallet-address-123',
          expect.any(Function)
        )
      })
    })

    it('should not subscribe to portfolio updates when wallet is not connected', async () => {
      const mockSubscribeToPortfolio = vi.fn()
      vi.mocked(require('@solana/wallet-adapter-react').useWallet).mockReturnValue({
        publicKey: null,
        connected: false,
      })
      vi.mocked(require('../providers/WebSocketProvider').useWebSocket).mockReturnValue({
        subscribeToPortfolio: mockSubscribeToPortfolio,
        isConnected: true,
        reconnectAttempts: 0,
      })

      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        expect(mockSubscribeToPortfolio).not.toHaveBeenCalled()
      })
    })

    it('should update portfolio data when WebSocket receives updates', async () => {
      let portfolioCallback: (data: any) => void
      const mockSubscribeToPortfolio = vi.fn((userId: string, callback: (data: any) => void) => {
        portfolioCallback = callback
        return () => {}
      })

      vi.mocked(require('../providers/WebSocketProvider').useWebSocket).mockReturnValue({
        subscribeToPortfolio: mockSubscribeToPortfolio,
        isConnected: true,
        reconnectAttempts: 0,
      })

      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      // Simulate WebSocket update
      act(() => {
        portfolioCallback!({
          summary: {
            totalEquity: 15000,
            totalUnrealizedPnl: 1000,
            totalRealizedPnl: 500,
            marginRatio: 80,
          },
          riskMetrics: {
            totalTrades: 30,
            winRate: 70,
            avgTradeSize: 1200,
            maxDrawdown: 4.5,
          },
          performanceMetrics: {
            dailyReturn: 3.2,
            weeklyReturn: 10.5,
            monthlyReturn: 18.2,
            yearlyReturn: 52.1,
          },
        })
      })

      await waitFor(() => {
        // Check that values have been updated
        expect(screen.getByText('$15,000.00')).toBeInTheDocument()
        expect(screen.getByText('$1,000.00')).toBeInTheDocument()
        expect(screen.getByText('$500.00')).toBeInTheDocument()
        expect(screen.getByText('80.0%')).toBeInTheDocument()
      })
    })
  })

  describe('Visual Indicators', () => {
    it('should display correct colors for positive P&L', async () => {
      vi.mocked(portfolioService.instance.getPortfolioSummary).mockReturnValue({
        totalEquity: 10000,
        totalUnrealizedPnl: 500, // Positive
        totalRealizedPnl: 200, // Positive
        marginRatio: 75,
      })

      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        // Check for positive P&L styling
        const unrealizedPnlElement = screen.getByText('$500.00')
        const realizedPnlElement = screen.getByText('$200.00')
        
        expect(unrealizedPnlElement).toHaveClass('text-green-400')
        expect(realizedPnlElement).toHaveClass('text-green-400')
      })
    })

    it('should display correct colors for negative P&L', async () => {
      vi.mocked(portfolioService.instance.getPortfolioSummary).mockReturnValue({
        totalEquity: 10000,
        totalUnrealizedPnl: -500, // Negative
        totalRealizedPnl: -200, // Negative
        marginRatio: 75,
      })

      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        // Check for negative P&L styling
        const unrealizedPnlElement = screen.getByText('-$500.00')
        const realizedPnlElement = screen.getByText('-$200.00')
        
        expect(unrealizedPnlElement).toHaveClass('text-red-400')
        expect(realizedPnlElement).toHaveClass('text-red-400')
      })
    })

    it('should display correct colors for margin ratio levels', async () => {
      // Test high margin ratio (red)
      vi.mocked(portfolioService.instance.getPortfolioSummary).mockReturnValue({
        totalEquity: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 85, // High - should be red
      })

      const { rerender } = render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        const marginRatioElement = screen.getByText('85.0%')
        expect(marginRatioElement).toHaveClass('text-red-400')
      })

      // Test medium margin ratio (yellow)
      vi.mocked(portfolioService.instance.getPortfolioSummary).mockReturnValue({
        totalEquity: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 70, // Medium - should be yellow
      })

      rerender(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        const marginRatioElement = screen.getByText('70.0%')
        expect(marginRatioElement).toHaveClass('text-yellow-400')
      })

      // Test low margin ratio (green)
      vi.mocked(portfolioService.instance.getPortfolioSummary).mockReturnValue({
        totalEquity: 10000,
        totalUnrealizedPnl: 500,
        totalRealizedPnl: 200,
        marginRatio: 50, // Low - should be green
      })

      rerender(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        const marginRatioElement = screen.getByText('50.0%')
        expect(marginRatioElement).toHaveClass('text-green-400')
      })
    })
  })

  describe('Responsive Design', () => {
    it('should render correctly on different screen sizes', async () => {
      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      await waitFor(() => {
        // Check that grid classes are applied for responsive design
        const portfolioCards = screen.getByText('Total Equity').closest('.grid')
        expect(portfolioCards).toHaveClass('grid-cols-1', 'md:grid-cols-2', 'lg:grid-cols-4')
      })
    })
  })

  describe('Error Handling', () => {
    it('should handle WebSocket connection errors gracefully', async () => {
      vi.mocked(require('../providers/WebSocketProvider').useWebSocket).mockReturnValue({
        subscribeToPortfolio: vi.fn(() => {
          throw new Error('WebSocket connection failed')
        }),
        isConnected: false,
        reconnectAttempts: 3,
      })

      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      // Should still render the component without crashing
      await waitFor(() => {
        expect(screen.getByText('Total Equity')).toBeInTheDocument()
      })
    })

    it('should handle missing portfolio data gracefully', async () => {
      vi.mocked(portfolioService.instance.getPortfolioSummary).mockReturnValue(null as any)
      vi.mocked(portfolioService.instance.getRiskMetrics).mockReturnValue(null as any)
      vi.mocked(portfolioService.instance.getPerformanceMetrics).mockReturnValue(null as any)

      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      // Should show loading state
      expect(screen.getByText('Loading portfolio data...')).toBeInTheDocument()
    })
  })

  describe('Performance', () => {
    it('should handle rapid WebSocket updates efficiently', async () => {
      let portfolioCallback: (data: any) => void
      const mockSubscribeToPortfolio = vi.fn((userId: string, callback: (data: any) => void) => {
        portfolioCallback = callback
        return () => {}
      })

      vi.mocked(require('../providers/WebSocketProvider').useWebSocket).mockReturnValue({
        subscribeToPortfolio: mockSubscribeToPortfolio,
        isConnected: true,
        reconnectAttempts: 0,
      })

      render(
        <BrowserRouter>
          <MockWalletProvider>
            <MockWebSocketProvider>
              <PortfolioDashboard />
            </MockWebSocketProvider>
          </MockWalletProvider>
        </BrowserRouter>
      )

      // Simulate rapid updates
      for (let i = 0; i < 10; i++) {
        act(() => {
          portfolioCallback!({
            summary: {
              totalEquity: 10000 + i * 100,
              totalUnrealizedPnl: 500 + i * 50,
              totalRealizedPnl: 200 + i * 20,
              marginRatio: 75 + i,
            },
            riskMetrics: {
              totalTrades: 25 + i,
              winRate: 68 + i,
              avgTradeSize: 1000 + i * 100,
              maxDrawdown: 5.2 - i * 0.1,
            },
            performanceMetrics: {
              dailyReturn: 2.5 + i * 0.1,
              weeklyReturn: 8.3 + i * 0.2,
              monthlyReturn: 15.7 + i * 0.3,
              yearlyReturn: 45.2 + i * 0.5,
            },
          })
        })
      }

      await waitFor(() => {
        // Should display the latest values
        expect(screen.getByText('$10,900.00')).toBeInTheDocument() // 10000 + 9*100
        expect(screen.getByText('$950.00')).toBeInTheDocument() // 500 + 9*50
        expect(screen.getByText('$380.00')).toBeInTheDocument() // 200 + 9*20
        expect(screen.getByText('84.0%')).toBeInTheDocument() // 75 + 9
      })
    })
  })
})
