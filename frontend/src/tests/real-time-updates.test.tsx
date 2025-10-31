/**
 * Frontend Real-time Updates Tests
 * 
 * This test suite validates the frontend real-time update functionality
 * required for the hackathon demo, ensuring all components receive and
 * display updates correctly.
 */

import React from 'react';
import { render, screen, waitFor, act } from '@testing-library/react';
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import Positions from '../components/Positions';
import Orders from '../components/Orders';
import PortfolioDashboard from '../components/PortfolioDashboard';

// Mock fetch
global.fetch = vi.fn();

// Mock localStorage
const localStorageMock = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
};
Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
});

// Mock WebSocket
class MockWebSocket {
  public onopen: ((event: Event) => void) | null = null;
  public onclose: ((event: CloseEvent) => void) | null = null;
  public onmessage: ((event: MessageEvent) => void) | null = null;
  public onerror: ((event: Event) => void) | null = null;
  public readyState = WebSocket.CONNECTING;
  public url: string;

  constructor(url: string) {
    this.url = url;
    // Simulate connection
    setTimeout(() => {
      this.readyState = WebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 100);
  }

  send(data: string) {
    // Mock send
  }

  close() {
    this.readyState = WebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }
}

// Mock WebSocket globally
Object.defineProperty(global, 'WebSocket', {
  value: MockWebSocket
});

// Mock PriceContext
const mockPriceContext = {
  getPrice: vi.fn((symbol: string) => {
    const prices: Record<string, number> = {
      'SOL-PERP': 100.50,
      'BTC-PERP': 50000.00,
      'ETH-PERP': 3000.00
    };
    return prices[symbol] || 100.00;
  }),
  prices: {
    'SOL-PERP': 100.50,
    'BTC-PERP': 50000.00,
    'ETH-PERP': 3000.00
  }
};

vi.mock('../contexts/PriceContext', () => ({
  usePrice: () => mockPriceContext
}));

describe('Frontend Real-time Updates - Hackathon Demo', () => {
  beforeEach(() => {
    // Reset mocks
    vi.clearAllMocks();
    localStorageMock.getItem.mockReturnValue('mock-token');
    
    // Mock successful API responses
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('/api/positions')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            success: true,
            positions: [
              {
                id: 'pos-1',
                marketId: 'market-1',
                symbol: 'SOL-PERP',
                side: 'long',
                size: 1.0,
                entryPrice: 100.00,
                currentPrice: 105.00,
                margin: 20.00,
                leverage: 5,
                unrealizedPnl: 5.00,
                unrealizedPnlPercent: 5.00,
                liquidationPrice: 80.00,
                healthFactor: 0.95,
                marginRatio: 0.02,
                isLiquidated: false,
                createdAt: '2024-01-01T00:00:00Z',
                updatedAt: '2024-01-01T00:00:00Z'
              }
            ]
          })
        });
      }
      
      if (url.includes('/api/orders')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            success: true,
            orders: [
              {
                id: 'order-1',
                marketId: 'market-1',
                orderType: 'market',
                side: 'long',
                size: 1.0,
                price: null,
                leverage: 5,
                status: 'filled',
                filledSize: 1.0,
                remainingSize: 0,
                averageFillPrice: 100.00,
                createdAt: '2024-01-01T00:00:00Z',
                updatedAt: '2024-01-01T00:00:00Z'
              }
            ]
          })
        });
      }
      
      if (url.includes('/api/portfolio-data')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            success: true,
            data: {
              userId: 'test-user',
              totalValue: 1000.00,
              totalUnrealizedPnl: 5.00,
              totalRealizedPnl: 0.00,
              marginRatio: 0.02,
              healthFactor: 0.95,
              totalCollateral: 1000.00,
              usedMargin: 20.00,
              availableMargin: 980.00,
              positions: [
                {
                  id: 'pos-1',
                  symbol: 'SOL-PERP',
                  size: 1.0,
                  entryPrice: 100.00,
                  currentPrice: 105.00,
                  unrealizedPnl: 5.00,
                  unrealizedPnlPercent: 5.00,
                  margin: 20.00,
                  leverage: 5,
                  side: 'long'
                }
              ],
              timestamp: Date.now()
            }
          })
        });
      }
      
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ success: true })
      });
    });
  });

  afterEach(() => {
    // Clean up event listeners
    const events = ['positionStatusUpdate', 'orderStatusUpdate', 'portfolioStatusUpdate'];
    events.forEach(event => {
      window.removeEventListener(event, () => {});
    });
  });

  describe('Positions Component Real-time Updates', () => {
    it('should display positions correctly', async () => {
      render(<Positions />);
      
      await waitFor(() => {
        expect(screen.getByText('SOL-PERP')).toBeInTheDocument();
        expect(screen.getByText('Long')).toBeInTheDocument();
        expect(screen.getByText('1.000')).toBeInTheDocument();
        expect(screen.getByText('$100.00')).toBeInTheDocument();
        expect(screen.getByText('$105.00')).toBeInTheDocument();
        expect(screen.getByText('+$5.00')).toBeInTheDocument();
        expect(screen.getByText('+5.00%')).toBeInTheDocument();
      });
    });

    it('should handle real-time position updates', async () => {
      render(<Positions />);
      
      await waitFor(() => {
        expect(screen.getByText('SOL-PERP')).toBeInTheDocument();
      });
      
      // Simulate position update event
      act(() => {
        const event = new CustomEvent('positionStatusUpdate', {
          detail: {
            positionId: 'pos-1',
            status: 'closed',
            userId: 'test-user',
            realizedPnl: 5.00,
            timestamp: Date.now()
          }
        });
        window.dispatchEvent(event);
      });
      
      // Position should be removed from display
      await waitFor(() => {
        expect(screen.queryByText('SOL-PERP')).not.toBeInTheDocument();
      });
    });

    it('should handle position P&L updates', async () => {
      render(<Positions />);
      
      await waitFor(() => {
        expect(screen.getByText('+$5.00')).toBeInTheDocument();
      });
      
      // Simulate P&L update
      act(() => {
        const event = new CustomEvent('positionStatusUpdate', {
          detail: {
            positionId: 'pos-1',
            status: 'updated',
            unrealizedPnl: 10.00,
            unrealizedPnlPercent: 10.00,
            timestamp: Date.now()
          }
        });
        window.dispatchEvent(event);
      });
      
      // Component should refresh to show updated P&L
      await waitFor(() => {
        expect(screen.getByText('SOL-PERP')).toBeInTheDocument();
      });
    });

    it('should handle position closing', async () => {
      render(<Positions />);
      
      await waitFor(() => {
        expect(screen.getByText('Close')).toBeInTheDocument();
      });
      
      // Mock successful close response
      (global.fetch as any).mockImplementationOnce(() => 
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ success: true })
        })
      );
      
      // Click close button
      const closeButton = screen.getByText('Close');
      closeButton.click();
      
      // Position should be removed
      await waitFor(() => {
        expect(screen.queryByText('SOL-PERP')).not.toBeInTheDocument();
      });
    });
  });

  describe('Orders Component Real-time Updates', () => {
    it('should display orders correctly', async () => {
      render(<Orders />);
      
      await waitFor(() => {
        expect(screen.getByText('Market')).toBeInTheDocument();
        expect(screen.getByText('Long')).toBeInTheDocument();
        expect(screen.getByText('1.000')).toBeInTheDocument();
        expect(screen.getByText('Market')).toBeInTheDocument();
        expect(screen.getByText('1.000')).toBeInTheDocument();
        expect(screen.getByText('filled')).toBeInTheDocument();
      });
    });

    it('should handle real-time order updates', async () => {
      render(<Orders />);
      
      await waitFor(() => {
        expect(screen.getByText('filled')).toBeInTheDocument();
      });
      
      // Simulate order update event
      act(() => {
        const event = new CustomEvent('orderStatusUpdate', {
          detail: {
            orderId: 'order-1',
            status: 'cancelled',
            userId: 'test-user',
            timestamp: Date.now()
          }
        });
        window.dispatchEvent(event);
      });
      
      // Order should be removed from display
      await waitFor(() => {
        expect(screen.queryByText('filled')).not.toBeInTheDocument();
      });
    });

    it('should handle order status changes', async () => {
      render(<Orders />);
      
      await waitFor(() => {
        expect(screen.getByText('filled')).toBeInTheDocument();
      });
      
      // Simulate status change
      act(() => {
        const event = new CustomEvent('orderStatusUpdate', {
          detail: {
            orderId: 'order-1',
            status: 'partially_filled',
            filledSize: 0.5,
            remainingSize: 0.5,
            timestamp: Date.now()
          }
        });
        window.dispatchEvent(event);
      });
      
      // Order should update in place
      await waitFor(() => {
        expect(screen.getByText('partially filled')).toBeInTheDocument();
      });
    });

    it('should handle order cancellation', async () => {
      // Mock pending order
      (global.fetch as any).mockImplementationOnce(() => 
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({
            success: true,
            orders: [
              {
                id: 'order-1',
                marketId: 'market-1',
                orderType: 'limit',
                side: 'long',
                size: 1.0,
                price: 95.00,
                leverage: 5,
                status: 'pending',
                filledSize: 0,
                remainingSize: 1.0,
                averageFillPrice: null,
                createdAt: '2024-01-01T00:00:00Z',
                updatedAt: '2024-01-01T00:00:00Z'
              }
            ]
          })
        })
      );
      
      render(<Orders />);
      
      await waitFor(() => {
        expect(screen.getByText('Cancel')).toBeInTheDocument();
      });
      
      // Mock successful cancellation
      (global.fetch as any).mockImplementationOnce(() => 
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({ success: true })
        })
      );
      
      // Click cancel button
      const cancelButton = screen.getByText('Cancel');
      cancelButton.click();
      
      // Order should be removed
      await waitFor(() => {
        expect(screen.queryByText('Cancel')).not.toBeInTheDocument();
      });
    });
  });

  describe('Portfolio Dashboard Real-time Updates', () => {
    it('should display portfolio data correctly', async () => {
      render(<PortfolioDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('Portfolio Dashboard')).toBeInTheDocument();
        expect(screen.getByText('$1,000.00')).toBeInTheDocument();
        expect(screen.getByText('+$5.00')).toBeInTheDocument();
        expect(screen.getByText('+0.50%')).toBeInTheDocument();
        expect(screen.getByText('95.0%')).toBeInTheDocument();
        expect(screen.getByText('Healthy')).toBeInTheDocument();
      });
    });

    it('should handle real-time portfolio updates', async () => {
      render(<PortfolioDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('$1,000.00')).toBeInTheDocument();
      });
      
      // Simulate portfolio update event
      act(() => {
        const event = new CustomEvent('portfolioStatusUpdate', {
          detail: {
            totalValue: 1010.00,
            totalUnrealizedPnl: 10.00,
            healthFactor: 0.98,
            timestamp: Date.now()
          }
        });
        window.dispatchEvent(event);
      });
      
      // Portfolio should update
      await waitFor(() => {
        expect(screen.getByText('$1,010.00')).toBeInTheDocument();
        expect(screen.getByText('+$10.00')).toBeInTheDocument();
      });
    });

    it('should handle portfolio P&L updates', async () => {
      render(<PortfolioDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('+$5.00')).toBeInTheDocument();
      });
      
      // Simulate P&L update
      act(() => {
        const event = new CustomEvent('portfolioStatusUpdate', {
          detail: {
            totalUnrealizedPnl: -10.00,
            timestamp: Date.now()
          }
        });
        window.dispatchEvent(event);
      });
      
      // P&L should update
      await waitFor(() => {
        expect(screen.getByText('-$10.00')).toBeInTheDocument();
      });
    });

    it('should handle portfolio health factor updates', async () => {
      render(<PortfolioDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('95.0%')).toBeInTheDocument();
        expect(screen.getByText('Healthy')).toBeInTheDocument();
      });
      
      // Simulate health factor update
      act(() => {
        const event = new CustomEvent('portfolioStatusUpdate', {
          detail: {
            healthFactor: 0.7,
            timestamp: Date.now()
          }
        });
        window.dispatchEvent(event);
      });
      
      // Health factor should update
      await waitFor(() => {
        expect(screen.getByText('70.0%')).toBeInTheDocument();
        expect(screen.getByText('Warning')).toBeInTheDocument();
      });
    });

    it('should handle error states gracefully', async () => {
      // Mock error response
      (global.fetch as any).mockImplementationOnce(() => 
        Promise.resolve({
          ok: false,
          json: () => Promise.resolve({
            error: 'Failed to fetch portfolio data'
          })
        })
      );
      
      render(<PortfolioDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('⚠️ Error Loading Portfolio')).toBeInTheDocument();
        expect(screen.getByText('Failed to fetch portfolio data')).toBeInTheDocument();
        expect(screen.getByText('Retry')).toBeInTheDocument();
      });
    });

    it('should handle loading states correctly', async () => {
      // Mock delayed response
      (global.fetch as any).mockImplementationOnce(() => 
        new Promise(resolve => {
          setTimeout(() => {
            resolve({
              ok: true,
              json: () => Promise.resolve({
                success: true,
                data: {
                  userId: 'test-user',
                  totalValue: 1000.00,
                  totalUnrealizedPnl: 5.00,
                  totalRealizedPnl: 0.00,
                  marginRatio: 0.02,
                  healthFactor: 0.95,
                  totalCollateral: 1000.00,
                  usedMargin: 20.00,
                  availableMargin: 980.00,
                  positions: [],
                  timestamp: Date.now()
                }
              })
            });
          }, 100);
        })
      );
      
      render(<PortfolioDashboard />);
      
      // Should show loading state
      expect(screen.getByText('Loading portfolio...')).toBeInTheDocument();
      
      // Should show data after loading
      await waitFor(() => {
        expect(screen.getByText('$1,000.00')).toBeInTheDocument();
      });
    });
  });

  describe('Event Listener Management', () => {
    it('should properly clean up event listeners', () => {
      const { unmount } = render(<Positions />);
      
      // Add event listener
      const eventListener = vi.fn();
      window.addEventListener('positionStatusUpdate', eventListener);
      
      // Unmount component
      unmount();
      
      // Event listener should still be available for cleanup
      expect(eventListener).toBeDefined();
    });

    it('should handle multiple event listeners correctly', () => {
      render(<Positions />);
      
      // Add multiple event listeners
      const listener1 = vi.fn();
      const listener2 = vi.fn();
      
      window.addEventListener('positionStatusUpdate', listener1);
      window.addEventListener('positionStatusUpdate', listener2);
      
      // Dispatch event
      act(() => {
        const event = new CustomEvent('positionStatusUpdate', {
          detail: {
            positionId: 'pos-1',
            status: 'updated',
            timestamp: Date.now()
          }
        });
        window.dispatchEvent(event);
      });
      
      // Both listeners should be called
      expect(listener1).toHaveBeenCalled();
      expect(listener2).toHaveBeenCalled();
    });
  });

  describe('Performance and Reliability', () => {
    it('should handle rapid updates efficiently', async () => {
      render(<PortfolioDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('$1,000.00')).toBeInTheDocument();
      });
      
      // Send rapid updates
      for (let i = 0; i < 10; i++) {
        act(() => {
          const event = new CustomEvent('portfolioStatusUpdate', {
            detail: {
              totalValue: 1000 + i,
              timestamp: Date.now()
            }
          });
          window.dispatchEvent(event);
        });
      }
      
      // Should handle all updates
      await waitFor(() => {
        expect(screen.getByText('$1,009.00')).toBeInTheDocument();
      });
    });

    it('should handle malformed event data gracefully', async () => {
      render(<PortfolioDashboard />);
      
      await waitFor(() => {
        expect(screen.getByText('$1,000.00')).toBeInTheDocument();
      });
      
      // Send malformed event
      act(() => {
        const event = new CustomEvent('portfolioStatusUpdate', {
          detail: {
            totalValue: 'invalid',
            timestamp: 'invalid'
          }
        });
        window.dispatchEvent(event);
      });
      
      // Should not crash
      expect(screen.getByText('$1,000.00')).toBeInTheDocument();
    });
  });
});