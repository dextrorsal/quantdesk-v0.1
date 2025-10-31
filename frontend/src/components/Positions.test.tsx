import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Positions from '../src/components/Positions';

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
  value: localStorageMock,
});

// Mock console methods
const consoleSpy = {
  log: vi.spyOn(console, 'log').mockImplementation(() => {}),
  error: vi.spyOn(console, 'error').mockImplementation(() => {}),
  warn: vi.spyOn(console, 'warn').mockImplementation(() => {}),
};

describe('Positions Component', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorageMock.getItem.mockReturnValue('test-token');
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should render loading state initially', () => {
    render(<Positions />);
    
    expect(screen.getByText('Loading positions...')).toBeInTheDocument();
    expect(screen.getByRole('status')).toBeInTheDocument();
  });

  it('should render positions table when data is loaded', async () => {
    const mockPositions = [
      {
        id: '1',
        marketId: 'BTC-PERP',
        symbol: 'BTC-PERP',
        side: 'long',
        size: 1.0,
        entryPrice: 50000,
        currentPrice: 55000,
        margin: 5000,
        leverage: 10,
        unrealizedPnl: 5000,
        unrealizedPnlPercent: 100,
        liquidationPrice: 45000,
        healthFactor: 1.1,
        marginRatio: 100,
        isLiquidated: false,
        createdAt: '2024-01-01T00:00:00Z',
        updatedAt: '2024-01-01T00:00:00Z'
      }
    ];

    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        positions: mockPositions,
        validation: { isValid: true, errors: [] }
      })
    });

    render(<Positions />);

    await waitFor(() => {
      expect(screen.getByText('BTC-PERP')).toBeInTheDocument();
      expect(screen.getByText('Long')).toBeInTheDocument();
      expect(screen.getByText('1.000')).toBeInTheDocument();
      expect(screen.getByText('$50,000.00')).toBeInTheDocument();
      expect(screen.getByText('$55,000.00')).toBeInTheDocument();
      expect(screen.getByText('+$5,000.00')).toBeInTheDocument();
      expect(screen.getByText('+100.00%')).toBeInTheDocument();
    });
  });

  it('should render empty state when no positions', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        positions: [],
        message: 'No open positions found'
      })
    });

    render(<Positions />);

    await waitFor(() => {
      expect(screen.getByText('No open positions yet')).toBeInTheDocument();
      expect(screen.getByText('Start trading to see your positions here')).toBeInTheDocument();
    });
  });

  it('should render error state when API fails', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: async () => ({ error: 'Database connection failed' })
    });

    render(<Positions />);

    await waitFor(() => {
      expect(screen.getByText('⚠️ Error Loading Positions')).toBeInTheDocument();
      expect(screen.getByText('Database connection failed')).toBeInTheDocument();
      expect(screen.getByText('Retry')).toBeInTheDocument();
    });
  });

  it('should render validation warnings when P&L validation fails', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        positions: [],
        validation: {
          isValid: false,
          errors: ['Position 1: Invalid unrealized P&L', 'Position 2: Invalid liquidation price']
        }
      })
    });

    render(<Positions />);

    await waitFor(() => {
      expect(screen.getByText('P&L Validation Warnings:')).toBeInTheDocument();
      expect(screen.getByText('• Position 1: Invalid unrealized P&L')).toBeInTheDocument();
      expect(screen.getByText('• Position 2: Invalid liquidation price')).toBeInTheDocument();
    });
  });

  it('should handle position closing', async () => {
    const mockPositions = [
      {
        id: '1',
        marketId: 'BTC-PERP',
        symbol: 'BTC-PERP',
        side: 'long',
        size: 1.0,
        entryPrice: 50000,
        currentPrice: 55000,
        margin: 5000,
        leverage: 10,
        unrealizedPnl: 5000,
        unrealizedPnlPercent: 100,
        liquidationPrice: 45000,
        healthFactor: 1.1,
        marginRatio: 100,
        isLiquidated: false,
        createdAt: '2024-01-01T00:00:00Z',
        updatedAt: '2024-01-01T00:00:00Z'
      }
    ];

    // Mock initial fetch
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        positions: mockPositions,
        validation: { isValid: true, errors: [] }
      })
    });

    render(<Positions />);

    await waitFor(() => {
      expect(screen.getByText('BTC-PERP')).toBeInTheDocument();
    });

    // Mock close position API call
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        message: 'Position closed successfully',
        positionId: '1'
      })
    });

    const closeButton = screen.getByText('Close');
    fireEvent.click(closeButton);

    await waitFor(() => {
      expect(screen.getByText('No open positions yet')).toBeInTheDocument();
    });
  });

  it('should handle liquidated positions', async () => {
    const mockPositions = [
      {
        id: '1',
        marketId: 'BTC-PERP',
        symbol: 'BTC-PERP',
        side: 'long',
        size: 1.0,
        entryPrice: 50000,
        currentPrice: 45000,
        margin: 5000,
        leverage: 10,
        unrealizedPnl: -5000,
        unrealizedPnlPercent: -100,
        liquidationPrice: 45000,
        healthFactor: 0,
        marginRatio: -100,
        isLiquidated: true,
        createdAt: '2024-01-01T00:00:00Z',
        updatedAt: '2024-01-01T00:00:00Z'
      }
    ];

    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        positions: mockPositions,
        validation: { isValid: true, errors: [] }
      })
    });

    render(<Positions />);

    await waitFor(() => {
      expect(screen.getByText('LIQUIDATED')).toBeInTheDocument();
      expect(screen.getByText('Liquidated')).toBeInTheDocument();
      expect(screen.getByText('0.0%')).toBeInTheDocument();
    });
  });

  it('should handle authentication errors', async () => {
    localStorageMock.getItem.mockReturnValue(null);

    render(<Positions />);

    await waitFor(() => {
      expect(screen.getByText('⚠️ Error Loading Positions')).toBeInTheDocument();
      expect(screen.getByText('Authentication required. Please connect your wallet.')).toBeInTheDocument();
    });
  });

  it('should retry on error', async () => {
    (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

    render(<Positions />);

    await waitFor(() => {
      expect(screen.getByText('⚠️ Error Loading Positions')).toBeInTheDocument();
    });

    // Mock successful retry
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        positions: [],
        validation: { isValid: true, errors: [] }
      })
    });

    const retryButton = screen.getByText('Retry');
    fireEvent.click(retryButton);

    await waitFor(() => {
      expect(screen.getByText('No open positions yet')).toBeInTheDocument();
    });
  });

  it('should handle real-time position updates', async () => {
    const mockPositions = [
      {
        id: '1',
        marketId: 'BTC-PERP',
        symbol: 'BTC-PERP',
        side: 'long',
        size: 1.0,
        entryPrice: 50000,
        currentPrice: 55000,
        margin: 5000,
        leverage: 10,
        unrealizedPnl: 5000,
        unrealizedPnlPercent: 100,
        liquidationPrice: 45000,
        healthFactor: 1.1,
        marginRatio: 100,
        isLiquidated: false,
        createdAt: '2024-01-01T00:00:00Z',
        updatedAt: '2024-01-01T00:00:00Z'
      }
    ];

    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        positions: mockPositions,
        validation: { isValid: true, errors: [] }
      })
    });

    render(<Positions />);

    await waitFor(() => {
      expect(screen.getByText('BTC-PERP')).toBeInTheDocument();
    });

    // Simulate position close event
    const positionUpdateEvent = new CustomEvent('positionStatusUpdate', {
      detail: {
        positionId: '1',
        status: 'closed',
        userId: 'test-user',
        timestamp: '2024-01-01T00:00:00Z'
      }
    });

    window.dispatchEvent(positionUpdateEvent);

    await waitFor(() => {
      expect(screen.getByText('No open positions yet')).toBeInTheDocument();
    });
  });
});
