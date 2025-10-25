/**
 * QuantDesk UI Component Library Example
 * 
 * This example demonstrates how to build a comprehensive UI component library
 * for trading applications using React, TypeScript, and Tailwind CSS.
 */

import React, { useState, useEffect, useCallback } from 'react';
import { ChevronUpIcon, ChevronDownIcon, ChartBarIcon, CurrencyDollarIcon } from '@heroicons/react/24/outline';

// Base UI Components
export interface ButtonProps {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'warning';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  loading?: boolean;
  className?: string;
}

export const Button: React.FC<ButtonProps> = ({
  children,
  onClick,
  variant = 'primary',
  size = 'md',
  disabled = false,
  loading = false,
  className = ''
}) => {
  const baseClasses = 'inline-flex items-center justify-center font-medium rounded-lg transition-colors focus:outline-none focus:ring-2 focus:ring-offset-2';
  
  const variantClasses = {
    primary: 'bg-blue-600 text-white hover:bg-blue-700 focus:ring-blue-500',
    secondary: 'bg-gray-200 text-gray-900 hover:bg-gray-300 focus:ring-gray-500',
    success: 'bg-green-600 text-white hover:bg-green-700 focus:ring-green-500',
    danger: 'bg-red-600 text-white hover:bg-red-700 focus:ring-red-500',
    warning: 'bg-yellow-600 text-white hover:bg-yellow-700 focus:ring-yellow-500'
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm',
    md: 'px-4 py-2 text-base',
    lg: 'px-6 py-3 text-lg'
  };

  const disabledClasses = disabled ? 'opacity-50 cursor-not-allowed' : '';

  return (
    <button
      onClick={onClick}
      disabled={disabled || loading}
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${disabledClasses} ${className}`}
    >
      {loading && (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
        </svg>
      )}
      {children}
    </button>
  );
};

// Card Component
export interface CardProps {
  children: React.ReactNode;
  title?: string;
  className?: string;
  header?: React.ReactNode;
  footer?: React.ReactNode;
}

export const Card: React.FC<CardProps> = ({
  children,
  title,
  className = '',
  header,
  footer
}) => {
  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      {(title || header) && (
        <div className="px-6 py-4 border-b border-gray-200">
          {header || <h3 className="text-lg font-medium text-gray-900">{title}</h3>}
        </div>
      )}
      <div className="px-6 py-4">
        {children}
      </div>
      {footer && (
        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
          {footer}
        </div>
      )}
    </div>
  );
};

// Trading-Specific Components
export interface PriceDisplayProps {
  symbol: string;
  price: number;
  change24h: number;
  volume?: number;
  className?: string;
}

export const PriceDisplay: React.FC<PriceDisplayProps> = ({
  symbol,
  price,
  change24h,
  volume,
  className = ''
}) => {
  const isPositive = change24h >= 0;
  const changeColor = isPositive ? 'text-green-600' : 'text-red-600';
  const changeIcon = isPositive ? ChevronUpIcon : ChevronDownIcon;
  const ChangeIcon = changeIcon;

  return (
    <Card className={className}>
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900">{symbol}</h3>
          <p className="text-2xl font-bold text-gray-900">
            ${price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
        </div>
        <div className="text-right">
          <div className={`flex items-center ${changeColor}`}>
            <ChangeIcon className="h-4 w-4 mr-1" />
            <span className="font-medium">
              {isPositive ? '+' : ''}{change24h.toFixed(2)}%
            </span>
          </div>
          {volume && (
            <p className="text-sm text-gray-500 mt-1">
              Vol: {volume.toLocaleString()}
            </p>
          )}
        </div>
      </div>
    </Card>
  );
};

// Order Form Component
export interface OrderFormProps {
  onSubmit: (order: {
    symbol: string;
    side: 'buy' | 'sell';
    amount: number;
    price?: number;
    type: 'market' | 'limit';
  }) => void;
  loading?: boolean;
  availableSymbols?: string[];
}

export const OrderForm: React.FC<OrderFormProps> = ({
  onSubmit,
  loading = false,
  availableSymbols = ['SOL', 'BTC', 'ETH', 'USDC']
}) => {
  const [symbol, setSymbol] = useState(availableSymbols[0]);
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [amount, setAmount] = useState('');
  const [price, setPrice] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    const orderData = {
      symbol,
      side,
      amount: parseFloat(amount),
      price: orderType === 'limit' ? parseFloat(price) : undefined,
      type: orderType
    };

    onSubmit(orderData);
  };

  return (
    <Card title="Place Order">
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Symbol Selection */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Symbol
          </label>
          <select
            value={symbol}
            onChange={(e) => setSymbol(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {availableSymbols.map(sym => (
              <option key={sym} value={sym}>{sym}</option>
            ))}
          </select>
        </div>

        {/* Buy/Sell Toggle */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Side
          </label>
          <div className="flex space-x-2">
            <Button
              type="button"
              variant={side === 'buy' ? 'success' : 'secondary'}
              onClick={() => setSide('buy')}
              className="flex-1"
            >
              Buy
            </Button>
            <Button
              type="button"
              variant={side === 'sell' ? 'danger' : 'secondary'}
              onClick={() => setSide('sell')}
              className="flex-1"
            >
              Sell
            </Button>
          </div>
        </div>

        {/* Order Type */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Order Type
          </label>
          <div className="flex space-x-2">
            <Button
              type="button"
              variant={orderType === 'market' ? 'primary' : 'secondary'}
              onClick={() => setOrderType('market')}
              className="flex-1"
            >
              Market
            </Button>
            <Button
              type="button"
              variant={orderType === 'limit' ? 'primary' : 'secondary'}
              onClick={() => setOrderType('limit')}
              className="flex-1"
            >
              Limit
            </Button>
          </div>
        </div>

        {/* Amount */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Amount
          </label>
          <input
            type="number"
            step="0.0001"
            min="0"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
            required
          />
        </div>

        {/* Price (for limit orders) */}
        {orderType === 'limit' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Price
            </label>
            <input
              type="number"
              step="0.01"
              min="0"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>
        )}

        {/* Submit Button */}
        <Button
          type="submit"
          variant={side === 'buy' ? 'success' : 'danger'}
          loading={loading}
          className="w-full"
        >
          {side === 'buy' ? 'Buy' : 'Sell'} {symbol}
        </Button>
      </form>
    </Card>
  );
};

// Portfolio Summary Component
export interface PortfolioSummaryProps {
  totalValue: number;
  positions: Array<{
    symbol: string;
    amount: number;
    value: number;
    pnl: number;
  }>;
  className?: string;
}

export const PortfolioSummary: React.FC<PortfolioSummaryProps> = ({
  totalValue,
  positions,
  className = ''
}) => {
  const totalPnl = positions.reduce((sum, pos) => sum + pos.pnl, 0);
  const isPositive = totalPnl >= 0;

  return (
    <Card title="Portfolio Summary" className={className}>
      <div className="text-center mb-6">
        <p className="text-sm text-gray-500">Total Value</p>
        <p className="text-3xl font-bold text-gray-900">
          ${totalValue.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
        </p>
        <p className={`text-sm font-medium ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
          {isPositive ? '+' : ''}${totalPnl.toFixed(2)} P&L
        </p>
      </div>

      <div className="space-y-3">
        {positions.map((position) => (
          <div key={position.symbol} className="flex justify-between items-center py-2 border-b border-gray-100 last:border-b-0">
            <div className="flex items-center">
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mr-3">
                <CurrencyDollarIcon className="h-4 w-4 text-blue-600" />
              </div>
              <div>
                <p className="font-medium text-gray-900">{position.symbol}</p>
                <p className="text-sm text-gray-500">{position.amount.toFixed(4)}</p>
              </div>
            </div>
            <div className="text-right">
              <p className="font-medium text-gray-900">
                ${position.value.toFixed(2)}
              </p>
              <p className={`text-sm ${position.pnl >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                {position.pnl >= 0 ? '+' : ''}${position.pnl.toFixed(2)}
              </p>
            </div>
          </div>
        ))}
      </div>
    </Card>
  );
};

// Market Data Table Component
export interface MarketDataTableProps {
  data: Array<{
    symbol: string;
    price: number;
    change24h: number;
    volume: number;
  }>;
  onSymbolClick?: (symbol: string) => void;
  className?: string;
}

export const MarketDataTable: React.FC<MarketDataTableProps> = ({
  data,
  onSymbolClick,
  className = ''
}) => {
  return (
    <Card title="Market Data" className={className}>
      <div className="overflow-x-auto">
        <table className="min-w-full divide-y divide-gray-200">
          <thead className="bg-gray-50">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Symbol
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Price
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                24h Change
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                Volume
              </th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {data.map((item) => {
              const isPositive = item.change24h >= 0;
              const changeColor = isPositive ? 'text-green-600' : 'text-red-600';
              const ChangeIcon = isPositive ? ChevronUpIcon : ChevronDownIcon;

              return (
                <tr
                  key={item.symbol}
                  className="hover:bg-gray-50 cursor-pointer"
                  onClick={() => onSymbolClick?.(item.symbol)}
                >
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center mr-3">
                        <ChartBarIcon className="h-4 w-4 text-blue-600" />
                      </div>
                      <span className="font-medium text-gray-900">{item.symbol}</span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-900">
                    ${item.price.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className={`flex items-center ${changeColor}`}>
                      <ChangeIcon className="h-4 w-4 mr-1" />
                      <span className="font-medium">
                        {isPositive ? '+' : ''}{item.change24h.toFixed(2)}%
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-gray-900">
                    {item.volume.toLocaleString()}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
};

// Custom Hook for Market Data
export const useMarketData = (symbol: string, interval: number = 5000) => {
  const [data, setData] = useState<{
    price: number;
    change24h: number;
    volume: number;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Simulate API call - replace with actual implementation
      const response = await fetch(`/api/market-data/${symbol}`);
      if (!response.ok) throw new Error('Failed to fetch market data');
      
      const marketData = await response.json();
      setData(marketData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [symbol]);

  useEffect(() => {
    fetchData();
    const intervalId = setInterval(fetchData, interval);
    return () => clearInterval(intervalId);
  }, [fetchData, interval]);

  return { data, loading, error, refetch: fetchData };
};

// Example Usage Component
export const TradingDashboard: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('SOL');
  const { data: marketData } = useMarketData(selectedSymbol);

  const handleOrderSubmit = (order: any) => {
    console.log('Order submitted:', order);
    // Handle order submission
  };

  const mockPositions = [
    { symbol: 'SOL', amount: 10.5, value: 1050, pnl: 50 },
    { symbol: 'BTC', amount: 0.1, value: 3000, pnl: -100 },
    { symbol: 'ETH', amount: 2.0, value: 4000, pnl: 200 }
  ];

  const mockMarketData = [
    { symbol: 'SOL', price: 100, change24h: 5.2, volume: 1000000 },
    { symbol: 'BTC', price: 30000, change24h: -2.1, volume: 5000000 },
    { symbol: 'ETH', price: 2000, change24h: 3.5, volume: 2000000 }
  ];

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-8">Trading Dashboard</h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Market Data */}
          <div className="lg:col-span-2">
            <MarketDataTable
              data={mockMarketData}
              onSymbolClick={setSelectedSymbol}
              className="mb-6"
            />
            
            {marketData && (
              <PriceDisplay
                symbol={selectedSymbol}
                price={marketData.price}
                change24h={marketData.change24h}
                volume={marketData.volume}
              />
            )}
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <OrderForm onSubmit={handleOrderSubmit} />
            <PortfolioSummary
              totalValue={8050}
              positions={mockPositions}
            />
          </div>
        </div>
      </div>
    </div>
  );
};

// Export all components
export default {
  Button,
  Card,
  PriceDisplay,
  OrderForm,
  PortfolioSummary,
  MarketDataTable,
  useMarketData,
  TradingDashboard
};
