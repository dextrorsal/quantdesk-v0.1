/**
 * QuantDesk Frontend UI Component Examples
 * 
 * This file demonstrates reusable React components for building trading interfaces.
 * These components are open source and can be used by the community.
 */

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Badge } from './ui/badge';

// Example: Price Display Component
export const PriceDisplay: React.FC<{
  symbol: string;
  price: number;
  change24h: number;
  className?: string;
}> = ({ symbol, price, change24h, className }) => {
  const isPositive = change24h >= 0;
  
  return (
    <Card className={className}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between">
          <span>{symbol}</span>
          <Badge variant={isPositive ? "default" : "destructive"}>
            {isPositive ? '+' : ''}{change24h.toFixed(2)}%
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">
          ${price.toLocaleString()}
        </div>
      </CardContent>
    </Card>
  );
};

// Example: Trading Button Component
export const TradingButton: React.FC<{
  action: 'buy' | 'sell';
  onClick: () => void;
  disabled?: boolean;
  loading?: boolean;
}> = ({ action, onClick, disabled = false, loading = false }) => {
  const isBuy = action === 'buy';
  
  return (
    <Button
      onClick={onClick}
      disabled={disabled || loading}
      className={`w-full ${
        isBuy 
          ? 'bg-green-600 hover:bg-green-700' 
          : 'bg-red-600 hover:bg-red-700'
      }`}
    >
      {loading ? 'Processing...' : `${isBuy ? 'Buy' : 'Sell'}`}
    </Button>
  );
};

// Example: Order Form Component
export const OrderForm: React.FC<{
  onSubmit: (order: { side: 'buy' | 'sell'; amount: number; price?: number }) => void;
  loading?: boolean;
}> = ({ onSubmit, loading = false }) => {
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [amount, setAmount] = useState('');
  const [price, setPrice] = useState('');
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({
      side,
      amount: parseFloat(amount),
      price: orderType === 'limit' ? parseFloat(price) : undefined
    });
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>Place Order</CardTitle>
      </CardHeader>
      <CardContent>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="flex space-x-2">
            <Button
              type="button"
              variant={side === 'buy' ? 'default' : 'outline'}
              onClick={() => setSide('buy')}
            >
              Buy
            </Button>
            <Button
              type="button"
              variant={side === 'sell' ? 'default' : 'outline'}
              onClick={() => setSide('sell')}
            >
              Sell
            </Button>
          </div>
          
          <div className="flex space-x-2">
            <Button
              type="button"
              variant={orderType === 'market' ? 'default' : 'outline'}
              onClick={() => setOrderType('market')}
            >
              Market
            </Button>
            <Button
              type="button"
              variant={orderType === 'limit' ? 'default' : 'outline'}
              onClick={() => setOrderType('limit')}
            >
              Limit
            </Button>
          </div>

          <Input
            type="number"
            placeholder="Amount"
            value={amount}
            onChange={(e) => setAmount(e.target.value)}
            required
          />

          {orderType === 'limit' && (
            <Input
              type="number"
              placeholder="Price"
              value={price}
              onChange={(e) => setPrice(e.target.value)}
              required
            />
          )}

          <TradingButton
            action={side}
            onClick={() => {}}
            loading={loading}
          />
        </form>
      </CardContent>
    </Card>
  );
};

// Example: Portfolio Summary Component
export const PortfolioSummary: React.FC<{
  totalValue: number;
  positions: Array<{
    symbol: string;
    amount: number;
    value: number;
    pnl: number;
  }>;
}> = ({ totalValue, positions }) => {
  const totalPnl = positions.reduce((sum, pos) => sum + pos.pnl, 0);
  const isPositive = totalPnl >= 0;

  return (
    <Card>
      <CardHeader>
        <CardTitle>Portfolio Summary</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="text-center">
          <div className="text-3xl font-bold">
            ${totalValue.toLocaleString()}
          </div>
          <div className={`text-sm ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
            {isPositive ? '+' : ''}${totalPnl.toFixed(2)} P&L
          </div>
        </div>
        
        <div className="space-y-2">
          {positions.map((position) => (
            <div key={position.symbol} className="flex justify-between">
              <span>{position.symbol}</span>
              <span>${position.value.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
};

// Example: Market Data Hook
export const useMarketData = (symbol: string) => {
  const [data, setData] = useState<{
    price: number;
    change24h: number;
    volume: number;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        // Example API call - replace with actual implementation
        const response = await fetch(`/api/market-data/${symbol}`);
        const marketData = await response.json();
        setData(marketData);
        setError(null);
      } catch (err) {
        setError('Failed to fetch market data');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 5000); // Update every 5 seconds
    
    return () => clearInterval(interval);
  }, [symbol]);

  return { data, loading, error };
};

export default {
  PriceDisplay,
  TradingButton,
  OrderForm,
  PortfolioSummary,
  useMarketData
};
