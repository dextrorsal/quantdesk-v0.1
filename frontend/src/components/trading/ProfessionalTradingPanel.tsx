  import React, { useState, useCallback, useMemo } from 'react';
  import { useWallet } from '@solana/wallet-adapter-react';
  import { useAccount } from '../../contexts/AccountContext';
  import { useMarkets } from '../../contexts/MarketContext';
  import { TrendingUp, TrendingDown, Settings } from 'lucide-react';
  
  // Terminal-style form components

/**
 * Professional Trading Panel inspired by Drift.trade
 * Features:
 * - Clean, professional UI with proper spacing
 * - Real-time PnL updates
 * - Advanced order types
 * - Risk management indicators
 * - Responsive design
 */

interface OrderFormData {
  side: 'long' | 'short';
  size: number;
  price?: number;
  orderType: 'market' | 'limit' | 'stop';
  leverage: number;
  reduceOnly?: boolean;
}

interface ProfessionalTradingPanelProps {
  symbol: string;
  currentPrice: number;
  onOrderPlaced?: (order: any) => void;
  className?: string;
}

const ProfessionalTradingPanel: React.FC<ProfessionalTradingPanelProps> = ({
  symbol,
  currentPrice,
  onOrderPlaced,
  className = ''
}) => {
  const { connected, publicKey } = useWallet();
  const { accountState, totalBalance, accountHealth } = useAccount();
  
  const [formData, setFormData] = useState<OrderFormData>({
    side: 'long',
    size: 0,
    price: undefined,
    orderType: 'market',
    leverage: 1,
    reduceOnly: false
  });
  
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // Get current market data
  const currentMarket = useMemo(() => 
    markets.find(m => m.symbol === symbol), [markets, symbol]
  );

  // Calculate position size percentages
  const sizePercentages = [25, 50, 75, 100];
  
  // Calculate available margin
  const availableMargin = useMemo(() => {
    if (!accountState) return 0;
    return accountState.totalCollateral * 0.95; // 95% of collateral available
  }, [accountState]);

  // Calculate max position size
  const maxPositionSize = useMemo(() => {
    if (!currentMarket || !availableMargin) return 0;
    return (availableMargin * formData.leverage) / currentPrice;
  }, [currentMarket, availableMargin, formData.leverage, currentPrice]);

  // Handle form input changes
  const handleInputChange = useCallback((field: keyof OrderFormData, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    setError(null);
  }, []);

  // Handle size percentage click
  const handleSizePercentage = useCallback((percentage: number) => {
    const size = (maxPositionSize * percentage) / 100;
    setFormData(prev => ({
      ...prev,
      size: Math.round(size * 1000) / 1000 // Round to 3 decimal places
    }));
  }, [maxPositionSize]);

  // Handle order submission
  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!connected || !publicKey) {
      setError('Please connect your wallet');
      return;
    }

    if (!accountState?.canTrade) {
      setError('Account not ready for trading');
      return;
    }

    // Validation
    if (formData.size <= 0) {
      setError('Size must be greater than 0');
      return;
    }

    if (formData.orderType === 'limit' && (!formData.price || formData.price <= 0)) {
      setError('Limit orders require a positive price');
      return;
    }

    if (formData.leverage < 1 || formData.leverage > 20) {
      setError('Leverage must be between 1 and 20');
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      // TODO: Implement actual order placement with smart contract
      console.log('Placing order:', formData);
      
      // Simulate order placement
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      onOrderPlaced?.(formData);
      
      // Reset form
      setFormData(prev => ({
        ...prev,
        size: 0,
        price: undefined
      }));
      
    } catch (err: any) {
      setError(err.message || 'Failed to place order');
    } finally {
      setIsSubmitting(false);
    }
  }, [connected, publicKey, accountState, formData, onOrderPlaced]);

  // Calculate estimated PnL
  const estimatedPnL = useMemo(() => {
    if (!formData.size || !currentPrice) return 0;
    
    const priceChange = formData.side === 'long' ? 0.01 : -0.01; // 1% price change
    const newPrice = currentPrice * (1 + priceChange);
    const pnl = formData.side === 'long' 
      ? (newPrice - currentPrice) * formData.size * formData.leverage
      : (currentPrice - newPrice) * formData.size * formData.leverage;
    
    return pnl;
  }, [formData.size, formData.side, formData.leverage, currentPrice]);

  return (
    <div className={`bg-gray-900 border border-gray-700 rounded-lg ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-white">Trade {symbol}</h3>
            <p className="text-sm text-gray-400">${currentPrice.toFixed(2)}</p>
          </div>
          <button
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="p-2 text-gray-400 hover:text-white transition-colors"
          >
            <Settings className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Account Status */}
      {accountState && (
        <div className="p-4 border-b border-gray-700">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Account Health</span>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                accountHealth > 80 ? 'bg-green-500' : 
                accountHealth > 50 ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="text-white">{accountHealth.toFixed(1)}%</span>
            </div>
          </div>
          <div className="flex items-center justify-between text-sm mt-1">
            <span className="text-gray-400">Available Margin</span>
            <span className="text-white">${availableMargin.toFixed(2)}</span>
          </div>
        </div>
      )}

      {/* Order Form */}
      <form onSubmit={handleSubmit} className="p-4 space-y-4">
        {/* Side Selection */}
        <div className="grid grid-cols-2 gap-2">
          <button
            type="button"
            onClick={() => handleInputChange('side', 'long')}
            className={`p-3 rounded-lg font-semibold transition-colors ${
              formData.side === 'long'
                ? 'bg-green-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            <TrendingUp className="w-4 h-4 inline mr-2" />
            Long
          </button>
          <button
            type="button"
            onClick={() => handleInputChange('side', 'short')}
            className={`p-3 rounded-lg font-semibold transition-colors ${
              formData.side === 'short'
                ? 'bg-red-600 text-white'
                : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            <TrendingDown className="w-4 h-4 inline mr-2" />
            Short
          </button>
        </div>

        {/* Order Type */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Order Type</label>
          <select
            value={formData.orderType}
            onChange={(e) => handleInputChange('orderType', e.target.value)}
                className="w-full p-3 bg-gray-800 border border-gray-600 text-white focus:border-blue-500 focus:outline-none font-mono transition-all"
          >
            <option value="market">Market</option>
            <option value="limit">Limit</option>
            <option value="stop">Stop</option>
          </select>
        </div>

        {/* Size Input */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Size</label>
          <div className="relative">
            <input
              type="number"
              value={formData.size || ''}
              onChange={(e) => handleInputChange('size', parseFloat(e.target.value) || 0)}
              placeholder="0.00"
              step="0.001"
              min="0"
              className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-blue-500 focus:outline-none pr-16"
            />
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 text-sm">
              {symbol.split('-')[0]}
            </div>
          </div>
          
          {/* Size Percentage Buttons */}
          <div className="grid grid-cols-4 gap-2 mt-2">
            {sizePercentages.map(percentage => (
              <button
                key={percentage}
                type="button"
                onClick={() => handleSizePercentage(percentage)}
                className="p-2 text-xs bg-gray-700 text-gray-300 rounded hover:bg-gray-600 transition-colors"
              >
                {percentage}%
              </button>
            ))}
          </div>
        </div>

        {/* Price Input (for limit orders) */}
        {formData.orderType === 'limit' && (
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">Price</label>
            <input
              type="number"
              value={formData.price || ''}
              onChange={(e) => handleInputChange('price', parseFloat(e.target.value) || undefined)}
              placeholder="0.00"
              step="0.01"
              min="0"
                className="w-full p-3 bg-gray-800 border border-gray-600 text-white focus:border-blue-500 focus:outline-none font-mono transition-all"
            />
          </div>
        )}

        {/* Leverage */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">Leverage</label>
          <div className="flex items-center space-x-2">
            <input
              type="range"
              min="1"
              max="20"
              value={formData.leverage}
              onChange={(e) => handleInputChange('leverage', parseInt(e.target.value))}
              className="flex-1"
            />
            <span className="text-white font-semibold min-w-[3rem] text-right">
              {formData.leverage}x
            </span>
          </div>
        </div>

        {/* Advanced Options */}
        {showAdvanced && (
          <div className="space-y-3 pt-3 border-t border-gray-700">
            <div className="flex items-center">
              <input
                type="checkbox"
                id="reduceOnly"
                checked={formData.reduceOnly}
                onChange={(e) => handleInputChange('reduceOnly', e.target.checked)}
                className="mr-2"
              />
              <label htmlFor="reduceOnly" className="text-sm text-gray-300">
                Reduce Only
              </label>
            </div>
          </div>
        )}

        {/* Estimated PnL */}
        {formData.size > 0 && (
          <div className="p-3 bg-gray-800 rounded-lg">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400">Est. PnL (1% move)</span>
              <span className={`font-semibold ${
                estimatedPnL >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                ${estimatedPnL.toFixed(2)}
              </span>
            </div>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="p-3 bg-red-900/20 border border-red-500 rounded-lg">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={!connected || isSubmitting || !accountState?.canTrade}
          className={`w-full p-3 rounded-lg font-semibold transition-colors ${
            formData.side === 'long'
              ? 'bg-green-600 hover:bg-green-700 text-white'
              : 'bg-red-600 hover:bg-red-700 text-white'
          } disabled:bg-gray-600 disabled:cursor-not-allowed`}
        >
          {isSubmitting ? (
            <div className="flex items-center justify-center">
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
              Placing Order...
            </div>
          ) : (
            `${formData.side === 'long' ? 'Buy' : 'Sell'} ${symbol}`
          )}
        </button>
      </form>

      {/* Footer */}
      <div className="p-4 border-t border-gray-700 text-xs text-gray-400">
        <p>Max position size: {maxPositionSize.toFixed(3)} {symbol.split('-')[0]}</p>
        <p>Required margin: ${(formData.size * currentPrice / formData.leverage).toFixed(2)}</p>
      </div>
    </div>
  );
};

export default ProfessionalTradingPanel;
