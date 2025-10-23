import React, { useState, useCallback } from 'react';
import { useOrderStore, Order } from '../stores/orderStore';
import { useOrderStatus } from '../hooks/useOrderStatus';
import { logger } from '../utils/logger';

export interface OrderFormData {
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  price?: number;
  orderType: 'market' | 'limit';
  leverage: number;
}

export interface OrderPlacementProps {
  symbol: string;
  currentPrice: number;
  onOrderPlaced?: (order: Order) => void;
  onOrderError?: (error: string) => void;
}

export const OrderPlacement: React.FC<OrderPlacementProps> = ({
  symbol,
  currentPrice,
  onOrderPlaced,
  onOrderError
}) => {
  const [formData, setFormData] = useState<OrderFormData>({
    symbol,
    side: 'buy',
    size: 0,
    price: undefined,
    orderType: 'market',
    leverage: 1
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { addOrder, setLoading, setError: setStoreError } = useOrderStore();
  const { isConnected } = useOrderStatus();

  // Handle form input changes
  const handleInputChange = useCallback((field: keyof OrderFormData, value: any) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  }, []);

  // Handle order submission
  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!isConnected) {
      const error = 'WebSocket not connected. Please check your connection.';
      setError(error);
      setStoreError(error);
      onOrderError?.(error);
      return;
    }

    // Validation
    if (formData.size <= 0) {
      const error = 'Size must be greater than 0';
      setError(error);
      setStoreError(error);
      onOrderError?.(error);
      return;
    }

    if (formData.orderType === 'limit' && (!formData.price || formData.price <= 0)) {
      const error = 'Limit orders require a positive price';
      setError(error);
      setStoreError(error);
      onOrderError?.(error);
      return;
    }

    if (formData.leverage < 1 || formData.leverage > 100) {
      const error = 'Leverage must be between 1 and 100';
      setError(error);
      setStoreError(error);
      onOrderError?.(error);
      return;
    }

    setIsSubmitting(true);
    setLoading(true);
    setError(null);
    setStoreError(null);

    try {
      // Create order object
      const orderId = `order_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      const newOrder: Order = {
        id: orderId,
        symbol: formData.symbol,
        side: formData.side,
        size: formData.size,
        price: formData.price,
        orderType: formData.orderType,
        status: 'pending',
        filledSize: 0,
        averageFillPrice: undefined,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };

      // Add to store immediately for optimistic UI
      addOrder(newOrder);

      // Submit order to backend
      const response = await fetch('/api/orders', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          symbol: formData.symbol,
          side: formData.side,
          size: formData.size,
          orderType: formData.orderType,
          price: formData.price,
          leverage: formData.leverage
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        
        // Handle specific error codes
        switch (errorData.code) {
          case 'MISSING_FIELDS':
            throw new Error('Please fill in all required fields');
          case 'INVALID_SIZE':
            throw new Error('Size must be a positive number');
          case 'INVALID_PRICE':
            throw new Error('Price must be a positive number for limit orders');
          case 'INVALID_LEVERAGE':
            throw new Error('Leverage must be between 1 and 100');
          case 'INVALID_SIDE':
            throw new Error('Side must be either Buy or Sell');
          case 'INVALID_ORDER_TYPE':
            throw new Error('Order type must be either Market or Limit');
          case 'PRICE_UNAVAILABLE':
            throw new Error('Market data is currently unavailable. Please try again later.');
          case 'SMART_CONTRACT_ERROR':
            throw new Error('Order was created but failed to execute on blockchain. Please check your order status.');
          default:
            throw new Error(errorData.details || errorData.error || 'Failed to place order');
        }
      }

      const result = await response.json();
      
      // Update order with backend response
      if (result.orderId) {
        addOrder({
          ...newOrder,
          id: result.orderId
        });
      }

      logger.info('Order placed successfully:', result);
      onOrderPlaced?.(newOrder);

      // Reset form
      setFormData(prev => ({
        ...prev,
        size: 0,
        price: undefined
      }));

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
      logger.error('Failed to place order:', error);
      setError(errorMessage);
      setStoreError(errorMessage);
      onOrderError?.(errorMessage);
    } finally {
      setIsSubmitting(false);
      setLoading(false);
    }
  }, [formData, isConnected, addOrder, setLoading, setError, onOrderPlaced, onOrderError]);

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h3 className="text-lg font-semibold mb-4 text-white">Place Order</h3>
      
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Order Type */}
        <div>
          <label className="block text-sm font-medium mb-2 text-gray-300">Order Type</label>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => handleInputChange('orderType', 'market')}
              className={`py-2 px-3 rounded-lg text-sm font-medium transition-colors ${
                formData.orderType === 'market'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Market
            </button>
            <button
              type="button"
              onClick={() => handleInputChange('orderType', 'limit')}
              className={`py-2 px-3 rounded-lg text-sm font-medium transition-colors ${
                formData.orderType === 'limit'
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Limit
            </button>
          </div>
        </div>

        {/* Side */}
        <div>
          <label className="block text-sm font-medium mb-2 text-gray-300">Side</label>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => handleInputChange('side', 'buy')}
              className={`py-2 px-3 rounded-lg text-sm font-medium transition-colors ${
                formData.side === 'buy'
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Buy
            </button>
            <button
              type="button"
              onClick={() => handleInputChange('side', 'sell')}
              className={`py-2 px-3 rounded-lg text-sm font-medium transition-colors ${
                formData.side === 'sell'
                  ? 'bg-red-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              Sell
            </button>
          </div>
        </div>

        {/* Size */}
        <div>
          <label htmlFor="size" className="block text-sm font-medium mb-2 text-gray-300">
            Size
          </label>
          <input
            id="size"
            type="number"
            step="0.001"
            min="0"
            placeholder="0.00"
            value={formData.size || ''}
            onChange={(e) => handleInputChange('size', parseFloat(e.target.value) || 0)}
            className="w-full py-2 px-3 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
            required
          />
        </div>

        {/* Price (for limit orders) */}
        {formData.orderType === 'limit' && (
          <div>
            <label htmlFor="price" className="block text-sm font-medium mb-2 text-gray-300">
              Price
            </label>
            <input
              id="price"
              type="number"
              step="0.01"
              min="0"
              placeholder="0.00"
              value={formData.price || ''}
              onChange={(e) => handleInputChange('price', parseFloat(e.target.value) || undefined)}
              className="w-full py-2 px-3 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
              required
            />
            <p className="text-xs text-gray-400 mt-1">
              Current price: ${currentPrice.toLocaleString()}
            </p>
          </div>
        )}

        {/* Leverage */}
        <div>
          <label htmlFor="leverage" className="block text-sm font-medium mb-2 text-gray-300">
            Leverage
          </label>
          <input
            id="leverage"
            type="number"
            min="1"
            max="100"
            value={formData.leverage}
            onChange={(e) => handleInputChange('leverage', parseInt(e.target.value) || 1)}
            className="w-full py-2 px-3 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
          />
        </div>

        {/* Connection Status */}
        {!isConnected && (
          <div className="bg-yellow-900 border border-yellow-600 rounded-lg p-3">
            <p className="text-yellow-200 text-sm">
              ⚠️ WebSocket disconnected. Order status updates may not work properly.
            </p>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="bg-red-900 border border-red-600 rounded-lg p-3">
            <p className="text-red-200 text-sm">
              ❌ {error}
            </p>
          </div>
        )}

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isSubmitting || !isConnected}
          className={`w-full py-3 rounded-lg font-semibold transition-colors ${
            isSubmitting || !isConnected
              ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
              : formData.side === 'buy'
              ? 'bg-green-600 text-white hover:bg-green-700'
              : 'bg-red-600 text-white hover:bg-red-700'
          }`}
        >
          {isSubmitting ? 'Placing Order...' : `${formData.side === 'buy' ? 'Buy' : 'Sell'} ${formData.symbol}`}
        </button>
      </form>
    </div>
  );
};
