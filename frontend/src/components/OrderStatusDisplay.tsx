import React from 'react';
import { useOrderStore, Order } from '../stores/orderStore';
import { useOrderStatusTracking } from '../hooks/useOrderStatus';

export interface OrderStatusDisplayProps {
  orderId: string;
  showDetails?: boolean;
}

export const OrderStatusDisplay: React.FC<OrderStatusDisplayProps> = ({
  orderId,
  showDetails = true
}) => {
  const order = useOrderStore(state => state.getOrder(orderId));
  const { orderStatus, isLoading } = useOrderStatusTracking(orderId);

  if (!order && !orderStatus) {
    return (
      <div className="text-gray-400 text-sm">
        Order not found
      </div>
    );
  }

  const displayOrder = order || orderStatus;
  if (!displayOrder) return null;

  const getStatusColor = (status: Order['status']) => {
    switch (status) {
      case 'pending':
        return 'text-yellow-400';
      case 'filled':
        return 'text-green-400';
      case 'partially_filled':
        return 'text-blue-400';
      case 'cancelled':
        return 'text-gray-400';
      case 'failed':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getStatusIcon = (status: Order['status']) => {
    switch (status) {
      case 'pending':
        return '⏳';
      case 'filled':
        return '✅';
      case 'partially_filled':
        return '🔄';
      case 'cancelled':
        return '❌';
      case 'failed':
        return '💥';
      default:
        return '❓';
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center space-x-2">
          <span className="text-lg">{getStatusIcon(displayOrder.status)}</span>
          <span className={`font-semibold ${getStatusColor(displayOrder.status)}`}>
            {displayOrder.status.replace('_', ' ').toUpperCase()}
          </span>
        </div>
        {isLoading && (
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400"></div>
        )}
      </div>

      <div className="space-y-2 text-sm">
        <div className="flex justify-between">
          <span className="text-gray-400">Order ID:</span>
          <span className="text-white font-mono text-xs">{displayOrder.id}</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-400">Symbol:</span>
          <span className="text-white">{displayOrder.symbol}</span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-400">Side:</span>
          <span className={`font-semibold ${
            displayOrder.side === 'buy' ? 'text-green-400' : 'text-red-400'
          }`}>
            {displayOrder.side.toUpperCase()}
          </span>
        </div>
        
        <div className="flex justify-between">
          <span className="text-gray-400">Size:</span>
          <span className="text-white">{displayOrder.size}</span>
        </div>
        
        {displayOrder.price && (
          <div className="flex justify-between">
            <span className="text-gray-400">Price:</span>
            <span className="text-white">${displayOrder.price.toLocaleString()}</span>
          </div>
        )}
        
        <div className="flex justify-between">
          <span className="text-gray-400">Filled:</span>
          <span className="text-white">
            {displayOrder.filledSize} / {displayOrder.size}
          </span>
        </div>
        
        {displayOrder.averageFillPrice && (
          <div className="flex justify-between">
            <span className="text-gray-400">Avg Fill Price:</span>
            <span className="text-white">${displayOrder.averageFillPrice.toLocaleString()}</span>
          </div>
        )}
        
        {displayOrder.smartContractTx && (
          <div className="flex justify-between">
            <span className="text-gray-400">Smart Contract TX:</span>
            <span className="text-blue-400 font-mono text-xs">
              {displayOrder.smartContractTx.slice(0, 8)}...
            </span>
          </div>
        )}
        
        {displayOrder.smartContractPositionId && (
          <div className="flex justify-between">
            <span className="text-gray-400">Position ID:</span>
            <span className="text-green-400 font-mono text-xs">
              {displayOrder.smartContractPositionId.slice(0, 8)}...
            </span>
          </div>
        )}
        
        {displayOrder.atomicPositionCreation && (
          <div className="flex justify-between">
            <span className="text-gray-400">Atomic Creation:</span>
            <span className="text-green-400">✅ Yes</span>
          </div>
        )}
        
        {displayOrder.errorMessage && (
          <div className="mt-3 p-2 bg-red-900 border border-red-600 rounded">
            <span className="text-red-200 text-xs">
              Error: {displayOrder.errorMessage}
            </span>
          </div>
        )}
      </div>

      {showDetails && (
        <div className="mt-4 pt-3 border-t border-gray-700">
          <div className="text-xs text-gray-400 space-y-1">
            <div>Created: {new Date(displayOrder.createdAt).toLocaleString()}</div>
            <div>Updated: {new Date(displayOrder.updatedAt).toLocaleString()}</div>
          </div>
        </div>
      )}
    </div>
  );
};

export interface OrderListProps {
  symbol?: string;
  status?: Order['status'];
  limit?: number;
}

export const OrderList: React.FC<OrderListProps> = ({
  symbol,
  status,
  limit = 10
}) => {
  const orders = useOrderStore(state => {
    let filteredOrders = state.getAllOrders();
    
    if (symbol) {
      filteredOrders = filteredOrders.filter(order => order.symbol === symbol);
    }
    
    if (status) {
      filteredOrders = filteredOrders.filter(order => order.status === status);
    }
    
    return filteredOrders.slice(0, limit);
  });

  if (orders.length === 0) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 text-center">
        <p className="text-gray-400">No orders found</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {orders.map(order => (
        <OrderStatusDisplay
          key={order.id}
          orderId={order.id}
          showDetails={false}
        />
      ))}
    </div>
  );
};
