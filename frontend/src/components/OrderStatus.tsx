import React, { useState, useEffect } from 'react';
import { useOrderUpdates, OrderStatusUpdate } from '../hooks/useOrderUpdates';

interface OrderStatusProps {
  orderId: string;
  symbol?: string;
  className?: string;
}

export const OrderStatus: React.FC<OrderStatusProps> = ({ 
  orderId, 
  symbol, 
  className = '' 
}) => {
  const [currentStatus, setCurrentStatus] = useState<OrderStatusUpdate | null>(null);
  const { subscribeToOrderUpdates } = useOrderUpdates();

  useEffect(() => {
    const unsubscribe = subscribeToOrderUpdates((update) => {
      if (update.orderId === orderId) {
        setCurrentStatus(update);
      }
    });

    return unsubscribe;
  }, [orderId, subscribeToOrderUpdates]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'pending':
        return 'text-yellow-500';
      case 'filled':
        return 'text-green-500';
      case 'partially_filled':
        return 'text-blue-500';
      case 'cancelled':
        return 'text-red-500';
      case 'expired':
        return 'text-gray-500';
      default:
        return 'text-gray-500';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'pending':
        return 'Pending';
      case 'filled':
        return 'Filled';
      case 'partially_filled':
        return 'Partially Filled';
      case 'cancelled':
        return 'Cancelled';
      case 'expired':
        return 'Expired';
      default:
        return status;
    }
  };

  if (!currentStatus) {
    return (
      <span className={`text-gray-500 ${className}`}>
        {symbol ? `${symbol} Order` : 'Order'}
      </span>
    );
  }

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <span className={`font-medium ${getStatusColor(currentStatus.status)}`}>
        {getStatusText(currentStatus.status)}
      </span>
      
      {currentStatus.filledSize && (
        <span className="text-sm text-gray-500">
          ({currentStatus.filledSize} filled)
        </span>
      )}
      
      {currentStatus.averageFillPrice && (
        <span className="text-sm text-gray-500">
          @ ${currentStatus.averageFillPrice.toFixed(2)}
        </span>
      )}
    </div>
  );
};

interface OrderUpdatesListProps {
  symbol?: string;
  maxItems?: number;
  className?: string;
}

export const OrderUpdatesList: React.FC<OrderUpdatesListProps> = ({ 
  symbol, 
  maxItems = 10,
  className = '' 
}) => {
  const { orderUpdates } = useOrderUpdates();
  
  const filteredUpdates = symbol 
    ? orderUpdates.filter(update => update.symbol === symbol)
    : orderUpdates;

  const recentUpdates = filteredUpdates.slice(0, maxItems);

  if (recentUpdates.length === 0) {
    return (
      <div className={`text-gray-500 text-sm ${className}`}>
        No recent order updates
      </div>
    );
  }

  return (
    <div className={`space-y-2 ${className}`}>
      <h3 className="font-medium text-gray-900">Recent Order Updates</h3>
      {recentUpdates.map((update) => (
        <div key={`${update.orderId}-${update.timestamp}`} className="flex justify-between items-center p-2 bg-gray-50 rounded">
          <div className="flex items-center space-x-2">
            <span className="text-sm font-medium">{update.symbol}</span>
            <OrderStatus orderId={update.orderId} />
          </div>
          <span className="text-xs text-gray-500">
            {new Date(update.timestamp).toLocaleTimeString()}
          </span>
        </div>
      ))}
    </div>
  );
};
