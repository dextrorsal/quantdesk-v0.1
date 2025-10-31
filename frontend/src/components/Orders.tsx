import React, { useState, useEffect } from 'react'

interface Order {
  id: string;
  marketId: string;
  orderType: 'market' | 'limit' | 'stop_loss' | 'take_profit' | 'trailing_stop' | 'post_only' | 'ioc' | 'fok';
  side: 'long' | 'short';
  size: number;
  price?: number;
  stopPrice?: number;
  trailingDistance?: number;
  leverage: number;
  status: 'pending' | 'filled' | 'partially_filled' | 'cancelled' | 'failed';
  filledSize?: number;
  remainingSize: number;
  averageFillPrice?: number;
  createdAt: string;
  updatedAt?: string;
  expiresAt?: string;
  filledAt?: string;
  cancelledAt?: string;
}

const Orders: React.FC = () => {
  const [orders, setOrders] = useState<Order[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadOrders()

        // 🚀 NEW: Listen for real-time order updates
        const handleOrderUpdate = (event: CustomEvent) => {
          const update = event.detail;
          console.log('📋 Received order update:', update);
          
          if (update.status === 'cancelled' || update.status === 'filled' || update.status === 'failed') {
            // Remove completed orders from list
            setOrders(prev => prev.filter(o => o.id !== update.orderId));
          } else {
            // Update order in place for other status changes
            setOrders(prev => prev.map(o => 
              o.id === update.orderId 
                ? { ...o, status: update.status, ...update }
                : o
            ));
          }
        };

        window.addEventListener('orderStatusUpdate', handleOrderUpdate as EventListener);

        return () => {
          window.removeEventListener('orderStatusUpdate', handleOrderUpdate as EventListener);
        };
  }, [])

  const loadOrders = async () => {
    try {
      setLoading(true)
      setError(null)
      
      const response = await fetch('/api/orders', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        if (data.success && data.orders) {
          setOrders(data.orders);
        } else {
          setError('Failed to load orders data');
          setOrders([]);
        }
      } else {
        const errorData = await response.json().catch(() => ({}));
        setError(errorData.error || `Failed to fetch orders: ${response.statusText}`);
        setOrders([]);
      }
    } catch (err: any) {
      console.error('Error loading orders:', err)
      setError('Network error: Unable to fetch orders');
      setOrders([]);
    } finally {
      setLoading(false);
    }
  }

  const handleCancelOrder = async (orderId: string) => {
    try {
      const response = await fetch(`/api/orders/${orderId}`, {
        method: 'DELETE',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        // Remove cancelled order from list
        setOrders(prev => prev.filter(order => order.id !== orderId));
        console.log('Order cancelled successfully');
      } else {
        const errorData = await response.json().catch(() => ({}));
        const errorMessage = errorData.error || 'Failed to cancel order';
        alert(`Order cancellation failed: ${errorMessage}`);
      }
    } catch (err: any) {
      console.error('Error cancelling order:', err)
      alert('Network error: Unable to cancel order');
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'filled': return 'text-green-400 bg-green-900/30 border border-green-500/30'
      case 'partially_filled': return 'text-yellow-400 bg-yellow-900/30 border border-yellow-500/30'
      case 'pending': return 'text-blue-400 bg-blue-900/30 border border-blue-500/30'
      case 'cancelled': return 'text-gray-400 bg-gray-900/30 border border-gray-500/30'
      case 'failed': return 'text-red-400 bg-red-900/30 border border-red-500/30'
      default: return 'text-gray-400 bg-gray-900/30 border border-gray-500/30'
    }
  }

  const getSideColor = (side: string) => {
    return side === 'long' 
      ? 'text-green-400 bg-green-900/30 border border-green-500/30' 
      : 'text-red-400 bg-red-900/30 border border-red-500/30'
  }

  const getOrderTypeDisplay = (orderType: string) => {
    switch (orderType) {
      case 'market': return 'Market'
      case 'limit': return 'Limit'
      case 'stop_loss': return 'Stop Loss'
      case 'take_profit': return 'Take Profit'
      case 'trailing_stop': return 'Trailing Stop'
      case 'post_only': return 'Post Only'
      case 'ioc': return 'IOC'
      case 'fok': return 'FOK'
      default: return orderType
    }
  }

  if (loading) {
    return (
      <div className="h-full bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <div className="text-gray-400">Loading orders...</div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="h-full bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-400 mb-2">⚠️ Error Loading Orders</div>
          <div className="text-gray-400 text-sm mb-4">{error}</div>
          <button 
            onClick={loadOrders}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full bg-gray-900 flex flex-col">
      <div className="flex-1 flex items-center justify-center">
        {orders.length > 0 ? (
          <div className="overflow-x-auto w-full">
            <table className="min-w-full text-xs">
              <thead className="text-gray-400">
                <tr className="text-left">
                  <th className="py-2 px-4">Time</th>
                  <th className="py-2 px-4">Market</th>
                  <th className="py-2 px-4">Side</th>
                  <th className="py-2 px-4">Type</th>
                  <th className="py-2 px-4 text-right">Size</th>
                  <th className="py-2 px-4 text-right">Price</th>
                  <th className="py-2 px-4 text-right">Filled</th>
                  <th className="py-2 px-4 text-right">Status</th>
                  <th className="py-2 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {orders.map((order) => (
                  <tr key={order.id} className="border-t border-gray-800 hover:bg-gray-800/50">
                    <td className="py-2 px-4 text-gray-400 font-mono text-xs">
                      {formatDate(order.createdAt)}
                    </td>
                    <td className="py-2 px-4 text-white font-medium">SOL-PERP</td>
                    <td className="py-2 px-4">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getSideColor(order.side)}`}>
                        {order.side === 'long' ? 'Long' : 'Short'}
                      </span>
                    </td>
                    <td className="py-2 px-4 text-white text-xs">
                      {getOrderTypeDisplay(order.orderType)}
                    </td>
                    <td className="py-2 px-4 text-right text-white font-mono">
                      {order.size.toFixed(3)}
                    </td>
                    <td className="py-2 px-4 text-right text-white font-mono">
                      {order.price ? `$${order.price.toFixed(2)}` : 'Market'}
                    </td>
                    <td className="py-2 px-4 text-right">
                      <div className="flex flex-col items-end">
                        <span className="text-white font-mono text-xs">
                          {order.filledSize ? order.filledSize.toFixed(3) : '0.000'}
                        </span>
                        <span className="text-gray-500 text-xs">
                          {order.remainingSize.toFixed(3)} remaining
                        </span>
                      </div>
                    </td>
                    <td className="py-2 px-4 text-right">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(order.status)}`}>
                        {order.status.replace('_', ' ')}
                      </span>
                    </td>
                    <td className="py-2 px-4 text-right">
                      {order.status === 'pending' && (
                        <button
                          onClick={() => handleCancelOrder(order.id)}
                          className="px-2 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700 transition-colors"
                          title="Cancel Order"
                        >
                          Cancel
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center text-gray-400">
            <div className="text-sm mb-2">No orders found</div>
            <div className="text-xs">Place an order to see it here</div>
          </div>
        )}
      </div>
    </div>
  )
}

export default Orders
