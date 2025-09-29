import React, { useState, useEffect } from 'react'
import { TradingService, Order } from '../services/tradingService'

const Orders: React.FC = () => {
  const [orders, setOrders] = useState<Order[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    loadOrders()
  }, [])

  const loadOrders = async () => {
    try {
      setLoading(true)
      setError(null)
      const ordersData = await TradingService.getOrders()
      setOrders(ordersData)
    } catch (err: any) {
      console.error('Error loading orders:', err)
      setError(err.response?.data?.message || 'Failed to load orders')
    } finally {
      setLoading(false)
    }
  }

  const handleCancelOrder = async (orderId: string) => {
    try {
      await TradingService.cancelOrder(orderId)
      setOrders(prev => prev.filter(order => order.id !== orderId))
    } catch (err: any) {
      console.error('Error cancelling order:', err)
      alert(err.response?.data?.message || 'Failed to cancel order')
    }
  }

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString()
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'filled': return 'text-green-400'
      case 'partially_filled': return 'text-yellow-400'
      case 'pending': return 'text-gray-400'
      case 'cancelled': return 'text-gray-400'
      default: return 'text-gray-400'
    }
  }

  const getSideColor = (side: string) => {
    return side === 'long' ? 'text-green-400' : 'text-red-400'
  }

  if (loading) {
    return (
      <div className="h-full panel-blue flex items-center justify-center">
        <div className="text-gray-400">Loading orders...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="h-full panel-blue flex items-center justify-center">
        <div className="text-center text-gray-400">
          <div className="text-sm mb-2 text-red-400">Error loading orders</div>
          <div className="text-xs">{error}</div>
          <button 
            onClick={loadOrders}
            className="mt-2 px-3 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700"
          >
            Retry
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="h-full panel-blue flex flex-col">
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
                  <th className="py-2 px-4 text-right">Status</th>
                  <th className="py-2 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {orders.map((order) => (
                  <tr key={order.id} className="border-t border-gray-800">
                    <td className="py-2 px-4 text-gray-400">{formatDate(order.createdAt)}</td>
                    <td className="py-2 px-4 text-white">BTC-PERP</td>
                    <td className="py-2 px-4">
                      <span className={getSideColor(order.side)}>
                        {order.side === 'long' ? 'Buy' : 'Sell'}
                      </span>
                    </td>
                    <td className="py-2 px-4 text-white capitalize">{order.orderType}</td>
                    <td className="py-2 px-4 text-right text-white">{order.size}</td>
                    <td className="py-2 px-4 text-right text-white">
                      {order.price ? `$${parseFloat(order.price).toFixed(2)}` : 'Market'}
                    </td>
                    <td className="py-2 px-4 text-right">
                      <span className={getStatusColor(order.status)}>
                        {order.status.replace('_', ' ')}
                      </span>
                    </td>
                    <td className="py-2 px-4 text-right">
                      {order.status === 'pending' && (
                        <button
                          onClick={() => handleCancelOrder(order.id)}
                          className="text-red-400 hover:text-red-300 text-xs"
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
