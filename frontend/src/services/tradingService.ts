import { apiClient } from './apiClient'

export interface PlaceOrderRequest {
  symbol: string
  side: 'buy' | 'sell'
  size: number
  orderType: 'market' | 'limit'
  price?: number
  leverage?: number
}

export interface PlaceOrderResponse {
  orderId: string
  filled: boolean
  fills: Array<{ price: number; size: number }>
  averageFillPrice?: number
}

export interface Order {
  id: string
  userId: string
  marketId: string
  orderAccount: string
  orderType: 'market' | 'limit'
  side: 'long' | 'short'
  size: string
  price?: string
  leverage: number
  status: 'pending' | 'filled' | 'partially_filled' | 'cancelled'
  filledSize?: string
  averageFillPrice?: string
  createdAt: string
  updatedAt?: string
  filledAt?: string
}

export class TradingService {
  /**
   * Place a new order
   */
  static async placeOrder(request: PlaceOrderRequest): Promise<PlaceOrderResponse> {
    const response = await apiClient.post('/orders', request)
    return response.data
  }

  /**
   * Get user's orders
   */
  static async getOrders(): Promise<Order[]> {
    const response = await apiClient.get('/orders')
    return response.data
  }

  /**
   * Get a specific order by ID
   */
  static async getOrder(orderId: string): Promise<Order> {
    const response = await apiClient.get(`/orders/${orderId}`)
    return response.data
  }

  /**
   * Cancel an order
   */
  static async cancelOrder(orderId: string): Promise<void> {
    await apiClient.delete(`/orders/${orderId}`)
  }
}
