import React, { useState } from 'react'
import { useWallet } from '@solana/wallet-adapter-react'
import { TrendingUp, TrendingDown, Settings, Calculator } from 'lucide-react'
import { TradingService, PlaceOrderRequest } from '../services/tradingService'

interface OrderForm {
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stopLoss' | 'takeProfit' | 'trailingStop' | 'postOnly' | 'ioc' | 'fok'
  size: string
  price: string
  stopPrice: string
  trailingDistance: string
  leverage: number
  reduceOnly: boolean
  takeProfit: boolean
  stopLoss: boolean
}

const TradingInterface: React.FC = () => {
  const { connected } = useWallet()
  const [orderForm, setOrderForm] = useState<OrderForm>({
    side: 'buy',
    type: 'market',
    size: '',
    price: '',
    stopPrice: '',
    trailingDistance: '',
    leverage: 1,
    reduceOnly: false,
    takeProfit: false,
    stopLoss: false,
  })

  const [availableBalance] = useState(1000.00)
  // Future features (for later implementation)
  // const [estimatedLiquidation] = useState(0.00)
  // const [maxOrderSize] = useState(0.0)
  const [fees] = useState({ maker: 0.0000, taker: 0.0000 })
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [toast, setToast] = useState<{ type: 'success' | 'error'; message: string } | null>(null)

  const handleOrderSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!connected) {
      setToast({ type: 'error', message: 'Please connect your wallet first' })
      return
    }

    if (!orderForm.size || parseFloat(orderForm.size) <= 0) {
      setToast({ type: 'error', message: 'Please enter a valid size' })
      return
    }

    if (orderForm.type === 'limit' && (!orderForm.price || parseFloat(orderForm.price) <= 0)) {
      setToast({ type: 'error', message: 'Please enter a valid price for limit orders' })
      return
    }

    setIsSubmitting(true)
    setToast(null)

    try {
      const orderRequest: PlaceOrderRequest = {
        symbol: 'BTC-PERP', // TODO: Get from context/state
        side: orderForm.side,
        size: parseFloat(orderForm.size),
        orderType: orderForm.type === 'limit' ? 'limit' : 'market',
        price: orderForm.type === 'limit' ? parseFloat(orderForm.price) : undefined,
        leverage: orderForm.leverage,
      }

      const result = await TradingService.placeOrder(orderRequest)
      
      if (result.filled) {
        setToast({ 
          type: 'success', 
          message: `Order filled! Average price: $${result.averageFillPrice?.toFixed(2)}` 
        })
        // Reset form
        setOrderForm(prev => ({ ...prev, size: '', price: '' }))
      } else {
        setToast({ 
          type: 'success', 
          message: `Order placed! Order ID: ${result.orderId}` 
        })
      }
    } catch (error: any) {
      console.error('Order submission error:', error)
      setToast({ 
        type: 'error', 
        message: error.response?.data?.message || 'Failed to place order' 
      })
    } finally {
      setIsSubmitting(false)
    }
  }

  const setOrderSide = (side: 'buy' | 'sell') => {
    setOrderForm(prev => ({ ...prev, side }))
  }

  const setOrderType = (type: 'market' | 'limit' | 'stopLoss' | 'takeProfit' | 'trailingStop' | 'postOnly' | 'ioc' | 'fok') => {
    setOrderForm(prev => ({ ...prev, type }))
  }

  const updateLeverage = (leverage: number) => {
    setOrderForm(prev => ({ ...prev, leverage }))
  }

  return (
    <div className="h-full bg-gray-900 flex flex-col">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <h3 className="text-sm font-semibold text-white">Trade</h3>
        <div className="flex space-x-2">
          <button className="p-1 text-gray-400 hover:text-white transition-colors">
            <Settings className="h-4 w-4" />
          </button>
          <button className="p-1 text-gray-400 hover:text-white transition-colors">
            <Calculator className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 flex flex-col p-4">
        {/* Order Type Tabs */}
        <div className="flex mb-4 bg-gray-800 rounded-lg p-1">
          {['market', 'limit', 'stopLoss', 'takeProfit', 'trailingStop'].map((type) => (
            <button
              key={type}
              onClick={() => setOrderType(type as any)}
              className={`flex-1 py-2 px-2 rounded-md text-xs font-medium transition-colors ${
                orderForm.type === type
                  ? 'bg-white text-black'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700'
              }`}
            >
              {type === 'stopLoss' ? 'SL' : 
               type === 'takeProfit' ? 'TP' : 
               type === 'trailingStop' ? 'TS' :
               type.charAt(0).toUpperCase() + type.slice(1)}
            </button>
          ))}
        </div>

        {/* Buy/Sell Buttons */}
        <div className="grid grid-cols-2 gap-2 mb-4">
          <button
            onClick={() => setOrderSide('buy')}
            className={`py-3 px-4 rounded-lg font-semibold transition-colors ${
              orderForm.side === 'buy'
                ? 'bg-green-600 text-white'
                : 'bg-gray-800 text-green-400 hover:bg-green-600 hover:text-white'
            }`}
          >
            <TrendingUp className="h-4 w-4 inline mr-2" />
            Buy / Long
          </button>
          <button
            onClick={() => setOrderSide('sell')}
            className={`py-3 px-4 rounded-lg font-semibold transition-colors ${
              orderForm.side === 'sell'
                ? 'bg-red-600 text-white'
                : 'bg-gray-800 text-red-400 hover:bg-red-600 hover:text-white'
            }`}
          >
            <TrendingDown className="h-4 w-4 inline mr-2" />
            Sell / Short
          </button>
        </div>

        {/* Order Form */}
        <form onSubmit={handleOrderSubmit} className="flex-1 flex flex-col space-y-3">
          {/* Available Balance */}
          <div className="flex justify-between text-xs">
            <span className="text-gray-400">Available</span>
            <span className="text-white">{availableBalance.toFixed(2)} USDT</span>
          </div>

          {/* Leverage Slider */}
          <div className="space-y-2">
            <div className="flex justify-between text-xs">
              <span className="text-gray-400">Leverage</span>
              <span className="text-white">{orderForm.leverage}x</span>
            </div>
            <div className="flex space-x-1">
              {[1, 2, 5, 10, 25, 50, 100].map((leverage) => (
                <button
                  key={leverage}
                  type="button"
                  onClick={() => updateLeverage(leverage)}
                  className={`px-2 py-1 rounded text-xs font-medium transition-colors ${
                    orderForm.leverage === leverage
                      ? 'bg-white text-black'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {leverage}x
                </button>
              ))}
            </div>
          </div>

          {/* Size Input */}
          <div className="space-y-2">
            <label className="text-xs text-gray-400">Size</label>
            <div className="relative">
              <input
                type="number"
                value={orderForm.size}
                onChange={(e) => setOrderForm(prev => ({ ...prev, size: e.target.value }))}
                placeholder="0.00"
                className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-blue-500"
                step="0.01"
                min="0"
              />
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 text-xs">
                BTC
              </div>
            </div>
          </div>

          {/* Price Input (for limit orders) */}
          {orderForm.type === 'limit' && (
            <div className="space-y-2">
              <label className="text-xs text-gray-400">Price</label>
              <div className="relative">
                <input
                  type="number"
                  value={orderForm.price}
                  onChange={(e) => setOrderForm(prev => ({ ...prev, price: e.target.value }))}
                  placeholder="0.00"
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-blue-500"
                  step="0.01"
                  min="0"
                />
                <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 text-xs">
                  USDT
                </div>
              </div>
            </div>
          )}

          {/* Stop Price Input (for stop-loss and take-profit orders) */}
          {(orderForm.type === 'stopLoss' || orderForm.type === 'takeProfit') && (
            <div className="space-y-2">
              <label className="text-xs text-gray-400">
                {orderForm.type === 'stopLoss' ? 'Stop Price' : 'Take Profit Price'}
              </label>
              <div className="relative">
                <input
                  type="number"
                  value={orderForm.stopPrice}
                  onChange={(e) => setOrderForm(prev => ({ ...prev, stopPrice: e.target.value }))}
                  placeholder="0.00"
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-blue-500"
                  step="0.01"
                  min="0"
                />
                <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 text-xs">
                  USDT
                </div>
              </div>
            </div>
          )}

          {/* Trailing Distance Input (for trailing stop orders) */}
          {orderForm.type === 'trailingStop' && (
            <div className="space-y-2">
              <label className="text-xs text-gray-400">Trailing Distance</label>
              <div className="relative">
                <input
                  type="number"
                  value={orderForm.trailingDistance}
                  onChange={(e) => setOrderForm(prev => ({ ...prev, trailingDistance: e.target.value }))}
                  placeholder="0.00"
                  className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm placeholder-gray-500 focus:outline-none focus:border-blue-500"
                  step="0.01"
                  min="0"
                />
                <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400 text-xs">
                  USDT
                </div>
              </div>
            </div>
          )}

          {/* Options */}
          <div className="space-y-2">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={orderForm.reduceOnly}
                onChange={(e) => setOrderForm(prev => ({ ...prev, reduceOnly: e.target.checked }))}
                className="rounded border-gray-600 bg-gray-800 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-xs text-gray-300">Reduce Only</span>
            </label>
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={orderForm.takeProfit}
                onChange={(e) => setOrderForm(prev => ({ ...prev, takeProfit: e.target.checked }))}
                className="rounded border-gray-600 bg-gray-800 text-blue-600 focus:ring-blue-500"
              />
              <span className="text-xs text-gray-300">TP/SL</span>
            </label>
          </div>

          {/* Order Summary */}
          <div className="bg-gray-800 rounded-lg p-3 space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-400">Order Type</span>
              <span className="text-white">
                {orderForm.type === 'stopLoss' ? 'Stop Loss' : 
                 orderForm.type === 'takeProfit' ? 'Take Profit' : 
                 orderForm.type === 'trailingStop' ? 'Trailing Stop' :
                 orderForm.type.charAt(0).toUpperCase() + orderForm.type.slice(1)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Margin</span>
              <span className="text-white">0.00 / 0.00 USDT</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Est. Liq. Price</span>
              <span className="text-white">-- / -- USDT</span>
            </div>
            {(orderForm.type === 'stopLoss' || orderForm.type === 'takeProfit') && orderForm.stopPrice && (
              <div className="flex justify-between">
                <span className="text-gray-400">Stop Price</span>
                <span className="text-white">{orderForm.stopPrice} USDT</span>
              </div>
            )}
            {orderForm.type === 'trailingStop' && orderForm.trailingDistance && (
              <div className="flex justify-between">
                <span className="text-gray-400">Trailing Distance</span>
                <span className="text-white">{orderForm.trailingDistance} USDT</span>
              </div>
            )}
            <div className="flex justify-between">
              <span className="text-gray-400">Fee</span>
              <span className="text-white">{fees.maker}% / {fees.taker}%</span>
            </div>
          </div>

          {/* Submit Button */}
          <button
            type="submit"
            className={`w-full py-3 px-4 rounded-lg font-semibold transition-colors ${
              connected && !isSubmitting
                ? orderForm.side === 'buy'
                  ? 'bg-green-600 hover:bg-green-700 text-white'
                  : 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-gray-600 text-gray-400 cursor-not-allowed'
            }`}
            disabled={!connected || isSubmitting}
          >
            {isSubmitting 
              ? 'Placing Order...' 
              : connected 
                ? `${orderForm.side === 'buy' ? 'Buy' : 'Sell'} ${orderForm.size || '0.0'} BTC` 
                : 'Connect Wallet'
            }
          </button>

          {/* Toast Notification */}
          {toast && (
            <div className={`fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${
              toast.type === 'success' ? 'bg-green-600 text-white' : 'bg-red-600 text-white'
            }`}>
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{toast.message}</span>
                <button
                  onClick={() => setToast(null)}
                  className="ml-4 text-white hover:text-gray-200"
                >
                  ×
                </button>
              </div>
            </div>
          )}
        </form>
      </div>
    </div>
  )
}

export default TradingInterface
