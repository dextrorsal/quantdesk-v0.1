import React from 'react'
import { usePrice } from '../contexts/PriceContext'

const Positions: React.FC = () => {
  const hasPositions = false // This would come from your trading state
  const { getPrice } = usePrice()
  
  // Get current BTC price for display
  const btcPrice = getPrice('BTC/USDT')?.price || 0

  return (
    <div className="h-full bg-gray-900 flex flex-col">
      <div className="flex-1 flex items-center justify-center">
        {hasPositions ? (
          <div className="overflow-x-auto w-full">
            <table className="min-w-full text-xs">
              <thead className="text-gray-400">
                <tr className="text-left">
                  <th className="py-2 px-4">Market</th>
                  <th className="py-2 px-4">Side</th>
                  <th className="py-2 px-4 text-right">Size</th>
                  <th className="py-2 px-4 text-right">Entry</th>
                  <th className="py-2 px-4 text-right">Mark</th>
                  <th className="py-2 px-4 text-right">Unrealized PnL</th>
                  <th className="py-2 px-4 text-right">Margin</th>
                  <th className="py-2 px-4 text-right">Leverage</th>
                  <th className="py-2 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-t border-gray-800">
                  <td className="py-2 px-4 text-white">BTC/USDT</td>
                  <td className="py-2 px-4"><span className="text-green-400">Long</span></td>
                  <td className="py-2 px-4 text-right text-white">0.250 BTC</td>
                  <td className="py-2 px-4 text-right text-white">{btcPrice.toFixed(2)}</td>
                  <td className="py-2 px-4 text-right text-white">{btcPrice.toFixed(2)}</td>
                  <td className="py-2 px-4 text-right text-green-400">+120.34</td>
                  <td className="py-2 px-4 text-right text-white">500.00</td>
                  <td className="py-2 px-4 text-right text-white">10x</td>
                  <td className="py-2 px-4 text-right">
                    <button className="px-3 py-1 mr-2 bg-gray-800 text-gray-300 rounded text-xs hover:bg-gray-700">TP/SL</button>
                    <button className="px-3 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700">Close</button>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center text-gray-400">
            <div className="text-sm mb-2">No open positions yet</div>
            <div className="text-xs">Log in or Sign Up to start trading</div>
          </div>
        )}
      </div>
    </div>
  )
}

export default Positions
