/**
 * Advanced Orders Example
 * 
 * This example shows how to use advanced order types:
 * - Stop-loss orders
 * - Take-profit orders
 * - Trailing stop orders
 * - TWAP orders
 * - Iceberg orders
 */

import { QuantDeskClient } from '@quantdesk/sdk'
import { PhantomWalletAdapter } from '@solana/wallet-adapter-wallets'

async function advancedOrdersExample() {
  const client = new QuantDeskClient(
    'https://api.quantdesk.app',
    'https://api.mainnet-beta.solana.com',
    new PhantomWalletAdapter()
  )

  try {
    console.log('🎯 Advanced Orders Example')
    console.log('===========================')

    // Get current BTC price
    const btcPrice = await client.getMarketPrice('BTC-PERP')
    console.log(`Current BTC Price: $${btcPrice.toFixed(2)}`)

    // 1. Stop-Loss Order
    console.log('\n🛑 Stop-Loss Order Example')
    const stopLossOrder = await client.placeOrder({
      symbol: 'BTC-PERP',
      side: 'sell',
      type: 'stop',
      amount: 0.1,
      stopPrice: btcPrice * 0.95, // 5% below current price
      timeInForce: 'GTC'
    })
    console.log(`Stop-loss order placed: ${stopLossOrder.id}`)
    console.log(`Stop price: $${stopLossOrder.stopPrice}`)

    // 2. Take-Profit Order
    console.log('\n💰 Take-Profit Order Example')
    const takeProfitOrder = await client.placeOrder({
      symbol: 'BTC-PERP',
      side: 'sell',
      type: 'take-profit',
      amount: 0.1,
      price: btcPrice * 1.1, // 10% above current price
      timeInForce: 'GTC'
    })
    console.log(`Take-profit order placed: ${takeProfitOrder.id}`)
    console.log(`Target price: $${takeProfitOrder.price}`)

    // 3. Trailing Stop Order
    console.log('\n📈 Trailing Stop Order Example')
    const trailingStopOrder = await client.placeOrder({
      symbol: 'BTC-PERP',
      side: 'sell',
      type: 'trailing-stop',
      amount: 0.1,
      trailingDistance: btcPrice * 0.02, // 2% trailing distance
      timeInForce: 'GTC'
    })
    console.log(`Trailing stop order placed: ${trailingStopOrder.id}`)
    console.log(`Trailing distance: $${trailingStopOrder.trailingDistance}`)

    // 4. TWAP Order (Time-Weighted Average Price)
    console.log('\n⏰ TWAP Order Example')
    const twapOrder = await client.placeOrder({
      symbol: 'BTC-PERP',
      side: 'buy',
      type: 'twap',
      amount: 1.0,
      price: btcPrice * 1.02, // 2% above current price
      timeInForce: 'GTD' // Good Till Date
    })
    console.log(`TWAP order placed: ${twapOrder.id}`)
    console.log(`Total amount: ${twapOrder.amount} BTC`)
    console.log(`Max price: $${twapOrder.price}`)

    // 5. Iceberg Order
    console.log('\n🧊 Iceberg Order Example')
    const icebergOrder = await client.placeOrder({
      symbol: 'BTC-PERP',
      side: 'sell',
      type: 'iceberg',
      amount: 10.0, // Total amount
      price: btcPrice * 1.05, // 5% above current price
      timeInForce: 'GTC'
    })
    console.log(`Iceberg order placed: ${icebergOrder.id}`)
    console.log(`Total amount: ${icebergOrder.amount} BTC`)
    console.log(`Visible amount: ${icebergOrder.amount * 0.1} BTC (10% visible)`)

    // 6. Monitor orders
    console.log('\n📊 Monitoring orders...')
    const orders = await client.getOrders('BTC-PERP')
    console.log(`Active orders: ${orders.length}`)
    
    orders.forEach(order => {
      console.log(`  ${order.type.toUpperCase()}: ${order.side} ${order.amount} @ $${order.price || order.stopPrice}`)
      console.log(`    Status: ${order.status}`)
      console.log(`    Created: ${new Date(order.createdAt).toLocaleString()}`)
    })

    console.log('\n✅ Advanced orders example completed!')

  } catch (error) {
    console.error('❌ Error:', error)
  }
}

// Run the example
if (require.main === module) {
  advancedOrdersExample()
}

export { advancedOrdersExample }
