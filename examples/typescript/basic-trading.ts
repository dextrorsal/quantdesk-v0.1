/**
 * Basic Trading Example
 * 
 * This example shows how to:
 * - Connect to QuantDesk Protocol
 * - Get market data
 * - Place and manage orders
 * - Monitor positions
 */

import { QuantDeskClient } from '@quantdesk/sdk'
import { PhantomWalletAdapter } from '@solana/wallet-adapter-wallets'

async function basicTradingExample() {
  // Initialize client
  const client = new QuantDeskClient(
    'https://api.quantdesk.com',
    'https://api.mainnet-beta.solana.com',
    new PhantomWalletAdapter()
  )

  try {
    console.log('🚀 QuantDesk Basic Trading Example')
    console.log('=====================================')

    // 1. Get available markets
    console.log('\n📊 Fetching markets...')
    const markets = await client.getMarkets()
    console.log(`Found ${markets.length} markets:`)
    markets.slice(0, 5).forEach(market => {
      console.log(`  ${market.symbol}: $${market.price.toFixed(2)} (${market.changePercent24h.toFixed(2)}%)`)
    })

    // 2. Get specific market data
    console.log('\n💰 Getting BTC market data...')
    const btcMarket = await client.getMarket('BTC-PERP')
    console.log(`BTC Price: $${btcMarket.price.toFixed(2)}`)
    console.log(`24h Change: ${btcMarket.changePercent24h.toFixed(2)}%`)
    console.log(`24h Volume: $${btcMarket.volume24h.toLocaleString()}`)

    // 3. Get current positions
    console.log('\n📈 Current positions...')
    const positions = await client.getPositions()
    if (positions.length === 0) {
      console.log('  No open positions')
    } else {
      positions.forEach(position => {
        console.log(`  ${position.symbol} ${position.side}: ${position.size} @ $${position.entryPrice}`)
        console.log(`    PnL: $${position.pnl.toFixed(2)} (${position.pnlPercent.toFixed(2)}%)`)
      })
    }

    // 4. Place a limit order (example - uncomment to actually place)
    /*
    console.log('\n📝 Placing limit order...')
    const order = await client.placeOrder({
      symbol: 'BTC-PERP',
      side: 'buy',
      type: 'limit',
      amount: 0.001,
      price: btcMarket.price * 0.95, // 5% below current price
      timeInForce: 'GTC'
    })
    console.log(`Order placed: ${order.id}`)
    console.log(`Status: ${order.status}`)
    */

    // 5. Get portfolio summary
    console.log('\n💼 Portfolio summary...')
    const portfolio = await client.getPortfolioSummary()
    console.log(`Total Value: $${portfolio.totalValue.toFixed(2)}`)
    console.log(`Total PnL: $${portfolio.totalPnl.toFixed(2)} (${portfolio.totalPnlPercent.toFixed(2)}%)`)
    console.log(`Available Balance: $${portfolio.availableBalance.toFixed(2)}`)
    console.log(`Used Margin: $${portfolio.usedMargin.toFixed(2)}`)

    // 6. Get risk metrics
    console.log('\n⚠️ Risk metrics...')
    const riskMetrics = await client.getRiskMetrics()
    console.log(`Portfolio VaR: $${riskMetrics.portfolioVaR.toFixed(2)}`)
    console.log(`Max Drawdown: ${riskMetrics.maxDrawdown.toFixed(2)}%`)
    console.log(`Leverage Ratio: ${riskMetrics.leverageRatio.toFixed(2)}x`)
    console.log(`Risk Score: ${riskMetrics.overallRiskScore}/100`)

    console.log('\n✅ Example completed successfully!')

  } catch (error) {
    console.error('❌ Error:', error)
  }
}

// Run the example
if (require.main === module) {
  basicTradingExample()
}

export { basicTradingExample }
