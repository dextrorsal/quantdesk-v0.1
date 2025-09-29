import React from 'react'
import TradingTab from './TradingTab'
import { useTabContext } from '../contexts/TabContext'
import { useTheme } from '../contexts/ThemeContext'
import { usePrice } from '../contexts/PriceContext'
import TradingViewTickerTape from '../components/TradingViewTickerTape'
import TradingViewHeatmap from '../components/TradingViewHeatmap'
import TradingViewScreener from '../components/TradingViewScreener'
import TradingViewCryptoMarkets from '../components/TradingViewCryptoMarkets'

// Comprehensive Dashboard Overview Component
const DashboardOverview: React.FC = () => {
  // Mock data for dashboard
  const [portfolioStats] = React.useState({
    totalValue: 125430.50,
    dayChange: 7.2,
    dayChangeAmount: 2450,
    activeStrategies: 3,
    winRate: 87,
    totalTrades: 12
  })

  const [activePositions] = React.useState([
    { symbol: 'BTC/USDT', side: 'long', size: 0.5, entryPrice: 43250, currentPrice: 44120, pnl: 435, pnlPercent: 2.01 },
    { symbol: 'ETH/USDT', side: 'short', size: 2.0, entryPrice: 3200, currentPrice: 3150, pnl: 100, pnlPercent: 1.56 },
    { symbol: 'SOL/USDT', side: 'long', size: 10, entryPrice: 220, currentPrice: 225, pnl: 50, pnlPercent: 2.27 }
  ])

  const [botPerformance] = React.useState([
    { name: 'Lorentzian ML', status: 'active', pnl: 1250, trades: 8, winRate: 87.5 },
    { name: 'RSI Momentum', status: 'active', pnl: 890, trades: 12, winRate: 75.0 },
    { name: 'MACD Cross', status: 'paused', pnl: -120, trades: 5, winRate: 60.0 }
  ])

  const [marketHeatmap] = React.useState([
    { symbol: 'BTC', change: 2.1, volume: 2.5 },
    { symbol: 'ETH', change: -1.2, volume: 1.8 },
    { symbol: 'SOL', change: 3.4, volume: 0.9 },
    { symbol: 'ADA', change: -0.8, volume: 0.7 },
    { symbol: 'DOT', change: 1.5, volume: 0.4 },
    { symbol: 'MATIC', change: -2.3, volume: 0.6 },
    { symbol: 'AVAX', change: 0.9, volume: 0.3 },
    { symbol: 'LINK', change: -1.7, volume: 0.5 }
  ])

  const [newsFeed] = React.useState([
    { title: 'Bitcoin ETF Approval Expected This Week', source: 'CoinDesk', time: '2m ago', sentiment: 'bullish' },
    { title: 'Ethereum Layer 2 Solutions See Record Growth', source: 'The Block', time: '15m ago', sentiment: 'bullish' },
    { title: 'Fed Signals Potential Rate Cut in Q2', source: 'Reuters', time: '1h ago', sentiment: 'neutral' },
    { title: 'Solana DeFi TVL Reaches New All-Time High', source: 'DeFi Pulse', time: '2h ago', sentiment: 'bullish' }
  ])

  const [twitterFeed] = React.useState([
    { user: 'CryptoWhale', content: 'BTC breaking through $90K resistance with massive institutional buying. This could be the start of the next leg up! 🚀', time: '5m ago', likes: 1240 },
    { user: 'DeFiAlpha', content: 'Just discovered this new yield farming opportunity on @Solana. APY looks insane but DYOR! 💎', time: '12m ago', likes: 890 },
    { user: 'TradingPro', content: 'Market structure looking bullish. Expecting continuation move in the next 24-48 hours. 📈', time: '25m ago', likes: 567 }
  ])

  const formatCurrency = (value: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)
  const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`

  const getHeatmapColor = (change: number) => {
    if (change > 2) return 'bg-green-600'
    if (change > 0) return 'bg-green-500'
    if (change > -2) return 'bg-red-500'
    return 'bg-red-600'
  }

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish': return 'price-positive'
      case 'bearish': return 'price-negative'
      default: return 'price-neutral'
    }
  }

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'bullish': return '📈'
      case 'bearish': return '📉'
      default: return '➡️'
    }
  }

  return (
    <div className="p-6 h-full overflow-auto">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-white mb-2">📊 Trading Dashboard</h1>
        <p className="text-gray-400">Your comprehensive trading overview - Upgrade to PRO for full customization</p>
      </div>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div className="panel-blue p-4 rounded-lg border border-primary-500">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm text-gray-400">Portfolio Value</h3>
            <span className="text-xs text-gray-500">24h</span>
          </div>
          <p className="text-2xl font-bold text-white">{formatCurrency(portfolioStats.totalValue)}</p>
          <p className={`text-sm ${portfolioStats.dayChange >= 0 ? 'price-positive' : 'price-negative'}`}>
            {formatPercent(portfolioStats.dayChange)} ({formatCurrency(portfolioStats.dayChangeAmount)})
          </p>
        </div>
        
        <div className="panel-blue p-4 rounded-lg border border-primary-500">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm text-gray-400">Active Strategies</h3>
            <span className="text-xs text-gray-500">Bots</span>
          </div>
          <p className="text-2xl font-bold text-white">{portfolioStats.activeStrategies}</p>
          <p className="text-sm text-gray-400">{portfolioStats.totalTrades} trades today</p>
        </div>
        
        <div className="panel-blue p-4 rounded-lg border border-primary-500">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm text-gray-400">Win Rate</h3>
            <span className="text-xs text-gray-500">30d</span>
          </div>
          <p className="text-2xl font-bold price-positive">{portfolioStats.winRate}%</p>
          <p className="text-sm text-gray-400">Last 30 days</p>
        </div>
        
        <div className="panel-blue p-4 rounded-lg border border-primary-500">
          <div className="flex items-center justify-between mb-2">
            <h3 className="text-sm text-gray-400">Today's P&L</h3>
            <span className="text-xs text-gray-500">Live</span>
          </div>
          <p className="text-2xl font-bold price-positive">{formatCurrency(portfolioStats.dayChangeAmount)}</p>
          <p className="text-sm text-gray-400">{portfolioStats.totalTrades} trades</p>
        </div>
      </div>

      {/* Two Column Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Active Positions */}
        <div className="panel-blue rounded-lg border border-primary-500">
          <div className="p-4 border-b border-primary-500">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">Active Positions</h3>
              <button 
                onClick={() => window.location.hash = '#trading'}
                className="text-xs text-primary-500 hover:text-primary-400"
              >
                View All →
              </button>
            </div>
          </div>
          <div className="p-4">
            {activePositions.map((position, index) => (
              <div key={index} className="flex items-center justify-between py-2 border-b border-primary-500 last:border-b-0">
                <div className="flex items-center gap-3">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    position.side === 'long' ? 'bg-green-600 text-green-100' : 'bg-red-600 text-red-100'
                  }`}>
                    {position.side.toUpperCase()}
                  </span>
                  <div>
                    <p className="text-white font-medium">{position.symbol}</p>
                    <p className="text-xs text-gray-400">{position.size} @ {formatCurrency(position.entryPrice)}</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className={`font-medium ${position.pnl >= 0 ? 'price-positive' : 'price-negative'}`}>
                    {formatCurrency(position.pnl)}
                  </p>
                  <p className={`text-xs ${position.pnl >= 0 ? 'price-positive' : 'price-negative'}`}>
                    {formatPercent(position.pnlPercent)}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Bot Performance */}
        <div className="panel-blue rounded-lg border border-primary-500">
          <div className="p-4 border-b border-primary-500">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">Bot Performance</h3>
              <button 
                onClick={() => window.location.hash = '#strategies'}
                className="text-xs text-primary-500 hover:text-primary-400"
              >
                Manage →
              </button>
            </div>
          </div>
          <div className="p-4">
            {botPerformance.map((bot, index) => (
              <div key={index} className="flex items-center justify-between py-2 border-b border-primary-500 last:border-b-0">
                <div className="flex items-center gap-3">
                  <span className={`w-2 h-2 rounded-full ${
                    bot.status === 'active' ? 'bg-green-400' : 'bg-yellow-400'
                  }`}></span>
                  <div>
                    <p className="text-white font-medium">{bot.name}</p>
                    <p className="text-xs text-gray-400">{bot.trades} trades • {bot.winRate}% win rate</p>
                  </div>
                </div>
                <div className="text-right">
                  <p className={`font-medium ${bot.pnl >= 0 ? 'price-positive' : 'price-negative'}`}>
                    {formatCurrency(bot.pnl)}
                  </p>
                  <p className="text-xs text-gray-400">{bot.status}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Market Heatmap */}
      <div className="panel-blue rounded-lg border border-primary-500 mb-6">
        <div className="p-4 border-b border-primary-500">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold text-white">Market Heatmap</h3>
            <button 
              onClick={() => window.location.hash = '#market'}
              className="text-xs text-primary-500 hover:text-primary-400"
            >
              Full View →
            </button>
          </div>
        </div>
        <div className="p-4">
          <div className="grid grid-cols-4 md:grid-cols-8 gap-2">
            {marketHeatmap.map((asset, index) => (
              <div key={index} className="text-center">
                <div className={`p-2 rounded text-white font-medium text-sm ${getHeatmapColor(asset.change)}`}>
                  {asset.symbol}
                </div>
                <p className={`text-xs mt-1 ${asset.change >= 0 ? 'price-positive' : 'price-negative'}`}>
                  {formatPercent(asset.change)}
                </p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* News & Social Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* News Feed */}
        <div className="panel-blue rounded-lg border border-primary-500">
          <div className="p-4 border-b border-primary-500">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">📰 Market News</h3>
              <button 
                onClick={() => window.location.hash = '#sentiment'}
                className="text-xs text-primary-500 hover:text-primary-400"
              >
                More News →
              </button>
            </div>
          </div>
          <div className="p-4">
            {newsFeed.map((news, index) => (
              <div key={index} className="py-3 border-b border-primary-500 last:border-b-0">
                <div className="flex items-start gap-3">
                  <span className="text-lg">{getSentimentIcon(news.sentiment)}</span>
                  <div className="flex-1">
                    <p className="text-white text-sm font-medium mb-1">{news.title}</p>
                    <div className="flex items-center gap-2 text-xs text-gray-400">
                      <span>{news.source}</span>
                      <span>•</span>
                      <span>{news.time}</span>
                      <span className={`ml-auto ${getSentimentColor(news.sentiment)}`}>
                        {news.sentiment}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Twitter Feed */}
        <div className="panel-blue rounded-lg border border-primary-500">
          <div className="p-4 border-b border-primary-500">
            <div className="flex items-center justify-between">
              <h3 className="text-lg font-semibold text-white">🐦 Crypto Twitter</h3>
              <button 
                onClick={() => window.location.hash = '#social'}
                className="text-xs text-primary-500 hover:text-primary-400"
              >
                Full Feed →
              </button>
            </div>
          </div>
          <div className="p-4">
            {twitterFeed.map((tweet, index) => (
              <div key={index} className="py-3 border-b border-primary-500 last:border-b-0">
                <div className="flex items-start gap-3">
                  <div className="w-8 h-8 bg-primary-600 rounded-full flex items-center justify-center text-white text-sm font-bold">
                    {tweet.user.charAt(0)}
                  </div>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-white font-medium text-sm">@{tweet.user}</span>
                      <span className="text-gray-400 text-xs">{tweet.time}</span>
                    </div>
                    <p className="text-gray-300 text-sm mb-2">{tweet.content}</p>
                    <div className="flex items-center gap-4 text-xs text-gray-400">
                      <span>❤️ {tweet.likes}</span>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Upgrade Prompt */}
      <div className="mt-6 panel-blue rounded-lg border border-primary-500 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-white mb-2">🚀 Unlock Full Dashboard Customization</h3>
            <p className="text-gray-400 text-sm">Upgrade to PRO to customize widgets, add more data feeds, and create your perfect trading dashboard</p>
          </div>
          <button className="btn btn-primary">
            Upgrade to PRO
          </button>
        </div>
      </div>
    </div>
  )
}

const LiteRouter: React.FC = () => {
  const { activeTab } = useTabContext()
  const { setTheme } = useTheme()
  
  // Set theme to lite when component mounts
  React.useEffect(() => {
    setTheme('lite')
  }, [setTheme])
  
  // Debug logging (commented out for production)
  // console.log(`🎯 LiteRouter render - activeTab: ${activeTab}`)

  // Comprehensive Market Data Component with all QuantDesk Lite features
  const MarketTab = () => {
    const [activeSubTab, setActiveSubTab] = React.useState<'tickers' | 'orderbook' | 'heatmap' | 'volume' | 'markets'>('tickers')
    const [selectedSymbol, setSelectedSymbol] = React.useState('BTC')
    const [timeframe, setTimeframe] = React.useState('24h')
    const [loading, setLoading] = React.useState(true)
    
    // Get real-time prices from PriceContext
    const { getPrice } = usePrice()
    
    // Real market data from Pyth Network
    const [marketData, setMarketData] = React.useState<any[]>([])
    
    // Use real Pyth prices directly instead of API calls
    React.useEffect(() => {
      const fetchMarketData = async () => {
        try {
          setLoading(true)
          
          // Use real Pyth prices directly
          const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 'MATIC/USDT', 'AVAX/USDT', 'LINK/USDT']
          const names = ['Bitcoin', 'Ethereum', 'Solana', 'Cardano', 'Polkadot', 'Polygon', 'Avalanche', 'Chainlink']
          const categories = ['Large Cap', 'Large Cap', 'Mid Cap', 'Mid Cap', 'Mid Cap', 'Mid Cap', 'Mid Cap', 'Mid Cap']
          
          const marketData = symbols.map((symbol, index) => {
            const priceData = getPrice(symbol)
            const price = priceData?.price || 0
            const change = priceData?.change || 0
            const changePercent = priceData?.changePercent || 0
            
            return {
              symbol: symbol.split('/')[0], // Extract BTC from BTC/USDT
              name: names[index],
              price: price,
              change: change,
              changePercent: changePercent,
              volume: Math.floor(Math.random() * 10000000) + 1000000, // Random volume
              marketCap: price * 1000000, // Approximate market cap
              high24h: price * 1.02, // Approximate 24h high
              low24h: price * 0.98, // Approximate 24h low
              category: categories[index]
            }
          })
          
          setMarketData(marketData)
        } catch (error) {
          console.error('Error fetching market data:', error)
          setMarketData([])
        } finally {
          setLoading(false)
        }
      }
      
      fetchMarketData()
    }, [getPrice])

    // Mock order book data
    const [orderBook] = React.useState({
      bids: [
        { price: 115750, amount: 0.5, total: 0.5 },
        { price: 115740, amount: 1.2, total: 1.7 },
        { price: 115730, amount: 0.8, total: 2.5 },
        { price: 115720, amount: 2.1, total: 4.6 },
        { price: 115710, amount: 1.5, total: 6.1 },
      ],
      asks: [
        { price: 115760, amount: 0.7, total: 0.7 },
        { price: 115770, amount: 1.1, total: 1.8 },
        { price: 115780, amount: 0.9, total: 2.7 },
        { price: 115790, amount: 1.8, total: 4.5 },
        { price: 115800, amount: 1.3, total: 5.8 },
      ]
    })

    const formatCurrency = (value: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)
    const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
    const formatNumber = (value: number) => {
      if (value >= 1e12) return (value / 1e12).toFixed(1) + 'T'
      if (value >= 1e9) return (value / 1e9).toFixed(1) + 'B'
      if (value >= 1e6) return (value / 1e6).toFixed(1) + 'M'
      if (value >= 1e3) return (value / 1e3).toFixed(1) + 'K'
      return value.toFixed(2)
    }
    const getChangeColor = (value: number) => value >= 0 ? 'text-primary-500' : 'text-red-400'
    const getHeatmapColor = (changePercent: number) => {
      const absPercent = Math.abs(changePercent)
      if (absPercent >= 5) return changePercent > 0 ? 'bg-primary-600' : 'bg-red-600'
      if (absPercent >= 3) return changePercent > 0 ? 'bg-primary-500' : 'bg-red-500'
      if (absPercent >= 1) return changePercent > 0 ? 'bg-primary-400' : 'bg-red-400'
      return 'bg-gray-500'
    }

    return (
      <div className="p-6 h-full overflow-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">📈 Market Data</h2>
          <p className="text-gray-400">Real-time market data, order books, heatmaps, and volume analysis</p>
        </div>

        {/* Sub-navigation tabs */}
        <div className="flex gap-2 mb-6 border-b border-primary-500">
          {[
            { id: 'tickers', label: '📊 Tickers', icon: '📊' },
            { id: 'orderbook', label: '📖 Order Book', icon: '📖' },
            { id: 'heatmap', label: '🔥 Heatmap', icon: '🔥' },
            { id: 'volume', label: '📊 Volume', icon: '📊' },
            { id: 'markets', label: '🏪 Markets', icon: '🏪' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id as any)}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                activeSubTab === tab.id
                  ? 'bg-primary-500 text-white border-b-2 border-primary-500'
                  : 'text-gray-400 hover:text-white hover:bg-primary-500/10'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tickers Tab */}
        {activeSubTab === 'tickers' && (
          <div className="space-y-4">
            {/* TradingView Ticker Tape */}
            <div>
              <TradingViewTickerTape />
            </div>
            
            {/* Two Column Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left Column - Crypto Screener */}
              <div>
                <h3 className="text-lg font-semibold text-white mb-3">🔍 Crypto Screener</h3>
                <div className="panel-blue rounded-lg border border-primary-500 p-4">
                  <TradingViewScreener />
                </div>
              </div>

              {/* Right Column - Market Data */}
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold text-white">📊 Market Data</h3>
                  <div className="flex items-center gap-4">
                    <select
                      value={timeframe}
                      onChange={(e) => setTimeframe(e.target.value)}
                      className="panel-blue border border-gray-600 rounded px-3 py-2 text-white text-sm"
                    >
                      <option value="1h">1H</option>
                      <option value="24h">24H</option>
                      <option value="7d">7D</option>
                      <option value="30d">30D</option>
                    </select>
                    <div className="text-sm text-gray-400">
                      Last update: {new Date().toLocaleTimeString()}
                    </div>
                  </div>
                </div>
                
                <div className="space-y-4 max-h-[550px] overflow-y-auto">
                  {loading ? (
                    <div className="text-center py-8">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-500 mx-auto"></div>
                      <p className="mt-2 text-gray-400">Loading market data...</p>
                    </div>
                  ) : (
                    marketData.map((market) => (
                    <div key={market.symbol} className="panel-blue p-4 rounded-lg border border-primary-500 hover:border-primary-400 transition-colors cursor-pointer"
                         onClick={() => setSelectedSymbol(market.symbol)}>
                      <div className="flex justify-between items-center">
                        <div className="flex items-center gap-4">
                          <div className="w-12 h-12 bg-gray-700 rounded-full flex items-center justify-center text-white font-bold text-lg">
                            {market.symbol.charAt(0)}
                          </div>
                          <div>
                            <div className="text-lg font-semibold text-white">{market.symbol}</div>
                            <div className="text-sm text-gray-400">{market.name}</div>
                            <div className="text-xs text-gray-500">Vol: {formatNumber(market.volume)}</div>
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-lg font-semibold text-white">{formatCurrency(market.price)}</div>
                          <div className={`text-sm ${getChangeColor(market.changePercent)}`}>
                            {formatCurrency(market.change)} ({formatPercent(market.changePercent)})
                          </div>
                        </div>
                        <div className="text-right min-w-[120px]">
                          <div className="text-sm text-gray-400">High: {formatCurrency(market.high24h)}</div>
                          <div className="text-sm text-gray-400">Low: {formatCurrency(market.low24h)}</div>
                        </div>
                        <div className="text-right min-w-[100px]">
                          <div className="text-sm text-gray-400">Cap: {formatNumber(market.marketCap)}</div>
                          <div className="text-xs text-gray-500">{market.category}</div>
                        </div>
                      </div>
                    </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Order Book Tab */}
        {activeSubTab === 'orderbook' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div>
              <h3 className="text-lg font-semibold text-white mb-4">📖 Order Book - {selectedSymbol}</h3>
              <div className="panel-blue rounded-lg border border-primary-500 p-4">
                {/* Asks */}
                <div className="mb-4">
                  <div className="grid grid-cols-3 gap-2 mb-2 text-xs text-gray-400 font-semibold">
                    <div>Price</div>
                    <div className="text-right">Amount</div>
                    <div className="text-right">Total</div>
                  </div>
                  {orderBook.asks.slice().reverse().map((ask, idx) => (
                    <div key={idx} className="grid grid-cols-3 gap-2 py-1 text-sm text-red-400">
                      <div>{formatCurrency(ask.price)}</div>
                      <div className="text-right">{ask.amount.toFixed(4)}</div>
                      <div className="text-right">{ask.total.toFixed(4)}</div>
                    </div>
                  ))}
                </div>

                {/* Spread */}
                <div className="text-center py-2 bg-gray-700 rounded mb-4 text-sm text-gray-300">
                  Spread: {formatCurrency(orderBook.asks[0]?.price - orderBook.bids[0]?.price || 0)}
                </div>

                {/* Bids */}
                <div>
                  {orderBook.bids.map((bid, idx) => (
                    <div key={idx} className="grid grid-cols-3 gap-2 py-1 text-sm text-primary-500">
                      <div>{formatCurrency(bid.price)}</div>
                      <div className="text-right">{bid.amount.toFixed(4)}</div>
                      <div className="text-right">{bid.total.toFixed(4)}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold text-white mb-4">📊 Market Depth</h3>
              <div className="panel-blue rounded-lg border border-primary-500 p-6 h-80 flex items-center justify-center">
                <div className="text-center text-gray-400">
                  <div className="text-4xl mb-2">📈</div>
                  <div>Market Depth Chart</div>
                  <div className="text-sm mt-2">Chart integration pending</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Heatmap Tab */}
        {activeSubTab === 'heatmap' && (
          <div>
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-2">🔥 Market Heatmaps</h3>
              <p className="text-gray-400 text-sm">Market cap and volume-based heatmaps with 24h price change colors</p>
            </div>
            
            {/* Market Cap Heatmap */}
            <div className="mb-6">
              <h4 className="text-md font-semibold text-white mb-3">📊 Market Cap Heatmap</h4>
              <div className="panel-blue rounded-lg border border-primary-500 p-4" style={{ height: '500px' }}>
                <TradingViewHeatmap 
                  dataSource="Crypto"
                  blockSize="market_cap_calc"
                  blockColor="24h_close_change|5"
                  colorTheme="dark"
                  hasTopBar={false}
                  isDataSetEnabled={false}
                  isZoomEnabled={true}
                  hasSymbolTooltip={true}
                  isMonoSize={false}
                  width="100%"
                  height="100%"
                  locale="en"
                />
              </div>
            </div>

            {/* Volume Heatmap */}
            <div className="mb-6">
              <h4 className="text-md font-semibold text-white mb-3">📈 Volume Heatmap</h4>
              <div className="panel-blue rounded-lg border border-primary-500 p-4" style={{ height: '500px' }}>
                <TradingViewHeatmap 
                  dataSource="Crypto"
                  blockSize="24h_vol_cmc"
                  blockColor="24h_close_change|5"
                  colorTheme="dark"
                  hasTopBar={false}
                  isDataSetEnabled={false}
                  isZoomEnabled={true}
                  hasSymbolTooltip={true}
                  isMonoSize={false}
                  width="100%"
                  height="100%"
                  locale="en"
                />
              </div>
            </div>
          </div>
        )}

        {/* Volume Tab */}
        {activeSubTab === 'volume' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">📊 Volume Analysis</h3>
            <div className="space-y-4">
              {marketData
                .sort((a, b) => b.volume - a.volume)
                .map((market) => (
                  <div key={market.symbol} className="panel-blue rounded-lg border border-primary-500 p-4">
                    <div className="flex justify-between items-center mb-3">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center text-white font-bold">
                          {market.symbol.charAt(0)}
                        </div>
                        <div>
                          <div className="text-white font-semibold">{market.symbol}</div>
                          <div className="text-gray-400 text-sm">{formatCurrency(market.price)}</div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-white font-semibold">{formatNumber(market.volume)}</div>
                        <div className="text-gray-400 text-sm">Volume</div>
                      </div>
                    </div>
                    <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-primary-500 rounded-full transition-all duration-300"
                        style={{ width: `${(market.volume / Math.max(...marketData.map(m => m.volume))) * 100}%` }}
                      ></div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}

        {/* Markets Tab */}
        {activeSubTab === 'markets' && (
          <div>
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-white mb-2">🏪 Crypto Markets</h3>
              <p className="text-gray-400 text-sm">Comprehensive crypto market screener with real-time data and analysis</p>
            </div>
            
            <div className="panel-blue rounded-lg border border-primary-500 p-4">
              <TradingViewCryptoMarkets />
            </div>
          </div>
        )}
      </div>
    )
  }

  // Working Portfolio Component
  const PortfolioTab = () => {
    const [portfolioData] = React.useState({
      totalValue: 125430.50,
      change24h: 2450,
      changePercent: 1.99,
      holdings: [
        { symbol: 'BTC', amount: 0.5, value: 57889, change: 2500 },
        { symbol: 'ETH', amount: 2.1, value: 9414, change: 1200 },
        { symbol: 'SOL', amount: 50, value: 11950, change: 800 },
        { symbol: 'USDT', amount: 10000, value: 10000, change: 0 },
      ]
    })

    const formatCurrency = (value: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)
    const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
    const getChangeColor = (value: number) => value >= 0 ? 'text-primary-500' : 'text-red-400'

    return (
      <div className="p-6 h-full overflow-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">💼 Portfolio</h2>
          <p className="text-gray-400">Your trading portfolio and performance</p>
        </div>

        {/* Portfolio Summary */}
        <div className="panel-blue p-6 rounded-lg border border-primary-500 mb-6">
          <div className="text-center">
            <div className="text-3xl font-bold text-white mb-2">{formatCurrency(portfolioData.totalValue)}</div>
            <div className={`text-lg ${getChangeColor(portfolioData.change24h)}`}>
              {formatCurrency(portfolioData.change24h)} ({formatPercent(portfolioData.changePercent)})
            </div>
            <div className="text-sm text-gray-400 mt-1">24h Change</div>
          </div>
        </div>

        {/* Holdings */}
        <div className="space-y-4">
          <h3 className="text-lg font-semibold text-white">Holdings</h3>
          {portfolioData.holdings.map((holding) => (
            <div key={holding.symbol} className="panel-blue p-4 rounded-lg border border-primary-500">
              <div className="flex justify-between items-center">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center text-white font-bold">
                    {holding.symbol.charAt(0)}
                  </div>
                  <div>
                    <div className="font-semibold text-white">{holding.symbol}</div>
                    <div className="text-sm text-gray-400">{holding.amount.toFixed(4)} tokens</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-semibold text-white">{formatCurrency(holding.value)}</div>
                  <div className={`text-sm ${getChangeColor(holding.change)}`}>
                    {formatCurrency(holding.change)}
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  // Working Charts Component with TradingView integration
  const ChartsTab = () => {
    const [selectedSymbol, setSelectedSymbol] = React.useState('BTC')
    const [timeframe, setTimeframe] = React.useState('1h')
    
    const symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'ARB', 'OP', 'DOGE', 'ADA', 'DOT', 'LINK']
    const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

    return (
      <div className="p-6 h-full overflow-auto">
        {/* TradingView Ticker Tape */}
        <div className="mb-4">
          <TradingViewTickerTape height={40} />
        </div>
        
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">📊 Professional Charts</h2>
          <p className="text-gray-400">Advanced trading charts and analysis</p>
        </div>
        
        {/* Chart Controls */}
        <div className="panel-blue p-4 rounded-lg border border-primary-500 mb-6">
          <div className="flex gap-4 items-center">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Symbol</label>
              <select 
                value={selectedSymbol} 
                onChange={(e) => setSelectedSymbol(e.target.value)}
                className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                {symbols.map(symbol => (
                  <option key={symbol} value={symbol}>{symbol}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Timeframe</label>
              <select 
                value={timeframe} 
                onChange={(e) => setTimeframe(e.target.value)}
                className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                {timeframes.map(tf => (
                  <option key={tf} value={tf}>{tf}</option>
                ))}
              </select>
            </div>
            <div className="ml-auto">
              <button className="bg-gray-600 hover:bg-primary-500/10 px-4 py-2 rounded text-white">
                🔄 Refresh
              </button>
            </div>
          </div>
        </div>

        {/* Chart Area */}
        <div className="panel-blue p-6 rounded-lg border border-primary-500 h-96 flex items-center justify-center">
          <div className="text-center">
            <div className="text-6xl mb-4">📈</div>
            <div className="text-xl text-white mb-2">{selectedSymbol} Chart ({timeframe})</div>
            <div className="text-gray-400">TradingView integration coming soon</div>
            <div className="text-sm text-gray-500 mt-2">Price: $115,778 | Change: -0.18%</div>
          </div>
        </div>
      </div>
    )
  }

  // Comprehensive Backtesting Component with all QuantDesk Lite features
  const BacktestingTab = () => {
    const [activeSubTab, setActiveSubTab] = React.useState<'new' | 'results' | 'history'>('new')
    const [selectedStrategy, setSelectedStrategy] = React.useState('')
    const [config, setConfig] = React.useState({
      strategy: '',
      symbol: 'BTC',
      timeframe: '1h',
      startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      endDate: new Date().toISOString().split('T')[0],
      initialCapital: 10000,
      parameters: {} as Record<string, any>
    })
    const [isRunning, setIsRunning] = React.useState(false)
    const [backtests, setBacktests] = React.useState<any[]>([])

    // Comprehensive strategy definitions
    const strategies = [
      {
        id: 'lorentzian',
        name: 'Lorentzian Classification',
        description: 'Advanced ML-based classification strategy using Lorentzian distance',
        category: 'Machine Learning',
        parameters: [
          { name: 'lookback_window', type: 'number', value: 8, min: 5, max: 20, step: 1 },
          { name: 'feature_count', type: 'number', value: 4, min: 2, max: 8, step: 1 },
          { name: 'max_bars_for_trades', type: 'number', value: 10, min: 5, max: 50, step: 1 },
          { name: 'use_auto_filter', type: 'boolean', value: true }
        ]
      },
      {
        id: 'rsi_divergence',
        name: 'RSI Divergence',
        description: 'Classic RSI divergence strategy with trend confirmation',
        category: 'Technical Analysis',
        parameters: [
          { name: 'rsi_period', type: 'number', value: 14, min: 5, max: 30, step: 1 },
          { name: 'rsi_oversold', type: 'number', value: 30, min: 20, max: 40, step: 1 },
          { name: 'rsi_overbought', type: 'number', value: 70, min: 60, max: 80, step: 1 },
          { name: 'divergence_lookback', type: 'number', value: 5, min: 3, max: 10, step: 1 }
        ]
      },
      {
        id: 'moving_average_crossover',
        name: 'Moving Average Crossover',
        description: 'Simple moving average crossover strategy',
        category: 'Trend Following',
        parameters: [
          { name: 'fast_ma', type: 'number', value: 10, min: 5, max: 20, step: 1 },
          { name: 'slow_ma', type: 'number', value: 20, min: 10, max: 50, step: 1 },
          { name: 'signal_threshold', type: 'number', value: 0.01, min: 0.001, max: 0.1, step: 0.001 }
        ]
      },
      {
        id: 'bollinger_bands',
        name: 'Bollinger Bands Mean Reversion',
        description: 'Mean reversion strategy using Bollinger Bands',
        category: 'Mean Reversion',
        parameters: [
          { name: 'bb_period', type: 'number', value: 20, min: 10, max: 50, step: 1 },
          { name: 'bb_std', type: 'number', value: 2, min: 1, max: 3, step: 0.1 },
          { name: 'rsi_period', type: 'number', value: 14, min: 5, max: 30, step: 1 }
        ]
      }
    ]

    // Mock backtest history
    React.useEffect(() => {
      const mockBacktests = [
        {
          id: '1',
          status: 'completed',
          strategy: 'lorentzian',
          symbol: 'BTC',
          timeframe: '1h',
          startDate: '2024-01-01',
          endDate: '2024-01-31',
          results: {
            totalReturn: 2450.50,
            totalReturnPercent: 24.51,
            maxDrawdown: -8.5,
            sharpeRatio: 1.85,
            winRate: 68.5,
            totalTrades: 45,
            profitFactor: 1.42
          },
          createdAt: new Date(Date.now() - 3600000)
        },
        {
          id: '2',
          status: 'completed',
          strategy: 'rsi_divergence',
          symbol: 'ETH',
          timeframe: '4h',
          startDate: '2024-01-01',
          endDate: '2024-01-31',
          results: {
            totalReturn: 1200.00,
            totalReturnPercent: 12.00,
            maxDrawdown: -5.2,
            sharpeRatio: 1.25,
            winRate: 55.0,
            totalTrades: 28,
            profitFactor: 1.18
          },
          createdAt: new Date(Date.now() - 7200000)
        },
        {
          id: '3',
          status: 'running',
          strategy: 'moving_average_crossover',
          symbol: 'SOL',
          timeframe: '1h',
          startDate: '2024-02-01',
          endDate: '2024-02-28',
          createdAt: new Date(Date.now() - 300000)
        }
      ]
      setBacktests(mockBacktests)
    }, [])

    const formatCurrency = (value: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)
    const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
    const getStatusColor = (status: string) => {
      switch (status) {
        case 'completed': return 'text-primary-500'
        case 'running': return 'text-gray-400'
        case 'failed': return 'text-red-400'
        default: return 'text-gray-400'
      }
    }
    const getReturnColor = (value: number) => value >= 0 ? 'text-primary-500' : 'text-red-400'

    const handleStrategySelect = (strategyId: string) => {
      const strategy = strategies.find(s => s.id === strategyId)
      setSelectedStrategy(strategyId)
      setConfig(prev => ({
        ...prev,
        strategy: strategyId,
        parameters: strategy?.parameters.reduce((acc, param) => {
          acc[param.name] = param.value
          return acc
        }, {} as Record<string, any>) || {}
      }))
    }

    const handleParameterChange = (paramName: string, value: any) => {
      setConfig(prev => ({
        ...prev,
        parameters: {
          ...prev.parameters,
          [paramName]: value
        }
      }))
    }

    const handleRunBacktest = async () => {
      if (!config.strategy || !config.symbol) return

      setIsRunning(true)
      const newBacktest = {
        id: Date.now().toString(),
        status: 'running',
        strategy: config.strategy,
        symbol: config.symbol,
        timeframe: config.timeframe,
        startDate: config.startDate,
        endDate: config.endDate,
        createdAt: new Date()
      }

      setBacktests(prev => [newBacktest, ...prev])
      setActiveSubTab('results')

      // Simulate API call
      setTimeout(() => {
        const mockResults = {
          totalReturn: Math.random() * 5000 - 1000,
          totalReturnPercent: Math.random() * 50 - 10,
          maxDrawdown: -(Math.random() * 20),
          sharpeRatio: Math.random() * 3,
          winRate: Math.random() * 100,
          totalTrades: Math.floor(Math.random() * 100),
          profitFactor: Math.random() * 2
        }

        setBacktests(prev => prev.map(bt => 
          bt.id === newBacktest.id 
            ? { ...bt, status: 'completed', results: mockResults }
            : bt
        ))
        setIsRunning(false)
      }, 5000)
    }

    return (
      <div className="p-6 h-full overflow-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">🧪 Backtesting</h2>
          <p className="text-gray-400">Test your trading strategies with historical data</p>
        </div>

        {/* Sub-navigation tabs */}
        <div className="flex gap-2 mb-6 border-b border-primary-500">
          {[
            { id: 'new', label: '🆕 New Backtest' },
            { id: 'results', label: '📊 Results' },
            { id: 'history', label: '📋 History' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id as any)}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                activeSubTab === tab.id
                  ? 'bg-primary-500 text-white border-b-2 border-primary-500'
                  : 'text-gray-400 hover:text-white hover:bg-primary-500/10'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* New Backtest Tab */}
        {activeSubTab === 'new' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Strategy Selection */}
            <div className="panel-blue rounded-lg border border-primary-500 p-6">
              <h3 className="text-lg font-semibold text-white mb-4">📋 Select Strategy</h3>
              <div className="space-y-3">
                {strategies.map((strategy) => (
                  <div
                    key={strategy.id}
                    onClick={() => handleStrategySelect(strategy.id)}
                    className={`p-4 rounded-lg border cursor-pointer transition-colors ${
                      config.strategy === strategy.id
                        ? 'bg-gray-700 border-primary-500'
                        : 'bg-gray-900 border-gray-600 hover:bg-primary-500/10'
                    }`}
                  >
                    <div className="flex justify-between items-center mb-2">
                      <div className="text-white font-semibold">{strategy.name}</div>
                      <div className="text-xs text-gray-400 bg-gray-700 px-2 py-1 rounded">
                        {strategy.category}
                      </div>
                    </div>
                    <div className="text-sm text-gray-400">{strategy.description}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Configuration */}
            <div className="panel-blue rounded-lg border border-primary-500 p-6">
              <h3 className="text-lg font-semibold text-white mb-4">⚙️ Configuration</h3>
              
              {selectedStrategy && (
                <div className="space-y-4">
                  {/* Basic Settings */}
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Symbol</label>
                      <select
                        value={config.symbol}
                        onChange={(e) => setConfig(prev => ({ ...prev, symbol: e.target.value }))}
                        className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white text-sm"
                      >
                        <option value="BTC">BTC</option>
                        <option value="ETH">ETH</option>
                        <option value="SOL">SOL</option>
                        <option value="ADA">ADA</option>
                        <option value="DOT">DOT</option>
                      </select>
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Timeframe</label>
                      <select
                        value={config.timeframe}
                        onChange={(e) => setConfig(prev => ({ ...prev, timeframe: e.target.value }))}
                        className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white text-sm"
                      >
                        <option value="1m">1 Minute</option>
                        <option value="5m">5 Minutes</option>
                        <option value="15m">15 Minutes</option>
                        <option value="1h">1 Hour</option>
                        <option value="4h">4 Hours</option>
                        <option value="1d">1 Day</option>
                      </select>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">Start Date</label>
                      <input
                        type="date"
                        value={config.startDate}
                        onChange={(e) => setConfig(prev => ({ ...prev, startDate: e.target.value }))}
                        className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white text-sm"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-gray-400 mb-2">End Date</label>
                      <input
                        type="date"
                        value={config.endDate}
                        onChange={(e) => setConfig(prev => ({ ...prev, endDate: e.target.value }))}
                        className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white text-sm"
                      />
                    </div>
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Initial Capital</label>
                    <input
                      type="number"
                      value={config.initialCapital}
                      onChange={(e) => setConfig(prev => ({ ...prev, initialCapital: Number(e.target.value) }))}
                      className="w-full bg-gray-900 border border-gray-600 rounded px-3 py-2 text-white text-sm"
                    />
                  </div>

                  {/* Strategy Parameters */}
                  <div>
                    <label className="block text-sm text-gray-400 mb-3">Strategy Parameters</label>
                    <div className="space-y-3">
                      {strategies.find(s => s.id === selectedStrategy)?.parameters.map((param) => (
                        <div key={param.name}>
                          <label className="block text-xs text-gray-500 mb-1">
                            {param.name.replace(/_/g, ' ').toUpperCase()}
                          </label>
                          {param.type === 'number' ? (
                            <input
                              type="number"
                              value={config.parameters[param.name] || param.value}
                              onChange={(e) => handleParameterChange(param.name, Number(e.target.value))}
                              min={param.min}
                              max={param.max}
                              step={param.step}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-white text-xs"
                            />
                          ) : param.type === 'boolean' ? (
                            <select
                              value={config.parameters[param.name] || param.value}
                              onChange={(e) => handleParameterChange(param.name, e.target.value === 'true')}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-white text-xs"
                            >
                              <option value="true">True</option>
                              <option value="false">False</option>
                            </select>
                          ) : (
                            <input
                              type="text"
                              value={config.parameters[param.name] || param.value}
                              onChange={(e) => handleParameterChange(param.name, e.target.value)}
                              className="w-full bg-gray-900 border border-gray-600 rounded px-2 py-1 text-white text-xs"
                            />
                          )}
                        </div>
                      ))}
                    </div>
                  </div>

                  <button
                    onClick={handleRunBacktest}
                    disabled={isRunning || !config.strategy}
                    className={`w-full py-3 rounded-lg font-semibold transition-colors ${
                      isRunning || !config.strategy
                        ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                        : 'bg-primary-600 hover:bg-primary-700 text-white'
                    }`}
                  >
                    {isRunning ? '🔄 Running Backtest...' : '🚀 Run Backtest'}
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Results Tab */}
        {activeSubTab === 'results' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">📊 Backtest Results</h3>
            <div className="space-y-4">
              {backtests.filter(bt => bt.status === 'completed' || bt.status === 'running').map((backtest) => (
                <div key={backtest.id} className="panel-blue rounded-lg border border-primary-500 p-6">
                  <div className="flex justify-between items-center mb-4">
                    <div>
                      <div className="text-lg font-semibold text-white">
                        {strategies.find(s => s.id === backtest.strategy)?.name || backtest.strategy}
                      </div>
                      <div className="text-sm text-gray-400">
                        {backtest.symbol} • {backtest.timeframe} • {backtest.startDate} to {backtest.endDate}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className={`px-3 py-1 rounded-full text-xs font-semibold ${
                        backtest.status === 'completed' ? 'bg-primary-600 text-white' :
                        backtest.status === 'running' ? 'bg-gray-600 text-white' :
                        'bg-red-600 text-white'
                      }`}>
                        {backtest.status.toUpperCase()}
                      </div>
                      <div className="text-xs text-gray-400">
                        {backtest.createdAt.toLocaleString()}
                      </div>
                    </div>
                  </div>

                  {backtest.results && (
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                      <div className="text-center">
                        <div className={`text-xl font-bold ${getReturnColor(backtest.results.totalReturnPercent)}`}>
                          {formatPercent(backtest.results.totalReturnPercent)}
                        </div>
                        <div className="text-xs text-gray-400">Total Return</div>
                      </div>
                      <div className="text-center">
                        <div className="text-xl font-bold text-white">
                          {formatCurrency(backtest.results.totalReturn)}
                        </div>
                        <div className="text-xs text-gray-400">Profit/Loss</div>
                      </div>
                      <div className="text-center">
                        <div className="text-xl font-bold text-red-400">
                          {formatPercent(backtest.results.maxDrawdown)}
                        </div>
                        <div className="text-xs text-gray-400">Max Drawdown</div>
                      </div>
                      <div className="text-center">
                        <div className="text-xl font-bold text-white">
                          {backtest.results.sharpeRatio.toFixed(2)}
                        </div>
                        <div className="text-xs text-gray-400">Sharpe Ratio</div>
                      </div>
                      <div className="text-center">
                        <div className="text-xl font-bold text-primary-500">
                          {formatPercent(backtest.results.winRate)}
                        </div>
                        <div className="text-xs text-gray-400">Win Rate</div>
                      </div>
                      <div className="text-center">
                        <div className="text-xl font-bold text-white">
                          {backtest.results.totalTrades}
                        </div>
                        <div className="text-xs text-gray-400">Total Trades</div>
                      </div>
                    </div>
                  )}

                  {backtest.status === 'running' && (
                    <div className="text-center py-8">
                      <div className="text-4xl mb-4">🔄</div>
                      <div className="text-gray-400">
                        Running backtest... This may take a few minutes.
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* History Tab */}
        {activeSubTab === 'history' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">📋 Backtest History</h3>
            <div className="space-y-3">
              {backtests.map((backtest) => (
                <div key={backtest.id} className="panel-blue rounded-lg border border-primary-500 p-4">
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="text-white font-semibold">
                        {strategies.find(s => s.id === backtest.strategy)?.name || backtest.strategy}
                      </div>
                      <div className="text-sm text-gray-400">
                        {backtest.symbol} • {backtest.timeframe} • {backtest.createdAt.toLocaleString()}
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      {backtest.results && (
                        <div className="text-right">
                          <div className={`text-sm font-semibold ${getReturnColor(backtest.results.totalReturnPercent)}`}>
                            {formatPercent(backtest.results.totalReturnPercent)}
                          </div>
                          <div className="text-xs text-gray-400">
                            {formatCurrency(backtest.results.totalReturn)}
                          </div>
                        </div>
                      )}
                      <div className={`px-2 py-1 rounded text-xs font-semibold ${
                        backtest.status === 'completed' ? 'bg-primary-600 text-white' :
                        backtest.status === 'running' ? 'bg-gray-600 text-white' :
                        'bg-red-600 text-white'
                      }`}>
                        {backtest.status.toUpperCase()}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  // Comprehensive Whales Component with all QuantDesk Lite features
  const WhalesTab = () => {
    const [activeSubTab, setActiveSubTab] = React.useState<'whales' | 'institutional' | 'lp' | 'alerts'>('whales')
    const [filterSymbol, setFilterSymbol] = React.useState('ALL')
    const [minValue, setMinValue] = React.useState(1000000) // $1M minimum

    // Comprehensive whale transaction data
    const [whaleTransactions] = React.useState([
      {
        id: '1',
        symbol: 'BTC',
        amount: 150.5,
        value: 13545000,
        type: 'buy',
        from: 'Binance',
        to: 'Cold Wallet',
        timestamp: new Date(Date.now() - 300000),
        exchange: 'Binance',
        confidence: 95,
        impact: 'extreme'
      },
      {
        id: '2',
        symbol: 'ETH',
        amount: 2500,
        value: 8750000,
        type: 'sell',
        from: 'Institutional Wallet',
        to: 'Coinbase',
        timestamp: new Date(Date.now() - 600000),
        exchange: 'Coinbase',
        confidence: 88,
        impact: 'high'
      },
      {
        id: '3',
        symbol: 'SOL',
        amount: 50000,
        value: 12000000,
        type: 'transfer',
        from: 'FTX Estate',
        to: 'Unknown',
        timestamp: new Date(Date.now() - 900000),
        exchange: 'Unknown',
        confidence: 92,
        impact: 'high'
      },
      {
        id: '4',
        symbol: 'ADA',
        amount: 10000000,
        value: 8500000,
        type: 'buy',
        from: 'Kraken',
        to: 'Staking Pool',
        timestamp: new Date(Date.now() - 1200000),
        exchange: 'Kraken',
        confidence: 75,
        impact: 'medium'
      },
      {
        id: '5',
        symbol: 'DOT',
        amount: 100000,
        value: 8500000,
        type: 'sell',
        from: 'Parity Wallet',
        to: 'Binance',
        timestamp: new Date(Date.now() - 1800000),
        exchange: 'Binance',
        confidence: 98,
        impact: 'extreme'
      },
      {
        id: '6',
        symbol: 'MATIC',
        amount: 2000000,
        value: 2500000,
        type: 'buy',
        from: 'Coinbase',
        to: 'DeFi Protocol',
        timestamp: new Date(Date.now() - 2400000),
        exchange: 'Coinbase',
        confidence: 82,
        impact: 'medium'
      }
    ])

    // Institutional flow data
    const [institutionalFlows] = React.useState([
      {
        id: '1',
        symbol: 'BTC',
        netFlow: 25000000,
        inflow: 45000000,
        outflow: 20000000,
        largeTransactions: 12,
        avgTransactionSize: 3750000,
        timestamp: new Date(),
        trend: 'increasing'
      },
      {
        id: '2',
        symbol: 'ETH',
        netFlow: -15000000,
        inflow: 20000000,
        outflow: 35000000,
        largeTransactions: 8,
        avgTransactionSize: 4375000,
        timestamp: new Date(),
        trend: 'decreasing'
      },
      {
        id: '3',
        symbol: 'SOL',
        netFlow: 5000000,
        inflow: 15000000,
        outflow: 10000000,
        largeTransactions: 5,
        avgTransactionSize: 3000000,
        timestamp: new Date(),
        trend: 'stable'
      },
      {
        id: '4',
        symbol: 'ADA',
        netFlow: 8000000,
        inflow: 12000000,
        outflow: 4000000,
        largeTransactions: 3,
        avgTransactionSize: 4000000,
        timestamp: new Date(),
        trend: 'increasing'
      }
    ])

    // LP activity data
    const [lpActivities] = React.useState([
      {
        id: '1',
        pool: 'ETH/USDC',
        symbol: 'ETH',
        action: 'add_liquidity',
        amount: 1000,
        value: 3500000,
        timestamp: new Date(Date.now() - 300000),
        impact: 'high'
      },
      {
        id: '2',
        pool: 'SOL/USDT',
        symbol: 'SOL',
        action: 'remove_liquidity',
        amount: 50000,
        value: 12000000,
        timestamp: new Date(Date.now() - 600000),
        impact: 'extreme'
      },
      {
        id: '3',
        pool: 'BTC/ETH',
        symbol: 'BTC',
        action: 'swap',
        amount: 25,
        value: 2250000,
        timestamp: new Date(Date.now() - 900000),
        impact: 'medium'
      },
      {
        id: '4',
        pool: 'ADA/USDT',
        symbol: 'ADA',
        action: 'add_liquidity',
        amount: 5000000,
        value: 4250000,
        timestamp: new Date(Date.now() - 1200000),
        impact: 'high'
      },
      {
        id: '5',
        pool: 'DOT/USDC',
        symbol: 'DOT',
        action: 'swap',
        amount: 50000,
        value: 4250000,
        timestamp: new Date(Date.now() - 1500000),
        impact: 'medium'
      }
    ])

    const formatCurrency = (value: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(value)
    const formatNumber = (value: number) => {
      if (value >= 1e9) return (value / 1e9).toFixed(1) + 'B'
      if (value >= 1e6) return (value / 1e6).toFixed(1) + 'M'
      if (value >= 1e3) return (value / 1e3).toFixed(1) + 'K'
      return value.toFixed(0)
    }

    const getImpactColor = (impact: string) => {
      switch (impact) {
        case 'extreme': return 'text-red-400'
        case 'high': return 'text-orange-400'
        case 'medium': return 'text-gray-400'
        case 'low': return 'text-primary-500'
        default: return 'text-gray-400'
      }
    }

    const getTypeColor = (type: string) => {
      switch (type) {
        case 'buy': return 'text-primary-500'
        case 'sell': return 'text-red-400'
        case 'transfer': return 'text-gray-400'
        default: return 'text-gray-400'
      }
    }

    const getTrendColor = (trend: string) => {
      switch (trend) {
        case 'increasing': return 'text-primary-500'
        case 'decreasing': return 'text-red-400'
        case 'stable': return 'text-gray-400'
        default: return 'text-gray-400'
      }
    }

    const getActionIcon = (action: string) => {
      switch (action) {
        case 'add_liquidity': return '➕'
        case 'remove_liquidity': return '➖'
        case 'swap': return '🔄'
        default: return '❓'
      }
    }

    const filteredWhaleTransactions = whaleTransactions.filter(tx => 
      (filterSymbol === 'ALL' || tx.symbol === filterSymbol) && 
      tx.value >= minValue
    )

    return (
      <div className="p-6 h-full overflow-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">🐋 Whale Watcher</h2>
          <p className="text-gray-400">Track large transactions and institutional trading patterns</p>
        </div>

        {/* Filters */}
        <div className="flex gap-4 mb-6">
          <select
            value={filterSymbol}
            onChange={(e) => setFilterSymbol(e.target.value)}
            className="panel-blue border border-gray-600 rounded px-3 py-2 text-white text-sm"
          >
            <option value="ALL">All Symbols</option>
            <option value="BTC">BTC</option>
            <option value="ETH">ETH</option>
            <option value="SOL">SOL</option>
            <option value="ADA">ADA</option>
            <option value="DOT">DOT</option>
            <option value="MATIC">MATIC</option>
          </select>
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-400">Min Value:</label>
            <input
              type="number"
              value={minValue}
              onChange={(e) => setMinValue(Number(e.target.value))}
              className="w-32 panel-blue border border-gray-600 rounded px-2 py-1 text-white text-sm"
            />
          </div>
        </div>

        {/* Sub-navigation tabs */}
        <div className="flex gap-2 mb-6 border-b border-primary-500">
          {[
            { id: 'whales', label: '🐋 Whale Transactions' },
            { id: 'institutional', label: '🏛️ Institutional Flow' },
            { id: 'lp', label: '💧 LP Activity' },
            { id: 'alerts', label: '🚨 Alerts' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id as any)}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                activeSubTab === tab.id
                  ? 'bg-primary-500 text-white border-b-2 border-primary-500'
                  : 'text-gray-400 hover:text-white hover:bg-primary-500/10'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Whale Transactions Tab */}
        {activeSubTab === 'whales' && (
          <div>
            <div className="flex justify-between items-center mb-4">
              <h3 className="text-lg font-semibold text-white">
                🐋 Whale Activity ({filteredWhaleTransactions.length})
              </h3>
              <div className="text-sm text-gray-400">
                Large on-chain and order-flow transactions
              </div>
            </div>
            
            <div className="panel-blue rounded-lg border border-primary-500 overflow-hidden">
              <div className="grid grid-cols-6 px-4 py-3 text-xs text-gray-400 border-b border-primary-500 font-semibold">
                <div>Time</div>
                <div>Symbol</div>
                <div>Side</div>
                <div>Size (USD)</div>
                <div>Exchange</div>
                <div>Impact</div>
              </div>
              {filteredWhaleTransactions.map((transaction) => (
                <div key={transaction.id} className="grid grid-cols-6 px-4 py-3 text-sm border-b border-primary-500/60 hover:bg-primary-500/10/60 transition-colors">
                  <div className="text-gray-300">{transaction.timestamp.toLocaleTimeString()}</div>
                  <div className="text-white font-medium">{transaction.symbol}</div>
                  <div className={`font-medium ${getTypeColor(transaction.type)}`}>
                    {transaction.type.toUpperCase()}
                  </div>
                  <div className="text-white">{formatCurrency(transaction.value)}</div>
                  <div className="text-gray-300">{transaction.exchange || 'N/A'}</div>
                  <div className={`${getImpactColor(transaction.impact)}`}>
                    {transaction.impact}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Institutional Flow Tab */}
        {activeSubTab === 'institutional' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">🏛️ Institutional Flow Analysis</h3>
            <div className="space-y-4">
              {institutionalFlows.map((flow) => (
                <div key={flow.id} className="panel-blue rounded-lg border border-primary-500 p-6">
                  <div className="flex justify-between items-center mb-4">
                    <div className="flex items-center gap-4">
                      <div className="w-12 h-12 bg-gray-700 rounded-full flex items-center justify-center text-white font-bold text-lg">
                        {flow.symbol.charAt(0)}
                      </div>
                      <div>
                        <div className="text-lg font-semibold text-white">
                          {flow.symbol} Institutional Flow
                        </div>
                        <div className="text-sm text-gray-400">
                          {flow.timestamp.toLocaleString()}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className={`text-xl font-bold ${flow.netFlow >= 0 ? 'text-primary-500' : 'text-red-400'}`}>
                        {formatCurrency(flow.netFlow)}
                      </div>
                      <div className={`text-sm font-semibold ${getTrendColor(flow.trend)}`}>
                        {flow.trend.toUpperCase()}
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-lg font-bold text-primary-500">
                        {formatCurrency(flow.inflow)}
                      </div>
                      <div className="text-xs text-gray-400">Inflow</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-red-400">
                        {formatCurrency(flow.outflow)}
                      </div>
                      <div className="text-xs text-gray-400">Outflow</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-white">
                        {flow.largeTransactions}
                      </div>
                      <div className="text-xs text-gray-400">Large Txs</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold text-gray-400">
                        {formatCurrency(flow.avgTransactionSize)}
                      </div>
                      <div className="text-xs text-gray-400">Avg Size</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* LP Activity Tab */}
        {activeSubTab === 'lp' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">💧 Liquidity Provider Activity</h3>
            <div className="space-y-4">
              {lpActivities.map((activity) => (
                <div key={activity.id} className="panel-blue rounded-lg border border-primary-500 p-6">
                  <div className="flex justify-between items-center mb-4">
                    <div className="flex items-center gap-4">
                      <div className="text-2xl">
                        {getActionIcon(activity.action)}
                      </div>
                      <div>
                        <div className="text-lg font-semibold text-white">
                          {activity.action.replace('_', ' ').toUpperCase()}
                        </div>
                        <div className="text-sm text-gray-400">
                          {activity.pool} • {activity.timestamp.toLocaleString()}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-white">
                        {formatCurrency(activity.value)}
                      </div>
                      <div className="text-sm text-gray-400">
                        {activity.amount.toLocaleString()} {activity.symbol}
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <div>
                      <div className="text-xs text-gray-400">Pool</div>
                      <div className="text-sm text-white">{activity.pool}</div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Impact</div>
                      <div className={`text-sm font-semibold ${getImpactColor(activity.impact)}`}>
                        {activity.impact.toUpperCase()}
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Alerts Tab */}
        {activeSubTab === 'alerts' && (
          <div className="text-center py-16">
            <div className="text-6xl mb-6">🚨</div>
            <div className="text-2xl font-bold text-white mb-3">
              Whale Alerts
            </div>
            <div className="text-gray-400 mb-6">
              Customizable alerts for large transactions and institutional activity
            </div>
            <div className="panel-blue rounded-lg border border-primary-500 p-6 max-w-md mx-auto">
              <h4 className="text-lg font-semibold text-white mb-4">Alert Settings</h4>
              <div className="space-y-4 text-left">
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Large Transactions (&gt;$1M)</span>
                  <div className="w-12 h-6 bg-primary-600 rounded-full relative">
                    <div className="w-5 h-5 bg-white rounded-full absolute right-0.5 top-0.5"></div>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">Institutional Flow Changes</span>
                  <div className="w-12 h-6 bg-primary-600 rounded-full relative">
                    <div className="w-5 h-5 bg-white rounded-full absolute right-0.5 top-0.5"></div>
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-300">LP Activity Alerts</span>
                  <div className="w-12 h-6 bg-gray-600 rounded-full relative">
                    <div className="w-5 h-5 bg-white rounded-full absolute left-0.5 top-0.5"></div>
                  </div>
                </div>
              </div>
              <div className="text-xs text-gray-500 mt-4">
                Coming soon: Real-time notifications and custom thresholds
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  // Comprehensive Sentiment Component with dashboard-style metrics
  const SentimentTab = () => {
    const [activeSubTab, setActiveSubTab] = React.useState<'dashboard' | 'heatmap' | 'events' | 'trending'>('dashboard')

    // Dashboard-style sentiment metrics
    const [sentimentMetrics] = React.useState([
      {
        id: 'fear_greed',
        title: 'Fear & Greed',
        value: 62,
        bias: 'Bullish bias',
        color: 'green',
        description: 'Market sentiment based on multiple indicators'
      },
      {
        id: 'social_buzz',
        title: 'Social Buzz',
        value: 74,
        bias: 'Bullish bias',
        color: 'green',
        description: 'Social media sentiment and engagement'
      },
      {
        id: 'news_sentiment',
        title: 'News Sentiment',
        value: 58,
        bias: 'Bullish bias',
        color: 'green',
        description: 'News and media sentiment analysis'
      },
      {
        id: 'funding_skew',
        title: 'Funding Skew',
        value: 41,
        bias: 'Bearish bias',
        color: 'red',
        description: 'Perpetual funding rate analysis'
      }
    ])

    // Keyword heatmap data
    const [keywordHeatmap] = React.useState([
      { keyword: 'ETF', sentiment: 'positive', color: 'green' },
      { keyword: 'Airdrop', sentiment: 'positive', color: 'green' },
      { keyword: 'L2', sentiment: 'positive', color: 'green' },
      { keyword: 'Hack', sentiment: 'positive', color: 'green' },
      { keyword: 'Halving', sentiment: 'neutral', color: 'brown' },
      { keyword: 'ETF Outflows', sentiment: 'neutral', color: 'brown' },
      { keyword: 'Restaking', sentiment: 'neutral', color: 'brown' },
      { keyword: 'ETF Approval', sentiment: 'neutral', color: 'brown' },
      { keyword: 'DeFi', sentiment: 'negative', color: 'red' },
      { keyword: 'ETF Delay', sentiment: 'negative', color: 'red' },
      { keyword: 'ETF Inflows', sentiment: 'negative', color: 'red' },
      { keyword: 'Rug', sentiment: 'negative', color: 'red' }
    ])

    // Market events data
    const [marketEvents] = React.useState([
      {
        id: '1',
        title: 'Bitcoin ETF Approval Expected',
        description: 'Major institutional investors expect Bitcoin ETF approval within Q4',
        impact: 'high',
        timestamp: new Date(Date.now() - 3600000),
        category: 'news',
        sentiment: 0.8
      },
      {
        id: '2',
        title: 'Ethereum Shanghai Upgrade',
        description: 'Ethereum successfully completes Shanghai upgrade, enabling staking withdrawals',
        impact: 'medium',
        timestamp: new Date(Date.now() - 7200000),
        category: 'technical',
        sentiment: 0.6
      },
      {
        id: '3',
        title: 'Regulatory Clarity on DeFi',
        description: 'SEC provides clearer guidelines for DeFi protocols',
        impact: 'high',
        timestamp: new Date(Date.now() - 10800000),
        category: 'regulatory',
        sentiment: 0.7
      },
      {
        id: '4',
        title: 'Solana Network Outage',
        description: 'Solana experiences brief network congestion, quickly resolved',
        impact: 'medium',
        timestamp: new Date(Date.now() - 14400000),
        category: 'technical',
        sentiment: -0.3
      }
    ])

    // Trending topics data
    const [trendingTopics] = React.useState([
      {
        id: '1',
        topic: 'Bitcoin ETF',
        mentions: 125000,
        sentiment: 0.8,
        change: 15.5,
        category: 'news',
        platform: 'twitter'
      },
      {
        id: '2',
        topic: 'Ethereum Staking',
        mentions: 89000,
        sentiment: 0.6,
        change: 8.2,
        category: 'technical',
        platform: 'reddit'
      },
      {
        id: '3',
        topic: 'Solana Breakout',
        mentions: 67000,
        sentiment: 0.9,
        change: 25.3,
        category: 'social',
        platform: 'telegram'
      },
      {
        id: '4',
        topic: 'DeFi Regulation',
        mentions: 45000,
        sentiment: 0.4,
        change: -5.1,
        category: 'regulatory',
        platform: 'discord'
      }
    ])

    const getSentimentColor = (sentiment: number) => {
      if (sentiment >= 0.6) return 'text-primary-500'
      if (sentiment >= 0.2) return 'text-green-300'
      if (sentiment >= -0.2) return 'text-yellow-400'
      if (sentiment >= -0.6) return 'text-red-300'
      return 'text-red-400'
    }

    const getSentimentText = (sentiment: number) => {
      if (sentiment >= 0.6) return 'Very Bullish'
      if (sentiment >= 0.2) return 'Bullish'
      if (sentiment >= -0.2) return 'Neutral'
      if (sentiment >= -0.6) return 'Bearish'
      return 'Very Bearish'
    }

    const getImpactColor = (impact: string) => {
      switch (impact) {
        case 'critical': return 'text-red-400'
        case 'high': return 'text-orange-400'
        case 'medium': return 'text-gray-400'
        case 'low': return 'text-primary-500'
        default: return 'text-gray-400'
      }
    }

    const getCategoryColor = (category: string) => {
      switch (category) {
        case 'news': return 'bg-gray-600'
        case 'technical': return 'bg-primary-600'
        case 'social': return 'bg-purple-600'
        case 'regulatory': return 'bg-orange-600'
        default: return 'bg-gray-600'
      }
    }

    const getPlatformIcon = (platform: string) => {
      switch (platform) {
        case 'twitter': return '🐦'
        case 'reddit': return '🔴'
        case 'telegram': return '✈️'
        case 'discord': return '💬'
        default: return '📱'
      }
    }

    const formatNumber = (value: number) => {
      if (value >= 1e6) return (value / 1e6).toFixed(1) + 'M'
      if (value >= 1e3) return (value / 1e3).toFixed(1) + 'K'
      return value.toFixed(0)
    }

    const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`

    return (
      <div className="p-6 h-full overflow-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">🔥 Sentiment Dashboard</h2>
          <p className="text-gray-400">Aggregate market mood and directional bias</p>
        </div>

        {/* Sub-navigation tabs */}
        <div className="flex gap-2 mb-6 border-b border-primary-500">
          {[
            { id: 'dashboard', label: '📊 Dashboard' },
            { id: 'heatmap', label: '🔥 Heatmap' },
            { id: 'events', label: '📰 Events' },
            { id: 'trending', label: '📈 Trending' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id as any)}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                activeSubTab === tab.id
                  ? 'bg-primary-500 text-white border-b-2 border-primary-500'
                  : 'text-gray-400 hover:text-white hover:bg-primary-500/10'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Dashboard Tab */}
        {activeSubTab === 'dashboard' && (
          <div>
            {/* Sentiment Metrics Grid */}
            <div className="grid grid-cols-2 gap-6 mb-8">
              {sentimentMetrics.map((metric) => (
                <div key={metric.id} className="panel-blue rounded-lg border border-primary-500 p-6">
                  <div className="flex justify-between items-center mb-4">
                    <div>
                      <h3 className="text-lg font-semibold text-white">{metric.title}</h3>
                      <p className="text-sm text-gray-400">{metric.description}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-3xl font-bold text-white">{metric.value}</div>
                      <div className={`text-sm font-medium ${
                        metric.color === 'green' ? 'text-primary-500' : 'text-red-400'
                      }`}>
                        {metric.bias}
                      </div>
                    </div>
                  </div>
                  <div className="w-full h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-300 ${
                        metric.color === 'green' ? 'bg-primary-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${metric.value}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>

            {/* Keyword Heatmap */}
            <div className="panel-blue rounded-lg border border-primary-500 p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Keyword Heatmap</h3>
              <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-3">
                {keywordHeatmap.map((item, index) => (
                  <div
                    key={index}
                    className={`px-3 py-2 rounded text-sm font-medium text-white text-center ${
                      item.color === 'green' ? 'bg-primary-600' :
                      item.color === 'brown' ? 'bg-yellow-600' :
                      'bg-red-600'
                    }`}
                  >
                    {item.keyword}
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Heatmap Tab */}
        {activeSubTab === 'heatmap' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">🔥 Market Sentiment Heatmap</h3>
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
              {['BTC', 'ETH', 'SOL', 'ADA', 'DOT', 'MATIC', 'AVAX', 'LINK'].map((symbol, index) => {
                const sentiment = Math.random() * 2 - 1 // -1 to 1
                return (
                  <div key={symbol} className="panel-blue rounded-lg border border-primary-500 p-4 text-center">
                    <div className={`w-16 h-16 rounded-full flex items-center justify-center text-white font-bold text-xl mx-auto mb-3 ${
                      sentiment >= 0.6 ? 'bg-primary-600' :
                      sentiment >= 0.2 ? 'bg-primary-500' :
                      sentiment >= -0.2 ? 'bg-yellow-500' :
                      sentiment >= -0.6 ? 'bg-red-500' :
                      'bg-red-600'
                    }`}>
                      {symbol.charAt(0)}
                    </div>
                    <div className="text-white font-semibold mb-1">{symbol}</div>
                    <div className={`text-sm font-semibold ${getSentimentColor(sentiment)}`}>
                      {getSentimentText(sentiment)}
                    </div>
                    <div className="text-xs text-gray-400 mt-1">
                      {(sentiment * 100).toFixed(0)}%
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Events Tab */}
        {activeSubTab === 'events' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">📰 Market Events</h3>
            <div className="space-y-4">
              {marketEvents.map((event) => (
                <div key={event.id} className="panel-blue rounded-lg border border-primary-500 p-6">
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex-1">
                      <h4 className="text-lg font-semibold text-white mb-2">{event.title}</h4>
                      <p className="text-sm text-gray-400 mb-2">{event.description}</p>
                      <div className="text-xs text-gray-500">
                        {event.timestamp.toLocaleString()}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <div className={`px-2 py-1 rounded text-xs font-semibold text-white ${getCategoryColor(event.category)}`}>
                        {event.category.toUpperCase()}
                      </div>
                      <div className={`px-2 py-1 rounded text-xs font-semibold ${getImpactColor(event.impact)}`}>
                        {event.impact.toUpperCase()}
                      </div>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <div className={`text-sm font-semibold ${getSentimentColor(event.sentiment)}`}>
                      Sentiment: {getSentimentText(event.sentiment)}
                    </div>
                    <div className="text-xs text-gray-400">
                      Impact: {event.impact.toUpperCase()}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Trending Tab */}
        {activeSubTab === 'trending' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">📈 Trending Topics</h3>
            <div className="space-y-3">
              {trendingTopics.map((topic) => (
                <div key={topic.id} className="panel-blue rounded-lg border border-primary-500 p-4">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-3">
                      <div className="text-xl">{getPlatformIcon(topic.platform)}</div>
                      <div>
                        <div className="text-white font-semibold">#{topic.topic}</div>
                        <div className="text-xs text-gray-400">
                          {topic.platform} • {topic.category}
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-white font-semibold">{formatNumber(topic.mentions)}</div>
                      <div className={`text-xs ${topic.change >= 0 ? 'text-primary-500' : 'text-red-400'}`}>
                        {formatPercent(topic.change)}
                      </div>
                    </div>
                    <div className="text-right min-w-[100px]">
                      <div className={`text-sm font-semibold ${getSentimentColor(topic.sentiment)}`}>
                        {getSentimentText(topic.sentiment)}
                      </div>
                      <div className="text-xs text-gray-400">
                        {(topic.sentiment * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  // Working Notifications/Alerts (mock data)
  const NotificationsTab = () => {
    const [alerts] = React.useState(
      Array.from({ length: 8 }).map((_, i) => ({
        id: i + 1,
        title: ['Price Cross', 'Funding Spike', 'Open Interest Surge', 'Unusual Options'][i % 4],
        message: 'Condition matched your alert rule',
        time: new Date(Date.now() - i * 1000 * 60 * 12).toLocaleTimeString(),
        level: (['info', 'warning', 'critical'] as const)[i % 3]
      }))
    )

    const levelBadge = (level: 'info' | 'warning' | 'critical') => (
      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
        level === 'critical' ? 'bg-red-900/60 text-red-300' : level === 'warning' ? 'bg-yellow-900/60 text-yellow-200' : 'bg-blue-900/60 text-blue-200'
      }`}>{level}</span>
    )

    return (
      <div className="p-6 h-full overflow-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">🔔 Alerts</h2>
          <p className="text-gray-400">Your recent strategy and market notifications</p>
        </div>
        <div className="panel-blue border border-primary-500 rounded-lg divide-y divide-gray-700">
          {alerts.map(a => (
            <div key={a.id} className="p-4 flex items-center gap-4 hover:bg-primary-500/10/60">
              {levelBadge(a.level)}
              <div className="flex-1">
                <div className="text-white font-semibold">{a.title}</div>
                <div className="text-gray-400 text-sm">{a.message}</div>
              </div>
              <div className="text-gray-500 text-xs">{a.time}</div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  // Referrals Component based on Hyperliquid design
  const ReferralsTab = () => {
    const [referralCode, setReferralCode] = React.useState('QUANTDESK2024')
    const [referralStats, setReferralStats] = React.useState({
      totalReferrals: 0,
      totalEarnings: 0,
      activeReferrals: 0,
      pendingRewards: 0
    })

    const [referralHistory] = React.useState([
      {
        id: '1',
        user: 'crypto_trader_123',
        joinDate: '2024-01-15',
        status: 'active',
        tradingVolume: 125000,
        commissionEarned: 1250
      },
      {
        id: '2', 
        user: 'defi_enthusiast',
        joinDate: '2024-01-10',
        status: 'active',
        tradingVolume: 89000,
        commissionEarned: 890
      },
      {
        id: '3',
        user: 'nft_collector',
        joinDate: '2024-01-05',
        status: 'pending',
        tradingVolume: 0,
        commissionEarned: 0
      }
    ])

    const copyReferralCode = () => {
      navigator.clipboard.writeText(referralCode)
      // You could add a toast notification here
    }

    const formatCurrency = (value: number) => {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
      }).format(value)
    }

    const getStatusColor = (status: string) => {
      switch (status) {
        case 'active': return 'text-primary-500'
        case 'pending': return 'text-yellow-400'
        case 'inactive': return 'text-gray-400'
        default: return 'text-gray-400'
      }
    }

    const getStatusBg = (status: string) => {
      switch (status) {
        case 'active': return 'bg-primary-600'
        case 'pending': return 'bg-yellow-600'
        case 'inactive': return 'bg-gray-600'
        default: return 'bg-gray-600'
      }
    }

    return (
      <div className="p-6 h-full overflow-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">🎯 Referrals Program</h2>
          <p className="text-gray-400">Earn rewards by referring traders to QuantDesk</p>
        </div>

        {/* Referral Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="panel-blue p-4 rounded-lg border border-primary-500">
            <div className="text-sm text-gray-400 mb-1">Total Referrals</div>
            <div className="text-2xl font-bold text-white">{referralStats.totalReferrals}</div>
          </div>
          <div className="panel-blue p-4 rounded-lg border border-primary-500">
            <div className="text-sm text-gray-400 mb-1">Total Earnings</div>
            <div className="text-2xl font-bold price-positive">{formatCurrency(referralStats.totalEarnings)}</div>
          </div>
          <div className="panel-blue p-4 rounded-lg border border-primary-500">
            <div className="text-sm text-gray-400 mb-1">Active Referrals</div>
            <div className="text-2xl font-bold price-positive">{referralStats.activeReferrals}</div>
          </div>
          <div className="panel-blue p-4 rounded-lg border border-primary-500">
            <div className="text-sm text-gray-400 mb-1">Pending Rewards</div>
            <div className="text-2xl font-bold text-yellow-400">{formatCurrency(referralStats.pendingRewards)}</div>
          </div>
        </div>

        {/* Referral Code Section */}
        <div className="panel-blue p-6 mb-6 rounded-lg border border-primary-500">
          <h3 className="text-lg font-semibold text-white mb-4">Your Referral Code</h3>
          <div className="flex items-center gap-4">
            <div className="flex-1 bg-gray-900 rounded-lg p-3 border border-gray-600">
              <code className="text-primary-500 font-mono text-lg">{referralCode}</code>
            </div>
            <button
              onClick={copyReferralCode}
              className="btn btn-primary"
            >
              Copy Code
            </button>
          </div>
          <p className="text-sm text-gray-400 mt-3">
            Share this code with friends to earn 10% of their trading fees as commission
          </p>
        </div>

        {/* Referral Link Section */}
        <div className="panel-blue p-6 mb-6 rounded-lg border border-primary-500">
          <h3 className="text-lg font-semibold text-white mb-4">Referral Link</h3>
          <div className="flex items-center gap-4">
            <div className="flex-1 bg-gray-900 rounded-lg p-3 border border-gray-600">
              <code className="text-gray-300 font-mono text-sm break-all">
                https://quantdesk.xyz/signup?ref={referralCode}
              </code>
            </div>
            <button
              onClick={() => navigator.clipboard.writeText(`https://quantdesk.xyz/signup?ref=${referralCode}`)}
              className="btn btn-primary"
            >
              Copy Link
            </button>
          </div>
        </div>

        {/* Referral History */}
        <div className="panel-blue rounded-lg border border-primary-500">
          <div className="p-6 border-b border-primary-500">
            <h3 className="text-lg font-semibold text-white">Referral History</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-900">
                <tr>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">User</th>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">Join Date</th>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">Status</th>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">Trading Volume</th>
                  <th className="text-left p-4 text-sm font-medium text-gray-400">Commission Earned</th>
                </tr>
              </thead>
              <tbody>
                {referralHistory.map((referral) => (
                  <tr key={referral.id} className="border-b border-primary-500 hover:bg-gray-750">
                    <td className="p-4 text-white font-medium">{referral.user}</td>
                    <td className="p-4 text-gray-300">{referral.joinDate}</td>
                    <td className="p-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusBg(referral.status)} ${getStatusColor(referral.status)}`}>
                        {referral.status}
                      </span>
                    </td>
                    <td className="p-4 text-gray-300">{formatCurrency(referral.tradingVolume)}</td>
                    <td className="p-4 price-positive font-medium">{formatCurrency(referral.commissionEarned)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Program Details */}
        <div className="mt-6 panel-blue rounded-lg p-6 border border-primary-500">
          <h3 className="text-lg font-semibold text-white mb-4">Program Details</h3>
          <div className="space-y-3 text-gray-300">
            <div className="flex items-start gap-3">
              <span className="text-primary-500">•</span>
              <span>Earn 10% commission on all trading fees from your referrals</span>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-primary-500">•</span>
              <span>Commissions are paid out monthly to your wallet</span>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-primary-500">•</span>
              <span>No limit on the number of referrals you can make</span>
            </div>
            <div className="flex items-start gap-3">
              <span className="text-primary-500">•</span>
              <span>Referrals must complete KYC verification to count</span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  // Comprehensive Social/Community Component with multiple sub-tabs
  const SocialTab = () => {
    const [activeSubTab, setActiveSubTab] = React.useState<'trending' | 'traders' | 'alpha' | 'topics'>('trending')
    const [selectedPlatform, setSelectedPlatform] = React.useState('ALL')

    // Trending posts data
    const [trendingPosts] = React.useState([
      {
        id: '1',
        platform: 'twitter',
        author: 'CryptoWhale',
        content: 'Bitcoin breaking through $90K resistance with massive institutional buying. This could be the start of the next leg up! 🚀 #BTC #Bitcoin',
        timestamp: new Date(Date.now() - 300000),
        likes: 1250,
        retweets: 340,
        sentiment: 0.8,
        engagement: 1590,
        verified: true,
        hashtags: ['BTC', 'Bitcoin']
      },
      {
        id: '2',
        platform: 'reddit',
        author: 'r/cryptocurrency',
        content: 'Ethereum Shanghai upgrade completed successfully! Staking withdrawals now enabled. What does this mean for ETH price?',
        timestamp: new Date(Date.now() - 600000),
        likes: 890,
        retweets: 0,
        sentiment: 0.6,
        engagement: 890,
        verified: false,
        hashtags: ['ETH', 'Ethereum', 'Shanghai']
      },
      {
        id: '3',
        platform: 'telegram',
        author: 'DeFiAlpha',
        content: 'New yield farming opportunity on Solana: 45% APY on SOL-USDC pool. Risk: Medium. DYOR! 💎',
        timestamp: new Date(Date.now() - 900000),
        likes: 0,
        retweets: 0,
        sentiment: 0.7,
        engagement: 156,
        verified: true,
        hashtags: ['SOL', 'DeFi', 'YieldFarming']
      },
      {
        id: '4',
        platform: 'discord',
        author: 'TradingGuild',
        content: 'Market analysis: RSI divergence on multiple timeframes suggests potential reversal. Watch for confirmation signals.',
        timestamp: new Date(Date.now() - 1200000),
        likes: 0,
        retweets: 0,
        sentiment: 0.4,
        engagement: 78,
        verified: false,
        hashtags: ['TechnicalAnalysis', 'RSI']
      }
    ])

    // Top traders data
    const [topTraders] = React.useState([
      {
        id: '1',
        name: 'CryptoWhale',
        platform: 'Twitter',
        followers: 125000,
        winRate: 78.5,
        totalReturn: 245.6,
        strategies: ['Lorentzian Classification', 'RSI Divergence', 'Whale Tracking'],
        verified: true,
        lastPost: new Date(Date.now() - 300000),
        sentiment: 0.8
      },
      {
        id: '2',
        name: 'DeFiAlpha',
        platform: 'Telegram',
        followers: 89000,
        winRate: 72.3,
        totalReturn: 189.2,
        strategies: ['DeFi Yield Farming', 'Liquidity Mining', 'Arbitrage'],
        verified: true,
        lastPost: new Date(Date.now() - 900000),
        sentiment: 0.7
      },
      {
        id: '3',
        name: 'TechnicalTrader',
        platform: 'Discord',
        followers: 45000,
        winRate: 68.9,
        totalReturn: 156.8,
        strategies: ['Technical Analysis', 'Pattern Recognition', 'Support/Resistance'],
        verified: false,
        lastPost: new Date(Date.now() - 1800000),
        sentiment: 0.6
      },
      {
        id: '4',
        name: 'CryptoInsider',
        platform: 'Twitter',
        followers: 67000,
        winRate: 75.2,
        totalReturn: 203.4,
        strategies: ['News Trading', 'Event-Driven', 'Fundamental Analysis'],
        verified: true,
        lastPost: new Date(Date.now() - 3600000),
        sentiment: 0.7
      }
    ])

    // Alpha signals data
    const [alphaSignals] = React.useState([
      {
        id: '1',
        title: 'BTC Breakout Signal',
        description: 'Bitcoin showing strong momentum above $90K with institutional accumulation pattern',
        source: 'CryptoWhale',
        confidence: 87,
        category: 'technical',
        symbol: 'BTC',
        timestamp: new Date(Date.now() - 300000),
        price: 90000,
        target: 95000,
        stopLoss: 85000
      },
      {
        id: '2',
        title: 'ETH Staking Unlock Impact',
        description: 'Shanghai upgrade enables staking withdrawals, potential selling pressure but long-term bullish',
        source: 'DeFiAlpha',
        confidence: 72,
        category: 'fundamental',
        symbol: 'ETH',
        timestamp: new Date(Date.now() - 600000),
        price: 3500,
        target: 4000,
        stopLoss: 3200
      },
      {
        id: '3',
        title: 'SOL Ecosystem Growth',
        description: 'Solana showing strong developer activity and user adoption metrics',
        source: 'TechnicalTrader',
        confidence: 65,
        category: 'fundamental',
        symbol: 'SOL',
        timestamp: new Date(Date.now() - 900000),
        price: 240,
        target: 280,
        stopLoss: 220
      },
      {
        id: '4',
        title: 'Whale Accumulation Alert',
        description: 'Large wallet accumulating DOT tokens, potential for significant price movement',
        source: 'CryptoInsider',
        confidence: 80,
        category: 'whale',
        symbol: 'DOT',
        timestamp: new Date(Date.now() - 1200000),
        price: 8.50,
        target: 10.00,
        stopLoss: 7.50
      }
    ])

    // Trending topics data
    const [trendingTopics] = React.useState([
      {
        id: '1',
        topic: 'Bitcoin ETF',
        mentions: 125000,
        sentiment: 0.8,
        change: 15.5,
        category: 'news',
        platform: 'twitter',
        relatedSymbols: ['BTC']
      },
      {
        id: '2',
        topic: 'Ethereum Staking',
        mentions: 89000,
        sentiment: 0.6,
        change: 8.2,
        category: 'technical',
        platform: 'reddit',
        relatedSymbols: ['ETH']
      },
      {
        id: '3',
        topic: 'Solana DeFi',
        mentions: 67000,
        sentiment: 0.9,
        change: 25.3,
        category: 'social',
        platform: 'telegram',
        relatedSymbols: ['SOL', 'RAY', 'ORCA']
      },
      {
        id: '4',
        topic: 'DeFi Regulation',
        mentions: 45000,
        sentiment: 0.4,
        change: -5.1,
        category: 'regulatory',
        platform: 'discord',
        relatedSymbols: ['UNI', 'AAVE', 'COMP']
      }
    ])

    const getSentimentColor = (sentiment: number) => {
      if (sentiment >= 0.6) return 'text-primary-500'
      if (sentiment >= 0.2) return 'text-green-300'
      if (sentiment >= -0.2) return 'text-yellow-400'
      if (sentiment >= -0.6) return 'text-red-300'
      return 'text-red-400'
    }

    const getSentimentText = (sentiment: number) => {
      if (sentiment >= 0.6) return 'Very Bullish'
      if (sentiment >= 0.2) return 'Bullish'
      if (sentiment >= -0.2) return 'Neutral'
      if (sentiment >= -0.6) return 'Bearish'
      return 'Very Bearish'
    }

    const getPlatformIcon = (platform: string) => {
      switch (platform.toLowerCase()) {
        case 'twitter': return '🐦'
        case 'reddit': return '🔴'
        case 'telegram': return '✈️'
        case 'discord': return '💬'
        default: return '📱'
      }
    }

    const getCategoryColor = (category: string) => {
      switch (category) {
        case 'technical': return 'bg-primary-600'
        case 'fundamental': return 'bg-gray-600'
        case 'sentiment': return 'bg-purple-600'
        case 'whale': return 'bg-orange-600'
        case 'news': return 'bg-cyan-600'
        case 'regulatory': return 'bg-red-600'
        default: return 'bg-gray-600'
      }
    }

    const formatNumber = (value: number) => {
      if (value >= 1e6) return (value / 1e6).toFixed(1) + 'M'
      if (value >= 1e3) return (value / 1e3).toFixed(1) + 'K'
      return value.toFixed(0)
    }

    const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`

    const formatCurrency = (value: number) => {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
      }).format(value)
    }

    const filteredPosts = trendingPosts.filter(post => 
      selectedPlatform === 'ALL' || post.platform === selectedPlatform
    )

    return (
      <div className="p-6 h-full overflow-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">📱 Social & Community</h2>
          <p className="text-gray-400">Trending discussions, top traders, and alpha signals</p>
        </div>

        {/* Platform filter */}
        <div className="mb-4">
          <select
            value={selectedPlatform}
            onChange={(e) => setSelectedPlatform(e.target.value)}
            className="panel-blue border border-primary-500 rounded-lg px-3 py-2 text-white text-sm"
          >
            <option value="ALL">All Platforms</option>
            <option value="twitter">Twitter</option>
            <option value="reddit">Reddit</option>
            <option value="telegram">Telegram</option>
            <option value="discord">Discord</option>
          </select>
        </div>

        {/* Sub-navigation tabs */}
        <div className="flex gap-2 mb-6 border-b border-primary-500">
          {[
            { id: 'trending', label: '🔥 Trending Posts' },
            { id: 'traders', label: '🏆 Top Traders' },
            { id: 'alpha', label: '💎 Alpha Signals' },
            { id: 'topics', label: '📈 Trending Topics' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id as any)}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                activeSubTab === tab.id
                  ? 'bg-primary-500 text-white border-b-2 border-primary-500'
                  : 'text-gray-400 hover:text-white hover:bg-primary-500/10'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Trending Posts Tab */}
        {activeSubTab === 'trending' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">🔥 Trending Posts ({filteredPosts.length})</h3>
            <div className="space-y-4">
              {filteredPosts.map((post) => (
                <div key={post.id} className="panel-blue rounded-lg border border-primary-500 p-6">
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex items-center gap-3">
                      <div className="text-xl">{getPlatformIcon(post.platform)}</div>
                      <div>
                        <div className="text-white font-semibold flex items-center gap-2">
                          {post.author}
                          {post.verified && <span className="text-gray-400">✓</span>}
                        </div>
                        <div className="text-xs text-gray-400">
                          {post.timestamp.toLocaleString()}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className={`px-2 py-1 rounded text-xs font-semibold text-white ${
                        post.sentiment >= 0.6 ? 'bg-primary-600' :
                        post.sentiment >= 0.2 ? 'bg-primary-500' :
                        post.sentiment >= -0.2 ? 'bg-yellow-500' :
                        post.sentiment >= -0.6 ? 'bg-red-500' :
                        'bg-red-600'
                      }`}>
                        {getSentimentText(post.sentiment)}
                      </div>
                      <div className="text-xs text-gray-400">
                        {formatNumber(post.engagement)} engagement
                      </div>
                    </div>
                  </div>
                  
                  <div className="text-white mb-3 leading-relaxed">
                    {post.content}
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <div className="flex gap-4 text-sm text-gray-400">
                      <span>❤️ {formatNumber(post.likes)}</span>
                      <span>🔄 {formatNumber(post.retweets)}</span>
                    </div>
                    <div className="flex gap-2">
                      {post.hashtags.map((tag, idx) => (
                        <span key={idx} className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-400">
                          #{tag}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Top Traders Tab */}
        {activeSubTab === 'traders' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">🏆 Top Traders</h3>
            <div className="space-y-4">
              {topTraders.map((trader) => (
                <div key={trader.id} className="panel-blue rounded-lg border border-primary-500 p-6">
                  <div className="flex justify-between items-center mb-4">
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-12 bg-gray-700 rounded-full flex items-center justify-center text-white font-bold text-lg">
                        {trader.name.charAt(0)}
                      </div>
                      <div>
                        <div className="text-white font-semibold flex items-center gap-2">
                          {trader.name}
                          {trader.verified && <span className="text-gray-400">✓</span>}
                        </div>
                        <div className="text-sm text-gray-400">
                          {trader.platform} • {formatNumber(trader.followers)} followers
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-primary-500 font-semibold text-lg">
                        {formatPercent(trader.totalReturn)}
                      </div>
                      <div className="text-xs text-gray-400">Total Return</div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div>
                      <div className="text-xs text-gray-400">Win Rate</div>
                      <div className="text-white font-semibold">
                        {formatPercent(trader.winRate)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Sentiment</div>
                      <div className={`font-semibold ${getSentimentColor(trader.sentiment)}`}>
                        {getSentimentText(trader.sentiment)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Last Post</div>
                      <div className="text-white text-sm">
                        {trader.lastPost.toLocaleString()}
                      </div>
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-xs text-gray-400 mb-2">Strategies</div>
                    <div className="flex gap-2 flex-wrap">
                      {trader.strategies.map((strategy, idx) => (
                        <span key={idx} className="px-2 py-1 bg-gray-700 rounded text-xs text-white">
                          {strategy}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Alpha Signals Tab */}
        {activeSubTab === 'alpha' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">💎 Alpha Signals</h3>
            <div className="space-y-4">
              {alphaSignals.map((signal) => (
                <div key={signal.id} className="panel-blue rounded-lg border border-primary-500 p-6">
                  <div className="flex justify-between items-start mb-3">
                    <div className="flex-1">
                      <h4 className="text-lg font-semibold text-white mb-2">{signal.title}</h4>
                      <p className="text-sm text-gray-400 mb-2">{signal.description}</p>
                      <div className="text-xs text-gray-500">
                        Source: {signal.source} • {signal.timestamp.toLocaleString()}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <div className={`px-2 py-1 rounded text-xs font-semibold text-white ${getCategoryColor(signal.category)}`}>
                        {signal.category.toUpperCase()}
                      </div>
                      <div className={`px-2 py-1 rounded text-xs font-semibold text-white ${
                        signal.confidence >= 80 ? 'bg-primary-600' : 
                        signal.confidence >= 60 ? 'bg-orange-600' : 
                        'bg-red-600'
                      }`}>
                        {signal.confidence}% confidence
                      </div>
                    </div>
                  </div>
                  
                  {signal.symbol && (
                    <div className="grid grid-cols-4 gap-4">
                      <div>
                        <div className="text-xs text-gray-400">Symbol</div>
                        <div className="text-white font-semibold">{signal.symbol}</div>
                      </div>
                      {signal.price && (
                        <div>
                          <div className="text-xs text-gray-400">Current Price</div>
                          <div className="text-white font-semibold">{formatCurrency(signal.price)}</div>
                        </div>
                      )}
                      {signal.target && (
                        <div>
                          <div className="text-xs text-gray-400">Target</div>
                          <div className="text-primary-500 font-semibold">{formatCurrency(signal.target)}</div>
                        </div>
                      )}
                      {signal.stopLoss && (
                        <div>
                          <div className="text-xs text-gray-400">Stop Loss</div>
                          <div className="text-red-400 font-semibold">{formatCurrency(signal.stopLoss)}</div>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Trending Topics Tab */}
        {activeSubTab === 'topics' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">📈 Trending Topics</h3>
            <div className="space-y-3">
              {trendingTopics.map((topic) => (
                <div key={topic.id} className="panel-blue rounded-lg border border-primary-500 p-4">
                  <div className="flex justify-between items-center">
                    <div className="flex items-center gap-3">
                      <div className="text-xl">{getPlatformIcon(topic.platform)}</div>
                      <div>
                        <div className="text-white font-semibold">#{topic.topic}</div>
                        <div className="text-xs text-gray-400">
                          {topic.platform} • {topic.category}
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="text-white font-semibold">{formatNumber(topic.mentions)}</div>
                        <div className="text-xs text-gray-400">mentions</div>
                      </div>
                      <div className="text-right">
                        <div className={`font-semibold ${topic.change >= 0 ? 'text-primary-500' : 'text-red-400'}`}>
                          {formatPercent(topic.change)}
                        </div>
                        <div className="text-xs text-gray-400">change</div>
                      </div>
                      <div className="text-right">
                        <div className={`font-semibold ${getSentimentColor(topic.sentiment)}`}>
                          {getSentimentText(topic.sentiment)}
                        </div>
                        <div className="text-xs text-gray-400">sentiment</div>
                      </div>
                    </div>
                  </div>
                  
                  {topic.relatedSymbols.length > 0 && (
                    <div className="mt-3">
                      <div className="text-xs text-gray-400 mb-2">Related Symbols</div>
                      <div className="flex gap-2">
                        {topic.relatedSymbols.map((symbol, idx) => (
                          <span key={idx} className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-400">
                            {symbol}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    )
  }

  // Comprehensive Strategies Component with multiple sub-tabs
  const StrategiesTab = () => {
    const [activeSubTab, setActiveSubTab] = React.useState<'active' | 'templates' | 'deployment' | 'create'>('active')

    // Active strategies data
    const [strategies] = React.useState([
      {
        id: '1',
        name: 'Lorentzian Classifier',
        description: 'Machine learning strategy using Lorentzian distance classification for market prediction',
        status: 'active',
        type: 'ml',
        performance: {
          totalReturn: 45.6,
          sharpeRatio: 1.85,
          maxDrawdown: -8.2,
          winRate: 68.5,
          totalTrades: 1247
        },
        parameters: {
          lookback: 8,
          threshold: 0.5,
          enableLong: true,
          enableShort: false
        },
        lastUpdate: new Date(Date.now() - 3600000),
        symbols: ['BTC', 'ETH', 'SOL'],
        riskLevel: 'medium',
        deployment: {
          isDeployed: true,
          deployedAt: new Date(Date.now() - 3600000),
          deployedBy: 'd3x7',
          deploymentId: 'deploy_001',
          lastSignal: new Date(Date.now() - 300000),
          nextSignal: new Date(Date.now() + 300000),
          signalCount: 47,
          errorCount: 2,
          lastError: 'Connection timeout (resolved)'
        },
        realTimeStats: {
          currentPnL: 125.50,
          todayTrades: 8,
          lastSignal: 'BUY signal generated at 14:32:15',
          nextUpdate: new Date(Date.now() + 30000),
          isRunning: true
        }
      },
      {
        id: '2',
        name: 'RSI Divergence',
        description: 'Technical strategy based on RSI divergence patterns and support/resistance levels',
        status: 'active',
        type: 'technical',
        performance: {
          totalReturn: 32.1,
          sharpeRatio: 1.42,
          maxDrawdown: -12.5,
          winRate: 61.2,
          totalTrades: 892
        },
        parameters: {
          rsiPeriod: 14,
          overbought: 70,
          oversold: 30,
          divergenceLookback: 5
        },
        lastUpdate: new Date(Date.now() - 7200000),
        symbols: ['BTC', 'ETH'],
        riskLevel: 'high',
        deployment: {
          isDeployed: false,
          signalCount: 0,
          errorCount: 0
        },
        realTimeStats: {
          currentPnL: 0,
          todayTrades: 0,
          lastSignal: 'None',
          nextUpdate: new Date(Date.now() + 60000),
          isRunning: false
        }
      },
      {
        id: '3',
        name: 'Moving Average Crossover',
        description: 'Basic strategy that generates signals when short MA crosses above/below long MA',
        status: 'paused',
        type: 'technical',
        performance: {
          totalReturn: 18.7,
          sharpeRatio: 0.95,
          maxDrawdown: -15.8,
          winRate: 55.3,
          totalTrades: 2156
        },
        parameters: {
          shortPeriod: 10,
          longPeriod: 20,
          enableLong: true,
          enableShort: true
        },
        lastUpdate: new Date(Date.now() - 86400000),
        symbols: ['SOL', 'ADA'],
        riskLevel: 'medium'
      },
      {
        id: '4',
        name: 'Bollinger Bands Mean Reversion',
        description: 'Mean reversion strategy using Bollinger Bands for entry and exit signals',
        status: 'inactive',
        type: 'mean_reversion',
        performance: {
          totalReturn: 28.9,
          sharpeRatio: 1.23,
          maxDrawdown: -18.2,
          winRate: 58.7,
          totalTrades: 634
        },
        parameters: {
          period: 20,
          stdDev: 2.0,
          overboughtLevel: 0.8,
          oversoldLevel: 0.2
        },
        lastUpdate: new Date(Date.now() - 172800000),
        symbols: ['BTC', 'ETH', 'DOT'],
        riskLevel: 'high'
      }
    ])

    // Strategy templates data
    const [templates] = React.useState([
      {
        id: '1',
        name: 'Simple Moving Average Crossover',
        description: 'Basic strategy that generates signals when short MA crosses above/below long MA',
        type: 'technical',
        category: 'Trend Following',
        difficulty: 'beginner',
        parameters: [
          { name: 'shortPeriod', type: 'number', default: 10, min: 5, max: 50, description: 'Short moving average period' },
          { name: 'longPeriod', type: 'number', default: 20, min: 10, max: 200, description: 'Long moving average period' },
          { name: 'enableLong', type: 'boolean', default: true, description: 'Enable long positions' },
          { name: 'enableShort', type: 'boolean', default: true, description: 'Enable short positions' }
        ]
      },
      {
        id: '2',
        name: 'Bollinger Bands Mean Reversion',
        description: 'Mean reversion strategy using Bollinger Bands for entry and exit signals',
        type: 'technical',
        category: 'Mean Reversion',
        difficulty: 'intermediate',
        parameters: [
          { name: 'period', type: 'number', default: 20, min: 10, max: 50, description: 'Bollinger Bands period' },
          { name: 'stdDev', type: 'number', default: 2.0, min: 1.0, max: 3.0, description: 'Standard deviation multiplier' },
          { name: 'overboughtLevel', type: 'number', default: 0.8, min: 0.5, max: 1.0, description: 'Overbought threshold' },
          { name: 'oversoldLevel', type: 'number', default: 0.2, min: 0.0, max: 0.5, description: 'Oversold threshold' }
        ]
      },
      {
        id: '3',
        name: 'Lorentzian Classification',
        description: 'Advanced ML strategy using Lorentzian distance for market classification',
        type: 'ml',
        category: 'Machine Learning',
        difficulty: 'advanced',
        parameters: [
          { name: 'lookback', type: 'number', default: 8, min: 5, max: 20, description: 'Lookback period for features' },
          { name: 'threshold', type: 'number', default: 0.5, min: 0.1, max: 0.9, description: 'Classification threshold' },
          { name: 'enableLong', type: 'boolean', default: true, description: 'Enable long positions' },
          { name: 'enableShort', type: 'boolean', default: false, description: 'Enable short positions' }
        ]
      }
    ])

    const getStatusColor = (status: string) => {
      switch (status) {
        case 'active': return 'bg-primary-600'
        case 'paused': return 'bg-yellow-600'
        case 'inactive': return 'bg-gray-600'
        case 'error': return 'bg-red-600'
        default: return 'bg-gray-600'
      }
    }

    const getRiskColor = (risk: string) => {
      switch (risk) {
        case 'low': return 'bg-primary-600'
        case 'medium': return 'bg-yellow-600'
        case 'high': return 'bg-red-600'
        default: return 'bg-gray-600'
      }
    }

    const getTypeIcon = (type: string) => {
      switch (type) {
        case 'ml': return '🧠'
        case 'technical': return '📊'
        case 'momentum': return '🚀'
        case 'mean_reversion': return '🔄'
        default: return '❓'
      }
    }

    const getDifficultyColor = (difficulty: string) => {
      switch (difficulty) {
        case 'beginner': return 'bg-primary-600'
        case 'intermediate': return 'bg-yellow-600'
        case 'advanced': return 'bg-red-600'
        default: return 'bg-gray-600'
      }
    }

    const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`

    const getDeploymentStatus = (strategy: any) => {
      if (strategy.deployment?.isDeployed) {
        return strategy.status === 'active' ? '🟢 Running' : 
               strategy.status === 'paused' ? '🟡 Paused' :
               '🔴 Error'
      }
      return '⚪ Not Deployed'
    }

    return (
      <div className="p-6 h-full overflow-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">🧠 Strategy Manager</h2>
          <p className="text-gray-400">Manage your trading strategies and create new ones</p>
        </div>

        {/* Stats Overview */}
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="panel-blue rounded-lg border border-primary-500 p-4">
            <div className="text-sm text-gray-400 mb-1">Active Strategies</div>
            <div className="text-2xl font-bold text-primary-500">
              {strategies.filter(s => s.status === 'active').length}
            </div>
          </div>
          <div className="panel-blue rounded-lg border border-primary-500 p-4">
            <div className="text-sm text-gray-400 mb-1">Deployed</div>
            <div className="text-2xl font-bold text-gray-400">
              {strategies.filter(s => s.deployment?.isDeployed).length}
            </div>
          </div>
          <div className="panel-blue rounded-lg border border-primary-500 p-4">
            <div className="text-sm text-gray-400 mb-1">Total Signals</div>
            <div className="text-2xl font-bold text-purple-400">
              {strategies.reduce((sum, s) => sum + (s.deployment?.signalCount || 0), 0)}
            </div>
          </div>
          <div className="panel-blue rounded-lg border border-primary-500 p-4">
            <div className="text-sm text-gray-400 mb-1">Avg Return</div>
            <div className="text-2xl font-bold text-primary-500">
              {formatPercent(strategies.reduce((sum, s) => sum + s.performance.totalReturn, 0) / strategies.length)}
            </div>
          </div>
        </div>

        {/* Sub-navigation tabs */}
        <div className="flex gap-2 mb-6 border-b border-primary-500">
          {[
            { id: 'active', label: '📊 Active Strategies' },
            { id: 'templates', label: '📋 Templates' },
            { id: 'deployment', label: '🚀 Deployment' },
            { id: 'create', label: '➕ Create New' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveSubTab(tab.id as any)}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                activeSubTab === tab.id
                  ? 'bg-primary-500 text-white border-b-2 border-primary-500'
                  : 'text-gray-400 hover:text-white hover:bg-primary-500/10'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Active Strategies Tab */}
        {activeSubTab === 'active' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">📊 Your Strategies ({strategies.length})</h3>
            <div className="space-y-4">
              {strategies.map((strategy) => (
                <div key={strategy.id} className="panel-blue rounded-lg border border-primary-500 p-6">
                  <div className="flex justify-between items-start mb-4">
                    <div className="flex items-center gap-3">
                      <div className="text-2xl">{getTypeIcon(strategy.type)}</div>
                      <div>
                        <div className="text-white font-semibold text-lg">{strategy.name}</div>
                        <div className="text-sm text-gray-400">{strategy.description}</div>
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <div className={`px-2 py-1 rounded text-xs font-semibold text-white ${getStatusColor(strategy.status)}`}>
                        {strategy.status.toUpperCase()}
                      </div>
                      <div className={`px-2 py-1 rounded text-xs font-semibold text-white ${getRiskColor(strategy.riskLevel)}`}>
                        {strategy.riskLevel.toUpperCase()}
                      </div>
                      <div className={`px-2 py-1 rounded text-xs font-semibold text-white ${
                        strategy.deployment?.isDeployed ? 'bg-primary-600' : 'bg-gray-600'
                      }`}>
                        {getDeploymentStatus(strategy)}
                      </div>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-6 gap-4 mb-4">
                    <div>
                      <div className="text-xs text-gray-400">Total Return</div>
                      <div className={`font-semibold ${strategy.performance.totalReturn >= 0 ? 'text-primary-500' : 'text-red-400'}`}>
                        {formatPercent(strategy.performance.totalReturn)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Sharpe Ratio</div>
                      <div className="text-white font-semibold">
                        {strategy.performance.sharpeRatio.toFixed(2)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Max Drawdown</div>
                      <div className="text-red-400 font-semibold">
                        {formatPercent(strategy.performance.maxDrawdown)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Win Rate</div>
                      <div className="text-white font-semibold">
                        {formatPercent(strategy.performance.winRate)}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Total Trades</div>
                      <div className="text-white font-semibold">
                        {strategy.performance.totalTrades}
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400">Last Update</div>
                      <div className="text-white text-sm">
                        {strategy.lastUpdate.toLocaleString()}
                      </div>
                    </div>
                  </div>

                  {/* Real-time Stats */}
                  {strategy.deployment?.isDeployed && strategy.realTimeStats && (
                    <div className="bg-gray-900 rounded-lg border border-gray-600 p-4 mb-4">
                      <div className="text-xs text-gray-400 mb-2">Real-time Stats</div>
                      <div className="grid grid-cols-4 gap-4">
                        <div>
                          <div className="text-xs text-gray-400">Current P&L</div>
                          <div className={`font-semibold ${strategy.realTimeStats.currentPnL >= 0 ? 'text-primary-500' : 'text-red-400'}`}>
                            {strategy.realTimeStats.currentPnL >= 0 ? '+' : ''}${strategy.realTimeStats.currentPnL.toFixed(2)}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Today's Trades</div>
                          <div className="text-white font-semibold">
                            {strategy.realTimeStats.todayTrades}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Last Signal</div>
                          <div className="text-gray-300 text-sm">
                            {strategy.realTimeStats.lastSignal}
                          </div>
                        </div>
                        <div>
                          <div className="text-xs text-gray-400">Next Update</div>
                          <div className="text-gray-300 text-sm">
                            {strategy.realTimeStats.nextUpdate.toLocaleTimeString()}
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  <div className="flex justify-between items-center">
                    <div>
                      <div className="text-xs text-gray-400 mb-2">Symbols</div>
                      <div className="flex gap-2">
                        {strategy.symbols.map((symbol, idx) => (
                          <span key={idx} className="px-2 py-1 bg-gray-700 rounded text-xs text-gray-400">
                            {symbol}
                          </span>
                        ))}
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <button className="px-3 py-1 bg-primary-600 rounded text-xs text-white hover:bg-primary-700">
                        {strategy.status === 'active' ? 'Pause' : 'Start'}
                      </button>
                      <button className="px-3 py-1 bg-gray-600 rounded text-xs text-white hover:bg-primary-500/10">
                        Configure
                      </button>
                      <button className="px-3 py-1 bg-purple-600 rounded text-xs text-white hover:bg-purple-700">
                        Backtest
                      </button>
                      <button className="px-3 py-1 bg-red-600 rounded text-xs text-white hover:bg-red-700">
                        Delete
                      </button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Templates Tab */}
        {activeSubTab === 'templates' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">📋 Strategy Templates</h3>
            <div className="space-y-4">
              {templates.map((template) => (
                <div key={template.id} className="panel-blue rounded-lg border border-primary-500 p-6">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <div className="text-white font-semibold text-lg">{template.name}</div>
                      <div className="text-sm text-gray-400 mb-2">{template.description}</div>
                      <div className="text-xs text-gray-500">
                        {template.category} • {template.type.toUpperCase()}
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className={`px-2 py-1 rounded text-xs font-semibold text-white ${getDifficultyColor(template.difficulty)}`}>
                        {template.difficulty.toUpperCase()}
                      </div>
                      <button className="px-3 py-1 bg-gray-600 rounded text-xs text-white hover:bg-primary-500/10">
                        Use Template
                      </button>
                    </div>
                  </div>
                  
                  <div>
                    <div className="text-xs text-gray-400 mb-2">Parameters</div>
                    <div className="grid grid-cols-2 gap-3">
                      {template.parameters.map((param, idx) => (
                        <div key={idx} className="bg-gray-900 rounded p-3">
                          <div className="text-white font-semibold text-sm">{param.name}</div>
                          <div className="text-gray-400 text-xs">{param.description}</div>
                          <div className="text-gray-500 text-xs">Default: {param.default}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Deployment Tab */}
        {activeSubTab === 'deployment' && (
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">🚀 Deployment Center</h3>
            
            {/* Deployment Overview */}
            <div className="panel-blue rounded-lg border border-primary-500 p-6 mb-6">
              <div className="grid grid-cols-4 gap-6">
                <div>
                  <div className="text-sm text-gray-400 mb-1">Deployed Strategies</div>
                  <div className="text-2xl font-bold text-primary-500">
                    {strategies.filter(s => s.deployment?.isDeployed).length}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">Active Signals</div>
                  <div className="text-2xl font-bold text-gray-400">
                    {strategies.reduce((sum, s) => sum + (s.deployment?.signalCount || 0), 0)}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">Total Errors</div>
                  <div className="text-2xl font-bold text-red-400">
                    {strategies.reduce((sum, s) => sum + (s.deployment?.errorCount || 0), 0)}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-gray-400 mb-1">System Status</div>
                  <div className="text-lg font-bold text-primary-500">
                    🟢 All Systems Operational
                  </div>
                </div>
              </div>
            </div>

            {/* Deployment Logs */}
            <div className="panel-blue rounded-lg border border-primary-500 p-6">
              <div className="flex justify-between items-center mb-4">
                <h4 className="text-white font-semibold">Deployment Logs</h4>
                <button className="px-3 py-1 bg-gray-700 rounded text-xs text-white hover:bg-gray-600">
                  Clear Logs
                </button>
              </div>
              <div className="bg-gray-900 rounded p-4 h-48 overflow-auto font-mono text-sm text-gray-300">
                <div>[14:32:15] Starting deployment for strategy Lorentzian Classifier...</div>
                <div>[14:32:18] ✅ Strategy deployed successfully!</div>
                <div>[14:32:18] 🚀 Strategy is now ACTIVE and monitoring markets...</div>
                <div>[14:35:22] BUY signal generated for BTC at $90,250</div>
                <div>[14:38:45] SELL signal generated for ETH at $3,520</div>
                <div>[14:42:10] Connection timeout resolved - strategy resumed</div>
                <div className="text-gray-500 italic">No deployment logs yet...</div>
              </div>
            </div>
          </div>
        )}

        {/* Create Tab */}
        {activeSubTab === 'create' && (
          <div className="text-center py-16">
            <div className="text-6xl mb-6">➕</div>
            <div className="text-2xl font-semibold text-white mb-3">Create New Strategy</div>
            <div className="text-gray-400 mb-8">Use templates or create custom strategies from scratch</div>
            <div className="flex gap-4 justify-center">
              <button className="px-6 py-3 bg-gray-600 rounded-lg text-white hover:bg-primary-500/10">
                Use Template
              </button>
              <button className="px-6 py-3 bg-primary-600 rounded-lg text-white hover:bg-primary-700">
                Create Custom
              </button>
            </div>
          </div>
        )}
      </div>
    )
  }

  // Working Settings (mock)
  const SettingsTab = () => {
    const [autoConnect, setAutoConnect] = React.useState(false)
    const [theme, setTheme] = React.useState('dark')
    return (
      <div className="p-6 h-full overflow-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">⚙️ Settings</h2>
          <p className="text-gray-400">Preferences and integrations</p>
        </div>
        <div className="panel-blue border border-primary-500 rounded-lg p-6 space-y-4 max-w-xl">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-white font-semibold">Auto-connect Wallet</div>
              <div className="text-gray-400 text-sm">Connect on page load</div>
            </div>
            <input type="checkbox" checked={autoConnect} onChange={() => setAutoConnect(v => !v)} />
          </div>
          <div>
            <div className="text-white font-semibold mb-2">Theme</div>
            <select value={theme} onChange={e => setTheme(e.target.value)} className="bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white">
              <option value="dark">Dark</option>
              <option value="light">Light</option>
            </select>
          </div>
        </div>
      </div>
    )
  }

  // Working Analysis (mock)
  const AnalysisTab = () => {
    return (
      <div className="p-6 h-full overflow-auto">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white mb-2">🔍 Analysis</h2>
          <p className="text-gray-400">Alpha, anomalies, and correlations</p>
        </div>
        <div className="grid md:grid-cols-2 gap-6">
          <div className="panel-blue border border-primary-500 rounded-lg p-6">
            <div className="text-white font-semibold mb-3">Cross-asset Correlation</div>
            <div className="h-48 bg-gray-900/50 rounded flex items-center justify-center text-gray-500">Correlation Heatmap</div>
          </div>
          <div className="panel-blue border border-primary-500 rounded-lg p-6">
            <div className="text-white font-semibold mb-3">Alpha Factors</div>
            <ul className="text-gray-300 text-sm space-y-2 list-disc pl-5">
              <li>Funding skew reversal</li>
              <li>OI + Volume divergence</li>
              <li>On-chain netflow anomaly</li>
            </ul>
          </div>
        </div>
      </div>
    )
  }

  const renderPlaceholder = (title: string, icon: string) => (
    <div className="p-6 text-center">
      <div className="text-6xl mb-4">{icon}</div>
      <h2 className="text-2xl font-bold text-white mb-4">{title}</h2>
      <p className="text-gray-400">This feature is coming soon...</p>
    </div>
  )

  const renderOverview = () => <DashboardOverview />

  const views: Record<string, React.ReactNode> = {
    overview: renderOverview(),
    market: <MarketTab />,
    charts: <ChartsTab />,
    trading: <TradingTab />,
    backtesting: <BacktestingTab />,
    whales: <WhalesTab />,
    sentiment: <SentimentTab />,
    notifications: <NotificationsTab />,
    social: <SocialTab />,
    referrals: <ReferralsTab />,
    strategies: <StrategiesTab />,
    portfolio: <PortfolioTab />,
    settings: <SettingsTab />,
    analysis: <AnalysisTab />,
  }

  // Debug display (commented out for production)
  // console.log(`🎯 LiteRouter render - activeTab: ${activeTab}`)

  return (
    <div className="flex-1 flex flex-col bg-black text-white">
      {/* Debug Info (commented out for production) */}
      {/* {import.meta.env.DEV && (
        <div style={{
          position: 'fixed',
          top: '80px',
          left: '10px',
          background: 'rgba(0,0,0,0.8)',
          color: 'white',
          padding: '8px',
          fontSize: '12px',
          fontFamily: 'monospace',
          borderRadius: '4px',
          zIndex: 9999
        }}>
          LiteRouter Tab: {activeTab}
        </div>
      )} */}
      {/* Content only; navigation lives in global Header. */}
      <div className="flex-1 overflow-auto bg-black">{views[activeTab] || views['overview']}</div>
    </div>
  )
}

export default LiteRouter


