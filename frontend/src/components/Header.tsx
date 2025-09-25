import React from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Settings, ChevronDown, Globe } from 'lucide-react'
import { useTabContext } from '../contexts/TabContext'
import WalletButton from './WalletButton'

// Market interface (for future use)
// interface Market {
//   symbol: string
//   name: string
//   price: number
//   change24h: number
//   volume24h: number
// }

// Market data (for future use)
// const markets: Market[] = [
//   { symbol: 'BTC/USDT', name: 'Bitcoin', price: 43250.50, change24h: 2.45, volume24h: 2.94 },
//   { symbol: 'ETH/USDT', name: 'Ethereum', price: 3192.30, change24h: -1.23, volume24h: 1.87 },
//   { symbol: 'SOL/USDT', name: 'Solana', price: 220.65, change24h: -7.05, volume24h: 0.89 },
//   { symbol: 'BNB/USDT', name: 'BNB', price: 993.10, change24h: -5.40, volume24h: 0.45 },
// ]

const Header: React.FC = React.memo(() => {
  const navigate = useNavigate()
  const { activeTab: activeLiteTab, goToTab } = useTabContext()

  const goLiteTab = (tabId: string) => {
    console.log(`🖱️ Header tab clicked: ${tabId}`)
    console.log(`📊 Current activeLiteTab: ${activeLiteTab}`)
    console.log(`🔄 Setting activeLiteTab to: ${tabId}`)
    
    goToTab(tabId)
    // also reflect in URL hash for deep-linking
    navigate(`/lite#${tabId}`)
    
    console.log(`✅ Navigation complete: /lite#${tabId}`)
  }
  // Market selection state (for future use)
  // const [selectedMarket, setSelectedMarket] = useState<Market>(markets[0])
  // const [showMarketDropdown, setShowMarketDropdown] = useState(false)

  // Debug display
  console.log(`🎯 Header render - activeLiteTab: ${activeLiteTab}`)

  return (
    <>
      {/* Debug Info */}
      {process.env.NODE_ENV === 'development' && (
        <div style={{
          position: 'fixed',
          top: '50px',
          left: '10px',
          background: 'rgba(0,0,0,0.8)',
          color: 'white',
          padding: '8px',
          fontSize: '12px',
          fontFamily: 'monospace',
          borderRadius: '4px',
          zIndex: 9999
        }}>
          Active Tab: {activeLiteTab}
        </div>
      )}
      <header className="bg-black border-b border-gray-800 px-6 py-3">
      <div className="flex items-center justify-between">
        {/* Logo and Navigation */}
        <div className="flex items-center space-x-8">
          <Link to="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-sm">Q</span>
            </div>
            <span className="text-xl font-bold text-white">QuantDesk</span>
          </Link>
          
          <nav className="hidden md:flex space-x-2">
            <button 
              onClick={() => goLiteTab('trading')} 
              className={`px-3 py-1.5 text-xs rounded border ${
                activeLiteTab === 'trading' ? 'bg-green-600/90 text-white border-green-500' : 'bg-gray-900 text-gray-300 border-gray-700 hover:text-white'
              }`}
              aria-label="Trade"
            >
              Trade
            </button>
            <button 
              onClick={() => goLiteTab('portfolio')} 
              className={`px-3 py-1.5 text-xs rounded border ${
                activeLiteTab === 'portfolio' ? 'bg-green-600/90 text-white border-green-500' : 'bg-gray-900 text-gray-300 border-gray-700 hover:text-white'
              }`}
              aria-label="Portfolio"
            >
              Portfolio
            </button>
            <button 
              onClick={() => goLiteTab('market')} 
              className={`px-3 py-1.5 text-xs rounded border ${
                activeLiteTab === 'market' ? 'bg-green-600/90 text-white border-green-500' : 'bg-gray-900 text-gray-300 border-gray-700 hover:text-white'
              }`}
              aria-label="Markets"
            >
              Markets
            </button>
            <button 
              onClick={() => goLiteTab('strategies')} 
              className={`px-3 py-1.5 text-xs rounded border ${
                activeLiteTab === 'strategies' ? 'bg-green-600/90 text-white border-green-500' : 'bg-gray-900 text-gray-300 border-gray-700 hover:text-white'
              }`}
              aria-label="Vaults"
            >
              Vaults
            </button>
            <button 
              onClick={() => goLiteTab('backtesting')} 
              className={`px-3 py-1.5 text-xs rounded border ${
                activeLiteTab === 'backtesting' ? 'bg-green-600/90 text-white border-green-500' : 'bg-gray-900 text-gray-300 border-gray-700 hover:text-white'
              }`}
              aria-label="Staking"
            >
              Staking
            </button>
            <button 
              onClick={() => goLiteTab('social')} 
              className={`px-3 py-1.5 text-xs rounded border ${
                activeLiteTab === 'social' ? 'bg-green-600/90 text-white border-green-500' : 'bg-gray-900 text-gray-300 border-gray-700 hover:text-white'
              }`}
              aria-label="Referrals"
            >
              Referrals
            </button>
            <button 
              onClick={() => goLiteTab('analysis')} 
              className={`px-3 py-1.5 text-xs rounded border ${
                activeLiteTab === 'analysis' ? 'bg-green-600/90 text-white border-green-500' : 'bg-gray-900 text-gray-300 border-gray-700 hover:text-white'
              }`}
              aria-label="Leaderboard"
            >
              Leaderboard
            </button>
            <div className="relative group">
              <div className="flex items-center text-gray-400 group-hover:text-white transition-colors text-sm font-medium cursor-pointer">
                More
                <ChevronDown className="h-3 w-3 ml-1" />
              </div>
              <div className="absolute left-0 mt-2 hidden group-hover:block bg-gray-900 border border-gray-800 rounded-lg shadow-lg z-20 min-w-[200px]">
                {[
                  { id: 'overview', label: 'Overview' },
                  { id: 'backtesting', label: 'Backtest' },
                  { id: 'whales', label: 'Whales' },
                  { id: 'sentiment', label: 'Sentiment' },
                  { id: 'notifications', label: 'Alerts' },
                  { id: 'strategies', label: 'Strategies' },
                  { id: 'social', label: 'Social' },
                  { id: 'settings', label: 'Settings' },
                ].map(item => (
                  <button
                    key={item.id}
                    onClick={() => goLiteTab(item.id)}
                    className="w-full text-left px-4 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-800"
                  >
                    {item.label}
                  </button>
                ))}
              </div>
            </div>
          </nav>
        </div>

        {/* Right side actions */}
        <div className="flex items-center space-x-4">
          <WalletButton />
          <button className="p-2 text-gray-400 hover:text-white transition-colors" aria-label="Language" title="Language">
            <Globe className="h-4 w-4" />
          </button>
          <button className="p-2 text-gray-400 hover:text-white transition-colors" aria-label="Settings" title="Settings">
            <Settings className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* primary nav now controls Lite tabs; secondary tabs removed */}
    </header>
    </>
  )
})

export default Header
