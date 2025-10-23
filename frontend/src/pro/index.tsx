import React from 'react'
import { useNavigate } from 'react-router-dom'
import { ThemeProvider, useTheme } from '../contexts/ThemeContext'
import { usePrice } from '../contexts/PriceContext'
import ThemeToggle from '../components/ThemeToggle'
import ProTerminalSettings from '../components/ProTerminalSettings'
import WalletButton from '../components/WalletButton'
import ChatWindow from '../components/ChatWindow'
import MessageInput from '../components/MessageInput'

// Component to set Pro theme
const ProThemeSetter: React.FC = () => {
  const { setTheme } = useTheme()
  
  React.useEffect(() => {
    setTheme('pro')
  }, [setTheme])
  
  return null
}

// Enhanced Pro Terminal with Mode Selector in Taskbar
const ProTerminalWithTaskbar: React.FC = () => {
  const [command, setCommand] = React.useState('')
  const [windows, setWindows] = React.useState<any[]>([])
  const [showBacktickMenu, setShowBacktickMenu] = React.useState(false)
  const [nextZIndex, setNextZIndex] = React.useState(100)
  const [commandHistory, setCommandHistory] = React.useState<string[]>([])
  const [historyIndex, setHistoryIndex] = React.useState(-1)
  const [selectedCommandIndex, setSelectedCommandIndex] = React.useState(0)
  const [terminalActive, setTerminalActive] = React.useState(false)
  const [marketData, setMarketData] = React.useState<Record<string, any>>({})
  const [dragData, setDragData] = React.useState<{ windowId: string; startX: number; startY: number; startMouseX: number; startMouseY: number } | null>(null)
  const [resizeData, setResizeData] = React.useState<{ windowId: string; startWidth: number; startHeight: number; startMouseX: number; startMouseY: number } | null>(null)
  const [showSettings, setShowSettings] = React.useState(false)
  const [currentProfile, setCurrentProfile] = React.useState<any>(null)
  const [filteredCommands, setFilteredCommands] = React.useState<any[]>([])
  const [filteredInstruments, setFilteredInstruments] = React.useState<any[]>([])
  const [filteredNews, setFilteredNews] = React.useState<any[]>([])
  const [showSearchResults, setShowSearchResults] = React.useState(false)
  const [inputFocused, setInputFocused] = React.useState(false)
  const [mode, setMode] = React.useState<'lite' | 'pro'>('pro')

  const commandInputRef = React.useRef<HTMLInputElement>(null)
  const navigate = useNavigate()
  
  // Get real-time prices from PriceContext
  const { getPrice } = usePrice()

  // Get username for terminal prompt (Linux-style)
  const getUsername = () => {
    // Try to get username from various sources
    const userAgent = navigator.userAgent
    const platform = navigator.platform
    
    // Check if we're on Linux
    if (platform.includes('Linux')) {
      // Try to extract username from user agent or use common Linux usernames
      const commonUsernames = ['d3x7', 'user', 'admin', 'ubuntu', 'debian', 'root']
      // For demo purposes, we'll use 'd3x7' as requested
      return 'd3x7'
    }
    
    // Fallback for other platforms
    return 'user'
  }

  // Crypto trading pairs and tokens data - using real Pyth prices
  const instruments = React.useMemo(() => {
    const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 'MATIC/USDT', 'AVAX/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT']
    const names = ['Bitcoin', 'Ethereum', 'Solana', 'Cardano', 'Polkadot', 'Polygon', 'Avalanche', 'Chainlink', 'Uniswap', 'Cosmos']
    const types = ['SPOT', 'SPOT', 'SPOT', 'SPOT', 'SPOT', 'SPOT', 'SPOT', 'SPOT', 'SPOT', 'SPOT']
    const volumes = ['2.4B', '1.8B', '890M', '456M', '234M', '345M', '567M', '678M', '123M', '89M']
    
    return symbols.map((symbol, index) => {
      const priceData = getPrice(symbol)
      return {
        symbol,
        name: names[index],
        price: priceData?.price || 0,
        change: priceData?.change || 0,
        changePercent: priceData?.changePercent || 0,
        type: types[index],
        volume: volumes[index]
      }
    })
  }, [getPrice])

  // Crypto news data
  const newsStories = [
    { headline: 'Bitcoin Surges Past $67,000 as Institutional Adoption Grows', ticker: 'BTC', time: '2 hours ago', source: 'CoinDesk' },
    { headline: 'Ethereum 2.0 Upgrade Shows Promising Results', ticker: 'ETH', time: '4 hours ago', source: 'The Block' },
    { headline: 'Solana Network Activity Hits New Highs', ticker: 'SOL', time: '6 hours ago', source: 'CoinTelegraph' },
    { headline: 'DeFi Total Value Locked Reaches $200B', ticker: 'DEFI', time: '8 hours ago', source: 'DeFi Pulse' },
    { headline: 'Cardano Smart Contracts Launch Successfully', ticker: 'ADA', time: '10 hours ago', source: 'Crypto News' },
    { headline: 'Polygon Announces Major Partnership', ticker: 'MATIC', time: '12 hours ago', source: 'CoinDesk' },
    { headline: 'Avalanche Ecosystem Expands with New Projects', ticker: 'AVAX', time: '1 day ago', source: 'The Block' },
    { headline: 'Chainlink Oracle Network Reaches New Milestone', ticker: 'LINK', time: '1 day ago', source: 'CoinTelegraph' }
  ]

  // Search function (Godel Terminal style)
  const performSearch = (query: string) => {
    if (!query.trim()) {
      setFilteredCommands([])
      setFilteredInstruments([])
      setFilteredNews([])
      setShowSearchResults(false)
      return
    }

    const lowerQuery = query.toLowerCase()

    // Filter commands
    const matchingCommands = backtickCommands.filter(cmd => 
      cmd.command.toLowerCase().includes(lowerQuery) ||
      cmd.description.toLowerCase().includes(lowerQuery) ||
      cmd.category.toLowerCase().includes(lowerQuery)
    ).slice(0, 8)

    // Filter instruments
    const matchingInstruments = instruments.filter(instrument => 
      instrument.symbol.toLowerCase().includes(lowerQuery) ||
      instrument.name.toLowerCase().includes(lowerQuery)
    ).slice(0, 8)

    // Filter news
    const matchingNews = newsStories.filter(news => 
      news.headline.toLowerCase().includes(lowerQuery) ||
      news.ticker.toLowerCase().includes(lowerQuery)
    ).slice(0, 6)

    setFilteredCommands(matchingCommands)
    setFilteredInstruments(matchingInstruments)
    setFilteredNews(matchingNews)
    setShowSearchResults(true)
  }

  // Godel Terminal Commands (based on backtick menu analysis)
  const backtickCommands = [
    // Core Crypto Trading & Market Data
    { command: 'QM', description: 'Quote Monitor - Real-time crypto prices', category: 'Market Data' },
    { command: 'CHART', description: 'Advanced crypto charting with technical indicators', category: 'Analysis' },
    { command: 'ORDER', description: 'Place buy/sell orders', category: 'Trading' },
    { command: 'POSITIONS', description: 'View current trading positions', category: 'Trading' },
    { command: 'PF', description: 'Portfolio Performance & Analytics', category: 'Trading' },
    { command: 'AL', description: 'Price Alerts & Notifications', category: 'Trading' },
    { command: 'BT', description: 'Strategy Backtesting', category: 'Trading' },
    
    // Crypto Market Analysis
    { command: 'VOLUME', description: 'Trading volume analysis', category: 'Analysis' },
    { command: 'FEAR', description: 'Fear & Greed Index', category: 'Analysis' },
    { command: 'CORR', description: 'Crypto correlation analysis', category: 'Analysis' },
    { command: 'FLOW', description: 'On-chain data & whale movements', category: 'Analysis' },
    { command: 'DEFI', description: 'DeFi protocols & yield farming', category: 'Analysis' },
    { command: 'NFT', description: 'NFT market analysis', category: 'Analysis' },
    
    // Crypto News & Research
    { command: 'N', description: 'Real-time crypto news with advanced filtering', category: 'News' },
    { command: 'RES', description: 'Crypto research reports', category: 'Research' },
    { command: 'WHITEPAPER', description: 'Token whitepapers', category: 'Research' },
    
    // Account & Settings
    { command: 'ACCT', description: 'Account Management', category: 'Account' },
    { command: 'WALLET', description: 'Wallet connections & balances', category: 'Account' },
    { command: 'API', description: 'API key management', category: 'Account' },
    { command: 'SETTINGS', description: 'User settings & preferences', category: 'Settings' },
    
    // Tools & Utilities
    { command: 'CALC', description: 'Crypto calculator & converters', category: 'Tools' },
    { command: 'GAS', description: 'Gas fee tracker', category: 'Tools' },
    { command: 'STAKING', description: 'Staking rewards calculator', category: 'Tools' },
    { command: 'NOTE', description: 'Rich Text Notes Editor', category: 'Tools' },
    
    // Social & Community
    { command: 'CHAT', description: 'Live crypto chat', category: 'Social' },
    { command: 'TWITTER', description: 'Crypto Twitter feed', category: 'Social' },
    { command: 'REDDIT', description: 'Crypto Reddit discussions', category: 'Social' },
    
    // Additional Market Analysis
    { command: 'ORDERBOOK', description: 'Real-time order book with bids/asks', category: 'Market Data' },
    { command: 'HEATMAP', description: 'Market heatmap showing price changes', category: 'Analysis' },
    
    // Trading & Strategy Management
    { command: 'STRATEGIES', description: 'Trading strategies management', category: 'Trading' },
    { command: 'NOTIFICATIONS', description: 'Comprehensive notification center', category: 'Trading' },
    { command: 'ANALYSIS', description: 'Technical analysis tools and indicators', category: 'Analysis' },
    
    // Dashboard & Overview
    { command: 'OVERVIEW', description: 'Trading dashboard with key metrics', category: 'System' },
    
    // System Commands
    { command: 'HELP', description: 'Crypto Terminal Documentation', category: 'System' },
    { command: 'S', description: 'Keyboard Shortcuts', category: 'System' },
    { command: 'CLEAR', description: 'Clear all windows', category: 'System' },
    { command: 'LAYOUT', description: 'Save/Load window layouts', category: 'System' },
    { command: 'ERR', description: 'Report bugs and get support', category: 'System' }
  ];

  // Create a new window
  const createWindow = (type: string, title: string, content?: any) => {
    // Define window sizes for different crypto trading tools
    const getWindowSize = (windowType: string) => {
      switch (windowType) {
        case 'CHART': return { width: 800, height: 500 };
        case 'QM': return { width: 400, height: 600 };
        case 'NEWS': return { width: 900, height: 600 };
        case 'ORDER': return { width: 350, height: 400 };
        case 'POSITIONS': return { width: 500, height: 350 };
        case 'PF': return { width: 600, height: 400 };
        case 'VOLUME': return { width: 450, height: 300 };
        case 'FEAR': return { width: 300, height: 200 };
        case 'DEFI': return { width: 500, height: 350 };
        case 'WALLET': return { width: 400, height: 300 };
        case 'GAS': return { width: 350, height: 250 };
        case 'STAKING': return { width: 400, height: 300 };
        case 'ORDERBOOK': return { width: 500, height: 400 };
        case 'HEATMAP': return { width: 600, height: 450 };
        case 'STRATEGIES': return { width: 550, height: 400 };
        case 'NOTIFICATIONS': return { width: 450, height: 500 };
        case 'ANALYSIS': return { width: 500, height: 400 };
        case 'OVERVIEW': return { width: 700, height: 500 };
        default: return { width: 500, height: 300 };
      }
    };

    const { width, height } = getWindowSize(type);
    
    const newWindow = {
      id: `${type}-${Date.now()}`,
      type,
      title,
      x: Math.random() * (window.innerWidth - width) + 50,
      y: Math.random() * (window.innerHeight - height) + 100,
      width,
      height,
      isMinimized: false,
      zIndex: nextZIndex,
      content
    };

    setWindows(prev => [...prev, newWindow]);
    setNextZIndex(prev => prev + 1);
  };

  // Close a window
  const closeWindow = (id: string) => {
    setWindows(prev => prev.filter(w => w.id !== id));
  };

  // Bring window to front
  const bringToFront = (id: string) => {
    setWindows(prev => prev.map(w => 
      w.id === id ? { ...w, zIndex: nextZIndex } : w
    ));
    setNextZIndex(prev => prev + 1);
  };

  // Fetch alerts data
  const fetchAlertsData = async () => {
    try {
      // Get all alerts
      const alertsResponse = await fetch('/api/alerts');
      const alertsResult = await alertsResponse.json();
      
      // Get notifications
      const notificationsResponse = await fetch('/api/alerts/notifications?limit=100');
      const notificationsResult = await notificationsResponse.json();
      
      // Get alert summary
      const summaryResponse = await fetch('/api/alerts/summary');
      const summaryResult = await summaryResponse.json();
      
      return {
        alerts: alertsResult.success ? alertsResult.data : [],
        notifications: notificationsResult.success ? notificationsResult.data : [],
        summary: summaryResult.success ? summaryResult.data : {}
      };
    } catch (error) {
      console.error('Error fetching alerts data:', error);
      return {
        alerts: [],
        notifications: [],
        summary: {}
      };
    }
  };

  // Fetch portfolio data
  const fetchPortfolioData = async () => {
    try {
      // Get current portfolio
      const currentResponse = await fetch('/api/portfolio/current');
      const currentResult = await currentResponse.json();
      
      // Get performance metrics
      const performanceResponse = await fetch('/api/portfolio/performance');
      const performanceResult = await performanceResponse.json();
      
      // Get portfolio history
      const historyResponse = await fetch('/api/portfolio/history?days=30');
      const historyResult = await historyResponse.json();
      
      // Get trade history
      const tradesResponse = await fetch('/api/portfolio/trades?limit=50');
      const tradesResult = await tradesResponse.json();
      
      return {
        current: currentResult.success ? currentResult.data : {},
        performance: performanceResult.success ? performanceResult.data : {},
        history: historyResult.success ? historyResult.data : {},
        trades: tradesResult.success ? tradesResult.data : {}
      };
    } catch (error) {
      console.error('Error fetching portfolio data:', error);
      return {
        current: {},
        performance: {},
        history: {},
        trades: {}
      };
    }
  };

  // Fetch backtest data
  const fetchBacktestData = async () => {
    try {
      // Get all backtests
      const backtestsResponse = await fetch('/api/backtest/all');
      const backtestsResult = await backtestsResponse.json();
      
      // Get available strategies
      const strategiesResponse = await fetch('/api/backtest/strategies');
      const strategiesResult = await strategiesResponse.json();
      
      // Get summary
      const summaryResponse = await fetch('/api/backtest/summary');
      const summaryResult = await summaryResponse.json();
      
      return {
        backtests: backtestsResult.success ? backtestsResult.data : { active: [], completed: [] },
        strategies: strategiesResult.success ? strategiesResult.data : {},
        summary: summaryResult.success ? summaryResult.data : {}
      };
    } catch (error) {
      console.error('Error fetching backtest data:', error);
      return {
        backtests: { active: [], completed: [] },
        strategies: {},
        summary: {}
      };
    }
  };

  // Fetch chart data from data pipeline
  const fetchChartData = async (symbol: string, interval: string = '1h', days: number = 7) => {
    try {
      const response = await fetch(`/api/data/unified/${symbol}?interval=${interval}&days=${days}`);
      const result = await response.json();
      
      if (result.success && result.data) {
        return {
          symbol,
          interval,
          data: result.data,
          count: result.count
        };
      } else {
        console.error('Failed to fetch chart data:', result);
        return { symbol, interval, data: [], count: 0 };
      }
    } catch (error) {
      console.error('Error fetching chart data:', error);
      return { symbol, interval, data: [], count: 0 };
    }
  };
  // Execute command
  const executeCommand = (cmd: string) => {
    const upperCmd = cmd.toUpperCase().trim();
    
    switch (upperCmd) {
      case 'QM':
        // Create Quote Monitor with real Pyth prices
        const qmPairs = instruments.map(instrument => ({
          symbol: instrument.symbol,
          price: instrument.price,
          change: instrument.change,
          changePercent: instrument.changePercent,
          volume: instrument.volume,
          high24h: instrument.price * 1.02, // Approximate 24h high
          low24h: instrument.price * 0.98  // Approximate 24h low
        }))
        
        createWindow('QM', 'Live Quote Monitor', {
          pairs: qmPairs,
          lastUpdate: new Date().toLocaleTimeString(),
          marketCap: '2.1T',
          dominance: { btc: 42.3, eth: 18.7, others: 39.0 }
        });
        break;
      case 'N':
      case 'NEWS':
        createWindow('NEWS', 'Crypto News Feed', {
          searchQuery: '',
          filters: {
            sources: 'All',
            categories: 'All',
            languages: 'English',
            includeText: '',
            excludeText: ''
          },
          newsItems: [
            {
              headline: 'Bitcoin Surges Past $67,000 as Institutional Adoption Grows',
              date: '2025-10-18',
              time: '21:52',
              ticker: 'BTC',
              source: 'CoinDesk',
              category: 'Market Analysis'
            },
            {
              headline: 'Ethereum 2.0 Upgrade Shows Promising Results in Latest Testnet',
              date: '2025-10-18',
              time: '21:51',
              ticker: 'ETH',
              source: 'The Block',
              category: 'Technology'
            },
            {
              headline: 'DeFi Total Value Locked Reaches $200B Milestone',
              date: '2025-10-18',
              time: '21:50',
              ticker: 'DEFI',
              source: 'DeFi Pulse',
              category: 'DeFi'
            },
            {
              headline: 'Solana Network Activity Hits New Highs Amid Ecosystem Growth',
              date: '2025-10-18',
              time: '21:49',
              ticker: 'SOL',
              source: 'CoinTelegraph',
              category: 'Ecosystem'
            },
            {
              headline: 'SEC Approves First Bitcoin ETF After Years of Deliberation',
              date: '2025-10-18',
              time: '21:48',
              ticker: 'BTC',
              source: 'Reuters',
              category: 'Regulatory'
            },
            {
              headline: 'Cardano Smart Contracts Launch Successfully on Mainnet',
              date: '2025-10-18',
              time: '21:47',
              ticker: 'ADA',
              source: 'Cardano Foundation',
              category: 'Technology'
            },
            {
              headline: 'Polygon Announces Major Partnership with Global Payment Processor',
              date: '2025-10-18',
              time: '21:46',
              ticker: 'MATIC',
              source: 'Polygon Blog',
              category: 'Partnerships'
            },
            {
              headline: 'Avalanche Ecosystem Expands with 50+ New Projects',
              date: '2025-10-18',
              time: '21:45',
              ticker: 'AVAX',
              source: 'Avalanche Foundation',
              category: 'Ecosystem'
            },
            {
              headline: 'Chainlink Oracle Network Reaches New Milestone in Data Feeds',
              date: '2025-10-18',
              time: '21:44',
              ticker: 'LINK',
              source: 'Chainlink Labs',
              category: 'Infrastructure'
            },
            {
              headline: 'Uniswap V4 Protocol Upgrade Enters Final Testing Phase',
              date: '2025-10-18',
              time: '21:43',
              ticker: 'UNI',
              source: 'Uniswap Labs',
              category: 'Technology'
            }
          ],
          totalResults: '487,247,752',
          showingResults: '200',
          sourceTypes: [
            { name: 'CoinDesk', count: '1.2M', selected: true },
            { name: 'CoinTelegraph', count: '890K', selected: true },
            { name: 'The Block', count: '650K', selected: true },
            { name: 'Reuters', count: '2.1M', selected: true },
            { name: 'Bloomberg', count: '1.8M', selected: true },
            { name: 'DeFi Pulse', count: '45K', selected: true },
            { name: 'Crypto News', count: '234K', selected: false },
            { name: 'Bitcoin Magazine', count: '123K', selected: false },
            { name: 'Ethereum Foundation', count: '67K', selected: false },
            { name: 'Binance Blog', count: '89K', selected: false }
          ],
          categories: [
            { name: 'Market Analysis', count: '45.2M', selected: true },
            { name: 'Technology', count: '23.1M', selected: true },
            { name: 'Regulatory', count: '12.8M', selected: true },
            { name: 'DeFi', count: '8.9M', selected: true },
            { name: 'NFTs', count: '5.4M', selected: false },
            { name: 'Ecosystem', count: '7.2M', selected: true },
            { name: 'Partnerships', count: '3.1M', selected: false },
            { name: 'Infrastructure', count: '2.8M', selected: true }
          ]
        });
        break;
      case 'CHART':
        // Fetch chart data for BTC by default
        fetchChartData('BTCUSDT', '1h').then(data => {
          createWindow('CHART', 'Crypto Chart - BTC/USDT', data);
        });
        break;
      case 'CHAT':
        return (
          <Box key="chat" sx={{ height: '100%', width: '100%', display: 'flex', flexDirection: 'column' }}>
            <Typography variant="h6" sx={{ color: proTerminalConfig.themes.dark.textPrimary, mb: 1 }}>Global Chat</Typography>
            {chatError && <Typography color="error" sx={{ mb: 1 }}>{chatError}</Typography>}
            <Box sx={{ flexGrow: 1, overflowY: 'auto', mb: 2, p: 1, border: `1px solid ${proTerminalConfig.themes.dark.border}`, borderRadius: '4px', backgroundColor: proTerminalConfig.themes.dark.surface }}>
              <ChatWindow channel="global" />
            </Box>
            <MessageInput onSendMessage={handleSendMessage} isLoading={isSendingMessage} />
          </Box>
        );
      case 'CALC':
        createWindow('CALC', 'Crypto Calculator & Converters', { 
          functions: [
            { 
              name: 'Position Size Calculator',
              inputs: {
                accountBalance: '10000',
                riskPercent: '2',
                entryPrice: getPrice('BTC/USDT')?.price?.toFixed(0) || '67543',
                stopLoss: '65000',
                currency: 'BTC'
              },
              result: {
                positionSize: '0.148 BTC',
                riskAmount: '$200',
                maxLoss: '$200'
              }
            },
            { 
              name: 'DCA Calculator',
              inputs: {
                totalAmount: '5000',
                frequency: 'Weekly',
                duration: '12 weeks',
                currentPrice: getPrice('BTC/USDT')?.price?.toFixed(0) || '67543'
              },
              result: {
                weeklyAmount: '$416.67',
                totalBTC: '0.074 BTC',
                avgPrice: '$67,567'
              }
            },
            { 
              name: 'Profit/Loss Calculator',
              inputs: {
                entryPrice: '65000',
                exitPrice: getPrice('BTC/USDT')?.price?.toFixed(0) || '67543',
                amount: '0.1',
                fees: '0.1%'
              },
              result: {
                profit: '$254.30',
                profitPercent: '3.91%',
                afterFees: '$253.05'
              }
            },
            { 
              name: 'Currency Converter',
              inputs: {
                amount: '1000',
                from: 'USD',
                to: 'BTC'
              },
              result: {
                converted: '0.0148 BTC',
                rate: `${getPrice('BTC/USDT')?.price?.toFixed(0) || '67543'} USD/BTC`
              }
            },
            { 
              name: 'Compound Interest',
              inputs: {
                principal: '10000',
                apy: '8',
                years: '5',
                compoundFreq: 'Daily'
              },
              result: {
                finalAmount: '$14,918.25',
                totalInterest: '$4,918.25'
              }
            }
          ],
          quickConversions: {
            '1 BTC': '$67,543',
            '1 ETH': '$3,246',
            '1 SOL': '$247',
            '1 ADA': '$0.46',
            '1 DOT': '$8.92'
          }
        });
        break;
      case 'CF':
        createWindow('CF', 'SEC Filings', [
          { company: 'AAPL', filing: '10-K', date: '2025-09-15', status: 'Filed' },
          { company: 'GOOGL', filing: '10-Q', date: '2025-09-10', status: 'Filed' },
          { company: 'MSFT', filing: '8-K', date: '2025-09-08', status: 'Filed' }
        ]);
        break;
      case 'EQS':
        createWindow('EQS', 'Equity Screener', {
          filters: ['Market Cap > 1B', 'P/E < 20', 'Volume > 1M'],
          results: ['AAPL', 'MSFT', 'GOOGL', 'NVDA']
        });
        break;
      case 'IPO':
        createWindow('IPO', 'IPO Calendar', [
          { company: 'TechCorp', date: '2025-11-15', priceRange: '$18-22', status: 'Upcoming' },
          { company: 'DataInc', date: '2025-11-20', priceRange: '$25-30', status: 'Upcoming' }
        ]);
        break;
      case 'NOTE':
        createWindow('NOTE', 'Trading Notes', {
          content: 'Market Analysis - Oct 2025\n\n• Tech sector showing strength\n• Watch for Fed announcements\n• SOL breakout pattern forming'
        });
        break;
      case 'PRT':
        createWindow('PRT', 'Pattern Search - Systematic', {
          patterns: [
            { name: 'Head & Shoulders', confidence: 0.85, forward_return: '+12.3%', occurrences: 23 },
            { name: 'Double Bottom', confidence: 0.92, forward_return: '+8.7%', occurrences: 15 },
            { name: 'Bull Flag', confidence: 0.78, forward_return: '+15.1%', occurrences: 31 },
            { name: 'Cup & Handle', confidence: 0.88, forward_return: '+18.4%', occurrences: 12 }
          ]
        });
        break;
      case 'BT':
        // Fetch backtest data and create window
        fetchBacktestData().then(data => {
          createWindow('BT', 'Strategy Backtesting', data);
        });
        break;
      case 'PF':
        // Fetch portfolio data and create window
        fetchPortfolioData().then(data => {
          createWindow('PF', 'Portfolio Performance', data);
        });
        break;
      case 'AL':
        // Fetch alerts data and create window
        fetchAlertsData().then(data => {
          createWindow('AL', 'Price Alerts & Notifications', data);
        });
        break;
      case 'BROK':
        createWindow('BROK', 'Trading Interface', { mode: 'positions' });
        break;
      case 'TOP':
        createWindow('TOP', 'Top Reuters News', [
          { headline: 'Federal Reserve Hints at Rate Pause', time: '2 hours ago', source: 'Reuters', priority: 'High' },
          { headline: 'Tech Earnings Beat Expectations', time: '4 hours ago', source: 'Reuters', priority: 'Medium' },
          { headline: 'Oil Prices Surge on Supply Concerns', time: '6 hours ago', source: 'Reuters', priority: 'High' },
          { headline: 'Crypto Regulation Update from SEC', time: '8 hours ago', source: 'Reuters', priority: 'Medium' }
        ]);
        break;
      case 'GR':
        createWindow('GR', 'Security Correlation Graph', {
          pair: 'AAPL vs SPY',
          correlation: 0.85,
          timeframe: '1Y',
          data_points: 252,
          r_squared: 0.72
        });
        break;
      case 'HPS':
        createWindow('HPS', 'Historical Performance Comparison', {
          securities: ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
          timeframe: '5Y',
          returns: [
            { symbol: 'AAPL', return_1y: '+28.7%', return_3y: '+156.2%', return_5y: '+312.8%' },
            { symbol: 'MSFT', return_1y: '+32.1%', return_3y: '+142.9%', return_5y: '+298.4%' },
            { symbol: 'GOOGL', return_1y: '+15.3%', return_3y: '+98.7%', return_5y: '+201.5%' },
            { symbol: 'TSLA', return_1y: '-23.4%', return_3y: '+245.8%', return_5y: '+1,247.3%' }
          ]
        });
        break;
      case 'TAS':
        createWindow('TAS', 'Time & Sales Data', [
          { time: '15:59:58', price: 171.35, size: 500, side: 'BUY' },
          { time: '15:59:57', price: 171.34, size: 200, side: 'SELL' },
          { time: '15:59:56', price: 171.36, size: 1000, side: 'BUY' },
          { time: '15:59:55', price: 171.33, size: 300, side: 'SELL' },
          { time: '15:59:54', price: 171.37, size: 750, side: 'BUY' }
        ]);
        break;
      case 'RSS':
        createWindow('RSS', 'Research Reports', [
          { title: 'Q4 2025 Market Outlook', analyst: 'Goldman Sachs', date: '2025-09-15', rating: 'BUY' },
          { title: 'Tech Sector Deep Dive', analyst: 'Morgan Stanley', date: '2025-09-12', rating: 'OVERWEIGHT' },
          { title: 'Crypto Winter Analysis', analyst: 'JPMorgan', date: '2025-09-10', rating: 'NEUTRAL' },
          { title: 'AI Stocks Valuation Report', analyst: 'Bank of America', date: '2025-09-08', rating: 'BUY' }
        ]);
        break;
      case 'XPRT':
        createWindow('XPRT', 'Expert Reports', [
          { expert: 'Ray Dalio', topic: 'Economic Cycles', confidence: 'High', recommendation: 'Diversify into commodities' },
          { expert: 'Cathie Wood', topic: 'Innovation Stocks', confidence: 'Very High', recommendation: 'Long-term AI plays' },
          { expert: 'Warren Buffett', topic: 'Value Investing', confidence: 'High', recommendation: 'Focus on fundamentals' },
          { expert: 'Michael Burry', topic: 'Market Bubbles', confidence: 'Medium', recommendation: 'Cautious positioning' }
        ]);
        break;
      case 'MET':
        createWindow('MET', 'Global Market Indices', [
          { index: 'S&P 500', value: 4756.50, change: '+1.2%', country: 'USA' },
          { index: 'NASDAQ', value: 14845.73, change: '+1.8%', country: 'USA' },
          { index: 'FTSE 100', value: 7456.89, change: '+0.7%', country: 'UK' },
          { index: 'DAX', value: 16789.34, change: '+0.9%', country: 'Germany' },
          { index: 'Nikkei 225', value: 33456.78, change: '-0.3%', country: 'Japan' },
          { index: 'Shanghai Composite', value: 2987.65, change: '+0.5%', country: 'China' }
        ]);
        break;
      case 'ORG':
        createWindow('ORG', 'Organization Billing', {
          plan: 'Professional',
          monthly_cost: '$299',
          usage: {
            api_calls: '847,392 / 1,000,000',
            data_feeds: '12 / 15',
            users: '3 / 5'
          },
          next_billing: '2025-11-15'
        });
        break;
      case 'ACCT':
        createWindow('ACCT', 'Account Management', {
          user: 'QuantDesk Trader',
          email: 'trader@quantdesk.app',
          plan: 'Professional',
          member_since: '2023-06-15',
          permissions: ['Trading', 'Research', 'Analytics', 'API Access']
        });
        break;
      case 'PDF':
        createWindow('PDF', 'User Settings', {
          preferences: {
            theme: 'Dark',
            timezone: 'EST',
            default_chart: 'Candlestick',
            notifications: 'Enabled',
            auto_save: 'Every 5 minutes'
          }
        });
        break;
      case 'CHANGE':
        createWindow('CHANGE', 'QuantDesk Terminal Changelog', [
          { version: 'v1.0.1', date: '2025-09-15', changes: ['Added Pro Terminal mode', 'Improved window management', 'Enhanced backtick menu'] },
          { version: 'v1.0.0', date: '2025-09-01', changes: ['Initial release', 'Basic trading features', 'Paper trading framework'] }
        ]);
        break;
      case 'CITADEL':
        createWindow('CITADEL', 'Citadel Overview', {
          company: 'Citadel LLC',
          founded: '1990',
          aum: '$59 billion',
          founder: 'Ken Griffin',
          strategies: ['Equity Market Making', 'Fixed Income', 'Commodities', 'Credit'],
          performance: 'Annual return: ~20% (since inception)'
        });
        break;
      case 'MARTIN':
        createWindow('MARTIN', 'Files & Links', [
          { name: 'Trading Strategies.pdf', type: 'Document', size: '2.3 MB' },
          { name: 'Market Analysis Q4.xlsx', type: 'Spreadsheet', size: '1.8 MB' },
          { name: 'Risk Management Guide', type: 'Link', url: 'https://quantdesk.app/risk' },
          { name: 'API Documentation', type: 'Link', url: 'https://quantdesk.app/docs' }
        ]);
        break;
      case 'NEJM':
        createWindow('NEJM', 'Medical Journal Research', [
          { title: 'AI in Healthcare Investing', date: '2025-09-10', relevance: 'Biotech sector analysis' },
          { title: 'Pharmaceutical Patent Cliff', date: '2025-09-05', relevance: 'Drug company valuations' },
          { title: 'Medical Device Innovation', date: '2023-12-28', relevance: 'MedTech investment opportunities' }
        ]);
        break;
      case 'GH':
        createWindow('GH', 'GitHub News Feed', [
          { repo: 'quantdesk-trading/core', event: 'New release v2.1.0', time: '2 hours ago' },
          { repo: 'microsoft/vscode', event: 'Security update', time: '4 hours ago' },
          { repo: 'openai/gpt-4', event: 'Model improvements', time: '1 day ago' }
        ]);
        break;
      case 'NH':
        createWindow('NH', 'News Hub', [
          { category: 'Markets', headline: 'Futures point to higher open', source: 'CNBC', time: '30 min ago' },
          { category: 'Crypto', headline: 'Bitcoin approaches $50K resistance', source: 'CoinDesk', time: '1 hour ago' },
          { category: 'Earnings', headline: 'Tech earnings season begins', source: 'Bloomberg', time: '2 hours ago' }
        ]);
        break;
      case 'NT':
        createWindow('NT', 'News Search', {
          search_term: 'Enter search term...',
          recent_searches: ['Fed rate decision', 'AI stocks', 'Crypto regulation', 'Earnings calendar']
        });
        break;
      case 'BUG':
      case 'ERR':
        createWindow('ERR', 'Bug Report & Support', {
          contact: 'contact@quantdesk.app',
          common_issues: [
            'Window not responding',
            'Data feed disconnected', 
            'Login issues',
            'Chart not loading'
          ],
          system_info: {
            version: 'v1.0.1',
            browser: navigator.userAgent.split(' ')[0],
            platform: navigator.platform
          }
        });
        break;
      case 'RES':
        createWindow('RES', 'Research Reports', [
          { title: 'Crypto Market Analysis Q4 2025', author: 'QuantDesk Research', date: '2025-09-15', rating: 'BUY' },
          { title: 'DeFi Protocol Valuation Models', author: 'Blockchain Analytics', date: '2025-09-12', rating: 'HOLD' },
          { title: 'Meme Coin Trend Analysis', author: 'Social Trading Desk', date: '2025-09-10', rating: 'SPECULATIVE' }
        ]);
        break;
      case 'CN':
        createWindow('CN', 'China News Feed', [
          { title: 'Shanghai Composite Index Update', time: '5 min ago', source: 'Shanghai Exchange' },
          { title: 'PBOC Policy Statement', time: '15 min ago', source: 'People\'s Bank of China' },
          { title: 'Tech Sector Regulations Update', time: '1 hour ago', source: 'CSRC' }
        ]);
        break;
      case 'NI':
        createWindow('NI', 'News Search Interface', {
          placeholder: 'Enter search terms (e.g., "Bitcoin", "Federal Reserve", "Earnings")',
          recent_searches: ['Bitcoin ETF', 'Interest rates', 'Tech earnings', 'Crypto regulation'],
          trending: ['AI stocks', 'Green energy', 'Inflation data']
        });
        break;
      case 'HMS':
        createWindow('HMS', 'Historical Multi-Security Comparison', {
          securities: ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD'],
          timeframes: ['1D', '1W', '1M', '3M', '1Y'],
          metrics: ['Price Change %', 'Volume', 'Market Cap', 'Volatility']
        });
        break;
      case 'WEI':
        createWindow('WEI', 'World Equity Indices Monitor', [
          { index: 'S&P 500', value: '4,756.50', change: '+0.85%', region: 'US' },
          { index: 'NASDAQ', value: '14,893.23', change: '+1.12%', region: 'US' },
          { index: 'FTSE 100', value: '7,445.12', change: '+0.23%', region: 'UK' },
          { index: 'Nikkei 225', value: '33,288.47', change: '-0.45%', region: 'Japan' },
          { index: 'DAX', value: '16,751.64', change: '+0.67%', region: 'Germany' }
        ]);
        break;
      case 'MOST':
        createWindow('MOST', 'Most Active', [
          { ticker: 'SPY', volume: '125.2M', price: 425.67, change: 1.23 },
          { ticker: 'QQQ', volume: '89.5M', price: 367.89, change: -0.45 },
          { ticker: 'AAPL', volume: '67.3M', price: 171.35, change: -2.02 },
          { ticker: 'TSLA', volume: '45.8M', price: 333.50, change: -0.47 }
        ]);
        break;
      case 'CLEAR':
        setWindows([]);
        break;
      case 'HELP':
        createWindow('HELP', 'Help & Commands', backtickCommands);
        break;
      case 'S':
      case 'SHORTCUTS':
        createWindow('SHORTCUTS', 'Keyboard Shortcuts', {});
        break;
      
      // New Crypto-Specific Commands
      case 'ORDER':
        createWindow('ORDER', 'Place Trading Order', {
          pair: 'BTC/USDT',
          currentPrice: 67543.21,
          side: 'BUY',
          amount: '0.1',
          price: '67543.21',
          orderType: 'LIMIT',
          totalValue: '6754.32',
          availableBalance: { USDT: '15000.00', BTC: '0.25' },
          orderBook: {
            bids: [
              { price: 67540.00, amount: '0.5', total: '33770.00' },
              { price: 67535.00, amount: '1.2', total: '81042.00' },
              { price: 67530.00, amount: '0.8', total: '54024.00' }
            ],
            asks: [
              { price: 67545.00, amount: '0.3', total: '20263.50' },
              { price: 67550.00, amount: '0.7', total: '47285.00' },
              { price: 67555.00, amount: '1.1', total: '74310.50' }
            ]
          },
          recentTrades: [
            { price: 67543.21, amount: '0.05', side: 'BUY', time: '14:32:15' },
            { price: 67542.80, amount: '0.12', side: 'SELL', time: '14:32:12' },
            { price: 67544.50, amount: '0.08', side: 'BUY', time: '14:32:08' }
          ]
        });
        break;
      case 'POSITIONS':
        createWindow('POSITIONS', 'Current Positions', {
          positions: [
            { 
              pair: 'BTC/USDT', 
              side: 'LONG', 
              size: '0.5', 
              entry: '65000.00', 
              current: '67543.21', 
              pnl: '+1271.61', 
              pnlPercent: '+3.91%',
              margin: '3250.00',
              leverage: '10x',
              openTime: '2025-09-15 09:30:00'
            },
            { 
              pair: 'ETH/USDT', 
              side: 'SHORT', 
              size: '2.0', 
              entry: '3300.00', 
              current: '3245.67', 
              pnl: '+108.66', 
              pnlPercent: '+1.65%',
              margin: '330.00',
              leverage: '20x',
              openTime: '2025-09-16 14:22:00'
            },
            { 
              pair: 'SOL/USDT', 
              side: 'LONG', 
              size: '10.0', 
              entry: '240.00', 
              current: '246.78', 
              pnl: '+67.80', 
              pnlPercent: '+2.83%',
              margin: '240.00',
              leverage: '5x',
              openTime: '2025-09-17 11:15:00'
            },
            { 
              pair: 'ADA/USDT', 
              side: 'LONG', 
              size: '1000.0', 
              entry: '0.44', 
              current: '0.4567', 
              pnl: '+16.70', 
              pnlPercent: '+3.80%',
              margin: '44.00',
              leverage: '10x',
              openTime: '2025-09-18 16:45:00'
            }
          ],
          totalPnL: '+1464.77',
          totalPnLPercent: '+3.12%',
          totalMargin: '3864.00',
          totalValue: '48528.77'
        });
        break;
      case 'VOLUME':
        createWindow('VOLUME', 'Trading Volume Analysis', {
          topPairs: [
            { pair: 'BTC/USDT', volume24h: '2.4B', change: '+15.2%', avgVolume: '2.1B', volumeRatio: 1.14 },
            { pair: 'ETH/USDT', volume24h: '1.8B', change: '+8.7%', avgVolume: '1.6B', volumeRatio: 1.13 },
            { pair: 'SOL/USDT', volume24h: '890M', change: '+23.1%', avgVolume: '720M', volumeRatio: 1.24 },
            { pair: 'ADA/USDT', volume24h: '456M', change: '+12.3%', avgVolume: '405M', volumeRatio: 1.13 },
            { pair: 'DOT/USDT', volume24h: '234M', change: '-5.2%', avgVolume: '247M', volumeRatio: 0.95 },
            { pair: 'MATIC/USDT', volume24h: '345M', change: '+18.9%', avgVolume: '290M', volumeRatio: 1.19 }
          ],
          volumeProfile: {
            trend: 'Increasing across major pairs',
            totalVolume24h: '6.1B',
            volumeChange: '+12.8%',
            activePairs: 156,
            volumeDistribution: {
              spot: '4.2B (68.9%)',
              futures: '1.9B (31.1%)'
            }
          },
          hourlyVolume: [
            { hour: '00:00', volume: '245M' },
            { hour: '01:00', volume: '198M' },
            { hour: '02:00', volume: '167M' },
            { hour: '03:00', volume: '189M' },
            { hour: '04:00', volume: '223M' },
            { hour: '05:00', volume: '256M' },
            { hour: '06:00', volume: '289M' },
            { hour: '07:00', volume: '312M' },
            { hour: '08:00', volume: '345M' },
            { hour: '09:00', volume: '378M' },
            { hour: '10:00', volume: '401M' },
            { hour: '11:00', volume: '423M' },
            { hour: '12:00', volume: '445M' },
            { hour: '13:00', volume: '467M' },
            { hour: '14:00', volume: '489M' },
            { hour: '15:00', volume: '512M' },
            { hour: '16:00', volume: '534M' },
            { hour: '17:00', volume: '556M' },
            { hour: '18:00', volume: '578M' },
            { hour: '19:00', volume: '601M' },
            { hour: '20:00', volume: '623M' },
            { hour: '21:00', volume: '645M' },
            { hour: '22:00', volume: '667M' },
            { hour: '23:00', volume: '689M' }
          ]
        });
        break;
      case 'FEAR':
        createWindow('FEAR', 'Fear & Greed Index', {
          current: 65,
          label: 'Greed',
          previous: 58,
          change: '+7',
          timeframe: '24h',
          components: {
            volatility: 65,
            marketMomentum: 70,
            socialMedia: 60,
            surveys: 55,
            dominance: 75
          },
          historicalData: [
            { date: '2025-09-15', value: 58, label: 'Greed' },
            { date: '2025-09-16', value: 62, label: 'Greed' },
            { date: '2025-09-17', value: 59, label: 'Greed' },
            { date: '2025-09-18', value: 65, label: 'Greed' },
            { date: '2025-09-19', value: 68, label: 'Greed' },
            { date: '2025-09-20', value: 65, label: 'Greed' }
          ],
          interpretation: 'Market showing strong bullish sentiment with increasing greed levels. Caution advised for contrarian positions.',
          extremeLevels: {
            extremeFear: 0,
            fear: 25,
            neutral: 50,
            greed: 75,
            extremeGreed: 100
          }
        });
        break;
      case 'CORR':
        createWindow('CORR', 'Crypto Correlation Matrix', {
          pairs: [
            { pair: 'BTC-ETH', correlation: 0.85, strength: 'Strong', trend: 'Increasing' },
            { pair: 'BTC-SOL', correlation: 0.72, strength: 'Moderate', trend: 'Stable' },
            { pair: 'ETH-SOL', correlation: 0.78, strength: 'Moderate', trend: 'Increasing' },
            { pair: 'BTC-ADA', correlation: 0.68, strength: 'Moderate', trend: 'Decreasing' },
            { pair: 'ETH-ADA', correlation: 0.71, strength: 'Moderate', trend: 'Stable' },
            { pair: 'SOL-ADA', correlation: 0.65, strength: 'Moderate', trend: 'Increasing' },
            { pair: 'BTC-DOT', correlation: 0.62, strength: 'Moderate', trend: 'Stable' },
            { pair: 'ETH-DOT', correlation: 0.69, strength: 'Moderate', trend: 'Increasing' }
          ],
          timeframe: '30D',
          matrix: {
            BTC: { BTC: 1.00, ETH: 0.85, SOL: 0.72, ADA: 0.68, DOT: 0.62 },
            ETH: { BTC: 0.85, ETH: 1.00, SOL: 0.78, ADA: 0.71, DOT: 0.69 },
            SOL: { BTC: 0.72, ETH: 0.78, SOL: 1.00, ADA: 0.65, DOT: 0.58 },
              ADA: { BTC: 0.68, ETH: 0.71, SOL: 0.65, ADA: 1.00, DOT: 0.61 },
            DOT: { BTC: 0.62, ETH: 0.69, SOL: 0.58, ADA: 0.61, DOT: 1.00 }
          },
          insights: [
            'Bitcoin maintains strong correlation with Ethereum (0.85)',
            'Altcoins showing moderate correlation with BTC (0.62-0.72)',
            'DeFi tokens (ETH, SOL) show higher inter-correlation',
            'Correlation levels increasing during market volatility'
          ]
        });
        break;
      case 'FLOW':
        createWindow('FLOW', 'On-Chain Data & Whale Movements', {
          whaleTransactions: [
            { 
              address: '1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa', 
              amount: '500 BTC', 
              value: '$33.7M',
              type: 'INFLOW', 
              time: '2h ago',
              exchange: 'Binance',
              confidence: 'High'
            },
            { 
              address: '0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6', 
              amount: '10,000 ETH', 
              value: '$32.5M',
              type: 'OUTFLOW', 
              time: '4h ago',
              exchange: 'Coinbase',
              confidence: 'High'
            },
            { 
              address: 'bc1qxy2kgdygjrsqtzq2n0yrf2493p83kkfjhx0wlh', 
              amount: '1,200 BTC', 
              value: '$81.1M',
              type: 'INFLOW', 
              time: '6h ago',
              exchange: 'Kraken',
              confidence: 'Medium'
            },
            { 
              address: '0x28C6c06298d514Db089934071355E5743bf21d60', 
              amount: '25,000 ETH', 
              value: '$81.1M',
              type: 'OUTFLOW', 
              time: '8h ago',
              exchange: 'Binance',
              confidence: 'High'
            }
          ],
          exchangeFlows: { 
            binance: { flow: '+1.2B', change: '+15.3%', trend: 'Inflow' },
            coinbase: { flow: '-800M', change: '-8.7%', trend: 'Outflow' },
            kraken: { flow: '+300M', change: '+12.1%', trend: 'Inflow' },
            ftx: { flow: '+150M', change: '+5.2%', trend: 'Inflow' }
          },
          onChainMetrics: {
            activeAddresses: '1.2M',
            transactionCount: '285K',
            networkHashRate: '456.7 EH/s',
            difficultyAdjustment: '+2.3%',
            mempoolSize: '12.5K',
            averageFee: '8.5 sat/vB'
          },
          whaleAlerts: [
            'Large BTC accumulation detected (500+ BTC)',
            'Institutional wallet movement: 10K ETH',
            'Exchange outflow spike: Coinbase -800M',
            'Whale distribution pattern emerging'
          ]
        });
        break;
      case 'DEFI':
        createWindow('DEFI', 'DeFi Protocols & Yield Farming', {
          protocols: [
            { 
              name: 'Uniswap V3', 
              tvl: '4.2B', 
              apy: '12.5%', 
              risk: 'Medium',
              change24h: '+2.3%',
              tokens: ['ETH', 'USDC', 'WBTC'],
              category: 'DEX'
            },
            { 
              name: 'Compound', 
              tvl: '2.8B', 
              apy: '8.3%', 
              risk: 'Low',
              change24h: '+1.1%',
              tokens: ['ETH', 'USDC', 'DAI'],
              category: 'Lending'
            },
            { 
              name: 'Aave', 
              tvl: '6.1B', 
              apy: '15.7%', 
              risk: 'Medium',
              change24h: '+3.2%',
              tokens: ['ETH', 'USDC', 'WBTC', 'LINK'],
              category: 'Lending'
            },
            { 
              name: 'Curve Finance', 
              tvl: '3.4B', 
              apy: '18.2%', 
              risk: 'Medium',
              change24h: '+0.8%',
              tokens: ['USDC', 'USDT', 'DAI'],
              category: 'Stablecoin'
            },
            { 
              name: 'PancakeSwap', 
              tvl: '1.9B', 
              apy: '22.1%', 
              risk: 'High',
              change24h: '+4.5%',
              tokens: ['BNB', 'CAKE', 'BUSD'],
              category: 'DEX'
            }
          ],
          totalTVL: '89.2B',
          totalTVLChange: '+2.8%',
          categories: {
            lending: '45.2B',
            dex: '28.7B',
            stablecoin: '12.3B',
            derivatives: '3.0B'
          },
          topYields: [
            { protocol: 'PancakeSwap', pair: 'CAKE-BNB', apy: '45.2%' },
            { protocol: 'Uniswap V3', pair: 'ETH-USDC', apy: '28.7%' },
            { protocol: 'Curve Finance', pair: '3CRV', apy: '18.2%' },
            { protocol: 'Aave', token: 'USDC', apy: '15.7%' }
          ],
          riskMetrics: {
            totalRiskScore: 'Medium',
            smartContractRisk: 'Low',
            impermanentLossRisk: 'Medium',
            liquidityRisk: 'Low'
          }
        });
        break;
      case 'NFT':
        createWindow('NFT', 'NFT Market Analysis', {
          collections: [
            { 
              name: 'Bored Ape Yacht Club', 
              floor: '45.2 ETH', 
              volume24h: '234 ETH', 
              change: '+12.3%',
              sales24h: 12,
              avgPrice: '19.5 ETH',
              marketCap: '1.2B ETH'
            },
            { 
              name: 'CryptoPunks', 
              floor: '67.8 ETH', 
              volume24h: '156 ETH', 
              change: '+8.7%',
              sales24h: 8,
              avgPrice: '19.5 ETH',
              marketCap: '2.1B ETH'
            },
            { 
              name: 'Mutant Ape Yacht Club', 
              floor: '12.3 ETH', 
              volume24h: '89 ETH', 
              change: '+15.2%',
              sales24h: 15,
              avgPrice: '5.9 ETH',
              marketCap: '456M ETH'
            },
            { 
              name: 'Azuki', 
              floor: '8.7 ETH', 
              volume24h: '67 ETH', 
              change: '+6.8%',
              sales24h: 9,
              avgPrice: '7.4 ETH',
              marketCap: '234M ETH'
            },
            { 
              name: 'CloneX', 
              floor: '6.2 ETH', 
              volume24h: '45 ETH', 
              change: '+3.2%',
              sales24h: 7,
              avgPrice: '6.4 ETH',
              marketCap: '189M ETH'
            }
          ],
          totalVolume: '1.2B ETH',
          totalVolumeChange: '+8.9%',
          marketMetrics: {
            totalSales: '2,456',
            avgPrice: '12.3 ETH',
            uniqueBuyers: '1,234',
            uniqueSellers: '987'
          },
          topSales: [
            { collection: 'CryptoPunks', price: '125 ETH', buyer: '0x1234...', time: '2h ago' },
            { collection: 'Bored Ape Yacht Club', price: '98 ETH', buyer: '0x5678...', time: '4h ago' },
            { collection: 'Mutant Ape Yacht Club', price: '67 ETH', buyer: '0x9abc...', time: '6h ago' }
          ],
          trendingCollections: [
            { name: 'Art Blocks', change: '+25.3%', volume: '89 ETH' },
            { name: 'World of Women', change: '+18.7%', volume: '67 ETH' },
            { name: 'Cool Cats', change: '+12.4%', volume: '45 ETH' }
          ]
        });
        break;
      case 'WHITEPAPER':
        createWindow('WHITEPAPER', 'Token Whitepapers', [
          { token: 'Bitcoin', version: 'v0.1', author: 'Satoshi Nakamoto', date: '2008-10-31' },
          { token: 'Ethereum', version: 'v1.0', author: 'Vitalik Buterin', date: '2013-11-19' },
          { token: 'Solana', version: 'v1.0', author: 'Anatoly Yakovenko', date: '2017-11-06' }
        ]);
        break;
      case 'WALLET':
        createWindow('WALLET', 'Wallet Connections & Balances', {
          connectedWallets: [
            { 
              name: 'MetaMask', 
              address: '0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6', 
              balance: '2.5 ETH',
              value: '$8,114',
              network: 'Ethereum',
              status: 'Connected',
              lastUsed: '2 min ago'
            },
            { 
              name: 'Phantom', 
              address: '9WzDXwBbmk8DdS1HWH9pp5QbE1xNATcoDxX9JhS8gF7k', 
              balance: '150 SOL',
              value: '$37,017',
              network: 'Solana',
              status: 'Connected',
              lastUsed: '5 min ago'
            },
            { 
              name: 'Trust Wallet', 
              address: '0x1234567890abcdef1234567890abcdef12345678', 
              balance: '0.8 BTC',
              value: '$54,034',
              network: 'Bitcoin',
              status: 'Connected',
              lastUsed: '1 hour ago'
            }
          ],
          totalValue: '$99,165',
          totalValueChange: '+2.3%',
          portfolio: {
            ETH: { amount: '2.5', value: '$8,114', percentage: '8.2%' },
            SOL: { amount: '150', value: '$37,017', percentage: '37.3%' },
            BTC: { amount: '0.8', value: '$54,034', percentage: '54.5%' }
          },
          recentTransactions: [
            { type: 'Received', amount: '0.5 ETH', from: '0x1234...', time: '2h ago', status: 'Confirmed' },
            { type: 'Sent', amount: '25 SOL', to: '9WzDX...', time: '4h ago', status: 'Confirmed' },
            { type: 'Received', amount: '0.1 BTC', from: 'bc1q...', time: '6h ago', status: 'Confirmed' }
          ],
          gasFees: {
            ethereum: '0.002 ETH ($6.48)',
            solana: '0.000005 SOL ($0.001)',
            bitcoin: '0.0001 BTC ($6.75)'
          }
        });
        break;
      case 'API':
        createWindow('API', 'API Key Management', {
          keys: [
            { name: 'Binance API', status: 'Active', permissions: ['Read', 'Trade'], lastUsed: '2 min ago' },
            { name: 'CoinGecko API', status: 'Active', permissions: ['Read'], lastUsed: '5 min ago' }
          ]
        });
        break;
      case 'SETTINGS':
        createWindow('SETTINGS', 'User Settings & Preferences', {
          trading: { defaultPair: 'BTC/USDT', orderType: 'LIMIT', riskLevel: 'Medium' },
          notifications: { priceAlerts: true, newsAlerts: true, tradeAlerts: true },
          display: { theme: 'Dark', timezone: 'UTC', currency: 'USD' }
        });
        break;
      case 'GAS':
        createWindow('GAS', 'Gas Fee Tracker', {
          ethereum: { 
            slow: '15 gwei', 
            standard: '20 gwei', 
            fast: '25 gwei',
            slowCost: '$2.10',
            standardCost: '$2.80',
            fastCost: '$3.50',
            networkCongestion: 'Medium'
          },
          polygon: { 
            slow: '1 gwei', 
            standard: '2 gwei', 
            fast: '5 gwei',
            slowCost: '$0.001',
            standardCost: '$0.002',
            fastCost: '$0.005',
            networkCongestion: 'Low'
          },
          arbitrum: { 
            slow: '0.1 gwei', 
            standard: '0.2 gwei', 
            fast: '0.5 gwei',
            slowCost: '$0.0001',
            standardCost: '$0.0002',
            fastCost: '$0.0005',
            networkCongestion: 'Low'
          },
          bsc: {
            slow: '3 gwei',
            standard: '5 gwei',
            fast: '8 gwei',
            slowCost: '$0.15',
            standardCost: '$0.25',
            fastCost: '$0.40',
            networkCongestion: 'Low'
          },
          avalanche: {
            slow: '25 nAVAX',
            standard: '30 nAVAX',
            fast: '50 nAVAX',
            slowCost: '$0.75',
            standardCost: '$0.90',
            fastCost: '$1.50',
            networkCongestion: 'Low'
          },
          gasTrends: {
            ethereum: { trend: 'Decreasing', change: '-12%', avg24h: '22 gwei' },
            polygon: { trend: 'Stable', change: '+2%', avg24h: '2 gwei' },
            arbitrum: { trend: 'Stable', change: '0%', avg24h: '0.2 gwei' }
          },
          recommendations: [
            'Use Polygon for low-cost transactions',
            'Arbitrum offers best value for DeFi',
            'Ethereum gas fees decreasing - good time for mainnet',
            'Consider BSC for Binance ecosystem tokens'
          ]
        });
        break;
      case 'STAKING':
        createWindow('STAKING', 'Staking Rewards Calculator', {
          protocols: [
            { 
              name: 'Ethereum 2.0', 
              apy: '4.2%', 
              minStake: '32 ETH', 
              risk: 'Low',
              currentStaked: '28.5M ETH',
              validatorCount: '850K',
              rewardFrequency: 'Daily',
              unstakingPeriod: '7 days'
            },
            { 
              name: 'Solana', 
              apy: '6.8%', 
              minStake: '1 SOL', 
              risk: 'Medium',
              currentStaked: '420M SOL',
              validatorCount: '1.2K',
              rewardFrequency: 'Epoch (2-3 days)',
              unstakingPeriod: '2-3 days'
            },
            { 
              name: 'Cardano', 
              apy: '5.1%', 
              minStake: '1 ADA', 
              risk: 'Low',
              currentStaked: '22.8B ADA',
              validatorCount: '3.2K',
              rewardFrequency: 'Epoch (5 days)',
              unstakingPeriod: '2 epochs'
            },
            { 
              name: 'Polkadot', 
              apy: '12.3%', 
              minStake: '10 DOT', 
              risk: 'Medium',
              currentStaked: '650M DOT',
              validatorCount: '297',
              rewardFrequency: 'Era (24 hours)',
              unstakingPeriod: '28 days'
            },
            { 
              name: 'Cosmos', 
              apy: '8.7%', 
              minStake: '1 ATOM', 
              risk: 'Low',
              currentStaked: '180M ATOM',
              validatorCount: '150',
              rewardFrequency: 'Daily',
              unstakingPeriod: '21 days'
            }
          ],
          stakingRewards: {
            totalStaked: '$89.2B',
            totalRewards: '$3.7B annually',
            avgAPY: '7.4%',
            activeStakers: '2.1M'
          },
          calculator: {
            amount: '1000',
            currency: 'ETH',
            protocol: 'Ethereum 2.0',
            estimatedRewards: {
              daily: '0.115 ETH',
              monthly: '3.5 ETH',
              yearly: '42 ETH'
            }
          },
          risks: [
            'Slashing risk for validator misbehavior',
            'Liquidity lock-up during unstaking period',
            'Validator downtime affects rewards',
            'Network congestion may delay unstaking'
          ]
        });
        break;
      case 'TWITTER':
        createWindow('TWITTER', 'Crypto Twitter Feed', {
          tweets: [
            { 
              user: '@elonmusk', 
              handle: 'Elon Musk',
              tweet: 'Bitcoin is the future of money 💰', 
              time: '1h ago', 
              likes: '45K',
              retweets: '12K',
              replies: '3.2K',
              verified: true
            },
            { 
              user: '@VitalikButerin', 
              handle: 'Vitalik Buterin',
              tweet: 'Ethereum scaling solutions are progressing well. Layer 2 adoption accelerating! 🚀', 
              time: '2h ago', 
              likes: '23K',
              retweets: '5.1K',
              replies: '1.8K',
              verified: true
            },
            { 
              user: '@naval', 
              handle: 'Naval Ravikant',
              tweet: 'The best time to buy Bitcoin was 10 years ago. The second best time is now.', 
              time: '3h ago', 
              likes: '18K',
              retweets: '4.2K',
              replies: '2.1K',
              verified: true
            },
            { 
              user: '@APompliano', 
              handle: 'Anthony Pompliano',
              tweet: 'Institutional adoption of Bitcoin continues to accelerate. The future is bright! 🌅', 
              time: '4h ago', 
              likes: '15K',
              retweets: '3.8K',
              replies: '1.5K',
              verified: true
            },
            { 
              user: '@aantonop', 
              handle: 'Andreas M. Antonopoulos',
              tweet: 'Bitcoin is not just digital gold, it\'s programmable money for the internet age.', 
              time: '5h ago', 
              likes: '12K',
              retweets: '2.9K',
              replies: '1.2K',
              verified: true
            }
          ],
          trending: [
            '#Bitcoin', '#Ethereum', '#DeFi', '#NFTs', '#Web3', '#Crypto', '#Blockchain'
          ],
          influencers: [
            '@elonmusk', '@VitalikButerin', '@naval', '@APompliano', '@aantonop'
          ]
        });
        break;
      case 'REDDIT':
        createWindow('REDDIT', 'Crypto Reddit Discussions', {
          posts: [
            { 
              subreddit: 'r/Bitcoin', 
              title: 'BTC breaks $67K resistance - Bull run confirmed?', 
              upvotes: '2.3K', 
              comments: '456',
              time: '2h ago',
              author: 'u/BitcoinHodler',
              flair: 'Price Discussion'
            },
            { 
              subreddit: 'r/ethereum', 
              title: 'Layer 2 adoption accelerating - Polygon hits new ATH', 
              upvotes: '1.8K', 
              comments: '234',
              time: '4h ago',
              author: 'u/EthMaximalist',
              flair: 'Technology'
            },
            { 
              subreddit: 'r/CryptoCurrency', 
              title: 'DeFi yield farming opportunities - 18% APY on stablecoins', 
              upvotes: '1.2K', 
              comments: '189',
              time: '6h ago',
              author: 'u/DeFiTrader',
              flair: 'DeFi'
            },
            { 
              subreddit: 'r/Solana', 
              title: 'Solana ecosystem expanding - New DEX launches', 
              upvotes: '890', 
              comments: '156',
              time: '8h ago',
              author: 'u/SolanaFan',
              flair: 'Ecosystem'
            },
            { 
              subreddit: 'r/Cardano', 
              title: 'Cardano staking rewards remain stable at 5.1% APY', 
              upvotes: '756', 
              comments: '98',
              time: '10h ago',
              author: 'u/AdaStaker',
              flair: 'Staking'
            }
          ],
          subreddits: [
            { name: 'r/Bitcoin', members: '4.2M', online: '12.5K' },
            { name: 'r/ethereum', members: '1.8M', online: '8.9K' },
            { name: 'r/CryptoCurrency', members: '6.7M', online: '25.3K' },
            { name: 'r/Solana', members: '456K', online: '3.2K' },
            { name: 'r/Cardano', members: '789K', online: '4.1K' }
          ],
          trending: [
            'Bitcoin ETF approval', 'Ethereum 2.0 upgrade', 'DeFi yield farming', 'NFT market recovery'
          ]
        });
        break;
      
      // Additional Missing Commands from Lite Version
      case 'ORDERBOOK':
        createWindow('ORDERBOOK', 'Order Book - BTC/USDT', {
          pair: 'BTC/USDT',
          bids: [
            { price: 67540.00, amount: '0.5', total: '33770.00' },
            { price: 67535.00, amount: '1.2', total: '81042.00' },
            { price: 67530.00, amount: '0.8', total: '54024.00' },
            { price: 67525.00, amount: '2.1', total: '141802.50' },
            { price: 67520.00, amount: '0.9', total: '60768.00' },
            { price: 67515.00, amount: '1.5', total: '101272.50' },
            { price: 67510.00, amount: '0.7', total: '47257.00' },
            { price: 67505.00, amount: '1.8', total: '121509.00' },
            { price: 67500.00, amount: '3.2', total: '216000.00' },
            { price: 67495.00, amount: '0.6', total: '40497.00' }
          ],
          asks: [
            { price: 67545.00, amount: '0.3', total: '20263.50' },
            { price: 67550.00, amount: '0.7', total: '47285.00' },
            { price: 67555.00, amount: '1.1', total: '74310.50' },
            { price: 67560.00, amount: '0.8', total: '54048.00' },
            { price: 67565.00, amount: '1.4', total: '94591.00' },
            { price: 67570.00, amount: '0.9', total: '60813.00' },
            { price: 67575.00, amount: '1.6', total: '108120.00' },
            { price: 67580.00, amount: '0.5', total: '33790.00' },
            { price: 67585.00, amount: '2.3', total: '155445.50' },
            { price: 67590.00, amount: '0.7', total: '47313.00' }
          ],
          spread: '5.00',
          spreadPercent: '0.007%',
          lastUpdate: new Date().toLocaleTimeString()
        });
        break;
      
      case 'HEATMAP':
        createWindow('HEATMAP', 'Market Heatmap', {
          pairs: [
            { symbol: 'BTC/USDT', price: 67543.21, change: 1.86, color: 'green', intensity: 'high' },
            { symbol: 'ETH/USDT', price: 3245.67, change: -1.37, color: 'red', intensity: 'medium' },
            { symbol: 'SOL/USDT', price: 246.78, change: 5.26, color: 'green', intensity: 'high' },
            { symbol: 'ADA/USDT', price: 0.4567, change: 2.78, color: 'green', intensity: 'medium' },
            { symbol: 'DOT/USDT', price: 8.92, change: -2.52, color: 'red', intensity: 'medium' },
            { symbol: 'MATIC/USDT', price: 0.89, change: 3.98, color: 'green', intensity: 'high' },
            { symbol: 'AVAX/USDT', price: 34.56, change: 3.68, color: 'green', intensity: 'medium' },
            { symbol: 'LINK/USDT', price: 14.78, change: -2.95, color: 'red', intensity: 'medium' },
            { symbol: 'UNI/USDT', price: 6.23, change: 2.98, color: 'green', intensity: 'medium' },
            { symbol: 'ATOM/USDT', price: 9.45, change: 3.73, color: 'green', intensity: 'medium' }
          ],
          timeframe: '24h',
          totalPairs: 156,
          greenCount: 89,
          redCount: 67,
          marketSentiment: 'Bullish'
        });
        break;
      
      case 'STRATEGIES':
        createWindow('STRATEGIES', 'Trading Strategies', {
          strategies: [
            {
              name: 'DCA Bot',
              description: 'Dollar Cost Averaging strategy',
              apy: '12.5%',
              risk: 'Low',
              status: 'Active',
              trades: 45,
              winRate: '78%'
            },
            {
              name: 'Grid Trading',
              description: 'Grid trading bot for range-bound markets',
              apy: '18.3%',
              risk: 'Medium',
              status: 'Active',
              trades: 123,
              winRate: '65%'
            },
            {
              name: 'Momentum Trader',
              description: 'Momentum-based trading strategy',
              apy: '25.7%',
              risk: 'High',
              status: 'Paused',
              trades: 67,
              winRate: '58%'
            },
            {
              name: 'Arbitrage Bot',
              description: 'Cross-exchange arbitrage opportunities',
              apy: '8.9%',
              risk: 'Low',
              status: 'Active',
              trades: 234,
              winRate: '92%'
            }
          ],
          totalStrategies: 4,
          activeStrategies: 3,
          totalAPY: '16.1%',
          totalTrades: 469
        });
        break;
      
      case 'NOTIFICATIONS':
        createWindow('NOTIFICATIONS', 'Notifications Center', {
          notifications: [
            {
              id: '1',
              type: 'price_alert',
              title: 'BTC Price Alert',
              message: 'BTC reached $67,000 target price',
              time: '2 min ago',
              read: false,
              priority: 'high'
            },
            {
              id: '2',
              type: 'trade_executed',
              title: 'Trade Executed',
              message: 'Buy order filled: 0.1 BTC at $67,543',
              time: '15 min ago',
              read: false,
              priority: 'medium'
            },
            {
              id: '3',
              type: 'strategy_update',
              title: 'Strategy Update',
              message: 'DCA Bot completed 10th purchase',
              time: '1 hour ago',
              read: true,
              priority: 'low'
            },
            {
              id: '4',
              type: 'market_news',
              title: 'Market News',
              message: 'Bitcoin ETF approval expected this week',
              time: '2 hours ago',
              read: true,
              priority: 'medium'
            }
          ],
          unreadCount: 2,
          settings: {
            priceAlerts: true,
            tradeAlerts: true,
            newsAlerts: true,
            strategyAlerts: true,
            emailNotifications: false,
            pushNotifications: true
          }
        });
        break;
      
      case 'ANALYSIS':
        createWindow('ANALYSIS', 'Technical Analysis Tools', {
          indicators: [
            {
              name: 'RSI',
              value: 65.4,
              signal: 'Neutral',
              timeframe: '1h',
              description: 'Relative Strength Index'
            },
            {
              name: 'MACD',
              value: 'Bullish',
              signal: 'Buy',
              timeframe: '4h',
              description: 'Moving Average Convergence Divergence'
            },
            {
              name: 'Bollinger Bands',
              value: 'Upper Band',
              signal: 'Overbought',
              timeframe: '1d',
              description: 'Price volatility indicator'
            },
            {
              name: 'Moving Average',
              value: 'Above MA50',
              signal: 'Bullish',
              timeframe: '1d',
              description: 'Trend following indicator'
            }
          ],
          patterns: [
            {
              name: 'Head & Shoulders',
              confidence: 0.85,
              timeframe: '4h',
              target: 'Bearish reversal'
            },
            {
              name: 'Double Bottom',
              confidence: 0.72,
              timeframe: '1d',
              target: 'Bullish reversal'
            }
          ],
          signals: {
            buy: 3,
            sell: 1,
            neutral: 2
          }
        });
        break;
      
      case 'OVERVIEW':
        createWindow('OVERVIEW', 'Trading Dashboard', {
          portfolio: {
            totalValue: '$99,165',
            change24h: '+2.3%',
            changeAmount: '+$2,234'
          },
          positions: {
            active: 4,
            totalPnL: '+$1,464.77',
            winRate: '75%'
          },
          market: {
            btcPrice: '$67,543.21',
            btcChange: '+1.86%',
            marketCap: '$2.1T',
            dominance: '42.3%'
          },
          alerts: {
            active: 8,
            triggered: 2,
            pending: 6
          },
          strategies: {
            active: 3,
            totalAPY: '16.1%',
            trades: 469
          },
          recentActivity: [
            { action: 'Buy Order', pair: 'BTC/USDT', amount: '0.1 BTC', time: '15 min ago' },
            { action: 'Price Alert', pair: 'ETH/USDT', price: '$3,300', time: '1 hour ago' },
            { action: 'Strategy Update', strategy: 'DCA Bot', status: 'Completed', time: '2 hours ago' }
          ]
        });
        break;
      
      default:
        if (upperCmd.startsWith('QUOTE ')) {
          const symbol = upperCmd.split(' ')[1];
        // Fetch real quote data for window
        fetch(`http://localhost:8000/api/market/quote/${symbol}USDT`)
          .then(response => response.json())
          .then(data => {
            if (data.success) {
              const quote = data.data
              createWindow('QUOTE', `Quote: ${symbol}`, {
                symbol,
                price: quote.price.toFixed(2),
                change: quote.change_24h.toFixed(2),
                changePercent: quote.change_24h_percent.toFixed(2),
                volume: quote.volume_24h.toLocaleString()
              });
            } else {
              createWindow('QUOTE', `Quote: ${symbol}`, {
                symbol,
                price: 'N/A',
                change: 'N/A',
                changePercent: 'N/A',
                volume: 'N/A'
              });
            }
          })
          .catch(error => {
            createWindow('QUOTE', `Quote: ${symbol}`, {
              symbol,
              price: 'Error',
              change: 'Error',
              changePercent: 'Error',
              volume: 'Error'
            });
          });
        } else {
          alert(`Unknown command: ${cmd}\nPress backtick (\`) to see all commands`);
        }
    }

    // Add to history
    if (cmd.trim() && !commandHistory.includes(cmd.trim())) {
      setCommandHistory(prev => [cmd.trim(), ...prev.slice(0, 19)]);
    }
    setHistoryIndex(-1);
    setCommand('');
  };
  // Render window content
  const renderWindowContent = (window: any) => {
    // Safety check to prevent object-to-primitive conversion errors
    if (!window || !window.type) {
      return (
        <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>
          Invalid window data
        </div>
      );
    }
    
    switch (window.type) {
      case 'QM':
        const qmData = Array.isArray(window.content?.pairs) ? window.content.pairs : [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              alignItems: 'center',
              marginBottom: '10px',
              fontSize: '12px'
            }}>
              <span style={{ color: 'var(--success-500)', fontWeight: 'bold' }}>LIVE QUOTES</span>
              <span style={{ color: 'var(--text-muted)' }}>
                {qmData.length} symbols • {window.content?.lastUpdate || new Date().toLocaleTimeString()}
              </span>
            </div>
            
            {/* Market Overview */}
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Market Cap:</span>
                <span style={{ color: 'var(--success-500)' }}>{window.content?.marketCap || '$2.1T'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>BTC Dominance:</span>
                <span style={{ color: 'var(--primary-500)' }}>{window.content?.dominance?.btc || '42.3'}%</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>ETH Dominance:</span>
                <span style={{ color: 'var(--primary-500)' }}>{window.content?.dominance?.eth || '18.7'}%</span>
              </div>
            </div>
            
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Symbol</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Price</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>24h%</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Volume</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>High/Low</th>
                </tr>
              </thead>
              <tbody>
                {qmData.map((pair: any, idx: number) => (
                  <tr key={idx} style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                    <td style={{ padding: '4px', color: 'var(--success-500)', fontWeight: 'bold' }}>
                      {pair.symbol}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--text-primary)' }}>
                      ${pair.price ? pair.price.toFixed(pair.price < 1 ? 4 : 2) : 'N/A'}
                    </td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: (pair.changePercent || 0) >= 0 ? 'var(--success-500)' : 'var(--danger-500)',
                      fontWeight: 'bold'
                    }}>
                      {(Number(pair.changePercent) || 0) >= 0 ? '+' : ''}{(Number(pair.changePercent) || 0).toFixed(2)}%
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--primary-500)' }}>
                      {pair.volume || 'N/A'}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--warning-500)' }}>
                      {pair.high24h ? `$${Number(pair.high24h).toFixed(0)}/${Number(pair.low24h || 0).toFixed(0)}` : 'N/A'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'NEWS':
        const newsData = Array.isArray(window.content?.newsItems) ? window.content.newsItems : [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            {/* Search Bar */}
            <div style={{ marginBottom: '10px' }}>
              <input 
                type="text" 
                placeholder="Press / to search"
                style={{ 
                  width: '100%', 
                padding: '8px', 
                  backgroundColor: 'var(--bg-secondary)', 
                  border: '1px solid var(--primary-500)', 
                  borderRadius: '4px',
                  color: 'white'
                }}
              />
                </div>
            
            {/* Filter Controls */}
            <div style={{ marginBottom: '10px', display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
              <button style={{ padding: '4px 8px', backgroundColor: 'var(--primary-500)', border: 'none', borderRadius: '4px', color: 'white', fontSize: '10px' }}>Watchlists</button>
              <button style={{ padding: '4px 8px', backgroundColor: 'var(--primary-500)', border: 'none', borderRadius: '4px', color: 'white', fontSize: '10px' }}>All</button>
              <button style={{ padding: '4px 8px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--primary-500)', borderRadius: '4px', color: 'white', fontSize: '10px' }}>Clear</button>
              <button style={{ padding: '4px 8px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--primary-500)', borderRadius: '4px', color: 'white', fontSize: '10px' }}>Pause</button>
              <button style={{ padding: '4px 8px', backgroundColor: 'var(--bg-secondary)', border: '1px solid var(--primary-500)', borderRadius: '4px', color: 'white', fontSize: '10px' }}>1 Filter</button>
            </div>
            
            {/* News Table Header */}
            <div style={{ display: 'flex', fontSize: '10px', fontWeight: 'bold', marginBottom: '8px', borderBottom: '1px solid var(--primary-500)', paddingBottom: '4px' }}>
              <div style={{ flex: '3', color: 'var(--text-primary)' }}>Headline</div>
              <div style={{ flex: '1', color: 'var(--text-primary)' }}>Date</div>
              <div style={{ flex: '1', color: 'var(--text-primary)' }}>Time</div>
              <div style={{ flex: '1', color: 'var(--text-primary)' }}>Ticker</div>
              <div style={{ flex: '1', color: 'var(--text-primary)' }}>Source</div>
            </div>
            
            {/* News Items */}
            {newsData.map((news: any, idx: number) => (
              <div key={idx} style={{ 
                display: 'flex',
                fontSize: '10px',
                padding: '6px 0',
                borderBottom: '1px solid var(--bg-secondary)',
                cursor: 'pointer'
              }}>
                <div style={{ flex: '3', color: 'white' }}>{news.headline}</div>
                <div style={{ flex: '1', color: 'var(--text-muted)' }}>{news.date}</div>
                <div style={{ flex: '1', color: 'var(--text-muted)' }}>{news.time}</div>
                <div style={{ flex: '1', color: 'var(--primary-500)' }}>{news.ticker}</div>
                <div style={{ flex: '1', color: 'var(--text-muted)' }}>{news.source}</div>
              </div>
            ))}
            
            {/* Footer */}
            <div style={{ marginTop: '10px', fontSize: '10px', color: 'var(--text-muted)', textAlign: 'center' }}>
              Showing {newsData.length} of {window.content?.totalResults || '0'} Results
            </div>
          </div>
        );

      case 'CHART':
        if (window.content?.data && window.content.data.length > 0) {
          return (
            <div style={{ height: '100%', overflow: 'hidden' }}>
              <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '10px',
                borderBottom: '1px solid #333',
                backgroundColor: 'var(--bg-secondary)'
              }}>
                <span style={{ color: 'var(--success-500)', fontWeight: 'bold', fontSize: '14px' }}>
                  {window.content.symbol} - {window.content.interval?.toUpperCase()} Chart
                </span>
                <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
                  {window.content.count} data points
                </span>
              </div>
              
              {/* Simple chart visualization */}
              <div style={{ height: 'calc(100% - 50px)', padding: '15px' }}>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(8px, 1fr))',
                  gap: '1px',
                  height: '60%',
                  alignItems: 'end',
                  backgroundColor: 'var(--bg-tertiary)',
                  padding: '10px',
                  borderRadius: '4px',
                  marginBottom: '15px'
                }}>
                  {window.content.data.slice(-50).map((candle: any, idx: number) => {
                    const maxPrice = Math.max(...window.content.data.slice(-50).map((d: any) => d.high));
                    const minPrice = Math.min(...window.content.data.slice(-50).map((d: any) => d.low));
                    const priceRange = maxPrice - minPrice;
                    const candleHeight = Math.max(5, ((candle.high - minPrice) / priceRange) * 180);
                    const isGreen = candle.close >= candle.open;
                    
                    return (
                      <div
                        key={idx}
                        style={{
                          height: `${candleHeight}px`,
                          backgroundColor: isGreen ? 'var(--success-500)' : 'var(--danger-500)',
                          minWidth: '4px',
                          opacity: 0.8,
                          cursor: 'pointer'
                        }}
                        title={`${new Date(candle.timestamp).toLocaleString()}\nO: $${candle.open?.toFixed(2)}\nH: $${candle.high?.toFixed(2)}\nL: $${candle.low?.toFixed(2)}\nC: $${candle.close?.toFixed(2)}`}
                      />
                    );
                  })}
                </div>
                
                {/* Chart stats */}
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(4, 1fr)',
                  gap: '10px',
                  fontSize: '11px'
                }}>
                  <div>
                    <div style={{ color: '#999' }}>Latest</div>
                    <div style={{ color: 'var(--text-primary)', fontWeight: 'bold' }}>
                      ${window.content.data[window.content.data.length - 1]?.close?.toFixed(2) || 'N/A'}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#999' }}>High</div>
                    <div style={{ color: 'var(--success-500)' }}>
                      ${Math.max(...window.content.data.map((d: any) => d.high)).toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#999' }}>Low</div>
                    <div style={{ color: 'var(--danger-500)' }}>
                      ${Math.min(...window.content.data.map((d: any) => d.low)).toFixed(2)}
                    </div>
                  </div>
                  <div>
                    <div style={{ color: '#999' }}>Points</div>
                    <div style={{ color: 'var(--primary-500)' }}>
                      {window.content.count}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          );
        } else {
          return (
            <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>
              <div style={{ fontSize: '14px', marginBottom: '10px' }}>Loading chart data...</div>
              <div style={{ fontSize: '11px' }}>
                Fetching historical and live data for {window.content?.symbol || 'symbol'}
              </div>
            </div>
          );
        };

      case 'CALC':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Financial Calculator</h3>
            {window.content?.functions?.map((func: string, idx: number) => (
              <button
                key={idx}
                style={{
                  display: 'block',
                  width: '100%',
                  padding: '8px',
                  marginBottom: '8px',
                  background: '#333',
                  border: 'none',
                  color: 'var(--text-primary)',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '11px'
                }}
                onClick={() => alert(`${func} calculator - Implementation pending`)}
              >
                {func}
              </button>
            ))}
          </div>
        );

      case 'CF':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Company</th>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Filing</th>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Date</th>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {window.content?.map((filing: any, idx: number) => (
                  <tr key={idx}>
                    <td style={{ padding: '4px', color: 'var(--success-500)' }}>{filing.company}</td>
                    <td style={{ padding: '4px' }}>{filing.filing}</td>
                    <td style={{ padding: '4px' }}>{filing.date}</td>
                    <td style={{ padding: '4px', color: 'var(--primary-500)' }}>{filing.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'EQS':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Equity Screener</h3>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>Active Filters:</div>
              {window.content?.filters?.map((filter: string, idx: number) => (
                <div key={idx} style={{ fontSize: '11px', color: 'var(--primary-500)', marginBottom: '4px' }}>
                  • {filter}
                </div>
              ))}
            </div>
            <div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>Results:</div>
              {window.content?.results?.map((symbol: string, idx: number) => (
                <div key={idx} style={{ fontSize: '11px', color: 'var(--success-500)', marginBottom: '4px' }}>
                  {symbol}
                </div>
              ))}
            </div>
          </div>
        );

      case 'NOTE':
        return (
          <div style={{ padding: '10px', height: '100%' }}>
            <textarea
              defaultValue={window.content?.content || ''}
              style={{
                width: '100%',
                height: '100%',
                background: '#000',
                border: '1px solid #333',
                color: 'var(--text-primary)',
                padding: '8px',
                fontSize: '11px',
                fontFamily: 'monospace',
                resize: 'none'
              }}
              placeholder="Enter your trading notes..."
            />
          </div>
        );


      case 'IPO':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Company</th>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Date</th>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Price Range</th>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Status</th>
                </tr>
              </thead>
              <tbody>
                {window.content?.map((ipo: any, idx: number) => (
                  <tr key={idx}>
                    <td style={{ padding: '4px', color: 'var(--success-500)' }}>{ipo.company}</td>
                    <td style={{ padding: '4px' }}>{ipo.date}</td>
                    <td style={{ padding: '4px' }}>{ipo.priceRange}</td>
                    <td style={{ padding: '4px', color: 'var(--primary-500)' }}>{ipo.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'MOST':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Ticker</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Volume</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Price</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Change</th>
                </tr>
              </thead>
              <tbody>
                {window.content?.map((stock: any, idx: number) => (
                  <tr key={idx}>
                    <td style={{ padding: '4px', color: 'var(--success-500)' }}>{stock.ticker}</td>
                    <td style={{ padding: '4px', textAlign: 'right' }}>{stock.volume}</td>
                    <td style={{ padding: '4px', textAlign: 'right' }}>{stock.price}</td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: stock.change > 0 ? 'var(--success-500)' : 'var(--danger-500)'
                    }}>
                      {stock.change > 0 ? '+' : ''}{stock.change}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'PRT':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Pattern Recognition Results</h3>
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Pattern</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Confidence</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Forward Return</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Count</th>
                </tr>
              </thead>
              <tbody>
                {window.content?.patterns?.map((pattern: any, idx: number) => (
                  <tr key={idx}>
                    <td style={{ padding: '4px', color: 'var(--success-500)' }}>{pattern.name}</td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--primary-500)' }}>{(pattern.confidence * 100).toFixed(0)}%</td>
                    <td style={{ padding: '4px', textAlign: 'right', color: pattern.forward_return.includes('+') ? 'var(--success-500)' : 'var(--danger-500)' }}>
                      {pattern.forward_return}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right' }}>{pattern.occurrences}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'BT':
        return (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '10px',
              borderBottom: '1px solid #333',
              backgroundColor: 'var(--bg-secondary)'
            }}>
              <span style={{ color: 'var(--success-500)', fontWeight: 'bold', fontSize: '14px' }}>
                STRATEGY BACKTESTING
              </span>
              <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
                {window.content?.summary?.total_backtests || 0} total tests
              </span>
            </div>

            <div style={{ flex: 1, overflow: 'auto', padding: '15px' }}>
              {/* Quick Start Section */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Quick Start Backtest
                </div>
                <div style={{
                  padding: '15px',
                  backgroundColor: 'var(--bg-tertiary)',
                  borderRadius: '4px',
                  marginBottom: '15px'
                }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '10px', marginBottom: '10px' }}>
                    <select style={{
                      padding: '6px',
                      backgroundColor: '#333',
                      color: 'var(--text-primary)',
                      border: '1px solid #555',
                      borderRadius: '2px',
                      fontSize: '11px'
                    }}>
                      <option>Select Strategy</option>
                      {Object.keys(window.content?.strategies || {}).map(strategy => (
                        <option key={strategy} value={strategy}>{strategy}</option>
                      ))}
                    </select>
                    <select style={{
                      padding: '6px',
                      backgroundColor: '#333',
                      color: 'var(--text-primary)',
                      border: '1px solid #555',
                      borderRadius: '2px',
                      fontSize: '11px'
                    }}>
                      <option>BTCUSDT</option>
                      <option>ETHUSDT</option>
                      <option>SOLUSDT</option>
                    </select>
                    <button style={{
                      padding: '6px 12px',
                      backgroundColor: 'var(--success-500)',
                      color: '#000',
                      border: 'none',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 'bold',
                      cursor: 'pointer'
                    }}>
                      START BACKTEST
                    </button>
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                    Default: Last 30 days, $10K capital, 0.1% commission
                  </div>
                </div>
              </div>

              {/* Active Backtests */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Active Backtests ({window.content?.backtests?.active?.length || 0})
                </div>
                {window.content?.backtests?.active?.length > 0 ? 
                  window.content.backtests.active.map((backtest: any, idx: number) => (
                    <div key={idx} style={{
                      padding: '10px',
                      backgroundColor: 'var(--bg-tertiary)',
                      borderRadius: '4px',
                      marginBottom: '8px',
                      borderLeft: '3px solid var(--warning-500)'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{ fontSize: '11px', fontWeight: 'bold' }}>
                          {backtest.strategy_name} - {backtest.symbol}
                        </span>
                        <span style={{ fontSize: '10px', color: 'var(--warning-500)' }}>
                          {backtest.status.toUpperCase()}
                        </span>
                      </div>
                      <div style={{ fontSize: '10px', color: '#999' }}>
                        Started: {new Date(backtest.start_time).toLocaleString()}
                      </div>
                    </div>
                  )) : (
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', padding: '20px' }}>
                      No active backtests
                    </div>
                  )
                }
              </div>

              {/* Completed Backtests */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Recent Results ({window.content?.backtests?.completed?.slice(0, 5).length || 0})
                </div>
                {window.content?.backtests?.completed?.slice(0, 5).map((backtest: any, idx: number) => (
                  <div key={idx} style={{
                    padding: '10px',
                    backgroundColor: 'var(--bg-tertiary)',
                    borderRadius: '4px',
                    marginBottom: '8px',
                    borderLeft: `3px solid ${backtest.status === 'completed' ? 'var(--success-500)' : 'var(--danger-500)'}`
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                      <span style={{ fontSize: '11px', fontWeight: 'bold' }}>
                        {backtest.strategy_name} - {backtest.symbol}
                      </span>
                      <span style={{ 
                        fontSize: '10px', 
                        color: backtest.status === 'completed' ? 'var(--success-500)' : 'var(--danger-500)'
                      }}>
                        {backtest.status.toUpperCase()}
                      </span>
                    </div>
                    <div style={{ fontSize: '10px', color: '#999' }}>
                      Duration: {backtest.end_time ? 
                        Math.round((new Date(backtest.end_time).getTime() - new Date(backtest.start_time).getTime()) / 1000 / 60) + 'm' : 
                        'N/A'
                      }
                    </div>
                    {backtest.error && (
                      <div style={{ fontSize: '10px', color: 'var(--danger-500)', marginTop: '4px' }}>
                        Error: {backtest.error}
                      </div>
                    )}
                  </div>
                )) || (
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', padding: '20px' }}>
                    No completed backtests
                  </div>
                )}
              </div>

              {/* Available Strategies */}
              <div>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Available Strategies ({Object.keys(window.content?.strategies || {}).length})
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '8px' }}>
                  {Object.entries(window.content?.strategies || {}).map(([key, strategy]: [string, any]) => (
                    <div key={key} style={{
                      padding: '8px',
                      backgroundColor: 'var(--bg-tertiary)',
                      borderRadius: '4px',
                      cursor: 'pointer'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#333'}
                    onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-tertiary)'}
                    >
                      <div style={{ fontSize: '11px', fontWeight: 'bold', marginBottom: '4px' }}>
                        {strategy.name || key}
                      </div>
                      <div style={{ fontSize: '10px', color: '#999' }}>
                        {strategy.description || 'No description'}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        );
      case 'PF':
        const formatCurrency = (value: number) => {
          return new Intl.NumberFormat('en-US', { 
            style: 'currency', 
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
          }).format(value);
        };
        
        const formatPercent = (value: number) => {
          return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;
        };

        return (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '10px',
              borderBottom: '1px solid #333',
              backgroundColor: 'var(--bg-secondary)'
            }}>
              <span style={{ color: 'var(--success-500)', fontWeight: 'bold', fontSize: '14px' }}>
                PORTFOLIO PERFORMANCE
              </span>
              <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
                {window.content?.current?.position_count || 0} positions
              </span>
            </div>

            <div style={{ flex: 1, overflow: 'auto', padding: '15px' }}>
              {/* Portfolio Summary */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Portfolio Summary
                </div>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                  gap: '15px',
                  marginBottom: '15px'
                }}>
                  <div style={{
                    padding: '12px',
                    backgroundColor: 'var(--bg-tertiary)',
                    borderRadius: '4px',
                    textAlign: 'center'
                  }}>
                    <div style={{ fontSize: '16px', fontWeight: 'bold', color: 'var(--text-primary)' }}>
                      {formatCurrency(window.content?.current?.total_value || 0)}
                    </div>
                    <div style={{ fontSize: '10px', color: '#999' }}>Total Value</div>
                  </div>
                  
                  <div style={{
                    padding: '12px',
                    backgroundColor: 'var(--bg-tertiary)',
                    borderRadius: '4px',
                    textAlign: 'center'
                  }}>
                    <div style={{ 
                      fontSize: '16px', 
                      fontWeight: 'bold',
                      color: (window.content?.current?.total_pnl || 0) >= 0 ? 'var(--success-500)' : 'var(--danger-500)'
                    }}>
                      {formatCurrency(window.content?.current?.total_pnl || 0)}
                    </div>
                    <div style={{ fontSize: '10px', color: '#999' }}>Total P&L</div>
                  </div>
                  
                  <div style={{
                    padding: '12px',
                    backgroundColor: 'var(--bg-tertiary)',
                    borderRadius: '4px',
                    textAlign: 'center'
                  }}>
                    <div style={{ 
                      fontSize: '16px', 
                      fontWeight: 'bold',
                      color: (window.content?.current?.total_return_percent || 0) >= 0 ? 'var(--success-500)' : 'var(--danger-500)'
                    }}>
                      {formatPercent(window.content?.current?.total_return_percent || 0)}
                    </div>
                    <div style={{ fontSize: '10px', color: '#999' }}>Total Return</div>
                  </div>
                </div>
              </div>

              {/* Performance Metrics */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Performance Metrics
                </div>
                <div style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))',
                  gap: '10px',
                  fontSize: '11px'
                }}>
                  <div style={{ padding: '8px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px' }}>
                    <div style={{ color: '#999' }}>Win Rate</div>
                    <div style={{ color: 'var(--success-500)', fontWeight: 'bold' }}>
                      {formatPercent(window.content?.performance?.win_rate || 0)}
                    </div>
                  </div>
                  
                  <div style={{ padding: '8px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px' }}>
                    <div style={{ color: '#999' }}>Sharpe Ratio</div>
                    <div style={{ color: 'var(--primary-500)', fontWeight: 'bold' }}>
                      {(window.content?.performance?.sharpe_ratio || 0).toFixed(2)}
                    </div>
                  </div>
                  
                  <div style={{ padding: '8px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px' }}>
                    <div style={{ color: '#999' }}>Max Drawdown</div>
                    <div style={{ color: 'var(--danger-500)', fontWeight: 'bold' }}>
                      {formatPercent(window.content?.performance?.max_drawdown_percent || 0)}
                    </div>
                  </div>
                  
                  <div style={{ padding: '8px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px' }}>
                    <div style={{ color: '#999' }}>Volatility</div>
                    <div style={{ color: 'var(--warning-500)', fontWeight: 'bold' }}>
                      {formatPercent(window.content?.performance?.volatility || 0)}
                    </div>
                  </div>
                </div>
              </div>

              {/* Current Positions */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Current Positions ({Object.keys(window.content?.current?.positions || {}).length})
                </div>
                {Object.entries(window.content?.current?.positions || {}).length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {Object.entries(window.content.current.positions).map(([symbol, position]: [string, any]) => (
                      <div key={symbol} style={{
                        padding: '10px',
                        backgroundColor: 'var(--bg-tertiary)',
                        borderRadius: '4px',
                        borderLeft: `3px solid ${position.unrealized_pnl >= 0 ? 'var(--success-500)' : 'var(--danger-500)'}`
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                          <span style={{ fontSize: '11px', fontWeight: 'bold' }}>{symbol}</span>
                          <span style={{ 
                            fontSize: '11px', 
                            color: position.unrealized_pnl >= 0 ? 'var(--success-500)' : 'var(--danger-500)',
                            fontWeight: 'bold'
                          }}>
                            {formatCurrency(position.unrealized_pnl)} ({formatPercent(position.unrealized_pnl_percent)})
                          </span>
                        </div>
                        <div style={{ fontSize: '10px', color: '#999' }}>
                          Qty: {position.quantity} | Entry: {formatCurrency(position.entry_price)} | Current: {formatCurrency(position.current_price)}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', padding: '20px' }}>
                    No current positions
                  </div>
                )}
              </div>

              {/* Recent Trades */}
              <div>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Recent Trades ({window.content?.trades?.trades?.length || 0})
                </div>
                {window.content?.trades?.trades?.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {window.content.trades.trades.slice(0, 10).map((trade: any, idx: number) => (
                      <div key={idx} style={{
                        padding: '8px',
                        backgroundColor: 'var(--bg-tertiary)',
                        borderRadius: '4px',
                        fontSize: '10px'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <span style={{ fontWeight: 'bold' }}>
                            {trade.symbol} {trade.side.toUpperCase()}
                          </span>
                          <span style={{ 
                            color: trade.status === 'closed' ? 
                              (trade.pnl >= 0 ? 'var(--success-500)' : 'var(--danger-500)') : 'var(--warning-500)'
                          }}>
                            {trade.status === 'closed' ? formatCurrency(trade.pnl) : trade.status.toUpperCase()}
                          </span>
                        </div>
                        <div style={{ color: '#999', marginTop: '2px' }}>
                          {trade.quantity} @ {formatCurrency(trade.entry_price)} • {new Date(trade.entry_time).toLocaleDateString()}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', padding: '20px' }}>
                    No trade history available
                  </div>
                )}
              </div>
            </div>
          </div>
        );

      case 'BROK':
        return (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '10px',
              borderBottom: '1px solid #333',
              backgroundColor: 'var(--bg-secondary)'
            }}>
              <span style={{ color: 'var(--success-500)', fontWeight: 'bold', fontSize: '14px' }}>
                TRADING INTERFACE
              </span>
              <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
                Bitget Integration
              </span>
            </div>

            <div style={{ flex: 1, overflow: 'auto', padding: '15px' }}>
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Quick Trade (Demo Mode)
                </div>
                <div style={{
                  padding: '15px',
                  backgroundColor: 'var(--bg-tertiary)',
                  borderRadius: '4px',
                  marginBottom: '15px'
                }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '10px' }}>
                    <button style={{
                      padding: '8px',
                      backgroundColor: 'var(--success-500)',
                      color: '#000',
                      border: 'none',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 'bold',
                      cursor: 'pointer'
                    }}>
                      BUY BTCUSDT
                    </button>
                    <button style={{
                      padding: '8px',
                      backgroundColor: 'var(--danger-500)',
                      color: 'var(--text-primary)',
                      border: 'none',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 'bold',
                      cursor: 'pointer'
                    }}>
                      SELL BTCUSDT
                    </button>
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', textAlign: 'center' }}>
                    Connect API for live trading
                  </div>
                </div>
              </div>

              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Demo Positions
                </div>
                <div style={{
                  padding: '10px',
                  backgroundColor: 'var(--bg-tertiary)',
                  borderRadius: '4px',
                  borderLeft: '3px solid var(--success-500)'
                }}>
                  <div style={{ fontSize: '11px', fontWeight: 'bold', marginBottom: '4px' }}>
                    BTCUSDT LONG
                  </div>
                  <div style={{ fontSize: '10px', color: '#999' }}>
                    Size: 0.1 BTC | Entry: $43,500 | Current: $44,000
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--success-500)', marginTop: '4px' }}>
                    PnL: +$50.00 (+1.15%)
                  </div>
                </div>
              </div>

              <div>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Account Balance
                </div>
                <div style={{
                  padding: '10px',
                  backgroundColor: 'var(--bg-tertiary)',
                  borderRadius: '4px'
                }}>
                  <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '8px' }}>
                    $10,000.00 USDT
                  </div>
                  <div style={{ fontSize: '10px', color: '#999' }}>
                    Available: $8,750.00 | Used: $1,250.00
                  </div>
                </div>
              </div>
            </div>
          </div>
        );

      case 'TOP':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Top Reuters News</h3>
            {window.content?.map((news: any, idx: number) => (
              <div key={idx} style={{ 
                marginBottom: '10px', 
                padding: '8px', 
                borderLeft: '3px solid ' + (news.priority === 'High' ? 'var(--danger-500)' : 'var(--primary-500)'),
                backgroundColor: 'var(--bg-secondary)'
              }}>
                <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{news.headline}</div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  {news.time} - {news.source} - Priority: {news.priority}
                </div>
              </div>
            ))}
          </div>
        );

      case 'GR':
        return (
          <div style={{ padding: '10px', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
            <div style={{ fontSize: '48px', marginBottom: '20px' }}>📊</div>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: '14px', fontWeight: 'bold', marginBottom: '10px' }}>
                Correlation: {window.content?.pair}
              </div>
              <div style={{ fontSize: '24px', color: 'var(--success-500)', fontWeight: 'bold', marginBottom: '10px' }}>
                {window.content?.correlation}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Timeframe: {window.content?.timeframe} | Data Points: {window.content?.data_points}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginTop: '5px' }}>
                R²: {window.content?.r_squared}
              </div>
            </div>
          </div>
        );

      case 'HPS':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Historical Performance Comparison</h3>
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Symbol</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>1Y Return</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>3Y Return</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>5Y Return</th>
                </tr>
              </thead>
              <tbody>
                {window.content?.returns?.map((perf: any, idx: number) => (
                  <tr key={idx}>
                    <td style={{ padding: '4px', color: 'var(--success-500)' }}>{perf.symbol}</td>
                    <td style={{ padding: '4px', textAlign: 'right', color: perf.return_1y.includes('+') ? 'var(--success-500)' : 'var(--danger-500)' }}>
                      {perf.return_1y}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: perf.return_3y.includes('+') ? 'var(--success-500)' : 'var(--danger-500)' }}>
                      {perf.return_3y}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: perf.return_5y.includes('+') ? 'var(--success-500)' : 'var(--danger-500)' }}>
                      {perf.return_5y}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'TAS':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Time & Sales Data</h3>
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Time</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Price</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Size</th>
                  <th style={{ textAlign: 'center', padding: '4px' }}>Side</th>
                </tr>
              </thead>
              <tbody>
                {window.content?.map((trade: any, idx: number) => (
                  <tr key={idx}>
                    <td style={{ padding: '4px' }}>{trade.time}</td>
                    <td style={{ padding: '4px', textAlign: 'right' }}>{trade.price}</td>
                    <td style={{ padding: '4px', textAlign: 'right' }}>{trade.size}</td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'center',
                      color: trade.side === 'BUY' ? 'var(--success-500)' : 'var(--danger-500)'
                    }}>
                      {trade.side}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'RSS':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Research Reports</h3>
            {window.content?.map((report: any, idx: number) => (
              <div key={idx} style={{ 
                marginBottom: '12px', 
                padding: '10px', 
                borderLeft: '3px solid var(--primary-500)',
                backgroundColor: 'var(--bg-secondary)'
              }}>
                <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{report.title}</div>
                <div style={{ fontSize: '11px', color: '#999', marginTop: '4px' }}>
                  {report.analyst} • {report.date}
                </div>
                <div style={{ 
                  fontSize: '11px', 
                  marginTop: '4px',
                  color: report.rating === 'BUY' ? 'var(--success-500)' : report.rating === 'SELL' ? 'var(--danger-500)' : 'var(--warning-500)'
                }}>
                  Rating: {report.rating}
                </div>
              </div>
            ))}
          </div>
        );

      case 'XPRT':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Expert Reports</h3>
            {window.content?.map((expert: any, idx: number) => (
              <div key={idx} style={{ 
                marginBottom: '12px', 
                padding: '10px', 
                borderLeft: '3px solid var(--warning-500)',
                backgroundColor: 'var(--bg-secondary)'
              }}>
                <div style={{ fontSize: '12px', fontWeight: 'bold', color: 'var(--warning-500)' }}>{expert.expert}</div>
                <div style={{ fontSize: '11px', color: 'var(--text-primary)', marginTop: '4px' }}>Topic: {expert.topic}</div>
                <div style={{ fontSize: '11px', color: '#999', marginTop: '4px' }}>
                  Confidence: {expert.confidence}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--primary-500)', marginTop: '6px' }}>
                  "{expert.recommendation}"
                </div>
              </div>
            ))}
          </div>
        );

      case 'MET':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Global Market Indices</h3>
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Index</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Value</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Change</th>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Country</th>
                </tr>
              </thead>
              <tbody>
                {window.content?.map((index: any, idx: number) => (
                  <tr key={idx}>
                    <td style={{ padding: '4px', color: 'var(--success-500)' }}>{index.index}</td>
                    <td style={{ padding: '4px', textAlign: 'right' }}>{index.value}</td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: index.change.includes('+') ? 'var(--success-500)' : 'var(--danger-500)'
                    }}>
                      {index.change}
                    </td>
                    <td style={{ padding: '4px', color: '#999' }}>{index.country}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'ORG':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Organization Billing</h3>
            <div style={{ marginBottom: '20px' }}>
              <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>Current Plan:</div>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: 'var(--primary-500)' }}>{window.content?.plan}</div>
              <div style={{ fontSize: '14px', color: 'var(--success-500)', marginTop: '4px' }}>{window.content?.monthly_cost}/month</div>
            </div>
            <div style={{ marginBottom: '20px' }}>
              <div style={{ fontSize: '12px', color: '#999', marginBottom: '8px' }}>Usage This Month:</div>
              <div style={{ fontSize: '11px', marginBottom: '4px' }}>
                API Calls: <span style={{ color: 'var(--primary-500)' }}>{window.content?.usage?.api_calls}</span>
              </div>
              <div style={{ fontSize: '11px', marginBottom: '4px' }}>
                Data Feeds: <span style={{ color: 'var(--primary-500)' }}>{window.content?.usage?.data_feeds}</span>
              </div>
              <div style={{ fontSize: '11px', marginBottom: '4px' }}>
                Users: <span style={{ color: 'var(--primary-500)' }}>{window.content?.usage?.users}</span>
              </div>
            </div>
            <div>
              <div style={{ fontSize: '12px', color: '#999' }}>Next Billing Date:</div>
              <div style={{ fontSize: '12px', color: 'var(--warning-500)' }}>{window.content?.next_billing}</div>
            </div>
          </div>
        );

      case 'ACCT':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Account Management</h3>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: '#999' }}>User:</div>
              <div style={{ fontSize: '14px', fontWeight: 'bold', color: 'var(--primary-500)' }}>{window.content?.user}</div>
            </div>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: '#999' }}>Email:</div>
              <div style={{ fontSize: '12px' }}>{window.content?.email}</div>
            </div>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: '#999' }}>Plan:</div>
              <div style={{ fontSize: '12px', color: 'var(--success-500)' }}>{window.content?.plan}</div>
            </div>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: '#999' }}>Member Since:</div>
              <div style={{ fontSize: '12px' }}>{window.content?.member_since}</div>
            </div>
            <div>
              <div style={{ fontSize: '12px', color: '#999', marginBottom: '8px' }}>Permissions:</div>
              {window.content?.permissions?.map((perm: string, idx: number) => (
                <div key={idx} style={{ fontSize: '11px', color: 'var(--success-500)', marginBottom: '4px' }}>
                  • {perm}
                </div>
              ))}
            </div>
          </div>
        );

      case 'PDF':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>User Settings</h3>
            {Object.entries(window.content?.preferences || {}).map(([key, value]) => (
              <div key={key} style={{ marginBottom: '12px' }}>
                <div style={{ fontSize: '12px', color: '#999', marginBottom: '4px' }}>
                  {key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}:
                </div>
                <div style={{ fontSize: '12px', color: 'var(--primary-500)' }}>{value as string}</div>
              </div>
            ))}
          </div>
        );

      case 'CHANGE':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>QuantDesk Terminal Changelog</h3>
            {window.content?.map((version: any, idx: number) => (
              <div key={idx} style={{ 
                marginBottom: '15px', 
                padding: '10px', 
                borderLeft: '3px solid var(--success-500)',
                backgroundColor: 'var(--bg-secondary)'
              }}>
                <div style={{ fontSize: '12px', fontWeight: 'bold', color: 'var(--success-500)' }}>
                  {version.version} - {version.date}
                </div>
                <div style={{ marginTop: '8px' }}>
                  {version.changes.map((change: string, changeIdx: number) => (
                    <div key={changeIdx} style={{ fontSize: '11px', color: '#999', marginBottom: '4px' }}>
                      • {change}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        );

      case 'CITADEL':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Citadel Overview</h3>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: 'var(--primary-500)' }}>{window.content?.company}</div>
              <div style={{ fontSize: '12px', color: '#999', marginTop: '4px' }}>
                Founded: {window.content?.founded} • Founder: {window.content?.founder}
              </div>
            </div>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: '#999' }}>Assets Under Management:</div>
              <div style={{ fontSize: '18px', fontWeight: 'bold', color: 'var(--success-500)' }}>{window.content?.aum}</div>
            </div>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: '#999', marginBottom: '8px' }}>Strategies:</div>
              {window.content?.strategies?.map((strategy: string, idx: number) => (
                <div key={idx} style={{ fontSize: '11px', color: 'var(--primary-500)', marginBottom: '4px' }}>
                  • {strategy}
                </div>
              ))}
            </div>
            <div>
              <div style={{ fontSize: '12px', color: '#999' }}>Performance:</div>
              <div style={{ fontSize: '12px', color: 'var(--warning-500)' }}>{window.content?.performance}</div>
            </div>
          </div>
        );

      case 'MARTIN':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Files & Links</h3>
            {window.content?.map((file: any, idx: number) => (
              <div key={idx} style={{ 
                marginBottom: '10px', 
                padding: '8px', 
                borderLeft: '3px solid var(--primary-500)',
                backgroundColor: 'var(--bg-secondary)',
                cursor: 'pointer'
              }}
              onClick={() => file.url && window.open(file.url, '_blank')}
              >
                <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{file.name}</div>
                <div style={{ fontSize: '11px', color: '#999', marginTop: '4px' }}>
                  {file.type} {file.size && `• ${file.size}`}
                </div>
              </div>
            ))}
          </div>
        );

      case 'NEJM':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Medical Journal Research</h3>
            {window.content?.map((article: any, idx: number) => (
              <div key={idx} style={{ 
                marginBottom: '12px', 
                padding: '10px', 
                borderLeft: '3px solid var(--warning-500)',
                backgroundColor: 'var(--bg-secondary)'
              }}>
                <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{article.title}</div>
                <div style={{ fontSize: '11px', color: '#999', marginTop: '4px' }}>
                  Date: {article.date}
                </div>
                <div style={{ fontSize: '11px', color: 'var(--primary-500)', marginTop: '6px' }}>
                  Relevance: {article.relevance}
                </div>
              </div>
            ))}
          </div>
        );

      case 'GH':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>GitHub News Feed</h3>
            {window.content?.map((event: any, idx: number) => (
              <div key={idx} style={{ 
                marginBottom: '10px', 
                padding: '8px', 
                borderLeft: '3px solid var(--success-500)',
                backgroundColor: 'var(--bg-secondary)'
              }}>
                <div style={{ fontSize: '12px', fontWeight: 'bold', color: 'var(--success-500)' }}>{event.repo}</div>
                <div style={{ fontSize: '11px', color: 'var(--text-primary)', marginTop: '4px' }}>{event.event}</div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>{event.time}</div>
              </div>
            ))}
          </div>
        );

      case 'NH':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>News Hub</h3>
            {window.content?.map((news: any, idx: number) => (
              <div key={idx} style={{ 
                marginBottom: '10px', 
                padding: '8px', 
                borderLeft: '3px solid var(--primary-500)',
                backgroundColor: 'var(--bg-secondary)'
              }}>
                <div style={{ fontSize: '11px', color: 'var(--warning-500)', fontWeight: 'bold' }}>[{news.category}]</div>
                <div style={{ fontSize: '12px', fontWeight: 'bold', marginTop: '4px' }}>{news.headline}</div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  {news.source} • {news.time}
                </div>
              </div>
            ))}
          </div>
        );

      case 'NT':
        return (
          <div style={{ padding: '10px', height: '100%', display: 'flex', flexDirection: 'column' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>News Search</h3>
            <input 
              placeholder={window.content?.search_term || 'Enter search term...'}
              style={{ 
                width: '100%', 
                padding: '8px', 
                background: '#000', 
                border: '1px solid #333', 
                color: 'var(--text-primary)',
                fontSize: '12px',
                marginBottom: '15px'
              }}
            />
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: '#999', marginBottom: '8px' }}>Recent Searches:</div>
              {window.content?.recent_searches?.map((search: string, idx: number) => (
                <div key={idx} style={{ 
                  fontSize: '11px', 
                  color: 'var(--primary-500)', 
                  marginBottom: '4px',
                  cursor: 'pointer'
                }}
                onClick={() => alert(`Searching for: ${search}`)}
                >
                  • {search}
                </div>
              ))}
            </div>
          </div>
        );

      case 'BUG':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Bug Report & Support</h3>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: '#999' }}>Contact Support:</div>
              <div style={{ fontSize: '12px', color: 'var(--primary-500)' }}>{window.content?.contact}</div>
            </div>
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: '#999', marginBottom: '8px' }}>Common Issues:</div>
              {window.content?.common_issues?.map((issue: string, idx: number) => (
                <div key={idx} style={{ fontSize: '11px', color: 'var(--warning-500)', marginBottom: '4px' }}>
                  • {issue}
                </div>
              ))}
            </div>
            <div>
              <div style={{ fontSize: '12px', color: '#999', marginBottom: '8px' }}>System Info:</div>
              <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                Version: {window.content?.system_info?.version}<br/>
                Browser: {window.content?.system_info?.browser}<br/>
                Platform: {window.content?.system_info?.platform}
              </div>
            </div>
          </div>
        );

      case 'HELP':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Available Commands:</h3>
            {backtickCommands.map((cmd, idx) => (
              <div key={idx} style={{ marginBottom: '8px', fontSize: '11px' }}>
                <span style={{ color: 'var(--success-500)', fontWeight: 'bold', minWidth: '60px', display: 'inline-block' }}>
                  {cmd.command}
                </span>
                <span style={{ color: '#999' }}> - {cmd.description}</span>
              </div>
            ))}
          </div>
        );
      case 'AL':
        return (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '10px',
              borderBottom: '1px solid #333',
              backgroundColor: 'var(--bg-secondary)'
            }}>
              <span style={{ color: 'var(--success-500)', fontWeight: 'bold', fontSize: '14px' }}>
                PRICE ALERTS & NOTIFICATIONS
              </span>
              <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
                {window.content?.summary?.active_alerts || 0} active alerts
              </span>
            </div>

            <div style={{ flex: 1, overflow: 'auto', padding: '15px' }}>
              {/* Quick Alert Creation */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Quick Alerts
                </div>
                <div style={{
                  padding: '15px',
                  backgroundColor: 'var(--bg-tertiary)',
                  borderRadius: '4px',
                  marginBottom: '15px'
                }}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr auto auto', gap: '10px', marginBottom: '10px' }}>
                    <select style={{
                      padding: '6px',
                      backgroundColor: '#333',
                      color: 'var(--text-primary)',
                      border: '1px solid #555',
                      borderRadius: '2px',
                      fontSize: '11px'
                    }}>
                      <option>BTCUSDT</option>
                      <option>ETHUSDT</option>
                      <option>SOLUSDT</option>
                      <option>ADAUSDT</option>
                    </select>
                    <button style={{
                      padding: '6px 12px',
                      backgroundColor: 'var(--success-500)',
                      color: '#000',
                      border: 'none',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 'bold',
                      cursor: 'pointer'
                    }}>
                      Quick Alerts
                    </button>
                    <button style={{
                      padding: '6px 12px',
                      backgroundColor: 'var(--primary-500)',
                      color: 'var(--text-primary)',
                      border: 'none',
                      borderRadius: '4px',
                      fontSize: '11px',
                      fontWeight: 'bold',
                      cursor: 'pointer'
                    }}>
                      Custom Alert
                    </button>
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                    Quick Alerts: ±5%, ±10% price levels | Custom: Set your own conditions
                  </div>
                </div>
              </div>

              {/* Active Alerts */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Active Alerts ({window.content?.alerts?.filter((a: any) => a.status === 'active').length || 0})
                </div>
                {window.content?.alerts?.filter((a: any) => a.status === 'active').length > 0 ? 
                  window.content.alerts.filter((a: any) => a.status === 'active').map((alert: any, idx: number) => (
                    <div key={idx} style={{
                      padding: '10px',
                      backgroundColor: 'var(--bg-tertiary)',
                      borderRadius: '4px',
                      marginBottom: '8px',
                      borderLeft: '3px solid var(--success-500)'
                    }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                        <span style={{ fontSize: '11px', fontWeight: 'bold' }}>
                          {alert.symbol} {alert.alert_type.replace('_', ' ').toUpperCase()}
                        </span>
                        <button style={{
                          padding: '2px 6px',
                          backgroundColor: 'var(--danger-500)',
                          color: 'var(--text-primary)',
                          border: 'none',
                          borderRadius: '2px',
                          fontSize: '10px',
                          cursor: 'pointer'
                        }}>
                          Delete
                        </button>
                      </div>
                      <div style={{ fontSize: '10px', color: '#999' }}>
                        Condition: {alert.condition_operator} ${alert.condition_value?.toFixed(4)}
                      </div>
                      <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                        Created: {new Date(alert.created_at).toLocaleDateString()}
                      </div>
                    </div>
                  )) : (
                    <div style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', padding: '20px' }}>
                      No active alerts
                    </div>
                  )
                }
              </div>

              {/* Recent Notifications */}
              <div>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Recent Notifications ({window.content?.notifications?.length || 0})
                  {window.content?.notifications?.filter((n: any) => !n.read).length > 0 && (
                    <span style={{ 
                      marginLeft: '10px',
                      fontSize: '10px',
                      padding: '2px 6px',
                      backgroundColor: 'var(--danger-500)',
                      borderRadius: '10px'
                    }}>
                      {window.content.notifications.filter((n: any) => !n.read).length} unread
                    </span>
                  )}
                </div>
                {window.content?.notifications?.length > 0 ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                    {window.content.notifications.slice(0, 10).map((notification: any, idx: number) => (
                      <div key={idx} style={{
                        padding: '8px',
                        backgroundColor: notification.read ? 'var(--bg-tertiary)' : '#1f3a3a',
                        borderRadius: '4px',
                        fontSize: '10px',
                        borderLeft: notification.read ? '3px solid var(--text-muted)' : '3px solid var(--success-500)'
                      }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                          <span style={{ 
                            fontWeight: 'bold',
                            color: notification.read ? '#999' : 'var(--text-primary)'
                          }}>
                            {notification.title}
                          </span>
                          <span style={{ color: 'var(--text-muted)', fontSize: '9px' }}>
                            {new Date(notification.timestamp).toLocaleTimeString()}
                          </span>
                        </div>
                        <div style={{ 
                          color: notification.read ? 'var(--text-muted)' : '#999',
                          marginBottom: '4px'
                        }}>
                          {notification.message}
                        </div>
                        {!notification.read && (
                          <button style={{
                            padding: '2px 6px',
                            backgroundColor: 'var(--primary-500)',
                            color: 'var(--text-primary)',
                            border: 'none',
                            borderRadius: '2px',
                            fontSize: '10px',
                            cursor: 'pointer'
                          }}>
                            Mark Read
                          </button>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', textAlign: 'center', padding: '20px' }}>
                    No notifications
                  </div>
                )}
              </div>
            </div>
          </div>
        );

      case 'SHORTCUTS':
        return (
          <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '10px',
              borderBottom: '1px solid #333',
              backgroundColor: 'var(--bg-secondary)'
            }}>
              <span style={{ color: 'var(--success-500)', fontWeight: 'bold', fontSize: '14px' }}>
                KEYBOARD SHORTCUTS
              </span>
              <span style={{ color: 'var(--text-muted)', fontSize: '11px' }}>
                Professional Trading Hotkeys
              </span>
            </div>

            <div style={{ flex: 1, overflow: 'auto', padding: '15px' }}>
              {/* Quick Commands */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Quick Commands (Ctrl + Key)
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '8px' }}>
                  {[
                    { key: 'Ctrl + Q', desc: 'Quote Monitor' },
                    { key: 'Ctrl + P', desc: 'Portfolio Performance' },
                    { key: 'Ctrl + B', desc: 'Trading Interface' },
                    { key: 'Ctrl + T', desc: 'Strategy Backtesting' },
                    { key: 'Ctrl + C', desc: 'Advanced Charts' },
                    { key: 'Ctrl + N', desc: 'News Feed' },
                    { key: 'Ctrl + H', desc: 'Help & Commands' },
                    { key: 'Ctrl + R', desc: 'Refresh Data' },
                    { key: 'Ctrl + W', desc: 'Close Active Window' }
                  ].map((shortcut, idx) => (
                    <div key={idx} style={{
                      padding: '8px',
                      backgroundColor: 'var(--bg-tertiary)',
                      borderRadius: '4px',
                      fontSize: '11px'
                    }}>
                      <div style={{ color: 'var(--success-500)', fontWeight: 'bold', marginBottom: '2px' }}>
                        {shortcut.key}
                      </div>
                      <div style={{ color: '#999' }}>
                        {shortcut.desc}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Window Management */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Window Management (Alt + Key)
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '8px' }}>
                  {[
                    { key: 'Alt + 1-9', desc: 'Focus Window by Number' },
                    { key: 'Alt + F', desc: 'Fit All Windows' },
                    { key: 'Alt + L', desc: 'Reset Layout' },
                    { key: 'Alt + M', desc: 'Toggle Lite/Pro Mode' }
                  ].map((shortcut, idx) => (
                    <div key={idx} style={{
                      padding: '8px',
                      backgroundColor: 'var(--bg-tertiary)',
                      borderRadius: '4px',
                      fontSize: '11px'
                    }}>
                      <div style={{ color: 'var(--primary-500)', fontWeight: 'bold', marginBottom: '2px' }}>
                        {shortcut.key}
                      </div>
                      <div style={{ color: '#999' }}>
                        {shortcut.desc}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Function Keys */}
              <div style={{ marginBottom: '20px' }}>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Function Keys
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '8px' }}>
                  {[
                    { key: 'F1', desc: 'Help' },
                    { key: 'F2', desc: 'Quotes' },
                    { key: 'F3', desc: 'Portfolio' },
                    { key: 'F4', desc: 'Trading' },
                    { key: 'F5', desc: 'Charts' },
                    { key: 'F6', desc: 'Backtest' },
                    { key: 'F9', desc: 'Quick Buy' },
                    { key: 'F10', desc: 'Quick Sell' },
                    { key: 'F11', desc: 'Fullscreen' }
                  ].map((shortcut, idx) => (
                    <div key={idx} style={{
                      padding: '6px',
                      backgroundColor: 'var(--bg-tertiary)',
                      borderRadius: '4px',
                      fontSize: '10px',
                      textAlign: 'center'
                    }}>
                      <div style={{ color: 'var(--danger-500)', fontWeight: 'bold', marginBottom: '2px' }}>
                        {shortcut.key}
                      </div>
                      <div style={{ color: '#999' }}>
                        {shortcut.desc}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Terminal Commands */}
              <div>
                <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px' }}>
                  Terminal Commands
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '8px' }}>
                  {[
                    { key: '` (Backtick)', desc: 'Open Command Menu' },
                    { key: 'ESC', desc: 'Close Command Menu' },
                    { key: '↑ ↓ Arrows', desc: 'Navigate Commands' },
                    { key: 'Enter', desc: 'Execute Command' }
                  ].map((shortcut, idx) => (
                    <div key={idx} style={{
                      padding: '8px',
                      backgroundColor: 'var(--bg-tertiary)',
                      borderRadius: '4px',
                      fontSize: '11px'
                    }}>
                      <div style={{ color: '#722ed1', fontWeight: 'bold', marginBottom: '2px' }}>
                        {shortcut.key}
                      </div>
                      <div style={{ color: '#999' }}>
                        {shortcut.desc}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        );

      case 'ORDER':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Place Trading Order</h3>
            
            {/* Trading Pair */}
            <div style={{ marginBottom: '15px' }}>
              <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '5px' }}>Trading Pair</label>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: 'var(--success-500)' }}>
                {window.content?.pair || 'BTC/USDT'}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                Current Price: ${window.content?.currentPrice || '67,543.21'}
              </div>
            </div>
            
            {/* Order Form */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '15px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '5px' }}>Side</label>
                <select style={{ 
                  width: '100%', 
                  padding: '8px', 
                  backgroundColor: 'var(--bg-secondary)', 
                  border: '1px solid var(--primary-500)', 
                  borderRadius: '4px',
                  color: 'white'
                }}>
                  <option value="BUY">BUY</option>
                  <option value="SELL">SELL</option>
                </select>
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '5px' }}>Order Type</label>
                <select style={{ 
                  width: '100%', 
                  padding: '8px', 
                  backgroundColor: 'var(--bg-secondary)', 
                  border: '1px solid var(--primary-500)', 
                  borderRadius: '4px',
                  color: 'white'
                }}>
                  <option value="LIMIT">LIMIT</option>
                  <option value="MARKET">MARKET</option>
                </select>
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px', marginBottom: '15px' }}>
              <div>
                <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '5px' }}>Amount</label>
                <input 
                  type="text" 
                  defaultValue={window.content?.amount || '0.1'}
                  style={{ 
                    width: '100%', 
                    padding: '8px', 
                    backgroundColor: 'var(--bg-secondary)', 
                    border: '1px solid var(--primary-500)', 
                    borderRadius: '4px',
                    color: 'white'
                  }} 
                />
              </div>
              <div>
                <label style={{ display: 'block', fontSize: '12px', color: 'var(--text-muted)', marginBottom: '5px' }}>Price</label>
                <input 
                  type="text" 
                  defaultValue={window.content?.price || '67543.21'}
                  style={{ 
                    width: '100%', 
                    padding: '8px', 
                    backgroundColor: 'var(--bg-secondary)', 
                    border: '1px solid var(--primary-500)', 
                    borderRadius: '4px',
                    color: 'white'
                  }} 
                />
              </div>
            </div>
            
            {/* Order Summary */}
            <div style={{ marginBottom: '15px', padding: '10px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '5px' }}>Order Summary</div>
              <div style={{ fontSize: '14px', fontWeight: 'bold' }}>
                Total Value: ${window.content?.totalValue || '6,754.32'}
              </div>
            </div>
            
            {/* Available Balance */}
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '5px' }}>Available Balance</div>
              <div style={{ fontSize: '11px' }}>
                <div>USDT: ${window.content?.availableBalance?.USDT || '15,000.00'}</div>
                <div>BTC: {window.content?.availableBalance?.BTC || '0.25'}</div>
              </div>
            </div>
            
            {/* Place Order Button */}
            <button style={{ 
              width: '100%', 
              padding: '12px', 
              backgroundColor: 'var(--primary-500)', 
              border: 'none', 
              borderRadius: '4px',
              color: 'white',
              fontSize: '14px',
              fontWeight: 'bold',
              cursor: 'pointer'
            }}>
              Place Order
            </button>
          </div>
        );
      
      case 'VOLUME':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Trading Volume Analysis</h3>
            
            {/* Volume Summary */}
            <div style={{ marginBottom: '15px', padding: '10px', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '5px' }}>24h Volume Summary</div>
              <div style={{ fontSize: '16px', fontWeight: 'bold', color: 'var(--success-500)' }}>
                {window.content?.volumeProfile?.totalVolume24h || '6.1B'}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--success-500)' }}>
                Change: {window.content?.volumeProfile?.volumeChange || '+12.8%'}
              </div>
            </div>
            
            {/* Top Pairs */}
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>Top Trading Pairs</div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
                {(window.content?.topPairs || []).map((pair: any, idx: number) => (
                  <div key={idx} style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    alignItems: 'center',
                    padding: '8px', 
                    backgroundColor: 'var(--bg-secondary)', 
                    borderRadius: '4px'
                  }}>
                    <div style={{ fontSize: '12px', fontWeight: 'bold', color: 'var(--success-500)' }}>
                      {pair.pair}
                    </div>
                    <div style={{ fontSize: '11px', color: 'var(--text-primary)' }}>
                      {pair.volume24h}
                    </div>
                    <div style={{ 
                      fontSize: '11px', 
                      color: pair.change.includes('+') ? 'var(--success-500)' : 'var(--danger-500)'
                    }}>
                      {pair.change}
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Volume Distribution */}
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>Volume Distribution</div>
              <div style={{ fontSize: '11px' }}>
                <div style={{ marginBottom: '4px' }}>
                  Spot: {window.content?.volumeProfile?.volumeDistribution?.spot || '4.2B (68.9%)'}
                </div>
                <div>
                  Futures: {window.content?.volumeProfile?.volumeDistribution?.futures || '1.9B (31.1%)'}
                </div>
              </div>
            </div>
            
            {/* Active Pairs */}
            <div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>Market Activity</div>
              <div style={{ fontSize: '11px' }}>
                <div>Active Pairs: {window.content?.volumeProfile?.activePairs || '156'}</div>
                <div>Trend: {window.content?.volumeProfile?.trend || 'Increasing across major pairs'}</div>
              </div>
            </div>
          </div>
        );

      case 'POSITIONS':
        const positionsData = window.content?.positions || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Current Positions</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Total Positions:</span>
                <span style={{ color: 'var(--success-500)' }}>{positionsData.length}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Total Value:</span>
                <span style={{ color: 'var(--primary-500)' }}>${window.content?.totalValue || '0.00'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Total P&L:</span>
                <span style={{ color: (Number(window.content?.totalPnL) || 0) >= 0 ? 'var(--success-500)' : 'var(--danger-500)' }}>
                  ${(Number(window.content?.totalPnL) || 0).toFixed(2)}
                </span>
              </div>
            </div>
            
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Pair</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Side</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Amount</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Entry Price</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>P&L</th>
                </tr>
              </thead>
              <tbody>
                {positionsData.map((position: any, idx: number) => (
                  <tr key={idx} style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                    <td style={{ padding: '4px', color: 'var(--success-500)', fontWeight: 'bold' }}>
                      {position.pair}
                    </td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: position.side === 'BUY' ? 'var(--success-500)' : 'var(--danger-500)',
                      fontWeight: 'bold'
                    }}>
                      {position.side}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--text-primary)' }}>
                      {position.amount}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--text-primary)' }}>
                      ${Number(position.entryPrice) ? Number(position.entryPrice).toFixed(2) : 'N/A'}
                    </td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: (Number(position.pnl) || 0) >= 0 ? 'var(--success-500)' : 'var(--danger-500)',
                      fontWeight: 'bold'
                    }}>
                      ${(Number(position.pnl) || 0).toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'FEAR':
        const fearData = window.content;
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Fear & Greed Index</h3>
            <div style={{ textAlign: 'center', marginBottom: '20px' }}>
              <div style={{ 
                fontSize: '48px', 
                fontWeight: 'bold',
                color: fearData.current >= 75 ? 'var(--danger-500)' : 
                       fearData.current >= 50 ? 'var(--warning-500)' : 'var(--success-500)'
              }}>
                {fearData.current}
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                {fearData.current >= 75 ? 'Extreme Greed' :
                 fearData.current >= 50 ? 'Neutral' : 'Fear'}
              </div>
            </div>
            
            <div style={{ marginBottom: '15px' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>Historical Data</div>
              <div style={{ fontSize: '11px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span>7 Days Ago:</span>
                  <span style={{ color: 'var(--primary-500)' }}>{fearData.historical?.sevenDaysAgo || 'N/A'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span>30 Days Ago:</span>
                  <span style={{ color: 'var(--primary-500)' }}>{fearData.historical?.thirtyDaysAgo || 'N/A'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>90 Days Ago:</span>
                  <span style={{ color: 'var(--primary-500)' }}>{fearData.historical?.ninetyDaysAgo || 'N/A'}</span>
                </div>
              </div>
            </div>
            
            <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              <div style={{ marginBottom: '5px' }}>Market Sentiment:</div>
              <div>{fearData.description || 'Market showing mixed signals'}</div>
            </div>
          </div>
        );

      case 'DEFI':
        const defiData = window.content?.protocols || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>DeFi Protocols & Yield Farming</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Total TVL:</span>
                <span style={{ color: 'var(--success-500)' }}>${window.content?.totalTVL || '45.2B'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Active Protocols:</span>
                <span style={{ color: 'var(--primary-500)' }}>{defiData.length}</span>
              </div>
            </div>
            
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Protocol</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>TVL</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>APY</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Chain</th>
                </tr>
              </thead>
              <tbody>
                {defiData.map((protocol: any, idx: number) => (
                  <tr key={idx} style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                    <td style={{ padding: '4px', color: 'var(--success-500)', fontWeight: 'bold' }}>
                      {protocol.name}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--text-primary)' }}>
                      ${protocol.tvl}
                    </td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: 'var(--success-500)',
                      fontWeight: 'bold'
                    }}>
                      {protocol.apy}%
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--primary-500)' }}>
                      {protocol.chain}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'WALLET':
        const walletData = window.content?.connectedWallets || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Wallet Connections & Balances</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Connected Wallets:</span>
                <span style={{ color: 'var(--success-500)' }}>{walletData.length}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Total Balance:</span>
                <span style={{ color: 'var(--primary-500)' }}>${window.content?.totalBalance || '0.00'}</span>
              </div>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {walletData.map((wallet: any, idx: number) => (
                <div key={idx} style={{ 
                  padding: '10px', 
                  backgroundColor: 'var(--bg-secondary)', 
                  borderRadius: '4px',
                  border: '1px solid var(--bg-tertiary)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
                    <span style={{ color: 'var(--success-500)', fontWeight: 'bold', fontSize: '12px' }}>
                      {wallet.name}
                    </span>
                    <span style={{ 
                      color: wallet.connected ? 'var(--success-500)' : 'var(--danger-500)',
                      fontSize: '10px'
                    }}>
                      {wallet.connected ? 'CONNECTED' : 'DISCONNECTED'}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                    <div>Address: {wallet.address?.slice(0, 8)}...{wallet.address?.slice(-6)}</div>
                    <div>Balance: ${Number(wallet.balance) ? Number(wallet.balance).toFixed(2) : '0.00'}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'GAS':
        const gasData = window.content;
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Gas Fee Tracker</h3>
            
            {/* Ethereum Gas */}
            <div style={{ marginBottom: '20px' }}>
              <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px', fontWeight: 'bold' }}>
                Ethereum Network
              </div>
              <div style={{ fontSize: '11px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Slow:</span>
                  <span style={{ color: 'var(--success-500)' }}>{gasData.ethereum?.slow || '15'} gwei</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Standard:</span>
                  <span style={{ color: 'var(--warning-500)' }}>{gasData.ethereum?.standard || '20'} gwei</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Fast:</span>
                  <span style={{ color: 'var(--danger-500)' }}>{gasData.ethereum?.fast || '25'} gwei</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>USD Cost:</span>
                  <span style={{ color: 'var(--primary-500)' }}>${gasData.ethereum?.usdCost || '2.50'}</span>
                </div>
              </div>
            </div>
            
            {/* Polygon Gas */}
            <div style={{ marginBottom: '20px' }}>
              <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px', fontWeight: 'bold' }}>
                Polygon Network
              </div>
              <div style={{ fontSize: '11px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Slow:</span>
                  <span style={{ color: 'var(--success-500)' }}>{gasData.polygon?.slow || '30'} gwei</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Standard:</span>
                  <span style={{ color: 'var(--warning-500)' }}>{gasData.polygon?.standard || '35'} gwei</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Fast:</span>
                  <span style={{ color: 'var(--danger-500)' }}>{gasData.polygon?.fast || '40'} gwei</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>USD Cost:</span>
                  <span style={{ color: 'var(--primary-500)' }}>${gasData.polygon?.usdCost || '0.01'}</span>
                </div>
              </div>
            </div>
            
            <div style={{ fontSize: '10px', color: 'var(--text-muted)', textAlign: 'center' }}>
              Last updated: {new Date().toLocaleTimeString()}
            </div>
          </div>
        );
      case 'STAKING':
        const stakingData = window.content?.protocols || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Staking Rewards Calculator</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Total Staked:</span>
                <span style={{ color: 'var(--success-500)' }}>${window.content?.totalStaked || '125.4M'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Average APY:</span>
                <span style={{ color: 'var(--primary-500)' }}>{window.content?.averageAPY || '5.2'}%</span>
              </div>
            </div>
            
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Token</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>APY</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Min Stake</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Lock Period</th>
                </tr>
              </thead>
              <tbody>
                {stakingData.map((protocol: any, idx: number) => (
                  <tr key={idx} style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                    <td style={{ padding: '4px', color: 'var(--success-500)', fontWeight: 'bold' }}>
                      {protocol.token}
                    </td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: 'var(--success-500)',
                      fontWeight: 'bold'
                    }}>
                      {protocol.apy}%
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--text-primary)' }}>
                      {protocol.minStake}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--primary-500)' }}>
                      {protocol.lockPeriod}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'ORDERBOOK':
        const orderbookData = window.content;
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Order Book - {orderbookData.pair}</h3>
            
            {/* Spread */}
            <div style={{ textAlign: 'center', marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ color: 'var(--text-muted)' }}>Spread</div>
              <div style={{ color: 'var(--warning-500)', fontWeight: 'bold' }}>
                ${Number(orderbookData.spread) ? Number(orderbookData.spread).toFixed(2) : '0.05'} ({orderbookData.spreadPercent || '0.01'}%)
              </div>
            </div>
            
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-muted)', marginBottom: '5px' }}>
              <span>Price (USDT)</span>
              <span>Amount ({orderbookData.pair?.split('/')[0]})</span>
            </div>
            
            {/* Asks (Sell Orders) */}
            <div style={{ marginBottom: '10px' }}>
              <div style={{ fontSize: '10px', color: 'var(--danger-500)', marginBottom: '5px', fontWeight: 'bold' }}>
                ASKS (Sell Orders)
              </div>
              <div style={{ display: 'flex', flexDirection: 'column-reverse', gap: '2px' }}>
                {(orderbookData.asks || []).slice(0, 8).map((ask: any, i: number) => (
                  <div key={`ask-${i}`} style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    backgroundColor: 'rgba(255,0,0,0.1)', 
                    padding: '2px 5px', 
                    borderRadius: '2px',
                    fontSize: '10px'
                  }}>
                    <span style={{ color: 'var(--danger-500)' }}>{Number(ask.price) ? Number(ask.price).toFixed(2) : 'N/A'}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{Number(ask.amount) ? Number(ask.amount).toFixed(4) : 'N/A'}</span>
                  </div>
                ))}
              </div>
            </div>
            
            {/* Bids (Buy Orders) */}
            <div>
              <div style={{ fontSize: '10px', color: 'var(--success-500)', marginBottom: '5px', fontWeight: 'bold' }}>
                BIDS (Buy Orders)
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '2px' }}>
                {(orderbookData.bids || []).slice(0, 8).map((bid: any, i: number) => (
                  <div key={`bid-${i}`} style={{ 
                    display: 'flex', 
                    justifyContent: 'space-between', 
                    backgroundColor: 'rgba(0,255,0,0.1)', 
                    padding: '2px 5px', 
                    borderRadius: '2px',
                    fontSize: '10px'
                  }}>
                    <span style={{ color: 'var(--success-500)' }}>{Number(bid.price) ? Number(bid.price).toFixed(2) : 'N/A'}</span>
                    <span style={{ color: 'var(--text-muted)' }}>{Number(bid.amount) ? Number(bid.amount).toFixed(4) : 'N/A'}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        );

      case 'HEATMAP':
        const heatmapData = window.content?.pairs || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Market Heatmap</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Total Pairs:</span>
                <span style={{ color: 'var(--success-500)' }}>{heatmapData.length}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Market Status:</span>
                <span style={{ color: 'var(--success-500)' }}>{window.content?.marketStatus || 'Bullish'}</span>
              </div>
            </div>
            
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: '8px' }}>
              {heatmapData.map((pair: any, idx: number) => (
                <div key={idx} style={{ 
                  padding: '8px', 
                  backgroundColor: 'var(--bg-secondary)', 
                  borderRadius: '4px',
                  textAlign: 'center',
                  border: '1px solid var(--bg-tertiary)'
                }}>
                  <div style={{ fontSize: '10px', color: 'var(--success-500)', fontWeight: 'bold', marginBottom: '4px' }}>
                    {pair.symbol}
                  </div>
                  <div style={{ 
                    fontSize: '11px', 
                    fontWeight: 'bold',
                    color: (pair.changePercent || 0) >= 0 ? 'var(--success-500)' : 'var(--danger-500)'
                  }}>
                    {(Number(pair.changePercent) || 0) >= 0 ? '+' : ''}{(Number(pair.changePercent) || 0).toFixed(2)}%
                  </div>
                  <div style={{ fontSize: '9px', color: 'var(--text-muted)', marginTop: '2px' }}>
                    ${Number(pair.price) ? Number(pair.price).toFixed(2) : 'N/A'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'STRATEGIES':
        const strategiesData = window.content?.strategies || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Trading Strategies</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Available Strategies:</span>
                <span style={{ color: 'var(--success-500)' }}>{strategiesData.length}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Active Backtests:</span>
                <span style={{ color: 'var(--primary-500)' }}>{window.content?.activeBacktests || '3'}</span>
              </div>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {strategiesData.map((strategy: any, idx: number) => (
                <div key={idx} style={{ 
                  padding: '10px', 
                  backgroundColor: 'var(--bg-secondary)', 
                  borderRadius: '4px',
                  border: '1px solid var(--bg-tertiary)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
                    <span style={{ color: 'var(--success-500)', fontWeight: 'bold', fontSize: '12px' }}>
                      {strategy.name}
                    </span>
                    <span style={{ 
                      color: strategy.status === 'Active' ? 'var(--success-500)' : 'var(--text-muted)',
                      fontSize: '10px'
                    }}>
                      {strategy.status}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '5px' }}>
                    {strategy.description}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px' }}>
                    <span style={{ color: 'var(--text-muted)' }}>Win Rate: <span style={{ color: 'var(--success-500)' }}>{strategy.winRate}%</span></span>
                    <span style={{ color: 'var(--text-muted)' }}>Profit: <span style={{ color: 'var(--success-500)' }}>{strategy.profit}%</span></span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'NOTIFICATIONS':
        const notificationsData = window.content?.notifications || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Notifications Center</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Unread:</span>
                <span style={{ color: 'var(--warning-500)' }}>{window.content?.unreadCount || '0'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Total:</span>
                <span style={{ color: 'var(--primary-500)' }}>{notificationsData.length}</span>
              </div>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {notificationsData.map((notification: any, idx: number) => (
                <div key={idx} style={{ 
                  padding: '8px', 
                  backgroundColor: notification.read ? 'var(--bg-secondary)' : 'var(--bg-tertiary)', 
                  borderRadius: '4px',
                  border: '1px solid var(--bg-tertiary)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
                    <span style={{ 
                      color: notification.type === 'alert' ? 'var(--warning-500)' : 
                             notification.type === 'trade' ? 'var(--success-500)' : 'var(--primary-500)',
                      fontWeight: 'bold', 
                      fontSize: '11px' 
                    }}>
                      {notification.type?.toUpperCase()}
                    </span>
                    <span style={{ fontSize: '9px', color: 'var(--text-muted)' }}>
                      {notification.time}
                    </span>
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-primary)' }}>
                    {notification.message}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'ANALYSIS':
        const analysisData = window.content?.indicators || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Technical Analysis Tools</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Active Indicators:</span>
                <span style={{ color: 'var(--success-500)' }}>{analysisData.length}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Signal:</span>
                <span style={{ color: window.content?.signal === 'BUY' ? 'var(--success-500)' : 
                               window.content?.signal === 'SELL' ? 'var(--danger-500)' : 'var(--warning-500)' }}>
                  {window.content?.signal || 'NEUTRAL'}
                </span>
              </div>
            </div>
            
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Indicator</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Value</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Signal</th>
                </tr>
              </thead>
              <tbody>
                {analysisData.map((indicator: any, idx: number) => (
                  <tr key={idx} style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                    <td style={{ padding: '4px', color: 'var(--success-500)', fontWeight: 'bold' }}>
                      {indicator.name}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--text-primary)' }}>
                      {indicator.value}
                    </td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: indicator.signal === 'BUY' ? 'var(--success-500)' : 
                             indicator.signal === 'SELL' ? 'var(--danger-500)' : 'var(--warning-500)',
                      fontWeight: 'bold'
                    }}>
                      {indicator.signal}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'OVERVIEW':
        const overviewData = window.content;
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Trading Dashboard</h3>
            
            {/* Key Metrics */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '10px', marginBottom: '20px' }}>
              <div style={{ padding: '10px', backgroundColor: 'var(--bg-secondary)', borderRadius: '4px', textAlign: 'center' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>Portfolio Value</div>
                <div style={{ fontSize: '16px', fontWeight: 'bold', color: 'var(--success-500)' }}>
                  ${overviewData?.portfolioValue || '0.00'}
                </div>
              </div>
              <div style={{ padding: '10px', backgroundColor: 'var(--bg-secondary)', borderRadius: '4px', textAlign: 'center' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>24h P&L</div>
                <div style={{ 
                  fontSize: '16px', 
                  fontWeight: 'bold', 
                  color: (overviewData?.pnl24h || 0) >= 0 ? 'var(--success-500)' : 'var(--danger-500)'
                }}>
                  ${(Number(overviewData?.pnl24h) || 0).toFixed(2)}
                </div>
              </div>
              <div style={{ padding: '10px', backgroundColor: 'var(--bg-secondary)', borderRadius: '4px', textAlign: 'center' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>Active Positions</div>
                <div style={{ fontSize: '16px', fontWeight: 'bold', color: 'var(--primary-500)' }}>
                  {overviewData?.activePositions || '0'}
                </div>
              </div>
              <div style={{ padding: '10px', backgroundColor: 'var(--bg-secondary)', borderRadius: '4px', textAlign: 'center' }}>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '4px' }}>Win Rate</div>
                <div style={{ fontSize: '16px', fontWeight: 'bold', color: 'var(--success-500)' }}>
                  {overviewData?.winRate || '0'}%
                </div>
              </div>
            </div>
            
            {/* Market Status */}
            <div style={{ marginBottom: '20px' }}>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>Market Status</div>
              <div style={{ fontSize: '11px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span>BTC Price:</span>
                  <span style={{ color: 'var(--success-500)' }}>${overviewData?.btcPrice || '67543.21'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span>Market Cap:</span>
                  <span style={{ color: 'var(--primary-500)' }}>${overviewData?.marketCap || '2.1T'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span>Fear & Greed:</span>
                  <span style={{ color: 'var(--warning-500)' }}>{overviewData?.fearGreed || '65'}</span>
                </div>
              </div>
            </div>
            
            {/* Quick Actions */}
            <div>
              <div style={{ fontSize: '12px', color: 'var(--text-muted)', marginBottom: '8px' }}>Quick Actions</div>
              <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                <button style={{ padding: '6px 12px', backgroundColor: 'var(--success-500)', border: 'none', borderRadius: '4px', color: 'white', fontSize: '10px' }}>
                  New Order
                </button>
                <button style={{ padding: '6px 12px', backgroundColor: 'var(--primary-500)', border: 'none', borderRadius: '4px', color: 'white', fontSize: '10px' }}>
                  View Charts
                </button>
                <button style={{ padding: '6px 12px', backgroundColor: 'var(--warning-500)', border: 'none', borderRadius: '4px', color: 'white', fontSize: '10px' }}>
                  Set Alert
                </button>
              </div>
            </div>
          </div>
        );

      case 'CORR':
        const corrData = window.content?.pairs || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Crypto Correlation Matrix</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Analysis Period:</span>
                <span style={{ color: 'var(--success-500)' }}>{window.content?.period || '30 days'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Strongest Correlation:</span>
                <span style={{ color: 'var(--primary-500)' }}>{window.content?.strongestCorr || 'BTC-ETH: 0.85'}</span>
              </div>
            </div>
            
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Pair</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Correlation</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Strength</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Trend</th>
                </tr>
              </thead>
              <tbody>
                {corrData.map((pair: any, idx: number) => (
                  <tr key={idx} style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                    <td style={{ padding: '4px', color: 'var(--success-500)', fontWeight: 'bold' }}>
                      {pair.pair}
                    </td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: Math.abs(pair.correlation) > 0.7 ? 'var(--warning-500)' : 'var(--text-primary)',
                      fontWeight: 'bold'
                    }}>
                      {pair.correlation?.toFixed(3) || 'N/A'}
                    </td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: Math.abs(pair.correlation) > 0.7 ? 'var(--danger-500)' : 
                             Math.abs(pair.correlation) > 0.4 ? 'var(--warning-500)' : 'var(--success-500)'
                    }}>
                      {Math.abs(pair.correlation) > 0.7 ? 'Strong' : 
                       Math.abs(pair.correlation) > 0.4 ? 'Moderate' : 'Weak'}
                    </td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: pair.trend === 'Increasing' ? 'var(--danger-500)' : 'var(--success-500)'
                    }}>
                      {pair.trend}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'FLOW':
        const flowData = window.content?.whaleTransactions || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>On-Chain Data & Whale Movements</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Total Volume (24h):</span>
                <span style={{ color: 'var(--success-500)' }}>${window.content?.totalVolume || '2.4B'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Large Transactions:</span>
                <span style={{ color: 'var(--primary-500)' }}>{window.content?.largeTxCount || '156'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Exchange Flow:</span>
                <span style={{ color: 'var(--warning-500)' }}>{window.content?.exchangeFlow || 'Net Inflow'}</span>
              </div>
            </div>
            
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Token</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Amount</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Value (USD)</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Type</th>
                </tr>
              </thead>
              <tbody>
                {flowData.map((tx: any, idx: number) => (
                  <tr key={idx} style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                    <td style={{ padding: '4px', color: 'var(--success-500)', fontWeight: 'bold' }}>
                      {tx.token}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--text-primary)' }}>
                      {tx.amount}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--primary-500)' }}>
                      ${tx.valueUSD}
                    </td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: tx.type === 'Inflow' ? 'var(--success-500)' : 'var(--danger-500)',
                      fontWeight: 'bold'
                    }}>
                      {tx.type}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'NFT':
        const nftData = window.content?.collections || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>NFT Market Analysis</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Total Volume (24h):</span>
                <span style={{ color: 'var(--success-500)' }}>${window.content?.totalVolume || '45.2M'}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Active Collections:</span>
                <span style={{ color: 'var(--primary-500)' }}>{nftData.length}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Floor Price Trend:</span>
                <span style={{ color: 'var(--success-500)' }}>{window.content?.floorTrend || 'Bullish'}</span>
              </div>
            </div>
            
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Collection</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Floor Price</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>24h Volume</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Change</th>
                </tr>
              </thead>
              <tbody>
                {nftData.map((collection: any, idx: number) => (
                  <tr key={idx} style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                    <td style={{ padding: '4px', color: 'var(--success-500)', fontWeight: 'bold' }}>
                      {collection.name}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--text-primary)' }}>
                      {collection.floorPrice}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--primary-500)' }}>
                      {collection.volume24h}
                    </td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: (collection.change24h || 0) >= 0 ? 'var(--success-500)' : 'var(--danger-500)',
                      fontWeight: 'bold'
                    }}>
                      {(collection.change24h || 0) >= 0 ? '+' : ''}{(collection.change24h || 0).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );

      case 'WHITEPAPER':
        const whitepaperData = window.content || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Token Whitepapers</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Available Papers:</span>
                <span style={{ color: 'var(--success-500)' }}>{whitepaperData.length}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Latest Update:</span>
                <span style={{ color: 'var(--primary-500)' }}>{whitepaperData[0]?.date || 'N/A'}</span>
              </div>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {whitepaperData.map((paper: any, idx: number) => (
                <div key={idx} style={{ 
                  padding: '10px', 
                  backgroundColor: 'var(--bg-secondary)', 
                  borderRadius: '4px',
                  border: '1px solid var(--bg-tertiary)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
                    <span style={{ color: 'var(--success-500)', fontWeight: 'bold', fontSize: '12px' }}>
                      {paper.token}
                    </span>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                      {paper.version}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '5px' }}>
                    Author: {paper.author} • {paper.date}
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-primary)' }}>
                    {paper.description || 'Technical documentation and tokenomics'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'API':
        const apiData = window.content?.keys || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>API Key Management</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Active Keys:</span>
                <span style={{ color: 'var(--success-500)' }}>{apiData.length}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Rate Limit:</span>
                <span style={{ color: 'var(--primary-500)' }}>{window.content?.rateLimit || '1000/min'}</span>
              </div>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {apiData.map((key: any, idx: number) => (
                <div key={idx} style={{ 
                  padding: '10px', 
                  backgroundColor: 'var(--bg-secondary)', 
                  borderRadius: '4px',
                  border: '1px solid var(--bg-tertiary)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
                    <span style={{ color: 'var(--success-500)', fontWeight: 'bold', fontSize: '12px' }}>
                      {key.name}
                    </span>
                    <span style={{ 
                      color: key.active ? 'var(--success-500)' : 'var(--danger-500)',
                      fontSize: '10px'
                    }}>
                      {key.active ? 'ACTIVE' : 'INACTIVE'}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '5px' }}>
                    Key: {key.key?.slice(0, 8)}...{key.key?.slice(-4)}
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-primary)' }}>
                    Permissions: {key.permissions?.join(', ') || 'Read-only'}
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      case 'SETTINGS':
        const settingsData = window.content;
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>User Settings & Preferences</h3>
            
            {/* Trading Settings */}
            <div style={{ marginBottom: '20px' }}>
              <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px', fontWeight: 'bold' }}>
                Trading Settings
              </div>
              <div style={{ fontSize: '11px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Default Pair:</span>
                  <span style={{ color: 'var(--primary-500)' }}>{settingsData?.trading?.defaultPair || 'BTC/USDT'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Order Type:</span>
                  <span style={{ color: 'var(--primary-500)' }}>{settingsData?.trading?.orderType || 'LIMIT'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Risk Level:</span>
                  <span style={{ color: 'var(--warning-500)' }}>{settingsData?.trading?.riskLevel || 'Medium'}</span>
                </div>
              </div>
            </div>
            
            {/* Display Settings */}
            <div style={{ marginBottom: '20px' }}>
              <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px', fontWeight: 'bold' }}>
                Display Settings
              </div>
              <div style={{ fontSize: '11px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Theme:</span>
                  <span style={{ color: 'var(--primary-500)' }}>{settingsData?.display?.theme || 'Dark'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Currency:</span>
                  <span style={{ color: 'var(--primary-500)' }}>{settingsData?.display?.currency || 'USD'}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Language:</span>
                  <span style={{ color: 'var(--primary-500)' }}>{settingsData?.display?.language || 'English'}</span>
                </div>
              </div>
            </div>
            
            {/* Notifications */}
            <div>
              <div style={{ fontSize: '12px', color: 'var(--success-500)', marginBottom: '10px', fontWeight: 'bold' }}>
                Notifications
              </div>
              <div style={{ fontSize: '11px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Price Alerts:</span>
                  <span style={{ color: settingsData?.notifications?.priceAlerts ? 'var(--success-500)' : 'var(--danger-500)' }}>
                    {settingsData?.notifications?.priceAlerts ? 'ON' : 'OFF'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Trade Notifications:</span>
                  <span style={{ color: settingsData?.notifications?.tradeNotifications ? 'var(--success-500)' : 'var(--danger-500)' }}>
                    {settingsData?.notifications?.tradeNotifications ? 'ON' : 'OFF'}
                  </span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ color: 'var(--text-muted)' }}>News Updates:</span>
                  <span style={{ color: settingsData?.notifications?.newsUpdates ? 'var(--success-500)' : 'var(--danger-500)' }}>
                    {settingsData?.notifications?.newsUpdates ? 'ON' : 'OFF'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        );

      case 'TWITTER':
        const twitterData = window.content?.tweets || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Crypto Twitter Feed</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Recent Tweets:</span>
                <span style={{ color: 'var(--success-500)' }}>{twitterData.length}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Sentiment:</span>
                <span style={{ color: 'var(--success-500)' }}>{window.content?.sentiment || 'Bullish'}</span>
              </div>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {twitterData.map((tweet: any, idx: number) => (
                <div key={idx} style={{ 
                  padding: '10px', 
                  backgroundColor: 'var(--bg-secondary)', 
                  borderRadius: '4px',
                  border: '1px solid var(--bg-tertiary)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
                    <span style={{ color: 'var(--success-500)', fontWeight: 'bold', fontSize: '12px' }}>
                      @{tweet.author}
                    </span>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                      {tweet.time}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-primary)', marginBottom: '5px' }}>
                    {tweet.content}
                  </div>
                  <div style={{ display: 'flex', gap: '15px', fontSize: '10px', color: 'var(--text-muted)' }}>
                    <span>❤️ {tweet.likes}</span>
                    <span>🔄 {tweet.retweets}</span>
                    <span>💬 {tweet.replies}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );
      case 'REDDIT':
        const redditData = window.content?.posts || [];
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            <h3 style={{ margin: '0 0 15px 0', fontSize: '14px' }}>Crypto Reddit Discussions</h3>
            <div style={{ marginBottom: '15px', fontSize: '11px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Active Discussions:</span>
                <span style={{ color: 'var(--success-500)' }}>{redditData.length}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '5px' }}>
                <span style={{ color: 'var(--text-muted)' }}>Hot Topics:</span>
                <span style={{ color: 'var(--primary-500)' }}>{window.content?.hotTopics || 'Bitcoin, DeFi, NFTs'}</span>
              </div>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              {redditData.map((post: any, idx: number) => (
                <div key={idx} style={{ 
                  padding: '10px', 
                  backgroundColor: 'var(--bg-secondary)', 
                  borderRadius: '4px',
                  border: '1px solid var(--bg-tertiary)'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '5px' }}>
                    <span style={{ color: 'var(--success-500)', fontWeight: 'bold', fontSize: '12px' }}>
                      r/{post.subreddit}
                    </span>
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                      {post.time}
                    </span>
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--text-primary)', marginBottom: '5px', fontWeight: 'bold' }}>
                    {post.title}
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: '5px' }}>
                    {post.content?.slice(0, 100)}...
                  </div>
                  <div style={{ display: 'flex', gap: '15px', fontSize: '10px', color: 'var(--text-muted)' }}>
                    <span>⬆️ {post.upvotes}</span>
                    <span>💬 {post.comments}</span>
                    <span>🏆 {post.awards}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        );

      default:
        return (
          <div style={{ padding: '20px', textAlign: 'center', color: 'var(--text-muted)' }}>
            {window.type} Component - Implementation pending
          </div>
        );
    }
  };

  // Mouse event handlers for window dragging
  const handleMouseDown = (e: React.MouseEvent, windowId: string) => {
    // Prevent text selection during drag
    e.preventDefault();
    
    const window = windows.find(w => w.id === windowId);
    if (!window) return;
    
    setDragData({
      windowId,
      startX: window.x,
      startY: window.y,
      startMouseX: e.clientX,
      startMouseY: e.clientY
    });
    
    bringToFront(windowId);
    
    // Disable text selection during drag
    document.body.style.userSelect = 'none';
    document.body.style.webkitUserSelect = 'none';
  };

  const handleMouseMove = React.useCallback((e: MouseEvent) => {
    if (!dragData) return;
    
    const deltaX = e.clientX - dragData.startMouseX;
    const deltaY = e.clientY - dragData.startMouseY;
    
    setWindows(prev => prev.map(w => 
      w.id === dragData.windowId 
        ? { 
            ...w, 
            x: Math.max(0, Math.min(window.innerWidth - w.width, dragData.startX + deltaX)),
            y: Math.max(50, Math.min(window.innerHeight - w.height - 30, dragData.startY + deltaY))
          }
        : w
    ));
  }, [dragData]);

  const handleMouseUp = React.useCallback(() => {
    setDragData(null);
    setResizeData(null);
    
    // Re-enable text selection after drag/resize
    document.body.style.userSelect = '';
    document.body.style.webkitUserSelect = '';
  }, []);

  // Resize event handlers
  const handleResizeMouseDown = (e: React.MouseEvent, windowId: string) => {
    e.stopPropagation();
    e.preventDefault(); // Prevent text selection during resize
    
    const window = windows.find(w => w.id === windowId);
    if (!window) return;
    
    setResizeData({
      windowId,
      startWidth: window.width,
      startHeight: window.height,
      startMouseX: e.clientX,
      startMouseY: e.clientY
    });
    
    bringToFront(windowId);
    
    // Disable text selection during resize
    document.body.style.userSelect = 'none';
    document.body.style.webkitUserSelect = 'none';
  };

  const handleResizeMouseMove = React.useCallback((e: MouseEvent) => {
    if (!resizeData) return;
    
    const deltaX = e.clientX - resizeData.startMouseX;
    const deltaY = e.clientY - resizeData.startMouseY;
    
    setWindows(prev => prev.map(w => 
      w.id === resizeData.windowId 
        ? { 
            ...w, 
            width: Math.max(250, resizeData.startWidth + deltaX),
            height: Math.max(200, resizeData.startHeight + deltaY)
          }
        : w
    ));
  }, [resizeData]);

  // Add global mouse event listeners
  React.useEffect(() => {
    if (dragData || resizeData) {
      if (dragData) {
        document.addEventListener('mousemove', handleMouseMove);
      }
      if (resizeData) {
        document.addEventListener('mousemove', handleResizeMouseMove);
      }
      document.addEventListener('mouseup', handleMouseUp);
      return () => {
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mousemove', handleResizeMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [dragData, resizeData, handleMouseMove, handleResizeMouseMove, handleMouseUp]);

  // Professional trading hotkeys
  const handleTradingHotkeys = React.useCallback((e: KeyboardEvent) => {
    // Only handle hotkeys if not typing in an input field and terminal is not active
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || terminalActive) {
      return;
    }

    // Modifier key combinations
    const isCtrl = e.ctrlKey || e.metaKey;
    const isShift = e.shiftKey;
    const isAlt = e.altKey;

    // Quick Command Shortcuts (Ctrl + Key)
    if (isCtrl && !isShift && !isAlt) {
      switch (e.key.toLowerCase()) {
        case 'q':
          e.preventDefault();
          executeCommand('QM'); // Quote Monitor
          break;
        case 'p':
          e.preventDefault();
          executeCommand('PF'); // Portfolio
          break;
        case 'b':
          e.preventDefault();
          executeCommand('BROK'); // Trading Interface
          break;
        case 't':
          e.preventDefault();
          executeCommand('BT'); // Backtesting
          break;
        case 'c':
          e.preventDefault();
          executeCommand('CHART'); // Charts
          break;
        case 'n':
          e.preventDefault();
          executeCommand('N'); // News
          break;
        case 'h':
          e.preventDefault();
          executeCommand('HELP'); // Help
          break;
        case 'r':
          e.preventDefault();
          // Refresh current data
          if (mode === 'pro') {
            // Refresh all windows
            setWindows(prev => prev.map(w => ({ ...w, lastRefresh: Date.now() })));
          }
          break;
        case 'w':
          e.preventDefault();
          // Close active window
          if (windows.length > 0) {
            const activeWindow = windows.reduce((prev, current) => 
              (prev.zIndex > current.zIndex) ? prev : current
            );
            closeWindow(activeWindow.id);
          }
          break;
      }
    }

    // Alt + Key shortcuts
    if (isAlt && !isCtrl && !isShift) {
      switch (e.key.toLowerCase()) {
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
          e.preventDefault();
          const windowIndex = parseInt(e.key) - 1;
          if (windows[windowIndex]) {
            bringToFront(windows[windowIndex].id);
          }
          break;
        case 'f':
          e.preventDefault();
          // Fit all windows to screen
          fitAllWindows();
          break;
        case 'l':
          e.preventDefault();
          // Reset layout
          resetLayout();
          break;
        case 'm':
          e.preventDefault();
          // Toggle between Lite and Pro mode
          setMode(prev => prev === 'lite' ? 'pro' : 'lite');
          break;
      }
    }

    // Ctrl + Shift combinations
    if (isCtrl && isShift && !isAlt) {
      switch (e.key.toLowerCase()) {
        case 'x':
          e.preventDefault();
          // Close all windows
          setWindows([]);
          break;
        case 'r':
          e.preventDefault();
          // Hard refresh - reload page
          window.location.reload();
          break;
        case 'f':
          e.preventDefault();
          // Toggle fullscreen
          if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
          } else {
            document.exitFullscreen();
          }
          break;
      }
    }

    // Function key shortcuts
    if (!isCtrl && !isShift && !isAlt) {
      switch (e.key) {
        case 'F1':
          e.preventDefault();
          executeCommand('HELP');
          break;
        case 'F2':
          e.preventDefault();
          executeCommand('QM');
          break;
        case 'F3':
          e.preventDefault();
          executeCommand('PF');
          break;
        case 'F4':
          e.preventDefault();
          executeCommand('BROK');
          break;
        case 'F5':
          e.preventDefault();
          executeCommand('CHART');
          break;
        case 'F6':
          e.preventDefault();
          executeCommand('BT');
          break;
        case 'F9':
          e.preventDefault();
          // Quick buy BTCUSDT
          if (mode === 'pro') {
            executeCommand('BROK');
          }
          break;
        case 'F10':
          e.preventDefault();
          // Quick sell BTCUSDT  
          if (mode === 'pro') {
            executeCommand('BROK');
          }
          break;
        case 'F11':
          e.preventDefault();
          // Toggle fullscreen
          if (!document.fullscreenElement) {
            document.documentElement.requestFullscreen();
          } else {
            document.exitFullscreen();
          }
          break;
      }
    }
  }, [executeCommand, mode, windows, closeWindow, bringToFront, terminalActive]);

  // Fit all windows to screen
  const fitAllWindows = React.useCallback(() => {
    if (windows.length === 0) return;

    const screenWidth = window.innerWidth;
    const screenHeight = window.innerHeight - 100; // Account for taskbar
    const cols = Math.ceil(Math.sqrt(windows.length));
    const rows = Math.ceil(windows.length / cols);
    const windowWidth = Math.floor(screenWidth / cols);
    const windowHeight = Math.floor(screenHeight / rows);

    setWindows(prev => prev.map((window, index) => {
      const col = index % cols;
      const row = Math.floor(index / cols);
      return {
        ...window,
        x: col * windowWidth,
        y: row * windowHeight,
        width: windowWidth - 10,
        height: windowHeight - 10
      };
    }));
  }, [windows]);

  // Reset layout
  const resetLayout = React.useCallback(() => {
    setWindows(prev => prev.map((window, index) => ({
      ...window,
      x: 50 + (index * 30),
      y: 50 + (index * 30),
      width: 600,
      height: 400,
      zIndex: 1000 + index
    })));
    setNextZIndex(1000 + windows.length);
  }, [windows]);

  // Global keyboard event handlers
  const handleGlobalKeyDown = React.useCallback((e: KeyboardEvent) => {
    // Handle backtick key globally (anywhere on the page)
    if (e.key === '`' && !showBacktickMenu) {
      e.preventDefault();
      setTerminalActive(true);
      setShowBacktickMenu(true);
      setSelectedCommandIndex(0);
      // Focus the input when activating terminal
      setTimeout(() => commandInputRef.current?.focus(), 0);
      return;
    }

    // Handle Escape key globally - completely deactivate terminal
    if (e.key === 'Escape') {
      if (showBacktickMenu || terminalActive) {
        e.preventDefault();
        setShowBacktickMenu(false);
        setSelectedCommandIndex(0);
        setTerminalActive(false);
        setInputFocused(false);
        // Blur the input to remove focus
        commandInputRef.current?.blur();
        return;
      }
    }

    // Handle arrow keys in command menu
    if (showBacktickMenu) {
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        const newIndex = selectedCommandIndex > 0 ? selectedCommandIndex - 1 : backtickCommands.length - 1;
        setSelectedCommandIndex(newIndex);
        // Scroll the selected item into view
        setTimeout(() => {
          const selectedItem = document.querySelector(`[data-command-idx="${newIndex}"]`);
          if (selectedItem) {
            selectedItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        }, 10);
        return;
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        const newIndex = selectedCommandIndex < backtickCommands.length - 1 ? selectedCommandIndex + 1 : 0;
        setSelectedCommandIndex(newIndex);
        // Scroll the selected item into view
        setTimeout(() => {
          const selectedItem = document.querySelector(`[data-command-idx="${newIndex}"]`);
          if (selectedItem) {
            selectedItem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        }, 10);
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        const selectedCommand = backtickCommands[selectedCommandIndex];
        executeCommand(selectedCommand.command);
        setShowBacktickMenu(false);
        setSelectedCommandIndex(0);
        return;
      }
    }

    // Handle trading hotkeys first (before other keys)
    handleTradingHotkeys(e);

    // Only handle other keys if command input is focused
    if (document.activeElement === commandInputRef.current) {
      if (e.key === 'Enter' && command.trim()) {
        e.preventDefault();
        executeCommand(command);
        setCommand('');
        setHistoryIndex(-1);
        return;
      }

      if (e.key === 'ArrowUp' && commandHistory.length > 0) {
        e.preventDefault();
        const newIndex = historyIndex === -1 ? commandHistory.length - 1 : Math.max(0, historyIndex - 1);
        setHistoryIndex(newIndex);
        setCommand(commandHistory[newIndex]);
        return;
      }

      if (e.key === 'ArrowDown' && commandHistory.length > 0) {
        e.preventDefault();
        if (historyIndex >= 0) {
          const newIndex = historyIndex + 1;
          if (newIndex >= commandHistory.length) {
            setHistoryIndex(-1);
            setCommand('');
          } else {
            setHistoryIndex(newIndex);
            setCommand(commandHistory[newIndex]);
          }
        }
        return;
      }
    }
  }, [showBacktickMenu, command, commandHistory, historyIndex, selectedCommandIndex, backtickCommands, executeCommand, terminalActive]);

  // Add global keyboard event listeners
  React.useEffect(() => {
    document.addEventListener('keydown', handleGlobalKeyDown);
    return () => {
      document.removeEventListener('keydown', handleGlobalKeyDown);
    };
  }, [handleGlobalKeyDown]);

  // Mobile/responsive detection
  const [isMobile, setIsMobile] = React.useState(false);
  const [isTablet, setIsTablet] = React.useState(false);
  const [screenSize, setScreenSize] = React.useState({ width: window.innerWidth, height: window.innerHeight });

  React.useEffect(() => {
    const handleResize = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      
      setScreenSize({ width, height });
      setIsMobile(width < 768);
      setIsTablet(width >= 768 && width < 1024);
    };

    handleResize(); // Initial check
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Initialize terminal in inactive state
  React.useEffect(() => {
    // Don't auto-focus on mount - wait for user interaction
    setTerminalActive(false);
    setInputFocused(false);
  }, []);

  // WebSocket connection for real-time market data
  React.useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    
    ws.onopen = () => {
      console.log('Connected to market data WebSocket');
      // Subscribe to market data
      ws.send(JSON.stringify({ type: 'subscribe_market_data' }));
    };
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        if (message.type === 'market_update') {
          setMarketData(message.data.quotes || {});
        } else if (message.type === 'market_data_snapshot') {
          setMarketData(message.data || {});
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };
    
    ws.onclose = () => {
      console.log('Market data WebSocket disconnected');
    };
    
    ws.onerror = (error) => {
      console.error('Market data WebSocket error:', error);
    };
    
    return () => {
      ws.close();
    };
  }, []);
  
  // Add blinking cursor animation and fat cursor styling
  React.useEffect(() => {
    const style = document.createElement('style');
    style.textContent = `
      @keyframes blink {
        0%, 50% { opacity: 1; }
        51%, 100% { opacity: 0; }
      }
      
      .terminal-input {
        caret-color: transparent !important;
      }
      
      .terminal-input:focus {
        caret-color: transparent !important;
      }
      
      .terminal-input.inactive {
        caret-color: transparent !important;
        pointer-events: none;
      }
      
      .fat-cursor {
        display: inline-block;
        width: 8px;
        height: 16px;
        background-color: var(--success-500);
        animation: blink 1s infinite;
        position: absolute;
        left: 0;
        top: 50%;
        transform: translateY(-50%);
        pointer-events: none;
      }
      
      .fat-cursor-active {
        display: inline-block;
        width: 8px;
        height: 16px;
        background-color: var(--success-500);
        animation: blink 1s infinite;
        position: absolute;
        top: 50%;
        transform: translateY(-50%);
        pointer-events: none;
        z-index: 10;
      }
    `;
    document.head.appendChild(style);
    
    return () => {
      document.head.removeChild(style);
    };
  }, []);
  return (
    <ThemeProvider>
      <ProThemeSetter />
      <div style={{
        width: '100vw',
        height: '100vh',
        backgroundColor: 'var(--bg-primary)',
        color: 'var(--text-primary)',
        fontFamily: 'JetBrains Mono, Monaco, Consolas, "Courier New", monospace',
      position: 'relative',
      overflow: 'hidden'
    }}>
      {/* Command Bar */}
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        height: '50px',
        backgroundColor: 'var(--bg-secondary)',
        borderBottom: '1px solid var(--bg-tertiary)',
        display: 'flex',
        alignItems: 'center',
        padding: '0 20px',
        zIndex: 1000
      }}>
        <span style={{ marginRight: '10px', color: 'var(--success-500)' }}>
          {getUsername()}@QuantDeskPro:~$
        </span>
        <div style={{ flex: 1, position: 'relative', display: 'flex', alignItems: 'center' }}>
          <input
            ref={commandInputRef}
            type="text"
            value={command}
            onChange={(e) => {
              setCommand(e.target.value)
              performSearch(e.target.value)
            }}
            onFocus={() => {
              setInputFocused(true);
              setTerminalActive(true);
              if (!showBacktickMenu) {
                setShowBacktickMenu(true);
                setSelectedCommandIndex(0);
              }
            }}
            onBlur={() => {
              // Only blur if terminal is not active (ESC was pressed)
              if (!terminalActive) {
                setInputFocused(false);
              }
            }}
            onClick={() => {
              if (!terminalActive) {
                setTerminalActive(true);
                setShowBacktickMenu(true);
                setSelectedCommandIndex(0);
              }
            }}
            placeholder={terminalActive ? " Type a command..." : "Backtick (`) to open terminal"}
            className={`terminal-input ${!terminalActive ? 'inactive' : ''}`}
            style={{
              width: '100%',
              backgroundColor: 'transparent',
              border: 'none',
              color: 'var(--text-primary)',
              fontSize: '14px',
              outline: 'none',
              fontFamily: 'inherit',
              pointerEvents: terminalActive ? 'auto' : 'none'
            }}
          />
          {terminalActive && (
            <div 
              className="fat-cursor-active"
              style={{
                left: `${command.length * 8.4}px` // Position cursor after text
              }}
            ></div>
          )}
        </div>
        {/* Right Side - Time with Market Status */}
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '15px',
          color: 'var(--text-muted)',
          fontSize: '11px',
          fontFamily: 'Monaco, Consolas, "Courier New", monospace'
        }}>
          {/* Time with Market Status Hover */}
          <div 
            style={{ 
              fontSize: '10px',
              color: '#f5f5dc',
              cursor: 'pointer',
              position: 'relative',
              padding: '4px 8px',
              borderRadius: '3px',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              const tooltip = e.currentTarget.querySelector('.market-status-tooltip') as HTMLElement;
              if (tooltip) tooltip.style.display = 'block';
              (e.currentTarget as HTMLElement).style.backgroundColor = 'rgba(82, 196, 26, 0.2)';
              (e.currentTarget as HTMLElement).style.border = '1px solid var(--success-500)';
              (e.currentTarget as HTMLElement).style.color = 'var(--text-primary)';
            }}
            onMouseLeave={(e) => {
              const tooltip = e.currentTarget.querySelector('.market-status-tooltip') as HTMLElement;
              if (tooltip) tooltip.style.display = 'none';
              (e.currentTarget as HTMLElement).style.backgroundColor = 'transparent';
              (e.currentTarget as HTMLElement).style.border = 'none';
              (e.currentTarget as HTMLElement).style.color = '#f5f5dc';
            }}
          >
            {new Date().toLocaleTimeString()}
            
            {/* Market Status Tooltip */}
            <div 
              className="market-status-tooltip"
              style={{
                display: 'none',
                position: 'absolute',
                top: '25px',
                right: '0px',
                backgroundColor: 'var(--bg-secondary)',
                border: '1px solid #333',
                borderRadius: '4px',
                padding: '12px',
                minWidth: '200px',
                zIndex: 2000,
                boxShadow: '0 4px 12px rgba(0, 0, 0, 0.5)',
                fontFamily: 'Monaco, Consolas, "Courier New", monospace'
              }}
            >
              <div style={{ 
                fontSize: '11px', 
                fontWeight: 'bold', 
                color: 'var(--success-500)', 
                marginBottom: '8px',
                borderBottom: '1px solid #333',
                paddingBottom: '4px'
              }}>
                MARKET STATUS
              </div>
              
              {/* US Markets */}
              <div style={{ marginBottom: '6px' }}>
                <div style={{ fontSize: '10px', color: '#999', marginBottom: '2px' }}>US Markets</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '9px' }}>
                  <div style={{ 
                    width: '6px', 
                    height: '6px', 
                    borderRadius: '50%', 
                    backgroundColor: 'var(--danger-500)' // Red for closed
                  }}></div>
                  <span style={{ color: 'var(--danger-500)' }}>NYSE - CLOSED</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '9px', marginTop: '2px' }}>
                  <div style={{ 
                    width: '6px', 
                    height: '6px', 
                    borderRadius: '50%', 
                    backgroundColor: 'var(--danger-500)' // Red for closed
                  }}></div>
                  <span style={{ color: 'var(--danger-500)' }}>NASDAQ - CLOSED</span>
                </div>
              </div>

              {/* Crypto Markets */}
              <div style={{ marginBottom: '6px' }}>
                <div style={{ fontSize: '10px', color: '#999', marginBottom: '2px' }}>Crypto</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '9px' }}>
                  <div style={{ 
                    width: '6px', 
                    height: '6px', 
                    borderRadius: '50%', 
                    backgroundColor: 'var(--success-500)' // Green for live
                  }}></div>
                  <span style={{ color: 'var(--success-500)' }}>CRYPTO - LIVE 24/7</span>
                </div>
              </div>

              {/* Forex Markets */}
              <div style={{ marginBottom: '6px' }}>
                <div style={{ fontSize: '10px', color: '#999', marginBottom: '2px' }}>Forex</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '9px' }}>
                  <div style={{ 
                    width: '6px', 
                    height: '6px', 
                    borderRadius: '50%', 
                    backgroundColor: 'var(--success-500)' // Green for live
                  }}></div>
                  <span style={{ color: 'var(--success-500)' }}>FOREX - LIVE</span>
                </div>
              </div>

              {/* News */}
              <div>
                <div style={{ fontSize: '10px', color: '#999', marginBottom: '2px' }}>News</div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '9px' }}>
                  <div style={{ 
                    width: '6px', 
                    height: '6px', 
                    borderRadius: '50%', 
                    backgroundColor: 'var(--success-500)' // Green for live
                  }}></div>
                  <span style={{ color: 'var(--success-500)' }}>NEWS - LIVE</span>
                </div>
              </div>
            </div>
          </div>

          {/* Wallet Connect Button */}
          <WalletButton />
        </div>
      </div>

      {/* Backtick Command Menu / Search Results */}
      {showBacktickMenu && (
        <div style={{
          position: 'fixed',
          top: '60px',
          left: '20px',
          width: '500px',
          maxHeight: '70vh',
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid #333',
          borderRadius: '4px',
          zIndex: 2000,
          overflow: 'auto'
        }}>
          {/* Show search results if there's a query, otherwise show all commands */}
          {command.trim() ? (
            <>
              {/* COMMANDS Section */}
              {filteredCommands.length > 0 && (
                <>
                  <div style={{
                    padding: '10px',
                    borderBottom: '1px solid #333',
                    fontWeight: 'bold',
                    fontSize: '12px',
                    color: 'var(--success-500)'
                  }}>
                    COMMANDS
                  </div>
                  <div style={{ padding: '10px' }}>
                    {filteredCommands.map((cmd, idx) => (
                      <div
                        key={idx}
                        onClick={() => {
                          executeCommand(cmd.command);
                          setShowBacktickMenu(false);
                          setSelectedCommandIndex(0);
                          setCommand('');
                        }}
                        style={{
                          padding: '6px 8px',
                          cursor: 'pointer',
                          fontSize: '11px',
                          borderRadius: '2px',
                          marginBottom: '2px',
                          backgroundColor: 'transparent',
                          border: '1px solid transparent',
                          transition: 'all 0.2s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = 'rgba(82, 196, 26, 0.2)';
                          e.currentTarget.style.border = '1px solid var(--success-500)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'transparent';
                          e.currentTarget.style.border = '1px solid transparent';
                        }}
                      >
                        <span style={{ color: 'var(--success-500)', fontWeight: 'bold', minWidth: '50px', display: 'inline-block' }}>
                          {cmd.command}
                        </span>
                        <span style={{ color: '#999' }}> {cmd.description}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {/* INSTRUMENTS Section */}
              {filteredInstruments.length > 0 && (
                <>
                  <div style={{
                    padding: '10px',
                    borderBottom: '1px solid #333',
                    fontWeight: 'bold',
                    fontSize: '12px',
                    color: 'var(--success-500)'
                  }}>
                    INSTRUMENTS
                  </div>
                  <div style={{ padding: '10px' }}>
                    {filteredInstruments.map((instrument, idx) => (
                      <div
                        key={idx}
                        onClick={() => {
                          executeCommand(`QUOTE ${instrument.symbol}`);
                          setShowBacktickMenu(false);
                          setCommand('');
                        }}
                        style={{
                          padding: '6px 8px',
                          cursor: 'pointer',
                          fontSize: '11px',
                          borderRadius: '2px',
                          marginBottom: '2px',
                          backgroundColor: 'transparent',
                          border: '1px solid transparent',
                          transition: 'all 0.2s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = 'rgba(82, 196, 26, 0.2)';
                          e.currentTarget.style.border = '1px solid var(--success-500)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'transparent';
                          e.currentTarget.style.border = '1px solid transparent';
                        }}
                      >
                        <span style={{ color: 'var(--primary-500)', fontWeight: 'bold', minWidth: '60px', display: 'inline-block' }}>
                          {instrument.type} {instrument.symbol}
                        </span>
                        <span style={{ color: 'var(--text-primary)' }}> {instrument.name}</span>
                        <span style={{ 
                          color: instrument.change >= 0 ? 'var(--success-500)' : 'var(--danger-500)',
                          float: 'right',
                          fontSize: '10px'
                        }}>
                          ${instrument.price} {instrument.change >= 0 ? '+' : ''}{instrument.changePercent}%
                        </span>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {/* NEWS STORIES Section */}
              {filteredNews.length > 0 && (
                <>
                  <div style={{
                    padding: '10px',
                    borderBottom: '1px solid #333',
                    fontWeight: 'bold',
                    fontSize: '12px',
                    color: 'var(--success-500)'
                  }}>
                    NEWS STORIES
                  </div>
                  <div style={{ padding: '10px' }}>
                    {filteredNews.map((news, idx) => (
                      <div
                        key={idx}
                        onClick={() => {
                          executeCommand(`NEWS ${news.ticker}`);
                          setShowBacktickMenu(false);
                          setCommand('');
                        }}
                        style={{
                          padding: '6px 8px',
                          cursor: 'pointer',
                          fontSize: '11px',
                          borderRadius: '2px',
                          marginBottom: '2px',
                          backgroundColor: 'transparent',
                          border: '1px solid transparent',
                          transition: 'all 0.2s ease'
                        }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.backgroundColor = 'rgba(82, 196, 26, 0.2)';
                          e.currentTarget.style.border = '1px solid var(--success-500)';
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.backgroundColor = 'transparent';
                          e.currentTarget.style.border = '1px solid transparent';
                        }}
                      >
                        <div style={{ color: 'var(--text-primary)', marginBottom: '2px' }}>
                          {news.headline}
                        </div>
                        <div style={{ color: '#999', fontSize: '10px' }}>
                          {news.ticker} • {news.time} • {news.source}
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              )}

              {/* No Results */}
              {filteredCommands.length === 0 && filteredInstruments.length === 0 && filteredNews.length === 0 && (
                <div style={{
                  padding: '20px',
                  textAlign: 'center',
                  color: 'var(--text-muted)',
                  fontSize: '12px'
                }}>
                  No results found for "{command}"
                </div>
              )}
            </>
          ) : (
            <>
              {/* Default Commands Menu */}
              <div style={{
                padding: '10px',
                borderBottom: '1px solid #333',
                fontWeight: 'bold',
                fontSize: '12px',
                color: 'var(--success-500)'
              }}>
                COMMANDS
              </div>
              <div style={{ padding: '10px' }}>
                {backtickCommands.map((cmd, idx) => (
                  <div
                    key={idx}
                    data-command-idx={idx}
                    onClick={() => {
                      executeCommand(cmd.command);
                      setShowBacktickMenu(false);
                      setSelectedCommandIndex(0);
                    }}
                    style={{
                      padding: '6px 8px',
                      cursor: 'pointer',
                      fontSize: '11px',
                      borderRadius: '2px',
                      marginBottom: '2px',
                      backgroundColor: idx === selectedCommandIndex ? 'rgba(82, 196, 26, 0.2)' : 'transparent',
                      border: idx === selectedCommandIndex ? '1px solid var(--success-500)' : '1px solid transparent',
                      transition: 'all 0.2s ease'
                    }}
                    onMouseEnter={(e) => {
                      if (idx !== selectedCommandIndex) {
                        e.currentTarget.style.backgroundColor = 'rgba(82, 196, 26, 0.2)';
                        e.currentTarget.style.border = '1px solid var(--success-500)';
                      }
                      setSelectedCommandIndex(idx);
                    }}
                    onMouseLeave={(e) => {
                      if (idx !== selectedCommandIndex) {
                        e.currentTarget.style.backgroundColor = 'transparent';
                        e.currentTarget.style.border = '1px solid transparent';
                      }
                    }}
                  >
                    <span style={{ color: 'var(--success-500)', fontWeight: 'bold', minWidth: '50px', display: 'inline-block' }}>
                      {cmd.command}
                    </span>
                    <span style={{ color: '#999' }}> {cmd.description}</span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* Windows */}
      {windows.map((window) => (
        <div
          key={window.id}
          style={{
            position: 'absolute',
            left: window.x,
            top: window.y,
            width: window.width,
            height: window.height,
            backgroundColor: 'var(--bg-secondary)',
            border: '1px solid #333',
            borderRadius: '4px',
            zIndex: window.zIndex,
            display: window.isMinimized ? 'none' : 'block'
          }}
          onClick={() => bringToFront(window.id)}
        >
          {/* Window Header */}
          <div 
            style={{
              height: '30px',
              backgroundColor: '#333',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '0 10px',
              cursor: 'move',
              borderRadius: '4px 4px 0 0'
            }}
            onMouseDown={(e) => handleMouseDown(e, window.id)}
          >
            <span style={{ fontSize: '12px', fontWeight: 'bold' }}>{window.title}</span>
            <button
              onClick={(e) => {
                e.stopPropagation();
                closeWindow(window.id);
              }}
              onMouseDown={(e) => e.stopPropagation()}
              style={{
                background: 'none',
                border: 'none',
                color: 'var(--danger-500)',
                cursor: 'pointer',
                fontSize: '16px',
                padding: '0',
                width: '20px',
                height: '20px'
              }}
            >
              ×
            </button>
          </div>

          {/* Window Content */}
          <div style={{
            height: 'calc(100% - 30px)',
            overflow: 'hidden'
          }}>
            {renderWindowContent(window)}
          </div>

          {/* Resize Handle */}
          <div
            onMouseDown={(e) => handleResizeMouseDown(e, window.id)}
            style={{
              position: 'absolute',
              bottom: 0,
              right: 0,
              width: '15px',
              height: '15px',
              cursor: 'nw-resize',
              background: 'linear-gradient(-45deg, transparent 0%, transparent 40%, var(--text-muted) 40%, var(--text-muted) 60%, transparent 60%)',
              zIndex: 1000
            }}
          />
        </div>
      ))}

      {/* Welcome Message */}
      {windows.length === 0 && (
        <div style={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          textAlign: 'center',
          color: 'var(--text-muted)'
        }}>
          <div style={{ fontSize: '24px', marginBottom: '20px' }}>🚀 QuantDesk Pro Terminal</div>
          <div style={{ fontSize: '16px', marginBottom: '10px' }}>Professional Trading Terminal</div>
          <div style={{ fontSize: '12px' }}>
            Press <span style={{ color: 'var(--success-500)', fontWeight: 'bold' }}>backtick (`)</span> to open command menu
          </div>
          <div style={{ fontSize: '12px', marginTop: '5px' }}>
            Try: <span style={{ color: 'var(--primary-500)' }}>QM</span>, <span style={{ color: 'var(--primary-500)' }}>CHART</span>, <span style={{ color: 'var(--primary-500)' }}>NEWS</span>, <span style={{ color: 'var(--primary-500)' }}>HELP</span>
          </div>
        </div>
      )}

      {/* Bottom Taskbar - Godel Terminal Style */}
      <div style={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        height: '28px',
        backgroundColor: 'var(--bg-primary)',
        borderTop: '1px solid var(--bg-tertiary)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 8px',
        fontSize: '11px',
        zIndex: 999,
        fontFamily: 'Monaco, Consolas, "Courier New", monospace'
      }}>
        {/* Left Side - Mode & Tools */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
          {/* Lite Mode Button */}
          <button
            onClick={() => {
              localStorage.setItem('quantdesk_ui_mode', 'lite')
              navigate('/lite')
            }}
            style={{
              padding: '2px 8px',
              backgroundColor: 'transparent',
              color: 'var(--text-muted)',
              border: '1px solid var(--bg-tertiary)',
              borderRadius: '2px',
              cursor: 'pointer',
              fontSize: '10px',
              fontWeight: 'bold'
            }}
            onMouseEnter={(e) => (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-primary)'}
            onMouseLeave={(e) => (e.currentTarget as HTMLButtonElement).style.color = 'var(--text-muted)'}
          >
            LITE
          </button>
          
          {/* Pro Mode Button */}
          <button
            style={{
              padding: '2px 8px',
              backgroundColor: 'var(--primary-500)',
              color: 'var(--text-primary)',
              border: '1px solid var(--primary-500)',
              borderRadius: '2px',
              cursor: 'pointer',
              fontSize: '10px',
              fontWeight: 'bold'
            }}
          >
            PRO
          </button>
          
          {/* Theme Switcher */}
          <div style={{ marginLeft: '4px' }}>
            <ThemeToggle />
          </div>
          
          {/* Settings Button */}
          <button
            onClick={() => setShowSettings(true)}
            title="Terminal Settings & Profiles"
            style={{
              background: 'none',
              border: '1px solid var(--bg-tertiary)',
              color: 'var(--text-muted)',
              padding: '3px 6px',
              fontSize: '10px',
              cursor: 'pointer',
              borderRadius: '2px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease',
              marginLeft: '4px'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(82, 196, 26, 0.2)';
              e.currentTarget.style.borderColor = 'var(--primary-500)';
              e.currentTarget.style.color = 'var(--primary-500)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.borderColor = 'var(--bg-tertiary)';
              e.currentTarget.style.color = 'var(--text-muted)';
            }}
          >
            ⚙️
          </button>
          
          {/* Visual Separator */}
          <div style={{ 
            width: '1px', 
            height: '20px', 
            backgroundColor: 'var(--bg-tertiary)', 
            margin: '0 4px' 
          }}></div>
          
          {/* Reset Layout Icon */}
          <button
            onClick={() => setWindows([])}
            title="Reset Layout - Clear all windows"
            style={{
              background: 'none',
              border: '1px solid var(--bg-tertiary)',
              color: 'var(--text-muted)',
              padding: '3px 6px',
              fontSize: '10px',
              cursor: 'pointer',
              borderRadius: '2px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(82, 196, 26, 0.2)';
              e.currentTarget.style.borderColor = 'var(--success-500)';
              e.currentTarget.style.color = 'var(--success-500)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.borderColor = '#333';
              e.currentTarget.style.color = '#999';
            }}
          >
            <svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor">
              <path fillRule="evenodd" clipRule="evenodd" d="M12.75 8a4.5 4.5 0 0 1-8.61 1.834l-1.391.565A6.001 6.001 0 0 0 14.25 8 6 6 0 0 0 3.5 4.334V2.5H2v4l.75.75h3.5v-1.5H4.352A4.5 4.5 0 0 1 12.75 8z"/>
            </svg>
          </button>

          {/* Fit All Icon */}
          <button
            onClick={() => {
              // Fit all windows to screen
              const screenWidth = window.innerWidth;
              const screenHeight = window.innerHeight - 80;
              const windowCount = windows.length;
              
              if (windowCount > 0) {
                const cols = Math.ceil(Math.sqrt(windowCount));
                const rows = Math.ceil(windowCount / cols);
                const windowWidth = Math.floor(screenWidth / cols) - 20;
                const windowHeight = Math.floor(screenHeight / rows) - 20;
                
                setWindows(prev => prev.map((window, idx) => ({
                  ...window,
                  x: (idx % cols) * (windowWidth + 20) + 10,
                  y: Math.floor(idx / cols) * (windowHeight + 20) + 60,
                  width: windowWidth,
                  height: windowHeight
                })));
              }
            }}
            title="Fit All - Arrange all windows to fit screen"
            style={{
              background: 'none',
              border: '1px solid var(--bg-tertiary)',
              color: 'var(--text-muted)',
              padding: '3px 6px',
              fontSize: '10px',
              cursor: 'pointer',
              borderRadius: '2px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(82, 196, 26, 0.2)';
              e.currentTarget.style.borderColor = 'var(--success-500)';
              e.currentTarget.style.color = 'var(--success-500)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.borderColor = '#333';
              e.currentTarget.style.color = '#999';
            }}
          >
            <svg width="10" height="10" viewBox="0 0 52 52" fill="currentColor">
              <path d="M50,6c0-2.2-1.8-4-4-4H6C3.8,2,2,3.8,2,6v27.7c0,2.2,1.8,4,4,4h40c2.2,0,4-1.8,4-4V6z M44,30.2c0,0.8-0.7,1.5-1.5,1.5h-33C8.7,31.7,8,31,8,30.2V9.5C8,8.7,8.7,8,9.5,8h33C43.3,8,44,8.7,44,9.5V30.2z M19,44c-2.2,0-4,1.8-4,4v0.5c0,0.8,0.7,1.5,1.5,1.5h19c0.8,0,1.5-0.7,1.5-1.5V48c0-2.2-1.8-4-4-4H19z"/>
              <path d="M18,26.7h-4.1c-0.6,0-1-0.5-1-1V14c0-0.6,0.4-1,1-1H18c0.5,0,1,0.4,1,1v11.7C19,26.3,18.5,26.7,18,26.7z"/>
              <path d="M38.1,26.7H24.8c-0.6,0-1-0.4-1-1V14c0-0.6,0.4-1,1-1h13.3c0.5,0,1,0.4,1,1v11.7C39.1,26.3,38.6,26.7,38.1,26.7z"/>
            </svg>
          </button>

          {/* Debug Icon */}
          <button
            onClick={() => {
              alert(`Debug Info:\n- Mode: Pro Terminal\n- Windows: ${windows.length}\n- Commands: 20+ available\n- Status: Live\n- Version: v1.0.1`);
            }}
            title="Debug - Show system information"
            style={{
              background: 'none',
              border: '1px solid var(--bg-tertiary)',
              color: 'var(--text-muted)',
              padding: '3px 6px',
              fontSize: '10px',
              cursor: 'pointer',
              borderRadius: '2px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.backgroundColor = 'rgba(82, 196, 26, 0.2)';
              e.currentTarget.style.borderColor = 'var(--success-500)';
              e.currentTarget.style.color = 'var(--success-500)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.backgroundColor = 'transparent';
              e.currentTarget.style.borderColor = '#333';
              e.currentTarget.style.color = '#999';
            }}
          >
            <svg width="10" height="10" viewBox="0 0 512.001 512.001" fill="currentColor">
              <path d="M387.184,124.822c12.533,12.531,23.284,26.853,31.831,42.545c10.825-2.54,21.103-8.026,29.546-16.467c14.721-14.722,20.469-35.016,17.246-54.087c-2.066-12.22-7.819-23.938-17.246-33.368c-9.431-9.432-21.15-15.18-33.368-17.246c-19.072-3.222-39.368,2.526-54.087,17.246c-8.443,8.441-13.931,18.716-16.47,29.543C360.328,101.536,374.65,112.282,387.184,124.822z"/>
              <path d="M124.823,387.179l262.361-262.357c-12.534-12.54-26.856-23.285-42.548-31.834c-26.334-14.354-56.534-22.505-88.635-22.505c-33.797,0-65.481,9.037-92.775,24.829c-28.152,16.289-51.626,39.762-67.914,67.914c-15.792,27.291-24.829,58.978-24.829,92.775C70.489,307.227,91.254,353.605,124.823,387.179z"/>
              <path d="M387.184,124.822L124.823,387.179c33.571,33.57,79.95,54.334,131.179,54.336c33.797,0,65.485-9.037,92.775-24.829c28.152-16.287,51.627-39.762,67.919-67.914c15.787-27.289,24.824-58.978,24.824-92.775c0-32.1-8.151-62.298-22.504-88.632C410.468,151.674,399.718,137.352,387.184,124.822z"/>
              <path d="M239.546,71.204c0.317,1.804,0.486,3.66,0.486,5.554c0,17.705-14.355,32.058-32.06,32.058c-13.857,0-25.666-8.799-30.139-21.112C196.873,78.849,217.663,73.126,239.546,71.204z"/>
              <path d="M176.98,273.155c17.658,0,32.058-14.402,32.058-32.057c0-17.658-14.4-32.057-32.058-32.057c-17.658,0-32.057,14.399-32.057,32.057C144.923,258.753,159.324,273.155,176.98,273.155z"/>
              <path d="M328.464,215.6c17.658,0,32.057-14.402,32.057-32.058c0-17.658-14.4-32.059-32.057-32.059c-17.659,0-32.057,14.402-32.057,32.059C296.407,201.198,310.805,215.6,328.464,215.6z"/>
              <path d="M337.435,376.559c17.659,0,32.057-14.4,32.057-32.057c0-17.656-14.399-32.06-32.057-32.06c-17.658,0-32.057,14.404-32.057,32.06C305.378,362.159,319.777,376.559,337.435,376.559z"/>
              <path d="M161.076,383.365c17.705,0,32.058,14.353,32.058,32.057c0,5.012-1.15,9.756-3.202,13.981c-20.218-7.707-38.748-18.855-54.864-32.724C140.889,388.615,150.373,383.365,161.076,383.365z"/>
              <path d="M114.646,236.21c0.774,0.168,1.546,0.249,2.306,0.249c4.954,0,9.413-3.439,10.513-8.477c5.38-24.673,17.662-47.151,35.517-65.003c4.206-4.205,4.207-11.024,0.001-15.231c-4.202-4.206-11.023-4.207-15.231-0.002c-20.779,20.775-35.072,46.934-41.333,75.647C105.152,229.205,108.836,234.943,114.646,236.21z"/>
              <path d="M397.351,275.788c-5.807-1.27-11.551,2.417-12.816,8.229c-13.076,59.994-67.131,103.537-128.533,103.537c-5.947,0-10.77,4.822-10.77,10.77c0,5.947,4.823,10.77,10.77,10.77c34.815,0,68.921-12.028,96.036-33.869c27.114-21.84,46.129-52.603,53.543-86.622C406.847,282.793,403.163,277.055,397.351,275.788z"/>
              <path d="M113.679,244.526c-5.947,0-10.77,4.822-10.77,10.77v0.704c0,5.948,4.823,10.77,10.77,10.77s10.77-4.822,10.77-10.77v-0.704C124.449,249.348,119.627,244.526,113.679,244.526z"/>
              <path d="M398.325,245.23c-5.947,0-10.77,4.822-10.77,10.77v0.704c0,5.948,4.823,10.77,10.77,10.77c5.948,0,10.77-4.822,10.77-10.77v-0.704C409.095,250.052,404.272,245.23,398.325,245.23z"/>
              <path d="M501.23,245.23h-49.241c-1.383-25.458-7.646-49.636-17.856-71.624c8.168-3.604,15.646-8.7,22.039-15.092c13.619-13.619,21.152-31.693,21.258-50.933h23.8c5.947,0,10.77-4.823,10.77-10.77c0-5.949-4.823-10.77-10.77-10.77h-26.897c-3.417-11.29-9.583-21.635-18.161-30.212c-8.576-8.578-18.921-14.745-30.213-18.162V10.771c0-5.948-4.822-10.77-10.77-10.77c-5.947,0-10.77,4.822-10.77,10.77v23.8c-19.239,0.108-37.314,7.642-50.928,21.258c-6.393,6.394-11.491,13.872-15.094,22.042c-21.988-10.211-46.166-16.474-71.624-17.857V10.771c0-5.948-4.823-10.77-10.77-10.77h-33.913c-5.948,0-10.77,4.822-10.77,10.77s4.822,10.77,10.77,10.77h23.143v38.473c-27.897,1.515-54.261,8.883-77.886,20.893l-24.63-42.665c-1.041-1.805-2.531-3.183-4.241-4.098c-2.882-1.543-6.4-1.75-9.532-0.326c-0.317,0.144-0.631,0.303-0.939,0.48c-0.003,0.002-0.005,0.004-0.009,0.005l-29.362,16.95c-5.151,2.975-6.918,9.561-3.943,14.712c1.995,3.456,5.614,5.387,9.338,5.387c1.826,0,3.678-0.465,5.373-1.444l20.044-11.57l19.269,33.375c-22.693,14.87-42.129,34.307-57,56.999l-42.701-24.655c-5.152-2.976-11.738-1.208-14.712,3.942L17.344,157.37c-2.975,5.151-1.209,11.737,3.942,14.712c1.696,0.979,3.548,1.444,5.374,1.444c3.722,0,7.343-1.931,9.338-5.386l11.571-20.044l33.339,19.248c-12.011,23.625-19.378,49.988-20.893,77.885H10.77c-5.948,0-10.77,4.822-10.77,10.77c0,0.014,0.002,0.027,0.002,0.041v33.873c0,5.947,4.823,10.77,10.77,10.77c5.948,0,10.77-4.823,10.77-10.77V266.77h38.473c2.681,49.327,23.66,93.86,56.248,126.928c0.29,0.38,0.599,0.75,0.946,1.099c0.347,0.347,0.718,0.655,1.099,0.945c33.066,32.588,77.6,53.566,126.926,56.248v38.472h-23.143c-5.948,0-10.77,4.822-10.77,10.77c0,5.947,4.822,10.77,10.77,10.77h33.913c5.947,0,10.77-4.823,10.77-10.77v-49.241c27.897-1.516,54.261-8.883,77.885-20.894l19.247,33.339l-20.041,11.571c-5.151,2.975-6.917,9.561-3.942,14.712c1.995,3.455,5.614,5.386,9.338,5.386c1.827,0,3.679-0.465,5.374-1.444l29.361-16.953c0.002-0.001,0.004-0.002,0.006-0.003c1.61-0.928,2.89-2.212,3.795-3.693c1.997-3.258,2.191-7.478,0.146-11.02c-0.002-0.003-0.004-0.005-0.005-0.009l-24.647-42.694c22.691-14.869,42.129-34.306,57-56.999l33.375,19.268l-11.572,20.041c-2.975,5.151-1.209,11.738,3.942,14.713c1.695,0.979,3.548,1.444,5.374,1.444c3.722,0,7.342-1.931,9.338-5.386l16.959-29.37c1.674-2.897,1.847-6.25,0.761-9.158c-0.001-0.001-0.001-0.002-0.002-0.003c-0.12-0.322-0.255-0.639-0.406-0.948c-0.909-1.865-2.364-3.488-4.295-4.604l-42.668-24.631c12.011-23.625,19.378-49.988,20.893-77.884h38.471v23.144c0,5.947,4.823,10.77,10.77,10.77s10.77-4.823,10.77-10.77v-33.914C512,250.052,507.177,245.23,501.23,245.23z M368.721,71.06c9.645-9.646,22.469-14.959,36.109-14.959s26.464,5.313,36.109,14.96c9.647,9.646,14.96,22.47,14.96,36.111c0,13.64-5.313,26.464-14.959,36.111c-4.864,4.863-10.612,8.666-16.903,11.218c-16.489-27.277-39.257-50.045-66.535-66.539C360.055,81.672,363.859,75.924,368.721,71.06z M256.002,81.254c44.31,0,84.813,16.586,115.649,43.865L125.123,371.652C97.843,340.815,81.256,300.312,81.256,256C81.255,159.646,159.646,81.254,256.002,81.254z M256.002,430.747c-44.31,0-84.812-16.586-115.649-43.865l246.53-246.531c27.278,30.836,43.865,71.339,43.865,115.65C430.748,352.356,352.357,430.747,256.002,430.747z"/>
            </svg>
          </button>
          

          {/* Settings Icon */}
          <button
            onClick={() => setShowSettings(true)}
            title="Settings - Manage profiles, layouts, and preferences"
            style={{
              background: 'var(--success-500)',
              border: '1px solid var(--success-500)',
              color: '#000',
              padding: '3px 6px',
              fontSize: '10px',
              cursor: 'pointer',
              borderRadius: '2px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontWeight: 'bold',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.background = '#73d13d';
              e.currentTarget.style.color = '#000';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.background = 'var(--success-500)';
              e.currentTarget.style.color = '#000';
            }}
          >
            ⚙
          </button>
        </div>

        {/* Center - Active Windows Count */}
        <div style={{ 
          color: 'var(--text-muted)', 
          fontSize: '10px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <span>Windows: {windows.length}</span>
          {windows.length > 0 && (
            <span style={{ color: 'var(--primary-500)' }}>
              • Active: {windows.filter(w => !w.isMinimized).length}
            </span>
          )}
        </div>

        {/* Right Side - Status & Version */}
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          gap: '12px',
          color: 'var(--text-muted)',
          fontSize: '10px',
          fontFamily: 'Monaco, Consolas, "Courier New", monospace'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
            <div style={{ 
              width: '6px', 
              height: '6px', 
              borderRadius: '50%', 
              backgroundColor: 'var(--success-500)',
              animation: 'pulse 2s infinite'
            }}></div>
            <span style={{ color: 'var(--success-500)', fontWeight: 'bold' }}>Live</span>
          </div>
          
          <div>v1.0.1</div>
        </div>
      </div>

      {/* Pro Terminal Settings Modal */}
      <ProTerminalSettings
        isOpen={showSettings}
        onClose={() => setShowSettings(false)}
        onProfileChange={(profile) => {
          setCurrentProfile(profile);
          console.log('Profile changed to:', profile.name);
        }}
        currentProfile={currentProfile}
      />
      </div>
    </ThemeProvider>
  )
}


export default ProTerminalWithTaskbar