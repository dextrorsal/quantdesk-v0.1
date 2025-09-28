import React from 'react'
import { useNavigate } from 'react-router-dom'
import { ThemeProvider, useTheme } from '../frontend/src/contexts/ThemeContext'

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

  // Sample instruments data (like Godel Terminal)
  const instruments = [
    { symbol: 'NVDA', name: 'NVIDIA Corporation', price: 176.37, change: 3.57, changePercent: 2.07, type: 'EQ' },
    { symbol: 'AAPL', name: 'Apple Inc.', price: 191.32, change: -0.68, changePercent: -0.35, type: 'EQ' },
    { symbol: 'MSFT', name: 'Microsoft Corporation', price: 508.95, change: -1.07, changePercent: -0.21, type: 'EQ' },
    { symbol: 'MSTR', name: 'MicroStrategy Incorporated', price: 350.00, change: 20.29, changePercent: 6.15, type: 'EQ' },
    { symbol: 'BTC', name: 'Bitcoin', price: 67543.21, change: 1234.56, changePercent: 1.86, type: 'CRYPTO' },
    { symbol: 'ETH', name: 'Ethereum', price: 3245.67, change: -45.23, changePercent: -1.37, type: 'CRYPTO' },
    { symbol: 'SOL', name: 'Solana', price: 246.78, change: 12.34, changePercent: 5.26, type: 'CRYPTO' },
    { symbol: 'TSLA', name: 'Tesla Inc.', price: 248.50, change: 5.20, changePercent: 2.14, type: 'EQ' },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 142.85, change: -1.25, changePercent: -0.87, type: 'EQ' },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 185.30, change: 2.45, changePercent: 1.34, type: 'EQ' }
  ]

  // Sample news data
  const newsStories = [
    { headline: 'Nvidia (NVDA) Reports Strong Q3 Earnings', ticker: 'NVDA', time: '2 hours ago', source: 'Reuters' },
    { headline: 'Bitcoin Surges Past $67,000 as Institutional Adoption Grows', ticker: 'BTC', time: '4 hours ago', source: 'CoinDesk' },
    { headline: 'Apple (AAPL) Announces New AI Features for iPhone', ticker: 'AAPL', time: '6 hours ago', source: 'TechCrunch' },
    { headline: 'Microsoft (MSFT) Azure Revenue Exceeds Expectations', ticker: 'MSFT', time: '8 hours ago', source: 'Bloomberg' },
    { headline: 'Tesla (TSLA) Stock Rises on Strong Delivery Numbers', ticker: 'TSLA', time: '10 hours ago', source: 'MarketWatch' },
    { headline: 'Solana (SOL) Network Activity Hits New Highs', ticker: 'SOL', time: '12 hours ago', source: 'The Block' },
    { headline: 'MicroStrategy (MSTR) Adds More Bitcoin to Treasury', ticker: 'MSTR', time: '1 day ago', source: 'Forbes' },
    { headline: 'Amazon (AMZN) AWS Growth Accelerates in Q3', ticker: 'AMZN', time: '1 day ago', source: 'CNBC' }
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
    // Core Market Data & Trading
    { command: 'CF', description: 'SEC Filings', category: 'Research' },
    { command: 'N', description: 'Realtime and historical news', category: 'News' },
    { command: 'HELP', description: 'Documentation about Godel Terminal', category: 'System' },
    { command: 'S', description: 'Keyboard Shortcuts', category: 'System' },
    { command: 'ORG', description: 'Organization Billing', category: 'Account' },
    { command: 'CHANGE', description: 'Godel Terminal ChangeLog', category: 'System' },
    { command: 'CHAT', description: 'Live Chat', category: 'Social' },
    { command: 'QM', description: 'Quote Monitor', category: 'Market Data' },
    { command: 'BT', description: 'Strategy Backtesting', category: 'Trading' },
    { command: 'PF', description: 'Portfolio Performance', category: 'Trading' },
    { command: 'AL', description: 'Price Alerts & Notifications', category: 'Trading' },
    { command: 'ACCT', description: 'Account Management', category: 'Account' },
    { command: 'PDF', description: 'Settings for your user account', category: 'Settings' },
    { command: 'TAS', description: 'Time and sale transaction data', category: 'Market Data' },
    { command: 'EQS', description: 'Equity Screener', category: 'Screening' },
    { command: 'IPO', description: 'List of upcoming and recent IPOs', category: 'Market Data' },
    { command: 'AL', description: 'Set desktop alerts for securities', category: 'Alerts' },
    { command: 'NOTE', description: 'Rich Text Notes Editor', category: 'Tools' },
    { command: 'PAT', description: 'Pattern search with forward-returns forecast', category: 'Analysis' },
    { command: 'PRT', description: 'Systematic pattern search (batch + ranking + realized outcome)', category: 'Analysis' },
    { command: 'MOST', description: 'Most active securities', category: 'Market Data' },
    { command: 'CALC', description: 'Financial Calculator', category: 'Tools' },
    { command: 'HMS', description: 'Compare multiple securities historically', category: 'Analysis' },
    { command: 'BROK', description: 'Connect your brokerage to godel', category: 'Trading' },
    { command: 'TOP', description: 'Top News from reuters today', category: 'News' },
    { command: 'WEI', description: 'Monitor and compare real-time prices for the world\'s equity indices', category: 'Market Data' },
    { command: 'GR', description: 'Graph Relationship between two securities over time', category: 'Analysis' },
    { command: 'MARTIN', description: 'Various files and links', category: 'Tools' },
    { command: 'NEJM', description: 'Medical Journal', category: 'Research' },
    { command: 'RES', description: 'Research Reports', category: 'Research' },
    { command: 'XPRT', description: 'Expert Reports', category: 'Research' },
    { command: 'CITADEL', description: 'Overview of Citadel', category: 'Research' },
    { command: 'CN', description: 'Realtime and historical news', category: 'News' },
    { command: 'NH', description: 'Realtime and historical news', category: 'News' },
    { command: 'NI', description: 'NI [Search Term] - opens news with search', category: 'News' },
    { command: 'ERR', description: 'Report bugs and get support', category: 'System' },
    
    // Additional System Commands
    { command: 'CLEAR', description: 'Clear all windows', category: 'System' },
    { command: 'CHART', description: 'Advanced charting with technical indicators', category: 'Analysis' },
    { command: 'LAYOUT', description: 'Save/Load window layouts', category: 'System' }
  ];

  // Create a new window
  const createWindow = (type: string, title: string, content?: any) => {
    const newWindow = {
      id: `${type}-${Date.now()}`,
      type,
      title,
      x: Math.random() * (window.innerWidth - 400) + 50,
      y: Math.random() * (window.innerHeight - 300) + 100,
      width: type === 'CHART' ? 600 : type === 'QM' ? 350 : 500,
      height: type === 'CHART' ? 400 : type === 'QM' ? 500 : 300,
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
        createWindow('QM', 'Live Quote Monitor', marketData);
        break;
      case 'N':
      case 'NEWS':
        createWindow('NEWS', 'Market News', [
          { headline: 'Market Opens Higher on Tech Earnings', time: '09:30', source: 'Reuters' },
          { headline: 'Fed Signals Potential Rate Changes', time: '09:15', source: 'Bloomberg' },
          { headline: 'AAPL Reports Strong Q3 Results', time: '08:45', source: 'CNBC' },
          { headline: 'Crypto Markets Show Volatility', time: '08:30', source: 'CoinDesk' }
        ]);
        break;
      case 'CHART':
        // Fetch chart data for BTC by default
        fetchChartData('BTCUSDT', '1h').then(data => {
          createWindow('CHART', 'Advanced Chart - BTCUSDT', data);
        });
        break;
      case 'CHAT':
        createWindow('CHAT', 'Community Chat', [
          { user: 'TraderBot', message: 'Welcome to QuantDesk Chat!', time: '14:30' },
          { user: 'CryptoKing', message: 'SOL looking bullish today', time: '14:32' },
          { user: 'AlgoTrader', message: 'My bot just triggered a buy signal', time: '14:35' }
        ]);
        break;
      case 'CALC':
        createWindow('CALC', 'Financial Calculator', { 
          functions: ['Position Size', 'Risk/Reward', 'Profit/Loss', 'Compound Interest']
        });
        break;
      case 'CF':
        createWindow('CF', 'SEC Filings', [
          { company: 'AAPL', filing: '10-K', date: '2024-01-15', status: 'Filed' },
          { company: 'GOOGL', filing: '10-Q', date: '2024-01-10', status: 'Filed' },
          { company: 'MSFT', filing: '8-K', date: '2024-01-08', status: 'Filed' }
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
          { company: 'TechCorp', date: '2024-02-15', priceRange: '$18-22', status: 'Upcoming' },
          { company: 'DataInc', date: '2024-02-20', priceRange: '$25-30', status: 'Upcoming' }
        ]);
        break;
      case 'NOTE':
        createWindow('NOTE', 'Trading Notes', {
          content: 'Market Analysis - Jan 2024\n\n• Tech sector showing strength\n• Watch for Fed announcements\n• SOL breakout pattern forming'
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
          { title: 'Q4 2024 Market Outlook', analyst: 'Goldman Sachs', date: '2024-01-15', rating: 'BUY' },
          { title: 'Tech Sector Deep Dive', analyst: 'Morgan Stanley', date: '2024-01-12', rating: 'OVERWEIGHT' },
          { title: 'Crypto Winter Analysis', analyst: 'JPMorgan', date: '2024-01-10', rating: 'NEUTRAL' },
          { title: 'AI Stocks Valuation Report', analyst: 'Bank of America', date: '2024-01-08', rating: 'BUY' }
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
          next_billing: '2024-02-15'
        });
        break;
      case 'ACCT':
        createWindow('ACCT', 'Account Management', {
          user: 'QuantDesk Trader',
          email: 'trader@QuantDesk.com',
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
          { version: 'v1.0.1', date: '2024-01-15', changes: ['Added Pro Terminal mode', 'Improved window management', 'Enhanced backtick menu'] },
          { version: 'v1.0.0', date: '2024-01-01', changes: ['Initial release', 'Basic trading features', 'Paper trading framework'] }
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
          { name: 'Risk Management Guide', type: 'Link', url: 'https://QuantDesk.com/risk' },
          { name: 'API Documentation', type: 'Link', url: 'https://docs.QuantDesk.com' }
        ]);
        break;
      case 'NEJM':
        createWindow('NEJM', 'Medical Journal Research', [
          { title: 'AI in Healthcare Investing', date: '2024-01-10', relevance: 'Biotech sector analysis' },
          { title: 'Pharmaceutical Patent Cliff', date: '2024-01-05', relevance: 'Drug company valuations' },
          { title: 'Medical Device Innovation', date: '2023-12-28', relevance: 'MedTech investment opportunities' }
        ]);
        break;
      case 'GH':
        createWindow('GH', 'GitHub News Feed', [
          { repo: 'QuantDesk-trading/core', event: 'New release v2.1.0', time: '2 hours ago' },
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
          contact: 'support@QuantDesk.com',
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
          { title: 'Crypto Market Analysis Q4 2024', author: 'QuantDesk Research', date: '2024-01-15', rating: 'BUY' },
          { title: 'DeFi Protocol Valuation Models', author: 'Blockchain Analytics', date: '2024-01-12', rating: 'HOLD' },
          { title: 'Meme Coin Trend Analysis', author: 'Social Trading Desk', date: '2024-01-10', rating: 'SPECULATIVE' }
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
    switch (window.type) {
      case 'QM':
        const marketQuotes = Object.entries(marketData).slice(0, 15);
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
                {marketQuotes.length} symbols • {new Date().toLocaleTimeString()}
              </span>
            </div>
            
            <table style={{ width: '100%', fontSize: '11px', fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                  <th style={{ textAlign: 'left', padding: '4px' }}>Symbol</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Price</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>24h%</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Volume</th>
                  <th style={{ textAlign: 'right', padding: '4px' }}>Spread</th>
                </tr>
              </thead>
              <tbody>
                {marketQuotes.length > 0 ? marketQuotes.map(([symbol, quote]: [string, any], idx: number) => (
                  <tr key={idx} style={{ borderBottom: '1px solid var(--bg-tertiary)' }}>
                    <td style={{ padding: '4px', color: 'var(--success-500)', fontWeight: 'bold' }}>
                      {symbol.replace('USDT', '')}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--text-primary)' }}>
                      ${quote.price ? quote.price.toFixed(quote.price < 1 ? 4 : 2) : 'N/A'}
                    </td>
                    <td style={{ 
                      padding: '4px', 
                      textAlign: 'right',
                      color: (quote.change_24h_percent || 0) >= 0 ? 'var(--success-500)' : 'var(--danger-500)',
                      fontWeight: 'bold'
                    }}>
                      {(quote.change_24h_percent || 0) >= 0 ? '+' : ''}{(quote.change_24h_percent || 0).toFixed(2)}%
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--primary-500)' }}>
                      {quote.volume_24h ? (quote.volume_24h / 1000000).toFixed(1) + 'M' : 'N/A'}
                    </td>
                    <td style={{ padding: '4px', textAlign: 'right', color: 'var(--warning-500)' }}>
                      {quote.bid && quote.ask ? ((quote.ask - quote.bid) / quote.price * 100).toFixed(3) + '%' : 'N/A'}
                    </td>
                  </tr>
                )) : (
                  <tr>
                    <td colSpan={5} style={{ 
                      padding: '20px', 
                      textAlign: 'center', 
                      color: 'var(--text-muted)',
                      fontStyle: 'italic'
                    }}>
                      Connecting to market data stream...
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        );

      case 'NEWS':
        return (
          <div style={{ padding: '10px', height: '100%', overflow: 'auto' }}>
            {window.content?.map((news: any, idx: number) => (
              <div key={idx} style={{ 
                marginBottom: '10px', 
                padding: '8px', 
                borderLeft: '3px solid var(--primary-500)',
                backgroundColor: 'var(--bg-secondary)'
              }}>
                <div style={{ fontSize: '12px', fontWeight: 'bold' }}>{news.headline}</div>
                <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginTop: '4px' }}>
                  {news.time} - {news.source}
                </div>
              </div>
            ))}
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

      case 'CHAT':
        return (
          <div style={{ padding: '10px', height: '100%', display: 'flex', flexDirection: 'column' }}>
            <div style={{ flex: 1, overflow: 'auto', marginBottom: '10px' }}>
              {window.content?.map((msg: any, idx: number) => (
                <div key={idx} style={{ marginBottom: '8px', fontSize: '11px' }}>
                  <span style={{ color: 'var(--primary-500)', fontWeight: 'bold' }}>{msg.user}</span>
                  <span style={{ color: 'var(--text-muted)', marginLeft: '8px', fontSize: '10px' }}>{msg.time}</span>
                  <div style={{ color: 'var(--text-primary)', marginTop: '2px' }}>{msg.message}</div>
                </div>
              ))}
            </div>
            <input 
              placeholder="Type a message..."
              style={{ 
                width: '100%', 
                padding: '6px', 
                background: '#000', 
                border: '1px solid #333', 
                color: 'var(--text-primary)',
                fontSize: '11px'
              }}
            />
          </div>
        );

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
          {getUsername()}@QuantifyPro:~$
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

          {/* Profile Button */}
          <button
            onClick={() => {
              alert('Profile/Login functionality coming soon!');
            }}
            style={{
              background: 'none',
              border: '1px solid var(--bg-tertiary)',
              color: 'var(--text-muted)',
              padding: '4px 8px',
              fontSize: '10px',
              cursor: 'pointer',
              borderRadius: '3px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              transition: 'all 0.2s ease'
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = 'var(--success-500)';
              e.currentTarget.style.color = 'var(--success-500)';
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = '#333';
              e.currentTarget.style.color = '#999';
            }}
          >
            👤 Profile
          </button>
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

      {/* Settings Modal - Placeholder */}
      {showSettings && (
        <div style={{
          position: 'fixed',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          backgroundColor: 'var(--bg-secondary)',
          border: '1px solid #333',
          padding: '20px',
          borderRadius: '8px',
          zIndex: 10000
        }}>
          <h3 style={{ color: 'var(--text-primary)', margin: '0 0 15px 0' }}>Pro Terminal Settings</h3>
          <p style={{ color: '#999', margin: '0 0 15px 0' }}>Settings panel coming soon...</p>
          <button 
            onClick={() => setShowSettings(false)}
            style={{
              backgroundColor: '#333',
              border: 'none',
              color: 'var(--text-primary)',
              padding: '8px 16px',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Close
          </button>
        </div>
      )}
      </div>
    </ThemeProvider>
  )
}


export default ProTerminalWithTaskbar
