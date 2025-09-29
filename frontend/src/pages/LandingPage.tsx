import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { 
  Terminal, 
  Zap, 
  BarChart3, 
  Shield, 
  TrendingUp, 
  Activity,
  Users,
  DollarSign,
  Monitor,
  Command,
  Cpu,
  Database,
  Lock,
  ArrowRight,
  Layers,
  Brain,
  Settings,
  TestTube,
  Target,
  AlertTriangle,
  Wrench,
  Twitter,
  ExternalLink
} from 'lucide-react'
import LivingBackground from '../components/LivingBackground'
import { usePrice } from '../contexts/PriceContext'
import { PriceDisplay } from '../components/PriceDisplay'
import TradingViewTickerTape from '../components/TradingViewTickerTape'
import { DOCS_URLS, GITHUB_URL } from '../utils/docsConfig'

const LandingPage: React.FC = () => {
  const [currentTime, setCurrentTime] = useState(new Date())
  const [terminalLines, setTerminalLines] = useState<string[]>([])
  const [isTyping, setIsTyping] = useState(false)
  
  // Use centralized price system
  const { connectionStatus } = usePrice()
  
  // Define symbols to display
  const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 'MATIC/USDT', 'AVAX/USDT', 'LINK/USDT']

  // Connection status indicator
  const getConnectionStatus = () => {
    switch (connectionStatus) {
      case 'connected':
        return (
          <div className="flex items-center gap-2 text-green-400 text-sm">
            <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
            <span>Live</span>
          </div>
        )
      case 'connecting':
        return (
          <div className="flex items-center gap-2 text-yellow-400 text-sm">
            <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
            <span>Connecting...</span>
          </div>
        )
      default:
        return (
          <div className="flex items-center gap-2 text-red-400 text-sm">
            <div className="w-2 h-2 bg-red-400 rounded-full"></div>
            <span>Disconnected</span>
          </div>
        )
    }
  }

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date())
      // Real-time data comes from WebSocket, no need to simulate
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  useEffect(() => {
    const commands = [
      'quantdesk@solana:~$ ./quantdesk --init',
      'Initializing QuantDesk Terminal v1.0.1...',
      '✓ Solana RPC connection established',
      '✓ Order book synchronization complete',
      '✓ Price feed integration active',
      '✓ Security protocols verified',
      '✓ Trading engine ready',
      '✓ ML algorithms loaded',
      '✓ Backtesting engine initialized',
      'quantdesk@solana:~$ ./quantdesk --status',
      'System Status: OPERATIONAL',
      'Active Markets: 24',
      'Live Users: 12,400+',
      'Total Volume: $2.1B',
      'ML Models: 8 Active',
      'Backtests: 1,247 Completed',
      'quantdesk@solana:~$ ./quantdesk --launch'
    ]

    let lineIndex = 0
    let charIndex = 0
    const interval = setInterval(() => {
      if (lineIndex < commands.length) {
        const currentLine = commands[lineIndex]
        if (charIndex < currentLine.length) {
          setTerminalLines(prev => {
            const newLines = [...prev]
            if (!newLines[lineIndex]) newLines[lineIndex] = ''
            newLines[lineIndex] = currentLine.slice(0, charIndex + 1)
            return newLines
          })
          charIndex++
        } else {
          lineIndex++
          charIndex = 0
        }
      } else {
        setIsTyping(false)
        clearInterval(interval)
      }
    }, 50)

    setIsTyping(true)
    return () => clearInterval(interval)
  }, [])

const stats = [
    { label: 'Total Volume', value: '$2.1B', change: '+12.5%', icon: DollarSign },
    { label: 'Open Interest', value: '$89.2M', change: '+8.2%', icon: Activity },
    { label: 'Active Users', value: '12.4K', change: '+15.3%', icon: Users },
    { label: 'ML Models', value: '8', change: '+2', icon: Brain },
]

const features = [
  {
    icon: <Zap className="h-6 w-6" />,
    title: 'Ultra-Low Latency',
    desc: 'Sub-millisecond execution on Solana',
      command: 'latency --check',
      color: 'text-yellow-400',
      status: 'operational'
    },
    {
      icon: <Brain className="h-6 w-6" />,
      title: 'ML Algorithms',
      desc: 'Customizable parameters and signal generation',
      command: 'ml --backtest --params',
      color: 'text-purple-400',
      status: 'beta'
    },
    {
      icon: <TestTube className="h-6 w-6" />,
      title: 'Backtesting Engine',
      desc: 'Historical data analysis and strategy validation',
      command: 'backtest --strategy --historical',
      color: 'text-green-400',
      status: 'operational'
  },
  {
    icon: <BarChart3 className="h-6 w-6" />,
    title: 'Advanced Trading',
    desc: 'Professional tools and deep liquidity',
      command: 'trading --advanced',
      color: 'text-blue-400',
      status: 'operational'
    },
    {
      icon: <Target className="h-6 w-6" />,
      title: 'Signal Generation',
      desc: 'AI-powered market signals and alerts',
      command: 'signals --generate --ai',
      color: 'text-orange-400',
      status: 'beta'
  },
  {
    icon: <Shield className="h-6 w-6" />,
    title: 'Battle-Tested',
    desc: 'Audited smart contracts and security',
      command: 'security --audit',
      color: 'text-green-400',
      status: 'operational'
    },
  ]

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { 
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }


  return (
    <div className="min-h-screen bg-black text-white font-mono relative">
      {/* Living Background System */}
      <LivingBackground />
      
      {/* Construction Warning Banner */}
      <div className="relative z-30">
        <div className="bg-gradient-to-r from-yellow-500/90 via-orange-500/90 to-yellow-500/90 backdrop-blur-sm border-b-2 border-yellow-400/50 px-6 py-2 animate-pulse">
          <div className="max-w-7xl mx-auto text-center">
            <div className="flex items-center justify-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-black animate-bounce" />
              <div>
                <p className="text-black font-bold text-sm">
                  ⚠️ UNDER CONSTRUCTION - NOT READY FOR LIVE TRADING ⚠️
                </p>
                <p className="text-black/80 font-medium text-xs">
                  Testing will commence soon. Follow our Twitter with alerts on!
                </p>
              </div>
              <AlertTriangle className="h-4 w-4 text-black animate-bounce" />
            </div>
          </div>
        </div>
      </div>
      
      {/* Animated Background Elements */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="absolute inset-0 bg-gradient-to-b from-black via-gray-900/30 to-black" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.08)_0%,transparent_70%)]" />
        
        {/* Grid Pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="w-full h-full" style={{
            backgroundImage: `radial-gradient(circle at 1px 1px, rgba(59,130,246,0.4) 1px, transparent 0)`,
            backgroundSize: '30px 30px'
          }} />
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-20">
        {/* TradingView Ticker Tape */}
        <div className="bg-black/80 backdrop-blur-sm border-b border-gray-800">
          <TradingViewTickerTape />
        </div>

        {/* Terminal Header Bar - Right Under Ticker Tape */}
        <div className="bg-black/95 backdrop-blur-sm border-b border-gray-800 px-6 py-3 flex items-center justify-between text-sm sticky top-0 z-50">
          <div className="flex items-center space-x-6">
            <div className="flex items-center space-x-2">
              <Terminal className="h-5 w-5 text-blue-400" />
              <span className="text-blue-400 font-semibold">QuantDesk Terminal</span>
            </div>
            <div className="text-gray-400">v1.0.1</div>
            <div className="flex items-center space-x-2">
              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
              <span className="text-green-400">LIVE ON SOLANA</span>
            </div>
            <div className="flex items-center space-x-1 bg-orange-500/20 px-2 py-1 rounded">
              <Wrench className="h-3 w-3 text-orange-400" />
              <span className="text-orange-400 text-xs">BETA</span>
            </div>
          </div>
          
          {/* Centered Twitter Follow Button */}
          <div className="absolute left-1/2 transform -translate-x-1/2">
            <a 
              href="https://twitter.com/quantdeskapp" 
              target="_blank" 
              rel="noopener noreferrer"
              className="group flex items-center space-x-2 bg-black border border-gray-700 hover:border-blue-400 text-white px-4 py-2 rounded text-sm font-semibold transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/25"
            >
              <Twitter className="h-4 w-4 text-blue-400 group-hover:scale-110 transition-transform" />
              <span>Follow with Alerts On</span>
              <ExternalLink className="h-3 w-3 group-hover:translate-x-1 transition-transform" />
            </a>
          </div>
          
          <div className="flex items-center space-x-4">
            <div className="text-gray-400">{formatTime(currentTime)}</div>
            <div className="flex items-center space-x-2">
              <Cpu className="h-4 w-4 text-blue-400" />
              <span className="text-xs">System Operational</span>
            </div>
            <a 
              href={DOCS_URLS.baseUrl} 
              target="_blank" 
              rel="noopener noreferrer"
              className="px-3 py-1 text-xs rounded border border-blue-400 text-blue-400 hover:text-white hover:border-blue-500 hover:bg-blue-500 transition-all duration-300"
              aria-label="Documentation"
            >
              📚 Docs
            </a>
          </div>
        </div>

        {/* Hero Section */}
        <section className="relative overflow-hidden py-20">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
            {/* Logo */}
            <div className="mb-8">
              <img 
                src="/quantdesk-logo.png" 
                alt="QuantDesk" 
                  className="h-20 mx-auto mb-6"
              />
            </div>
            
            {/* Main Headline */}
              <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-8">
              The Future of{' '}
                <span className="bg-gradient-to-r from-blue-600 via-orange-600 to-blue-600 bg-clip-text text-transparent">
                Trading
              </span>
            </h1>
            
            {/* Subtitle */}
              <p className="text-xl text-gray-300 max-w-3xl mx-auto mb-12 leading-relaxed">
                Professional-grade perpetual futures trading infrastructure on Solana. 
                Built for traders who demand institutional-level performance with AI-powered insights.
            </p>
            
            {/* CTA Buttons */}
              <div className="flex items-center justify-center gap-6 mb-16">
              <Link 
                to="/lite#trading" 
                  className="group flex items-center space-x-2 bg-black hover:bg-blue-400 border border-blue-400 hover:border-blue-400 text-blue-400 hover:text-black px-8 py-4 rounded-lg font-mono font-semibold transition-all duration-300 hover:shadow-lg hover:shadow-blue-400/50"
              >
                  <TrendingUp className="h-5 w-5" />
                  <span>Start Trading</span>
                  <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
              </Link>
              <Link 
                  to="/pro" 
                  className="group flex items-center space-x-2 bg-black hover:bg-orange-400 border border-orange-400 hover:border-orange-400 text-orange-400 hover:text-white px-8 py-4 rounded-lg font-mono font-semibold transition-all duration-300 hover:shadow-lg hover:shadow-orange-400/50"
              >
                  <Terminal className="h-5 w-5" />
                  <span>Launch Terminal</span>
                  <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
              </Link>
            </div>
          </div>

            {/* Live Terminal Demo */}
            <div className="max-w-4xl mx-auto mb-16">
              <div className="bg-black border border-gray-800 rounded-xl overflow-hidden shadow-2xl">
                <div className="bg-black px-4 py-3 border-b border-gray-800 flex items-center space-x-2">
                  <div className="flex space-x-2">
                    <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                    <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  </div>
                  <span className="text-sm text-gray-400 ml-4">quantdesk-terminal</span>
                  <div className="ml-auto flex items-center space-x-2">
                    <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                    <span className="text-xs text-green-400">ML Active</span>
                  </div>
                </div>
                <div className="p-6 font-mono text-sm">
                  {terminalLines.map((line, index) => (
                    <div key={index} className="mb-1">
                      {line.includes('✓') ? (
                        <span className="text-green-400">{line}</span>
                      ) : line.includes('quantdesk@solana') ? (
                        <span className="text-green-400">{line}</span>
                      ) : line.includes('System Status') ? (
                        <span className="text-blue-400 font-semibold">{line}</span>
                      ) : line.includes('ML Models') || line.includes('Backtests') ? (
                        <span className="text-purple-400">{line}</span>
                      ) : line.includes('Active Markets') || line.includes('Live Users') || line.includes('Total Volume') ? (
                        <span className="text-yellow-400">{line}</span>
                      ) : (
                        <span className="text-gray-300">{line}</span>
                      )}
                      {isTyping && index === terminalLines.length - 1 && (
                        <span className="text-blue-400 animate-pulse">_</span>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 mb-16">
              {stats.map((stat) => (
                <div key={stat.label} className="bg-black border border-gray-800 rounded-xl p-6 hover:border-gray-700 transition-all duration-300 group">
                  <div className="flex items-center space-x-3 mb-3">
                    <stat.icon className="h-5 w-5 text-blue-400" />
                    <span className="text-sm text-gray-400 uppercase tracking-wider">{stat.label}</span>
                  </div>
                  <div className="text-2xl font-bold text-white mb-1">{stat.value}</div>
                  <div className="text-sm text-green-400 font-medium">{stat.change}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-6">
              Built for Professional Traders
            </h2>
              <p className="text-lg text-gray-300 max-w-3xl mx-auto">
              Every feature designed with institutional-grade performance and security in mind.
                Command-line precision meets modern trading infrastructure with AI-powered insights.
            </p>
          </div>
          
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              {features.map((feature, index) => (
              <div 
                key={feature.title}
                  className="group bg-black border border-gray-800 rounded-xl p-6 hover:border-gray-700 transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/10 relative"
                >
                  {feature.status === 'beta' && (
                    <div className="absolute -top-2 -right-2 bg-orange-500 text-white text-xs px-2 py-1 rounded-full flex items-center space-x-1">
                      <Wrench className="h-3 w-3" />
                      <span>BETA</span>
                    </div>
                  )}
                  <div className={`${feature.color} mb-4 group-hover:scale-110 transition-transform duration-300`}>
                    {feature.icon}
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-3">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-gray-400 mb-4">
                    {feature.desc}
                  </p>
                  <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded border border-gray-700">
                    {feature.command}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

        {/* ML & Backtesting Section */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-white mb-6">
                AI-Powered Trading Intelligence
              </h2>
              <p className="text-lg text-gray-300 max-w-3xl mx-auto">
                Advanced machine learning algorithms with customizable parameters, 
                signal generation, and comprehensive backtesting capabilities.
              </p>
            </div>
            
            <div className="grid lg:grid-cols-2 gap-12">
              <div className="bg-black border border-gray-800 rounded-xl p-8 relative">
                <div className="absolute -top-2 -right-2 bg-purple-500 text-white text-xs px-2 py-1 rounded-full flex items-center space-x-1">
                  <Wrench className="h-3 w-3" />
                  <span>BETA</span>
                </div>
                <div className="flex items-center space-x-3 mb-6">
                  <Brain className="h-6 w-6 text-purple-400" />
                  <h3 className="text-xl font-semibold text-white">Machine Learning Engine</h3>
                </div>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Active Models:</span>
                    <span className="text-white font-medium">8</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Signal Accuracy:</span>
                    <span className="text-green-400 font-medium">87.3%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Custom Parameters:</span>
                    <span className="text-blue-400 font-medium">Unlimited</span>
                  </div>
                  <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded mt-4 border border-gray-700">
                    ml.train --custom --optimize
                  </div>
                </div>
              </div>
              
              <div className="bg-black border border-gray-800 rounded-xl p-8 relative">
                <div className="absolute -top-2 -right-2 bg-green-500 text-white text-xs px-2 py-1 rounded-full flex items-center space-x-1">
                  <Wrench className="h-3 w-3" />
                  <span>BETA</span>
                </div>
                <div className="flex items-center space-x-3 mb-6">
                  <TestTube className="h-6 w-6 text-green-400" />
                  <h3 className="text-xl font-semibold text-white">Backtesting Engine</h3>
                </div>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Completed Tests:</span>
                    <span className="text-white font-medium">1,247</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Historical Data:</span>
                    <span className="text-blue-400 font-medium">2+ Years</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Strategy Validation:</span>
                    <span className="text-green-400 font-medium">Real-time</span>
                  </div>
                  <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded mt-4 border border-gray-700">
                    backtest --momentum --period=1y
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Smart Money Flow Section */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-white mb-6">
                Smart Money Flow Intelligence
              </h2>
              <p className="text-lg text-gray-300 max-w-3xl mx-auto">
                Track whale movements, institutional activity, and large order detection. 
                Follow the smart money with real-time flow analysis and institutional-grade insights.
              </p>
            </div>
            
            <div className="grid lg:grid-cols-3 gap-8 mb-12">
              <div className="bg-black border border-gray-800 rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                  <h3 className="text-lg font-semibold text-white">Whale Tracking</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Large Orders (&gt;$1M):</span>
                    <span className="text-green-400 font-medium">47 today</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Whale Activity:</span>
                    <span className="text-blue-400 font-medium">High</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Flow Direction:</span>
                    <span className="text-green-400 font-medium">Inbound</span>
                  </div>
                </div>
                <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded mt-4 border border-gray-700">
                  whale.track --large-orders --threshold=1000000
                </div>
              </div>
              
              <div className="bg-black border border-gray-800 rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse"></div>
                  <h3 className="text-lg font-semibold text-white">Institutional Flow</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Block Trades:</span>
                    <span className="text-white font-medium">23</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Dark Pool Volume:</span>
                    <span className="text-blue-400 font-medium">$45.2M</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Institution Ratio:</span>
                    <span className="text-purple-400 font-medium">67%</span>
                  </div>
                </div>
                <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded mt-4 border border-gray-700">
                  flow.analyze --institutional --dark-pools
                </div>
              </div>
              
              <div className="bg-black border border-gray-800 rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-3 h-3 bg-orange-400 rounded-full animate-pulse"></div>
                  <h3 className="text-lg font-semibold text-white">Market Sentiment</h3>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Fear & Greed:</span>
                    <span className="text-orange-400 font-medium">Neutral</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Volume Spike:</span>
                    <span className="text-green-400 font-medium">+234%</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400 text-sm">Smart Money:</span>
                    <span className="text-blue-400 font-medium">Accumulating</span>
                  </div>
                </div>
                <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded mt-4 border border-gray-700">
                  sentiment.analyze --smart-money --volume
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Customizable Dashboard Section */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-white mb-6">
                Fully Customizable Trading Dashboard
              </h2>
              <p className="text-lg text-gray-300 max-w-3xl mx-auto">
                No more scrolling through 15 tabs. Everything you need in one unified interface. 
                Customize your workspace with drag-and-drop widgets, real-time data streams, and personalized layouts.
              </p>
            </div>
            
            <div className="grid lg:grid-cols-2 gap-12 mb-12">
              <div className="space-y-6">
                <div className="bg-black border border-gray-800 rounded-xl p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <Settings className="h-5 w-5 text-blue-400" />
                    <h3 className="text-lg font-semibold text-white">Widget Customization</h3>
                  </div>
                  <ul className="space-y-2 text-gray-300">
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span>Drag-and-drop interface</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span>Real-time data streams</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span>Multiple screen layouts</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span>Personalized alerts</span>
                    </li>
                  </ul>
                </div>
                
                <div className="bg-black border border-gray-800 rounded-xl p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <Monitor className="h-5 w-5 text-purple-400" />
                    <h3 className="text-lg font-semibold text-white">Unified Interface</h3>
                  </div>
                  <ul className="space-y-2 text-gray-300">
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span>All markets in one view</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span>Portfolio overview</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span>Order management</span>
                    </li>
                    <li className="flex items-center space-x-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                      <span>Risk monitoring</span>
                    </li>
                  </ul>
                </div>
              </div>
              
              <div className="bg-black border border-gray-800 rounded-xl p-8">
                <div className="text-center mb-6">
                  <h3 className="text-xl font-semibold text-white mb-2">Dashboard Preview</h3>
                  <p className="text-gray-400 text-sm">Everything in one unified interface</p>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-black border border-gray-700 rounded-lg p-4 text-center">
                    <BarChart3 className="h-8 w-8 text-blue-400 mx-auto mb-2" />
                    <p className="text-white text-sm">Live Charts</p>
                  </div>
                  <div className="bg-black border border-gray-700 rounded-lg p-4 text-center">
                    <Activity className="h-8 w-8 text-green-400 mx-auto mb-2" />
                    <p className="text-white text-sm">Order Book</p>
                  </div>
                  <div className="bg-black border border-gray-700 rounded-lg p-4 text-center">
                    <Brain className="h-8 w-8 text-purple-400 mx-auto mb-2" />
                    <p className="text-white text-sm">ML Signals</p>
                  </div>
                  <div className="bg-black border border-gray-700 rounded-lg p-4 text-center">
                    <Shield className="h-8 w-8 text-orange-400 mx-auto mb-2" />
                    <p className="text-white text-sm">Risk Monitor</p>
                  </div>
                </div>
                <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded mt-4 text-center border border-gray-700">
                  dashboard.customize --layout=professional --widgets=all
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Advanced Features Section */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-6">
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold text-white mb-6">
                Advanced Trading Infrastructure
              </h2>
              <p className="text-lg text-gray-300 max-w-3xl mx-auto">
                Institutional-grade features including cross-collateralization, JIT liquidity, 
                and advanced risk management protocols built for professional traders.
              </p>
            </div>
            
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
              <div className="bg-black border border-gray-800 rounded-xl p-6 relative">
                <div className="absolute -top-2 -right-2 bg-blue-500 text-white text-xs px-2 py-1 rounded-full flex items-center space-x-1">
                  <Wrench className="h-3 w-3" />
                  <span>BETA</span>
                </div>
                <div className="flex items-center space-x-3 mb-4">
                  <Layers className="h-6 w-6 text-blue-400" />
                  <h3 className="text-lg font-semibold text-white">Cross-Collateralization</h3>
                </div>
                <p className="text-gray-400 text-sm mb-4">
                  Use multiple assets as collateral for leveraged positions. 
                  Optimize capital efficiency across your entire portfolio.
                </p>
                <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded border border-gray-700">
                  collateral.cross --assets=BTC,ETH,SOL --leverage=5x
                </div>
              </div>
              
              <div className="bg-black border border-gray-800 rounded-xl p-6 relative">
                <div className="absolute -top-2 -right-2 bg-yellow-500 text-white text-xs px-2 py-1 rounded-full flex items-center space-x-1">
                  <Wrench className="h-3 w-3" />
                  <span>BETA</span>
                </div>
                <div className="flex items-center space-x-3 mb-4">
                  <Zap className="h-6 w-6 text-yellow-400" />
                  <h3 className="text-lg font-semibold text-white">JIT Liquidity</h3>
                </div>
                <p className="text-gray-400 text-sm mb-4">
                  Just-in-time liquidity provision for optimal execution. 
                  Minimize slippage with intelligent order routing.
                </p>
                <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded border border-gray-700">
                  liquidity.jit --route=optimal --slippage=min
                </div>
              </div>
              
              <div className="bg-black border border-gray-800 rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <Shield className="h-6 w-6 text-red-400" />
                  <h3 className="text-lg font-semibold text-white">Risk Management</h3>
                </div>
                <p className="text-gray-400 text-sm mb-4">
                  Advanced risk protocols with real-time monitoring, 
                  automatic position sizing, and dynamic stop-losses.
                </p>
                <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded border border-gray-700">
                  risk.manage --auto-stop --position-sizing
                </div>
              </div>
              
              <div className="bg-black border border-gray-800 rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <Command className="h-6 w-6 text-green-400" />
                  <h3 className="text-lg font-semibold text-white">Advanced Orders</h3>
                </div>
                <p className="text-gray-400 text-sm mb-4">
                  Professional order types including iceberg, TWAP, 
                  and algorithmic execution strategies.
                </p>
                <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded border border-gray-700">
                  order.advanced --type=iceberg --strategy=twap
                </div>
              </div>
              
              <div className="bg-black border border-gray-800 rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <Database className="h-6 w-6 text-purple-400" />
                  <h3 className="text-lg font-semibold text-white">Portfolio Analytics</h3>
                </div>
                <p className="text-gray-400 text-sm mb-4">
                  Comprehensive portfolio analysis with P&L tracking, 
                  performance metrics, and risk attribution.
                </p>
                <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded border border-gray-700">
                  portfolio.analyze --pnl --risk --attribution
                </div>
              </div>
              
              <div className="bg-black border border-gray-800 rounded-xl p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <Lock className="h-6 w-6 text-cyan-400" />
                  <h3 className="text-lg font-semibold text-white">Security Protocols</h3>
                </div>
                <p className="text-gray-400 text-sm mb-4">
                  Secure wallet connections, hardware security modules, 
                  and audited smart contracts for maximum security.
                </p>
                <div className="text-xs text-orange-400 font-mono bg-black px-3 py-2 rounded border border-gray-700">
                  security.protocol --secure-wallet --hsm --audited
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Trading Interface Preview */}
        <section className="py-20">
          <div className="max-w-7xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-6">
              Professional Trading Interface
            </h2>
              <p className="text-lg text-gray-300 max-w-3xl mx-auto">
              Advanced charting, real-time order book, and institutional-grade execution.
                Experience the power of terminal-based trading with AI insights.
            </p>
          </div>
          
          <div className="relative">
              <div className="bg-black border border-gray-800 rounded-2xl overflow-hidden shadow-2xl">
                <div className="bg-black px-6 py-4 border-b border-gray-700 flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    <div className="flex space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                    <span className="text-sm font-medium">QuantDesk Trading Terminal</span>
                  </div>
                  <div className="flex items-center space-x-4">
                    <div className="flex items-center space-x-2 text-sm text-gray-400">
                      <Database className="h-4 w-4" />
                      <span>Live Data</span>
                    </div>
                    <div className="flex items-center space-x-2 text-sm text-purple-400">
                      <Brain className="h-4 w-4" />
                      <span>ML Active</span>
                    </div>
                  </div>
                </div>
                
                <div className="p-8">
              <div className="grid lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2">
                      <div className="h-80 bg-black border border-gray-700 rounded-xl flex items-center justify-center">
                    <div className="text-center">
                      <TrendingUp className="h-16 w-16 text-blue-400 mx-auto mb-4" />
                          <p className="text-gray-400 mb-2">Advanced Charting Interface</p>
                          <p className="text-sm text-gray-500">Real-time price action with AI signals</p>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                      <div className="h-36 bg-black border border-gray-700 rounded-xl flex items-center justify-center">
                        <div className="text-center">
                          <BarChart3 className="h-8 w-8 text-green-400 mx-auto mb-2" />
                    <p className="text-gray-400 text-sm">Order Book</p>
                  </div>
                      </div>
                      <div className="h-36 bg-black border border-gray-700 rounded-xl flex items-center justify-center">
                        <div className="text-center">
                          <Brain className="h-8 w-8 text-purple-400 mx-auto mb-2" />
                          <p className="text-gray-400 text-sm">ML Signals</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Social Links Section */}
        <section className="py-16">
          <div className="max-w-4xl mx-auto px-6 text-center">
            <h2 className="text-3xl font-bold text-white mb-6">
              Stay Updated
            </h2>
            <p className="text-lg text-gray-300 mb-8">
              Follow us on social media for the latest updates, testing announcements, and trading insights.
            </p>
            <div className="flex items-center justify-center space-x-6">
              <a 
                href="https://twitter.com/quantdeskapp" 
                target="_blank" 
                rel="noopener noreferrer"
                className="group flex items-center space-x-3 bg-black border border-gray-700 hover:border-blue-400 text-white px-6 py-4 rounded-lg font-semibold transition-all duration-300 hover:shadow-lg hover:shadow-blue-500/25"
              >
                <Twitter className="h-6 w-6 text-blue-400 group-hover:scale-110 transition-transform" />
                <span>Follow @QuantDeskApp</span>
                <ExternalLink className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
              </a>
            </div>
            <div className="mt-8 text-sm text-gray-400">
              <p>Turn on notifications to get alerts when testing begins!</p>
          </div>
        </div>
      </section>

      {/* CTA Section */}
        <section className="py-20">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold text-white mb-6">
            Ready to Start Trading?
          </h2>
            <p className="text-lg text-gray-300 mb-12">
              Join thousands of professional traders already using QuantDesk for their perpetual futures trading. 
              Experience the power of terminal-based trading infrastructure with AI-powered insights.
          </p>
            <div className="flex items-center justify-center gap-6 mb-12">
            <Link 
              to="/lite#trading" 
                className="group flex items-center space-x-2 bg-black hover:bg-blue-400 border border-blue-400 hover:border-blue-400 text-blue-400 hover:text-black px-8 py-4 rounded-lg font-mono font-semibold transition-all duration-300 hover:shadow-lg hover:shadow-blue-400/50"
            >
                <TrendingUp className="h-5 w-5" />
                <span>Launch Trading Interface</span>
                <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </Link>
            <Link 
                to="/pro" 
                className="group flex items-center space-x-2 bg-black hover:bg-orange-400 border border-orange-400 hover:border-orange-400 text-orange-400 hover:text-white px-8 py-4 rounded-lg font-mono font-semibold transition-all duration-300 hover:shadow-lg hover:shadow-orange-400/50"
            >
                <Terminal className="h-5 w-5" />
                <span>Open Pro Terminal</span>
                <ArrowRight className="h-4 w-4 group-hover:translate-x-1 transition-transform" />
            </Link>
          </div>
          
            {/* Footer */}
          <div className="pt-8 border-t border-gray-800">
            <div className="mb-6">
              <img 
                src="/quantdesk-banner.png" 
                alt="QuantDesk" 
                className="h-8 mx-auto opacity-60 hover:opacity-80 transition-opacity mb-6"
              />
              
              {/* Documentation Links */}
              <div className="flex flex-wrap items-center justify-center gap-6 text-sm">
                <a 
                  href={DOCS_URLS.baseUrl} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="group flex items-center space-x-2 text-gray-400 hover:text-blue-400 transition-colors"
                >
                  <Terminal className="h-4 w-4" />
                  <span>Documentation</span>
                  <ExternalLink className="h-3 w-3 group-hover:translate-x-1 transition-transform" />
                </a>
                <a 
                  href={DOCS_URLS.technicalPortfolio} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="group flex items-center space-x-2 text-gray-400 hover:text-blue-400 transition-colors"
                >
                  <BarChart3 className="h-4 w-4" />
                  <span>Technical Portfolio</span>
                  <ExternalLink className="h-3 w-3 group-hover:translate-x-1 transition-transform" />
                </a>
                <a 
                  href={DOCS_URLS.performanceMetrics} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="group flex items-center space-x-2 text-gray-400 hover:text-blue-400 transition-colors"
                >
                  <TrendingUp className="h-4 w-4" />
                  <span>Performance Metrics</span>
                  <ExternalLink className="h-3 w-3 group-hover:translate-x-1 transition-transform" />
                </a>
                <a 
                  href={GITHUB_URL} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="group flex items-center space-x-2 text-gray-400 hover:text-blue-400 transition-colors"
                >
                  <Database className="h-4 w-4" />
                  <span>GitHub Repository</span>
                  <ExternalLink className="h-3 w-3 group-hover:translate-x-1 transition-transform" />
                </a>
              </div>
            </div>
            
            {/* Copyright */}
            <div className="text-center text-gray-500 text-xs">
              <p>&copy; 2024 QuantDesk Protocol. Built for professional traders.</p>
              <p className="mt-1">The Bloomberg Terminal for Crypto - Decentralized Perpetual Trading Infrastructure</p>
            </div>
          </div>
        </div>
      </section>
      </div>
    </div>
  )
}

export default LandingPage