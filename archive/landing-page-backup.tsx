import React from 'react'
import { Link } from 'react-router-dom'
import { Shield, Zap, BarChart3, Wallet, TrendingUp } from 'lucide-react'

const stats = [
  { label: 'Total Volume', value: '$2.1B', change: '+12.5%' },
  { label: 'Open Interest', value: '$89.2M', change: '+8.2%' },
  { label: 'Active Users', value: '12.4K', change: '+15.3%' },
  { label: 'Markets', value: '24', change: '+2' },
]

const features = [
  {
    icon: <Zap className="h-6 w-6" />,
    title: 'Ultra-Low Latency',
    desc: 'Sub-millisecond execution on Solana',
  },
  {
    icon: <BarChart3 className="h-6 w-6" />,
    title: 'Advanced Trading',
    desc: 'Professional tools and deep liquidity',
  },
  {
    icon: <Shield className="h-6 w-6" />,
    title: 'Battle-Tested',
    desc: 'Audited smart contracts and security',
  },
  {
    icon: <Wallet className="h-6 w-6" />,
    title: 'Non-Custodial',
    desc: 'Your keys, your assets, always',
  },
]

const LandingPage: React.FC = () => {
  return (
    <div className="min-h-screen bg-black text-white">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-b from-black via-gray-900 to-black" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(59,130,246,0.03)_0%,transparent_70%)]" />
        
        <div className="relative max-w-6xl mx-auto px-6 pt-24 pb-32">
          <div className="text-center">
            {/* Logo */}
            <div className="mb-8">
              <img 
                src="/quantdesk-logo.png" 
                alt="QuantDesk" 
                className="h-16 mx-auto mb-6"
              />
            </div>
            
            {/* Badge */}
            <div className="inline-flex items-center px-3 py-1.5 rounded-full bg-gray-900/50 border border-gray-800 mb-8">
              <div className="w-1.5 h-1.5 bg-green-400 rounded-full mr-2" />
              <span className="text-xs text-gray-300 font-medium">LIVE ON SOLANA</span>
            </div>
            
            {/* Main Headline */}
            <h1 className="text-6xl md:text-8xl font-bold tracking-tight mb-8">
              The Future of{' '}
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Trading
              </span>
            </h1>
            
            {/* Subtitle */}
            <p className="text-xl text-gray-400 max-w-2xl mx-auto mb-12 leading-relaxed">
              Trade perpetual futures with institutional-grade infrastructure on Solana.
            </p>
            
            {/* CTA Buttons */}
            <div className="flex items-center justify-center gap-4 mb-20">
              <Link 
                to="/lite#trading" 
                className="px-8 py-4 bg-white text-black rounded-lg font-semibold hover:bg-gray-100 transition-colors"
              >
                Start Trading
              </Link>
              <Link 
                to="/lite" 
                className="px-8 py-4 text-gray-400 hover:text-white transition-colors"
              >
                Learn More →
              </Link>
            </div>
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
            {stats.map((stat) => (
              <div key={stat.label} className="text-center">
                <div className="text-3xl font-bold text-white mb-1">
                  {stat.value}
                </div>
                <div className="text-sm text-gray-400 mb-1">{stat.label}</div>
                <div className="text-xs text-green-400 font-medium">
                  {stat.change}
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 bg-gray-900/30">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-20">
            <h2 className="text-4xl font-bold text-white mb-6">
              Built for Professional Traders
            </h2>
            <p className="text-lg text-gray-400 max-w-2xl mx-auto">
              Every feature designed with institutional-grade performance and security in mind.
            </p>
          </div>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature) => (
              <div 
                key={feature.title}
                className="group"
              >
                <div className="p-6 rounded-xl bg-gray-900/50 border border-gray-800 hover:border-gray-700 transition-all duration-300">
                  <div className="text-blue-400 mb-4">
                    {feature.icon}
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-sm text-gray-400">
                    {feature.desc}
                  </p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Trading Preview */}
      <section className="py-24">
        <div className="max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-white mb-6">
              Professional Trading Interface
            </h2>
            <p className="text-lg text-gray-400 max-w-2xl mx-auto">
              Advanced charting, real-time order book, and institutional-grade execution.
            </p>
          </div>
          
          <div className="relative">
            <div className="rounded-2xl bg-gray-900/50 border border-gray-800 p-8">
              <div className="grid lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2">
                  <div className="h-80 bg-gray-800/50 rounded-xl flex items-center justify-center border border-gray-700">
                    <div className="text-center">
                      <TrendingUp className="h-16 w-16 text-blue-400 mx-auto mb-4" />
                      <p className="text-gray-400">Advanced Charting Interface</p>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  <div className="h-36 bg-gray-800/50 rounded-xl flex items-center justify-center border border-gray-700">
                    <p className="text-gray-400 text-sm">Order Book</p>
                  </div>
                  <div className="h-36 bg-gray-800/50 rounded-xl flex items-center justify-center border border-gray-700">
                    <p className="text-gray-400 text-sm">Trading Panel</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold text-white mb-6">
            Ready to Start Trading?
          </h2>
          <p className="text-lg text-gray-400 mb-12">
            Join thousands of traders already using QuantDesk for their perpetual futures trading.
          </p>
          <div className="flex items-center justify-center gap-4 mb-12">
            <Link 
              to="/lite#trading" 
              className="px-8 py-4 bg-white text-black rounded-lg font-semibold hover:bg-gray-100 transition-colors"
            >
              Launch Trading Interface
            </Link>
            <Link 
              to="/lite#markets" 
              className="px-8 py-4 text-gray-400 hover:text-white transition-colors"
            >
              Browse Markets
            </Link>
          </div>
          
          {/* Footer Logo */}
          <div className="pt-8 border-t border-gray-800">
            <img 
              src="/quantdesk-banner.png" 
              alt="QuantDesk" 
              className="h-8 mx-auto opacity-60 hover:opacity-80 transition-opacity"
            />
          </div>
        </div>
      </section>
    </div>
  )
}

export default LandingPage
