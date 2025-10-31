// QuantDesk Developer Playground - SDK Integration Examples Component
// Phase 2: Enhanced Features (Exceed Drift's Capabilities)
// Strategy: "More Open Than Drift" - Comprehensive SDK integration

import React, { useState } from 'react';

interface SdkExample {
  id: string;
  title: string;
  description: string;
  category: 'trading' | 'ai' | 'community' | 'analytics' | 'portfolio';
  difficulty: 'beginner' | 'intermediate' | 'advanced';
  languages: string[];
  code: string;
  explanation: string;
  useCase: string;
}

export const SdkIntegrationExamples: React.FC = () => {
  const [selectedExample, setSelectedExample] = useState<string | null>(null);
  const [selectedLanguage, setSelectedLanguage] = useState<string>('typescript');
  const [copied, setCopied] = useState<boolean>(false);

  const sdkExamples: SdkExample[] = [
    {
      id: 'basic-trading',
      title: 'Basic Trading Operations',
      description: 'Open and close positions with the QuantDesk SDK',
      category: 'trading',
      difficulty: 'beginner',
      languages: ['typescript', 'javascript', 'python'],
      code: `// Basic Trading Operations with QuantDesk SDK
import { QuantDeskClient, PositionSide } from '@quantdesk/sdk';

async function basicTradingExample() {
  // Initialize the client
  const client = new QuantDeskClient({
    apiKey: process.env.QUANTDESK_API_KEY,
    baseUrl: 'https://api.quantdesk.com'
  });

  try {
    // Get available markets
    const markets = await client.getMarkets();
    console.log('Available markets:', markets);

    // Get market data for SOL-PERP
    const marketData = await client.getMarketData('SOL-PERP');
    console.log('SOL-PERP price:', marketData.price);

    // Open a long position
    const position = await client.openPosition({
      market: 'SOL-PERP',
      side: PositionSide.LONG,
      size: 1.0,
      leverage: 10,
      entryPrice: marketData.price
    });
    console.log('Position opened:', position);

    // Get portfolio
    const portfolio = await client.getPortfolio();
    console.log('Portfolio:', portfolio);

    // Close the position
    await client.closePosition(position.id);
    console.log('Position closed');

  } catch (error) {
    console.error('Trading error:', error);
  }
}

basicTradingExample();`,
      explanation: 'This example demonstrates basic trading operations including opening and closing positions. It shows how to get market data, open a position with leverage, and manage your portfolio.',
      useCase: 'Perfect for beginners learning to trade with QuantDesk SDK. Shows the complete trading workflow from market data to position management.'
    },
    {
      id: 'ai-integration',
      title: 'AI-Powered Trading with MIKEY',
      description: 'Integrate MIKEY AI for intelligent trading decisions',
      category: 'ai',
      difficulty: 'intermediate',
      languages: ['typescript', 'python'],
      code: `// AI-Powered Trading with MIKEY AI
import { QuantDeskClient, MikeyAI } from '@quantdesk/sdk';

async function aiTradingExample() {
  const client = new QuantDeskClient({
    apiKey: process.env.QUANTDESK_API_KEY
  });

  const mikey = new MikeyAI(client);

  try {
    // Get AI analysis for SOL-PERP
    const analysis = await mikey.getMarketAnalysis('SOL-PERP');
    console.log('AI Analysis:', analysis);

    // Get trading signals
    const signals = await mikey.getTradingSignals();
    console.log('Trading Signals:', signals);

    // Chat with MIKEY AI
    const response = await mikey.chat({
      message: 'What do you think about SOL-PERP?',
      context: {
        market: 'SOL-PERP',
        userPositions: []
      }
    });
    console.log('MIKEY Response:', response);

    // Execute AI-recommended trade
    if (analysis.recommendation === 'buy' && analysis.confidence > 0.8) {
      const position = await client.openPosition({
        market: 'SOL-PERP',
        side: 'long',
        size: 0.5,
        leverage: 5,
        entryPrice: analysis.currentPrice
      });
      console.log('AI-recommended position opened:', position);
    }

  } catch (error) {
    console.error('AI trading error:', error);
  }
}

aiTradingExample();`,
      explanation: 'This example shows how to integrate MIKEY AI for intelligent trading decisions. It demonstrates getting AI analysis, trading signals, and executing AI-recommended trades.',
      useCase: 'Ideal for traders who want to leverage AI for better trading decisions. Shows the complete AI integration workflow.'
    },
    {
      id: 'portfolio-management',
      title: 'Advanced Portfolio Management',
      description: 'Comprehensive portfolio tracking and risk management',
      category: 'portfolio',
      difficulty: 'intermediate',
      languages: ['typescript', 'python', 'javascript'],
      code: `// Advanced Portfolio Management
import { QuantDeskClient, PortfolioManager } from '@quantdesk/sdk';

async function portfolioManagementExample() {
  const client = new QuantDeskClient({
    apiKey: process.env.QUANTDESK_API_KEY
  });

  const portfolioManager = new PortfolioManager(client);

  try {
    // Get comprehensive portfolio data
    const portfolio = await portfolioManager.getPortfolio();
    console.log('Portfolio Overview:', portfolio);

    // Get performance metrics
    const performance = await portfolioManager.getPerformanceMetrics();
    console.log('Performance Metrics:', performance);

    // Get risk analysis
    const riskAnalysis = await portfolioManager.getRiskAnalysis();
    console.log('Risk Analysis:', riskAnalysis);

    // Get trading history
    const tradingHistory = await portfolioManager.getTradingHistory({
      startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
      endDate: new Date()
    });
    console.log('Trading History:', tradingHistory);

    // Calculate portfolio metrics
    const metrics = await portfolioManager.calculateMetrics({
      totalValue: portfolio.totalValue,
      positions: portfolio.positions,
      trades: tradingHistory.trades
    });
    console.log('Calculated Metrics:', metrics);

    // Risk management
    if (riskAnalysis.overallRisk === 'high') {
      console.log('High risk detected, consider reducing position sizes');
      // Implement risk management logic
    }

  } catch (error) {
    console.error('Portfolio management error:', error);
  }
}

portfolioManagementExample();`,
      explanation: 'This example demonstrates advanced portfolio management including performance tracking, risk analysis, and comprehensive metrics calculation.',
      useCase: 'Perfect for portfolio managers and advanced traders who need comprehensive portfolio tracking and risk management capabilities.'
    },
    {
      id: 'community-features',
      title: 'Community Points & Referrals',
      description: 'Integrate community features for user engagement',
      category: 'community',
      difficulty: 'beginner',
      languages: ['typescript', 'javascript', 'python'],
      code: `// Community Points & Referrals Integration
import { QuantDeskClient, CommunityManager } from '@quantdesk/sdk';

async function communityFeaturesExample() {
  const client = new QuantDeskClient({
    apiKey: process.env.QUANTDESK_API_KEY
  });

  const community = new CommunityManager(client);

  try {
    // Get user points
    const points = await community.getUserPoints();
    console.log('User Points:', points);

    // Get user badges
    const badges = await community.getUserBadges();
    console.log('User Badges:', badges);

    // Generate referral code
    const referralCode = await community.generateReferralCode();
    console.log('Referral Code:', referralCode);

    // Get referral stats
    const referralStats = await community.getReferralStats();
    console.log('Referral Stats:', referralStats);

    // Get community leaderboard
    const leaderboard = await community.getLeaderboard();
    console.log('Community Leaderboard:', leaderboard);

    // Award points for trading activity
    const pointsAwarded = await community.awardPoints({
      activity: 'trade',
      amount: 1000,
      market: 'SOL-PERP'
    });
    console.log('Points Awarded:', pointsAwarded);

    // Get redemption options
    const redemptions = await community.getRedemptionOptions();
    console.log('Redemption Options:', redemptions);

  } catch (error) {
    console.error('Community features error:', error);
  }
}

communityFeaturesExample();`,
      explanation: 'This example shows how to integrate community features including points system, referrals, badges, and leaderboards for enhanced user engagement.',
      useCase: 'Great for applications that want to gamify trading and increase user engagement through community features.'
    },
    {
      id: 'advanced-analytics',
      title: 'Advanced Analytics & Insights',
      description: 'Comprehensive analytics and market insights',
      category: 'analytics',
      difficulty: 'advanced',
      languages: ['typescript', 'python'],
      code: `// Advanced Analytics & Insights
import { QuantDeskClient, AnalyticsEngine } from '@quantdesk/sdk';

async function advancedAnalyticsExample() {
  const client = new QuantDeskClient({
    apiKey: process.env.QUANTDESK_API_KEY
  });

  const analytics = new AnalyticsEngine(client);

  try {
    // Get market analytics
    const marketAnalytics = await analytics.getMarketAnalytics('SOL-PERP');
    console.log('Market Analytics:', marketAnalytics);

    // Get trading performance analytics
    const performanceAnalytics = await analytics.getPerformanceAnalytics({
      startDate: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000), // 90 days ago
      endDate: new Date()
    });
    console.log('Performance Analytics:', performanceAnalytics);

    // Get risk metrics
    const riskMetrics = await analytics.getRiskMetrics();
    console.log('Risk Metrics:', riskMetrics);

    // Get correlation analysis
    const correlation = await analytics.getCorrelationAnalysis(['SOL-PERP', 'ETH-PERP', 'BTC-PERP']);
    console.log('Correlation Analysis:', correlation);

    // Get volatility analysis
    const volatility = await analytics.getVolatilityAnalysis('SOL-PERP');
    console.log('Volatility Analysis:', volatility);

    // Generate insights report
    const insightsReport = await analytics.generateInsightsReport({
      markets: ['SOL-PERP', 'ETH-PERP'],
      timeRange: '30d',
      includeRisk: true,
      includePerformance: true
    });
    console.log('Insights Report:', insightsReport);

  } catch (error) {
    console.error('Analytics error:', error);
  }
}

advancedAnalyticsExample();`,
      explanation: 'This example demonstrates advanced analytics capabilities including market analysis, performance metrics, risk analysis, and comprehensive insights generation.',
      useCase: 'Perfect for quantitative traders, analysts, and advanced users who need comprehensive market and performance analytics.'
    },
    {
      id: 'real-time-monitoring',
      title: 'Real-time Market Monitoring',
      description: 'Real-time market data and position monitoring',
      category: 'trading',
      difficulty: 'intermediate',
      languages: ['typescript', 'javascript'],
      code: `// Real-time Market Monitoring
import { QuantDeskClient, RealTimeMonitor } from '@quantdesk/sdk';

async function realTimeMonitoringExample() {
  const client = new QuantDeskClient({
    apiKey: process.env.QUANTDESK_API_KEY
  });

  const monitor = new RealTimeMonitor(client);

  try {
    // Subscribe to market data
    await monitor.subscribeToMarketData('SOL-PERP', (data) => {
      console.log('SOL-PERP Update:', data);
      
      // Check if price moved significantly
      if (data.priceChange > 0.05) { // 5% change
        console.log('Significant price movement detected!');
      }
    });

    // Subscribe to portfolio updates
    await monitor.subscribeToPortfolio((portfolio) => {
      console.log('Portfolio Update:', portfolio);
      
      // Check for new positions
      if (portfolio.positions.length > 0) {
        console.log('New positions detected');
      }
    });

    // Subscribe to order updates
    await monitor.subscribeToOrders((orders) => {
      console.log('Order Updates:', orders);
      
      // Check for filled orders
      const filledOrders = orders.filter(order => order.status === 'filled');
      if (filledOrders.length > 0) {
        console.log('Orders filled:', filledOrders);
      }
    });

    // Start monitoring
    await monitor.start();
    console.log('Real-time monitoring started');

    // Keep monitoring for 60 seconds
    setTimeout(async () => {
      await monitor.stop();
      console.log('Monitoring stopped');
    }, 60000);

  } catch (error) {
    console.error('Real-time monitoring error:', error);
  }
}

realTimeMonitoringExample();`,
      explanation: 'This example shows how to implement real-time monitoring for market data, portfolio updates, and order status changes using WebSocket connections.',
      useCase: 'Ideal for active traders who need real-time updates and automated responses to market changes.'
    }
  ];

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'trading': return '💼';
      case 'ai': return '🤖';
      case 'community': return '👥';
      case 'analytics': return '📊';
      case 'portfolio': return '📈';
      default: return '🔗';
    }
  };

  const getDifficultyColor = (difficulty: string) => {
    switch (difficulty) {
      case 'beginner': return 'bg-green-100 text-green-800';
      case 'intermediate': return 'bg-yellow-100 text-yellow-800';
      case 'advanced': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const copyToClipboard = async (code: string) => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy code:', err);
    }
  };

  const downloadCode = (code: string, title: string) => {
    const extension = selectedLanguage === 'typescript' ? 'ts' : 
                     selectedLanguage === 'javascript' ? 'js' : 
                     selectedLanguage === 'python' ? 'py' : 'txt';
    const filename = `${title.toLowerCase().replace(/\s+/g, '-')}.${extension}`;
    
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const selectedExampleData = sdkExamples.find(ex => ex.id === selectedExample);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          ⚡ SDK Integration Examples
        </h2>
        <p className="text-gray-600">
          Comprehensive SDK examples - Exceeding Drift's basic documentation
        </p>
      </div>

      {/* Examples Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {sdkExamples.map((example) => (
          <div
            key={example.id}
            className="bg-white p-6 rounded-lg shadow-sm border hover:shadow-md transition-shadow cursor-pointer"
            onClick={() => setSelectedExample(example.id)}
          >
            <div className="flex items-center mb-4">
              <span className="text-2xl mr-3">
                {getCategoryIcon(example.category)}
              </span>
              <div className="flex-1">
                <h3 className="font-semibold text-gray-900">
                  {example.title}
                </h3>
                <span className={`px-2 py-1 text-xs font-medium rounded ${getDifficultyColor(example.difficulty)}`}>
                  {example.difficulty}
                </span>
              </div>
            </div>
            
            <p className="text-sm text-gray-600 mb-4">
              {example.description}
            </p>
            
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-1">
                {example.languages.map((lang) => (
                  <span key={lang} className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                    {lang}
                  </span>
                ))}
              </div>
              <span className="text-sm text-blue-600">
                View Example →
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Selected Example Details */}
      {selectedExampleData && (
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center">
              <span className="text-3xl mr-4">
                {getCategoryIcon(selectedExampleData.category)}
              </span>
              <div>
                <h3 className="text-xl font-semibold text-gray-900">
                  {selectedExampleData.title}
                </h3>
                <div className="flex items-center space-x-2 mt-1">
                  <span className={`px-2 py-1 text-xs font-medium rounded ${getDifficultyColor(selectedExampleData.difficulty)}`}>
                    {selectedExampleData.difficulty}
                  </span>
                  <span className="text-sm text-gray-500">
                    {selectedExampleData.category}
                  </span>
                </div>
              </div>
            </div>
            <button
              onClick={() => setSelectedExample(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              <span className="text-2xl">×</span>
            </button>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Code Section */}
            <div>
              <div className="flex items-center justify-between mb-4">
                <h4 className="font-semibold text-gray-900">Code Example</h4>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => copyToClipboard(selectedExampleData.code)}
                    className={`px-3 py-1 text-sm rounded ${
                      copied 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {copied ? '✅ Copied!' : '📋 Copy'}
                  </button>
                  <button
                    onClick={() => downloadCode(selectedExampleData.code, selectedExampleData.title)}
                    className="px-3 py-1 bg-blue-100 text-blue-800 text-sm rounded hover:bg-blue-200"
                  >
                    💾 Download
                  </button>
                </div>
              </div>
              
              <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto">
                <pre className="text-green-400 text-sm">
                  <code>{selectedExampleData.code}</code>
                </pre>
              </div>
            </div>

            {/* Details Section */}
            <div className="space-y-4">
              <div>
                <h4 className="font-semibold text-gray-900 mb-2">Explanation</h4>
                <p className="text-sm text-gray-700">
                  {selectedExampleData.explanation}
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold text-gray-900 mb-2">Use Case</h4>
                <p className="text-sm text-gray-700">
                  {selectedExampleData.useCase}
                </p>
              </div>
              
              <div>
                <h4 className="font-semibold text-gray-900 mb-2">Supported Languages</h4>
                <div className="flex items-center space-x-2">
                  {selectedExampleData.languages.map((lang) => (
                    <span key={lang} className="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded">
                      {lang}
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* SDK Advantages */}
      <div className="bg-green-50 rounded-lg p-6 border border-green-200">
        <h3 className="text-lg font-semibold text-green-900 mb-4">
          🎯 SDK Advantages Over Drift
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-green-900 mb-2">Drift Protocol</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Basic SDK examples</li>
              <li>• Limited integration patterns</li>
              <li>• Simple trading operations</li>
              <li>• Basic documentation</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-green-900 mb-2">QuantDesk</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Comprehensive SDK examples</li>
              <li>• Advanced integration patterns</li>
              <li>• AI-powered trading</li>
              <li>• Community features</li>
              <li>• Real-time monitoring</li>
              <li>• Advanced analytics</li>
              <li>• Multi-language support</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Getting Started */}
      <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-4">
          🚀 Getting Started with SDK
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex items-start">
            <span className="text-2xl mr-3">1️⃣</span>
            <div>
              <h4 className="font-medium text-blue-900">Install SDK</h4>
              <p className="text-sm text-blue-700">
                npm install @quantdesk/sdk
              </p>
            </div>
          </div>
          
          <div className="flex items-start">
            <span className="text-2xl mr-3">2️⃣</span>
            <div>
              <h4 className="font-medium text-blue-900">Get API Key</h4>
              <p className="text-sm text-blue-700">
                Generate your API key from the dashboard
              </p>
            </div>
          </div>
          
          <div className="flex items-start">
            <span className="text-2xl mr-3">3️⃣</span>
            <div>
              <h4 className="font-medium text-blue-900">Start Coding</h4>
              <p className="text-sm text-blue-700">
                Use the examples above to get started
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
