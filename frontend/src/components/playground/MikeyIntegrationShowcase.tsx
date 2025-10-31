// QuantDesk MIKEY AI Integration Showcase
// Task 7: MIKEY Integration Showcase
// Strategy: "More Open Than Drift" - Show AI capabilities and interaction patterns

import React, { useState, useEffect } from 'react';
import { mikeyAI, MikeyAIResponse, LLMStatus } from '../../services/mikeyAI';

interface AIInsight {
  id: string;
  type: 'portfolio' | 'sentiment' | 'position' | 'market' | 'risk';
  title: string;
  description: string;
  confidence: number;
  timestamp: Date;
  data: any;
  recommendations?: string[];
}

interface TradingScenario {
  id: string;
  title: string;
  description: string;
  query: string;
  expectedResponse: string;
  capabilities: string[];
  category: 'analysis' | 'management' | 'insights' | 'risk';
}

export const MikeyIntegrationShowcase: React.FC = () => {
  const [selectedScenario, setSelectedScenario] = useState<string | null>(null);
  const [aiResponse, setAiResponse] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [insights, setInsights] = useState<AIInsight[]>([]);
  const [serviceStatus, setServiceStatus] = useState<boolean>(false);
  const [llmStatus, setLlmStatus] = useState<LLMStatus[]>([]);
  const [error, setError] = useState<string>('');

  // Check service health on component mount
  useEffect(() => {
    const checkHealth = async () => {
      const isHealthy = await mikeyAI.healthCheck();
      setServiceStatus(isHealthy);
      
      if (isHealthy) {
        const status = await mikeyAI.getLLMStatus();
        setLlmStatus(status);
      }
    };
    
    checkHealth();
  }, []);

  const tradingScenarios: TradingScenario[] = [
    {
      id: 'portfolio-analysis',
      title: 'Portfolio Analysis & Optimization',
      description: 'AI-powered portfolio analysis with risk assessment and optimization recommendations',
      query: 'Analyze my current portfolio and provide optimization recommendations based on market conditions',
      expectedResponse: 'Portfolio analysis with risk metrics, diversification insights, and optimization suggestions',
      capabilities: ['Multi-LLM routing', 'Real-time data processing', 'Risk assessment', 'Portfolio optimization'],
      category: 'analysis'
    },
    {
      id: 'market-sentiment',
      title: 'Market Sentiment Analysis',
      description: 'Real-time market sentiment analysis using news, social media, and trading data',
      query: 'What is the current market sentiment for SOL-PERP and how might it affect my positions?',
      expectedResponse: 'Sentiment analysis with confidence scores, key factors, and trading implications',
      capabilities: ['Multi-source sentiment', 'Real-time processing', 'Confidence scoring', 'Trend analysis'],
      category: 'insights'
    },
    {
      id: 'position-management',
      title: 'Intelligent Position Management',
      description: 'AI-driven position sizing, stop-loss, and take-profit recommendations',
      query: 'Help me manage my SOL-PERP position with optimal stop-loss and take-profit levels',
      expectedResponse: 'Position management recommendations with risk-adjusted sizing and exit strategies',
      capabilities: ['Position sizing', 'Risk management', 'Exit strategies', 'Dynamic adjustments'],
      category: 'management'
    },
    {
      id: 'risk-assessment',
      title: 'Risk Assessment & Management',
      description: 'Comprehensive risk analysis with portfolio-wide risk metrics and alerts',
      query: 'Assess the overall risk of my trading portfolio and suggest risk management strategies',
      expectedResponse: 'Risk assessment with portfolio metrics, correlation analysis, and risk mitigation strategies',
      capabilities: ['Portfolio risk analysis', 'Correlation analysis', 'Risk metrics', 'Mitigation strategies'],
      category: 'risk'
    },
    {
      id: 'market-opportunities',
      title: 'Market Opportunity Detection',
      description: 'AI-powered identification of trading opportunities based on technical and fundamental analysis',
      query: 'Identify potential trading opportunities in the current market conditions',
      expectedResponse: 'Market opportunities with entry signals, risk levels, and expected returns',
      capabilities: ['Pattern recognition', 'Signal generation', 'Risk assessment', 'Return estimation'],
      category: 'insights'
    },
    {
      id: 'trade-execution',
      title: 'Intelligent Trade Execution',
      description: 'AI-assisted trade execution with optimal timing and execution strategies',
      query: 'Help me execute a trade for SOL-PERP with optimal timing and execution strategy',
      expectedResponse: 'Trade execution plan with timing recommendations, execution strategy, and risk management',
      capabilities: ['Execution optimization', 'Timing analysis', 'Strategy selection', 'Risk control'],
      category: 'management'
    }
  ];

  const mockAIInsights: AIInsight[] = [
    {
      id: 'insight-1',
      type: 'portfolio',
      title: 'Portfolio Diversification Alert',
      description: 'Your portfolio is 85% concentrated in SOL-related assets. Consider diversifying to reduce risk.',
      confidence: 0.92,
      timestamp: new Date(Date.now() - 300000),
      data: {
        concentration: 0.85,
        recommendedAllocation: 0.60,
        riskLevel: 'high'
      },
      recommendations: [
        'Reduce SOL exposure to 60%',
        'Add BTC and ETH positions',
        'Consider stablecoin allocation'
      ]
    },
    {
      id: 'insight-2',
      type: 'sentiment',
      title: 'SOL Market Sentiment: Bullish',
      description: 'SOL sentiment is trending bullish with strong social media activity and positive news flow.',
      confidence: 0.78,
      timestamp: new Date(Date.now() - 180000),
      data: {
        sentiment: 0.78,
        socialActivity: 'high',
        newsSentiment: 'positive',
        priceTarget: 220
      },
      recommendations: [
        'Consider increasing SOL position',
        'Monitor resistance at $200',
        'Set stop-loss at $180'
      ]
    },
    {
      id: 'insight-3',
      type: 'risk',
      title: 'Risk Level: Moderate',
      description: 'Portfolio risk is within acceptable limits but monitor correlation with market volatility.',
      confidence: 0.85,
      timestamp: new Date(Date.now() - 120000),
      data: {
        portfolioVar: 0.12,
        correlation: 0.65,
        volatility: 'moderate'
      },
      recommendations: [
        'Maintain current position sizes',
        'Monitor market volatility',
        'Consider hedging strategies'
      ]
    }
  ];

  useEffect(() => {
    setInsights(mockAIInsights);
  }, []);

  const handleScenarioClick = async (scenario: TradingScenario) => {
    if (!serviceStatus) {
      setError('MIKEY-AI service is not available. Please check if the service is running on port 3000.');
      return;
    }

    setIsLoading(true);
    setSelectedScenario(scenario.id);
    setError('');
    
    try {
      // Make real API call to MIKEY-AI
      const response: MikeyAIResponse = await mikeyAI.queryAI(scenario.query);
      
      if (response.success) {
        setAiResponse(response.data.response);
        
        // Generate insights based on the real response
        const newInsights: AIInsight[] = [
          {
            id: `insight-${Date.now()}`,
            type: scenario.category as any,
            title: `${scenario.title} Analysis`,
            description: `AI analysis completed with ${(response.data.confidence * 100).toFixed(1)}% confidence`,
            confidence: response.data.confidence,
            timestamp: new Date(response.data.timestamp),
            data: {
              sources: response.data.sources,
              provider: response.data.provider,
              response: response.data.response
            },
            recommendations: [
              'Review the analysis results carefully',
              'Consider the confidence level when making decisions',
              'Cross-reference with other market data sources'
            ]
          }
        ];
        
        setInsights(prev => [...newInsights, ...prev.slice(0, 4)]); // Keep last 5 insights
      } else {
        setError(`AI query failed: ${response.error || 'Unknown error'}`);
        setAiResponse('Sorry, I encountered an error while processing your request.');
      }
    } catch (error) {
      console.error('MIKEY-AI query error:', error);
      setError(`Network error: ${error instanceof Error ? error.message : 'Unknown error'}`);
      setAiResponse('Sorry, I encountered a network error while processing your request.');
    } finally {
      setIsLoading(false);
    }
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'analysis': return '📊';
      case 'management': return '⚖️';
      case 'insights': return '💡';
      case 'risk': return '⚠️';
      default: return '🤖';
    }
  };

  const getInsightIcon = (type: string) => {
    switch (type) {
      case 'portfolio': return '💼';
      case 'sentiment': return '📈';
      case 'position': return '⚖️';
      case 'market': return '📊';
      case 'risk': return '⚠️';
      default: return '🤖';
    }
  };

  const selectedScenarioData = tradingScenarios.find(s => s.id === selectedScenario);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          🤖 MIKEY AI Integration Showcase
        </h2>
        <p className="text-gray-600">
          AI-powered trading intelligence - Beyond Drift's capabilities
        </p>
      </div>

      {/* Service Status */}
      <div className={`p-4 rounded-lg border ${
        serviceStatus 
          ? 'bg-green-50 border-green-200' 
          : 'bg-red-50 border-red-200'
      }`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <span className={`text-2xl mr-3 ${serviceStatus ? 'text-green-600' : 'text-red-600'}`}>
              {serviceStatus ? '🟢' : '🔴'}
            </span>
            <div>
              <h3 className={`font-medium ${serviceStatus ? 'text-green-800' : 'text-red-800'}`}>
                MIKEY-AI Service Status
              </h3>
              <p className={`text-sm ${serviceStatus ? 'text-green-600' : 'text-red-600'}`}>
                {serviceStatus 
                  ? `Connected to MIKEY-AI on port 3000 (${llmStatus.length} LLM providers available)`
                  : 'Service unavailable - Check if MIKEY-AI is running on port 3000'
                }
              </p>
            </div>
          </div>
          {serviceStatus && (
            <div className="text-sm text-green-600">
              <div className="font-medium">Available LLMs:</div>
              <div className="flex gap-2 mt-1">
                {llmStatus.map(llm => (
                  <span key={llm.name} className="px-2 py-1 bg-green-100 rounded text-xs">
                    {llm.name} ({llm.status})
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
        {error && (
          <div className="mt-3 p-3 bg-red-100 border border-red-300 rounded text-red-700 text-sm">
            <strong>Error:</strong> {error}
          </div>
        )}
      </div>

      {/* AI Insights Dashboard */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          📊 Real-time AI Insights
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {insights.map((insight) => (
            <div key={insight.id} className="p-4 bg-gray-50 rounded-lg">
              <div className="flex items-center mb-2">
                <span className="text-2xl mr-2">{getInsightIcon(insight.type)}</span>
                <div className="flex-1">
                  <h4 className="font-medium text-gray-900">{insight.title}</h4>
                  <div className="flex items-center">
                    <div className="w-16 bg-gray-200 rounded-full h-2 mr-2">
                      <div 
                        className="bg-green-600 h-2 rounded-full" 
                        style={{width: `${insight.confidence * 100}%`}}
                      ></div>
                    </div>
                    <span className="text-sm text-gray-600">
                      {Math.round(insight.confidence * 100)}%
                    </span>
                  </div>
                </div>
              </div>
              
              <p className="text-sm text-gray-700 mb-3">
                {insight.description}
              </p>
              
              {insight.recommendations && (
                <div className="space-y-1">
                  {insight.recommendations.map((rec, index) => (
                    <div key={index} className="text-xs text-blue-700">
                      • {rec}
                    </div>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Trading Scenarios */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Scenarios List */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            🎯 AI Trading Scenarios
          </h3>
          
          <div className="space-y-3">
            {tradingScenarios.map((scenario) => (
              <div
                key={scenario.id}
                className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                  selectedScenario === scenario.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => handleScenarioClick(scenario)}
              >
                <div className="flex items-center mb-2">
                  <span className="text-2xl mr-3">{getCategoryIcon(scenario.category)}</span>
                  <div className="flex-1">
                    <h4 className="font-medium text-gray-900">{scenario.title}</h4>
                    <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded">
                      {scenario.category}
                    </span>
                  </div>
                </div>
                
                <p className="text-sm text-gray-600 mb-2">
                  {scenario.description}
                </p>
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-1">
                    {scenario.capabilities.map((capability) => (
                      <span key={capability} className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                        {capability}
                      </span>
                    ))}
                  </div>
                  <span className="text-sm text-blue-600">
                    Try Scenario →
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* AI Response */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            🤖 AI Response
          </h3>
          
          {selectedScenarioData ? (
            <div className="h-96 overflow-y-auto">
              {isLoading ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                    <p className="text-gray-600">MIKEY AI is analyzing...</p>
                  </div>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="bg-blue-50 p-4 rounded-lg">
                    <h4 className="font-medium text-blue-900 mb-2">
                      Query: {selectedScenarioData.title}
                    </h4>
                    <p className="text-sm text-blue-700">
                      {selectedScenarioData.query}
                    </p>
                  </div>
                  
                  <div className="bg-gray-900 p-4 rounded-lg">
                    <pre className="text-green-400 text-sm whitespace-pre-wrap">
                      {aiResponse}
                    </pre>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="h-96 flex items-center justify-center text-gray-500">
              <div className="text-center">
                <div className="text-4xl mb-4">🤖</div>
                <h3 className="text-lg font-medium mb-2">Select a Scenario</h3>
                <p>Choose a trading scenario to see MIKEY AI in action</p>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* AI Capabilities */}
      <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-4">
          🚀 MIKEY AI Capabilities
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-blue-900 mb-2">What MIKEY Can Do</h4>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Analyze your portfolio and suggest optimizations</li>
              <li>• Monitor market sentiment across multiple sources</li>
              <li>• Provide intelligent position management advice</li>
              <li>• Assess portfolio risk and suggest mitigation</li>
              <li>• Detect trading opportunities in real-time</li>
              <li>• Optimize trade execution timing and strategy</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-blue-900 mb-2">How to Interact with MIKEY</h4>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Ask natural language questions about your portfolio</li>
              <li>• Request market analysis for specific assets</li>
              <li>• Get position management recommendations</li>
              <li>• Receive risk assessment reports</li>
              <li>• Access real-time trading opportunities</li>
              <li>• Integrate via API for automated strategies</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Competitive Advantages */}
      <div className="bg-green-50 rounded-lg p-6 border border-green-200">
        <h3 className="text-lg font-semibold text-green-900 mb-4">
          🎯 AI Advantages Over Drift Protocol
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-green-900 mb-2">Drift Protocol</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• No AI integration</li>
              <li>• Basic trading tools only</li>
              <li>• Manual analysis required</li>
              <li>• No intelligent recommendations</li>
              <li>• Limited market insights</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-green-900 mb-2">QuantDesk + MIKEY AI</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Intelligent AI trading assistant</li>
              <li>• Automated analysis and insights</li>
              <li>• Natural language interaction</li>
              <li>• AI-powered recommendations</li>
              <li>• Comprehensive market intelligence</li>
              <li>• Real-time data processing</li>
              <li>• Portfolio optimization</li>
              <li>• Risk management automation</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Integration Benefits */}
      <div className="bg-purple-50 rounded-lg p-6 border border-purple-200">
        <h3 className="text-lg font-semibold text-purple-900 mb-4">
          🔗 QuantDesk + MIKEY AI Integration Benefits
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div>
            <h4 className="font-medium text-purple-900 mb-2">For Traders</h4>
            <ul className="text-sm text-purple-700 space-y-1">
              <li>• AI-powered trading insights</li>
              <li>• Automated risk management</li>
              <li>• Intelligent position sizing</li>
              <li>• Market opportunity detection</li>
              <li>• Real-time portfolio optimization</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-purple-900 mb-2">For Developers</h4>
            <ul className="text-sm text-purple-700 space-y-1">
              <li>• AI API integration examples</li>
              <li>• Natural language query interface</li>
              <li>• Real-time data processing</li>
              <li>• Automated strategy development</li>
              <li>• MCP integration potential</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-purple-900 mb-2">For Platform</h4>
            <ul className="text-sm text-purple-700 space-y-1">
              <li>• Competitive differentiation</li>
              <li>• Enhanced user experience</li>
              <li>• Advanced analytics</li>
              <li>• Intelligent automation</li>
              <li>• Market leadership position</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
