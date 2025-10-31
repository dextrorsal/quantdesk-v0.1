// Demo Mock Tools for MIKEY-AI
// Temporary mock data tools for demo purposes

import { DynamicTool } from '@langchain/core/tools';

export class DemoMockTools {
  /**
   * Mock token analysis tool - Returns comprehensive market analysis
   */
  static createMockTokenAnalysisTool(): DynamicTool {
    return new DynamicTool({
      name: 'analyze_token_market',
      description: 'Analyze token market data including price, TVL, market cap, indicators, order book levels, sentiment. Input: { "token": "FARTCOIN" } or token symbol.',
      func: async (input: string) => {
        try {
          const body = typeof input === 'string' ? JSON.parse(input || '{}') : input;
          const token = (body.token || body.symbol || input || 'FARTCOIN').toUpperCase();
          
          // Mock comprehensive market analysis
          const mockAnalysis = {
            success: true,
            token: token,
            timestamp: new Date().toISOString(),
            analysis: {
              // Price Data
              currentPrice: 0.0247,
              priceChange24h: 12.5, // %
              priceChange7d: -3.2,
              high24h: 0.0258,
              low24h: 0.0231,
              
              // Market Metrics
              marketCap: 24700000, // $24.7M
              fullyDilutedValuation: 49200000, // $49.2M
              totalValueLocked: 8500000, // $8.5M TVL
              circulatingSupply: 1000000000, // 1B tokens
              totalSupply: 2000000000, // 2B tokens
              
              // Volume & Liquidity
              volume24h: 2450000, // $2.45M
              liquidity: 3200000, // $3.2M
              liquidityConcentration: 'HIGH', // Where most liquidity is
              
              // Technical Indicators
              indicators: {
                rsi: 58.5, // RSI
                macd: {
                  value: 0.0012,
                  signal: 'BULLISH',
                  histogram: 0.0008
                },
                supportLevel: 0.0230, // Major support
                resistanceLevel: 0.0265, // Major resistance
                movingAverage: {
                  sma20: 0.0241,
                  sma50: 0.0235,
                  ema: 0.0243
                }
              },
              
              // Order Book Analysis
              orderBook: {
                bidConcentration: {
                  level: 0.0245,
                  volume: 450000, // $450K in bids
                  description: 'Strong support zone'
                },
                askConcentration: {
                  level: 0.0252,
                  volume: 380000, // $380K in asks
                  description: 'Resistance zone'
                },
                orderFlow: 'BULLISH', // More buying pressure
                spread: 0.28, // 0.28%
                depth: 'MEDIUM' // Order book depth
              },
              
              // Sentiment Analysis
              sentiment: {
                overall: 'BULLISH',
                score: 0.68, // 0-1 scale
                socialSentiment: 'POSITIVE',
                fearGreedIndex: 65, // 0-100
                newsSentiment: 'MIXED',
                communityActivity: 'HIGH'
              },
              
              // Trading Recommendations
              recommendation: {
                action: 'BUY',
                confidence: 'MEDIUM-HIGH',
                reasoning: [
                  'Strong support at $0.0230 with high bid concentration',
                  'Positive price momentum (12.5% 24h gain)',
                  'Bullish MACD signal with RSI in healthy range',
                  'Resistance at $0.0265 provides clear target',
                  'Medium-high liquidity reduces slippage risk'
                ],
                riskLevel: 'MEDIUM',
                targetPrice: 0.0265,
                stopLoss: 0.0230,
                positionSize: 'SMALL-MEDIUM',
                timeframe: 'SHORT-TERM'
              },
              
              // Key Levels
              keyLevels: {
                immediateSupport: 0.0240,
                majorSupport: 0.0230,
                immediateResistance: 0.0255,
                majorResistance: 0.0265,
                liquidationZones: [0.0220, 0.0275]
              }
            }
          };
          
          return JSON.stringify(mockAnalysis, null, 2);
        } catch (error: any) {
          return JSON.stringify({ 
            success: false, 
            error: error.message,
            token: input 
          });
        }
      }
    });
  }

  /**
   * Mock position opening tool - Returns success without real transaction
   */
  static createMockOpenPositionTool(): DynamicTool {
    return new DynamicTool({
      name: 'open_position_mock',
      description: 'Open a mock position for demo purposes. Input: { "symbol": "FARTCOIN-PERP", "side": "buy|sell", "size": 0.1, "leverage": 5, "type": "market|limit" }. Returns mock success.',
      func: async (input: string) => {
        try {
          const body = typeof input === 'string' ? JSON.parse(input || '{}') : input;
          const symbol = body.symbol || 'FARTCOIN-PERP';
          const side = body.side || 'buy';
          const size = body.size || 0.1;
          const leverage = body.leverage || 5;
          const orderType = body.type || body.orderType || 'market';
          
          // Mock successful position opening
          const mockResponse = {
            success: true,
            demo: true, // Flag indicating this is mock data
            message: 'Position opened successfully (DEMO MODE)',
            position: {
              id: `demo-pos-${Date.now()}`,
              symbol: symbol,
              side: side.toUpperCase(),
              size: size,
              leverage: leverage,
              orderType: orderType.toUpperCase(),
              entryPrice: 0.0247,
              entryTime: new Date().toISOString(),
              unrealizedPnL: 0,
              liquidationPrice: side.toLowerCase() === 'buy' ? 0.0196 : 0.0298,
              marginUsed: (size * 0.0247) / leverage,
              status: 'OPEN'
            },
            transaction: {
              signature: `demo-sig-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
              type: 'MOCK',
              note: 'This is a demo transaction. No real funds were moved.',
              timestamp: new Date().toISOString()
            },
            note: 'Demo mode - This position is simulated and will not appear on-chain. For real trading, ensure backend is connected.'
          };
          
          return JSON.stringify(mockResponse, null, 2);
        } catch (error: any) {
          return JSON.stringify({ 
            success: false, 
            error: error.message,
            input: input 
          });
        }
      }
    });
  }

  /**
   * Get all demo mock tools
   */
  static getAllTools(): DynamicTool[] {
    return [
      this.createMockTokenAnalysisTool(),
      this.createMockOpenPositionTool()
    ];
  }
}

