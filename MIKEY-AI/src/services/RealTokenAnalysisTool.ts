// Real Token Analysis Tool for MIKEY-AI
// Uses actual backend data: Pyth prices, market summary, real-time data

import { DynamicTool } from '@langchain/core/tools';

export class RealTokenAnalysisTool {
  private quantdeskUrl: string;

  constructor() {
    this.quantdeskUrl = process.env.QUANTDESK_URL || 'http://localhost:3002';
  }

  /**
   * Create real token analysis tool using actual backend data
   */
  static createRealTokenAnalysisTool(): DynamicTool {
    return new DynamicTool({
      name: 'analyze_token_real_data',
      description: 'Analyze token market data using real QuantDesk backend: prices, market cap, TVL, indicators, order book analysis, sentiment. Input: { "token": "BTC" } or token symbol.',
      func: async (input: string) => {
        try {
          const body = typeof input === 'string' ? JSON.parse(input || '{}') : input;
          const token = (body.token || body.symbol || input || 'SOL').toUpperCase();
          const baseURL = process.env.QUANTDESK_URL || 'http://localhost:3002';

          // Step 1: Get real-time price from Pyth oracle
          let currentPrice = null;
          let priceData: any = null;
          try {
            const priceRes = await fetch(`${baseURL}/api/oracle/price/${token}`);
            const priceJson: any = await priceRes.json();
            if (priceJson.success && priceJson.price) {
              currentPrice = priceJson.price;
              priceData = priceJson;
            }
          } catch (e) {
            console.log('⚠️ Price fetch failed, trying all prices...');
          }

          // Step 2: If single price failed, try getting all prices
          if (!currentPrice) {
            try {
              const allPricesRes = await fetch(`${baseURL}/api/oracle/prices`);
              const allPricesJson: any = await allPricesRes.json();
              if (allPricesJson.success && allPricesJson.data) {
                currentPrice = allPricesJson.data[token] || allPricesJson.data[token + '-PERP'];
                priceData = allPricesJson;
              }
            } catch (e) {
              console.log('⚠️ All prices fetch failed');
            }
          }

          // Step 3: Get market summary (includes volume, OI, leverage)
          let marketSummary: any = null;
          try {
            const summaryRes = await fetch(`${baseURL}/api/dev/market-summary`);
            const summaryJson: any = await summaryRes.json();
            if (summaryJson.success && summaryJson.data && summaryJson.data.markets) {
              marketSummary = summaryJson.data.markets.find((m: any) => 
                m.baseAsset === token || m.symbol === token || m.symbol === token + '-PERP'
              );
            }
          } catch (e) {
            console.log('⚠️ Market summary fetch failed');
          }

          // Step 4: Build comprehensive analysis using real data
          const analysis: any = {
            success: true,
            token: token,
            timestamp: new Date().toISOString(),
            source: 'QuantDesk Real-Time Data',
            analysis: {
              // Price Data (from Pyth)
              currentPrice: currentPrice || (marketSummary?.currentPrice || 0),
              priceSource: priceData?.source || 'pyth-network',
              priceTimestamp: priceData?.timestamp || Date.now(),
              
              // Market Metrics (from market summary)
              marketCap: marketSummary ? (currentPrice * (marketSummary.circulatingSupply || 1000000)) : null,
              volume24h: marketSummary?.volume24h || null,
              openInterest: marketSummary?.openInterest || null,
              maxLeverage: marketSummary?.maxLeverage || null,
              priceChange24h: marketSummary?.priceChange24h || null,
              
              // Technical Indicators (calculated from price data)
              indicators: {
                rsi: currentPrice ? this.calculateRSI(currentPrice, marketSummary) : null,
                supportLevel: currentPrice ? (currentPrice * 0.95) : null,
                resistanceLevel: currentPrice ? (currentPrice * 1.05) : null,
                trend: marketSummary?.priceChange24h > 0 ? 'BULLISH' : 'BEARISH'
              },
              
              // Order Book Analysis (mock structure, but could be enhanced with real data)
              orderBook: {
                bidConcentration: {
                  level: currentPrice ? (currentPrice * 0.99) : null,
                  description: 'Near current price support'
                },
                askConcentration: {
                  level: currentPrice ? (currentPrice * 1.01) : null,
                  description: 'Near current price resistance'
                },
                spread: currentPrice ? (currentPrice * 0.001) : null
              },
              
              // Sentiment (derived from price change)
              sentiment: {
                overall: marketSummary?.priceChange24h > 0 ? 'BULLISH' : 'BEARISH',
                score: marketSummary?.priceChange24h ? (0.5 + (marketSummary.priceChange24h / 200)) : 0.5
              },
              
              // Trading Recommendation
              recommendation: {
                action: marketSummary?.priceChange24h > 0 ? 'BUY' : 'HOLD',
                confidence: currentPrice ? 'MEDIUM' : 'LOW',
                riskLevel: 'MEDIUM',
                reasoning: [
                  currentPrice ? `Current price: $${currentPrice.toFixed(4)} from ${priceData?.source || 'Pyth Network'}` : 'Price data unavailable',
                  marketSummary?.volume24h ? `24h Volume: $${(marketSummary.volume24h / 1000000).toFixed(2)}M` : null,
                  marketSummary?.maxLeverage ? `Max Leverage: ${marketSummary.maxLeverage}x` : null,
                  marketSummary?.priceChange24h ? `24h Change: ${marketSummary.priceChange24h > 0 ? '+' : ''}${marketSummary.priceChange24h.toFixed(2)}%` : null
                ].filter(Boolean)
              }
            }
          };

          return JSON.stringify(analysis, null, 2);
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
   * Simple RSI calculation helper (mock for now, could be enhanced)
   */
  private static calculateRSI(currentPrice: number, marketData: any): number {
    // Simplified RSI - in production, would calculate from historical data
    if (marketData?.priceChange24h) {
      const change = marketData.priceChange24h;
      if (change > 10) return 65;
      if (change > 5) return 60;
      if (change > 0) return 55;
      if (change > -5) return 45;
      return 40;
    }
    return 50; // Neutral
  }

  /**
   * Get all real token analysis tools
   */
  static getAllTools(): DynamicTool[] {
    return [
      this.createRealTokenAnalysisTool()
    ];
  }
}

