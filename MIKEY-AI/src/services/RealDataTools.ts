// Real Data Integration Tools for Mikey AI
import { DynamicTool } from '@langchain/core/tools';

export class RealDataTools {
  private quantdeskUrl: string;
  private supabaseUrl: string;
  private supabaseKey: string;

  constructor() {
    this.quantdeskUrl = process.env.QUANTDESK_URL || 'http://localhost:3002';
    this.supabaseUrl = process.env.SUPABASE_URL || '';
    this.supabaseKey = process.env.SUPABASE_ANON_KEY || '';
  }

  /**
   * Get all available real data tools
   */
  static getAllTools(): DynamicTool[] {
    const tools = new RealDataTools();
    return [
      tools.createPythPriceTool(),
      tools.createCoinGeckoPriceTool(),
      tools.createWhaleDataTool(),
      tools.createNewsDataTool(),
      tools.createMarketDataTool(),
      tools.createArbitrageTool()
    ];
  }

  /**
   * Pyth Oracle Price Tool - Real-time oracle prices
   */
  public createPythPriceTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_pyth_prices',
      description: 'Get real-time prices from Pyth Network oracle (BTC, ETH, SOL)',
      func: async (input: string) => {
        try {
          console.log('🔍 Calling Pyth price tool...');
          console.log('📡 URL:', `${this.quantdeskUrl}/api/oracle/prices`);

          const response = await fetch(`${this.quantdeskUrl}/api/oracle/prices`);
          const data = await response.json() as any;

          console.log('📊 Pyth response:', JSON.stringify(data).substring(0, 200));

          if (data.success && data.data) {
            const prices = Object.entries(data.data).map(([symbol, price]) => ({
              symbol,
              price: price as number,
              confidence: 0.95,
              timestamp: data.timestamp
            }));

            return JSON.stringify({
              success: true,
              source: data.source,
              prices: prices,
              count: prices.length,
              timestamp: data.timestamp
            });
          }

          return JSON.stringify({
            success: false,
            error: 'No Pyth price data available',
            response: data,
            url: `${this.quantdeskUrl}/api/oracle/prices`
          });
        } catch (error: any) {
          console.log('❌ Pyth tool error:', error.message);
          return JSON.stringify({
            success: false,
            error: error.message,
            url: `${this.quantdeskUrl}/api/oracle/prices`
          });
        }
      }
    });
  }

  /**
   * CoinGecko Price Tool - Market prices
   */
  public createCoinGeckoPriceTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_coingecko_prices',
      description: 'Get current market prices from CoinGecko API',
      func: async (input: string) => {
        try {
          const response = await fetch(`${this.quantdeskUrl}/api/coingecko/prices`);
          const data = await response.json() as any;
          
          if (data.success && data.prices) {
            return JSON.stringify({
              success: true,
              source: 'CoinGecko API',
              prices: data.prices.map((price: any) => ({
                symbol: price.symbol,
                price: price.price,
                change24h: price.change24h,
                marketCap: price.marketCap,
                volume: price.volume
              })),
              count: data.prices.length
            });
          }
          
          return JSON.stringify({ success: false, error: 'No CoinGecko price data available' });
        } catch (error) {
          return JSON.stringify({ success: false, error: error.message });
        }
      }
    });
  }

  /**
   * Whale Data Tool - Large transaction monitoring
   */
  public createWhaleDataTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_whale_data',
      description: 'Get recent whale movements and large transactions from Solana blockchain',
      func: async (input: string) => {
        try {
          console.log('🐋 Calling whale data tool...');
          console.log('📡 URL:', `${this.quantdeskUrl}/api/v1/trading/whales?threshold=100000&timeframe=24h`);
          
          const response = await fetch(`${this.quantdeskUrl}/api/v1/trading/whales?threshold=100000&timeframe=24h`);
          const data = await response.json() as any;
          
          console.log('📊 Whale response:', JSON.stringify(data).substring(0, 200));
          
          if (data.success && data.whales) {
            return JSON.stringify({
              success: true,
              source: 'Solana Blockchain',
              whales: data.whales.map((whale: any) => ({
                signature: whale.signature,
                wallet: whale.wallet,
                amountSOL: whale.amountSOL,
                amountUSD: whale.amountUSD,
                transactionType: whale.transactionType,
                timestamp: whale.timestamp
              })),
              count: data.whales.length
            });
          }
          
          return JSON.stringify({ 
            success: false, 
            error: 'No whale data available',
            response: data,
            url: `${this.quantdeskUrl}/api/v1/trading/whales`
          });
        } catch (error: any) {
          console.log('❌ Whale tool error:', error.message);
          return JSON.stringify({ 
            success: false, 
            error: error.message,
            url: `${this.quantdeskUrl}/api/v1/trading/whales`
          });
        }
      }
    });
  }

  /**
   * News Data Tool - Crypto news from top sources (CoinDesk, CoinTelegraph, The Block)
   */
  public createNewsDataTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_crypto_news',
      description: 'Get latest crypto news from top sources: CoinDesk, CoinTelegraph, and The Block. Can filter by ticker, category, or keyword. Input: { "ticker": "BTC", "sources": "CoinDesk,CoinTelegraph,The Block", "limit": 10 } or just ticker symbol like "BTC"',
      func: async (input: string) => {
        try {
          console.log('📰 Calling news data tool...');
          
          // Parse input - can be JSON or simple ticker string
          let params: any = {};
          try {
            params = typeof input === 'string' && input.trim() ? JSON.parse(input) : {};
          } catch (e) {
            // If not JSON, treat as ticker symbol
            if (input.trim()) {
              params = { ticker: input.trim().toUpperCase() };
            }
          }
          
          // Build query params - prioritize CoinDesk, CoinTelegraph, The Block
          const queryParams = new URLSearchParams();
          queryParams.append('sources', params.sources || 'CoinDesk,CoinTelegraph,The Block');
          if (params.ticker) queryParams.append('ticker', params.ticker.toUpperCase());
          if (params.category) queryParams.append('category', params.category);
          if (params.keyword) queryParams.append('keyword', params.keyword);
          queryParams.append('limit', params.limit?.toString() || '20');
          
          const url = `${this.quantdeskUrl}/api/news?${queryParams.toString()}`;
          console.log('📡 URL:', url);
          
          const response = await fetch(url);
          const data = await response.json() as any;
          
          if (data.success && data.articles && data.articles.length > 0) {
            // Format news articles nicely
            const formattedArticles = data.articles.map((article: any, index: number) => ({
              index: index + 1,
              headline: article.headline,
              source: article.source,
              ticker: article.ticker,
              category: article.category,
              date: article.date,
              time: article.time,
              snippet: article.snippet,
              url: article.url
            }));
            
            return JSON.stringify({
              success: true,
              source: 'Top Crypto News Sources',
              sources: data.sources || ['CoinDesk', 'CoinTelegraph', 'The Block'],
              articles: formattedArticles,
              count: data.count || formattedArticles.length,
              filteredBy: {
                ticker: params.ticker || 'all',
                category: params.category || 'all',
                keyword: params.keyword || 'none'
              }
            });
          }
          
          return JSON.stringify({ 
            success: false, 
            error: 'No news articles found',
            sources: data.sources || [],
            count: 0
          });
        } catch (error: any) {
          console.log('❌ News tool error:', error.message);
          return JSON.stringify({ 
            success: false, 
            error: error.message,
            url: `${this.quantdeskUrl}/api/news`
          });
        }
      }
    });
  }

  /**
   * Market Data Tool - Comprehensive market analysis
   */
  public createMarketDataTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_market_analysis',
      description: 'Get comprehensive market analysis including TVL, volume, and trends',
      func: async (input: string) => {
        try {
          console.log('📊 Calling market data tool...');
          console.log('📡 URL:', `${this.quantdeskUrl}/api/real-supabase-markets`);
          
          const response = await fetch(`${this.quantdeskUrl}/api/real-supabase-markets`);
          const data = await response.json() as any;
          
          if (data.success && data.markets) {
            return JSON.stringify({
              success: true,
              source: 'QuantDesk Supabase Markets',
              markets: data.markets.map((market: any) => ({
                symbol: market.symbol,
                baseAsset: market.baseAsset,
                quoteAsset: market.quoteAsset,
                type: market.type,
                status: market.status,
                price: market.price,
                volume24h: market.volume24h
              })),
              count: data.markets.length
            });
          }
          
          return JSON.stringify({ success: false, error: 'No market data available' });
        } catch (error) {
          return JSON.stringify({ success: false, error: error.message });
        }
      }
    });
  }

  /**
   * Arbitrage Tool - Cross-exchange arbitrage opportunities
   */
  public createArbitrageTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_arbitrage_opportunities',
      description: 'Find arbitrage opportunities across different exchanges and DEXs',
      func: async (input: string) => {
        try {
          const response = await fetch(`${this.quantdeskUrl}/api/arbitrage/opportunities`);
          const data = await response.json() as any;
          
          if (data.success && data.opportunities) {
            return JSON.stringify({
              success: true,
              source: 'Cross-Exchange Analysis',
              opportunities: data.opportunities.map((opp: any) => ({
                symbol: opp.symbol,
                buyExchange: opp.buyExchange,
                sellExchange: opp.sellExchange,
                buyPrice: opp.buyPrice,
                sellPrice: opp.sellPrice,
                profitPercent: opp.profitPercent,
                profitUSD: opp.profitUSD,
                volume: opp.volume
              })),
              count: data.opportunities.length
            });
          }
          
          return JSON.stringify({ success: false, error: 'No arbitrage opportunities found' });
        } catch (error) {
          return JSON.stringify({ success: false, error: error.message });
        }
      }
    });
  }
}
