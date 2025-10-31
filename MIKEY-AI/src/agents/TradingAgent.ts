import { DynamicTool } from '@langchain/core/tools';
import { systemLogger, errorLogger } from '../utils/logger';
import { QuantDeskTools } from '../services/QuantDeskTools';
import { QuantDeskTradingTools } from '../services/QuantDeskTradingTools';
import { QuantDeskProtocolTools } from '../services/QuantDeskProtocolTools';
import { DemoMockTools } from '../services/DemoMockTools';
import { RealTokenAnalysisTool } from '../services/RealTokenAnalysisTool';
import { SolanaWalletTools } from '../services/SolanaWalletTools';
import { ToolOrchestrator } from '../services/ToolOrchestrator';
import { SupabaseTools } from '../services/SupabaseTools';
import { RealDataTools } from '../services/RealDataTools';
import { SolanaAgentKitTools } from '../services/SolanaAgentKitTools';
import { EnhancedDeFiTools } from '../services/EnhancedDeFiTools';
import { officialLLMRouter } from '../services/OfficialLLMRouter';
import { AIQuery, AIResponse } from '../types';

/**
 * Solana DeFi Trading Intelligence AI Agent
 * Core working version with QuantDesk integration
 */

export class TradingAgent {
  private tools!: DynamicTool[];

  constructor() {
    this.initializeTools();
    systemLogger.startup('TradingAgent', 'Multi-LLM Router initialized');
  }

  /**
   * Process AI query with multi-LLM routing and tool usage
   */
  public async processQuery(query: AIQuery): Promise<AIResponse> {
    try {
      systemLogger.startup('TradingAgent', `Processing query with tools: ${query.query}`);
      
      // Check if query needs QuantDesk API data FIRST (simple price queries)
      if (this.needsQuantDeskData(query.query)) {
        systemLogger.startup('TradingAgent', `Detected QuantDesk query: ${query.query}`);
        return await this.processWithQuantDeskTools(query);
      }
      
      // Check if query needs real data from your pipeline
      if (this.needsRealData(query.query)) {
        systemLogger.startup('TradingAgent', `Detected real data query: ${query.query}`);
        return await this.processWithRealDataTools(query);
      }
      
      // Check if query needs demo mock data (token analysis, mock positions)
      if (this.needsDemoMockData(query.query)) {
        return await this.processWithDemoMockTools(query);
      }
      
      // Check if query needs QuantDesk protocol data (wallet, portfolio)
      if (this.needsQuantDeskProtocolData(query.query)) {
        return await this.processWithQuantDeskProtocolTools(query);
      }
      
      // Check if query needs Solana wallet data
      if (this.needsSolanaWalletData(query.query)) {
        return await this.processWithSolanaWalletTools(query);
      }
      
      // Check if query needs orchestration (complex multi-step workflows)
      if (this.needsOrchestration(query.query)) {
        return await this.processWithOrchestration(query);
      }
      
      // Check if query needs Supabase data
      if (this.needsSupabaseData(query.query)) {
        return await this.processWithSupabaseTools(query);
      }
      
      // Check if query needs Solana Agent Kit tools
      if (this.needsSolanaAgentKit(query.query)) {
        return await this.processWithSolanaAgentKitTools(query);
      }
      
      // Check if query needs enhanced DeFi tools (Jupiter, Raydium, Drift, Mango)
      if (this.needsEnhancedDeFiTools(query.query)) {
        return await this.processWithEnhancedDeFiTools(query);
      }
      
      // For general queries, use LLM directly
      const taskType = this.determineTaskType(query.query);
      const result = await officialLLMRouter.routeRequest(query.query, taskType);
      
      systemLogger.startup('TradingAgent', `Query processed using ${taskType} routing via ${result.provider}`);
      return { 
        response: result.response,
        sources: [],
        confidence: 0.8,
        timestamp: new Date(),
        provider: result.provider
      };
    } catch (error) {
      errorLogger.aiError(error as Error, 'AI query processing');
      return { 
        response: `Error processing your request: ${error instanceof Error ? error.message : 'Unknown error'}`,
        sources: [],
        confidence: 0.0,
        timestamp: new Date()
      };
    }
  }

  /**
   * Determine task type for smart LLM routing
   */
  private determineTaskType(text: string): string {
    const lowerText = text.toLowerCase();
    
    if (lowerText.includes('trade') || lowerText.includes('buy') || lowerText.includes('sell') || 
        lowerText.includes('analyze') || lowerText.includes('price') || lowerText.includes('trend') ||
        lowerText.includes('sol') || lowerText.includes('crypto') || lowerText.includes('market')) {
      return 'trading_analysis';
    }
    if (lowerText.includes('code') || lowerText.includes('function') || lowerText.includes('script') ||
        lowerText.includes('write') || lowerText.includes('create') || lowerText.includes('implement')) {
      return 'code_generation';
    }
    if (lowerText.includes('explain') || lowerText.includes('why') || lowerText.includes('how') ||
        lowerText.includes('what') || lowerText.includes('reasoning')) {
      return 'reasoning';
    }
    if (lowerText.includes('sentiment') || lowerText.includes('emotion') || lowerText.includes('mood') ||
        lowerText.includes('feeling') || lowerText.includes('opinion')) {
      return 'sentiment_analysis';
    }
    if (lowerText.includes('translate') || lowerText.includes('language') || lowerText.includes('chinese') || 
        lowerText.includes('spanish') || lowerText.includes('french') || lowerText.includes('german')) {
      return 'multilingual';
    }
    
    return 'general';
  }

  /**
   * Initialize custom trading tools
   */
  private initializeTools(): void {
    this.tools = [
      // Real Data Tools from your existing pipeline (Pyth, CoinGecko, etc.)
      ...RealDataTools.getAllTools(),
      
      // Real Token Analysis Tool (uses actual backend data)
      ...RealTokenAnalysisTool.getAllTools(),
      
      // QuantDesk API Tools (the main ones!)
      ...QuantDeskTools.getAllTools(),
      
      // QuantDesk Trading Tools (advanced trading operations)
      ...QuantDeskTradingTools.getAllTools(),
      
      // Enhanced DeFi Tools (Jupiter, Raydium, Drift, Mango, NFT markets)
      ...EnhancedDeFiTools.getAllTools(),
      
      // QuantDesk Protocol Tools (wallet integration, portfolio analysis)
      ...QuantDeskProtocolTools.getAllTools(),
      
      // Demo Mock Tools (for demo purposes - market analysis, mock position opening)
      ...DemoMockTools.getAllTools(),
      
      // Solana Wallet Tools (balance checking, transaction history)
      ...SolanaWalletTools.getAllTools(),
      
      // Tool Orchestration (complex multi-step workflows)
      ToolOrchestrator.createOrchestrationTool(),
      
      // Supabase Database Tools
      ...SupabaseTools.getAllTools(),
      
      // Solana Agent Kit Tools (POC - Trading-focused)
      ...SolanaAgentKitTools.getAllTools(),
      
      // Basic placeholder tools
      this.createPlaceholderWalletTool(),
      this.createPlaceholderPriceTool()
    ];
  }

  /**
   * Create placeholder wallet analysis tool
   */
  private createPlaceholderWalletTool(): DynamicTool {
    return new DynamicTool({
      name: 'analyze_solana_wallet',
      description: 'Analyze a Solana wallet address for basic information (placeholder)',
      func: async (input: string) => {
        try {
          const address = input.trim();
          return JSON.stringify({
            address,
            message: 'Wallet analysis service temporarily unavailable - will be restored soon!',
            status: 'placeholder'
          });
        } catch (error) {
          return `Error analyzing wallet: ${error}`;
        }
      }
    });
  }

  /**
   * Create placeholder price tool
   */
  private createPlaceholderPriceTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_crypto_price',
      description: 'Get cryptocurrency price from exchanges (placeholder)',
      func: async (input: string) => {
        try {
          const symbol = input.trim().toUpperCase();
          return JSON.stringify({
            symbol,
            price: 0,
            message: 'Price service temporarily unavailable - will be restored soon!',
            status: 'placeholder'
          });
        } catch (error) {
          return `Error fetching ${input} price: ${error}`;
        }
      }
    });
  }

  /**
   * Check if the query requires real data from your pipeline
   * Priority: Real data tools BEFORE mock tools for analysis queries
   */
  private needsRealData(queryText: string): boolean {
    const lowerQuery = queryText.toLowerCase();
    
    // Check for analyze requests first - should use REAL data unless explicitly mock/demo
    const isAnalyzeRequest = (lowerQuery.includes('analyze') || 
                              lowerQuery.includes('market summary') ||
                              lowerQuery.includes('token analysis') ||
                              lowerQuery.includes('sentiment') ||
                              lowerQuery.includes('tvl') ||
                              lowerQuery.includes('market cap') ||
                              lowerQuery.includes('indicators') ||
                              lowerQuery.includes('order book')) &&
                             !lowerQuery.includes('mock') && 
                             !lowerQuery.includes('demo');
    
    // If analyze request (and not mock), use REAL data
    if (isAnalyzeRequest) {
      return true;
    }
    
    // Other real data patterns
    return lowerQuery.includes('pyth') || 
           lowerQuery.includes('oracle') ||
           lowerQuery.includes('coingecko') ||
           lowerQuery.includes('whale') ||
           lowerQuery.includes('whales') ||
           lowerQuery.includes('large transaction') ||
           lowerQuery.includes('crypto news') ||
           lowerQuery.includes('news') ||
           lowerQuery.includes('article') ||
           lowerQuery.includes('headline') ||
           lowerQuery.includes('coindesk') ||
           lowerQuery.includes('cointelegraph') ||
           lowerQuery.includes('the block') ||
           lowerQuery.includes('arbitrage') ||
           lowerQuery.includes('market analysis') ||
           lowerQuery.includes('real-time') ||
           lowerQuery.includes('live data') ||
           lowerQuery.includes('live price');
  }

  /**
   * Process query using real data tools from your pipeline
   */
  private async processWithRealDataTools(query: AIQuery): Promise<AIResponse> {
    systemLogger.startup('TradingAgent', `Processing with real data tools: ${query.query}`);
    const realDataTools = new RealDataTools();
    const availableTools = RealDataTools.getAllTools();
    const lowerQuery = query.query.toLowerCase();
    
    let toolResponse: string | undefined;

    // Check for token analysis requests first (use RealTokenAnalysisTool)
    if (lowerQuery.includes('analyze') || lowerQuery.includes('token analysis') || 
        lowerQuery.includes('market summary') || lowerQuery.includes('tvl') || 
        lowerQuery.includes('market cap') || lowerQuery.includes('indicators')) {
      systemLogger.startup('TradingAgent', 'Calling Real Token Analysis tool...');
      const tokenMatch = query.query.match(/\b([A-Z]{2,}|[A-Z]+COIN|[A-Z]+)\b/);
      const token = tokenMatch ? tokenMatch[1] : 'BTC';
      const realAnalysisTool = RealTokenAnalysisTool.createRealTokenAnalysisTool();
      toolResponse = await realAnalysisTool.func(JSON.stringify({ token }));
    } else if (lowerQuery.includes('pyth') || lowerQuery.includes('oracle')) {
      systemLogger.startup('TradingAgent', 'Calling Pyth price tool...');
      toolResponse = await realDataTools.createPythPriceTool().func('');
    } else if (lowerQuery.includes('coingecko') || lowerQuery.includes('market price')) {
      systemLogger.startup('TradingAgent', 'Calling CoinGecko price tool...');
      toolResponse = await realDataTools.createCoinGeckoPriceTool().func('');
    } else if (lowerQuery.includes('whale') || lowerQuery.includes('large transaction')) {
      systemLogger.startup('TradingAgent', 'Calling whale data tool...');
      toolResponse = await realDataTools.createWhaleDataTool().func('');
    } else if (lowerQuery.includes('news') || lowerQuery.includes('article') || 
               lowerQuery.includes('headline') || lowerQuery.includes('coindesk') ||
               lowerQuery.includes('cointelegraph') || lowerQuery.includes('the block') ||
               (lowerQuery.includes('sentiment') && !lowerQuery.includes('analyze'))) {
      systemLogger.startup('TradingAgent', 'Calling news data tool...');
      
      // Extract ticker from query if mentioned (e.g., "BTC news", "news about ETH")
      const tickerMatch = query.query.match(/\b(BTC|ETH|SOL|ADA|DOT|LINK|AVAX|MATIC|BNB|XRP|DOGE|LTC|BCH|XLM|ALGO|ATOM|FIL|TRX|EOS|AAVE|MKR|SNX|SUSHI|CRV|COMP|YFI|RUNE|LUNA|ICP|FTT|BONK|WIF|PEPE|FLOKI|SHIB|JUP|JTO|WEN|PYTH|RAY)\b/i);
      const ticker = tickerMatch ? tickerMatch[1].toUpperCase() : null;
      
      // Prioritize CoinDesk, CoinTelegraph, The Block
      const newsInput = JSON.stringify({
        ticker: ticker || null,
        sources: 'CoinDesk,CoinTelegraph,The Block',
        limit: 15
      });
      
      toolResponse = await realDataTools.createNewsDataTool().func(newsInput);
    } else if (lowerQuery.includes('market analysis') || lowerQuery.includes('markets')) {
      systemLogger.startup('TradingAgent', 'Calling market data tool...');
      toolResponse = await realDataTools.createMarketDataTool().func('');
    } else if (lowerQuery.includes('arbitrage')) {
      systemLogger.startup('TradingAgent', 'Calling arbitrage tool...');
      toolResponse = await realDataTools.createArbitrageTool().func('');
    }

    if (toolResponse) {
      // Format response nicely for LLM
      try {
        const parsed = JSON.parse(toolResponse);
        
        // Handle news articles response
        if (parsed.articles && Array.isArray(parsed.articles)) {
          const articles = parsed.articles;
          const sourcesList = parsed.sources?.join(', ') || 'Top Crypto News Sources';
          const filteredBy = parsed.filteredBy;
          
          let formatted = `📰 Latest Crypto News from ${sourcesList} (${articles.length} articles)\n\n`;
          
          if (filteredBy?.ticker && filteredBy.ticker !== 'all') {
            formatted += `Filtered by ticker: ${filteredBy.ticker}\n`;
          }
          if (filteredBy?.keyword && filteredBy.keyword !== 'none') {
            formatted += `Search keyword: ${filteredBy.keyword}\n`;
          }
          if (filteredBy?.category && filteredBy.category !== 'all') {
            formatted += `Category: ${filteredBy.category}\n`;
          }
          formatted += '\n';
          
          articles.slice(0, 10).forEach((article: any) => {
            formatted += `${article.index}. **${article.headline}**\n`;
            formatted += `   Source: ${article.source} | ${article.date} ${article.time}\n`;
            if (article.ticker && article.ticker !== 'N/A') {
              formatted += `   Ticker: ${article.ticker} | Category: ${article.category}\n`;
            }
            if (article.snippet) {
              formatted += `   ${article.snippet.substring(0, 150)}${article.snippet.length > 150 ? '...' : ''}\n`;
            }
            if (article.url) {
              formatted += `   URL: ${article.url}\n`;
            }
            formatted += '\n';
          });
          
      return {
            response: formatted,
            sources: parsed.sources || ['CoinDesk', 'CoinTelegraph', 'The Block'],
            confidence: 0.9,
            timestamp: new Date(),
            provider: 'RealDataTools'
          };
        }
        
        // Handle token analysis response
        if (parsed.analysis) {
          // Format real token analysis response
          const analysis = parsed.analysis;
          const formatted = `Market Analysis for ${parsed.token} (Real-Time Data):
            
**Price Data:**
- Current Price: $${analysis.currentPrice?.toFixed(4) || 'N/A'} (Source: ${parsed.source || 'Pyth Network'})
${analysis.priceChange24h !== null ? `- 24h Change: ${analysis.priceChange24h > 0 ? '+' : ''}${analysis.priceChange24h.toFixed(2)}%` : ''}

**Market Metrics:**
${analysis.marketCap ? `- Market Cap: $${(analysis.marketCap / 1000000).toFixed(1)}M` : ''}
${analysis.volume24h ? `- 24h Volume: $${(analysis.volume24h / 1000000).toFixed(2)}M` : ''}
${analysis.openInterest ? `- Open Interest: $${(analysis.openInterest / 1000000).toFixed(2)}M` : ''}
${analysis.maxLeverage ? `- Max Leverage: ${analysis.maxLeverage}x` : ''}

**Technical Indicators:**
${analysis.indicators?.rsi ? `- RSI: ${analysis.indicators.rsi.toFixed(1)}` : ''}
${analysis.indicators?.supportLevel ? `- Support: $${analysis.indicators.supportLevel.toFixed(4)}` : ''}
${analysis.indicators?.resistanceLevel ? `- Resistance: $${analysis.indicators.resistanceLevel.toFixed(4)}` : ''}
${analysis.indicators?.trend ? `- Trend: ${analysis.indicators.trend}` : ''}

**Sentiment:**
${analysis.sentiment?.overall ? `- Overall: ${analysis.sentiment.overall}` : ''}
${analysis.sentiment?.score ? `- Score: ${(analysis.sentiment.score * 100).toFixed(0)}%` : ''}

**Trading Recommendation:**
${analysis.recommendation?.action ? `- Action: ${analysis.recommendation.action}` : ''}
${analysis.recommendation?.confidence ? `- Confidence: ${analysis.recommendation.confidence}` : ''}
${analysis.recommendation?.riskLevel ? `- Risk Level: ${analysis.recommendation.riskLevel}` : ''}
${analysis.recommendation?.reasoning?.length ? `- Reasoning: ${analysis.recommendation.reasoning.join('. ')}` : ''}

*Data sourced from QuantDesk backend: Pyth Network prices, market summary, and real-time metrics.*`;

          return {
            response: formatted,
            sources: ['QuantDesk Real-Time Data Pipeline'],
            confidence: 0.9,
            timestamp: new Date(),
            provider: 'RealTokenAnalysisTool'
          };
        }
      } catch (e) {
        // If parsing fails, return raw response
      }
      
      return {
        response: toolResponse,
        sources: ['QuantDesk Data Pipeline'],
        confidence: 0.9,
        timestamp: new Date(),
        provider: 'RealDataTools'
      };
    }

    // If no specific tool matched, try to use LLM to interpret and call tools
    const llmPrompt = `The user is asking about real market data. Here is the query: "${query.query}". Use the available real data tools to answer. Available tools: ${availableTools.map(t => t.name).join(', ')}.`;
    const result = await officialLLMRouter.routeRequest(llmPrompt, 'trading_analysis');
    
    return { 
      response: result.response,
      sources: ['QuantDesk Data Pipeline', result.provider],
      confidence: 0.7,
      timestamp: new Date(),
      provider: result.provider
    };
  }

  /**
   * Check if the query requires QuantDesk API data
   */
  private needsQuantDeskData(queryText: string): boolean {
    const lowerQuery = queryText.toLowerCase();
    
    // Simple price queries: "What is the price of X?", "Live price of ETH", etc.
    const isSimplePriceQuery = (lowerQuery.includes('price') || lowerQuery.includes('cost')) &&
                               (lowerQuery.includes('what is') || lowerQuery.includes('show me') ||
                                lowerQuery.includes('get') || lowerQuery.includes('live') ||
                                lowerQuery.includes('current') || lowerQuery.includes('how much')) &&
                               !lowerQuery.includes('analyze') && !lowerQuery.includes('analysis');
    
    if (isSimplePriceQuery) {
      return true;
    }
    
    return lowerQuery.includes('quantdesk') || 
           lowerQuery.includes('market data') || 
           lowerQuery.includes('price data') ||
           lowerQuery.includes('markets') ||
           lowerQuery.includes('prices') ||
           lowerQuery.includes('account info') ||
           lowerQuery.includes('trading data') ||
           lowerQuery.includes('health');
  }

  /**
   * Check if the query needs demo mock data (token analysis, market sentiment, mock trading)
   */
  private needsDemoMockData(queryText: string): boolean {
    const lowerQuery = queryText.toLowerCase();
    // Only use mock tools if explicitly requested or for position opening (demo mode)
    const isExplicitMock = lowerQuery.includes('mock') || lowerQuery.includes('demo');
    const isPositionOpening = lowerQuery.includes('open position') || 
                              lowerQuery.includes('open a position') || 
                              (lowerQuery.includes('place') && lowerQuery.includes('trade'));
    
    // For demo: Use mock tools only if explicitly requested OR for position opening
    return isExplicitMock || isPositionOpening;
  }

  /**
   * Check if the query requires QuantDesk protocol data (portfolio, positions, trading)
   */
  private needsQuantDeskProtocolData(queryText: string): boolean {
    const lowerQuery = queryText.toLowerCase();
    return lowerQuery.includes('portfolio') || 
           lowerQuery.includes('positions') ||
           lowerQuery.includes('place order') ||
           lowerQuery.includes('place trade') ||
           lowerQuery.includes('cancel order') ||
           lowerQuery.includes('close position') ||
           lowerQuery.includes('funding rate') ||
           lowerQuery.includes('open interest') ||
           lowerQuery.includes('risk analysis') ||
           lowerQuery.includes('margin ratio') ||
           lowerQuery.includes('liquidation') ||
           lowerQuery.includes('quantdesk port');
  }

  /**
   * Check if the query requires Solana wallet data
   */
  private needsSolanaWalletData(queryText: string): boolean {
    const lowerQuery = queryText.toLowerCase();
    return lowerQuery.includes('wallet balance') ||
           lowerQuery.includes('sol balance') ||
           lowerQuery.includes('check balance') ||
           lowerQuery.includes('wallet address') ||
           lowerQuery.includes('public key') ||
           lowerQuery.includes('transaction history') ||
           lowerQuery.includes('solana pub wallet') ||
           lowerQuery.includes('program account') ||
           (lowerQuery.includes('balance') && lowerQuery.includes('sol'));
  }

  private needsOrchestration(queryText: string): boolean {
    const lowerQuery = queryText.toLowerCase();
    
    // Complex queries that need multiple tools
    const orchestrationPatterns = [
      // Portfolio analysis (balance + portfolio)
      (lowerQuery.includes('balance') && lowerQuery.includes('portfolio')),
      
      // Comprehensive wallet analysis
      (lowerQuery.includes('wallet') && lowerQuery.includes('analysis') && 
       (lowerQuery.includes('comprehensive') || lowerQuery.includes('complete'))),
      
      // Market sentiment analysis
      (lowerQuery.includes('market') && lowerQuery.includes('sentiment') && 
       lowerQuery.includes('analysis')),
      
      // Multi-step queries
      (lowerQuery.includes('and') && (
        lowerQuery.includes('balance') || 
        lowerQuery.includes('portfolio') || 
        lowerQuery.includes('transaction')
      )),
      
      // Analysis requests
      (lowerQuery.includes('analyze') && (
        lowerQuery.includes('wallet') || 
        lowerQuery.includes('portfolio') || 
        lowerQuery.includes('risk')
      ))
    ];
    
    return orchestrationPatterns.some(pattern => pattern);
  }

  /**
   * Process query using QuantDesk API tools
   */
  private async processWithQuantDeskTools(query: AIQuery): Promise<AIResponse> {
    const quantDeskTools = new QuantDeskTools(process.env.QUANTDESK_URL, process.env.QUANTDESK_API_KEY);
    const availableTools = QuantDeskTools.getAllTools();
    const lowerQuery = query.query.toLowerCase();
    
    let toolResponse: string | undefined;

    // Handle simple price queries: "What is the price of ETH?" or "Live price of SOL"
    if ((lowerQuery.includes('price') || lowerQuery.includes('cost')) &&
        (lowerQuery.includes('what is') || lowerQuery.includes('show me') ||
         lowerQuery.includes('get') || lowerQuery.includes('live') ||
         lowerQuery.includes('current') || lowerQuery.includes('how much'))) {
      systemLogger.startup('TradingAgent', 'Detected simple price query, using QuantDeskProtocolTools...');
      
      // Extract asset symbol (BTC, ETH, SOL, etc.)
      const tokenMatch = query.query.match(/\b([A-Z]{2,}|[A-Z]+COIN|[A-Z]+)\b/);
      const asset = tokenMatch ? tokenMatch[1] : null;
      
      if (asset) {
        // Use QuantDeskProtocolTools for single asset price
        try {
          const livePriceTool = QuantDeskProtocolTools.createGetLivePriceTool();
          toolResponse = await livePriceTool.func(JSON.stringify({ asset }));
        
          if (toolResponse) {
            try {
              const parsed = JSON.parse(toolResponse);
              // Handle response structure: { success: true, price: ..., source: ... } OR { success: true, data: { price: ... } }
              const price = parsed.price || parsed.data?.price;
              const source = parsed.source || parsed.data?.source || 'QuantDesk Oracle';
              const timestamp = parsed.timestamp || parsed.data?.timestamp || new Date().toISOString();
              const change24h = parsed.change24h || parsed.data?.change24h;
              
              if (parsed.success && price) {
                const formatted = `Current ${asset} Price: $${typeof price === 'number' ? price.toFixed(4) : price}
Source: ${source}
Timestamp: ${timestamp}
${change24h ? `24h Change: ${change24h > 0 ? '+' : ''}${typeof change24h === 'number' ? change24h.toFixed(2) : change24h}%` : ''}`;
                
                return {
                  response: formatted,
                  sources: ['QuantDesk Oracle (Pyth Network)'],
                  confidence: 0.9,
                  timestamp: new Date(),
                  provider: 'QuantDeskProtocolTools'
                };
              }
            } catch (e) {
              // If parsing fails, return raw response
              systemLogger.startup('TradingAgent', `Error parsing price response: ${e}`);
            }
          } else {
            systemLogger.startup('TradingAgent', `No tool response for asset: ${asset}`);
          }
        } catch (error: any) {
          systemLogger.startup('TradingAgent', `Error in simple price query: ${error.message}`);
        }
      } else {
        systemLogger.startup('TradingAgent', `Could not extract asset from query: ${query.query}`);
      }
    }

    if (lowerQuery.includes('health')) {
      const healthTool = QuantDeskTools.getAllTools().find(tool => tool.name === 'check_quantdesk_health');
      toolResponse = healthTool ? await healthTool.func('') : null;
    } else if (lowerQuery.includes('markets')) {
      const marketsTool = QuantDeskTools.getAllTools().find(tool => tool.name === 'get_markets');
      toolResponse = marketsTool ? await marketsTool.func('') : null;
    } else if (lowerQuery.includes('prices') || lowerQuery.includes('price data')) {
      const pricesTool = QuantDeskTools.getAllTools().find(tool => tool.name === 'get_prices');
      toolResponse = pricesTool ? await pricesTool.func('') : null;
    } else if (lowerQuery.includes('account info')) {
      const accountTool = QuantDeskTools.getAllTools().find(tool => tool.name === 'get_account_info');
      toolResponse = accountTool ? await accountTool.func('') : null;
    } else if (lowerQuery.includes('trading data') || lowerQuery.includes('orders') || lowerQuery.includes('positions')) {
      const tradingTool = QuantDeskTools.getAllTools().find(tool => tool.name === 'get_trading_data');
      toolResponse = tradingTool ? await tradingTool.func('') : null;
    }

    if (toolResponse) {
      return {
        response: `QuantDesk API response: ${toolResponse}`,
        sources: ['QuantDesk API'],
        confidence: 0.9,
        timestamp: new Date(),
        provider: 'QuantDeskTools'
      };
    }

    // If no specific tool matched, try to use LLM to interpret and call tools
    const llmPrompt = `The user is asking about QuantDesk. Here is the query: "${query.query}". Use the available QuantDesk tools to answer. Available tools: ${availableTools.map(t => t.name).join(', ')}.`;
    const result = await officialLLMRouter.routeRequest(llmPrompt, 'trading_analysis');
    
    return { 
      response: result.response,
      sources: ['QuantDesk API', result.provider],
      confidence: 0.7,
      timestamp: new Date(),
      provider: result.provider
    };
  }

  /**
   * Check if the query requires Supabase data
   */
  private needsSupabaseData(queryText: string): boolean {
    const lowerQuery = queryText.toLowerCase();
    return lowerQuery.includes('historical data') ||
           lowerQuery.includes('news data') ||
           lowerQuery.includes('market data from database') ||
           lowerQuery.includes('user data');
  }

  /**
   * Check if the query requires Solana Agent Kit tools
   */
  private needsSolanaAgentKit(queryText: string): boolean {
    const lowerQuery = queryText.toLowerCase();
    return lowerQuery.includes('token swap') ||
           lowerQuery.includes('jupiter') ||
           lowerQuery.includes('token balance') ||
           lowerQuery.includes('spl token') ||
           lowerQuery.includes('wallet balance') ||
           lowerQuery.includes('sol balance') ||
           lowerQuery.includes('swap quote') ||
           lowerQuery.includes('token data') ||
           lowerQuery.includes('wallet info');
  }

  private needsEnhancedDeFiTools(queryText: string): boolean {
    const lowerQuery = queryText.toLowerCase();
    return lowerQuery.includes('jupiter') ||
           lowerQuery.includes('raydium') ||
           lowerQuery.includes('drift') ||
           lowerQuery.includes('mango') ||
           lowerQuery.includes('swap quote') ||
           lowerQuery.includes('liquidity pool') ||
           lowerQuery.includes('funding rate') ||
           lowerQuery.includes('yield farming') ||
           lowerQuery.includes('arbitrage') ||
           lowerQuery.includes('nft floor') ||
           lowerQuery.includes('tvl') ||
           lowerQuery.includes('apy');
  }

  /**
   * Process query using Supabase database tools
   */
  private async processWithSupabaseTools(query: AIQuery): Promise<AIResponse> {
    const supabaseTools = new SupabaseTools(process.env.SUPABASE_URL!, process.env.SUPABASE_ANON_KEY!);
    const availableTools = SupabaseTools.getAllTools();

    let toolResponse: string | undefined;

    if (query.query.toLowerCase().includes('historical data')) {
      const historicalTool = SupabaseTools.getAllTools().find(tool => tool.name === 'get_historical_data');
      toolResponse = historicalTool ? await historicalTool.func(JSON.stringify({ symbol: 'SOL', timeframe: '1d' })) : null;
    } else if (query.query.toLowerCase().includes('news data')) {
      const newsTool = SupabaseTools.getAllTools().find(tool => tool.name === 'get_news_data');
      toolResponse = newsTool ? await newsTool.func('') : null;
    } else if (query.query.toLowerCase().includes('market data from database')) {
      const marketTool = SupabaseTools.getAllTools().find(tool => tool.name === 'get_market_data');
      toolResponse = marketTool ? await marketTool.func('') : null;
    } else if (query.query.toLowerCase().includes('user data')) {
      const userTool = SupabaseTools.getAllTools().find(tool => tool.name === 'get_user_data');
      toolResponse = userTool ? await userTool.func('') : null;
    }

    if (toolResponse) {
      return {
        response: `Supabase Database response: ${toolResponse}`,
        sources: ['Supabase Database'],
        confidence: 0.9,
        timestamp: new Date(),
        provider: 'SupabaseTools'
      };
    }

    // If no specific tool matched, try to use LLM to interpret and call tools
    const llmPrompt = `The user is asking about Supabase data. Here is the query: "${query.query}". Use the available Supabase tools to answer. Available tools: ${availableTools.map(t => t.name).join(', ')}.`;
    const result = await officialLLMRouter.routeRequest(llmPrompt, 'general');
    
    return { 
      response: result.response,
      sources: ['Supabase Database', result.provider],
      confidence: 0.7,
      timestamp: new Date(),
      provider: result.provider
    };
  }

  /**
   * Process query using Solana Agent Kit tools
   */
  private async processWithSolanaAgentKitTools(query: AIQuery): Promise<AIResponse> {
    const solanaTools = new SolanaAgentKitTools();
    const availableTools = SolanaAgentKitTools.getAllTools();

    let toolResponse: string | undefined;

    if (query.query.toLowerCase().includes('wallet balance') || query.query.toLowerCase().includes('sol balance')) {
      const walletTool = availableTools.find(tool => tool.name === 'get_wallet_balance');
      toolResponse = walletTool ? await walletTool.func('mock-wallet-address') : null;
    } else if (query.query.toLowerCase().includes('token balance')) {
      const tokenTool = availableTools.find(tool => tool.name === 'get_token_balance');
      toolResponse = tokenTool ? await tokenTool.func('mock-wallet-address,EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v') : null;
    } else if (query.query.toLowerCase().includes('swap quote')) {
      const swapTool = availableTools.find(tool => tool.name === 'get_swap_quote');
      toolResponse = swapTool ? await swapTool.func('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v,So11111111111111111111111111111111111111112,100') : null;
    } else if (query.query.toLowerCase().includes('token data')) {
      const tokenDataTool = availableTools.find(tool => tool.name === 'get_token_data');
      toolResponse = tokenDataTool ? await tokenDataTool.func('EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v') : null;
    } else if (query.query.toLowerCase().includes('wallet info')) {
      const walletInfoTool = availableTools.find(tool => tool.name === 'get_wallet_info');
      toolResponse = walletInfoTool ? await walletInfoTool.func('mock-wallet-address') : null;
    }

    if (toolResponse) {
      return {
        response: `Solana Agent Kit response: ${toolResponse}`,
        sources: ['Solana Agent Kit'],
        confidence: 0.8,
        timestamp: new Date(),
        provider: 'SolanaAgentKitTools'
      };
    }

    // If no specific tool matched, use LLM
    const llmPrompt = `The user is asking about Solana Agent Kit operations: "${query.query}". Use available tools.`;
    const result = await officialLLMRouter.routeRequest(llmPrompt, 'trading_analysis');
    
    return {
      response: result.response,
      sources: ['Solana Agent Kit', result.provider],
      confidence: 0.7,
      timestamp: new Date(),
      provider: result.provider
    };
  }

  /**
   * Process query using enhanced DeFi tools
   */
  private async processWithEnhancedDeFiTools(query: AIQuery): Promise<AIResponse> {
    const enhancedTools = EnhancedDeFiTools.getAllTools();
    
    let toolResponse: string | undefined;
    let protocolUsed = 'Unknown';

    // Detect which tool to use based on query
    const lowerQuery = query.query.toLowerCase();
    
    if (lowerQuery.includes('jupiter') || lowerQuery.includes('swap quote')) {
      const swapTool = enhancedTools.find(tool => tool.name === 'get_jupiter_swap_quote');
      const params = {
        inputMint: 'So11111111111111111111111111111111111111112', // SOL
        outputMint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', // USDC
        amount: 1000000000,
        slippageBps: 50
      };
      toolResponse = swapTool ? await swapTool.func(JSON.stringify(params)) : null;
      protocolUsed = 'Jupiter';
    } else if (lowerQuery.includes('raydium') || lowerQuery.includes('liquidity pool')) {
      const poolTool = enhancedTools.find(tool => tool.name === 'analyze_raydium_pool');
      toolResponse = poolTool ? await poolTool.func('mock-pool-address') : null;
      protocolUsed = 'Raydium';
    } else if (lowerQuery.includes('drift') || lowerQuery.includes('funding rate')) {
      const driftTool = enhancedTools.find(tool => tool.name === 'check_drift_funding_rate');
      toolResponse = driftTool ? await driftTool.func(JSON.stringify({ market: 'SOL-PERP' })) : null;
      protocolUsed = 'Drift';
    } else if (lowerQuery.includes('mango') || lowerQuery.includes('account health')) {
      const mangoTool = enhancedTools.find(tool => tool.name === 'analyze_mango_account');
      toolResponse = mangoTool ? await mangoTool.func('mock-account-address') : null;
      protocolUsed = 'Mango';
    } else if (lowerQuery.includes('yield') || lowerQuery.includes('farming') || lowerQuery.includes('apy')) {
      const yieldTool = enhancedTools.find(tool => tool.name === 'find_yield_opportunities');
      toolResponse = yieldTool ? await yieldTool.func(JSON.stringify({ token: 'SOL', minAPY: 5 })) : null;
      protocolUsed = 'Multi-Protocol';
    } else if (lowerQuery.includes('arbitrage')) {
      const arbTool = enhancedTools.find(tool => tool.name === 'analyze_arbitrage_opportunities');
      toolResponse = arbTool ? await arbTool.func(JSON.stringify({ token: 'SOL' })) : null;
      protocolUsed = 'Multi-DEX';
    } else if (lowerQuery.includes('nft') || lowerQuery.includes('floor')) {
      const nftTool = enhancedTools.find(tool => tool.name === 'analyze_nft_market');
      toolResponse = nftTool ? await nftTool.func('famous-fox-federation') : null;
      protocolUsed = 'NFT Market';
    }

    if (toolResponse) {
      return {
        response: `${protocolUsed} response: ${toolResponse}`,
        sources: [protocolUsed],
        confidence: 0.85,
        timestamp: new Date(),
        provider: `${protocolUsed}Tools`
      };
    }

    // If no specific tool matched, use LLM
    const llmPrompt = `The user is asking about DeFi protocols: "${query.query}". Provide helpful information.`;
    const result = await officialLLMRouter.routeRequest(llmPrompt, 'trading_analysis');
    
    return {
      response: result.response,
      sources: ['Enhanced DeFi Tools', result.provider],
      confidence: 0.7,
      timestamp: new Date(),
      provider: result.provider
    };
  }

  /**
   * Process query using Demo Mock Tools (market analysis, mock positions for demo)
   */
  private async processWithDemoMockTools(query: AIQuery): Promise<AIResponse> {
    const availableTools = DemoMockTools.getAllTools();
    let toolResponse: string | undefined;
    const lowerQuery = query.query.toLowerCase();

    // Token/market analysis
    if (!toolResponse && (lowerQuery.includes('analyze') || lowerQuery.includes('fartcoin') || 
        lowerQuery.includes('market') || lowerQuery.includes('tvl') || lowerQuery.includes('sentiment') || 
        lowerQuery.includes('indicators') || lowerQuery.includes('order book') || 
        (lowerQuery.includes('support') && lowerQuery.includes('resistance')))) {
      const analysisTool = availableTools.find(tool => tool.name === 'analyze_token_market');
      // Extract token from query (FARTCOIN, BTC, ETH, etc.)
      const tokenMatch = query.query.match(/\b([A-Z]{2,}|[A-Z]+COIN|[A-Z]+)\b/);
      const token = tokenMatch ? tokenMatch[1] : 'FARTCOIN';
      toolResponse = analysisTool ? await analysisTool.func(JSON.stringify({ token })) : null;
    }
    
    // Mock position opening (after analysis or direct request)
    if (!toolResponse && (lowerQuery.includes('open position') || lowerQuery.includes('open a position') || 
        (lowerQuery.includes('place') && (lowerQuery.includes('trade') || lowerQuery.includes('order'))))) {
      const positionTool = availableTools.find(tool => tool.name === 'open_position_mock');
      // Extract order details from query if present
      const symbolMatch = query.query.match(/([A-Z-]+-PERP|[A-Z]+-PERP)/i);
      const sideMatch = query.query.match(/\b(buy|sell)\b/i);
      const sizeMatch = query.query.match(/(\d+\.?\d*)/);
      toolResponse = positionTool ? await positionTool.func(JSON.stringify({
        symbol: symbolMatch ? symbolMatch[1].toUpperCase() : 'FARTCOIN-PERP',
        side: sideMatch ? sideMatch[1].toLowerCase() : 'buy',
        size: sizeMatch ? parseFloat(sizeMatch[1]) : 0.1,
        leverage: 5,
        type: 'market'
      })) : null;
    }

    if (toolResponse) {
      // Parse JSON and format nicely for LLM consumption
      try {
        const parsed = JSON.parse(toolResponse);
        if (parsed.analysis) {
          // Format market analysis response for LLM to interpret
          const formatted = `Market Analysis for ${parsed.token}:
            
Current Price: $${parsed.analysis.currentPrice} (${parsed.analysis.priceChange24h > 0 ? '+' : ''}${parsed.analysis.priceChange24h}% 24h)

Market Metrics:
- Market Cap: $${(parsed.analysis.marketCap / 1000000).toFixed(1)}M
- TVL: $${(parsed.analysis.totalValueLocked / 1000000).toFixed(1)}M
- 24h Volume: $${(parsed.analysis.volume24h / 1000000).toFixed(2)}M

Technical Indicators:
- RSI: ${parsed.analysis.indicators.rsi} (${parsed.analysis.indicators.rsi > 70 ? 'Overbought' : parsed.analysis.indicators.rsi < 30 ? 'Oversold' : 'Neutral'})
- MACD: ${parsed.analysis.indicators.macd.signal}
- Support: $${parsed.analysis.indicators.supportLevel}
- Resistance: $${parsed.analysis.indicators.resistanceLevel}

Order Book Analysis:
- Bid Concentration: $${(parsed.analysis.orderBook.bidConcentration.volume / 1000).toFixed(0)}K at $${parsed.analysis.orderBook.bidConcentration.level} (${parsed.analysis.orderBook.bidConcentration.description})
- Ask Concentration: $${(parsed.analysis.orderBook.askConcentration.volume / 1000).toFixed(0)}K at $${parsed.analysis.orderBook.askConcentration.level} (${parsed.analysis.orderBook.askConcentration.description})
- Order Flow: ${parsed.analysis.orderBook.orderFlow}

Sentiment Analysis:
- Overall: ${parsed.analysis.sentiment.overall} (Score: ${(parsed.analysis.sentiment.score * 100).toFixed(0)}%)
- Social Sentiment: ${parsed.analysis.sentiment.socialSentiment}
- Fear & Greed Index: ${parsed.analysis.sentiment.fearGreedIndex}/100

Trading Recommendation:
- Action: ${parsed.analysis.recommendation.action}
- Confidence: ${parsed.analysis.recommendation.confidence}
- Risk Level: ${parsed.analysis.recommendation.riskLevel}
- Target Price: $${parsed.analysis.recommendation.targetPrice}
- Stop Loss: $${parsed.analysis.recommendation.stopLoss}
- Position Size: ${parsed.analysis.recommendation.positionSize}
- Timeframe: ${parsed.analysis.recommendation.timeframe}

Reasoning: ${parsed.analysis.recommendation.reasoning.join(' ')}`;
          
          return {
            response: formatted,
            sources: ['Demo Mock Tools - Market Analysis'],
            confidence: 0.9,
            timestamp: new Date(),
            provider: 'DemoMockTools'
          };
        } else if (parsed.position) {
          // Format position opening response
          const formatted = `✅ Position Opened Successfully (DEMO MODE)

Position Details:
- Symbol: ${parsed.position.symbol}
- Side: ${parsed.position.side}
- Size: ${parsed.position.size}
- Leverage: ${parsed.position.leverage}x
- Entry Price: $${parsed.position.entryPrice}
- Liquidation Price: $${parsed.position.liquidationPrice}
- Margin Used: $${parsed.position.marginUsed.toFixed(4)}

Transaction: ${parsed.transaction.signature}
Status: ${parsed.position.status}

⚠️ Note: This is a DEMO transaction. No real funds were moved.`;

          return {
            response: formatted,
            sources: ['Demo Mock Tools - Position Opening'],
            confidence: 0.9,
            timestamp: new Date(),
            provider: 'DemoMockTools'
          };
        }
      } catch (e) {
        // If parsing fails, return raw response
      }
      
      return {
        response: toolResponse,
        sources: ['Demo Mock Tools'],
        confidence: 0.9,
        timestamp: new Date(),
        provider: 'DemoMockTools'
      };
    }

    // Fallback to LLM with tool context
    const result = await officialLLMRouter.routeRequest(
      `${query.query}\n\nUse the available demo tools (analyze_token_market, open_position_mock) to answer. Available tools: ${availableTools.map(t => t.name).join(', ')}.`,
      'trading_analysis'
    );
    
    return { 
      response: result.response,
      sources: ['Demo Mock Tools', result.provider],
      confidence: 0.7,
      timestamp: new Date(),
      provider: result.provider
    };
  }

  /**
   * Process query using QuantDesk Protocol tools (portfolio, trading, risk analysis)
   */
  private async processWithQuantDeskProtocolTools(query: AIQuery): Promise<AIResponse> {
    const availableTools = QuantDeskProtocolTools.getAllTools();
    let toolResponse: string | undefined;

    if (query.query.toLowerCase().includes('portfolio')) {
      const portfolioTool = availableTools.find(tool => tool.name === 'check_quantdesk_portfolio');
      toolResponse = portfolioTool ? await portfolioTool.func(query.query) : null;
    } else if (query.query.toLowerCase().includes('market data') || query.query.toLowerCase().includes('funding rate')) {
      const marketTool = availableTools.find(tool => tool.name === 'get_quantdesk_market_data');
      toolResponse = marketTool ? await marketTool.func('') : null;
    } else if (query.query.toLowerCase().includes('place trade') || query.query.toLowerCase().includes('place order')) {
      const tradeTool = availableTools.find(tool => tool.name === 'place_quantdesk_trade');
      toolResponse = tradeTool ? await tradeTool.func(query.query) : null;
    } else if (query.query.toLowerCase().includes('risk analysis') || query.query.toLowerCase().includes('analyze wallet')) {
      const riskTool = availableTools.find(tool => tool.name === 'analyze_wallet_risk');
      toolResponse = riskTool ? await riskTool.func(query.query) : null;
    }

    if (toolResponse) {
      return {
        response: `QuantDesk Protocol response: ${toolResponse}`,
        sources: ['QuantDesk Protocol'],
        confidence: 0.9,
        timestamp: new Date(),
        provider: 'QuantDeskProtocolTools'
      };
    }

    const result = await officialLLMRouter.routeRequest(query.query, 'trading_analysis');
    return { 
      response: result.response,
      sources: ['QuantDesk Protocol', result.provider],
      confidence: 0.7,
      timestamp: new Date(),
      provider: result.provider
    };
  }

  /**
   * Process query using Solana Wallet tools (balance checking, transaction history)
   */
  private async processWithSolanaWalletTools(query: AIQuery): Promise<AIResponse> {
    const availableTools = SolanaWalletTools.getAllTools();
    let toolResponse: string | undefined;

    if (query.query.toLowerCase().includes('balance') || query.query.toLowerCase().includes('sol balance')) {
      const balanceTool = availableTools.find(tool => tool.name === 'check_sol_balance');
      toolResponse = balanceTool ? await balanceTool.func(query.query) : null;
    } else if (query.query.toLowerCase().includes('transaction history')) {
      const txTool = availableTools.find(tool => tool.name === 'get_transaction_history');
      toolResponse = txTool ? await txTool.func(query.query) : null;
    } else if (query.query.toLowerCase().includes('program account')) {
      const programTool = availableTools.find(tool => tool.name === 'check_program_account');
      toolResponse = programTool ? await programTool.func(query.query) : null;
    }

    if (toolResponse) {
      return {
        response: `Solana Wallet response: ${toolResponse}`,
        sources: ['Solana Network'],
        confidence: 0.9,
        timestamp: new Date(),
        provider: 'SolanaWalletTools'
      };
    }

    const result = await officialLLMRouter.routeRequest(query.query, 'trading_analysis');
    return { 
      response: result.response,
      sources: ['Solana Network', result.provider],
      confidence: 0.7,
      timestamp: new Date(),
      provider: result.provider
    };
  }

  private async processWithOrchestration(query: AIQuery): Promise<AIResponse> {
    const orchestrationTool = this.tools.find(tool => tool.name === 'orchestrate_workflow');
    
    if (!orchestrationTool) {
      errorLogger.externalApiError(new Error('Orchestration tool not found'), 'TradingAgent', 'processWithOrchestration');
      const result = await officialLLMRouter.routeRequest(query.query, 'trading_analysis');
      return { 
        response: result.response,
        sources: [result.provider],
        confidence: 0.7,
        timestamp: new Date(),
        provider: result.provider
      };
    }

    try {
      const orchestrationResult = await orchestrationTool.func(query.query);
      const parsedResult = JSON.parse(orchestrationResult);
      
      return {
        response: parsedResult.synthesis || orchestrationResult,
        sources: ['Tool Orchestration', 'Multiple Data Sources'],
        confidence: parsedResult.confidence || 0.8,
        timestamp: new Date(),
        provider: 'ToolOrchestrator'
      };
    } catch (error) {
      errorLogger.externalApiError(error as Error, 'TradingAgent', 'OrchestrateWorkflow');
      const result = await officialLLMRouter.routeRequest(query.query, 'trading_analysis');
      return { 
        response: result.response,
        sources: [result.provider],
        confidence: 0.6,
        timestamp: new Date(),
        provider: result.provider
      };
    }
  }

  /**
   * Get agent status
   */
  getStatus(): { initialized: boolean; toolsCount: number } {
    return {
      initialized: true, // Multi-LLM router is always available
      toolsCount: this.tools.length
    };
  }
}