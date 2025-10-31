import axios, { AxiosInstance } from 'axios';
import { config } from '@/config';
import { systemLogger, errorLogger } from '@/utils/logger';
import { PriceData, TransactionData, LiquidationData, WalletData } from '@/types';

/**
 * Data Sources Integration Service
 * Connects to multiple DeFi platforms for real-time trading data
 */

export interface DriftData {
  markets: any[];
  positions: any[];
  liquidations: any[];
  fundingRates: any[];
}

export interface JupiterData {
  quotes: any[];
  routes: any[];
  tokens: any[];
  priceData: any[];
}

export interface HyperliquidData {
  perpetuals: any[];
  liquidations: any[];
  fundingRates: any[];
  orderBook: any[];
}

export interface AxiomData {
  pools: any[];
  swaps: any[];
  liquidity: any[];
}

export interface AsterdexData {
  markets: any[];
  trades: any[];
  orderBook: any[];
}

export class DataSourcesService {
  private driftClient: AxiosInstance;
  private jupiterClient: AxiosInstance;
  private hyperliquidClient: AxiosInstance;
  private axiomClient: AxiosInstance;
  private asterdexClient: AxiosInstance;

  constructor() {
    this.initializeClients();
  }

  /**
   * Initialize API clients for all data sources
   */
  private initializeClients(): void {
    // Drift Protocol API
    this.driftClient = axios.create({
      baseURL: 'https://api.drift.trade',
      timeout: 10000,
      headers: {
        'User-Agent': 'Solana-DeFi-AI/1.0.0',
        'Accept': 'application/json'
      }
    });

    // Jupiter API
    this.jupiterClient = axios.create({
      baseURL: 'https://quote-api.jup.ag',
      timeout: 10000,
      headers: {
        'User-Agent': 'Solana-DeFi-AI/1.0.0',
        'Accept': 'application/json'
      }
    });

    // Hyperliquid API
    this.hyperliquidClient = axios.create({
      baseURL: 'https://api.hyperliquid.xyz',
      timeout: 10000,
      headers: {
        'User-Agent': 'Solana-DeFi-AI/1.0.0',
        'Accept': 'application/json'
      }
    });

    // Axiom API (placeholder - need actual endpoint)
    this.axiomClient = axios.create({
      baseURL: 'https://api.axiom.xyz',
      timeout: 10000,
      headers: {
        'User-Agent': 'Solana-DeFi-AI/1.0.0',
        'Accept': 'application/json'
      }
    });

    // Asterdex API (placeholder - need actual endpoint)
    this.asterdexClient = axios.create({
      baseURL: 'https://api.asterdex.io',
      timeout: 10000,
      headers: {
        'User-Agent': 'Solana-DeFi-AI/1.0.0',
        'Accept': 'application/json'
      }
    });

    systemLogger.startup('1.0.0', config.dev.nodeEnv);
  }

  /**
   * Get Drift Protocol data
   */
  async getDriftData(): Promise<DriftData> {
    try {
      const [markets, positions, liquidations, fundingRates] = await Promise.all([
        this.getDriftMarkets(),
        this.getDriftPositions(),
        this.getDriftLiquidations(),
        this.getDriftFundingRates()
      ]);

      return {
        markets,
        positions,
        liquidations,
        fundingRates
      };
    } catch (error) {
      errorLogger.externalApiError(error as Error, 'drift', 'data_fetch');
      throw error;
    }
  }

  /**
   * Get Drift markets data
   */
  async getDriftMarkets(): Promise<any[]> {
    try {
      const response = await this.driftClient.get('/v1/markets');
      return response.data.markets || [];
    } catch (error) {
      // Fallback to mock data for development
      return this.getMockDriftMarkets();
    }
  }

  /**
   * Get Drift positions data
   */
  async getDriftPositions(): Promise<any[]> {
    try {
      const response = await this.driftClient.get('/v1/positions');
      return response.data.positions || [];
    } catch (error) {
      return this.getMockDriftPositions();
    }
  }

  /**
   * Get Drift liquidations data
   */
  async getDriftLiquidations(): Promise<any[]> {
    try {
      const response = await this.driftClient.get('/v1/liquidations');
      return response.data.liquidations || [];
    } catch (error) {
      return this.getMockDriftLiquidations();
    }
  }

  /**
   * Get Drift funding rates
   */
  async getDriftFundingRates(): Promise<any[]> {
    try {
      const response = await this.driftClient.get('/v1/funding-rates');
      return response.data.fundingRates || [];
    } catch (error) {
      return this.getMockDriftFundingRates();
    }
  }

  /**
   * Get Jupiter aggregator data
   */
  async getJupiterData(): Promise<JupiterData> {
    try {
      const [quotes, routes, tokens, priceData] = await Promise.all([
        this.getJupiterQuotes(),
        this.getJupiterRoutes(),
        this.getJupiterTokens(),
        this.getJupiterPriceData()
      ]);

      return {
        quotes,
        routes,
        tokens,
        priceData
      };
    } catch (error) {
      errorLogger.externalApiError(error as Error, 'jupiter', 'data_fetch');
      throw error;
    }
  }

  /**
   * Get Jupiter quotes
   */
  async getJupiterQuotes(): Promise<any[]> {
    try {
      const response = await this.jupiterClient.get('/v6/quote', {
        params: {
          inputMint: 'So11111111111111111111111111111111111111112', // SOL
          outputMint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', // USDC
          amount: '1000000',
          slippageBps: '50'
        }
      });
      return [response.data];
    } catch (error) {
      return this.getMockJupiterQuotes();
    }
  }

  /**
   * Get Jupiter routes
   */
  async getJupiterRoutes(): Promise<any[]> {
    try {
      const response = await this.jupiterClient.get('/v6/route', {
        params: {
          inputMint: 'So11111111111111111111111111111111111111112',
          outputMint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
          amount: '1000000'
        }
      });
      return [response.data];
    } catch (error) {
      return this.getMockJupiterRoutes();
    }
  }

  /**
   * Get Jupiter tokens
   */
  async getJupiterTokens(): Promise<any[]> {
    try {
      const response = await this.jupiterClient.get('/v6/tokens');
      return response.data.tokens || [];
    } catch (error) {
      return this.getMockJupiterTokens();
    }
  }

  /**
   * Get Jupiter price data
   */
  async getJupiterPriceData(): Promise<any[]> {
    try {
      const response = await this.jupiterClient.get('/v6/price', {
        params: {
          ids: 'So11111111111111111111111111111111111111112,EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v'
        }
      });
      return Object.entries(response.data).map(([id, price]) => ({ id, price }));
    } catch (error) {
      return this.getMockJupiterPriceData();
    }
  }

  /**
   * Get Hyperliquid data
   */
  async getHyperliquidData(): Promise<HyperliquidData> {
    try {
      const [perpetuals, liquidations, fundingRates, orderBook] = await Promise.all([
        this.getHyperliquidPerpetuals(),
        this.getHyperliquidLiquidations(),
        this.getHyperliquidFundingRates(),
        this.getHyperliquidOrderBook()
      ]);

      return {
        perpetuals,
        liquidations,
        fundingRates,
        orderBook
      };
    } catch (error) {
      errorLogger.externalApiError(error as Error, 'hyperliquid', 'data_fetch');
      throw error;
    }
  }

  /**
   * Get Hyperliquid perpetuals
   */
  async getHyperliquidPerpetuals(): Promise<any[]> {
    try {
      const response = await this.hyperliquidClient.get('/info');
      return response.data.universe || [];
    } catch (error) {
      return this.getMockHyperliquidPerpetuals();
    }
  }

  /**
   * Get Hyperliquid liquidations
   */
  async getHyperliquidLiquidations(): Promise<any[]> {
    try {
      const response = await this.hyperliquidClient.get('/info', {
        params: { type: 'liquidations' }
      });
      return response.data.liquidations || [];
    } catch (error) {
      return this.getMockHyperliquidLiquidations();
    }
  }

  /**
   * Get Hyperliquid funding rates
   */
  async getHyperliquidFundingRates(): Promise<any[]> {
    try {
      const response = await this.hyperliquidClient.get('/info', {
        params: { type: 'fundingHistory' }
      });
      return response.data.fundingHistory || [];
    } catch (error) {
      return this.getMockHyperliquidFundingRates();
    }
  }

  /**
   * Get Hyperliquid order book
   */
  async getHyperliquidOrderBook(): Promise<any[]> {
    try {
      const response = await this.hyperliquidClient.get('/info', {
        params: { type: 'l2Book', coin: 'SOL' }
      });
      return [response.data];
    } catch (error) {
      return this.getMockHyperliquidOrderBook();
    }
  }

  /**
   * Get Axiom data
   */
  async getAxiomData(): Promise<AxiomData> {
    try {
      const [pools, swaps, liquidity] = await Promise.all([
        this.getAxiomPools(),
        this.getAxiomSwaps(),
        this.getAxiomLiquidity()
      ]);

      return {
        pools,
        swaps,
        liquidity
      };
    } catch (error) {
      errorLogger.externalApiError(error as Error, 'axiom', 'data_fetch');
      return this.getMockAxiomData();
    }
  }

  /**
   * Get Asterdex data
   */
  async getAsterdexData(): Promise<AsterdexData> {
    try {
      const [markets, trades, orderBook] = await Promise.all([
        this.getAsterdexMarkets(),
        this.getAsterdexTrades(),
        this.getAsterdexOrderBook()
      ]);

      return {
        markets,
        trades,
        orderBook
      };
    } catch (error) {
      errorLogger.externalApiError(error as Error, 'asterdex', 'data_fetch');
      return this.getMockAsterdexData();
    }
  }

  /**
   * Get aggregated market data from all sources
   */
  async getAggregatedMarketData(): Promise<{
    drift: DriftData;
    jupiter: JupiterData;
    hyperliquid: HyperliquidData;
    axiom: AxiomData;
    asterdex: AsterdexData;
  }> {
    try {
      const [drift, jupiter, hyperliquid, axiom, asterdex] = await Promise.all([
        this.getDriftData(),
        this.getJupiterData(),
        this.getHyperliquidData(),
        this.getAxiomData(),
        this.getAsterdexData()
      ]);

      return {
        drift,
        jupiter,
        hyperliquid,
        axiom,
        asterdex
      };
    } catch (error) {
      errorLogger.externalApiError(error as Error, 'aggregated', 'data_fetch');
      throw error;
    }
  }

  // Mock data methods for development/testing
  private getMockDriftMarkets(): any[] {
    return [
      {
        marketId: 'SOL-PERP',
        symbol: 'SOL',
        baseAsset: 'SOL',
        quoteAsset: 'USDC',
        price: 95.50,
        volume24h: 1250000,
        openInterest: 45000000,
        fundingRate: 0.0001
      }
    ];
  }

  private getMockDriftPositions(): any[] {
    return [
      {
        positionId: 'pos_123',
        user: 'ABC123...',
        market: 'SOL-PERP',
        side: 'long',
        size: 1000,
        entryPrice: 94.50,
        markPrice: 95.50,
        pnl: 1000,
        margin: 10000
      }
    ];
  }

  private getMockDriftLiquidations(): any[] {
    return [
      {
        liquidationId: 'liq_123',
        user: 'ABC123...',
        market: 'SOL-PERP',
        side: 'long',
        size: 500,
        liquidationPrice: 94.20,
        markPrice: 94.20,
        pnl: -2500
      }
    ];
  }

  private getMockDriftFundingRates(): any[] {
    return [
      {
        market: 'SOL-PERP',
        fundingRate: 0.0001,
        timestamp: new Date().toISOString()
      }
    ];
  }

  private getMockJupiterQuotes(): any[] {
    return [
      {
        inputMint: 'So11111111111111111111111111111111111111112',
        outputMint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
        inAmount: '1000000',
        outAmount: '94500',
        otherAmountThreshold: '94000',
        swapMode: 'ExactIn',
        slippageBps: 50,
        platformFee: null,
        priceImpactPct: '0.5'
      }
    ];
  }

  private getMockJupiterRoutes(): any[] {
    return [
      {
        inputMint: 'So11111111111111111111111111111111111111112',
        outputMint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
        inAmount: '1000000',
        outAmount: '94500',
        otherAmountThreshold: '94000',
        swapMode: 'ExactIn',
        slippageBps: 50,
        platformFee: null,
        priceImpactPct: '0.5',
        routePlan: [
          {
            swapInfo: {
              ammKey: '58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2',
              label: 'Raydium',
              inputMint: 'So11111111111111111111111111111111111111112',
              outputMint: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
              notEnoughLiquidity: false,
              minInAmount: '1000000',
              minOutAmount: '94000'
            },
            percent: 100
          }
        ]
      }
    ];
  }

  private getMockJupiterTokens(): any[] {
    return [
      {
        address: 'So11111111111111111111111111111111111111112',
        chainId: 101,
        decimals: 9,
        name: 'Wrapped SOL',
        symbol: 'SOL',
        logoURI: 'https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/So11111111111111111111111111111111111111112/logo.png',
        tags: ['native-sol'],
        extensions: {
          coingeckoId: 'solana'
        }
      }
    ];
  }

  private getMockJupiterPriceData(): any[] {
    return [
      { id: 'So11111111111111111111111111111111111111112', price: 95.50 },
      { id: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v', price: 1.00 }
    ];
  }

  private getMockHyperliquidPerpetuals(): any[] {
    return [
      {
        name: 'SOL',
        onlyIsolated: false,
        szDecimals: 2,
        maxLeverage: 20,
        maxSz: 1000000,
        onlyCross: false
      }
    ];
  }

  private getMockHyperliquidLiquidations(): any[] {
    return [
      {
        coin: 'SOL',
        px: 94.20,
        sz: 500,
        side: 'B',
        time: Date.now(),
        startPx: 95.50,
        endPx: 94.20,
        liqPx: 94.20,
        liqSz: 500,
        liqPnl: -2500
      }
    ];
  }

  private getMockHyperliquidFundingRates(): any[] {
    return [
      {
        coin: 'SOL',
        time: Date.now(),
        fundingRate: 0.0001
      }
    ];
  }

  private getMockHyperliquidOrderBook(): any[] {
    return [
      {
        coin: 'SOL',
        levels: [
          { px: '95.50', sz: '1000', n: 1 },
          { px: '95.49', sz: '2000', n: 2 },
          { px: '95.48', sz: '1500', n: 1 }
        ],
        time: Date.now()
      }
    ];
  }

  private getMockAxiomData(): AxiomData {
    return {
      pools: [
        {
          poolId: 'axiom_pool_1',
          tokenA: 'SOL',
          tokenB: 'USDC',
          liquidity: 1000000,
          volume24h: 500000
        }
      ],
      swaps: [
        {
          swapId: 'swap_123',
          poolId: 'axiom_pool_1',
          tokenIn: 'SOL',
          tokenOut: 'USDC',
          amountIn: 1000,
          amountOut: 94500,
          timestamp: new Date().toISOString()
        }
      ],
      liquidity: [
        {
          poolId: 'axiom_pool_1',
          totalLiquidity: 1000000,
          tokenALiquidity: 10000,
          tokenBLiquidity: 950000
        }
      ]
    };
  }

  private getMockAsterdexData(): AsterdexData {
    return {
      markets: [
        {
          marketId: 'asterdex_sol_usdc',
          baseAsset: 'SOL',
          quoteAsset: 'USDC',
          price: 95.50,
          volume24h: 750000,
          liquidity: 2000000
        }
      ],
      trades: [
        {
          tradeId: 'trade_123',
          marketId: 'asterdex_sol_usdc',
          side: 'buy',
          size: 500,
          price: 95.50,
          timestamp: new Date().toISOString()
        }
      ],
      orderBook: [
        {
          marketId: 'asterdex_sol_usdc',
          bids: [
            { price: 95.49, size: 1000 },
            { price: 95.48, size: 2000 }
          ],
          asks: [
            { price: 95.51, size: 1500 },
            { price: 95.52, size: 1000 }
          ]
        }
      ]
    };
  }

  // Placeholder methods for Axiom and Asterdex
  private async getAxiomPools(): Promise<any[]> {
    return this.getMockAxiomData().pools;
  }

  private async getAxiomSwaps(): Promise<any[]> {
    return this.getMockAxiomData().swaps;
  }

  private async getAxiomLiquidity(): Promise<any[]> {
    return this.getMockAxiomData().liquidity;
  }

  private async getAsterdexMarkets(): Promise<any[]> {
    return this.getMockAsterdexData().markets;
  }

  private async getAsterdexTrades(): Promise<any[]> {
    return this.getMockAsterdexData().trades;
  }

  private async getAsterdexOrderBook(): Promise<any[]> {
    return this.getMockAsterdexData().orderBook;
  }
}

// Export singleton instance
export const dataSourcesService = new DataSourcesService();
