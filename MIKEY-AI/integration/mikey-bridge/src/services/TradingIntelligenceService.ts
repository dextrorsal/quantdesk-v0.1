/**
 * Trading Intelligence Service
 * 
 * Combines MIKEY-AI intelligence with QuantDesk ML models
 * Provides unified trading intelligence and analysis
 */

import { QuantDeskClient } from '../clients/QuantDeskClient';
import { MikeyAIClient } from '../clients/MikeyAIClient';
import { logger } from '../utils/logger';

export interface UnifiedAnalysisRequest {
  symbol: string;
  analysisType: 'comprehensive' | 'ml-focused' | 'market-focused' | 'arbitrage-focused';
  includeMLPredictions?: boolean;
  includeArbitrage?: boolean;
  includeWhaleTracking?: boolean;
  includeSentiment?: boolean;
  includeTechnicalAnalysis?: boolean;
}

export interface UnifiedAnalysisResponse {
  symbol: string;
  timestamp: Date;
  analysis: {
    mlPredictions?: {
      lorentzian: any;
      logistic: any;
      chandelier: any;
      consensus: any;
    };
    marketData?: {
      cex: any;
      dex: any;
      arbitrage: any;
    };
    whaleActivity?: any;
    sentiment?: any;
    technicalAnalysis?: any;
    liquidations?: any;
  };
  recommendations: {
    action: 'buy' | 'sell' | 'hold' | 'arbitrage';
    confidence: number;
    reasoning: string;
    riskLevel: 'low' | 'medium' | 'high';
  };
  metadata: {
    dataSources: string[];
    confidence: number;
    processingTime: number;
  };
}

export interface MLPredictionRequest {
  symbol: string;
  timeframe: string;
  includeMarketData?: boolean;
  includeSentiment?: boolean;
  includeWhaleActivity?: boolean;
}

export interface MLPredictionResponse {
  symbol: string;
  timeframe: string;
  predictions: {
    lorentzian: {
      prediction: number;
      confidence: number;
      backtestMetrics: any;
    };
    logistic: {
      prediction: number;
      confidence: number;
      backtestMetrics: any;
    };
    chandelier: {
      prediction: number;
      confidence: number;
      backtestMetrics: any;
    };
  };
  consensus: {
    prediction: number;
    confidence: number;
    agreement: number;
  };
  marketContext: {
    sentiment?: any;
    whaleActivity?: any;
    arbitrageOpportunities?: any;
  };
  timestamp: Date;
}

export interface NaturalLanguageQueryRequest {
  query: string;
  context?: any;
  includeMLData?: boolean;
  includeMarketData?: boolean;
  includeArbitrageData?: boolean;
}

export interface NaturalLanguageQueryResponse {
  response: string;
  sources: string[];
  confidence: number;
  data: {
    mlInsights?: any;
    marketInsights?: any;
    arbitrageInsights?: any;
    whaleInsights?: any;
  };
  timestamp: Date;
}

export class TradingIntelligenceService {
  constructor(
    private quantDeskClient: QuantDeskClient,
    private mikeyAIClient: MikeyAIClient
  ) {}

  /**
   * Comprehensive market analysis combining both systems
   */
  public async analyzeMarket(request: UnifiedAnalysisRequest): Promise<UnifiedAnalysisResponse> {
    const startTime = Date.now();
    
    try {
      logger.info(`Starting comprehensive analysis for ${request.symbol}`);

      // Parallel data collection
      const [
        mlPredictions,
        marketData,
        whaleActivity,
        sentiment,
        technicalAnalysis,
        liquidations
      ] = await Promise.allSettled([
        request.includeMLPredictions ? this.getMLPredictions(request.symbol) : null,
        this.getMarketData(request.symbol),
        request.includeWhaleTracking ? this.getWhaleActivity(request.symbol) : null,
        request.includeSentiment ? this.getSentimentAnalysis(request.symbol) : null,
        request.includeTechnicalAnalysis ? this.getTechnicalAnalysis(request.symbol) : null,
        this.getLiquidationData(request.symbol)
      ]);

      // Process results
      const analysis = {
        mlPredictions: mlPredictions.status === 'fulfilled' ? mlPredictions.value : null,
        marketData: marketData.status === 'fulfilled' ? marketData.value : null,
        whaleActivity: whaleActivity.status === 'fulfilled' ? whaleActivity.value : null,
        sentiment: sentiment.status === 'fulfilled' ? sentiment.value : null,
        technicalAnalysis: technicalAnalysis.status === 'fulfilled' ? technicalAnalysis.value : null,
        liquidations: liquidations.status === 'fulfilled' ? liquidations.value : null
      };

      // Generate recommendations
      const recommendations = this.generateRecommendations(analysis);

      const processingTime = Date.now() - startTime;

      return {
        symbol: request.symbol,
        timestamp: new Date(),
        analysis,
        recommendations,
        metadata: {
          dataSources: this.extractDataSources(analysis),
          confidence: this.calculateOverallConfidence(analysis),
          processingTime
        }
      };
    } catch (error) {
      logger.error('Market analysis failed:', error);
      throw error;
    }
  }

  /**
   * Get ML predictions with market context
   */
  public async getMLPrediction(request: MLPredictionRequest): Promise<MLPredictionResponse> {
    try {
      logger.info(`Getting ML prediction for ${request.symbol}`);

      // Get ML predictions from QuantDesk
      const [lorentzianPred, logisticPred, chandelierPred] = await Promise.allSettled([
        this.quantDeskClient.getMLPrediction({
          symbol: request.symbol,
          timeframe: request.timeframe,
          modelType: 'lorentzian'
        }),
        this.quantDeskClient.getMLPrediction({
          symbol: request.symbol,
          timeframe: request.timeframe,
          modelType: 'logistic'
        }),
        this.quantDeskClient.getMLPrediction({
          symbol: request.symbol,
          timeframe: request.timeframe,
          modelType: 'chandelier'
        })
      ]);

      // Get market context from MIKEY-AI
      const marketContext = await this.getMarketContext(request);

      // Calculate consensus
      const predictions = {
        lorentzian: lorentzianPred.status === 'fulfilled' ? lorentzianPred.value : null,
        logistic: logisticPred.status === 'fulfilled' ? logisticPred.value : null,
        chandelier: chandelierPred.status === 'fulfilled' ? chandelierPred.value : null
      };

      const consensus = this.calculateConsensus(predictions);

      return {
        symbol: request.symbol,
        timeframe: request.timeframe,
        predictions,
        consensus,
        marketContext,
        timestamp: new Date()
      };
    } catch (error) {
      logger.error('ML prediction failed:', error);
      throw error;
    }
  }

  /**
   * Process natural language queries with both systems
   */
  public async processNaturalLanguageQuery(request: NaturalLanguageQueryRequest): Promise<NaturalLanguageQueryResponse> {
    try {
      logger.info(`Processing natural language query: ${request.query}`);

      // Get AI response from MIKEY-AI
      const aiResponse = await this.mikeyAIClient.processQuery({
        query: request.query,
        context: request.context,
        includeMarketData: request.includeMarketData,
        includeWhaleData: true,
        includeArbitrageData: request.includeArbitrageData
      });

      // Enhance with ML data if requested
      let mlInsights = null;
      if (request.includeMLData) {
        mlInsights = await this.extractMLInsights(request.query);
      }

      return {
        response: aiResponse.response,
        sources: aiResponse.sources,
        confidence: aiResponse.confidence,
        data: {
          mlInsights,
          marketInsights: aiResponse.data,
          arbitrageInsights: request.includeArbitrageData ? await this.getArbitrageInsights(request.query) : null,
          whaleInsights: await this.getWhaleInsights(request.query)
        },
        timestamp: new Date()
      };
    } catch (error) {
      logger.error('Natural language query processing failed:', error);
      throw error;
    }
  }

  /**
   * Track whales with ML impact analysis
   */
  public async trackWhales(request: { symbol: string; threshold: number; includeMLImpact?: boolean; includeArbitrageImpact?: boolean }): Promise<any> {
    try {
      logger.info(`Tracking whales for ${request.symbol}`);

      // Get whale data from MIKEY-AI
      const whaleData = await this.mikeyAIClient.trackWhales({
        symbol: request.symbol,
        threshold: request.threshold
      });

      // Enhance with ML impact analysis
      let mlImpact = null;
      if (request.includeMLImpact) {
        mlImpact = await this.analyzeWhaleMLImpact(whaleData);
      }

      // Enhance with arbitrage impact
      let arbitrageImpact = null;
      if (request.includeArbitrageImpact) {
        arbitrageImpact = await this.analyzeWhaleArbitrageImpact(whaleData);
      }

      return {
        ...whaleData,
        mlImpact,
        arbitrageImpact
      };
    } catch (error) {
      logger.error('Whale tracking failed:', error);
      throw error;
    }
  }

  /**
   * Create real-time data stream
   */
  public createDataStream(symbol: string): any {
    // Implementation for real-time data streaming
    // This would use WebSockets or Server-Sent Events
    return {
      on: (event: string, callback: Function) => {
        // Stream implementation
      },
      destroy: () => {
        // Cleanup
      }
    };
  }

  // Private helper methods
  private async getMLPredictions(symbol: string): Promise<any> {
    const [lorentzian, logistic, chandelier] = await Promise.allSettled([
      this.quantDeskClient.getMLPrediction({ symbol, timeframe: '15m', modelType: 'lorentzian' }),
      this.quantDeskClient.getMLPrediction({ symbol, timeframe: '15m', modelType: 'logistic' }),
      this.quantDeskClient.getMLPrediction({ symbol, timeframe: '15m', modelType: 'chandelier' })
    ]);

    return {
      lorentzian: lorentzian.status === 'fulfilled' ? lorentzian.value : null,
      logistic: logistic.status === 'fulfilled' ? logistic.value : null,
      chandelier: chandelier.status === 'fulfilled' ? chandelier.value : null
    };
  }

  private async getMarketData(symbol: string): Promise<any> {
    const [cexData, arbitrageData] = await Promise.allSettled([
      this.mikeyAIClient.getMarketData({ symbol }),
      this.mikeyAIClient.detectArbitrage({ symbol })
    ]);

    return {
      cex: cexData.status === 'fulfilled' ? cexData.value : null,
      arbitrage: arbitrageData.status === 'fulfilled' ? arbitrageData.value : null
    };
  }

  private async getWhaleActivity(symbol: string): Promise<any> {
    return await this.mikeyAIClient.trackWhales({ symbol, threshold: 100000 });
  }

  private async getSentimentAnalysis(symbol: string): Promise<any> {
    return await this.mikeyAIClient.getSentimentAnalysis(symbol);
  }

  private async getTechnicalAnalysis(symbol: string): Promise<any> {
    return await this.mikeyAIClient.getTechnicalAnalysis(symbol);
  }

  private async getLiquidationData(symbol: string): Promise<any> {
    return await this.mikeyAIClient.getLiquidations({ symbol });
  }

  private async getMarketContext(request: MLPredictionRequest): Promise<any> {
    const context: any = {};

    if (request.includeMarketData) {
      context.marketData = await this.mikeyAIClient.getMarketData({ symbol: request.symbol });
    }

    if (request.includeSentiment) {
      context.sentiment = await this.mikeyAIClient.getSentimentAnalysis(request.symbol);
    }

    if (request.includeWhaleActivity) {
      context.whaleActivity = await this.mikeyAIClient.trackWhales({ symbol: request.symbol });
    }

    return context;
  }

  private calculateConsensus(predictions: any): any {
    const validPredictions = Object.values(predictions).filter(p => p !== null);
    
    if (validPredictions.length === 0) {
      return { prediction: 0, confidence: 0, agreement: 0 };
    }

    const avgPrediction = validPredictions.reduce((sum: number, p: any) => sum + p.prediction, 0) / validPredictions.length;
    const avgConfidence = validPredictions.reduce((sum: number, p: any) => sum + p.confidence, 0) / validPredictions.length;
    
    // Calculate agreement (how close predictions are)
    const variance = validPredictions.reduce((sum: number, p: any) => sum + Math.pow(p.prediction - avgPrediction, 2), 0) / validPredictions.length;
    const agreement = Math.max(0, 1 - variance);

    return {
      prediction: avgPrediction,
      confidence: avgConfidence,
      agreement
    };
  }

  private generateRecommendations(analysis: any): any {
    // Simple recommendation logic - in production, this would be more sophisticated
    let action: 'buy' | 'sell' | 'hold' | 'arbitrage' = 'hold';
    let confidence = 0.5;
    let reasoning = 'Insufficient data for recommendation';
    let riskLevel: 'low' | 'medium' | 'high' = 'medium';

    // Check for arbitrage opportunities
    if (analysis.marketData?.arbitrage?.opportunities?.length > 0) {
      action = 'arbitrage';
      confidence = 0.8;
      reasoning = 'Arbitrage opportunities detected';
      riskLevel = 'low';
    }
    // Check ML predictions
    else if (analysis.mlPredictions?.consensus) {
      const consensus = analysis.mlPredictions.consensus;
      if (consensus.prediction > 0.6 && consensus.confidence > 0.7) {
        action = 'buy';
        confidence = consensus.confidence;
        reasoning = 'Strong ML consensus for bullish signal';
        riskLevel = consensus.agreement > 0.8 ? 'low' : 'medium';
      } else if (consensus.prediction < 0.4 && consensus.confidence > 0.7) {
        action = 'sell';
        confidence = consensus.confidence;
        reasoning = 'Strong ML consensus for bearish signal';
        riskLevel = consensus.agreement > 0.8 ? 'low' : 'medium';
      }
    }

    return {
      action,
      confidence,
      reasoning,
      riskLevel
    };
  }

  private extractDataSources(analysis: any): string[] {
    const sources: string[] = [];
    
    if (analysis.mlPredictions) sources.push('QuantDesk ML Models');
    if (analysis.marketData) sources.push('MIKEY-AI Market Data');
    if (analysis.whaleActivity) sources.push('MIKEY-AI Whale Tracking');
    if (analysis.sentiment) sources.push('MIKEY-AI Sentiment Analysis');
    if (analysis.technicalAnalysis) sources.push('MIKEY-AI Technical Analysis');
    if (analysis.liquidations) sources.push('MIKEY-AI Liquidation Data');

    return sources;
  }

  private calculateOverallConfidence(analysis: any): number {
    const confidences: number[] = [];
    
    if (analysis.mlPredictions?.consensus?.confidence) {
      confidences.push(analysis.mlPredictions.consensus.confidence);
    }
    if (analysis.marketData?.summary) {
      confidences.push(0.8); // Market data confidence
    }
    if (analysis.sentiment?.confidence) {
      confidences.push(analysis.sentiment.confidence);
    }

    return confidences.length > 0 ? confidences.reduce((sum, conf) => sum + conf, 0) / confidences.length : 0.5;
  }

  private async extractMLInsights(query: string): Promise<any> {
    // Extract ML-related insights from the query
    // This would analyze the query and return relevant ML data
    return null;
  }

  private async getArbitrageInsights(query: string): Promise<any> {
    // Extract arbitrage-related insights
    return null;
  }

  private async getWhaleInsights(query: string): Promise<any> {
    // Extract whale-related insights
    return null;
  }

  private async analyzeWhaleMLImpact(whaleData: any): Promise<any> {
    // Analyze how whale activity might impact ML predictions
    return null;
  }

  private async analyzeWhaleArbitrageImpact(whaleData: any): Promise<any> {
    // Analyze how whale activity might impact arbitrage opportunities
    return null;
  }
}
