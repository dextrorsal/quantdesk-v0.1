import { QuantDeskAI } from '@quantdesk/ai-sdk';

/**
 * AI-Powered Market Sentiment Analysis Example
 * 
 * This example demonstrates how to use QuantDesk's AI capabilities
 * for market sentiment analysis without exposing AI internals.
 */

export class AISentimentAnalyzer {
  private ai: QuantDeskAI;

  constructor(apiKey: string) {
    this.ai = new QuantDeskAI({ apiKey });
  }

  /**
   * Analyze market sentiment for multiple symbols
   */
  async analyzeMarketSentiment(symbols: string[], timeframe: string = '24h') {
    try {
      const sentiment = await this.ai.getMarketSentiment({
        symbols,
        timeframe,
        includeNewsAnalysis: true,
        includeSocialSentiment: true,
        includeTechnicalAnalysis: true
      });

      return {
        overallSentiment: sentiment.overallSentiment,
        symbolSentiments: sentiment.symbolSentiments,
        newsImpact: sentiment.newsImpact,
        socialSentiment: sentiment.socialSentiment,
        technicalSentiment: sentiment.technicalSentiment,
        confidence: sentiment.confidence,
        timestamp: new Date().toISOString()
      };
    } catch (error) {
      console.error('AI Sentiment Analysis Error:', error);
      throw new Error('Failed to analyze market sentiment with AI');
    }
  }

  /**
   * Get sentiment-based trading signals
   */
  async getSentimentSignals(symbols: string[]) {
    try {
      const signals = await this.ai.getSentimentSignals({
        symbols,
        includeEntrySignals: true,
        includeExitSignals: true,
        includeRiskSignals: true
      });

      return {
        entrySignals: signals.entrySignals,
        exitSignals: signals.exitSignals,
        riskSignals: signals.riskSignals,
        signalStrength: signals.signalStrength,
        recommendedActions: signals.recommendedActions
      };
    } catch (error) {
      console.error('AI Sentiment Signals Error:', error);
      throw new Error('Failed to get sentiment signals');
    }
  }

  /**
   * Analyze news impact on market sentiment
   */
  async analyzeNewsImpact(symbols: string[], newsKeywords: string[] = []) {
    try {
      const newsAnalysis = await this.ai.analyzeNewsImpact({
        symbols,
        keywords: newsKeywords,
        timeframe: '7d',
        includeSentimentScoring: true
      });

      return {
        newsSentiment: newsAnalysis.newsSentiment,
        keywordImpact: newsAnalysis.keywordImpact,
        sentimentTrend: newsAnalysis.sentimentTrend,
        newsVolume: newsAnalysis.newsVolume,
        impactScore: newsAnalysis.impactScore
      };
    } catch (error) {
      console.error('AI News Impact Analysis Error:', error);
      throw new Error('Failed to analyze news impact');
    }
  }

  /**
   * Get social media sentiment analysis
   */
  async getSocialSentiment(symbols: string[], platforms: string[] = ['twitter', 'reddit', 'telegram']) {
    try {
      const socialSentiment = await this.ai.getSocialSentiment({
        symbols,
        platforms,
        timeframe: '24h',
        includeInfluencerAnalysis: true
      });

      return {
        platformSentiments: socialSentiment.platformSentiments,
        influencerSentiment: socialSentiment.influencerSentiment,
        socialVolume: socialSentiment.socialVolume,
        sentimentMomentum: socialSentiment.sentimentMomentum,
        trendingTopics: socialSentiment.trendingTopics
      };
    } catch (error) {
      console.error('AI Social Sentiment Error:', error);
      throw new Error('Failed to get social sentiment');
    }
  }
}

/**
 * Example usage
 */
async function sentimentExample() {
  const analyzer = new AISentimentAnalyzer(process.env.QUANTDESK_AI_KEY!);

  const symbols = ['SOL', 'BTC', 'ETH', 'USDC'];

  try {
    // Analyze market sentiment
    const sentiment = await analyzer.analyzeMarketSentiment(symbols, '24h');
    console.log('Market Sentiment:', sentiment);

    // Get sentiment-based trading signals
    const signals = await analyzer.getSentimentSignals(symbols);
    console.log('Sentiment Signals:', signals);

    // Analyze news impact
    const newsImpact = await analyzer.analyzeNewsImpact(symbols, ['DeFi', 'Solana', 'crypto']);
    console.log('News Impact:', newsImpact);

    // Get social sentiment
    const socialSentiment = await analyzer.getSocialSentiment(symbols);
    console.log('Social Sentiment:', socialSentiment);

  } catch (error) {
    console.error('Sentiment Example Error:', error);
  }
}

export default AISentimentAnalyzer;
