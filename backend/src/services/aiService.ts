import axios from 'axios';
import { Logger } from '../utils/logger';

const logger = new Logger();

export class AIService {
  private readonly baseUrl: string;
  private readonly timeout: number;

  constructor() {
    this.baseUrl = process.env.MIKEY_AI_URL || 'http://localhost:3000';
    this.timeout = 30000; // 30 seconds
  }

  async queryAI(query: string, context?: any): Promise<any> {
    try {
      const response = await axios.post(
        `${this.baseUrl}/api/v1/ai/query`,
        { query, context },
        { timeout: this.timeout }
      );
      return response.data;
    } catch (error) {
      logger.error('AI query failed:', error);
      throw new Error('AI service unavailable');
    }
  }

  async getMarketAnalysis(symbol: string): Promise<any> {
    return this.queryAI(`Analyze market conditions for ${symbol}`);
  }

  async getWhaleActivity(): Promise<any> {
    return this.queryAI('Get recent whale movements and large transactions');
  }

  async getTradingInsights(marketData: any): Promise<any> {
    return this.queryAI('Provide trading insights based on market data', { marketData });
  }

  async getRiskAssessment(positionData: any): Promise<any> {
    return this.queryAI('Assess risk for trading positions', { positionData });
  }

  async healthCheck(): Promise<boolean> {
    try {
      const response = await axios.get(`${this.baseUrl}/health`, { timeout: 5000 });
      return response.status === 200;
    } catch (error) {
      return false;
    }
  }

  async getServiceStatus(): Promise<any> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/v1/status`, { timeout: 5000 });
      return response.data;
    } catch (error) {
      logger.error('Failed to get AI service status:', error);
      return {
        status: 'offline',
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
}

export const aiService = new AIService();
