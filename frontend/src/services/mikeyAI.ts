// QuantDesk MIKEY-AI Service Integration
// Real API calls to MIKEY-AI service running on port 3000

export interface MikeyAIResponse {
  success: boolean;
  data: {
    response: string;
    sources: string[];
    confidence: number;
    timestamp: string;
    provider: string;
  };
  error?: string;
}

export interface LLMStatus {
  name: string;
  status: string;
  configured: boolean;
}

export interface MarketPricesResponse {
  success: boolean;
  data?: Record<string, number>;
  error?: {
    code: string;
    message: string;
  };
}

export interface MarketSentimentResponse {
  success: boolean;
  data?: {
    sentiment: string;
    confidence: number;
    sources: string[];
    timestamp: string;
  };
  error?: string;
}

class MikeyAIService {
  private baseUrl: string;

  constructor() {
    // Use backend proxy to avoid CORS and centralize auth/rate limits
    this.baseUrl = 'http://localhost:3002/api/mikey';
  }

  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      const data = await response.json();
      return data.success && data.data.status === 'healthy';
    } catch (error) {
      console.error('MIKEY-AI health check failed:', error);
      return false;
    }
  }

  // AI Query - Main functionality
  async queryAI(query: string): Promise<MikeyAIResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('MIKEY-AI query failed:', error);
      return {
        success: false,
        data: {
          response: `Error: ${error instanceof Error ? error.message : 'Unknown error'}`,
          sources: ['Error'],
          confidence: 0,
          timestamp: new Date().toISOString(),
          provider: 'Error',
        },
      };
    }
  }

  // Get LLM status
  async getLLMStatus(): Promise<LLMStatus[]> {
    try {
      const response = await fetch(`${this.baseUrl}/llm/status`);
      const data = await response.json();
      return data.success ? data.data : [];
    } catch (error) {
      console.error('Failed to get LLM status:', error);
      return [];
    }
  }

  // Get market prices
  async getMarketPrices(symbols: string[]): Promise<MarketPricesResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/market/prices?symbols=${symbols.join(',')}`);
      return await response.json();
    } catch (error) {
      console.error('Failed to get market prices:', error);
      return {
        success: false,
        error: {
          code: 'NETWORK_ERROR',
          message: error instanceof Error ? error.message : 'Unknown error',
        },
      };
    }
  }

  // Get market sentiment
  async getMarketSentiment(): Promise<MarketSentimentResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/market/sentiment`);
      return await response.json();
    } catch (error) {
      console.error('Failed to get market sentiment:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  // Get trading liquidations
  async getLiquidations(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/trading/liquidations`);
      return await response.json();
    } catch (error) {
      console.error('Failed to get liquidations:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }

  // Get whale tracking data
  async getWhaleTracking(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v1/trading/whales`);
      return await response.json();
    } catch (error) {
      console.error('Failed to get whale tracking:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
}

// Export singleton instance
export const mikeyAI = new MikeyAIService();
export default mikeyAI;
