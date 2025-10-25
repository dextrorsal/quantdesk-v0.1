// QuantDesk API Client Example
// Realistic API client implementation with proper error handling and security

import { QuantDeskClient, MarketData, Position, Order, Portfolio } from '@quantdesk/sdk';

/**
 * API Client Example - Realistic Implementation
 * This example demonstrates practical API client usage with proper error handling and security
 */
export class QuantDeskAPIClient {
  private client: QuantDeskClient;
  private isInitialized: boolean = false;

  constructor(client: QuantDeskClient) {
    this.client = client;
  }

  /**
   * Initialize the API client
   */
  async initialize(): Promise<void> {
    try {
      await this.client.initialize();
      this.isInitialized = true;
      console.log('✅ API Client initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize API client:', error);
      throw new Error(`API Client initialization failed: ${error.message}`);
    }
  }

  /**
   * Get available markets with error handling
   */
  async getMarkets(): Promise<any[]> {
    this.ensureInitialized();
    
    try {
      const markets = await this.client.getMarkets();
      console.log(`📈 Retrieved ${markets.length} markets`);
      return markets;
    } catch (error) {
      console.error('❌ Failed to get markets:', error);
      throw new Error(`Failed to get markets: ${error.message}`);
    }
  }

  /**
   * Get market data with validation
   */
  async getMarketData(market: string): Promise<MarketData> {
    this.ensureInitialized();
    this.validateMarketSymbol(market);
    
    try {
      const marketData = await this.client.getMarketData(market);
      console.log(`📊 Retrieved market data for ${market}`);
      return marketData;
    } catch (error) {
      console.error(`❌ Failed to get market data for ${market}:`, error);
      throw new Error(`Failed to get market data for ${market}: ${error.message}`);
    }
  }

  /**
   * Get portfolio with error handling
   */
  async getPortfolio(): Promise<Portfolio> {
    this.ensureInitialized();
    
    try {
      const portfolio = await this.client.getPortfolio();
      console.log(`💼 Retrieved portfolio data`);
      return portfolio;
    } catch (error) {
      console.error('❌ Failed to get portfolio:', error);
      throw new Error(`Failed to get portfolio: ${error.message}`);
    }
  }

  /**
   * Get positions with error handling
   */
  async getPositions(): Promise<Position[]> {
    this.ensureInitialized();
    
    try {
      const positions = await this.client.getPositions();
      console.log(`📋 Retrieved ${positions.length} positions`);
      return positions;
    } catch (error) {
      console.error('❌ Failed to get positions:', error);
      throw new Error(`Failed to get positions: ${error.message}`);
    }
  }

  /**
   * Place order with validation and error handling
   */
  async placeOrder(orderData: {
    market: string;
    side: 'buy' | 'sell';
    size: number;
    price: number;
    orderType: 'limit' | 'market';
  }): Promise<Order> {
    this.ensureInitialized();
    this.validateOrderData(orderData);
    
    try {
      console.log(`📝 Placing ${orderData.side} order for ${orderData.size} ${orderData.market} at $${orderData.price}`);
      
      const order = await this.client.placeOrder(orderData);
      console.log(`✅ Order placed successfully: ${order.id}`);
      return order;
    } catch (error) {
      console.error('❌ Failed to place order:', error);
      throw new Error(`Failed to place order: ${error.message}`);
    }
  }

  /**
   * Get order status with error handling
   */
  async getOrder(orderId: string): Promise<Order> {
    this.ensureInitialized();
    this.validateOrderId(orderId);
    
    try {
      const order = await this.client.getOrder(orderId);
      console.log(`📋 Retrieved order ${orderId}: ${order.status}`);
      return order;
    } catch (error) {
      console.error(`❌ Failed to get order ${orderId}:`, error);
      throw new Error(`Failed to get order ${orderId}: ${error.message}`);
    }
  }

  /**
   * Cancel order with error handling
   */
  async cancelOrder(orderId: string): Promise<void> {
    this.ensureInitialized();
    this.validateOrderId(orderId);
    
    try {
      await this.client.cancelOrder(orderId);
      console.log(`❌ Order ${orderId} cancelled successfully`);
    } catch (error) {
      console.error(`❌ Failed to cancel order ${orderId}:`, error);
      throw new Error(`Failed to cancel order ${orderId}: ${error.message}`);
    }
  }

  /**
   * Close position with error handling
   */
  async closePosition(positionId: string): Promise<void> {
    this.ensureInitialized();
    this.validatePositionId(positionId);
    
    try {
      await this.client.closePosition(positionId);
      console.log(`🔒 Position ${positionId} closed successfully`);
    } catch (error) {
      console.error(`❌ Failed to close position ${positionId}:`, error);
      throw new Error(`Failed to close position ${positionId}: ${error.message}`);
    }
  }

  /**
   * Get AI analysis with error handling
   */
  async getAIAnalysis(market: string): Promise<any> {
    this.ensureInitialized();
    this.validateMarketSymbol(market);
    
    try {
      const analysis = await this.client.getAIAnalysis(market);
      console.log(`🤖 Retrieved AI analysis for ${market}`);
      return analysis;
    } catch (error) {
      console.error(`❌ Failed to get AI analysis for ${market}:`, error);
      throw new Error(`Failed to get AI analysis for ${market}: ${error.message}`);
    }
  }

  /**
   * Get trading signals with error handling
   */
  async getTradingSignals(): Promise<any[]> {
    this.ensureInitialized();
    
    try {
      const signals = await this.client.getTradingSignals();
      console.log(`📡 Retrieved ${signals.length} trading signals`);
      return signals;
    } catch (error) {
      console.error('❌ Failed to get trading signals:', error);
      throw new Error(`Failed to get trading signals: ${error.message}`);
    }
  }

  /**
   * Get risk assessment with error handling
   */
  async getRiskAssessment(): Promise<any> {
    this.ensureInitialized();
    
    try {
      const riskAssessment = await this.client.getRiskAssessment();
      console.log(`⚠️ Retrieved risk assessment`);
      return riskAssessment;
    } catch (error) {
      console.error('❌ Failed to get risk assessment:', error);
      throw new Error(`Failed to get risk assessment: ${error.message}`);
    }
  }

  /**
   * Chat with MIKEY-AI with error handling
   */
  async chatWithMIKEY(message: string, context?: any): Promise<any> {
    this.ensureInitialized();
    this.validateMessage(message);
    
    try {
      const response = await this.client.chatWithMIKEY(message, context);
      console.log(`💬 Received response from MIKEY-AI`);
      return response;
    } catch (error) {
      console.error('❌ Failed to chat with MIKEY-AI:', error);
      throw new Error(`Failed to chat with MIKEY-AI: ${error.message}`);
    }
  }

  // Validation methods
  private ensureInitialized(): void {
    if (!this.isInitialized) {
      throw new Error('API Client not initialized. Call initialize() first.');
    }
  }

  private validateMarketSymbol(symbol: string): void {
    if (!symbol || typeof symbol !== 'string') {
      throw new Error('Market symbol must be a non-empty string');
    }
    
    if (!symbol.includes('-PERP')) {
      throw new Error('Market symbol must end with -PERP');
    }
  }

  private validateOrderData(orderData: any): void {
    if (!orderData) {
      throw new Error('Order data is required');
    }
    
    if (!orderData.market || typeof orderData.market !== 'string') {
      throw new Error('Market is required and must be a string');
    }
    
    if (!orderData.side || !['buy', 'sell'].includes(orderData.side)) {
      throw new Error('Side must be either "buy" or "sell"');
    }
    
    if (typeof orderData.size !== 'number' || orderData.size <= 0) {
      throw new Error('Size must be a positive number');
    }
    
    if (typeof orderData.price !== 'number' || orderData.price <= 0) {
      throw new Error('Price must be a positive number');
    }
    
    if (!orderData.orderType || !['limit', 'market'].includes(orderData.orderType)) {
      throw new Error('Order type must be either "limit" or "market"');
    }
  }

  private validateOrderId(orderId: string): void {
    if (!orderId || typeof orderId !== 'string') {
      throw new Error('Order ID must be a non-empty string');
    }
  }

  private validatePositionId(positionId: string): void {
    if (!positionId || typeof positionId !== 'string') {
      throw new Error('Position ID must be a non-empty string');
    }
  }

  private validateMessage(message: string): void {
    if (!message || typeof message !== 'string') {
      throw new Error('Message must be a non-empty string');
    }
    
    if (message.length > 1000) {
      throw new Error('Message must be less than 1000 characters');
    }
  }
}

// Example usage - Realistic implementation
export async function runAPIClientExample() {
  // Initialize QuantDesk client
  const client = new QuantDeskClient({
    rpcUrl: 'https://api.devnet.solana.com',
    programId: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
    wallet: wallet // Your Solana wallet instance
  });

  // Create API client wrapper
  const apiClient = new QuantDeskAPIClient(client);
  
  try {
    // Initialize API client
    await apiClient.initialize();
    
    // Get markets
    const markets = await apiClient.getMarkets();
    console.log('Available markets:', markets.map(m => m.symbol));
    
    // Get market data for SOL-PERP
    const marketData = await apiClient.getMarketData('SOL-PERP');
    console.log('SOL-PERP price:', marketData.price);
    
    // Get portfolio
    const portfolio = await apiClient.getPortfolio();
    console.log('Portfolio value:', portfolio.totalValue);
    
    // Get AI analysis
    const analysis = await apiClient.getAIAnalysis('SOL-PERP');
    console.log('AI sentiment:', analysis.sentiment);
    
    // Chat with MIKEY
    const chatResponse = await apiClient.chatWithMIKEY('What do you think about SOL-PERP?');
    console.log('MIKEY response:', chatResponse.response);
    
  } catch (error) {
    console.error('API Client example failed:', error);
  }
}

// Export for use in other modules
export { QuantDeskAPIClient };
