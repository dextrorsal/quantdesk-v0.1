import { Connection, PublicKey } from '@solana/web3.js';
import { Logger } from '../utils/logger';

interface RPCProvider {
  name: string;
  url: string;
  priority: number;
  weight: number;
  isHealthy: boolean;
  lastUsed: number;
  requestCount: number;
  errorCount: number;
  avgResponseTime: number;
}

interface RPCConfig {
  providers: RPCProvider[];
  maxRetries: number;
  healthCheckInterval: number;
  circuitBreakerThreshold: number;
  rateLimitBuffer: number;
}

export class RPCLoadBalancer {
  private static instance: RPCLoadBalancer;
  private config: RPCConfig;
  private logger: Logger;
  private healthCheckTimer: NodeJS.Timeout | null = null;
  private currentProviderIndex: number = 0;

  private constructor() {
    this.logger = new Logger();
    this.config = this.initializeConfig();
    this.startHealthChecks();
  }

  public static getInstance(): RPCLoadBalancer {
    if (!RPCLoadBalancer.instance) {
      RPCLoadBalancer.instance = new RPCLoadBalancer();
    }
    return RPCLoadBalancer.instance;
  }

  private initializeConfig(): RPCConfig {
    return {
      providers: [
        // Primary providers (high priority) - Your 4 providers
        {
          name: 'Helius',
          url: process.env['HELIUS_RPC_URL'] || 'https://devnet.helius-rpc.com',
          priority: 1,
          weight: 3,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        {
          name: 'QuickNode',
          url: process.env['QUICKNODE_RPC_URL'] || 'https://solana-devnet.g.alchemy.com/v2/demo',
          priority: 1,
          weight: 3,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        {
          name: 'Alchemy',
          url: process.env['ALCHEMY_RPC_URL'] || 'https://solana-devnet.g.alchemy.com/v2/demo',
          priority: 1,
          weight: 2,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        {
          name: 'Helius-1',
          url: process.env['HELIUS_RPC_1_URL'] || '',
          priority: 1,
          weight: 3,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        {
          name: 'QuickNode-1',
          url: process.env['QUICKNODE_1_RPC_URL'] || '',
          priority: 1,
          weight: 3,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        {
          name: 'Alchemy-1',
          url: process.env['ALCHEMY_1_RPC_URL'] || '',
          priority: 1,
          weight: 2,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        {
          name: 'Syndica-1',
          url: process.env['SYNDICA_1_RPC_URL'] || '',
          priority: 1,
          weight: 2,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        {
          name: 'Chainstack-1',
          url: process.env['CHAINSTACK_1_RPC_URL'] || '',
          priority: 1,
          weight: 2,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        // Secondary providers (medium priority) - Business partner providers
        {
          name: 'Helius-2',
          url: process.env['HELIUS_RPC_2_URL'] || '',
          priority: 2,
          weight: 2,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        {
          name: 'QuickNode-2',
          url: process.env['QUICKNODE_2_RPC_URL'] || '',
          priority: 2,
          weight: 2,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        {
          name: 'Alchemy-2',
          url: process.env['ALCHEMY_2_RPC_URL'] || '',
          priority: 2,
          weight: 2,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        {
          name: 'Syndica-2',
          url: process.env['SYNDICA_2_RPC_URL'] || '',
          priority: 2,
          weight: 2,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        {
          name: 'Chainstack-2',
          url: process.env['CHAINSTACK_2_RPC_URL'] || '',
          priority: 2,
          weight: 2,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        },
        // Fallback providers (low priority)
        {
          name: 'Solana Foundation',
          url: 'https://api.devnet.solana.com',
          priority: 3,
          weight: 1,
          isHealthy: true,
          lastUsed: 0,
          requestCount: 0,
          errorCount: 0,
          avgResponseTime: 0
        }
      ],
      maxRetries: 3,
      healthCheckInterval: 30000, // 30 seconds
      circuitBreakerThreshold: 5, // Mark unhealthy after 5 consecutive errors
      rateLimitBuffer: 0.8 // Use 80% of rate limit to avoid hitting limits
    };
  }

  public getConnection(): Connection {
    const provider = this.selectProvider();
    return new Connection(provider.url, 'confirmed');
  }

  public async executeWithRetry<T>(
    operation: (connection: Connection) => Promise<T>,
    maxRetries: number = this.config.maxRetries
  ): Promise<T> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      const provider = this.selectProvider();
      const connection = new Connection(provider.url, 'confirmed');

      try {
        const startTime = Date.now();
        const result = await operation(connection);
        const responseTime = Date.now() - startTime;

        // Update provider stats
        this.updateProviderStats(provider, true, responseTime);
        
        this.logger.info(`RPC request successful via ${provider.name} (${responseTime}ms)`);
        return result;

      } catch (error: any) {
        lastError = error;
        this.updateProviderStats(provider, false, 0);

        // Check if it's a rate limit error
        if (this.isRateLimitError(error)) {
          this.logger.warn(`Rate limit hit on ${provider.name}, switching provider`);
          this.markProviderUnhealthy(provider);
        } else if (this.isNetworkError(error)) {
          this.logger.warn(`Network error on ${provider.name}: ${error.message}`);
        }

        // If this is the last attempt, throw the error
        if (attempt === maxRetries) {
          break;
        }

        // Wait before retry (exponential backoff)
        await this.delay(Math.pow(2, attempt) * 1000);
      }
    }

    throw lastError || new Error('All RPC providers failed');
  }

  private selectProvider(): RPCProvider {
    // Filter healthy providers
    const healthyProviders = this.config.providers
      .filter(p => p.isHealthy)
      .sort((a, b) => a.priority - b.priority);

    if (healthyProviders.length === 0) {
      // If no healthy providers, reset all and use the first one
      this.logger.warn('No healthy RPC providers, resetting health status');
      this.config.providers.forEach(p => p.isHealthy = true);
      return this.config.providers[0];
    }

    // Weighted round-robin selection within priority groups
    const priorityGroups = this.groupByPriority(healthyProviders);
    
    for (const [priority, providers] of priorityGroups) {
      const selectedProvider = this.selectFromGroup(providers);
      if (selectedProvider) {
        return selectedProvider;
      }
    }

    // Fallback to first healthy provider
    return healthyProviders[0];
  }

  private groupByPriority(providers: RPCProvider[]): Map<number, RPCProvider[]> {
    const groups = new Map<number, RPCProvider[]>();
    
    providers.forEach(provider => {
      if (!groups.has(provider.priority)) {
        groups.set(provider.priority, []);
      }
      groups.get(provider.priority)!.push(provider);
    });

    return groups;
  }

  private selectFromGroup(providers: RPCProvider[]): RPCProvider | null {
    if (providers.length === 0) return null;

    // Calculate total weight
    const totalWeight = providers.reduce((sum, p) => sum + p.weight, 0);
    
    // Weighted selection based on request count and weight
    let currentWeight = 0;
    const random = Math.random() * totalWeight;

    for (const provider of providers) {
      // Adjust weight based on recent usage (less used = higher effective weight)
      const usageFactor = Math.max(0.1, 1 - (provider.requestCount / 1000));
      const effectiveWeight = provider.weight * usageFactor;
      
      currentWeight += effectiveWeight;
      
      if (random <= currentWeight) {
        return provider;
      }
    }

    return providers[0];
  }

  private updateProviderStats(provider: RPCProvider, success: boolean, responseTime: number): void {
    provider.lastUsed = Date.now();
    provider.requestCount++;

    if (success) {
      // Update average response time
      provider.avgResponseTime = (provider.avgResponseTime + responseTime) / 2;
      
      // Reset error count on success
      if (provider.errorCount > 0) {
        provider.errorCount = Math.max(0, provider.errorCount - 1);
      }
    } else {
      provider.errorCount++;
      
      // Mark unhealthy if too many errors
      if (provider.errorCount >= this.config.circuitBreakerThreshold) {
        this.markProviderUnhealthy(provider);
      }
    }
  }

  private markProviderUnhealthy(provider: RPCProvider): void {
    provider.isHealthy = false;
    this.logger.warn(`Marked ${provider.name} as unhealthy (${provider.errorCount} errors)`);
  }

  private isRateLimitError(error: any): boolean {
    const message = error.message?.toLowerCase() || '';
    return message.includes('rate limit') || 
           message.includes('429') || 
           message.includes('too many requests') ||
           error.code === 429;
  }

  private isNetworkError(error: any): boolean {
    const message = error.message?.toLowerCase() || '';
    return message.includes('network') || 
           message.includes('timeout') || 
           message.includes('connection') ||
           message.includes('econnreset') ||
           message.includes('enotfound');
  }

  private startHealthChecks(): void {
    this.healthCheckTimer = setInterval(() => {
      this.performHealthChecks();
    }, this.config.healthCheckInterval);
  }

  private async performHealthChecks(): Promise<void> {
    const healthCheckPromises = this.config.providers.map(provider => 
      this.checkProviderHealth(provider)
    );

    await Promise.allSettled(healthCheckPromises);
  }

  private async checkProviderHealth(provider: RPCProvider): Promise<void> {
    try {
      const connection = new Connection(provider.url, 'confirmed');
      const startTime = Date.now();
      
      // Simple health check - get latest blockhash
      await connection.getLatestBlockhash();
      
      const responseTime = Date.now() - startTime;
      
      // If provider was unhealthy and now responding, mark as healthy
      if (!provider.isHealthy) {
        provider.isHealthy = true;
        provider.errorCount = 0;
        this.logger.info(`Provider ${provider.name} recovered (${responseTime}ms)`);
      }
      
      provider.avgResponseTime = (provider.avgResponseTime + responseTime) / 2;
      
    } catch (error) {
      this.logger.warn(`Health check failed for ${provider.name}: ${error}`);
      provider.errorCount++;
      
      if (provider.errorCount >= this.config.circuitBreakerThreshold) {
        provider.isHealthy = false;
      }
    }
  }

  public getProviderStats(): any {
    return {
      providers: this.config.providers.map(p => ({
        name: p.name,
        isHealthy: p.isHealthy,
        requestCount: p.requestCount,
        errorCount: p.errorCount,
        avgResponseTime: p.avgResponseTime,
        lastUsed: p.lastUsed
      })),
      totalRequests: this.config.providers.reduce((sum, p) => sum + p.requestCount, 0),
      healthyProviders: this.config.providers.filter(p => p.isHealthy).length
    };
  }

  public async getAccountInfo(publicKey: PublicKey): Promise<any> {
    return this.executeWithRetry(async (connection) => {
      return await connection.getAccountInfo(publicKey);
    });
  }

  public async getBalance(publicKey: PublicKey): Promise<number> {
    return this.executeWithRetry(async (connection) => {
      return await connection.getBalance(publicKey);
    });
  }

  public async getLatestBlockhash(): Promise<any> {
    return this.executeWithRetry(async (connection) => {
      return await connection.getLatestBlockhash();
    });
  }

  public async sendTransaction(transaction: any): Promise<string> {
    return this.executeWithRetry(async (connection) => {
      return await connection.sendRawTransaction(transaction);
    });
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  public destroy(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = null;
    }
  }
}
