/**
 * Tokenization Configuration
 * Provider-specific tokenization settings and encodings
 */

import { TokenizationProvider } from '../types/token-estimation';

export class TokenizationConfig {
  private static instance: TokenizationConfig;
  private providers: Map<string, TokenizationProvider> = new Map();

  private constructor() {
    this.initializeProviders();
  }

  public static getInstance(): TokenizationConfig {
    if (!TokenizationConfig.instance) {
      TokenizationConfig.instance = new TokenizationConfig();
    }
    return TokenizationConfig.instance;
  }

  private initializeProviders(): void {
    // OpenAI Provider Configuration
    this.providers.set('openai', {
      name: 'openai',
      encoding: 'cl100k_base',
      models: ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'],
      costPerToken: 0.0005
    });

    // Google Gemini Provider Configuration
    this.providers.set('google', {
      name: 'google',
      encoding: 'gemini',
      models: ['gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash'],
      costPerToken: 0.0001
    });

    // Mistral Provider Configuration
    this.providers.set('mistral', {
      name: 'mistral',
      encoding: 'mistral',
      models: ['mistral-large', 'mistral-medium', 'mistral-small'],
      costPerToken: 0.0001
    });

    // Cohere Provider Configuration
    this.providers.set('cohere', {
      name: 'cohere',
      encoding: 'cohere',
      models: ['command-a-03-2025', 'command-r-plus', 'command-r'],
      costPerToken: 0.0001
    });

    // HuggingFace Provider Configuration
    this.providers.set('huggingface', {
      name: 'huggingface',
      encoding: 'huggingface',
      models: ['meta-llama/Llama-2-70b-chat-hf', 'microsoft/DialoGPT-medium'],
      costPerToken: 0.00005
    });

    // XAI Provider Configuration
    this.providers.set('xai', {
      name: 'xai',
      encoding: 'xai',
      models: ['grok-beta', 'grok-2'],
      costPerToken: 0.0002
    });
  }

  public getProvider(providerName: string): TokenizationProvider | undefined {
    return this.providers.get(providerName.toLowerCase());
  }

  public getAllProviders(): TokenizationProvider[] {
    return Array.from(this.providers.values());
  }

  public getProviderNames(): string[] {
    return Array.from(this.providers.keys());
  }

  public isValidProvider(providerName: string): boolean {
    return this.providers.has(providerName.toLowerCase());
  }

  public isValidModel(providerName: string, modelName: string): boolean {
    const provider = this.getProvider(providerName);
    return provider ? provider.models.includes(modelName) : false;
  }

  public getDefaultModel(providerName: string): string | undefined {
    const provider = this.getProvider(providerName);
    return provider ? provider.models[0] : undefined;
  }

  public getCostPerToken(providerName: string): number {
    const provider = this.getProvider(providerName);
    return provider ? provider.costPerToken : 0.001; // Default fallback
  }

  public updateProviderCost(providerName: string, newCost: number): void {
    const provider = this.getProvider(providerName);
    if (provider) {
      provider.costPerToken = newCost;
    }
  }

  public addCustomProvider(provider: TokenizationProvider): void {
    this.providers.set(provider.name.toLowerCase(), provider);
  }

  public removeProvider(providerName: string): boolean {
    return this.providers.delete(providerName.toLowerCase());
  }

  public validateConfiguration(): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    this.providers.forEach((provider, name) => {
      if (!provider.encoding) {
        errors.push(`Provider ${name} missing encoding`);
      }
      if (!provider.models || provider.models.length === 0) {
        errors.push(`Provider ${name} missing models`);
      }
      if (provider.costPerToken <= 0) {
        errors.push(`Provider ${name} has invalid cost per token: ${provider.costPerToken}`);
      }
    });

    return {
      isValid: errors.length === 0,
      errors
    };
  }
}
