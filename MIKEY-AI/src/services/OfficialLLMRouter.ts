/**
 * Official LLM Router using correct SDKs for each provider
 * Based on official documentation from OpenAI, Google, and Cohere
 * Enhanced with cost optimization capabilities
 */

import { CostOptimizationEngine, CostMetrics } from './CostOptimizationEngine';
import { ProviderCostRanking } from './ProviderCostRanking';
import { TokenEstimationService } from './TokenEstimationService';
import { EnhancedCostMetrics } from '../types/token-estimation';
import { QualityThresholdManager } from './QualityThresholdManager';
import { QualityEvaluationResult, EscalationDecision } from '../types/quality-thresholds';
import { AnalyticsCollector } from './AnalyticsCollector';
import { RequestMetrics } from '../types/analytics';
import { ProviderHealthMonitor } from './ProviderHealthMonitor';
import { IntelligentFallbackManager } from './IntelligentFallbackManager';
import { MonitoringService } from './MonitoringService';
import { systemLogger, errorLogger, securityLogger } from '../utils/logger';

export class OfficialLLMRouter {
  private readonly providers: Map<string, any> = new Map();
  private readonly providerConfigs: Map<string, any> = new Map();
  private costOptimizationEngine: CostOptimizationEngine;
  private providerRanking: ProviderCostRanking;
  private tokenEstimationService: TokenEstimationService;
  private qualityThresholdManager: QualityThresholdManager;
  private analyticsCollector: AnalyticsCollector;
  private providerHealthMonitor: ProviderHealthMonitor;
  private intelligentFallbackManager: IntelligentFallbackManager;
  private monitoringService: MonitoringService;
  private usageHistory: Array<{provider: string, tokensUsed: number, timestamp: Date, task: string}> = [];
  private sessionRetryCounts: Map<string, number> = new Map();
  private sessionStartTimes: Map<string, Date> = new Map();

  constructor() {
    this.initializeProviders();
    this.costOptimizationEngine = new CostOptimizationEngine();
    this.providerRanking = new ProviderCostRanking();
    this.tokenEstimationService = new TokenEstimationService();
    this.qualityThresholdManager = new QualityThresholdManager();
    this.analyticsCollector = new AnalyticsCollector();
    this.providerHealthMonitor = new ProviderHealthMonitor();
    this.intelligentFallbackManager = new IntelligentFallbackManager();
    this.monitoringService = new MonitoringService();
    this.setupUsageTracking();
    
    systemLogger.startup('OfficialLLMRouter', 'initialized with enhanced security and monitoring');
  }

  private initializeProviders(): void {
    // OpenAI Provider - Official SDK
    if (process.env['OPENAI_API_KEY']) {
      this.providerConfigs.set('openai', {
        apiKey: process.env['OPENAI_API_KEY'],
        model: 'gpt-4o-mini', // Use cheaper model first
        temperature: 0.7
      });
      systemLogger.startup('OfficialLLMRouter', 'OpenAI provider configured');
    }

    // Google Gemini Provider - Official SDK
    if (process.env['GOOGLE_API_KEY']) {
      this.providerConfigs.set('google', {
        apiKey: process.env['GOOGLE_API_KEY'],
        model: 'gemini-2.5-flash', // From your docs
        temperature: 0.7
      });
      systemLogger.startup('OfficialLLMRouter', 'Google Gemini provider configured');
    }

    // Cohere Provider - Official SDK
    if (process.env['COHERE_API_KEY']) {
      this.providerConfigs.set('cohere', {
        apiKey: process.env['COHERE_API_KEY'],
        model: 'command-a-03-2025', // From your docs
        temperature: 0.7
      });
      systemLogger.startup('OfficialLLMRouter', 'Cohere provider configured');
    }

    // Mistral AI Provider (keep existing LangChain approach)
    if (process.env['MISTRAL_API_KEY']) {
      this.providerConfigs.set('mistral', {
        apiKey: process.env['MISTRAL_API_KEY'],
        model: 'mistral-small-latest',
        temperature: 0.7,
        baseURL: 'https://api.mistral.ai/v1'
      });
      systemLogger.startup('OfficialLLMRouter', 'Mistral provider configured');
    }

    // XAI (Grok) Provider - OpenAI-compatible API
    if (process.env['XAI_API_KEY']) {
      this.providerConfigs.set('xai', {
        apiKey: process.env['XAI_API_KEY'],
        model: 'grok-beta', // Official model name from XAI docs
        temperature: 0.7,
        baseURL: 'https://api.x.ai/v1'
      });
      systemLogger.startup('OfficialLLMRouter', 'XAI (Grok) provider configured');
    }

    // Qwen (Alibaba) Provider - OpenAI-compatible API
    if (process.env['QWEN_API_KEY']) {
      this.providerConfigs.set('qwen', {
        apiKey: process.env['QWEN_API_KEY'],
        model: 'qwen-plus', // Qwen Plus model
        temperature: 0.7,
        baseURL: 'https://dashscope.aliyuncs.com/compatible-mode/v1'
      });
      systemLogger.startup('OfficialLLMRouter', 'Qwen (Alibaba) provider configured');
    }

    // Hugging Face Provider - Latest Qwen3-Next-80B-A3B-Instruct
    if (process.env['HUGGING_FACE_BEARER_TOKEN']) {
      this.providerConfigs.set('huggingface', {
        apiKey: process.env['HUGGING_FACE_BEARER_TOKEN'],
        model: 'Qwen/Qwen3-Next-80B-A3B-Instruct', // Latest & Powerful!
        temperature: 0.7,
        baseURL: 'https://api-inference.huggingface.co/models'
      });
      systemLogger.startup('OfficialLLMRouter', 'Hugging Face (Qwen3-Next-80B) provider configured');
    }

    systemLogger.startup('OfficialLLMRouter', `Initialized ${this.providerConfigs.size} LLM providers: ${Array.from(this.providerConfigs.keys()).join(', ')}`);
  }

  /**
   * Validate and sanitize input for security
   */
  private validateInput(prompt: string): { isValid: boolean; sanitizedPrompt: string; error?: string } {
    try {
      // Check for malicious patterns
      const maliciousPatterns = [
        /<script[^>]*>.*?<\/script>/gi,
        /javascript:/gi,
        /on\w+\s*=/gi,
        /eval\s*\(/gi,
        /function\s*\(/gi,
        /import\s*\(/gi,
        /require\s*\(/gi
      ];

      for (const pattern of maliciousPatterns) {
        if (pattern.test(prompt)) {
          securityLogger.securityViolation('Malicious pattern detected', { pattern: pattern.source, prompt: prompt.substring(0, 100) });
          return { isValid: false, sanitizedPrompt: '', error: 'Malicious content detected' };
        }
      }

      // Check input length
      if (prompt.length > 50000) {
        securityLogger.securityViolation('Input too long', { length: prompt.length });
        return { isValid: false, sanitizedPrompt: '', error: 'Input too long' };
      }

      // Basic sanitization
      const sanitizedPrompt = prompt
        .replace(/[<>]/g, '') // Remove potential HTML tags
        .trim();

      return { isValid: true, sanitizedPrompt };
    } catch (error) {
      errorLogger.aiError(error as Error, 'Input validation');
      return { isValid: false, sanitizedPrompt: '', error: 'Validation error' };
    }
  }

  /**
   * Validate API keys for security
   */
  private validateApiKeys(): void {
    const requiredKeys = ['OPENAI_API_KEY', 'GOOGLE_API_KEY', 'COHERE_API_KEY', 'MISTRAL_API_KEY', 'XAI_API_KEY', 'QWEN_API_KEY'];
    
    for (const key of requiredKeys) {
      const value = process.env[key];
      if (value) {
        // Check for placeholder values
        if (value.includes('your_') || value.includes('placeholder') || value.length < 10) {
          securityLogger.securityViolation('Invalid API key format', { key, length: value.length });
        }
        
        // Log key presence (but not the actual key)
        systemLogger.startup('OfficialLLMRouter', `API key ${key} validated`);
      }
    }
  }

  /**
   * Smart routing with quality monitoring and escalation
   */
  public async routeRequest(prompt: string, taskType: string = 'general', sessionId?: string): Promise<{response: string, provider: string}> {
    const startTime = Date.now();
    
    // Security validation
    const validation = this.validateInput(prompt);
    if (!validation.isValid) {
      securityLogger.securityViolation('Invalid input rejected', { error: validation.error, taskType, sessionId });
      throw new Error(`Security validation failed: ${validation.error}`);
    }
    
    const sanitizedPrompt = validation.sanitizedPrompt;
    
    // Smart routing: select best provider for task type
    const selectedProvider = this.selectBestProvider(taskType);
    
    if (!selectedProvider) {
      errorLogger.aiError(new Error('No available providers'), 'Provider selection');
      throw new Error('No available providers');
    }

    try {
      systemLogger.startup('OfficialLLMRouter', `Smart routing ${taskType} to ${selectedProvider.name}: ${sanitizedPrompt.substring(0, 50)}...`);
      
      const response = await this.callProvider(selectedProvider.name, sanitizedPrompt);
      systemLogger.startup('OfficialLLMRouter', `${selectedProvider.name} responded successfully`);
      
      // Track usage with accurate token estimation
      const fullContent = prompt + response;
      const tokenEstimation = await this.tokenEstimationService.estimateTokens(
        fullContent, 
        selectedProvider.name.toLowerCase(), 
        this.getProviderModel(selectedProvider.name)
      );
      this.trackUsageWithAccurateTokens(selectedProvider.name, tokenEstimation, taskType);
      
      // Quality monitoring and escalation
      let qualityResult: any = null;
      if (this.qualityThresholdManager.isQualityEvaluationEnabled()) {
        qualityResult = await this.qualityThresholdManager.evaluateQuality(
          response, 
          taskType, 
          selectedProvider.name.toLowerCase()
        );
        
        // Check if escalation is needed
        if (qualityResult.shouldEscalate) {
          const escalationDecision = await this.qualityThresholdManager.makeEscalationDecision(
            selectedProvider.name.toLowerCase(),
            qualityResult.qualityScore,
            sessionId
          );
          
          if (escalationDecision.shouldEscalate && escalationDecision.suggestedProvider !== selectedProvider.name.toLowerCase()) {
            console.log(`📈 Quality escalation: ${selectedProvider.name} (${qualityResult.qualityScore.toFixed(3)}) → ${escalationDecision.suggestedProvider}`);
            
            // Try the suggested provider
            try {
              const escalatedResponse = await this.callProvider(escalationDecision.suggestedProvider, prompt);
              
              // Track escalated usage
              const escalatedFullContent = prompt + escalatedResponse;
              const escalatedTokenEstimation = await this.tokenEstimationService.estimateTokens(
                escalatedFullContent, 
                escalationDecision.suggestedProvider, 
                this.getProviderModel(escalationDecision.suggestedProvider)
              );
              this.trackUsageWithAccurateTokens(escalationDecision.suggestedProvider, escalatedTokenEstimation, taskType);
              
              console.log(`✅ Escalated to ${escalationDecision.suggestedProvider} successfully`);
              
              return {
                response: escalatedResponse,
                provider: escalationDecision.suggestedProvider
              };
            } catch (escalationError) {
              console.log(`❌ Escalation to ${escalationDecision.suggestedProvider} failed: ${escalationError instanceof Error ? escalationError.message : 'Unknown error'}`);
              // Continue with original response
            }
          }
        }
        
        // Log quality metrics
        systemLogger.startup('OfficialLLMRouter', `Quality: ${selectedProvider.name} scored ${qualityResult.qualityScore.toFixed(3)} (confidence: ${qualityResult.confidence.toFixed(3)})`);
      }
      
      // Track analytics metrics
      await this.trackAnalyticsMetrics(
        selectedProvider.name,
        tokenEstimation,
        qualityResult?.qualityScore || 0.5,
        Date.now() - startTime,
        taskType,
        sessionId,
        qualityResult?.shouldEscalate ? 1 : 0,
        false // fallbackUsed
      );
      
      // Collect monitoring metrics
      await this.collectMonitoringMetrics(
        selectedProvider.name,
        tokenEstimation,
        qualityResult?.qualityScore || 0.5,
        Date.now() - startTime,
        true, // success
        taskType,
        sessionId,
        false, // fallbackUsed
        qualityResult?.shouldEscalate || false // escalationUsed
      );
      
      return {
        response: response as string,
        provider: selectedProvider.name
      };
    } catch (error) {
      errorLogger.aiError(error as Error, `${selectedProvider.name} routing error`);
      
      // Advanced fallback mechanism
      return await this.handleAdvancedFallback(sanitizedPrompt, taskType, sessionId, error as Error, selectedProvider.name);
    }
  }

  private async callProvider(providerName: string, prompt: string): Promise<string> {
    const config = this.providerConfigs.get(providerName);
    if (!config) throw new Error(`Provider ${providerName} not configured`);

    switch (providerName) {
      case 'openai':
        return await this.callOpenAI(config, prompt);
      case 'google':
        return await this.callGoogle(config, prompt);
      case 'cohere':
        return await this.callCohere(config, prompt);
      case 'mistral':
        return await this.callMistral(config, prompt);
      case 'xai':
        return await this.callXAI(config, prompt);
      case 'qwen':
        return await this.callQwen(config, prompt);
      case 'huggingface':
        return await this.callHuggingFace(config, prompt);
      default:
        throw new Error(`Unknown provider: ${providerName}`);
    }
  }

  private async callOpenAI(config: any, prompt: string): Promise<string> {
    // Use official OpenAI SDK
    const OpenAI = require('openai');
    const client = new OpenAI({
      apiKey: config.apiKey,
    });

    const response = await client.chat.completions.create({
      model: config.model,
      messages: [{ role: 'user', content: prompt }],
      temperature: config.temperature,
    });

    return response.choices[0].message.content;
  }

  private async callGoogle(config: any, prompt: string): Promise<string> {
    // Use official Google GenAI SDK
    const { GoogleGenAI } = require('@google/genai');
    const ai = new GoogleGenAI({
      apiKey: config.apiKey,
    });

    const response = await ai.models.generateContent({
      model: config.model,
      contents: prompt,
    });

    return response.text;
  }

  private async callCohere(config: any, prompt: string): Promise<string> {
    // Use official Cohere SDK
    const { CohereClientV2 } = require('cohere-ai');
    const cohere = new CohereClientV2({
      token: config.apiKey,
    });

    const response = await cohere.chat({
      model: config.model,
      messages: [
        {
          role: 'user',
          content: prompt,
        },
      ],
    });

    // Cohere V2 response shape:
    // {
    //   message: { role: 'assistant', content: [{ type: 'text', text: '...' }, ...] },
    //   ...
    // }
    const contentArray = (response && response.message && Array.isArray(response.message.content))
      ? response.message.content
      : [];
    const firstTextPart = contentArray.find((part: any) => part && part.type === 'text');
    const text = firstTextPart && firstTextPart.text ? firstTextPart.text : undefined;

    return text || JSON.stringify(response);
  }

  private async callMistral(config: any, prompt: string): Promise<string> {
    // Use LangChain for Mistral (it works well)
    const { ChatOpenAI } = require('@langchain/openai');
    const model = new ChatOpenAI({
      modelName: config.model,
      temperature: config.temperature,
      openAIApiKey: config.apiKey,
      configuration: {
        baseURL: config.baseURL
      }
    });

    const response = await model.invoke(prompt);
    return response.content;
  }

  private async callXAI(config: any, prompt: string): Promise<string> {
    // XAI uses OpenAI-compatible API
    const OpenAI = require('openai');
    const client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL,
    });

    const response = await client.chat.completions.create({
      model: config.model,
      messages: [{ role: 'user', content: prompt }],
      temperature: config.temperature,
    });

    return response.choices[0].message.content;
  }

  private async callQwen(config: any, prompt: string): Promise<string> {
    // Qwen uses OpenAI-compatible API
    const OpenAI = require('openai');
    const client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL,
    });

    const response = await client.chat.completions.create({
      model: config.model,
      messages: [{ role: 'user', content: prompt }],
      temperature: config.temperature,
    });

    return response.choices[0].message.content;
  }

  private async callHuggingFace(config: any, prompt: string): Promise<string> {
    // Hugging Face uses OpenAI-compatible API
    const OpenAI = require('openai');
    const client = new OpenAI({
      apiKey: config.apiKey,
      baseURL: config.baseURL,
    });

    const response = await client.chat.completions.create({
      model: config.model,
      messages: [{ role: 'user', content: prompt }],
      temperature: config.temperature,
    });

    return response.choices[0].message.content;
  }

  private selectBestProvider(taskType: string): {name: string, config: any} | null {
    const availableProviders = Array.from(this.providerConfigs.entries()).map(([name, config]) => ({name, config}));
    
    if (availableProviders.length === 0) return null;

    // Get provider names for cost optimization
    const providerNames = availableProviders.map(p => p.name.toLowerCase());
    
    // Get cost-optimized ranking
    const rankedProviders = this.costOptimizationEngine.rankProvidersByCost(providerNames, taskType);
    
    // Find the best provider considering both cost optimization and task-specific requirements
    let selectedProvider: {name: string, config: any} | null = null;
    
    // First, try to find a provider that matches task-specific requirements AND is cost-optimized
    for (const providerName of rankedProviders) {
      const provider = availableProviders.find(p => p.name.toLowerCase() === providerName);
      if (provider && this.isProviderSuitableForTask(provider.name, taskType)) {
        selectedProvider = provider;
        break;
      }
    }
    
    // If no task-specific match found, use the most cost-efficient available provider
    if (!selectedProvider) {
      for (const providerName of rankedProviders) {
        const provider = availableProviders.find(p => p.name.toLowerCase() === providerName);
        if (provider) {
          selectedProvider = provider;
          break;
        }
      }
    }
    
    // Fallback to original logic if cost optimization fails
    if (!selectedProvider) {
      selectedProvider = this.selectBestProviderLegacy(taskType, availableProviders);
    }

    return selectedProvider;
  }

  /**
   * Check if provider is suitable for specific task type
   */
  private isProviderSuitableForTask(providerName: string, taskType: string): boolean {
    switch (taskType) {
      case 'trading_analysis':
        return ['mistral', 'xai', 'google'].includes(providerName);
      case 'code_generation':
        return ['openai', 'xai', 'mistral'].includes(providerName);
      case 'multilingual':
        return ['mistral', 'google', 'qwen'].includes(providerName);
      case 'sentiment_analysis':
        return ['cohere', 'mistral'].includes(providerName);
      case 'reasoning':
        return ['huggingface', 'mistral', 'xai'].includes(providerName);
      default:
        return true; // All providers suitable for general tasks
    }
  }

  /**
   * Legacy provider selection logic (preserved for fallback)
   */
  private selectBestProviderLegacy(taskType: string, availableProviders: Array<{name: string, config: any}>): {name: string, config: any} | null {
    // Task-specific routing (preserved from original implementation)
    switch (taskType) {
      case 'trading_analysis':
        // Prefer Mistral for trading analysis (good at reasoning)
        return availableProviders.find(p => p.name === 'mistral') || 
               availableProviders.find(p => p.name === 'xai') || 
               availableProviders.find(p => p.name === 'google') || 
               availableProviders[0];
               
      case 'code_generation':
        // Prefer OpenAI for code (when quota resets), then XAI
        return availableProviders.find(p => p.name === 'openai') || 
               availableProviders.find(p => p.name === 'xai') || 
               availableProviders.find(p => p.name === 'mistral') || 
               availableProviders[0];
               
      case 'multilingual':
        // Prefer Mistral for multilingual
        return availableProviders.find(p => p.name === 'mistral') || 
               availableProviders.find(p => p.name === 'google') || 
               availableProviders[0];
               
      case 'sentiment_analysis':
        // Prefer Cohere for sentiment
        return availableProviders.find(p => p.name === 'cohere') || 
               availableProviders.find(p => p.name === 'mistral') || 
               availableProviders[0];
               
      default: {
        // Round-robin for general tasks
        const index = Math.floor(Math.random() * availableProviders.length);
        return availableProviders[index];
      }
    }
  }

  private async fallbackRequest(prompt: string, taskType: string = 'general', sessionId?: string): Promise<{response: string, provider: string}> {
    const availableProviders = Array.from(this.providerConfigs.entries()).map(([name, config]) => ({name, config}));
    
    for (const provider of availableProviders) {
      try {
        console.log(`🔄 Fallback trying ${provider.name}...`);
        
        const response = await Promise.race([
          this.callProvider(provider.name, prompt),
          new Promise((_, reject) => 
            setTimeout(() => reject(new Error(`${provider.name} timeout`)), 8000)
          )
        ]);
        
        console.log(`✅ ${provider.name} fallback succeeded`);
        
        return {
          response: response as string,
          provider: provider.name
        };
      } catch (error) {
        console.log(`❌ ${provider.name} fallback failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
      }
    }
    
    throw new Error('All providers failed');
  }

  public getUsageStats(): any {
    return {
      providers: Array.from(this.providerConfigs.keys()),
      totalProviders: this.providerConfigs.size,
      status: 'official-sdk-ready',
      timestamp: new Date()
    };
  }

  public getProviderStatus(): any {
    return Array.from(this.providerConfigs.keys()).map(name => ({
      name: name,
      status: 'available',
      configured: true,
      sdk: 'official'
    }));
  }

  /**
   * Check provider health status
   */
  public async checkProviderHealth(providerName: string): Promise<boolean> {
    try {
      const config = this.providerConfigs.get(providerName);
      if (!config) return false;

      // Simple health check by making a minimal request
      const testPrompt = "Hello";
      const response = await this.callProvider(providerName, testPrompt);
      
      const isHealthy = response && response.length > 0;
      await this.providerHealthMonitor.updateProviderStatus(
        providerName,
        isHealthy,
        1000,
        isHealthy ? 'Health check passed' : 'Health check failed'
      );
      
      systemLogger.startup('OfficialLLMRouter', `Health check for ${providerName}: ${isHealthy ? 'PASSED' : 'FAILED'}`);
      return isHealthy;
    } catch (error) {
      errorLogger.aiError(error as Error, `Health check for ${providerName}`);
      await this.providerHealthMonitor.updateProviderStatus(
        providerName,
        false,
        5000,
        error instanceof Error ? error.message : 'Unknown error'
      );
      return false;
    }
  }

  /**
   * Get comprehensive provider statistics
   */
  public getProviderStats(): any {
    return {
      totalProviders: this.providerConfigs.size,
      healthyProviders: Array.from(this.providerConfigs.keys()).filter(name => 
        (this.providerHealthMonitor as any).isProviderHealthy?.(name) || true
      ).length,
      providers: Array.from(this.providerConfigs.keys()).map(name => ({
        name,
        configured: true,
        healthy: (this.providerHealthMonitor as any).isProviderHealthy?.(name) || true,
        sdk: 'official'
      })),
      timestamp: new Date().toISOString()
    };
  }

  /**
   * Track usage and cost metrics
   */
  private trackUsage(providerName: string, tokens: number, task: string): void {
    // Track cost metrics
    const costPerToken = this.getProviderCostPerToken(providerName);
    const costMetrics: CostMetrics = {
      provider: providerName.toLowerCase(),
      tokensUsed: tokens,
      costPerToken: costPerToken,
      totalCost: tokens * costPerToken,
      timestamp: new Date(),
      taskType: task
    };
    
    this.costOptimizationEngine.trackCostMetrics(costMetrics);
    
    // Track usage history
    this.usageHistory.push({
      provider: providerName,
      tokensUsed: tokens,
      timestamp: new Date(),
      task
    });
  }

  /**
   * Get provider model name for token estimation
   */
  private getProviderModel(providerName: string): string {
    const config = this.providerConfigs.get(providerName.toLowerCase());
    return config?.model || 'unknown';
  }

  /**
   * Track usage with accurate token estimation
   */
  private async trackUsageWithAccurateTokens(
    providerName: string, 
    tokenEstimation: any, 
    taskType: string
  ): Promise<void> {
    // Create enhanced cost metrics
    const costPerToken = this.getProviderCostPerToken(providerName);
    const costMetrics: CostMetrics = {
      provider: providerName,
      tokensUsed: tokenEstimation.tokenCount,
      costPerToken,
      totalCost: tokenEstimation.tokenCount * costPerToken,
      timestamp: new Date(),
      taskType
    };

    // Track in cost optimization engine
    this.costOptimizationEngine.trackCostMetrics(costMetrics);

    // Add to usage history
    this.usageHistory.push({
      provider: providerName,
      tokensUsed: tokenEstimation.tokenCount,
      timestamp: new Date(),
      task: taskType
    });

    // Log usage with accuracy info
    console.log(`📊 Used ${tokenEstimation.tokenCount} tokens (confidence: ${tokenEstimation.confidence.toFixed(2)}) from ${providerName}`);
  }

  /**
   * Estimate token count (legacy method - kept for backward compatibility)
   * @deprecated Use TokenEstimationService for accurate estimation
   */
  private estimateTokens(text: string): number {
    return Math.ceil(text.length / 4); // Rough estimate: 4 chars per token
  }

  /**
   * Get cost per token for a provider
   */
  private getProviderCostPerToken(providerName: string): number {
    const costMap: Record<string, number> = {
      'openai': 0.00015,
      'google': 0.000075,
      'cohere': 0.0002,
      'mistral': 0.0001,
      'xai': 0.0001,
      'qwen': 0.00008,        // Very affordable
      'huggingface': 0.00005  // Very cheap
    };
    
    return costMap[providerName.toLowerCase()] || 0.0001;
  }

  /**
   * Get cost optimization statistics
   */
  public getCostOptimizationStats(): any {
    return this.costOptimizationEngine.getCostStatistics();
  }

  /**
   * Get cost optimization configuration
   */
  public getCostOptimizationConfig(): any {
    return this.costOptimizationEngine.getConfiguration();
  }

  /**
   * Update provider availability for cost optimization
   */
  public updateProviderAvailability(providerName: string, isAvailable: boolean): void {
    this.costOptimizationEngine.updateProviderAvailability(providerName.toLowerCase(), isAvailable);
  }

  /**
   * Get token estimation statistics
   */
  public getTokenEstimationStats(): any {
    return this.tokenEstimationService.getStats();
  }

  /**
   * Get cache statistics for token estimation
   */
  public getTokenEstimationCacheStats(): any {
    return this.tokenEstimationService.getCacheStats();
  }

  /**
   * Clear token estimation cache
   */
  public async clearTokenEstimationCache(): Promise<void> {
    await this.tokenEstimationService.clearCache();
  }

  /**
   * Warm up token estimation cache with common patterns
   */
  public async warmupTokenEstimationCache(patterns: string[]): Promise<void> {
    await this.tokenEstimationService.warmupCache(patterns);
  }

  /**
   * Get quality threshold statistics
   */
  public getQualityStats(): any {
    return this.qualityThresholdManager.getQualityStats();
  }

  /**
   * Get provider quality profiles
   */
  public getProviderQualityProfiles(): any {
    return this.qualityThresholdManager.getProviderProfiles();
  }

  /**
   * Get quality evaluation history
   */
  public getQualityHistory(limit?: number): any {
    return this.qualityThresholdManager.getQualityHistory(limit);
  }

  /**
   * Update quality thresholds for a provider
   */
  public async updateQualityThresholds(provider: string, config: any): Promise<void> {
    await this.qualityThresholdManager.updateQualityThresholds(provider, config);
  }

  /**
   * Enable/disable quality evaluation
   */
  public setQualityEvaluationEnabled(enabled: boolean): void {
    this.qualityThresholdManager.setQualityEvaluationEnabled(enabled);
  }

  /**
   * Check if quality evaluation is enabled
   */
  public isQualityEvaluationEnabled(): boolean {
    return this.qualityThresholdManager.isQualityEvaluationEnabled();
  }

  /**
   * Reset escalation count for a session
   */
  public resetEscalationCount(sessionId: string): void {
    this.qualityThresholdManager.resetEscalationCount(sessionId);
  }

  /**
   * Get escalation count for a session
   */
  public getEscalationCount(sessionId: string): number {
    return this.qualityThresholdManager.getEscalationCount(sessionId);
  }

  /**
   * Clear quality history
   */
  public clearQualityHistory(): void {
    this.qualityThresholdManager.clearQualityHistory();
  }

  /**
   * Get quality metrics for a response
   */
  public async getQualityMetrics(response: string, taskType: string): Promise<any> {
    return await this.qualityThresholdManager.getQualityMetrics(response, taskType);
  }

  /**
   * Track analytics metrics for a request
   */
  private async trackAnalyticsMetrics(
    provider: string,
    tokenEstimation: any,
    qualityScore: number,
    responseTime: number,
    taskType: string,
    sessionId?: string,
    escalationCount: number = 0,
    fallbackUsed: boolean = false
  ): Promise<void> {
    try {
      const metrics: RequestMetrics = {
        requestId: `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        provider,
        tokensUsed: tokenEstimation.totalTokens || 0,
        cost: tokenEstimation.estimatedCost || 0,
        qualityScore,
        responseTime,
        timestamp: new Date(),
        taskType,
        sessionId,
        escalationCount,
        fallbackUsed
      };

      await this.analyticsCollector.trackRequestMetrics(metrics);
    } catch (error) {
      console.error('Analytics tracking error:', error);
    }
  }

  /**
   * Get analytics dashboard
   */
  public async getAnalyticsDashboard(timeRange?: any): Promise<any> {
    return await this.analyticsCollector.getAnalyticsDashboard(timeRange);
  }

  /**
   * Get cost report
   */
  public async getCostReport(timeRange: any): Promise<any> {
    return await this.analyticsCollector.generateCostReport(timeRange);
  }

  /**
   * Get provider utilization
   */
  public async getProviderUtilization(): Promise<any> {
    return await this.analyticsCollector.getProviderUtilization();
  }

  /**
   * Get user satisfaction metrics
   */
  public async getUserSatisfactionMetrics(): Promise<any> {
    return await this.analyticsCollector.getUserSatisfactionMetrics();
  }

  /**
   * Get analytics statistics
   */
  public getAnalyticsStats(): any {
    return this.analyticsCollector.getAnalyticsStats();
  }

  /**
   * Handle advanced fallback mechanism
   */
  private async handleAdvancedFallback(
    prompt: string,
    taskType: string,
    sessionId: string | undefined,
    error: Error,
    originalProvider: string
  ): Promise<{response: string, provider: string}> {
    try {
      const retryCount = this.getRetryCount(sessionId);
      
      // Make intelligent fallback decision
      const fallbackDecision = await this.intelligentFallbackManager.makeFallbackDecision(
        originalProvider,
        error,
        retryCount,
        taskType
      );

      if (!fallbackDecision.shouldFallback) {
        throw new Error(`Fallback not available: ${fallbackDecision.reason}`);
      }

      // Record fallback event
      this.intelligentFallbackManager.getHealthMonitor().recordFallbackEvent({
        originalProvider: originalProvider,
        fallbackProvider: fallbackDecision.suggestedProvider,
        reason: fallbackDecision.reason,
        timestamp: new Date(),
        success: false, // Will be updated if successful
        responseTime: 0,
        retryCount: fallbackDecision.retryCount
      });

      // Apply retry delay if needed
      if (fallbackDecision.estimatedDelay > 0) {
        await this.delay(fallbackDecision.estimatedDelay);
      }

      // Try the suggested fallback provider
      console.log(`🔄 Advanced fallback: ${originalProvider} → ${fallbackDecision.suggestedProvider} (${fallbackDecision.reason})`);

      try {
        const fallbackResponse = await this.callProvider(fallbackDecision.suggestedProvider, prompt);
        
        // Update fallback event as successful
        this.updateFallbackEventSuccess(fallbackDecision.suggestedProvider, true);
        
        // Update retry count
        this.updateRetryCount(sessionId, retryCount + 1);
        
        return {
          response: fallbackResponse,
          provider: fallbackDecision.suggestedProvider
        };
      } catch (fallbackError) {
        console.log(`❌ Fallback to ${fallbackDecision.suggestedProvider} failed: ${fallbackError instanceof Error ? fallbackError.message : 'Unknown error'}`);
        
        // Try cascading fallback
        return await this.handleCascadingFallback(prompt, taskType, sessionId, error, originalProvider, retryCount + 1);
      }

    } catch (fallbackError) {
      console.error('Advanced fallback handling error:', fallbackError);
      
      // Try cascading fallback as last resort
      return await this.handleCascadingFallback(prompt, taskType, sessionId, error, originalProvider, 0);
    }
  }

  /**
   * Handle cascading fallback to multiple providers
   */
  private async handleCascadingFallback(
    prompt: string,
    taskType: string,
    sessionId: string | undefined,
    originalError: Error,
    originalProvider: string,
    retryCount: number
  ): Promise<{response: string, provider: string}> {
    try {
      const availableProviders = await this.intelligentFallbackManager.getAvailableProviders(originalProvider);
      
      for (const provider of availableProviders) {
        try {
          console.log(`🔄 Cascading fallback attempt ${retryCount + 1}: trying ${provider}`);

          const response = await this.callProvider(provider, prompt);
          
          // Record successful cascading fallback
          this.intelligentFallbackManager.getHealthMonitor().recordFallbackEvent({
            originalProvider: originalProvider,
            fallbackProvider: provider,
            reason: `Cascading fallback attempt ${retryCount + 1}`,
            timestamp: new Date(),
            success: true,
            responseTime: 0,
            retryCount: retryCount + 1
          });

          return {
            response: response,
            provider: provider
          };
        } catch (providerError) {
          console.log(`❌ Cascading fallback to ${provider} failed: ${providerError instanceof Error ? providerError.message : 'Unknown error'}`);
          
          // Update provider health status
          await this.providerHealthMonitor.updateProviderStatus(
            provider,
            false,
            5000,
            providerError instanceof Error ? providerError.message : 'Unknown error'
          );
        }
      }

      // All fallback attempts failed
      throw new Error(`All fallback providers failed. Original error: ${originalError.message}`);

    } catch (cascadingError) {
      console.error('Cascading fallback handling error:', cascadingError);
      throw cascadingError;
    }
  }

  /**
   * Get retry count for session
   */
  private getRetryCount(sessionId: string | undefined): number {
    if (!sessionId) return 0;
    return this.sessionRetryCounts.get(sessionId) || 0;
  }

  /**
   * Update retry count for session
   */
  private updateRetryCount(sessionId: string | undefined, count: number): void {
    if (!sessionId) return;
    this.sessionRetryCounts.set(sessionId, count);
  }

  /**
   * Update fallback event success status
   */
  private updateFallbackEventSuccess(provider: string, success: boolean): void {
    const events = this.intelligentFallbackManager.getHealthMonitor().getFallbackEvents();
    const lastEvent = events[events.length - 1];
    
    if (lastEvent && lastEvent.fallbackProvider === provider) {
      lastEvent.success = success;
    }
  }

  /**
   * Setup usage tracking and reset
   */
  private setupUsageTracking(): void {
    // Reset usage daily
    setInterval(() => {
      this.resetDailyUsage();
    }, 24 * 60 * 60 * 1000); // 24 hours
  }

  /**
   * Reset daily usage counters
   */
  private resetDailyUsage(): void {
    // Clear session retry counts
    this.sessionRetryCounts.clear();
    this.sessionStartTimes.clear();
    
    systemLogger.startup('OfficialLLMRouter', 'Daily LLM usage counters reset');
  }

  /**
   * Delay execution for retry backoff
   */
  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Collect monitoring metrics
   */
  private async collectMonitoringMetrics(
    provider: string,
    tokenEstimation: any,
    qualityScore: number,
    responseTime: number,
    success: boolean,
    taskType: string,
    sessionId?: string,
    fallbackUsed: boolean = false,
    escalationUsed: boolean = false
  ): Promise<void> {
    try {
      const requestId = `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      
      // Collect monitoring metrics
      await this.monitoringService.collectMetrics({
        requestId,
        provider,
        cost: tokenEstimation.estimatedCost || 0,
        tokensUsed: tokenEstimation.totalTokens || 0,
        responseTime,
        qualityScore,
        success,
        errorType: success ? undefined : 'routing_error',
        fallbackUsed,
        escalationUsed,
        taskType,
        sessionId
      });

      // Collect performance metrics
      await this.monitoringService.collectPerformanceMetrics({
        routingDecisionTime: 50, // Estimated
        providerSelectionTime: 25, // Estimated
        tokenEstimationTime: 30, // Estimated
        qualityEvaluationTime: 40, // Estimated
        fallbackDecisionTime: 20, // Estimated
        totalRequestTime: responseTime,
        memoryUsage: process.memoryUsage().heapUsed,
        cpuUsage: 50 // Estimated
      });

      // Collect cost metrics
      await this.monitoringService.collectCostMetrics({
        totalCost: tokenEstimation.estimatedCost || 0,
        averageCost: tokenEstimation.estimatedCost || 0,
        costPerToken: tokenEstimation.totalTokens > 0 ? (tokenEstimation.estimatedCost || 0) / tokenEstimation.totalTokens : 0,
        costSavings: 0, // Would need historical data
        providerCostBreakdown: { [provider]: tokenEstimation.estimatedCost || 0 },
        dailyCost: 0, // Would need aggregation
        monthlyCost: 0, // Would need aggregation
        budgetUtilization: 0 // Would need budget data
      });

      // Collect quality metrics
      await this.monitoringService.collectQualityMetrics({
        averageQualityScore: qualityScore,
        qualityDegradationRate: 0, // Would need historical data
        escalationRate: escalationUsed ? 1 : 0,
        userSatisfactionScore: qualityScore,
        providerQualityBreakdown: { [provider]: qualityScore },
        qualityThresholdViolations: qualityScore < 0.7 ? 1 : 0
      });

    } catch (error) {
      console.error('Monitoring metrics collection error:', error);
    }
  }
}

export const officialLLMRouter = new OfficialLLMRouter();
