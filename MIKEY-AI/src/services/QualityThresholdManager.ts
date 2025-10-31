/**
 * Quality Threshold Manager
 * Intelligent quality evaluation and automatic escalation system
 */

import { 
  QualityEvaluationResult, 
  QualityMetrics, 
  EscalationDecision, 
  QualityStats,
  ProviderQualityProfile 
} from '../types/quality-thresholds';
import { QualityConfig } from '../config/quality-config';
import { CostOptimizationEngine } from './CostOptimizationEngine';

export class QualityThresholdManager {
  private config: QualityConfig;
  private costEngine: CostOptimizationEngine;
  private qualityHistory: QualityEvaluationResult[] = [];
  private providerProfiles: Map<string, ProviderQualityProfile> = new Map();
  private escalationCount: Map<string, number> = new Map(); // Track escalations per session

  constructor() {
    this.config = QualityConfig.getInstance();
    this.costEngine = new CostOptimizationEngine();
    this.initializeProviderProfiles();
  }

  /**
   * Evaluate quality of a response
   */
  public async evaluateQuality(
    response: string, 
    taskType: string, 
    provider: string
  ): Promise<QualityEvaluationResult> {
    const startTime = Date.now();
    
    try {
      // Get provider-specific threshold configuration
      const threshold = this.config.getThreshold(provider);
      if (!threshold) {
        throw new Error(`No quality threshold configuration found for provider: ${provider}`);
      }

      // Evaluate quality using configured criteria
      const evaluationCriteria = this.config.getEvaluationCriteria();
      const criteriaScores: Record<string, number> = {};
      
      let totalScore = 0;
      let totalWeight = 0;

      for (const criterion of evaluationCriteria) {
        if (threshold.evaluationCriteria.includes(criterion.name)) {
          const score = await criterion.evaluator(response, taskType);
          criteriaScores[criterion.name] = score;
          totalScore += score * criterion.weight;
          totalWeight += criterion.weight;
        }
      }

      const qualityScore = totalWeight > 0 ? totalScore / totalWeight : 0.5;
      const shouldEscalate = await this.shouldEscalate(qualityScore, threshold.escalationThreshold);

      const result: QualityEvaluationResult = {
        provider,
        qualityScore,
        evaluationCriteria: criteriaScores,
        shouldEscalate,
        confidence: this.calculateConfidence(criteriaScores),
        timestamp: new Date()
      };

      // Track quality history
      this.qualityHistory.push(result);

      // Update provider profile
      this.updateProviderProfile(provider, qualityScore);

      // Log quality evaluation
      console.log(`📊 Quality evaluation for ${provider}: ${qualityScore.toFixed(3)} (${shouldEscalate ? 'ESCALATE' : 'ACCEPT'})`);

      return result;

    } catch (error) {
      console.error(`Quality evaluation failed for ${provider}:`, error);
      
      // Return fallback result
      return {
        provider,
        qualityScore: 0.5, // Neutral score
        evaluationCriteria: {},
        shouldEscalate: false,
        confidence: 0.0,
        timestamp: new Date()
      };
    }
  }

  /**
   * Determine if escalation is needed
   */
  public async shouldEscalate(qualityScore: number, threshold: number): Promise<boolean> {
    return qualityScore < threshold;
  }

  /**
   * Get optimal fallback provider for escalation
   */
  public async getOptimalFallbackProvider(currentProvider: string): Promise<string> {
    const availableProviders = this.getAvailableProviders();
    const currentProviderLower = currentProvider.toLowerCase();
    
    // Filter out current provider
    const fallbackProviders = availableProviders.filter(p => p !== currentProviderLower);
    
    if (fallbackProviders.length === 0) {
      return currentProviderLower; // No fallback available
    }

    // Get provider profiles and sort by quality
    const providerScores = fallbackProviders.map(provider => {
      const profile = this.providerProfiles.get(provider);
      const costTier = this.costEngine.getProviderCostTier(provider);
      
      // Prefer affordable premium providers with good quality
      let score = profile?.averageQuality || 0.5;
      if (costTier?.tier === 'affordable_premium') {
        score += 0.1; // Bonus for affordable premium
      }
      
      return { provider, score };
    });

    // Sort by score (highest first)
    providerScores.sort((a, b) => b.score - a.score);
    
    return providerScores[0].provider;
  }

  /**
   * Make escalation decision
   */
  public async makeEscalationDecision(
    currentProvider: string,
    qualityScore: number,
    sessionId?: string
  ): Promise<EscalationDecision> {
    const sessionKey = sessionId || 'default';
    const escalationCount = this.escalationCount.get(sessionKey) || 0;
    const maxEscalations = this.config.getSettings().maxEscalationsPerSession;
    
    // Check if we've exceeded max escalations
    if (escalationCount >= maxEscalations) {
      return {
        shouldEscalate: false,
        reason: 'Maximum escalations per session exceeded',
        currentProvider,
        suggestedProvider: currentProvider,
        qualityScore,
        threshold: this.config.getProviderEscalationThreshold(currentProvider),
        confidence: 0.0
      };
    }

    const threshold = this.config.getProviderEscalationThreshold(currentProvider);
    const shouldEscalate = await this.shouldEscalate(qualityScore, threshold);
    
    if (!shouldEscalate) {
      return {
        shouldEscalate: false,
        reason: 'Quality score meets threshold',
        currentProvider,
        suggestedProvider: currentProvider,
        qualityScore,
        threshold,
        confidence: 1.0
      };
    }

    // Find optimal fallback provider
    const suggestedProvider = await this.getOptimalFallbackProvider(currentProvider);
    
    // Increment escalation count
    this.escalationCount.set(sessionKey, escalationCount + 1);

    return {
      shouldEscalate: true,
      reason: `Quality score ${qualityScore.toFixed(3)} below threshold ${threshold.toFixed(3)}`,
      currentProvider,
      suggestedProvider,
      qualityScore,
      threshold,
      confidence: 0.8
    };
  }

  /**
   * Update quality thresholds configuration
   */
  public async updateQualityThresholds(provider: string, config: Partial<any>): Promise<void> {
    this.config.updateThreshold(provider, config);
    console.log(`📝 Updated quality thresholds for ${provider}`);
  }

  /**
   * Get quality statistics
   */
  public getQualityStats(): QualityStats {
    const totalEvaluations = this.qualityHistory.length;
    const averageQualityScore = totalEvaluations > 0 
      ? this.qualityHistory.reduce((sum, result) => sum + result.qualityScore, 0) / totalEvaluations 
      : 0;

    const escalationCount = this.qualityHistory.filter(result => result.shouldEscalate).length;
    const escalationRate = totalEvaluations > 0 ? escalationCount / totalEvaluations : 0;

    // Calculate user satisfaction rate (simplified)
    const userSatisfactionRate = Math.max(0, averageQualityScore - 0.1);

    // Provider quality breakdown
    const providerQualityBreakdown: Record<string, number> = {};
    this.providerProfiles.forEach((profile, provider) => {
      providerQualityBreakdown[provider] = profile.averageQuality;
    });

    return {
      totalEvaluations,
      averageQualityScore,
      escalationCount,
      escalationRate,
      userSatisfactionRate,
      providerQualityBreakdown
    };
  }

  /**
   * Get provider quality profiles
   */
  public getProviderProfiles(): ProviderQualityProfile[] {
    return Array.from(this.providerProfiles.values());
  }

  /**
   * Reset escalation count for a session
   */
  public resetEscalationCount(sessionId: string): void {
    this.escalationCount.delete(sessionId);
  }

  /**
   * Get escalation count for a session
   */
  public getEscalationCount(sessionId: string): number {
    return this.escalationCount.get(sessionId) || 0;
  }

  /**
   * Check if quality evaluation is enabled
   */
  public isQualityEvaluationEnabled(): boolean {
    return this.config.getSettings().evaluationEnabled;
  }

  /**
   * Enable/disable quality evaluation
   */
  public setQualityEvaluationEnabled(enabled: boolean): void {
    this.config.updateSettings({ evaluationEnabled: enabled });
  }

  /**
   * Get quality evaluation history
   */
  public getQualityHistory(limit?: number): QualityEvaluationResult[] {
    if (limit) {
      return this.qualityHistory.slice(-limit);
    }
    return [...this.qualityHistory];
  }

  /**
   * Clear quality history
   */
  public clearQualityHistory(): void {
    this.qualityHistory = [];
    this.providerProfiles.clear();
    this.escalationCount.clear();
  }

  // Private helper methods
  private initializeProviderProfiles(): void {
    const providers = ['openai', 'google', 'mistral', 'cohere', 'huggingface', 'xai'];
    
    providers.forEach(provider => {
      this.providerProfiles.set(provider, {
        provider,
        averageQuality: 0.75, // Default quality
        qualityConsistency: 0.8,
        escalationRate: 0.1,
        userSatisfaction: 0.8,
        strengths: [],
        weaknesses: []
      });
    });
  }

  private updateProviderProfile(provider: string, qualityScore: number): void {
    const profile = this.providerProfiles.get(provider);
    if (!profile) return;

    // Update average quality (exponential moving average)
    const alpha = 0.1; // Learning rate
    profile.averageQuality = alpha * qualityScore + (1 - alpha) * profile.averageQuality;

    // Update escalation rate
    const recentEvaluations = this.qualityHistory
      .filter(r => r.provider === provider)
      .slice(-10); // Last 10 evaluations
    
    if (recentEvaluations.length > 0) {
      profile.escalationRate = recentEvaluations.filter(r => r.shouldEscalate).length / recentEvaluations.length;
    }

    // Update user satisfaction (simplified)
    profile.userSatisfaction = Math.max(0, profile.averageQuality - 0.1);
  }

  private calculateConfidence(criteriaScores: Record<string, number>): number {
    const scores = Object.values(criteriaScores);
    if (scores.length === 0) return 0.0;

    // Calculate confidence based on score consistency
    const average = scores.reduce((sum, score) => sum + score, 0) / scores.length;
    const variance = scores.reduce((sum, score) => sum + Math.pow(score - average, 2), 0) / scores.length;
    const standardDeviation = Math.sqrt(variance);

    // Lower standard deviation = higher confidence
    const confidence = Math.max(0, 1 - standardDeviation);
    return Math.min(1.0, confidence);
  }

  private getAvailableProviders(): string[] {
    return ['openai', 'google', 'mistral', 'cohere', 'huggingface', 'xai'];
  }

  /**
   * Get quality metrics for a response
   */
  public async getQualityMetrics(response: string, taskType: string): Promise<QualityMetrics> {
    const evaluationCriteria = this.config.getEvaluationCriteria();
    const metrics: Partial<QualityMetrics> = {};

    for (const criterion of evaluationCriteria) {
      const score = await criterion.evaluator(response, taskType);
      
      switch (criterion.name) {
        case 'coherence':
          metrics.coherenceScore = score;
          break;
        case 'relevance':
          metrics.relevanceScore = score;
          break;
        case 'completeness':
          metrics.completenessScore = score;
          break;
        case 'clarity':
          metrics.userSatisfactionScore = score;
          break;
      }
    }

    return {
      responseLength: response.length,
      coherenceScore: metrics.coherenceScore || 0.5,
      relevanceScore: metrics.relevanceScore || 0.5,
      completenessScore: metrics.completenessScore || 0.5,
      errorRate: 0.0, // Would need error detection logic
      userSatisfactionScore: metrics.userSatisfactionScore || 0.5
    };
  }
}
