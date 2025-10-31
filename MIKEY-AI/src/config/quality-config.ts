/**
 * Quality Configuration
 * Configurable quality thresholds and evaluation criteria
 */

import { QualityThresholdConfig, QualityThresholdSettings, QualityEvaluationCriteria } from '../types/quality-thresholds';

export class QualityConfig {
  private static instance: QualityConfig;
  private thresholds: Map<string, QualityThresholdConfig> = new Map();
  private settings: QualityThresholdSettings;
  private evaluationCriteria: QualityEvaluationCriteria[] = [];

  private constructor() {
    this.settings = this.loadDefaultSettings();
    this.initializeThresholds();
    this.initializeEvaluationCriteria();
  }

  public static getInstance(): QualityConfig {
    if (!QualityConfig.instance) {
      QualityConfig.instance = new QualityConfig();
    }
    return QualityConfig.instance;
  }

  private loadDefaultSettings(): QualityThresholdSettings {
    return {
      globalMinQuality: parseFloat(process.env.GLOBAL_MIN_QUALITY || '0.7'),
      escalationThreshold: parseFloat(process.env.ESCALATION_THRESHOLD || '0.6'),
      evaluationEnabled: process.env.QUALITY_EVALUATION_ENABLED !== 'false',
      fallbackEnabled: process.env.QUALITY_FALLBACK_ENABLED !== 'false',
      maxEscalationsPerSession: parseInt(process.env.MAX_ESCALATIONS_PER_SESSION || '3'),
      qualityEvaluationTimeout: parseInt(process.env.QUALITY_EVALUATION_TIMEOUT || '100')
    };
  }

  private initializeThresholds(): void {
    // OpenAI Provider Quality Threshold
    this.thresholds.set('openai', {
      provider: 'openai',
      minQualityScore: parseFloat(process.env.OPENAI_MIN_QUALITY || '0.8'),
      escalationThreshold: parseFloat(process.env.OPENAI_ESCALATION_THRESHOLD || '0.6'),
      evaluationCriteria: ['coherence', 'relevance', 'completeness', 'accuracy'],
      enabled: true
    });

    // Google Provider Quality Threshold
    this.thresholds.set('google', {
      provider: 'google',
      minQualityScore: parseFloat(process.env.GOOGLE_MIN_QUALITY || '0.75'),
      escalationThreshold: parseFloat(process.env.GOOGLE_ESCALATION_THRESHOLD || '0.6'),
      evaluationCriteria: ['coherence', 'relevance', 'completeness', 'accuracy'],
      enabled: true
    });

    // Mistral Provider Quality Threshold
    this.thresholds.set('mistral', {
      provider: 'mistral',
      minQualityScore: parseFloat(process.env.MISTRAL_MIN_QUALITY || '0.7'),
      escalationThreshold: parseFloat(process.env.MISTRAL_ESCALATION_THRESHOLD || '0.5'),
      evaluationCriteria: ['coherence', 'relevance', 'completeness'],
      enabled: true
    });

    // Cohere Provider Quality Threshold
    this.thresholds.set('cohere', {
      provider: 'cohere',
      minQualityScore: parseFloat(process.env.COHERE_MIN_QUALITY || '0.7'),
      escalationThreshold: parseFloat(process.env.COHERE_ESCALATION_THRESHOLD || '0.5'),
      evaluationCriteria: ['coherence', 'relevance', 'completeness'],
      enabled: true
    });

    // HuggingFace Provider Quality Threshold
    this.thresholds.set('huggingface', {
      provider: 'huggingface',
      minQualityScore: parseFloat(process.env.HUGGINGFACE_MIN_QUALITY || '0.65'),
      escalationThreshold: parseFloat(process.env.HUGGINGFACE_ESCALATION_THRESHOLD || '0.45'),
      evaluationCriteria: ['coherence', 'relevance'],
      enabled: true
    });

    // XAI Provider Quality Threshold
    this.thresholds.set('xai', {
      provider: 'xai',
      minQualityScore: parseFloat(process.env.XAI_MIN_QUALITY || '0.75'),
      escalationThreshold: parseFloat(process.env.XAI_ESCALATION_THRESHOLD || '0.55'),
      evaluationCriteria: ['coherence', 'relevance', 'completeness', 'accuracy'],
      enabled: true
    });
  }

  private initializeEvaluationCriteria(): void {
    this.evaluationCriteria = [
      {
        name: 'coherence',
        weight: 0.25,
        evaluator: this.evaluateCoherence.bind(this),
        description: 'Logical flow and consistency of response'
      },
      {
        name: 'relevance',
        weight: 0.25,
        evaluator: this.evaluateRelevance.bind(this),
        description: 'Relevance to the task and context'
      },
      {
        name: 'completeness',
        weight: 0.2,
        evaluator: this.evaluateCompleteness.bind(this),
        description: 'Completeness of the response'
      },
      {
        name: 'accuracy',
        weight: 0.2,
        evaluator: this.evaluateAccuracy.bind(this),
        description: 'Accuracy of information provided'
      },
      {
        name: 'clarity',
        weight: 0.1,
        evaluator: this.evaluateClarity.bind(this),
        description: 'Clarity and readability of response'
      }
    ];
  }

  // Quality evaluation methods
  private async evaluateCoherence(response: string, taskType: string): Promise<number> {
    // Simple coherence evaluation based on response structure
    const sentences = response.split(/[.!?]+/).filter(s => s.trim().length > 0);
    if (sentences.length < 2) return 0.5;
    
    // Check for logical connectors and transitions
    const connectors = ['however', 'therefore', 'moreover', 'furthermore', 'additionally', 'consequently'];
    const hasConnectors = connectors.some(connector => response.toLowerCase().includes(connector));
    
    // Check for repetition (negative indicator)
    const words = response.toLowerCase().split(/\s+/);
    const uniqueWords = new Set(words);
    const repetitionRate = 1 - (uniqueWords.size / words.length);
    
    let score = 0.7; // Base score
    if (hasConnectors) score += 0.2;
    if (repetitionRate < 0.3) score += 0.1;
    
    return Math.min(1.0, Math.max(0.0, score));
  }

  private async evaluateRelevance(response: string, taskType: string): Promise<number> {
    // Simple relevance evaluation based on task type keywords
    const taskKeywords: Record<string, string[]> = {
      'analysis': ['analyze', 'analysis', 'examine', 'evaluate', 'assess'],
      'general': ['help', 'assist', 'support', 'provide', 'explain'],
      'code': ['code', 'function', 'method', 'class', 'variable', 'algorithm'],
      'trading': ['trade', 'trading', 'market', 'price', 'strategy', 'portfolio']
    };
    
    const keywords = taskKeywords[taskType] || taskKeywords['general'];
    const responseLower = response.toLowerCase();
    const keywordMatches = keywords.filter(keyword => responseLower.includes(keyword)).length;
    
    const relevanceScore = keywordMatches / keywords.length;
    return Math.min(1.0, Math.max(0.0, relevanceScore));
  }

  private async evaluateCompleteness(response: string, taskType: string): Promise<number> {
    // Simple completeness evaluation based on response length and structure
    const minLength = 50; // Minimum expected response length
    const maxLength = 2000; // Maximum expected response length
    
    let score = 0.5; // Base score
    
    if (response.length >= minLength) {
      score += 0.3;
    }
    
    if (response.length <= maxLength) {
      score += 0.2;
    }
    
    // Check for question answering completeness
    if (response.includes('?') && response.length > 100) {
      score += 0.1;
    }
    
    return Math.min(1.0, Math.max(0.0, score));
  }

  private async evaluateAccuracy(response: string, taskType: string): Promise<number> {
    // Simple accuracy evaluation based on response characteristics
    let score = 0.7; // Base score
    
    // Check for specific claims (positive indicator)
    const specificIndicators = ['specifically', 'exactly', 'precisely', 'according to', 'based on'];
    const hasSpecificity = specificIndicators.some(indicator => response.toLowerCase().includes(indicator));
    
    if (hasSpecificity) {
      score += 0.2;
    }
    
    // Check for uncertainty markers (negative indicator)
    const uncertaintyMarkers = ['might', 'could', 'possibly', 'perhaps', 'maybe', 'unclear'];
    const uncertaintyCount = uncertaintyMarkers.filter(marker => response.toLowerCase().includes(marker)).length;
    
    if (uncertaintyCount > 3) {
      score -= 0.1;
    }
    
    return Math.min(1.0, Math.max(0.0, score));
  }

  private async evaluateClarity(response: string, taskType: string): Promise<number> {
    // Simple clarity evaluation based on readability
    const sentences = response.split(/[.!?]+/).filter(s => s.trim().length > 0);
    const avgSentenceLength = sentences.reduce((sum, sentence) => sum + sentence.split(/\s+/).length, 0) / sentences.length;
    
    let score = 0.7; // Base score
    
    // Optimal sentence length is around 15-20 words
    if (avgSentenceLength >= 10 && avgSentenceLength <= 25) {
      score += 0.2;
    } else if (avgSentenceLength > 30) {
      score -= 0.1;
    }
    
    // Check for complex words (negative indicator)
    const complexWords = response.split(/\s+/).filter(word => word.length > 10).length;
    const totalWords = response.split(/\s+/).length;
    const complexWordRate = complexWords / totalWords;
    
    if (complexWordRate > 0.2) {
      score -= 0.1;
    }
    
    return Math.min(1.0, Math.max(0.0, score));
  }

  // Public methods
  public getThreshold(provider: string): QualityThresholdConfig | undefined {
    return this.thresholds.get(provider.toLowerCase());
  }

  public getAllThresholds(): QualityThresholdConfig[] {
    return Array.from(this.thresholds.values());
  }

  public getSettings(): QualityThresholdSettings {
    return { ...this.settings };
  }

  public getEvaluationCriteria(): QualityEvaluationCriteria[] {
    return [...this.evaluationCriteria];
  }

  public updateThreshold(provider: string, config: Partial<QualityThresholdConfig>): void {
    const existing = this.thresholds.get(provider.toLowerCase());
    if (existing) {
      this.thresholds.set(provider.toLowerCase(), { ...existing, ...config });
    }
  }

  public updateSettings(settings: Partial<QualityThresholdSettings>): void {
    this.settings = { ...this.settings, ...settings };
  }

  public isProviderEnabled(provider: string): boolean {
    const threshold = this.getThreshold(provider);
    return threshold?.enabled ?? false;
  }

  public getProviderMinQuality(provider: string): number {
    const threshold = this.getThreshold(provider);
    return threshold?.minQualityScore ?? this.settings.globalMinQuality;
  }

  public getProviderEscalationThreshold(provider: string): number {
    const threshold = this.getThreshold(provider);
    return threshold?.escalationThreshold ?? this.settings.escalationThreshold;
  }

  public validateConfiguration(): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate thresholds
    this.thresholds.forEach((threshold, provider) => {
      if (threshold.minQualityScore < 0 || threshold.minQualityScore > 1) {
        errors.push(`Invalid minQualityScore for ${provider}: ${threshold.minQualityScore}`);
      }
      if (threshold.escalationThreshold < 0 || threshold.escalationThreshold > 1) {
        errors.push(`Invalid escalationThreshold for ${provider}: ${threshold.escalationThreshold}`);
      }
      if (threshold.minQualityScore <= threshold.escalationThreshold) {
        errors.push(`minQualityScore must be greater than escalationThreshold for ${provider}`);
      }
    });

    // Validate settings
    if (this.settings.globalMinQuality < 0 || this.settings.globalMinQuality > 1) {
      errors.push(`Invalid globalMinQuality: ${this.settings.globalMinQuality}`);
    }
    if (this.settings.escalationThreshold < 0 || this.settings.escalationThreshold > 1) {
      errors.push(`Invalid escalationThreshold: ${this.settings.escalationThreshold}`);
    }
    if (this.settings.maxEscalationsPerSession < 0) {
      errors.push(`Invalid maxEscalationsPerSession: ${this.settings.maxEscalationsPerSession}`);
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }
}
