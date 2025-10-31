/**
 * Quality Threshold Types
 * Type definitions for quality evaluation and threshold management
 */

export interface QualityThresholdConfig {
  provider: string;
  minQualityScore: number;
  escalationThreshold: number;
  evaluationCriteria: string[];
  enabled: boolean;
}

export interface QualityEvaluationResult {
  provider: string;
  qualityScore: number;
  evaluationCriteria: Record<string, number>;
  shouldEscalate: boolean;
  confidence: number;
  timestamp: Date;
}

export interface QualityMetrics {
  responseLength: number;
  coherenceScore: number;
  relevanceScore: number;
  completenessScore: number;
  errorRate: number;
  userSatisfactionScore: number;
}

export interface EscalationDecision {
  shouldEscalate: boolean;
  reason: string;
  currentProvider: string;
  suggestedProvider: string;
  qualityScore: number;
  threshold: number;
  confidence: number;
}

export interface QualityThresholdSettings {
  globalMinQuality: number;
  escalationThreshold: number;
  evaluationEnabled: boolean;
  fallbackEnabled: boolean;
  maxEscalationsPerSession: number;
  qualityEvaluationTimeout: number;
}

export interface QualityStats {
  totalEvaluations: number;
  averageQualityScore: number;
  escalationCount: number;
  escalationRate: number;
  userSatisfactionRate: number;
  providerQualityBreakdown: Record<string, number>;
}

export interface QualityEvaluationCriteria {
  name: string;
  weight: number;
  evaluator: (response: string, taskType: string) => Promise<number>;
  description: string;
}

export interface ProviderQualityProfile {
  provider: string;
  averageQuality: number;
  qualityConsistency: number;
  escalationRate: number;
  userSatisfaction: number;
  strengths: string[];
  weaknesses: string[];
}
