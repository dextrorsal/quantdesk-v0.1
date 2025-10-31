/**
 * Analytics Configuration
 * Configuration management for analytics collection and data retention
 */

import { AnalyticsConfig } from '../types/analytics';

export class AnalyticsConfiguration {
  private static instance: AnalyticsConfiguration;
  private config: AnalyticsConfig;

  private constructor() {
    this.config = this.loadConfiguration();
  }

  public static getInstance(): AnalyticsConfiguration {
    if (!AnalyticsConfiguration.instance) {
      AnalyticsConfiguration.instance = new AnalyticsConfiguration();
    }
    return AnalyticsConfiguration.instance;
  }

  private loadConfiguration(): AnalyticsConfig {
    return {
      dataRetentionDays: parseInt(process.env.ANALYTICS_DATA_RETENTION_DAYS || '90'),
      privacyCompliance: process.env.ANALYTICS_PRIVACY_COMPLIANCE !== 'false',
      anonymizeData: process.env.ANALYTICS_ANONYMIZE_DATA !== 'false',
      enableRealTimeTracking: process.env.ANALYTICS_REAL_TIME_TRACKING !== 'false',
      batchSize: parseInt(process.env.ANALYTICS_BATCH_SIZE || '100'),
      flushInterval: parseInt(process.env.ANALYTICS_FLUSH_INTERVAL || '30000') // 30 seconds
    };
  }

  public getConfiguration(): AnalyticsConfig {
    return { ...this.config };
  }

  public updateConfiguration(updates: Partial<AnalyticsConfig>): void {
    this.config = { ...this.config, ...updates };
  }

  public isDataRetentionEnabled(): boolean {
    return this.config.dataRetentionDays > 0;
  }

  public isPrivacyComplianceEnabled(): boolean {
    return this.config.privacyCompliance;
  }

  public isDataAnonymizationEnabled(): boolean {
    return this.config.anonymizeData;
  }

  public isRealTimeTrackingEnabled(): boolean {
    return this.config.enableRealTimeTracking;
  }

  public getDataRetentionDays(): number {
    return this.config.dataRetentionDays;
  }

  public getBatchSize(): number {
    return this.config.batchSize;
  }

  public getFlushInterval(): number {
    return this.config.flushInterval;
  }

  public validateConfiguration(): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    if (this.config.dataRetentionDays < 0) {
      errors.push('Data retention days must be non-negative');
    }

    if (this.config.batchSize <= 0) {
      errors.push('Batch size must be positive');
    }

    if (this.config.flushInterval <= 0) {
      errors.push('Flush interval must be positive');
    }

    if (this.config.dataRetentionDays > 365) {
      errors.push('Data retention days should not exceed 1 year for privacy compliance');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  public getPrivacyComplianceSettings(): {
    anonymizeUserData: boolean;
    retentionPolicy: string;
    dataProcessingBasis: string;
    userRights: string[];
  } {
    return {
      anonymizeUserData: this.config.anonymizeData,
      retentionPolicy: `${this.config.dataRetentionDays} days`,
      dataProcessingBasis: 'Legitimate interest for service optimization',
      userRights: [
        'Right to access personal data',
        'Right to rectification',
        'Right to erasure',
        'Right to data portability',
        'Right to object to processing'
      ]
    };
  }
}
