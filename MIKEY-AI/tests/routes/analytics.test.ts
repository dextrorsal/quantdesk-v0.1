/**
 * Unit tests for Analytics API Routes
 */

import request from 'supertest';
import express from 'express';
import analyticsRouter from '../routes/analytics';

// Mock dependencies
jest.mock('../services/AnalyticsCollector', () => {
  return {
    AnalyticsCollector: jest.fn().mockImplementation(() => ({
      generateCostReport: jest.fn().mockResolvedValue({
        timeRange: { start: new Date('2025-01-01'), end: new Date('2025-01-31') },
        totalCost: 100.50,
        costSavings: 25.75,
        providerBreakdown: [
          { provider: 'openai', totalCost: 60.30, tokenCount: 12000, requestCount: 150 },
          { provider: 'google', totalCost: 40.20, tokenCount: 8000, requestCount: 100 }
        ],
        qualityMetrics: { averageQuality: 0.85 },
        utilizationMetrics: { totalRequests: 250, totalTokens: 20000 }
      }),
      getProviderUtilization: jest.fn().mockResolvedValue([
        { provider: 'openai', requestCount: 150, totalTokens: 12000, utilizationPercentage: 60 },
        { provider: 'google', requestCount: 100, totalTokens: 8000, utilizationPercentage: 40 }
      ]),
      getUserSatisfactionMetrics: jest.fn().mockResolvedValue({
        averageQualityScore: 0.85,
        userSatisfactionRate: 0.75,
        escalationRate: 0.10,
        fallbackRate: 0.05,
        responseTimeMetrics: { average: 1500, median: 1400, p95: 2500, p99: 3000 },
        qualityDistribution: { excellent: 0.6, good: 0.3, average: 0.1, poor: 0.0 }
      }),
      generateCostSavingsReport: jest.fn().mockResolvedValue({
        timeRange: { start: new Date('2025-01-01'), end: new Date('2025-01-31') },
        baselineCost: 126.25,
        actualCost: 100.50,
        savingsAmount: 25.75,
        savingsPercentage: 20.4,
        roi: 25.6
      }),
      getAnalyticsDashboard: jest.fn().mockResolvedValue({
        overview: { totalRequests: 250, totalCost: 100.50, averageQualityScore: 0.85 },
        costReport: { totalCost: 100.50, costSavings: 25.75 },
        providerUtilization: [
          { provider: 'openai', requestCount: 150, utilizationPercentage: 60 }
        ],
        satisfactionMetrics: { averageQualityScore: 0.85, userSatisfactionRate: 0.75 },
        costSavings: { savingsAmount: 25.75, savingsPercentage: 20.4 },
        trends: { costTrend: [], qualityTrend: [], utilizationTrend: [], savingsTrend: [] }
      }),
      getAnalyticsStats: jest.fn().mockReturnValue({
        totalRequests: 250,
        totalCost: 100.50,
        totalSavings: 25.75,
        averageQualityScore: 0.85,
        dataPointsCollected: 250,
        lastUpdated: new Date()
      }),
      trackRequestMetrics: jest.fn().mockResolvedValue(undefined),
      exportAnalyticsData: jest.fn().mockResolvedValue([
        { requestId: 'req-1', provider: 'openai', cost: 0.05, qualityScore: 0.8 }
      ]),
      clearOldData: jest.fn().mockResolvedValue(undefined)
    }))
  };
});

jest.mock('../utils/logger', () => ({
  systemLogger: {
    startup: jest.fn()
  },
  errorLogger: {
    aiError: jest.fn()
  }
}));

describe('Analytics API Routes', () => {
  let app: express.Application;

  beforeEach(() => {
    app = express();
    app.use(express.json());
    app.use('/api/analytics', analyticsRouter);
  });

  describe('GET /api/analytics/cost-report', () => {
    test('should return cost report with valid parameters', async () => {
      const response = await request(app)
        .get('/api/analytics/cost-report')
        .query({ start: '2025-01-01', end: '2025-01-31' })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data).toBeDefined();
      expect(response.body.data.totalCost).toBe(100.50);
      expect(response.body.data.costSavings).toBe(25.75);
      expect(response.body.data.providerBreakdown).toHaveLength(2);
      expect(response.body.generatedAt).toBeDefined();
    });

    test('should return 400 for missing parameters', async () => {
      const response = await request(app)
        .get('/api/analytics/cost-report')
        .expect(400);

      expect(response.body.error).toBe('Missing required parameters');
      expect(response.body.message).toContain('start and end date parameters are required');
    });

    test('should return 400 for invalid date range', async () => {
      const response = await request(app)
        .get('/api/analytics/cost-report')
        .query({ start: '2025-01-31', end: '2025-01-01' })
        .expect(400);

      expect(response.body.error).toBe('Invalid date range');
      expect(response.body.message).toBe('Start date must be before end date');
    });
  });

  describe('GET /api/analytics/provider-utilization', () => {
    test('should return provider utilization metrics', async () => {
      const response = await request(app)
        .get('/api/analytics/provider-utilization')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data).toBeDefined();
      expect(response.body.data).toHaveLength(2);
      expect(response.body.data[0].provider).toBe('openai');
      expect(response.body.data[0].utilizationPercentage).toBe(60);
      expect(response.body.generatedAt).toBeDefined();
    });
  });

  describe('GET /api/analytics/user-satisfaction', () => {
    test('should return user satisfaction metrics', async () => {
      const response = await request(app)
        .get('/api/analytics/user-satisfaction')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data).toBeDefined();
      expect(response.body.data.averageQualityScore).toBe(0.85);
      expect(response.body.data.userSatisfactionRate).toBe(0.75);
      expect(response.body.data.escalationRate).toBe(0.10);
      expect(response.body.data.fallbackRate).toBe(0.05);
      expect(response.body.data.responseTimeMetrics).toBeDefined();
      expect(response.body.data.qualityDistribution).toBeDefined();
      expect(response.body.generatedAt).toBeDefined();
    });
  });

  describe('GET /api/analytics/cost-savings', () => {
    test('should return cost savings report with valid parameters', async () => {
      const response = await request(app)
        .get('/api/analytics/cost-savings')
        .query({ start: '2025-01-01', end: '2025-01-31' })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data).toBeDefined();
      expect(response.body.data.baselineCost).toBe(126.25);
      expect(response.body.data.actualCost).toBe(100.50);
      expect(response.body.data.savingsAmount).toBe(25.75);
      expect(response.body.data.savingsPercentage).toBe(20.4);
      expect(response.body.data.roi).toBe(25.6);
      expect(response.body.generatedAt).toBeDefined();
    });

    test('should return 400 for missing parameters', async () => {
      const response = await request(app)
        .get('/api/analytics/cost-savings')
        .expect(400);

      expect(response.body.error).toBe('Missing required parameters');
      expect(response.body.message).toContain('start and end date parameters are required');
    });
  });

  describe('GET /api/analytics/dashboard', () => {
    test('should return analytics dashboard', async () => {
      const response = await request(app)
        .get('/api/analytics/dashboard')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data).toBeDefined();
      expect(response.body.data.overview).toBeDefined();
      expect(response.body.data.costReport).toBeDefined();
      expect(response.body.data.providerUtilization).toBeDefined();
      expect(response.body.data.satisfactionMetrics).toBeDefined();
      expect(response.body.data.costSavings).toBeDefined();
      expect(response.body.data.trends).toBeDefined();
      expect(response.body.generatedAt).toBeDefined();
    });

    test('should return dashboard with time range parameters', async () => {
      const response = await request(app)
        .get('/api/analytics/dashboard')
        .query({ start: '2025-01-01', end: '2025-01-31' })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data).toBeDefined();
    });
  });

  describe('GET /api/analytics/stats', () => {
    test('should return analytics statistics', async () => {
      const response = await request(app)
        .get('/api/analytics/stats')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data).toBeDefined();
      expect(response.body.data.totalRequests).toBe(250);
      expect(response.body.data.totalCost).toBe(100.50);
      expect(response.body.data.averageQualityScore).toBe(0.85);
      expect(response.body.data.dataPointsCollected).toBe(250);
      expect(response.body.generatedAt).toBeDefined();
    });
  });

  describe('POST /api/analytics/track', () => {
    test('should track metrics successfully', async () => {
      const metrics = {
        requestId: 'test-request-1',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date().toISOString(),
        taskType: 'analysis'
      };

      const response = await request(app)
        .post('/api/analytics/track')
        .send(metrics)
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.message).toBe('Metrics tracked successfully');
      expect(response.body.requestId).toBe('test-request-1');
    });

    test('should return 400 for missing required fields', async () => {
      const invalidMetrics = {
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05
        // Missing required fields
      };

      const response = await request(app)
        .post('/api/analytics/track')
        .send(invalidMetrics)
        .expect(400);

      expect(response.body.error).toBe('Missing required fields');
      expect(response.body.message).toContain('requestId, provider, and taskType are required');
      expect(response.body.required).toBeDefined();
    });
  });

  describe('POST /api/analytics/export', () => {
    test('should export analytics data with filters', async () => {
      const query = {
        filter: {
          providers: ['openai'],
          timeRange: {
            start: '2025-01-01',
            end: '2025-01-31'
          }
        },
        limit: 10
      };

      const response = await request(app)
        .post('/api/analytics/export')
        .send(query)
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data).toBeDefined();
      expect(response.body.count).toBe(1);
      expect(response.body.data[0].provider).toBe('openai');
      expect(response.body.exportedAt).toBeDefined();
    });

    test('should return 400 for missing filter', async () => {
      const response = await request(app)
        .post('/api/analytics/export')
        .send({})
        .expect(400);

      expect(response.body.error).toBe('Missing filter');
      expect(response.body.message).toBe('Filter object is required');
    });
  });

  describe('POST /api/analytics/cleanup', () => {
    test('should trigger data cleanup', async () => {
      const response = await request(app)
        .post('/api/analytics/cleanup')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.message).toBe('Data cleanup completed successfully');
      expect(response.body.cleanedAt).toBeDefined();
    });
  });

  describe('GET /api/analytics/health', () => {
    test('should return health status', async () => {
      const response = await request(app)
        .get('/api/analytics/health')
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.status).toBe('healthy');
      expect(response.body.data).toBeDefined();
      expect(response.body.data.totalRequests).toBe(250);
      expect(response.body.data.dataPointsCollected).toBe(250);
      expect(response.body.data.uptime).toBeDefined();
      expect(response.body.checkedAt).toBeDefined();
    });
  });

  describe('Error Handling', () => {
    test('should handle service errors gracefully', async () => {
      // Mock service to throw error
      const mockAnalyticsCollector = require('../services/AnalyticsCollector').AnalyticsCollector;
      mockAnalyticsCollector.mockImplementation(() => ({
        generateCostReport: jest.fn().mockRejectedValue(new Error('Service error'))
      }));

      const response = await request(app)
        .get('/api/analytics/cost-report')
        .query({ start: '2025-01-01', end: '2025-01-31' })
        .expect(500);

      expect(response.body.error).toBe('Internal server error');
      expect(response.body.message).toBe('Failed to generate cost report');
    });

    test('should handle invalid JSON in POST requests', async () => {
      const response = await request(app)
        .post('/api/analytics/track')
        .set('Content-Type', 'application/json')
        .send('invalid json')
        .expect(400);

      // Express will handle JSON parsing errors
      expect(response.body).toBeDefined();
    });
  });

  describe('Request Validation', () => {
    test('should validate date format in query parameters', async () => {
      const response = await request(app)
        .get('/api/analytics/cost-report')
        .query({ start: 'invalid-date', end: '2025-01-31' })
        .expect(200); // Should still work, but dates will be invalid

      // The service should handle invalid dates gracefully
      expect(response.body.success).toBe(true);
    });

    test('should handle empty query parameters', async () => {
      const response = await request(app)
        .get('/api/analytics/cost-report')
        .query({ start: '', end: '' })
        .expect(400);

      expect(response.body.error).toBe('Missing required parameters');
    });
  });

  describe('Response Format', () => {
    test('should return consistent response format', async () => {
      const response = await request(app)
        .get('/api/analytics/stats')
        .expect(200);

      expect(response.body).toHaveProperty('success');
      expect(response.body).toHaveProperty('data');
      expect(response.body).toHaveProperty('generatedAt');
      expect(typeof response.body.success).toBe('boolean');
      expect(typeof response.body.generatedAt).toBe('string');
    });

    test('should return ISO timestamp format', async () => {
      const response = await request(app)
        .get('/api/analytics/stats')
        .expect(200);

      const timestamp = new Date(response.body.generatedAt);
      expect(timestamp.getTime()).not.toBeNaN();
    });
  });
});
