// QuantDesk Developer Playground - API Analytics Component
// Phase 3: Advanced Capabilities (Beyond Drift)
// Strategy: "More Open Than Drift" - Enterprise analytics features

import React, { useState, useEffect } from 'react';

interface ApiUsageStats {
  totalRequests: number;
  requestsToday: number;
  averageResponseTime: number;
  errorRate: number;
  topEndpoints: Array<{
    endpoint: string;
    requests: number;
    avgResponseTime: number;
  }>;
}

interface PerformanceMetrics {
  uptime: number;
  responseTime: {
    p50: number;
    p95: number;
    p99: number;
  };
  throughput: number;
  errorRate: number;
}

interface ErrorAnalytics {
  totalErrors: number;
  errorsByType: Array<{
    type: string;
    count: number;
    percentage: number;
  }>;
  errorsByEndpoint: Array<{
    endpoint: string;
    count: number;
  }>;
}

export const ApiAnalytics: React.FC = () => {
  const [usageStats, setUsageStats] = useState<ApiUsageStats | null>(null);
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null);
  const [errorAnalytics, setErrorAnalytics] = useState<ErrorAnalytics | null>(null);
  const [timeRange, setTimeRange] = useState<string>('24h');
  const [isLoading, setIsLoading] = useState<boolean>(true);

  useEffect(() => {
    // Simulate API analytics data loading
    const loadAnalytics = async () => {
      setIsLoading(true);
      
      // Mock data for demonstration
      setTimeout(() => {
        setUsageStats({
          totalRequests: 15420,
          requestsToday: 1250,
          averageResponseTime: 145,
          errorRate: 0.8,
          topEndpoints: [
            { endpoint: '/api/markets', requests: 3200, avgResponseTime: 120 },
            { endpoint: '/api/positions', requests: 2800, avgResponseTime: 150 },
            { endpoint: '/api/portfolio', requests: 2100, avgResponseTime: 135 },
            { endpoint: '/api/ai/analysis', requests: 1800, avgResponseTime: 200 },
            { endpoint: '/api/orders', requests: 1500, avgResponseTime: 160 }
          ]
        });

        setPerformanceMetrics({
          uptime: 99.9,
          responseTime: {
            p50: 120,
            p95: 250,
            p99: 450
          },
          throughput: 1250,
          errorRate: 0.8
        });

        setErrorAnalytics({
          totalErrors: 123,
          errorsByType: [
            { type: 'Rate Limited', count: 45, percentage: 36.6 },
            { type: 'Authentication', count: 32, percentage: 26.0 },
            { type: 'Validation', count: 28, percentage: 22.8 },
            { type: 'Server Error', count: 18, percentage: 14.6 }
          ],
          errorsByEndpoint: [
            { endpoint: '/api/positions', count: 35 },
            { endpoint: '/api/orders', count: 28 },
            { endpoint: '/api/ai/analysis', count: 22 },
            { endpoint: '/api/markets', count: 18 },
            { endpoint: '/api/portfolio', count: 15 }
          ]
        });

        setIsLoading(false);
      }, 1000);
    };

    loadAnalytics();
  }, [timeRange]);

  const getUptimeColor = (uptime: number) => {
    if (uptime >= 99.9) return 'text-green-600';
    if (uptime >= 99.0) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getErrorRateColor = (errorRate: number) => {
    if (errorRate <= 1.0) return 'text-green-600';
    if (errorRate <= 5.0) return 'text-yellow-600';
    return 'text-red-600';
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="animate-pulse">
            <div className="h-8 bg-gray-200 rounded w-1/4 mb-4"></div>
            <div className="h-4 bg-gray-200 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900 mb-2">
              📊 API Analytics & Monitoring
            </h2>
            <p className="text-gray-600">
              Comprehensive API usage monitoring and performance analytics - Beyond Drift's capabilities
            </p>
          </div>
          <div className="flex items-center space-x-4">
            <select
              value={timeRange}
              onChange={(e) => setTimeRange(e.target.value)}
              className="p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              <option value="1h">Last Hour</option>
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
            </select>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="text-2xl mr-3">📈</div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {usageStats?.totalRequests.toLocaleString()}
              </div>
              <div className="text-sm text-gray-600">Total Requests</div>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="text-2xl mr-3">⚡</div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {usageStats?.averageResponseTime}ms
              </div>
              <div className="text-sm text-gray-600">Avg Response Time</div>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="text-2xl mr-3">✅</div>
            <div>
              <div className={`text-2xl font-bold ${getUptimeColor(performanceMetrics?.uptime || 0)}`}>
                {performanceMetrics?.uptime}%
              </div>
              <div className="text-sm text-gray-600">Uptime</div>
            </div>
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <div className="flex items-center">
            <div className="text-2xl mr-3">❌</div>
            <div>
              <div className={`text-2xl font-bold ${getErrorRateColor(usageStats?.errorRate || 0)}`}>
                {usageStats?.errorRate}%
              </div>
              <div className="text-sm text-gray-600">Error Rate</div>
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Top Endpoints */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Top Endpoints
          </h3>
          
          <div className="space-y-3">
            {usageStats?.topEndpoints.map((endpoint, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded">
                <div className="flex-1">
                  <div className="font-medium text-gray-900">{endpoint.endpoint}</div>
                  <div className="text-sm text-gray-600">
                    {endpoint.requests.toLocaleString()} requests
                  </div>
                </div>
                <div className="text-sm text-gray-600">
                  {endpoint.avgResponseTime}ms avg
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="bg-white p-6 rounded-lg shadow-sm border">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Performance Metrics
          </h3>
          
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>Response Time (P50)</span>
                <span>{performanceMetrics?.responseTime.p50}ms</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-blue-600 h-2 rounded-full" style={{width: '60%'}}></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>Response Time (P95)</span>
                <span>{performanceMetrics?.responseTime.p95}ms</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-yellow-600 h-2 rounded-full" style={{width: '80%'}}></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>Response Time (P99)</span>
                <span>{performanceMetrics?.responseTime.p99}ms</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-red-600 h-2 rounded-full" style={{width: '95%'}}></div>
              </div>
            </div>
            
            <div>
              <div className="flex justify-between text-sm text-gray-600 mb-1">
                <span>Throughput</span>
                <span>{performanceMetrics?.throughput} req/min</span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div className="bg-green-600 h-2 rounded-full" style={{width: '70%'}}></div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Error Analytics */}
      <div className="bg-white p-6 rounded-lg shadow-sm border">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">
          Error Analytics
        </h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-900 mb-3">Errors by Type</h4>
            <div className="space-y-2">
              {errorAnalytics?.errorsByType.map((error, index) => (
                <div key={index} className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">{error.type}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-20 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-red-600 h-2 rounded-full" 
                        style={{width: `${error.percentage}%`}}
                      ></div>
                    </div>
                    <span className="text-sm text-gray-600 w-12 text-right">
                      {error.count} ({error.percentage}%)
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-900 mb-3">Errors by Endpoint</h4>
            <div className="space-y-2">
              {errorAnalytics?.errorsByEndpoint.map((error, index) => (
                <div key={index} className="flex items-center justify-between">
                  <span className="text-sm text-gray-700 font-mono">{error.endpoint}</span>
                  <span className="text-sm text-gray-600">{error.count}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Analytics Features */}
      <div className="bg-blue-50 rounded-lg p-6 border border-blue-200">
        <h3 className="text-lg font-semibold text-blue-900 mb-4">
          🚀 Analytics Features
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Real-time Monitoring</h4>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Real-time API usage tracking</li>
              <li>• Performance metrics dashboard</li>
              <li>• Error rate monitoring</li>
              <li>• Usage pattern analysis</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-blue-900 mb-2">Advanced Analytics</h4>
            <ul className="text-sm text-blue-700 space-y-1">
              <li>• Cost optimization insights</li>
              <li>• Custom analytics reports</li>
              <li>• Historical data analysis</li>
              <li>• Predictive analytics</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Competitive Advantages */}
      <div className="bg-green-50 rounded-lg p-6 border border-green-200">
        <h3 className="text-lg font-semibold text-green-900 mb-4">
          🎯 Analytics Advantages Over Drift
        </h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-green-900 mb-2">Drift Protocol</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Basic API testing</li>
              <li>• No analytics dashboard</li>
              <li>• No performance monitoring</li>
              <li>• No error tracking</li>
            </ul>
          </div>
          
          <div>
            <h4 className="font-medium text-green-900 mb-2">QuantDesk</h4>
            <ul className="text-sm text-green-700 space-y-1">
              <li>• Comprehensive analytics dashboard</li>
              <li>• Real-time performance monitoring</li>
              <li>• Advanced error analytics</li>
              <li>• Usage pattern analysis</li>
              <li>• Cost optimization insights</li>
              <li>• Custom reporting</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};
