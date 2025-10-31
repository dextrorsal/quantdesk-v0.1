// MIKEY AI Monitoring Dashboard Component
// Real-time monitoring of LLM providers, usage, and performance

import React, { useState, useEffect } from 'react';
// import { mikeyAI, LLMStatus } from '../services/mikeyAI';

interface LLMStatus {
  name: string;
  provider: string;
  status: 'healthy' | 'degraded' | 'unhealthy' | string;
  queries?: number;
  cost?: number;
  avgResponseTime?: number;
  lastUsed?: string | number | Date | null;
  notes?: string;
}

interface UsageStats {
  totalQueries: number;
  totalCost: number;
  averageResponseTime: number;
  providerUsage: Record<string, { queries: number; cost: number; avgTime: number }>;
}

export const MikeyAIMonitoringDashboard: React.FC = () => {
  const [llmStatus, setLlmStatus] = useState<LLMStatus[]>([]);
  const [usageStats, setUsageStats] = useState<UsageStats | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [autoRefresh, setAutoRefresh] = useState(true);

  // Fetch monitoring data via backend proxy to avoid CORS
  const fetchMonitoringData = async () => {
    try {
      setIsLoading(true);
      // LLM status
      const statusRes = await fetch('/api/mikey/llm/status');
      if (statusRes.ok) {
        const statusJson = await statusRes.json();
        const items = Array.isArray(statusJson?.data) ? statusJson.data : (statusJson?.providers || []);
        setLlmStatus(items || []);
      } else {
        setLlmStatus([]);
      }

      // Usage stats
      const usageRes = await fetch('/api/mikey/llm/usage');
      if (usageRes.ok) {
        const usageJson = await usageRes.json();
        setUsageStats(usageJson?.data || null);
      } else {
        setUsageStats(null);
      }

      setError('');
    } catch (err) {
      setError('Failed to fetch monitoring data');
      console.error('Monitoring error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchMonitoringData();
    if (autoRefresh) {
      const interval = setInterval(fetchMonitoringData, 10000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-green-500';
      case 'degraded': return 'bg-yellow-500';
      case 'unhealthy': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const formatCurrency = (amount: number) => new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(amount || 0);
  const formatDuration = (ms: number) => (ms < 1000 ? `${(ms || 0).toFixed(0)}ms` : `${((ms || 0) / 1000).toFixed(2)}s`);

  return (
    <div className="p-6 bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 text-white min-h-screen">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold mb-2">🤖 MIKEY AI Monitoring Dashboard</h1>
            <p className="text-gray-400">Real-time LLM provider performance and usage analytics</p>
          </div>
          <div className="flex items-center gap-4">
            <button onClick={() => setAutoRefresh(!autoRefresh)} className={`px-4 py-2 rounded-lg transition-colors ${autoRefresh ? 'bg-green-600 hover:bg-green-700' : 'bg-gray-700 hover:bg-gray-600'}`}>
              {autoRefresh ? '🟢 Auto-Refresh ON' : '⚫ Auto-Refresh OFF'}
            </button>
            <button onClick={fetchMonitoringData} className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg transition-colors">🔄 Refresh Now</button>
          </div>
        </div>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-900/50 border border-red-500 rounded-lg"><p className="text-red-300">❌ {error}</p></div>
      )}

      {/* LLM Provider Status Grid */}
      <div className="mb-8">
        <h2 className="text-xl font-semibold mb-4">📊 LLM Provider Status</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {llmStatus.map((provider) => (
            <div
              key={provider.name}
              className="rounded-lg p-4 transition-all"
              style={{
                background: 'var(--bg-secondary)',
                border: '1px solid var(--bg-tertiary)',
                color: 'var(--text-primary)'
              }}
            >
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-lg">{provider.name}</h3>
                <div className="flex items-center gap-2">
                  <div className={`w-3 h-3 rounded-full ${getStatusColor(provider.status)}`}></div>
                  <span className="text-xs text-gray-400 capitalize">{provider.status}</span>
                </div>
              </div>

              <div className="space-y-2 text-sm">
                <div className="flex justify-between"><span className="text-gray-400">Provider:</span><span className="text-blue-400">{provider.provider}</span></div>
                <div className="flex justify-between"><span className="text-gray-400">Queries:</span><span>{Number(provider.queries || 0).toLocaleString()}</span></div>
                <div className="flex justify-between"><span className="text-gray-400">Total Cost:</span><span className="text-green-400">{formatCurrency(provider.cost || 0)}</span></div>
                {provider.avgResponseTime != null && (
                  <div className="flex justify-between"><span className="text-gray-400">Avg Response:</span><span>{formatDuration(provider.avgResponseTime)}</span></div>
                )}
                <div className="flex justify-between"><span className="text-gray-400">Last Used:</span><span className="text-gray-500">{provider.lastUsed ? new Date(provider.lastUsed as any).toLocaleTimeString() : 'Never'}</span></div>
              </div>

              {provider.notes && (
                <div className="mt-3 pt-3" style={{ borderTop: '1px solid var(--bg-tertiary)' }}>
                  <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>
                    {provider.notes}
                  </p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Usage Statistics */}
      {usageStats && (
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4">📈 Usage Statistics</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="rounded-lg p-6" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--bg-tertiary)' }}>
              <div className="flex items-center justify-between mb-2"><span className="text-gray-400">Total Queries</span><span className="text-2xl">📊</span></div>
              <div className="text-3xl font-bold" style={{ color: 'var(--primary-400)' }}>{Number(usageStats.totalQueries || 0).toLocaleString()}</div>
            </div>
            <div className="rounded-lg p-6" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--bg-tertiary)' }}>
              <div className="flex items-center justify-between mb-2"><span className="text-gray-400">Total Cost</span><span className="text-2xl">💰</span></div>
              <div className="text-3xl font-bold" style={{ color: 'var(--success-500)' }}>{formatCurrency(usageStats.totalCost || 0)}</div>
            </div>
            <div className="rounded-lg p-6" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--bg-tertiary)' }}>
              <div className="flex items-center justify-between mb-2"><span className="text-gray-400">Avg Response Time</span><span className="text-2xl">⚡</span></div>
              <div className="text-3xl font-bold" style={{ color: 'var(--warning-400)' }}>{formatDuration(usageStats.averageResponseTime || 0)}</div>
            </div>
          </div>
        </div>
      )}

      {/* Provider Usage Breakdown */}
      {usageStats?.providerUsage && (
        <div className="mb-8">
          <h2 className="text-xl font-semibold mb-4">🔍 Provider Usage Breakdown</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {Object.entries(usageStats.providerUsage).map(([providerName, stats]) => (
              <div key={providerName} className="rounded-lg p-4" style={{ background: 'var(--bg-secondary)', border: '1px solid var(--bg-tertiary)' }}>
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold">{providerName}</h3>
                  <span className="text-xs text-gray-400">Usage</span>
                </div>
                <div className="space-y-1 text-sm">
                  <div className="flex justify-between"><span className="text-gray-400">Queries</span><span>{Number(stats.queries || 0).toLocaleString()}</span></div>
                  <div className="flex justify-between"><span className="text-gray-400">Cost</span><span style={{ color: 'var(--success-500)' }}>{formatCurrency(stats.cost || 0)}</span></div>
                  <div className="flex justify-between"><span className="text-gray-400">Avg Time</span><span>{formatDuration(stats.avgTime || 0)}</span></div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {isLoading && <div className="text-gray-400">Loading...</div>}
    </div>
  );
}

