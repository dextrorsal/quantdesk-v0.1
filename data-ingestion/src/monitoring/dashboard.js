const express = require('express');
const { redis, logger } = require('../config');

class MonitoringDashboard {
  constructor() {
    this.app = express();
    this.port = process.env.METRICS_PORT || 3003;
    this.isRunning = false;
  }

  async start() {
    if (this.isRunning) {
      logger.warn('Monitoring dashboard already running');
      return;
    }

    logger.info('Starting monitoring dashboard...');
    this.isRunning = true;

    this.setupRoutes();
    this.startMetricsCollection();

    this.app.listen(this.port, () => {
      logger.info(`Monitoring dashboard running on port ${this.port}`);
    });
  }

  setupRoutes() {
    // Health check endpoint
    this.app.get('/health', async (req, res) => {
      try {
        const redisStatus = await this.checkRedisHealth();
        const dbStatus = await this.checkDatabaseHealth();
        
        res.json({
          status: 'healthy',
          timestamp: new Date().toISOString(),
          services: {
            redis: redisStatus,
            database: dbStatus
          }
        });
      } catch (error) {
        res.status(500).json({
          status: 'unhealthy',
          error: error.message
        });
      }
    });

    // Stream metrics endpoint
    this.app.get('/metrics/streams', async (req, res) => {
      try {
        const streamMetrics = await this.getStreamMetrics();
        res.json(streamMetrics);
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });

    // Consumer group metrics endpoint
    this.app.get('/metrics/consumers', async (req, res) => {
      try {
        const consumerMetrics = await this.getConsumerMetrics();
        res.json(consumerMetrics);
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });

    // System metrics endpoint
    this.app.get('/metrics/system', async (req, res) => {
      try {
        const systemMetrics = await this.getSystemMetrics();
        res.json(systemMetrics);
      } catch (error) {
        res.status(500).json({ error: error.message });
      }
    });

    // Dashboard HTML
    this.app.get('/', (req, res) => {
      res.send(this.getDashboardHTML());
    });
  }

  async checkRedisHealth() {
    try {
      await redis.ping();
      return { status: 'healthy', latency: Date.now() };
    } catch (error) {
      return { status: 'unhealthy', error: error.message };
    }
  }

  async checkDatabaseHealth() {
    try {
      const { dbPool } = require('../config');
      const start = Date.now();
      await dbPool.query('SELECT 1');
      const latency = Date.now() - start;
      return { status: 'healthy', latency };
    } catch (error) {
      return { status: 'unhealthy', error: error.message };
    }
  }

  async getStreamMetrics() {
    try {
      const streams = [
        'ticks.raw',
        'whales.raw',
        'news.raw',
        'user.events',
        'system.events'
      ];

      const metrics = {};
      
      for (const stream of streams) {
        try {
          const info = await redis.xinfo('STREAM', stream);
          metrics[stream] = {
            length: info[1], // Length
            firstEntry: info[3], // First entry ID
            lastEntry: info[5], // Last entry ID
            groups: info[7] // Number of consumer groups
          };
        } catch (error) {
          metrics[stream] = { error: error.message };
        }
      }

      return metrics;
    } catch (error) {
      throw error;
    }
  }

  async getConsumerMetrics() {
    try {
      const streams = [
        'ticks.raw',
        'whales.raw',
        'news.raw',
        'system.events'
      ];

      const metrics = {};
      
      for (const stream of streams) {
        try {
          const groups = await redis.xinfo('GROUPS', stream);
          metrics[stream] = { groups: groups.length };
          
          for (const group of groups) {
            const groupName = group[1];
            const consumers = await redis.xinfo('CONSUMERS', stream, groupName);
            metrics[stream][groupName] = {
              consumers: consumers.length,
              pending: group[3] // Pending messages
            };
          }
        } catch (error) {
          metrics[stream] = { error: error.message };
        }
      }

      return metrics;
    } catch (error) {
      throw error;
    }
  }

  async getSystemMetrics() {
    try {
      const info = await redis.info();
      const lines = info.split('\r\n');
      const metrics = {};
      
      for (const line of lines) {
        if (line.includes(':')) {
          const [key, value] = line.split(':');
          metrics[key] = value;
        }
      }

      return {
        redis: {
          used_memory: metrics.used_memory_human,
          connected_clients: metrics.connected_clients,
          total_commands_processed: metrics.total_commands_processed,
          keyspace_hits: metrics.keyspace_hits,
          keyspace_misses: metrics.keyspace_misses
        },
        system: {
          uptime: process.uptime(),
          memory_usage: process.memoryUsage(),
          cpu_usage: process.cpuUsage()
        }
      };
    } catch (error) {
      throw error;
    }
  }

  startMetricsCollection() {
    // Collect metrics every 30 seconds
    setInterval(async () => {
      try {
        const streamMetrics = await this.getStreamMetrics();
        const consumerMetrics = await this.getConsumerMetrics();
        
        logger.info('Pipeline Metrics:', {
          streams: streamMetrics,
          consumers: consumerMetrics,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        logger.error('Error collecting metrics:', error);
      }
    }, 30000);
  }

  getDashboardHTML() {
    return `
<!DOCTYPE html>
<html>
<head>
    <title>QuantDesk Data Pipeline Monitor</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; }
        .card { background: white; padding: 20px; margin: 10px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #e8f4fd; border-radius: 4px; }
        .status-healthy { color: green; }
        .status-unhealthy { color: red; }
        .refresh-btn { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; }
        .refresh-btn:hover { background: #0056b3; }
    </style>
</head>
<body>
    <div class="container">
        <h1>QuantDesk Data Pipeline Monitor</h1>
        
        <div class="card">
            <h2>System Health</h2>
            <div id="health-status">Loading...</div>
            <button class="refresh-btn" onclick="refreshHealth()">Refresh</button>
        </div>
        
        <div class="card">
            <h2>Stream Metrics</h2>
            <div id="stream-metrics">Loading...</div>
            <button class="refresh-btn" onclick="refreshStreams()">Refresh</button>
        </div>
        
        <div class="card">
            <h2>Consumer Metrics</h2>
            <div id="consumer-metrics">Loading...</div>
            <button class="refresh-btn" onclick="refreshConsumers()">Refresh</button>
        </div>
        
        <div class="card">
            <h2>System Metrics</h2>
            <div id="system-metrics">Loading...</div>
            <button class="refresh-btn" onclick="refreshSystem()">Refresh</button>
        </div>
    </div>

    <script>
        async function fetchData(url) {
            const response = await fetch(url);
            return await response.json();
        }

        async function refreshHealth() {
            const data = await fetchData('/health');
            document.getElementById('health-status').innerHTML = 
                '<div class="metric">Status: <span class="status-' + data.status + '">' + data.status + '</span></div>' +
                '<div class="metric">Redis: ' + data.services.redis.status + '</div>' +
                '<div class="metric">Database: ' + data.services.database.status + '</div>';
        }

        async function refreshStreams() {
            const data = await fetchData('/metrics/streams');
            let html = '';
            for (const [stream, metrics] of Object.entries(data)) {
                html += '<div class="metric">' + stream + ': ' + (metrics.length || 'N/A') + ' messages</div>';
            }
            document.getElementById('stream-metrics').innerHTML = html;
        }

        async function refreshConsumers() {
            const data = await fetchData('/metrics/consumers');
            let html = '';
            for (const [stream, metrics] of Object.entries(data)) {
                html += '<div class="metric">' + stream + ': ' + (metrics.groups || 0) + ' groups</div>';
            }
            document.getElementById('consumer-metrics').innerHTML = html;
        }

        async function refreshSystem() {
            const data = await fetchData('/metrics/system');
            document.getElementById('system-metrics').innerHTML = 
                '<div class="metric">Memory: ' + data.system.memory_usage.heapUsed + ' bytes</div>' +
                '<div class="metric">Uptime: ' + Math.floor(data.system.uptime) + ' seconds</div>' +
                '<div class="metric">Redis Memory: ' + data.redis.used_memory + '</div>';
        }

        // Auto-refresh every 30 seconds
        setInterval(() => {
            refreshHealth();
            refreshStreams();
            refreshConsumers();
            refreshSystem();
        }, 30000);

        // Initial load
        refreshHealth();
        refreshStreams();
        refreshConsumers();
        refreshSystem();
    </script>
</body>
</html>
    `;
  }
}

// Start dashboard if run directly
if (require.main === module) {
  const dashboard = new MonitoringDashboard();
  
  dashboard.start().catch(error => {
    logger.error('Failed to start monitoring dashboard:', error);
    process.exit(1);
  });
}

module.exports = MonitoringDashboard;
