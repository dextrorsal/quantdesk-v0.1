const express = require('express');
const cors = require('cors');
const { createProxyMiddleware } = require('http-proxy-middleware');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json());

// Service URLs
const ML_SERVICE_URL = process.env.ML_SERVICE_URL || 'http://ml-service:8000';
const TRADING_ENGINE_URL = process.env.TRADING_ENGINE_URL || 'http://trading-engine:8080';

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    service: 'api-gateway',
    timestamp: new Date().toISOString()
  });
});

// Proxy to ML Service
app.use('/api/ml', createProxyMiddleware({
  target: ML_SERVICE_URL,
  changeOrigin: true,
  pathRewrite: {
    '^/api/ml': '', // Remove /api/ml prefix when forwarding to ML service
  },
  onError: (err, req, res) => {
    console.error('ML Service proxy error:', err.message);
    res.status(502).json({ error: 'ML Service unavailable' });
  }
}));

// Proxy to Trading Engine (when implemented)
app.use('/api/trading', createProxyMiddleware({
  target: TRADING_ENGINE_URL,
  changeOrigin: true,
  pathRewrite: {
    '^/api/trading': '',
  },
  onError: (err, req, res) => {
    console.error('Trading Engine proxy error:', err.message);
    res.status(502).json({ error: 'Trading Engine unavailable' });
  }
}));

// Service discovery endpoint
app.get('/api/services', async (req, res) => {
  const services = {
    ml_service: { url: ML_SERVICE_URL, status: 'unknown' },
    trading_engine: { url: TRADING_ENGINE_URL, status: 'unknown' }
  };

  // Check ML Service health
  try {
    const mlResponse = await axios.get(`${ML_SERVICE_URL}/health`, { timeout: 5000 });
    services.ml_service.status = mlResponse.data.status;
  } catch (error) {
    services.ml_service.status = 'unhealthy';
  }

  // Check Trading Engine health (when implemented)
  try {
    const tradingResponse = await axios.get(`${TRADING_ENGINE_URL}/health`, { timeout: 5000 });
    services.trading_engine.status = tradingResponse.data.status;
  } catch (error) {
    services.trading_engine.status = 'unhealthy';
  }

  res.json(services);
});

// Root endpoint
app.get('/', (req, res) => {
  res.json({
    service: 'Quantify API Gateway',
    version: '1.0.0',
    endpoints: {
      health: '/health',
      ml_service: '/api/ml/*',
      trading_engine: '/api/trading/*',
      services: '/api/services'
    }
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 API Gateway running on port ${PORT}`);
  console.log(`📡 ML Service: ${ML_SERVICE_URL}`);
  console.log(`⚡ Trading Engine: ${TRADING_ENGINE_URL}`);
});
