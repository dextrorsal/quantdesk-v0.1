/**
 * Documentation Configuration
 * Centralized configuration for documentation URLs
 */

// Environment-based documentation URL configuration
export const DOCS_CONFIG = {
  // Local development
  local: {
    baseUrl: 'http://localhost:8080',
    technicalPortfolio: 'http://localhost:8080/html/docs_TECHNICAL_EVOLUTION_PORTFOLIO.html',
    performanceMetrics: 'http://localhost:8080/html/docs_PERFORMANCE_METRICS.html',
    gettingStarted: 'http://localhost:8080/html/docs_getting-started/README.html',
    apiDocs: 'http://localhost:8080/html/docs_api/API.html',
  },
  
  // Production (when deployed)
  production: {
    baseUrl: 'https://docs.quantdesk.app',
    technicalPortfolio: 'https://docs.quantdesk.app/html/docs_TECHNICAL_EVOLUTION_PORTFOLIO.html',
    performanceMetrics: 'https://docs.quantdesk.app/html/docs_PERFORMANCE_METRICS.html',
    gettingStarted: 'https://docs.quantdesk.app/html/docs_getting-started/README.html',
    apiDocs: 'https://docs.quantdesk.app/html/docs_api/API.html',
  }
}

// Determine current environment
const isDevelopment = import.meta.env.DEV

// Export current configuration
export const DOCS_URLS = isDevelopment ? DOCS_CONFIG.local : DOCS_CONFIG.production

// Helper function to get documentation URL
export const getDocsUrl = (path: keyof typeof DOCS_CONFIG.local) => {
  return DOCS_URLS[path]
}

// GitHub repository URL
export const GITHUB_URL = 'https://github.com/dextrorsal/quantdesk'
