import winston from 'winston';
import { config } from '../config';
import { SecurityUtils } from './security';

/**
 * Secure logging utility for Solana DeFi Trading Intelligence AI
 * Implements structured logging with sensitive data masking
 */

// Custom log format that masks sensitive data
const sensitiveDataMasker = winston.format((info) => {
  const sensitiveKeys = [
    'password',
    'privateKey',
    'apiKey',
    'token',
    'secret',
    'authorization',
    'cookie',
    'session'
  ];

  const maskSensitiveData = (obj: any): any => {
    if (typeof obj !== 'object' || obj === null) {
      return obj;
    }

    if (Array.isArray(obj)) {
      return obj.map(maskSensitiveData);
    }

    const masked: any = {};
    for (const [key, value] of Object.entries(obj)) {
      const lowerKey = key.toLowerCase();
      const isSensitive = sensitiveKeys.some(sensitiveKey => 
        lowerKey.includes(sensitiveKey)
      );

      if (isSensitive && typeof value === 'string') {
        masked[key] = SecurityUtils.maskSensitiveData(value);
      } else if (typeof value === 'object' && value !== null) {
        masked[key] = maskSensitiveData(value);
      } else {
        masked[key] = value;
      }
    }

    return masked;
  };

  return maskSensitiveData(info);
});

// Create logger instance
const logger = winston.createLogger({
  level: config.monitoring.logLevel,
  format: winston.format.combine(
    winston.format.timestamp({
      format: 'YYYY-MM-DD HH:mm:ss'
    }),
    winston.format.errors({ stack: true }),
    sensitiveDataMasker(),
    winston.format.json()
  ),
  defaultMeta: {
    service: 'solana-defi-ai',
    version: process.env['npm_package_version'] || '1.0.0',
    environment: config.dev.nodeEnv
  },
  transports: [
    // Console transport for development
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    
    // File transport for errors
    new winston.transports.File({
      filename: 'logs/error.log',
      level: 'error',
      maxsize: 5242880, // 5MB
      maxFiles: 5
    }),
    
    // File transport for all logs
    new winston.transports.File({
      filename: 'logs/combined.log',
      maxsize: 5242880, // 5MB
      maxFiles: 5
    })
  ],
  
  // Handle uncaught exceptions
  exceptionHandlers: [
    new winston.transports.File({ filename: 'logs/exceptions.log' })
  ],
  
  // Handle unhandled promise rejections
  rejectionHandlers: [
    new winston.transports.File({ filename: 'logs/rejections.log' })
  ]
});

// Add request logging middleware
export const requestLogger = (req: any, res: any, next: any) => {
  const start = Date.now();
  
  res.on('finish', () => {
    const duration = Date.now() - start;
    const logData = {
      method: req.method,
      url: req.url,
      statusCode: res.statusCode,
      duration: `${duration}ms`,
      userAgent: req.get('User-Agent'),
      ip: req.ip,
      timestamp: new Date().toISOString()
    };

    if (res.statusCode >= 400) {
      logger.error('HTTP Request Error', logData);
    } else {
      logger.info('HTTP Request', logData);
    }
  });

  next();
};

// Security event logger
export const securityLogger = {
  loginAttempt: (ip: string, success: boolean, reason?: string) => {
    logger.warn('Login Attempt', {
      ip,
      success,
      reason,
      timestamp: new Date().toISOString()
    });
  },

  apiKeyUsage: (apiKey: string, endpoint: string, success: boolean) => {
    logger.info('API Key Usage', {
      apiKey: SecurityUtils.maskSensitiveData(apiKey),
      endpoint,
      success,
      timestamp: new Date().toISOString()
    });
  },

  suspiciousActivity: (activity: string, details: any) => {
    logger.warn('Suspicious Activity Detected', {
      activity,
      details: SecurityUtils.maskSensitiveData(JSON.stringify(details)),
      timestamp: new Date().toISOString()
    });
  },

  rateLimitExceeded: (ip: string, endpoint: string) => {
    logger.warn('Rate Limit Exceeded', {
      ip,
      endpoint,
      timestamp: new Date().toISOString()
    });
  },

  securityViolation: (violation: string, details: any) => {
    logger.error('Security Violation', {
      violation,
      details: SecurityUtils.maskSensitiveData(JSON.stringify(details)),
      timestamp: new Date().toISOString()
    });
  }
};

// Trading activity logger
export const tradingLogger = {
  walletActivity: (walletAddress: string, activity: string, details: any) => {
    logger.info('Wallet Activity', {
      walletAddress: SecurityUtils.maskSensitiveData(walletAddress),
      activity,
      details,
      timestamp: new Date().toISOString()
    });
  },

  liquidationDetected: (liquidationData: any) => {
    logger.warn('Liquidation Detected', {
      ...liquidationData,
      timestamp: new Date().toISOString()
    });
  },

  priceUpdate: (symbol: string, price: number, change: number) => {
    logger.debug('Price Update', {
      symbol,
      price,
      change,
      timestamp: new Date().toISOString()
    });
  },

  aiQuery: (query: string, response: string, confidence: number) => {
    logger.info('AI Query Processed', {
      query: SecurityUtils.sanitizeInput(query),
      responseLength: response.length,
      confidence,
      timestamp: new Date().toISOString()
    });
  }
};

// System health logger
export const systemLogger = {
  startup: (version: string, environment: string) => {
    logger.info('System Startup', {
      version,
      environment,
      timestamp: new Date().toISOString()
    });
  },

  shutdown: (reason: string) => {
    logger.info('System Shutdown', {
      reason,
      timestamp: new Date().toISOString()
    });
  },

  healthCheck: (status: 'healthy' | 'unhealthy', details: any) => {
    logger.info('Health Check', {
      status,
      details,
      timestamp: new Date().toISOString()
    });
  },

  databaseConnection: (status: 'connected' | 'disconnected', error?: string) => {
    logger.info('Database Connection', {
      status,
      error,
      timestamp: new Date().toISOString()
    });
  },

  externalApiCall: (service: string, endpoint: string, success: boolean, duration: number) => {
    logger.info('External API Call', {
      service,
      endpoint,
      success,
      duration: `${duration}ms`,
      timestamp: new Date().toISOString()
    });
  },

  info: (message: string, data?: any) => {
    logger.info(message, data);
  },

  error: (message: string, error?: Error | any) => {
    logger.error(message, error);
  }
};

// Error logger with context
export const errorLogger = {
  solanaError: (error: Error, context: any) => {
    logger.error('Solana Error', {
      error: error.message,
      stack: error.stack,
      context,
      timestamp: new Date().toISOString()
    });
  },

  aiError: (error: Error, query: string) => {
    logger.error('AI Error', {
      error: error.message,
      stack: error.stack,
      query: SecurityUtils.sanitizeInput(query),
      timestamp: new Date().toISOString()
    });
  },

  databaseError: (error: Error, query: string) => {
    logger.error('Database Error', {
      error: error.message,
      stack: error.stack,
      query: SecurityUtils.sanitizeInput(query),
      timestamp: new Date().toISOString()
    });
  },

  externalApiError: (error: Error, service: string, endpoint: string) => {
    logger.error('External API Error', {
      error: error.message,
      stack: error.stack,
      service,
      endpoint,
      timestamp: new Date().toISOString()
    });
  },

  toolError: (error: Error, toolName: string, context?: any) => {
    logger.error('Tool Error', {
      tool: toolName,
      error: error.message,
      stack: error.stack,
      context,
      timestamp: new Date().toISOString()
    });
  }
};

export default logger;
