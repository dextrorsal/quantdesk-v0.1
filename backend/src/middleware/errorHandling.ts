import { Request, Response, NextFunction } from 'express';
import { Logger } from '../utils/logger';

const logger = new Logger();

// Custom error classes
export class QuantDeskError extends Error {
  public statusCode: number;
  public code: string;
  public details?: any;
  public timestamp: number;
  
  constructor(
    message: string,
    statusCode: number = 500,
    code: string = 'INTERNAL_ERROR',
    details?: any
  ) {
    super(message);
    this.name = 'QuantDeskError';
    this.statusCode = statusCode;
    this.code = code;
    this.details = details;
    this.timestamp = Date.now();
    
    // Maintain proper stack trace
    Error.captureStackTrace(this, QuantDeskError);
  }
}

export class ValidationError extends QuantDeskError {
  constructor(message: string, details?: any) {
    super(message, 400, 'VALIDATION_ERROR', details);
    this.name = 'ValidationError';
  }
}

export class AuthenticationError extends QuantDeskError {
  constructor(message: string = 'Authentication required') {
    super(message, 401, 'AUTHENTICATION_ERROR');
    this.name = 'AuthenticationError';
  }
}

export class AuthorizationError extends QuantDeskError {
  constructor(message: string = 'Insufficient permissions') {
    super(message, 403, 'AUTHORIZATION_ERROR');
    this.name = 'AuthorizationError';
  }
}

export class NotFoundError extends QuantDeskError {
  constructor(resource: string = 'Resource') {
    super(`${resource} not found`, 404, 'NOT_FOUND');
    this.name = 'NotFoundError';
  }
}

export class ConflictError extends QuantDeskError {
  constructor(message: string, details?: any) {
    super(message, 409, 'CONFLICT_ERROR', details);
    this.name = 'ConflictError';
  }
}

export class RateLimitError extends QuantDeskError {
  constructor(message: string = 'Rate limit exceeded', retryAfter?: number) {
    super(message, 429, 'RATE_LIMIT_ERROR', { retryAfter });
    this.name = 'RateLimitError';
  }
}

export class ServiceUnavailableError extends QuantDeskError {
  constructor(message: string = 'Service temporarily unavailable') {
    super(message, 503, 'SERVICE_UNAVAILABLE');
    this.name = 'ServiceUnavailableError';
  }
}

// Error response interface
interface ErrorResponse {
  success: false;
  error: string;
  code: string;
  message: string;
  details?: any;
  timestamp: number;
  request_id?: string;
  path?: string;
  method?: string;
}

// Error handling middleware
export function errorHandler(
  error: Error,
  req: Request,
  res: Response,
  next: NextFunction
): void {
  const requestId = req.headers['x-request-id'] as string || generateRequestId();
  
  // Log error
  logger.error('API Error:', {
    error: error.message,
    stack: error.stack,
    requestId,
    path: req.path,
    method: req.method,
    userAgent: req.get('User-Agent'),
    ip: req.ip
  });
  
  let errorResponse: ErrorResponse;
  
  if (error instanceof QuantDeskError) {
    // Handle custom QuantDesk errors
    errorResponse = {
      success: false,
      error: error.message,
      code: error.code,
      message: error.message,
      details: error.details,
      timestamp: error.timestamp,
      request_id: requestId,
      path: req.path,
      method: req.method
    };
    
    res.status(error.statusCode).json(errorResponse);
  } else if (error.name === 'ValidationError') {
    // Handle validation errors
    errorResponse = {
      success: false,
      error: 'Validation failed',
      code: 'VALIDATION_ERROR',
      message: error.message,
      timestamp: Date.now(),
      request_id: requestId,
      path: req.path,
      method: req.method
    };
    
    res.status(400).json(errorResponse);
  } else if (error.name === 'CastError') {
    // Handle MongoDB cast errors
    errorResponse = {
      success: false,
      error: 'Invalid data format',
      code: 'CAST_ERROR',
      message: 'Invalid data format provided',
      timestamp: Date.now(),
      request_id: requestId,
      path: req.path,
      method: req.method
    };
    
    res.status(400).json(errorResponse);
  } else if (error.name === 'MongoError' || error.name === 'MongooseError') {
    // Handle database errors
    errorResponse = {
      success: false,
      error: 'Database error',
      code: 'DATABASE_ERROR',
      message: 'A database error occurred',
      timestamp: Date.now(),
      request_id: requestId,
      path: req.path,
      method: req.method
    };
    
    res.status(500).json(errorResponse);
  } else if (error.name === 'JsonWebTokenError') {
    // Handle JWT errors
    errorResponse = {
      success: false,
      error: 'Invalid token',
      code: 'JWT_ERROR',
      message: 'Invalid or expired token',
      timestamp: Date.now(),
      request_id: requestId,
      path: req.path,
      method: req.method
    };
    
    res.status(401).json(errorResponse);
  } else if (error.name === 'TokenExpiredError') {
    // Handle expired token errors
    errorResponse = {
      success: false,
      error: 'Token expired',
      code: 'TOKEN_EXPIRED',
      message: 'Token has expired',
      timestamp: Date.now(),
      request_id: requestId,
      path: req.path,
      method: req.method
    };
    
    res.status(401).json(errorResponse);
  } else {
    // Handle unknown errors
    errorResponse = {
      success: false,
      error: 'Internal server error',
      code: 'INTERNAL_ERROR',
      message: process.env.NODE_ENV === 'production' 
        ? 'An unexpected error occurred' 
        : error.message,
      timestamp: Date.now(),
      request_id: requestId,
      path: req.path,
      method: req.method
    };
    
    res.status(500).json(errorResponse);
  }
}

// 404 handler
export function notFoundHandler(req: Request, res: Response): void {
  const requestId = req.headers['x-request-id'] as string || generateRequestId();
  
  logger.warn('404 Not Found:', {
    requestId,
    path: req.path,
    method: req.method,
    userAgent: req.get('User-Agent'),
    ip: req.ip
  });
  
  const errorResponse: ErrorResponse = {
    success: false,
    error: 'Not found',
    code: 'NOT_FOUND',
    message: `Route ${req.method} ${req.path} not found`,
    timestamp: Date.now(),
    request_id: requestId,
    path: req.path,
    method: req.method
  };
  
  res.status(404).json(errorResponse);
}

// Async error wrapper
export function asyncHandler(fn: Function) {
  return (req: Request, res: Response, next: NextFunction) => {
    Promise.resolve(fn(req, res, next)).catch(next);
  };
}

// Request ID generator
function generateRequestId(): string {
  return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// Request ID middleware
export function requestIdMiddleware(req: Request, res: Response, next: NextFunction): void {
  const requestId = req.headers['x-request-id'] as string || generateRequestId();
  req.headers['x-request-id'] = requestId;
  res.set('X-Request-ID', requestId);
  next();
}

// Response time middleware
export function responseTimeMiddleware(req: Request, res: Response, next: NextFunction): void {
  const startTime = Date.now();
  
  res.on('finish', () => {
    const responseTime = Date.now() - startTime;
    
    // Log slow requests
    if (responseTime > 1000) {
      logger.warn('Slow request detected:', {
        requestId: req.headers['x-request-id'],
        path: req.path,
        method: req.method,
        responseTime: `${responseTime}ms`,
        statusCode: res.statusCode
      });
    }
  });
  
  next();
}

// Validation error formatter
export function formatValidationError(error: any): ValidationError {
  if (error.details && Array.isArray(error.details)) {
    const details = error.details.map((detail: any) => ({
      field: detail.path.join('.'),
      message: detail.message,
      value: detail.context?.value
    }));
    
    return new ValidationError('Validation failed', details);
  }
  
  return new ValidationError(error.message || 'Validation failed');
}

// Error monitoring and alerting
export class ErrorMonitor {
  private static instance: ErrorMonitor;
  private errorCounts: Map<string, number> = new Map();
  private lastAlertTime: Map<string, number> = new Map();
  
  private constructor() {}
  
  public static getInstance(): ErrorMonitor {
    if (!ErrorMonitor.instance) {
      ErrorMonitor.instance = new ErrorMonitor();
    }
    return ErrorMonitor.instance;
  }
  
  public trackError(error: QuantDeskError): void {
    const key = `${error.code}:${error.statusCode}`;
    const count = this.errorCounts.get(key) || 0;
    this.errorCounts.set(key, count + 1);
    
    // Alert on high error rates
    if (count > 0 && count % 10 === 0) {
      this.alertHighErrorRate(key, count);
    }
  }
  
  private alertHighErrorRate(key: string, count: number): void {
    const now = Date.now();
    const lastAlert = this.lastAlertTime.get(key) || 0;
    
    // Don't alert more than once per minute
    if (now - lastAlert < 60000) return;
    
    logger.error(`High error rate detected: ${key} (${count} errors)`);
    this.lastAlertTime.set(key, now);
    
    // In production, send to monitoring service
    // await sendToMonitoringService(key, count);
  }
  
  public getErrorStats(): any {
    return {
      error_counts: Object.fromEntries(this.errorCounts),
      last_alert_times: Object.fromEntries(this.lastAlertTime)
    };
  }
}

export const errorMonitor = ErrorMonitor.getInstance();

export default {
  QuantDeskError,
  ValidationError,
  AuthenticationError,
  AuthorizationError,
  NotFoundError,
  ConflictError,
  RateLimitError,
  ServiceUnavailableError,
  errorHandler,
  notFoundHandler,
  asyncHandler,
  requestIdMiddleware,
  responseTimeMiddleware,
  formatValidationError,
  errorMonitor
};
