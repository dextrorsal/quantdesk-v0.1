/**
 * Centralized Error Message Utility
 * Provides user-friendly error messages for common scenarios
 */

export interface ErrorContext {
  operation?: string;
  symbol?: string;
  amount?: number;
  userAction?: string;
  retryable?: boolean;
}

export class ErrorMessageFormatter {
  /**
   * Format error messages to be user-friendly
   */
  static formatError(error: any, context?: ErrorContext): string {
    // Handle string errors
    if (typeof error === 'string') {
      return this.formatStringError(error, context);
    }

    // Handle Error objects
    if (error instanceof Error) {
      return this.formatErrorObject(error, context);
    }

    // Handle API error responses
    if (error?.code && error?.message) {
      return this.formatApiError(error, context);
    }

    // Fallback
    return 'Something went wrong. Please try again.';
  }

  /**
   * Format string-based errors
   */
  private static formatStringError(error: string, context?: ErrorContext): string {
    const lowerError = error.toLowerCase();

    // Network errors
    if (lowerError.includes('network') || lowerError.includes('connection')) {
      return 'Connection issue. Please check your internet and try again.';
    }

    // Authentication errors
    if (lowerError.includes('auth') || lowerError.includes('unauthorized')) {
      return 'Please connect your wallet to continue.';
    }

    // Validation errors
    if (lowerError.includes('validation') || lowerError.includes('invalid')) {
      return this.formatValidationError(error, context);
    }

    // Trading-specific errors
    if (lowerError.includes('insufficient') || lowerError.includes('balance')) {
      return 'Insufficient balance. Please deposit more funds or reduce your order size.';
    }

    if (lowerError.includes('leverage')) {
      return 'Invalid leverage. Please choose a leverage between 1x and 100x.';
    }

    if (lowerError.includes('size') || lowerError.includes('amount')) {
      return 'Invalid order size. Please enter a positive amount.';
    }

    if (lowerError.includes('price')) {
      return 'Invalid price. Please enter a valid price for your limit order.';
    }

    // Return original error if no pattern matches
    return error;
  }

  /**
   * Format Error objects
   */
  private static formatErrorObject(error: Error, context?: ErrorContext): string {
    const message = error.message.toLowerCase();

    // Solana-specific errors
    if (message.includes('user rejected')) {
      return 'Transaction cancelled. You can try again when ready.';
    }

    if (message.includes('insufficient funds')) {
      return 'Insufficient funds. Please deposit more SOL or reduce your order size.';
    }

    if (message.includes('timeout')) {
      return 'Transaction timed out. Please try again.';
    }

    if (message.includes('blockhash')) {
      return 'Network congestion detected. Please try again in a moment.';
    }

    // WebSocket errors
    if (message.includes('websocket') || message.includes('socket')) {
      return 'Connection lost. We\'re reconnecting automatically...';
    }

    // Return formatted error message
    return this.formatStringError(error.message, context);
  }

  /**
   * Format API error responses
   */
  private static formatApiError(error: any, context?: ErrorContext): string {
    const code = error.code?.toLowerCase();
    const message = error.message || error.error || '';

    switch (code) {
      case 'validation_error':
        return this.formatValidationError(message, context);
      
      case 'authentication_error':
        return 'Please connect your wallet to continue.';
      
      case 'authorization_error':
        return 'You don\'t have permission to perform this action.';
      
      case 'not_found':
        return context?.operation 
          ? `${context.operation} not found. Please try again.`
          : 'Resource not found. Please refresh and try again.';
      
      case 'rate_limit_error':
        return 'Too many requests. Please wait a moment and try again.';
      
      case 'service_unavailable':
        return 'Service temporarily unavailable. Please try again in a few minutes.';
      
      case 'conflict_error':
        return 'This action conflicts with your current state. Please refresh and try again.';
      
      case 'missing_fields':
        return 'Please fill in all required fields.';
      
      case 'invalid_size':
        return 'Order size must be greater than 0.';
      
      case 'invalid_price':
        return 'Price must be a positive number for limit orders.';
      
      case 'invalid_leverage':
        return 'Leverage must be between 1x and 100x.';
      
      case 'invalid_side':
        return 'Please select either Buy or Sell.';
      
      case 'invalid_order_type':
        return 'Please select a valid order type.';
      
      case 'price_unavailable':
        return 'Market data is currently unavailable. Please try again later.';
      
      case 'smart_contract_error':
        return 'Order was created but failed to execute on blockchain. Please check your order status.';
      
      default:
        return message || 'An unexpected error occurred. Please try again.';
    }
  }

  /**
   * Format validation errors with context
   */
  private static formatValidationError(error: string, context?: ErrorContext): string {
    const lowerError = error.toLowerCase();

    if (lowerError.includes('required')) {
      return 'Please fill in all required fields.';
    }

    if (lowerError.includes('email')) {
      return 'Please enter a valid email address.';
    }

    if (lowerError.includes('password')) {
      return 'Password must be at least 8 characters long.';
    }

    if (context?.operation) {
      return `Invalid ${context.operation}. Please check your input and try again.`;
    }

    return 'Please check your input and try again.';
  }

  /**
   * Get retry suggestion for errors
   */
  static getRetrySuggestion(error: any, context?: ErrorContext): string | null {
    const lowerError = typeof error === 'string' ? error.toLowerCase() : 
                      error?.message?.toLowerCase() || '';

    // Network-related errors
    if (lowerError.includes('network') || lowerError.includes('connection') || 
        lowerError.includes('timeout') || lowerError.includes('websocket')) {
      return 'Check your internet connection and try again.';
    }

    // Rate limiting
    if (lowerError.includes('rate limit') || lowerError.includes('too many')) {
      return 'Wait a moment and try again.';
    }

    // Service unavailable
    if (lowerError.includes('service unavailable') || lowerError.includes('maintenance')) {
      return 'Service is temporarily down. Please try again later.';
    }

    // Trading-specific retry suggestions
    if (context?.operation === 'trading' && lowerError.includes('insufficient')) {
      return 'Deposit more funds or reduce your order size.';
    }

    return null;
  }

  /**
   * Get help text for errors
   */
  static getHelpText(error: any, context?: ErrorContext): string | null {
    const lowerError = typeof error === 'string' ? error.toLowerCase() : 
                      error?.message?.toLowerCase() || '';

    // Wallet connection issues
    if (lowerError.includes('wallet') || lowerError.includes('auth')) {
      return 'Make sure your wallet is connected and unlocked.';
    }

    // Trading issues
    if (context?.operation === 'trading') {
      if (lowerError.includes('leverage')) {
        return 'Leverage amplifies both gains and losses. Start with lower leverage.';
      }
      if (lowerError.includes('balance')) {
        return 'You need sufficient balance to cover the position size and fees.';
      }
    }

    return null;
  }
}

/**
 * Error severity levels
 */
export enum ErrorSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

/**
 * Get error severity based on error type
 */
export function getErrorSeverity(error: any): ErrorSeverity {
  const lowerError = typeof error === 'string' ? error.toLowerCase() : 
                    error?.message?.toLowerCase() || '';

  // Critical errors
  if (lowerError.includes('critical') || lowerError.includes('fatal')) {
    return ErrorSeverity.CRITICAL;
  }

  // High severity errors
  if (lowerError.includes('insufficient') || lowerError.includes('unauthorized') ||
      lowerError.includes('timeout') || lowerError.includes('network')) {
    return ErrorSeverity.HIGH;
  }

  // Medium severity errors
  if (lowerError.includes('validation') || lowerError.includes('invalid') ||
      lowerError.includes('conflict')) {
    return ErrorSeverity.MEDIUM;
  }

  // Low severity errors
  return ErrorSeverity.LOW;
}

/**
 * Check if error is retryable
 */
export function isRetryableError(error: any): boolean {
  const lowerError = typeof error === 'string' ? error.toLowerCase() : 
                    error?.message?.toLowerCase() || '';

  return lowerError.includes('network') || 
         lowerError.includes('timeout') || 
         lowerError.includes('connection') ||
         lowerError.includes('rate limit') ||
         lowerError.includes('service unavailable');
}

export default ErrorMessageFormatter;
