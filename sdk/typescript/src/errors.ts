/**
 * QuantDesk Protocol Errors
 */

export class QuantDeskError extends Error {
  constructor(
    message: string,
    public code: string,
    public statusCode?: number
  ) {
    super(message)
    this.name = 'QuantDeskError'
  }
}

export class InsufficientBalanceError extends QuantDeskError {
  constructor(required: number, available: number) {
    super(
      `Insufficient balance. Required: ${required}, Available: ${available}`,
      'INSUFFICIENT_BALANCE',
      400
    )
  }
}

export class OrderNotFoundError extends QuantDeskError {
  constructor(orderId: string) {
    super(`Order not found: ${orderId}`, 'ORDER_NOT_FOUND', 404)
  }
}

export class MarketNotFoundError extends QuantDeskError {
  constructor(symbol: string) {
    super(`Market not found: ${symbol}`, 'MARKET_NOT_FOUND', 404)
  }
}

export class InvalidOrderError extends QuantDeskError {
  constructor(message: string) {
    super(`Invalid order: ${message}`, 'INVALID_ORDER', 400)
  }
}

export class RiskLimitExceededError extends QuantDeskError {
  constructor(limit: string, value: number, maxValue: number) {
    super(
      `Risk limit exceeded for ${limit}. Value: ${value}, Max: ${maxValue}`,
      'RISK_LIMIT_EXCEEDED',
      400
    )
  }
}

export class NetworkError extends QuantDeskError {
  constructor(message: string) {
    super(`Network error: ${message}`, 'NETWORK_ERROR', 500)
  }
}

export class AuthenticationError extends QuantDeskError {
  constructor(message: string = 'Authentication required') {
    super(message, 'AUTHENTICATION_ERROR', 401)
  }
}

export class AuthorizationError extends QuantDeskError {
  constructor(message: string = 'Insufficient permissions') {
    super(message, 'AUTHORIZATION_ERROR', 403)
  }
}
