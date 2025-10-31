import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Logger } from '../utils/logger';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error?: Error;
  errorInfo?: ErrorInfo;
}

/**
 * Collateral Error Boundary
 * 
 * CRITICAL: This component prevents the entire app from crashing when
 * collateral operations fail, providing graceful error handling and
 * user-friendly error messages.
 */
export class CollateralErrorBoundary extends Component<Props, State> {
  private logger = new Logger();

  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log the error for debugging
    this.logger.error('Collateral Error Boundary caught an error:', error, errorInfo);
    
    // Update state with error details
    this.setState({
      error,
      errorInfo
    });

    // Call custom error handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    // Log to external service (e.g., Sentry, LogRocket)
    this.logToExternalService(error, errorInfo);
  }

  private logToExternalService(error: Error, errorInfo: ErrorInfo) {
    try {
      // This would integrate with external logging services
      console.error('External logging:', {
        error: error.message,
        stack: error.stack,
        componentStack: errorInfo.componentStack,
        timestamp: new Date().toISOString()
      });
    } catch (loggingError) {
      this.logger.error('Failed to log to external service:', loggingError);
    }
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  private handleReportError = () => {
    // This would open a support ticket or error reporting form
    console.log('Reporting error to support team...');
  };

  render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default error UI
      return (
        <div className="collateral-error-boundary">
          <div className="error-container">
            <div className="error-icon">⚠️</div>
            <h2 className="error-title">Collateral Operation Error</h2>
            <p className="error-message">
              Something went wrong with your collateral operation. Don't worry, your funds are safe.
            </p>
            
            <div className="error-details">
              <details>
                <summary>Technical Details</summary>
                <pre className="error-stack">
                  {this.state.error?.message}
                  {this.state.error?.stack}
                </pre>
              </details>
            </div>

            <div className="error-actions">
              <button 
                onClick={this.handleRetry}
                className="retry-button"
              >
                Try Again
              </button>
              <button 
                onClick={this.handleReportError}
                className="report-button"
              >
                Report Issue
              </button>
            </div>

            <div className="error-help">
              <p>
                If this problem persists, please contact our support team.
                Your transaction may still be processing on-chain.
              </p>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Deposit Error Boundary
 * Specific error boundary for deposit operations
 */
export class DepositErrorBoundary extends Component<Props, State> {
  private logger = new Logger();

  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.logger.error('Deposit Error Boundary caught an error:', error, errorInfo);
    
    this.setState({ error, errorInfo });
    
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="deposit-error-boundary">
          <div className="error-container">
            <div className="error-icon">💰</div>
            <h2 className="error-title">Deposit Error</h2>
            <p className="error-message">
              There was an issue processing your deposit. Your funds are safe and the transaction may still be processing.
            </p>
            
            <div className="error-actions">
              <button onClick={this.handleRetry} className="retry-button">
                Retry Deposit
              </button>
            </div>

            <div className="error-help">
              <p>
                <strong>What to do:</strong>
                <br />
                1. Check your wallet for pending transactions
                <br />
                2. Wait a few minutes and refresh the page
                <br />
                3. Contact support if the issue persists
              </p>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Withdrawal Error Boundary
 * Specific error boundary for withdrawal operations
 */
export class WithdrawalErrorBoundary extends Component<Props, State> {
  private logger = new Logger();

  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.logger.error('Withdrawal Error Boundary caught an error:', error, errorInfo);
    
    this.setState({ error, errorInfo });
    
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="withdrawal-error-boundary">
          <div className="error-container">
            <div className="error-icon">💸</div>
            <h2 className="error-title">Withdrawal Error</h2>
            <p className="error-message">
              There was an issue processing your withdrawal. Your funds remain in your account and are safe.
            </p>
            
            <div className="error-actions">
              <button onClick={this.handleRetry} className="retry-button">
                Retry Withdrawal
              </button>
            </div>

            <div className="error-help">
              <p>
                <strong>Security Notice:</strong>
                <br />
                • Your funds are protected by our security systems
                <br />
                • Withdrawals may be delayed for security verification
                <br />
                • Contact support if you need immediate assistance
              </p>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

/**
 * Balance Display Error Boundary
 * Specific error boundary for balance display components
 */
export class BalanceDisplayErrorBoundary extends Component<Props, State> {
  private logger = new Logger();

  constructor(props: Props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.logger.error('Balance Display Error Boundary caught an error:', error, errorInfo);
    
    this.setState({ error, errorInfo });
    
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }
  }

  private handleRetry = () => {
    this.setState({ hasError: false, error: undefined, errorInfo: undefined });
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="balance-error-boundary">
          <div className="error-container">
            <div className="error-icon">📊</div>
            <h2 className="error-title">Balance Display Error</h2>
            <p className="error-message">
              Unable to load your balance information. Please refresh the page or try again.
            </p>
            
            <div className="error-actions">
              <button onClick={this.handleRetry} className="retry-button">
                Refresh Balance
              </button>
            </div>

            <div className="error-help">
              <p>
                <strong>Note:</strong> This is a display issue only. Your actual balance is safe and accurate.
              </p>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// CSS styles for error boundaries
export const errorBoundaryStyles = `
.collateral-error-boundary,
.deposit-error-boundary,
.withdrawal-error-boundary,
.balance-error-boundary {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 200px;
  padding: 20px;
  background: var(--background-secondary);
  border-radius: 8px;
  border: 1px solid var(--border-color);
}

.error-container {
  text-align: center;
  max-width: 500px;
}

.error-icon {
  font-size: 48px;
  margin-bottom: 16px;
}

.error-title {
  color: var(--text-primary);
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 12px;
}

.error-message {
  color: var(--text-secondary);
  font-size: 16px;
  line-height: 1.5;
  margin-bottom: 20px;
}

.error-details {
  margin: 20px 0;
}

.error-details summary {
  cursor: pointer;
  color: var(--text-secondary);
  font-size: 14px;
}

.error-stack {
  background: var(--background-tertiary);
  padding: 12px;
  border-radius: 4px;
  font-family: monospace;
  font-size: 12px;
  text-align: left;
  margin-top: 8px;
  overflow-x: auto;
}

.error-actions {
  display: flex;
  gap: 12px;
  justify-content: center;
  margin: 20px 0;
}

.retry-button,
.report-button {
  padding: 10px 20px;
  border-radius: 6px;
  border: none;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.retry-button {
  background: var(--primary-color);
  color: white;
}

.retry-button:hover {
  background: var(--primary-hover);
}

.report-button {
  background: var(--secondary-color);
  color: var(--text-primary);
}

.report-button:hover {
  background: var(--secondary-hover);
}

.error-help {
  margin-top: 20px;
  padding: 16px;
  background: var(--background-tertiary);
  border-radius: 6px;
  text-align: left;
}

.error-help p {
  color: var(--text-secondary);
  font-size: 14px;
  line-height: 1.5;
  margin: 0;
}
`;

export default CollateralErrorBoundary;
