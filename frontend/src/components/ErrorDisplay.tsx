import React from 'react';
import { AlertCircle, RefreshCw, HelpCircle, X } from 'lucide-react';
import { ErrorMessageFormatter, ErrorSeverity, getErrorSeverity, isRetryableError, ErrorContext } from '../utils/errorMessages';

export interface ErrorDisplayProps {
  error: any;
  context?: ErrorContext;
  onRetry?: () => void;
  onDismiss?: () => void;
  showDetails?: boolean;
  className?: string;
  compact?: boolean;
}

export const ErrorDisplay: React.FC<ErrorDisplayProps> = ({
  error,
  context,
  onRetry,
  onDismiss,
  showDetails = false,
  className = '',
  compact = false
}) => {
  const [showHelp, setShowHelp] = React.useState(false);
  
  const formattedError = ErrorMessageFormatter.formatError(error, context);
  const retrySuggestion = ErrorMessageFormatter.getRetrySuggestion(error, context);
  const helpText = ErrorMessageFormatter.getHelpText(error, context);
  const severity = getErrorSeverity(error);
  const canRetry = isRetryableError(error) && onRetry;

  const getSeverityStyles = (severity: ErrorSeverity) => {
    switch (severity) {
      case ErrorSeverity.CRITICAL:
        return {
          container: 'bg-red-900/20 border-red-500/50',
          icon: 'text-red-400',
          title: 'text-red-400',
          message: 'text-red-300',
          button: 'bg-red-600 hover:bg-red-700 text-white'
        };
      case ErrorSeverity.HIGH:
        return {
          container: 'bg-orange-900/20 border-orange-500/50',
          icon: 'text-orange-400',
          title: 'text-orange-400',
          message: 'text-orange-300',
          button: 'bg-orange-600 hover:bg-orange-700 text-white'
        };
      case ErrorSeverity.MEDIUM:
        return {
          container: 'bg-yellow-900/20 border-yellow-500/50',
          icon: 'text-yellow-400',
          title: 'text-yellow-400',
          message: 'text-yellow-300',
          button: 'bg-yellow-600 hover:bg-yellow-700 text-white'
        };
      default:
        return {
          container: 'bg-blue-900/20 border-blue-500/50',
          icon: 'text-blue-400',
          title: 'text-blue-400',
          message: 'text-blue-300',
          button: 'bg-blue-600 hover:bg-blue-700 text-white'
        };
    }
  };

  const styles = getSeverityStyles(severity);

  if (compact) {
    return (
      <div className={`flex items-center space-x-2 p-2 rounded border ${styles.container} ${className}`}>
        <AlertCircle className={`w-4 h-4 ${styles.icon}`} />
        <span className={`text-sm ${styles.message}`}>{formattedError}</span>
        {canRetry && (
          <button
            onClick={onRetry}
            className="p-1 hover:bg-gray-700 rounded transition-colors"
            title="Retry"
          >
            <RefreshCw className="w-3 h-3 text-gray-400" />
          </button>
        )}
        {onDismiss && (
          <button
            onClick={onDismiss}
            className="p-1 hover:bg-gray-700 rounded transition-colors"
            title="Dismiss"
          >
            <X className="w-3 h-3 text-gray-400" />
          </button>
        )}
      </div>
    );
  }

  return (
    <div className={`p-4 rounded-lg border ${styles.container} ${className}`}>
      <div className="flex items-start space-x-3">
        <AlertCircle className={`w-5 h-5 ${styles.icon} mt-0.5 flex-shrink-0`} />
        
        <div className="flex-1 min-w-0">
          <div className="flex items-center justify-between">
            <h3 className={`text-sm font-medium ${styles.title}`}>
              {severity === ErrorSeverity.CRITICAL ? 'Critical Error' :
               severity === ErrorSeverity.HIGH ? 'Error' :
               severity === ErrorSeverity.MEDIUM ? 'Warning' : 'Notice'}
            </h3>
            
            <div className="flex items-center space-x-2">
              {helpText && (
                <button
                  onClick={() => setShowHelp(!showHelp)}
                  className="p-1 hover:bg-gray-700 rounded transition-colors"
                  title="Get help"
                >
                  <HelpCircle className="w-4 h-4 text-gray-400" />
                </button>
              )}
              {onDismiss && (
                <button
                  onClick={onDismiss}
                  className="p-1 hover:bg-gray-700 rounded transition-colors"
                  title="Dismiss"
                >
                  <X className="w-4 h-4 text-gray-400" />
                </button>
              )}
            </div>
          </div>
          
          <p className={`text-sm mt-1 ${styles.message}`}>
            {formattedError}
          </p>
          
          {retrySuggestion && (
            <p className="text-xs text-gray-400 mt-2">
              💡 {retrySuggestion}
            </p>
          )}
          
          {showHelp && helpText && (
            <div className="mt-3 p-3 bg-gray-800/50 rounded border border-gray-600">
              <p className="text-xs text-gray-300">
                <strong>Help:</strong> {helpText}
              </p>
            </div>
          )}
          
          {showDetails && (
            <details className="mt-3">
              <summary className="text-xs text-gray-400 cursor-pointer hover:text-gray-300">
                Technical Details
              </summary>
              <pre className="mt-2 text-xs text-gray-500 bg-gray-900/50 p-2 rounded overflow-x-auto">
                {typeof error === 'string' ? error : JSON.stringify(error, null, 2)}
              </pre>
            </details>
          )}
          
          {canRetry && (
            <div className="mt-3 flex space-x-2">
              <button
                onClick={onRetry}
                className={`px-3 py-1.5 text-xs font-medium rounded transition-colors ${styles.button}`}
              >
                <RefreshCw className="w-3 h-3 inline mr-1" />
                Try Again
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * Inline error display for forms
 */
export const InlineError: React.FC<{
  error: any;
  context?: ErrorContext;
  className?: string;
}> = ({ error, context, className = '' }) => {
  const formattedError = ErrorMessageFormatter.formatError(error, context);
  
  return (
    <div className={`flex items-center space-x-1 text-red-400 text-sm ${className}`}>
      <AlertCircle className="w-3 h-3 flex-shrink-0" />
      <span>{formattedError}</span>
    </div>
  );
};

/**
 * Toast-style error notification
 */
export const ErrorToast: React.FC<{
  error: any;
  context?: ErrorContext;
  onDismiss: () => void;
  autoHide?: boolean;
  duration?: number;
}> = ({ error, context, onDismiss, autoHide = true, duration = 5000 }) => {
  React.useEffect(() => {
    if (autoHide) {
      const timer = setTimeout(onDismiss, duration);
      return () => clearTimeout(timer);
    }
  }, [autoHide, duration, onDismiss]);

  return (
    <div className="fixed top-4 right-4 z-50 max-w-sm">
      <ErrorDisplay
        error={error}
        context={context}
        onDismiss={onDismiss}
        compact={true}
        className="shadow-lg"
      />
    </div>
  );
};

export default ErrorDisplay;
