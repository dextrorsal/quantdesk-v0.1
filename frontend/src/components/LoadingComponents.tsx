import React from 'react';
import { Loader2, RefreshCw } from 'lucide-react';

export interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl';
  color?: 'primary' | 'secondary' | 'white' | 'gray';
  text?: string;
  className?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  size = 'md',
  color = 'primary',
  text,
  className = ''
}) => {
  const sizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8',
    xl: 'w-12 h-12'
  };

  const colorClasses = {
    primary: 'text-blue-500',
    secondary: 'text-gray-500',
    white: 'text-white',
    gray: 'text-gray-400'
  };

  return (
    <div className={`flex items-center justify-center ${className}`}>
      <div className="flex flex-col items-center space-y-2">
        <Loader2 className={`${sizeClasses[size]} ${colorClasses[color]} animate-spin`} />
        {text && (
          <p className={`text-sm ${colorClasses[color]}`}>
            {text}
          </p>
        )}
      </div>
    </div>
  );
};

export interface LoadingOverlayProps {
  isLoading: boolean;
  text?: string;
  children: React.ReactNode;
  className?: string;
}

export const LoadingOverlay: React.FC<LoadingOverlayProps> = ({
  isLoading,
  text = 'Loading...',
  children,
  className = ''
}) => {
  if (!isLoading) {
    return <>{children}</>;
  }

  return (
    <div className={`relative ${className}`}>
      {children}
      <div className="absolute inset-0 bg-gray-900/80 backdrop-blur-sm flex items-center justify-center z-10">
        <LoadingSpinner size="lg" color="white" text={text} />
      </div>
    </div>
  );
};

export interface LoadingButtonProps {
  isLoading: boolean;
  loadingText?: string;
  children: React.ReactNode;
  disabled?: boolean;
  className?: string;
  onClick?: () => void;
  type?: 'button' | 'submit' | 'reset';
}

export const LoadingButton: React.FC<LoadingButtonProps> = ({
  isLoading,
  loadingText = 'Loading...',
  children,
  disabled = false,
  className = '',
  onClick,
  type = 'button'
}) => {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || isLoading}
      className={`relative ${className} ${(disabled || isLoading) ? 'opacity-50 cursor-not-allowed' : ''}`}
    >
      {isLoading ? (
        <div className="flex items-center justify-center space-x-2">
          <Loader2 className="w-4 h-4 animate-spin" />
          <span>{loadingText}</span>
        </div>
      ) : (
        children
      )}
    </button>
  );
};

export interface LoadingCardProps {
  isLoading: boolean;
  loadingText?: string;
  children: React.ReactNode;
  className?: string;
  skeleton?: boolean;
}

export const LoadingCard: React.FC<LoadingCardProps> = ({
  isLoading,
  loadingText = 'Loading...',
  children,
  className = '',
  skeleton = false
}) => {
  if (!isLoading) {
    return <>{children}</>;
  }

  if (skeleton) {
    return (
      <div className={`bg-gray-800 rounded-lg p-6 animate-pulse ${className}`}>
        <div className="space-y-4">
          <div className="h-4 bg-gray-700 rounded w-3/4"></div>
          <div className="h-4 bg-gray-700 rounded w-1/2"></div>
          <div className="h-4 bg-gray-700 rounded w-5/6"></div>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
      <LoadingSpinner size="lg" color="white" text={loadingText} />
    </div>
  );
};

export interface LoadingTableProps {
  isLoading: boolean;
  loadingText?: string;
  children: React.ReactNode;
  className?: string;
  rows?: number;
}

export const LoadingTable: React.FC<LoadingTableProps> = ({
  isLoading,
  loadingText = 'Loading data...',
  children,
  className = '',
  rows = 5
}) => {
  if (!isLoading) {
    return <>{children}</>;
  }

  return (
    <div className={`bg-gray-800 rounded-lg overflow-hidden ${className}`}>
      <div className="p-6">
        <LoadingSpinner size="lg" color="white" text={loadingText} />
      </div>
      <div className="border-t border-gray-700">
        {Array.from({ length: rows }).map((_, index) => (
          <div key={index} className="p-4 border-b border-gray-700 last:border-b-0">
            <div className="flex space-x-4 animate-pulse">
              <div className="h-4 bg-gray-700 rounded w-1/4"></div>
              <div className="h-4 bg-gray-700 rounded w-1/4"></div>
              <div className="h-4 bg-gray-700 rounded w-1/4"></div>
              <div className="h-4 bg-gray-700 rounded w-1/4"></div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export interface LoadingChartProps {
  isLoading: boolean;
  loadingText?: string;
  children: React.ReactNode;
  className?: string;
}

export const LoadingChart: React.FC<LoadingChartProps> = ({
  isLoading,
  loadingText = 'Loading chart...',
  children,
  className = ''
}) => {
  if (!isLoading) {
    return <>{children}</>;
  }

  return (
    <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
      <div className="h-64 flex items-center justify-center">
        <LoadingSpinner size="xl" color="white" text={loadingText} />
      </div>
    </div>
  );
};

/**
 * Hook for managing loading states
 */
export const useLoadingState = (initialState: boolean = false) => {
  const [isLoading, setIsLoading] = React.useState(initialState);
  const [loadingText, setLoadingText] = React.useState<string>('Loading...');

  const startLoading = React.useCallback((text?: string) => {
    setIsLoading(true);
    if (text) setLoadingText(text);
  }, []);

  const stopLoading = React.useCallback(() => {
    setIsLoading(false);
  }, []);

  const setLoading = React.useCallback((loading: boolean, text?: string) => {
    setIsLoading(loading);
    if (text) setLoadingText(text);
  }, []);

  return {
    isLoading,
    loadingText,
    startLoading,
    stopLoading,
    setLoading
  };
};

export default LoadingSpinner;
