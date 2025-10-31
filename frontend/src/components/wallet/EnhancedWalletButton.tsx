import React, { useState, useCallback, useEffect } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { useWalletModal } from '@solana/wallet-adapter-react-ui';
import { useWalletAuth } from '../../hooks/useWalletAuth';
import { AlertCircle, CheckCircle, Loader2, RefreshCw, ExternalLink } from 'lucide-react';

/**
 * Enhanced Wallet Connection Component
 * Features:
 * - Better error handling and user feedback
 * - Connection status indicators
 * - Retry mechanisms
 * - Professional styling
 * - Accessibility improvements
 */

interface EnhancedWalletButtonProps {
  className?: string;
  showAccountPanel?: boolean;
  onAccountPanelToggle?: (isOpen: boolean) => void;
}

const EnhancedWalletButton: React.FC<EnhancedWalletButtonProps> = ({
  className = '',
  showAccountPanel = false,
  onAccountPanelToggle
}) => {
  const { connected, publicKey, connecting, disconnecting } = useWallet();
  const { setVisible } = useWalletModal();
  const { authenticate, isAuthenticated, isLoading, error } = useWalletAuth();
  const [isRetrying, setIsRetrying] = useState(false);
  const [retryCount, setRetryCount] = useState(0);
  const [lastError, setLastError] = useState<string | null>(null);

  const maxRetries = 3;

  // Clear error when wallet connection changes
  useEffect(() => {
    if (connected && publicKey) {
      setLastError(null);
      setRetryCount(0);
    }
  }, [connected, publicKey]);

  // Handle wallet connection
  const handleConnect = useCallback(async () => {
    if (!connected) {
      setVisible(true);
      return;
    }

    // If connected but not authenticated, try to authenticate
    if (!isAuthenticated && !isLoading) {
      try {
        await authenticate();
      } catch (err: any) {
        setLastError(err.message || 'Authentication failed');
      }
    }

    // Toggle account panel
    if (onAccountPanelToggle) {
      onAccountPanelToggle(!showAccountPanel);
    }
  }, [connected, isAuthenticated, isLoading, authenticate, setVisible, showAccountPanel, onAccountPanelToggle]);

  // Handle retry authentication
  const handleRetry = useCallback(async () => {
    if (retryCount >= maxRetries) {
      setLastError('Maximum retry attempts reached. Please refresh the page.');
      return;
    }

    setIsRetrying(true);
    setLastError(null);
    
    try {
      await authenticate();
      setRetryCount(0);
    } catch (err: any) {
      setRetryCount(prev => prev + 1);
      setLastError(err.message || 'Authentication failed');
    } finally {
      setIsRetrying(false);
    }
  }, [retryCount, maxRetries, authenticate]);

  // Handle disconnect
  const handleDisconnect = useCallback(() => {
    if (onAccountPanelToggle) {
      onAccountPanelToggle(false);
    }
  }, [onAccountPanelToggle]);

  // Get connection status
  const getConnectionStatus = () => {
    if (connecting) return 'connecting';
    if (disconnecting) return 'disconnecting';
    if (isRetrying) return 'retrying';
    if (isLoading) return 'loading';
    if (error || lastError) return 'error';
    if (connected && isAuthenticated) return 'connected';
    if (connected && !isAuthenticated) return 'connected-not-authenticated';
    return 'disconnected';
  };

  const status = getConnectionStatus();

  // Get status color
  const getStatusColor = () => {
    switch (status) {
      case 'connected': return 'text-green-400';
      case 'connected-not-authenticated': return 'text-yellow-400';
      case 'error': return 'text-red-400';
      case 'connecting':
      case 'loading':
      case 'retrying': return 'text-blue-400';
      default: return 'text-gray-400';
    }
  };

  // Get status icon
  const getStatusIcon = () => {
    switch (status) {
      case 'connected': return <CheckCircle className="w-4 h-4" />;
      case 'error': return <AlertCircle className="w-4 h-4" />;
      case 'connecting':
      case 'loading':
      case 'retrying': return <Loader2 className="w-4 h-4 animate-spin" />;
      default: return null;
    }
  };

  // Get button text
  const getButtonText = () => {
    switch (status) {
      case 'connecting': return 'Connecting...';
      case 'disconnecting': return 'Disconnecting...';
      case 'loading': return 'Authenticating...';
      case 'retrying': return 'Retrying...';
      case 'connected': return `Connected: ${publicKey?.toString().slice(0, 4)}...${publicKey?.toString().slice(-4)}`;
      case 'connected-not-authenticated': return 'Authenticate Wallet';
      case 'error': return 'Retry Connection';
      default: return 'Connect Wallet';
    }
  };

  // Get button variant
  const getButtonVariant = () => {
    switch (status) {
      case 'connected': return 'bg-green-600 hover:bg-green-700';
      case 'error': return 'bg-red-600 hover:bg-red-700';
      case 'connected-not-authenticated': return 'bg-yellow-600 hover:bg-yellow-700';
      default: return 'bg-blue-600 hover:bg-blue-700';
    }
  };

  return (
    <div className={`relative ${className}`}>
      {/* Main Button */}
      <button
        onClick={status === 'error' ? handleRetry : handleConnect}
        disabled={connecting || disconnecting || isLoading || isRetrying}
        className={`
          flex items-center space-x-2 px-4 py-2 rounded-lg font-semibold text-white transition-all duration-200
          ${getButtonVariant()}
          disabled:opacity-50 disabled:cursor-not-allowed
          focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900
        `}
        aria-label={getButtonText()}
      >
        {getStatusIcon()}
        <span>{getButtonText()}</span>
        {status === 'connected' && (
          <ExternalLink className="w-3 h-3" />
        )}
      </button>

      {/* Error Message */}
      {(error || lastError) && (
        <div className="absolute top-full left-0 right-0 mt-2 p-3 bg-red-900/20 border border-red-500 rounded-lg">
          <div className="flex items-start space-x-2">
            <AlertCircle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
            <div className="flex-1">
              <p className="text-red-400 text-sm font-medium">Connection Error</p>
              <p className="text-red-300 text-xs mt-1">{error || lastError}</p>
              {retryCount < maxRetries && (
                <button
                  onClick={handleRetry}
                  disabled={isRetrying}
                  className="mt-2 flex items-center space-x-1 text-xs text-red-300 hover:text-red-200 transition-colors"
                >
                  <RefreshCw className={`w-3 h-3 ${isRetrying ? 'animate-spin' : ''}`} />
                  <span>Retry ({retryCount}/{maxRetries})</span>
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Connection Status Indicator */}
      {connected && (
        <div className="absolute -top-1 -right-1">
          <div className={`
            w-3 h-3 rounded-full border-2 border-gray-900
            ${status === 'connected' ? 'bg-green-500' : 
              status === 'connected-not-authenticated' ? 'bg-yellow-500' : 'bg-red-500'}
          `} />
        </div>
      )}

      {/* Loading Overlay */}
      {(connecting || isLoading || isRetrying) && (
        <div className="absolute inset-0 bg-gray-900/50 rounded-lg flex items-center justify-center">
          <Loader2 className="w-5 h-5 animate-spin text-white" />
        </div>
      )}
    </div>
  );
};

export default EnhancedWalletButton;
