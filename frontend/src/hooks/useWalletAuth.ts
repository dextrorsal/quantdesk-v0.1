import { useState, useCallback, useEffect } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { walletAuthService } from '../services/walletAuth';
import { Logger } from '../utils/logger';

const logger = new Logger();

export interface WalletAuthState {
  isAuthenticated: boolean;
  isLoading: boolean;
  error: string | null;
  user: any | null;
}

export const useWalletAuth = () => {
  const { connected, publicKey, signMessage, disconnect } = useWallet();
  const [state, setState] = useState<WalletAuthState>({
    isAuthenticated: false, // Will be checked on mount/initial load
    isLoading: true, // Set to true initially while checking auth status
    error: null,
    user: null,
  });

  // Check auth status on mount
  useEffect(() => {
    const checkAuthStatus = async () => {
      if (connected && publicKey) {
        const user = await walletAuthService.checkSession();
        if (user) {
          setState(prev => ({ ...prev, isAuthenticated: true, user, isLoading: false }));
        } else {
          setState(prev => ({ ...prev, isAuthenticated: false, user: null, isLoading: false }));
        }
      } else {
        setState(prev => ({ ...prev, isAuthenticated: false, user: null, isLoading: false }));
      }
    };

    checkAuthStatus();
  }, [connected, publicKey]);

  const authenticate = useCallback(async (): Promise<boolean> => {
    if (!connected || !publicKey || !signMessage) {
      setState(prev => ({
        ...prev,
        error: 'Wallet not connected or signing not supported',
      }));
      return false;
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const walletPubkey = publicKey.toString();

      // 1. Get nonce from backend
      const nonceResponse = await walletAuthService.getNonce(walletPubkey);
      if (!nonceResponse.success || !nonceResponse.nonce) {
        setState(prev => ({ ...prev, isLoading: false, error: nonceResponse.error || 'Failed to get nonce' }));
        return false;
      }
      const nonce = nonceResponse.nonce;

      // 2. Create message for signing
      const authMessage = walletAuthService.createAuthMessage(walletPubkey, nonce);

      // 3. Sign the message with the wallet
      const signature = await signMessage(new TextEncoder().encode(authMessage.message));

      // 4. Verify signature with backend
      const response = await walletAuthService.verifySignature(
        walletPubkey,
        signature,
        nonce,
      );

      if (response.success) {
        // Fetch user profile after successful authentication
        const userProfile = await walletAuthService.fetchUserProfile();
        setState(prev => ({
          ...prev,
          isAuthenticated: true,
          isLoading: false,
          error: null,
          user: userProfile.user,
        }));
        return true;
      } else {
        setState(prev => ({
          ...prev,
          isLoading: false,
          error: response.error || 'Authentication failed',
        }));
        return false;
      }
    } catch (error: any) {
      logger.error('Authentication error:', error);
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error.message || 'Authentication failed',
      }));
      return false;
    }
  }, [connected, publicKey, signMessage]);

  const logout = useCallback(async () => {
    await walletAuthService.logout();
    disconnect(); // Disconnect wallet adapter
    setState({
      isAuthenticated: false,
      isLoading: false,
      error: null,
      user: null,
    });
  }, [disconnect]);

  const checkAuth = useCallback(async () => {
    setState(prev => ({ ...prev, isLoading: true }));
    const user = await walletAuthService.checkSession();
    if (user) {
      setState(prev => ({ ...prev, isAuthenticated: true, user, isLoading: false }));
      return true;
    } else {
      setState(prev => ({ ...prev, isAuthenticated: false, user: null, isLoading: false }));
      return false;
    }
  }, []);

  return {
    ...state,
    authenticate,
    logout,
    checkAuth,
    walletAddress: publicKey?.toString() || null,
  };
};
