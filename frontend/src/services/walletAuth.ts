/**
 * Wallet Authentication Service
 * Handles wallet-based authentication for QuantDesk
 */

import { Logger } from '../utils/logger';
import { apiClient } from './apiClient';

const logger = new Logger();

export interface AuthResponse {
  success: boolean;
  error?: string;
  user?: any;
  nonce?: string;
}

export interface NonceResponse {
  success: boolean;
  nonce?: string;
  error?: string;
}

export interface AuthMessage {
  message: string;
  timestamp: number;
}

export class WalletAuthService {
  private static _instance: WalletAuthService;
  private baseUrl: string;
  private retryAttempts: number = 0;
  private maxRetries: number = 3;
  private retryDelay: number = 1000; // 1 second

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:3002';
  }

  static get instance(): WalletAuthService {
    if (!WalletAuthService._instance) {
      WalletAuthService._instance = new WalletAuthService();
    }
    return WalletAuthService._instance;
  }

  /**
   * Enhanced retry mechanism with exponential backoff
   */
  private async withRetry<T>(
    operation: () => Promise<T>,
    operationName: string = 'operation'
  ): Promise<T> {
    for (let attempt = 0; attempt <= this.maxRetries; attempt++) {
      try {
        const result = await operation();
        this.retryAttempts = 0; // Reset on success
        return result;
      } catch (error: any) {
        const isLastAttempt = attempt === this.maxRetries;
        const shouldRetry = this.shouldRetry(error) && !isLastAttempt;
        
        if (shouldRetry) {
          const delay = this.retryDelay * Math.pow(2, attempt);
          console.warn(`${operationName} failed (attempt ${attempt + 1}), retrying in ${delay}ms:`, error.message);
          await new Promise(resolve => setTimeout(resolve, delay));
        } else {
          console.error(`${operationName} failed after ${attempt + 1} attempts:`, error);
          throw error;
        }
      }
    }
    throw new Error(`${operationName} failed after ${this.maxRetries + 1} attempts`);
  }

  /**
   * Determine if an error should trigger a retry
   */
  private shouldRetry(error: any): boolean {
    // Retry on network errors, timeouts, and 5xx server errors
    if (!error.response) return true; // Network error
    if (error.code === 'ECONNABORTED') return true; // Timeout
    if (error.response.status >= 500) return true; // Server error
    if (error.response.status === 429) return true; // Rate limited
    
    // Don't retry on client errors (4xx except 429)
    return false;
  }

  /**
   * Convert technical errors to user-friendly messages
   */
  private getUserFriendlyError(error: any): string {
    if (!error) return 'An unexpected error occurred';
    
    // Network errors
    if (!error.response) {
      return 'Network connection failed. Please check your internet connection and try again.';
    }
    
    // HTTP status errors
    const status = error.response.status;
    switch (status) {
      case 400:
        return 'Invalid request. Please try again.';
      case 401:
        return 'Authentication failed. Please reconnect your wallet.';
      case 403:
        return 'Access denied. Please check your permissions.';
      case 404:
        return 'Service not found. Please try again later.';
      case 429:
        return 'Too many requests. Please wait a moment and try again.';
      case 500:
        return 'Server error. Please try again later.';
      case 502:
      case 503:
      case 504:
        return 'Service temporarily unavailable. Please try again later.';
      default:
        return error.message || 'An unexpected error occurred';
    }
  }

  /**
   * Get nonce for wallet authentication with enhanced error handling
   */
  async getNonce(walletAddress: string): Promise<NonceResponse> {
    try {
      return await this.withRetry(async () => {
        const response = await apiClient.post<NonceResponse>(`${this.baseUrl}/auth/nonce`, { walletAddress });
        return response.data;
      }, 'getNonce');
    } catch (error: any) {
      console.error('Failed to get nonce after retries:', error);
      return { 
        success: false, 
        error: this.getUserFriendlyError(error) || 'Failed to get authentication nonce' 
      };
    }
  }

  /**
   * Create authentication message for signing
   */
  createAuthMessage(walletAddress: string, nonce: string): AuthMessage {
    const timestamp = Date.now();
    const message = `QuantDesk Authentication\nWallet: ${walletAddress}\nNonce: ${nonce}\nTimestamp: ${timestamp}`;
    
    return {
      message,
      timestamp,
    };
  }

  /**
   * Verify signature with backend using enhanced error handling
   */
  async verifySignature(
    walletAddress: string,
    signature: Uint8Array,
    nonce: string
  ): Promise<AuthResponse> {
    try {
      return await this.withRetry(async () => {
        const response = await apiClient.post<{ user: any; token: string }>(`${this.baseUrl}/auth/verify`, {
          walletAddress,
          signature: Buffer.from(signature).toString('base64'),
          nonce,
        });
        localStorage.setItem('jwtToken', response.data.token);
        return { success: true, user: response.data.user };
      }, 'verifySignature');
    } catch (error: any) {
      console.error('Failed to verify signature after retries:', error);
      return { 
        success: false, 
        error: this.getUserFriendlyError(error) || 'Signature verification failed' 
      };
    }
  }

  /**
   * Check current session
   */
  async checkSession(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/api/auth/session`, {
        method: 'GET',
        credentials: 'include',
      });

      // Gracefully handle 404 or other errors
      if (!response.ok || response.status === 404) {
        return null;
      }

      const data = await response.json();
      return data.user;
    } catch (error: any) {
      // Don't log 404 errors as they're expected when not authenticated
      if (error.message?.includes('404')) {
        return null;
      }
      logger.error('Failed to check session:', error);
      return null;
    }
  }

  /**
   * Fetch user profile
   */
  async fetchUserProfile(): Promise<AuthResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/auth/profile`, {
        method: 'GET',
        credentials: 'include',
      });

      if (!response.ok) {
        return {
          success: false,
          error: `HTTP ${response.status}: ${response.statusText}`,
        };
      }

      const data = await response.json();
      return {
        success: true,
        user: data.user,
      };
    } catch (error: any) {
      logger.error('Failed to fetch user profile:', error);
      return {
        success: false,
        error: error.message || 'Failed to fetch user profile',
      };
    }
  }

  /**
   * Logout user
   */
  async logout(): Promise<void> {
    try {
      await fetch(`${this.baseUrl}/api/auth/logout`, {
        method: 'POST',
        credentials: 'include',
      });
    } catch (error: any) {
      logger.error('Failed to logout:', error);
    }
  }
}

export const walletAuthService = WalletAuthService.instance;
