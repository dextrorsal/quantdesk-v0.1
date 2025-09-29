import { PublicKey } from '@solana/web3.js'
import { sign } from '@solana/web3.js'

export interface AuthMessage {
  message: string
  timestamp: number
  nonce: string
}

export interface AuthResponse {
  success: boolean
  token?: string
  user?: {
    id: string
    walletAddress: string
    username?: string
    email?: string
    kycStatus: string
    riskLevel: string
    totalVolume: number
    totalTrades: number
  }
  error?: string
}

class WalletAuthService {
  private baseUrl: string

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:3002'
  }

  /**
   * Generate a nonce for wallet authentication
   */
  generateNonce(): string {
    return Math.random().toString(36).substring(2, 15) + 
           Math.random().toString(36).substring(2, 15)
  }

  /**
   * Create authentication message for wallet signing
   */
  createAuthMessage(walletAddress: string, nonce: string): AuthMessage {
    const timestamp = Date.now()
    const message = `QuantDesk Authentication\n\nWallet: ${walletAddress}\nNonce: ${nonce}\nTimestamp: ${timestamp}\n\nThis signature proves you own this wallet and allows you to access QuantDesk.`
    
    return {
      message,
      timestamp,
      nonce
    }
  }

  /**
   * Authenticate user with wallet signature
   */
  async authenticateWithWallet(
    walletAddress: string,
    signature: Uint8Array,
    message: string
  ): Promise<AuthResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/auth/authenticate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          walletAddress,
          signature: Array.from(signature), // Convert Uint8Array to regular array
          message
        })
      })

      const data = await response.json()

      if (data.success && data.token) {
        // Store token in localStorage
        localStorage.setItem('quantdesk_auth_token', data.token)
        localStorage.setItem('quantdesk_wallet_address', walletAddress)
      }

      return data
    } catch (error) {
      console.error('Authentication error:', error)
      return {
        success: false,
        error: 'Network error during authentication'
      }
    }
  }

  /**
   * Get stored authentication token
   */
  getAuthToken(): string | null {
    return localStorage.getItem('quantdesk_auth_token')
  }

  /**
   * Get stored wallet address
   */
  getWalletAddress(): string | null {
    return localStorage.getItem('quantdesk_wallet_address')
  }

  /**
   * Check if user is authenticated
   */
  isAuthenticated(): boolean {
    const token = this.getAuthToken()
    if (!token) return false

    try {
      // Check if token is expired (basic check)
      const payload = JSON.parse(atob(token.split('.')[1]))
      const now = Date.now() / 1000
      return payload.exp > now
    } catch {
      return false
    }
  }

  /**
   * Logout user
   */
  logout(): void {
    localStorage.removeItem('quantdesk_auth_token')
    localStorage.removeItem('quantdesk_wallet_address')
  }

  /**
   * Get authenticated headers for API requests
   */
  getAuthHeaders(): HeadersInit {
    const token = this.getAuthToken()
    return {
      'Content-Type': 'application/json',
      ...(token && { 'Authorization': `Bearer ${token}` })
    }
  }
}

// Lazy initialization to avoid constructor running at module level
let _walletAuthService: WalletAuthService | null = null

export const walletAuthService = {
  get instance() {
    if (!_walletAuthService) {
      _walletAuthService = new WalletAuthService()
    }
    return _walletAuthService
  }
}
