export interface NonceResponse {
  success: boolean;
  nonce?: string;
  error?: string;
}

export interface AuthResponse {
  success: boolean;
  user?: {
    id: string;
    wallet_pubkey: string; // Changed from walletAddress
    username?: string;
    email?: string;
    role?: string; // Added role
    created_at?: string; // Added created_at
    referrer_pubkey?: string; // Added referrer_pubkey
    is_activated?: boolean; // Added is_activated
  };
  error?: string;
}

class WalletAuthService {
  private baseUrl: string;

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:3002'; // Ensure correct backend port
  }

  async getNonce(walletPubkey: string): Promise<NonceResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/siws/nonce`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ walletPubkey }),
      });
      const data = await response.json();
      if (response.ok) {
        return { success: true, nonce: data.nonce };
      } else {
        return { success: false, error: data.error || 'Failed to get nonce' };
      }
    } catch (error) {
      console.error('Error fetching nonce:', error);
      return { success: false, error: 'Network error fetching nonce' };
    }
  }

  createAuthMessage(walletPubkey: string, nonce: string): { message: string } {
    const message = `QuantDesk Authentication\n\nWallet: ${walletPubkey}\nNonce: ${nonce}\n\nThis signature proves you own this wallet and allows you to access QuantDesk.`;
    return { message };
  }

  async verifySignature(
    walletPubkey: string,
    signature: Uint8Array,
    nonce: string,
    referrerPubkey?: string,
  ): Promise<AuthResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/siws/verify`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          walletPubkey,
          signature: Array.from(signature), // Convert Uint8Array to regular array
          nonce,
          ref: referrerPubkey, // Pass optional referrer
        }),
      });

      const data = await response.json();

      if (response.ok) {
        // The session is now managed by an HttpOnly cookie; no token in response body
        return { success: true };
      } else {
        return { success: false, error: data.error || 'Signature verification failed' };
      }
    } catch (error) {
      console.error('Signature verification error:', error);
      return { success: false, error: 'Network error during signature verification' };
    }
  }

  async logout(): Promise<AuthResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/api/siws/logout`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const data = await response.json();
      return { success: response.ok, error: data.error };
    } catch (error) {
      console.error('Logout error:', error);
      return { success: false, error: 'Network error during logout' };
    }
  }

  // New method to check session and fetch user profile via authenticated endpoint
  async checkSession(): Promise<any | null> {
    try {
      // Skip API call for now - just return null to avoid 401 errors
      // The account state will be fetched directly from blockchain via AccountContext
      console.log('🔍 WalletAuth: Skipping API session check to avoid 401 errors');
      return null;
    } catch (error) {
      console.error('Session check error:', error);
      return null;
    }
  }

  isAuthenticated(): boolean {
    return false; 
  }
}

export const walletAuthService = {
  instance: new WalletAuthService()
};
