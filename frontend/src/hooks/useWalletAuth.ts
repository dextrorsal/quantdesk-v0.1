import { useState, useCallback } from 'react'
import { useWallet } from '@solana/wallet-adapter-react'
import { walletAuthService, AuthResponse } from '../services/walletAuth'

export interface WalletAuthState {
  isAuthenticated: boolean
  isLoading: boolean
  error: string | null
  user: any | null
}

export const useWalletAuth = () => {
  const { connected, publicKey, signMessage } = useWallet()
  const [state, setState] = useState<WalletAuthState>({
    isAuthenticated: walletAuthService.instance.isAuthenticated(),
    isLoading: false,
    error: null,
    user: null
  })

  const authenticate = useCallback(async (): Promise<boolean> => {
    if (!connected || !publicKey || !signMessage) {
      setState(prev => ({
        ...prev,
        error: 'Wallet not connected or signing not supported'
      }))
      return false
    }

    setState(prev => ({ ...prev, isLoading: true, error: null }))

    try {
      const walletAddress = publicKey.toString()
      const nonce = walletAuthService.instance.generateNonce()
      const authMessage = walletAuthService.instance.createAuthMessage(walletAddress, nonce)

      // Sign the message with the wallet
      const signature = await signMessage(new TextEncoder().encode(authMessage.message))

      // Authenticate with backend
      const response: AuthResponse = await walletAuthService.instance.authenticateWithWallet(
        walletAddress,
        signature,
        authMessage.message
      )

      if (response.success) {
        setState(prev => ({
          ...prev,
          isAuthenticated: true,
          isLoading: false,
          error: null,
          user: response.user
        }))
        return true
      } else {
        setState(prev => ({
          ...prev,
          isLoading: false,
          error: response.error || 'Authentication failed'
        }))
        return false
      }
    } catch (error: any) {
      console.error('Authentication error:', error)
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: error.message || 'Authentication failed'
      }))
      return false
    }
  }, [connected, publicKey, signMessage])

  const logout = useCallback(() => {
    walletAuthService.instance.logout()
    setState({
      isAuthenticated: false,
      isLoading: false,
      error: null,
      user: null
    })
  }, [])

  const checkAuth = useCallback(() => {
    const isAuth = walletAuthService.instance.isAuthenticated()
    setState(prev => ({
      ...prev,
      isAuthenticated: isAuth
    }))
    return isAuth
  }, [])

  return {
    ...state,
    authenticate,
    logout,
    checkAuth,
    walletAddress: publicKey?.toString() || null
  }
}
