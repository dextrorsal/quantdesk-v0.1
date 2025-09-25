import React, { useEffect } from 'react'
import { useWallet } from '@solana/wallet-adapter-react'
import { WalletMultiButton } from '@solana/wallet-adapter-react-ui'
import { Copy, ExternalLink, LogOut, Shield } from 'lucide-react'
import { useWalletAuth } from '../hooks/useWalletAuth'

const WalletButton: React.FC = () => {
  const { connected, publicKey, disconnect } = useWallet()
  const { isAuthenticated, isLoading, authenticate, logout, user } = useWalletAuth()

  // Auto-authenticate when wallet connects
  useEffect(() => {
    if (connected && publicKey && !isAuthenticated && !isLoading) {
      authenticate()
    }
  }, [connected, publicKey, isAuthenticated, isLoading, authenticate])

  if (!connected) {
    return (
      <WalletMultiButton className="!bg-blue-600 hover:!bg-blue-700 !text-white !rounded-lg !px-4 !py-2 !text-sm !font-medium !transition-colors" />
    )
  }

  if (!publicKey) {
    return (
      <button 
        onClick={disconnect}
        className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-medium hover:bg-red-700 transition-colors"
      >
        Disconnect
      </button>
    )
  }

  // Format wallet address for display
  const address = publicKey.toString()
  const shortAddress = `${address.slice(0, 4)}...${address.slice(-4)}`

  // Show loading state during authentication
  if (isLoading) {
    return (
      <button className="flex items-center space-x-2 px-4 py-2 bg-yellow-600 text-white rounded-lg text-sm font-medium cursor-not-allowed">
        <div className="w-2 h-2 bg-yellow-400 rounded-full animate-pulse"></div>
        <span>Authenticating...</span>
      </button>
    )
  }

  // Show authentication status
  const buttonColor = isAuthenticated ? 'bg-green-600 hover:bg-green-700' : 'bg-orange-600 hover:bg-orange-700'
  const statusColor = isAuthenticated ? 'bg-green-400' : 'bg-orange-400'
  const statusText = isAuthenticated ? 'Authenticated' : 'Not Authenticated'

  return (
    <div className="relative group">
      <button className={`flex items-center space-x-2 px-4 py-2 ${buttonColor} text-white rounded-lg text-sm font-medium transition-colors`}>
        <div className={`w-2 h-2 ${statusColor} rounded-full`}></div>
        <span>{shortAddress}</span>
      </button>
      
      {/* Wallet dropdown */}
      <div className="absolute right-0 mt-2 hidden group-hover:block bg-gray-900 border border-gray-800 rounded-lg shadow-lg z-20 min-w-[250px]">
        {/* Authentication Status */}
        <div className="p-3 border-b border-gray-800">
          <div className="flex items-center space-x-2 mb-2">
            <Shield className={`h-4 w-4 ${isAuthenticated ? 'text-green-400' : 'text-orange-400'}`} />
            <span className={`text-sm font-medium ${isAuthenticated ? 'text-green-400' : 'text-orange-400'}`}>
              {statusText}
            </span>
          </div>
          
          {user && (
            <div className="space-y-1">
              <div className="text-xs text-gray-400">User ID: {user.id}</div>
              {user.username && <div className="text-xs text-gray-400">Username: {user.username}</div>}
              <div className="text-xs text-gray-400">Risk Level: {user.riskLevel}</div>
              <div className="text-xs text-gray-400">Total Volume: ${user.totalVolume.toLocaleString()}</div>
            </div>
          )}
        </div>

        {/* Wallet Address */}
        <div className="p-3 border-b border-gray-800">
          <div className="text-xs text-gray-400 mb-1">Wallet Address</div>
          <div className="flex items-center space-x-2">
            <span className="text-sm font-mono text-white">{shortAddress}</span>
            <button
              onClick={() => navigator.clipboard.writeText(address)}
              className="p-1 text-gray-400 hover:text-white transition-colors"
              title="Copy address"
            >
              <Copy className="h-3 w-3" />
            </button>
            <a
              href={`https://explorer.solana.com/address/${address}?cluster=devnet`}
              target="_blank"
              rel="noopener noreferrer"
              className="p-1 text-gray-400 hover:text-white transition-colors"
              title="View on Solana Explorer"
            >
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>
        </div>
        
        {/* Actions */}
        <div className="p-2 space-y-1">
          {!isAuthenticated && (
            <button
              onClick={authenticate}
              className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-white hover:bg-gray-800 rounded transition-colors bg-blue-600 hover:bg-blue-700"
            >
              <Shield className="h-4 w-4" />
              <span>Authenticate</span>
            </button>
          )}
          
          <button
            onClick={logout}
            className="w-full flex items-center space-x-2 px-3 py-2 text-sm text-gray-300 hover:text-white hover:bg-gray-800 rounded transition-colors"
          >
            <LogOut className="h-4 w-4" />
            <span>Disconnect</span>
          </button>
        </div>
      </div>
    </div>
  )
}

export default WalletButton
