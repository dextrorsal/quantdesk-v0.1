import React, { useState, useEffect } from 'react'
import { Connection, PublicKey } from '@solana/web3.js'
import { useWallet } from '@solana/wallet-adapter-react'

interface WalletTestingComponentProps {
  connection: Connection
  wallet: any
  connected: boolean
  connecting: boolean
  debugInfo: any
  updateDebugInfo: (key: string, value: any) => void
  errorLog: string[]
  addToErrorLog: (error: string) => void
  clearErrorLog: () => void
  testRPCConnection: () => void
}

const WalletTestingComponent: React.FC<WalletTestingComponentProps> = ({
  connection,
  wallet,
  connected,
  connecting,
  debugInfo,
  updateDebugInfo,
  addToErrorLog,
  testRPCConnection
}) => {
  const { connect, disconnect } = useWallet()
  const [walletBalance, setWalletBalance] = useState<number>(0)
  const [rpcStatus, setRpcStatus] = useState<'checking' | 'connected' | 'error'>('checking')

  // Test RPC connection
  useEffect(() => {
    const testConnection = async () => {
      try {
        setRpcStatus('checking')
        await connection.getVersion()
        setRpcStatus('connected')
        updateDebugInfo('rpcStatus', 'connected')
      } catch (error) {
        setRpcStatus('error')
        updateDebugInfo('rpcStatus', 'error')
        addToErrorLog(`RPC Connection Failed: ${error}`)
      }
    }
    testConnection()
  }, [connection])

  // Get wallet balance
  useEffect(() => {
    const getBalance = async () => {
      if (wallet?.adapter?.publicKey) {
        try {
          const balance = await connection.getBalance(wallet.adapter.publicKey)
          setWalletBalance(balance / 1e9) // Convert lamports to SOL
          updateDebugInfo('walletBalance', balance / 1e9)
        } catch (error) {
          addToErrorLog(`Balance Check Failed: ${error}`)
        }
      }
    }
    getBalance()
  }, [wallet, connection])

  const handleConnect = async () => {
    try {
      await connect()
      updateDebugInfo('lastConnectAttempt', new Date().toISOString())
    } catch (error) {
      addToErrorLog(`Wallet Connect Failed: ${error}`)
    }
  }

  const handleDisconnect = async () => {
    try {
      await disconnect()
      updateDebugInfo('lastDisconnectAttempt', new Date().toISOString())
    } catch (error) {
      addToErrorLog(`Wallet Disconnect Failed: ${error}`)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'text-green-400'
      case 'checking': return 'text-yellow-400'
      case 'error': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'connected': return '✅'
      case 'checking': return '⏳'
      case 'error': return '❌'
      default: return '❓'
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Wallet Connection Testing</h2>
        
        {/* Connection Status */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-medium mb-3">Connection Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-sm font-medium">Wallet Status:</span>
                <span className={`${getStatusColor(connected ? 'connected' : 'error')} flex items-center space-x-1`}>
                  <span>{getStatusIcon(connected ? 'connected' : 'error')}</span>
                  <span>{connected ? 'Connected' : 'Disconnected'}</span>
                </span>
              </div>
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-sm font-medium">RPC Status:</span>
                <span className={`${getStatusColor(rpcStatus)} flex items-center space-x-1`}>
                  <span>{getStatusIcon(rpcStatus)}</span>
                  <span className="capitalize">{rpcStatus}</span>
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium">Network:</span>
                <span className="text-gray-300">{connection.rpcEndpoint}</span>
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-sm font-medium">Wallet Balance:</span>
                <span className="text-gray-300">{walletBalance.toFixed(4)} SOL</span>
              </div>
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-sm font-medium">Wallet Address:</span>
                <span className="text-gray-300 font-mono text-xs">
                  {wallet?.adapter?.publicKey ? 
                    `${wallet.adapter.publicKey.toString().slice(0, 8)}...${wallet.adapter.publicKey.toString().slice(-8)}` : 
                    'Not connected'
                  }
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Wallet Actions */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-medium mb-3">Wallet Actions</h3>
          <div className="flex space-x-4">
            {!connected ? (
              <button
                onClick={handleConnect}
                disabled={connecting}
                className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {connecting ? 'Connecting...' : 'Connect Wallet'}
              </button>
            ) : (
              <button
                onClick={handleDisconnect}
                className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
              >
                Disconnect Wallet
              </button>
            )}
            <button
              onClick={testRPCConnection}
              className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
            >
              Test RPC Connection
            </button>
          </div>
        </div>

        {/* Wallet Information */}
        {connected && wallet && (
          <div className="bg-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-3">Wallet Information</h3>
            <div className="space-y-2">
              <div>
                <span className="text-sm font-medium">Wallet Name:</span>
                <span className="text-gray-300 ml-2">{wallet.adapter.name}</span>
              </div>
              <div>
                <span className="text-sm font-medium">Public Key:</span>
                <span className="text-gray-300 ml-2 font-mono text-xs break-all">
                  {wallet.adapter.publicKey?.toString()}
                </span>
              </div>
              <div>
                <span className="text-sm font-medium">Connected:</span>
                <span className="text-gray-300 ml-2">{wallet.adapter.connected ? 'Yes' : 'No'}</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default WalletTestingComponent
