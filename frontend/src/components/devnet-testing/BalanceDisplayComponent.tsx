import React, { useState, useEffect } from 'react'
import { Connection, PublicKey } from '@solana/web3.js'
import { useWallet } from '@solana/wallet-adapter-react'

interface BalanceDisplayComponentProps {
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

const BalanceDisplayComponent: React.FC<BalanceDisplayComponentProps> = ({
  connection,
  wallet,
  connected,
  debugInfo,
  updateDebugInfo,
  addToErrorLog
}) => {
  const [walletBalances, setWalletBalances] = useState<{[key: string]: number}>({
    SOL: 0,
    USDC: 0,
    USDT: 0
  })
  const [accountBalances, setAccountBalances] = useState<{[key: string]: number}>({
    SOL: 0,
    USDC: 0,
    USDT: 0
  })
  const [lastUpdated, setLastUpdated] = useState<Date>(new Date())
  const [isRefreshing, setIsRefreshing] = useState(false)

  // Mock price data (in real implementation, this would come from price oracle)
  const mockPrices = {
    SOL: 100.0,
    USDC: 1.0,
    USDT: 1.0
  }

  // Get wallet balances
  const getWalletBalances = async () => {
    if (!wallet?.adapter?.publicKey) {
      return
    }

    try {
      setIsRefreshing(true)
      
      // Get SOL balance
      const solBalance = await connection.getBalance(wallet.adapter.publicKey)
      const solAmount = solBalance / 1e9
      
      // Mock USDC and USDT balances (in real implementation, these would be token account balances)
      const mockUSDCBalance = 1000.0
      const mockUSDTBalance = 500.0
      
      setWalletBalances({
        SOL: solAmount,
        USDC: mockUSDCBalance,
        USDT: mockUSDTBalance
      })
      
      updateDebugInfo('walletBalances', {
        SOL: solAmount,
        USDC: mockUSDCBalance,
        USDT: mockUSDTBalance
      })
      
      setLastUpdated(new Date())
    } catch (error) {
      addToErrorLog(`Balance Fetch Failed: ${error}`)
    } finally {
      setIsRefreshing(false)
    }
  }

  // Get account balances (simulated)
  const getAccountBalances = async () => {
    try {
      // Mock account balances (in real implementation, these would come from program accounts)
      const mockAccountBalances = {
        SOL: 5.0,
        USDC: 2000.0,
        USDT: 1000.0
      }
      
      setAccountBalances(mockAccountBalances)
      updateDebugInfo('accountBalances', mockAccountBalances)
    } catch (error) {
      addToErrorLog(`Account Balance Fetch Failed: ${error}`)
    }
  }

  // Calculate total portfolio value
  const calculateTotalValue = (balances: {[key: string]: number}) => {
    return Object.entries(balances).reduce((total, [token, amount]) => {
      return total + (amount * mockPrices[token as keyof typeof mockPrices])
    }, 0)
  }

  // Refresh all balances
  const refreshBalances = async () => {
    await Promise.all([
      getWalletBalances(),
      getAccountBalances()
    ])
  }

  // Auto-refresh balances
  useEffect(() => {
    if (connected && wallet?.adapter?.publicKey) {
      refreshBalances()
      
      // Set up auto-refresh every 30 seconds
      const interval = setInterval(refreshBalances, 30000)
      return () => clearInterval(interval)
    }
  }, [connected, wallet])

  const walletTotalValue = calculateTotalValue(walletBalances)
  const accountTotalValue = calculateTotalValue(accountBalances)
  const totalPortfolioValue = walletTotalValue + accountTotalValue

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Balance Display</h2>
        
        {/* Portfolio Summary */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-medium">Portfolio Summary</h3>
            <button
              onClick={refreshBalances}
              disabled={isRefreshing}
              className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
            >
              {isRefreshing ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center">
              <div className="text-sm text-gray-400 mb-1">Wallet Value</div>
              <div className="text-xl font-semibold">${walletTotalValue.toFixed(2)}</div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-400 mb-1">Account Value</div>
              <div className="text-xl font-semibold">${accountTotalValue.toFixed(2)}</div>
            </div>
            <div className="text-center">
              <div className="text-sm text-gray-400 mb-1">Total Portfolio</div>
              <div className="text-2xl font-bold text-green-400">${totalPortfolioValue.toFixed(2)}</div>
            </div>
          </div>
          <div className="text-xs text-gray-400 mt-2 text-center">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </div>
        </div>

        {/* Wallet Balances */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-medium mb-3">Wallet Balances</h3>
          <div className="space-y-3">
            {Object.entries(walletBalances).map(([token, amount]) => (
              <div key={token} className="flex justify-between items-center p-3 bg-gray-600 rounded">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gray-500 rounded-full flex items-center justify-center text-xs font-bold">
                    {token.charAt(0)}
                  </div>
                  <div>
                    <div className="font-medium">{token}</div>
                    <div className="text-sm text-gray-400">Wallet</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium">{amount.toFixed(4)}</div>
                  <div className="text-sm text-gray-400">
                    ${(amount * mockPrices[token as keyof typeof mockPrices]).toFixed(2)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Account Balances */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-medium mb-3">Account Balances</h3>
          <div className="space-y-3">
            {Object.entries(accountBalances).map(([token, amount]) => (
              <div key={token} className="flex justify-between items-center p-3 bg-gray-600 rounded">
                <div className="flex items-center space-x-3">
                  <div className="w-8 h-8 bg-gray-500 rounded-full flex items-center justify-center text-xs font-bold">
                    {token.charAt(0)}
                  </div>
                  <div>
                    <div className="font-medium">{token}</div>
                    <div className="text-sm text-gray-400">Trading Account</div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="font-medium">{amount.toFixed(4)}</div>
                  <div className="text-sm text-gray-400">
                    ${(amount * mockPrices[token as keyof typeof mockPrices]).toFixed(2)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Price Information */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-medium mb-3">Current Prices</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {Object.entries(mockPrices).map(([token, price]) => (
              <div key={token} className="text-center p-3 bg-gray-600 rounded">
                <div className="font-medium">{token}</div>
                <div className="text-lg font-semibold">${price.toFixed(2)}</div>
                <div className="text-xs text-gray-400">Mock Price</div>
              </div>
            ))}
          </div>
        </div>

        {/* Instructions */}
        <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
          <h3 className="text-lg font-medium mb-2 text-blue-400">Instructions</h3>
          <div className="text-sm text-gray-300 space-y-1">
            <p>• Balances are automatically refreshed every 30 seconds</p>
            <p>• Click "Refresh" to manually update balances</p>
            <p>• Wallet balances show your actual wallet holdings</p>
            <p>• Account balances show your trading account holdings</p>
            <p>• Prices are currently mock data for testing</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default BalanceDisplayComponent
