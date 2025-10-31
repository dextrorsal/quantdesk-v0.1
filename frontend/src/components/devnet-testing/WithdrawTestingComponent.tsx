import React, { useState, useEffect } from 'react'
import { Connection, PublicKey, Transaction, SystemProgram } from '@solana/web3.js'
import { useWallet } from '@solana/wallet-adapter-react'

interface WithdrawTestingComponentProps {
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

const WithdrawTestingComponent: React.FC<WithdrawTestingComponentProps> = ({
  connection,
  wallet,
  connected,
  debugInfo,
  updateDebugInfo,
  addToErrorLog
}) => {
  const { signTransaction } = useWallet()
  const [amount, setAmount] = useState('')
  const [tokenType, setTokenType] = useState('SOL')
  const [transactionStatus, setTransactionStatus] = useState<'idle' | 'preparing' | 'signing' | 'submitting' | 'confirming' | 'success' | 'error'>('idle')
  const [transactionSignature, setTransactionSignature] = useState<string>('')
  const [walletBalance, setWalletBalance] = useState(0)
  const [accountBalance, setAccountBalance] = useState(5.0) // Simulated account balance

  // Get wallet balance
  useEffect(() => {
    const getBalance = async () => {
      if (wallet?.adapter?.publicKey) {
        try {
          const balance = await connection.getBalance(wallet.adapter.publicKey)
          setWalletBalance(balance / 1e9)
        } catch (error) {
          addToErrorLog(`Balance Check Failed: ${error}`)
        }
      }
    }
    getBalance()
  }, [wallet, connection])

  // Set amount to percentage of balance
  const setAmountPercentage = (percentage: number) => {
    if (percentage === 100) {
      setAmount(accountBalance.toString())
    } else {
      setAmount((accountBalance * percentage / 100).toString())
    }
  }

  // Validate amount
  const validateAmount = (value: string): string | null => {
    const num = parseFloat(value)
    if (isNaN(num) || num <= 0) {
      return 'Amount must be greater than 0'
    }
    if (num > accountBalance) {
      return 'Amount exceeds account balance'
    }
    return null
  }

  // Execute withdraw transaction
  const executeWithdraw = async () => {
    if (!wallet?.adapter?.publicKey || !signTransaction) {
      addToErrorLog('Wallet not connected')
      return
    }

    const validationError = validateAmount(amount)
    if (validationError) {
      addToErrorLog(validationError)
      return
    }

    try {
      setTransactionStatus('preparing')
      
      // Create a simple transfer transaction as a placeholder
      // In real implementation, this would call the program's withdraw_collateral instruction
      const transaction = new Transaction()
      
      // Add a simple SOL transfer as placeholder
      const lamports = parseFloat(amount) * 1e9
      transaction.add(
        SystemProgram.transfer({
          fromPubkey: wallet.adapter.publicKey, // Placeholder - would be program account
          toPubkey: wallet.adapter.publicKey,
          lamports: Math.floor(lamports)
        })
      )

      setTransactionStatus('signing')
      updateDebugInfo('lastWithdrawAttempt', new Date().toISOString())
      
      // Sign transaction
      const signedTransaction = await signTransaction(transaction)
      
      setTransactionStatus('submitting')
      
      // Submit transaction
      const signature = await connection.sendRawTransaction(signedTransaction.serialize())
      setTransactionSignature(signature)
      
      setTransactionStatus('confirming')
      
      // Wait for confirmation
      const confirmation = await connection.confirmTransaction(signature, 'confirmed')
      
      if (confirmation.value.err) {
        throw new Error('Transaction failed')
      }
      
      setTransactionStatus('success')
      updateDebugInfo('lastSuccessfulWithdraw', new Date().toISOString())
      addToErrorLog(`Withdrawal successful: ${signature}`)
      
      // Update balances
      const newBalance = await connection.getBalance(wallet.adapter.publicKey)
      setWalletBalance(newBalance / 1e9)
      setAccountBalance(accountBalance - parseFloat(amount))
      
    } catch (error) {
      setTransactionStatus('error')
      addToErrorLog(`Withdrawal Failed: ${error}`)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'success': return 'text-green-400'
      case 'error': return 'text-red-400'
      case 'preparing': case 'signing': case 'submitting': case 'confirming': return 'text-yellow-400'
      default: return 'text-gray-400'
    }
  }

  const getStatusMessage = (status: string) => {
    switch (status) {
      case 'preparing': return 'Preparing transaction...'
      case 'signing': return 'Please sign transaction in wallet...'
      case 'submitting': return 'Submitting transaction...'
      case 'confirming': return 'Waiting for confirmation...'
      case 'success': return 'Withdrawal successful!'
      case 'error': return 'Transaction failed'
      default: return 'Ready to withdraw'
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Withdraw Testing</h2>
        
        {/* Transaction Status */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-medium mb-3">Transaction Status</h3>
          <div className="flex items-center space-x-2 mb-2">
            <span className="text-sm font-medium">Status:</span>
            <span className={`${getStatusColor(transactionStatus)} flex items-center space-x-1`}>
              <span>{getStatusMessage(transactionStatus)}</span>
            </span>
          </div>
          {transactionSignature && (
            <div className="flex items-center space-x-2">
              <span className="text-sm font-medium">Signature:</span>
              <span className="text-gray-300 font-mono text-xs">
                {transactionSignature.slice(0, 8)}...{transactionSignature.slice(-8)}
              </span>
              <button
                onClick={() => window.open(`https://explorer.solana.com/tx/${transactionSignature}?cluster=devnet`, '_blank')}
                className="text-blue-400 hover:text-blue-300 text-xs"
              >
                View on Explorer
              </button>
            </div>
          )}
        </div>

        {/* Withdraw Form */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-medium mb-3">Withdraw Form</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">Amount</label>
              <div className="flex space-x-2">
                <input
                  type="number"
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  placeholder="Enter amount"
                  className="flex-1 px-3 py-2 bg-gray-600 text-white rounded-md border border-gray-500 focus:border-blue-500 focus:outline-none"
                  step="0.0001"
                  min="0"
                  max={accountBalance}
                />
                <select
                  value={tokenType}
                  onChange={(e) => setTokenType(e.target.value)}
                  className="px-3 py-2 bg-gray-600 text-white rounded-md border border-gray-500 focus:border-blue-500 focus:outline-none"
                >
                  <option value="SOL">SOL</option>
                  <option value="USDC">USDC</option>
                  <option value="USDT">USDT</option>
                </select>
              </div>
              <div className="flex space-x-2 mt-2">
                {[25, 50, 75, 100].map(percentage => (
                  <button
                    key={percentage}
                    onClick={() => setAmountPercentage(percentage)}
                    className="px-3 py-1 bg-gray-600 text-white rounded text-sm hover:bg-gray-500"
                  >
                    {percentage}%
                  </button>
                ))}
              </div>
            </div>
            
            <button
              onClick={executeWithdraw}
              disabled={!connected || !amount || transactionStatus !== 'idle' && transactionStatus !== 'success' && transactionStatus !== 'error'}
              className="w-full px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {transactionStatus === 'idle' ? 'Withdraw' : getStatusMessage(transactionStatus)}
            </button>
          </div>
        </div>

        {/* Balance Preview */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-medium mb-3">Balance Preview</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="text-sm font-medium mb-1">Current Wallet Balance</div>
              <div className="text-gray-300">{walletBalance.toFixed(4)} SOL</div>
            </div>
            <div>
              <div className="text-sm font-medium mb-1">Current Account Balance</div>
              <div className="text-gray-300">{accountBalance.toFixed(4)} SOL</div>
            </div>
            <div>
              <div className="text-sm font-medium mb-1">Withdraw Amount</div>
              <div className="text-gray-300">{amount || '0'} SOL</div>
            </div>
            <div>
              <div className="text-sm font-medium mb-1">New Wallet Balance</div>
              <div className="text-gray-300">
                {amount ? (walletBalance + parseFloat(amount)).toFixed(4) : walletBalance.toFixed(4)} SOL
              </div>
            </div>
          </div>
        </div>

        {/* Instructions */}
        <div className="bg-red-900/20 border border-red-500/30 rounded-lg p-4">
          <h3 className="text-lg font-medium mb-2 text-red-400">Instructions</h3>
          <div className="text-sm text-gray-300 space-y-1">
            <p>• Enter the amount you want to withdraw</p>
            <p>• Use percentage buttons for quick amount selection</p>
            <p>• Select the token type (SOL, USDC, USDT)</p>
            <p>• Click "Withdraw" to execute the transaction</p>
            <p>• Sign the transaction in your wallet when prompted</p>
            <p>• Ensure you have sufficient account balance</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default WithdrawTestingComponent
