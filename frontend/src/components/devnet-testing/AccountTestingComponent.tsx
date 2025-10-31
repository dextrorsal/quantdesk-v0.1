import React, { useState, useEffect } from 'react'
import { Connection, PublicKey } from '@solana/web3.js'
import { useWallet } from '@solana/wallet-adapter-react'

interface AccountTestingComponentProps {
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

const AccountTestingComponent: React.FC<AccountTestingComponentProps> = ({
  connection,
  wallet,
  connected,
  debugInfo,
  updateDebugInfo,
  addToErrorLog
}) => {
  const [accountStatus, setAccountStatus] = useState<'checking' | 'found' | 'notFound' | 'error'>('checking')
  const [accountData, setAccountData] = useState<any>(null)
  const [accountAddress, setAccountAddress] = useState<string>('')
  const [isInitializing, setIsInitializing] = useState(false)

  // Check if user account exists
  const checkAccountExists = async () => {
    if (!wallet?.adapter?.publicKey) {
      setAccountStatus('error')
      addToErrorLog('No wallet connected')
      return
    }

    try {
      setAccountStatus('checking')
      
      // Import program IDL to get correct program ID and PDA derivation
      const { programIdl } = await import('../../services/smartContractService')
      const programId = new PublicKey(programIdl.address)
      
      // Use CORRECT PDA derivation: ['user_account', userPubkey, accountIndexBuffer]
      // This matches what createUserAccount uses
      const userPubkey = wallet.adapter.publicKey
      const accountIndex = 0
      const accountIndexBuffer = Buffer.alloc(2)
      accountIndexBuffer.writeUInt16LE(accountIndex, 0)
      
      const [userAccountPDA] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('user_account'),
          userPubkey.toBuffer(),
          accountIndexBuffer
        ],
        programId
      )
      
      setAccountAddress(userAccountPDA.toString())
      updateDebugInfo('accountAddress', userAccountPDA.toString())
      
      // Try to fetch account data
      try {
        const accountInfo = await connection.getAccountInfo(userAccountPDA)
        if (accountInfo) {
          setAccountStatus('found')
          setAccountData(accountInfo)
          updateDebugInfo('accountData', accountInfo)
          addToErrorLog(`✅ Account found at: ${userAccountPDA.toString()}`)
        } else {
          setAccountStatus('notFound')
          setAccountData(null)
          addToErrorLog(`⚠️ Account not found at: ${userAccountPDA.toString()}`)
        }
      } catch (error) {
        setAccountStatus('notFound')
        setAccountData(null)
        addToErrorLog(`❌ Error checking account: ${error}`)
      }
    } catch (error) {
      setAccountStatus('error')
      addToErrorLog(`Account Check Failed: ${error}`)
    }
  }

  // Initialize account using real smartContractService
  const initializeAccount = async () => {
    if (!wallet?.adapter?.publicKey) {
      addToErrorLog('No wallet connected')
      return
    }

    try {
      setIsInitializing(true)
      addToErrorLog('🔄 Creating user account on-chain...')
      
      // Use real smartContractService to create account
      const { smartContractService } = await import('../../services/smartContractService')
      
      // Check if account already exists first
      const exists = await smartContractService.checkUserAccount(wallet.adapter.publicKey.toString())
      if (exists) {
        addToErrorLog('⚠️ Account already exists! Refreshing...')
        await checkAccountExists()
        setIsInitializing(false)
        return
      }
      
      // Create account
      const signature = await smartContractService.createUserAccount(wallet)
      addToErrorLog(`✅ Account created! Transaction: ${signature}`)
      
      // Wait a moment for confirmation, then refresh
      await new Promise(resolve => setTimeout(resolve, 2000))
      await checkAccountExists()
      
      updateDebugInfo('accountInitialized', new Date().toISOString())
    } catch (error: any) {
      addToErrorLog(`❌ Account Initialization Failed: ${error.message || error}`)
      setAccountStatus('error')
    } finally {
      setIsInitializing(false)
    }
  }

  // Check account on wallet connection
  useEffect(() => {
    if (connected && wallet?.adapter?.publicKey) {
      checkAccountExists()
    }
  }, [connected, wallet])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'found': return 'text-green-400'
      case 'notFound': return 'text-yellow-400'
      case 'checking': return 'text-blue-400'
      case 'error': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'found': return '✅'
      case 'notFound': return '⚠️'
      case 'checking': return '⏳'
      case 'error': return '❌'
      default: return '❓'
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Account Management Testing</h2>
        
        {/* Account Status */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-medium mb-3">Account Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-sm font-medium">Account Status:</span>
                <span className={`${getStatusColor(accountStatus)} flex items-center space-x-1`}>
                  <span>{getStatusIcon(accountStatus)}</span>
                  <span className="capitalize">{accountStatus.replace(/([A-Z])/g, ' $1').trim()}</span>
                </span>
              </div>
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-sm font-medium">Account Address:</span>
                <span className="text-gray-300 font-mono text-xs">
                  {accountAddress ? 
                    `${accountAddress.slice(0, 8)}...${accountAddress.slice(-8)}` : 
                    'Not detected'
                  }
                </span>
              </div>
            </div>
            <div>
              <div className="flex items-center space-x-2 mb-2">
                <span className="text-sm font-medium">Account Balance:</span>
                <span className="text-gray-300">
                  {accountData ? `${(accountData.lamports / 1e9).toFixed(4)} SOL` : 'N/A'}
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-sm font-medium">Last Check:</span>
                <span className="text-gray-300">{new Date().toLocaleTimeString()}</span>
              </div>
            </div>
          </div>
        </div>

        {/* Account Actions */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-medium mb-3">Account Actions</h3>
          <div className="flex space-x-4">
            <button
              onClick={checkAccountExists}
              disabled={!connected}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Check Account
            </button>
            {accountStatus === 'notFound' && (
              <button
                onClick={initializeAccount}
                disabled={isInitializing || !connected}
                className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isInitializing ? 'Initializing...' : 'Initialize Account'}
              </button>
            )}
            {accountAddress && (
              <button
                onClick={() => window.open(`https://explorer.solana.com/address/${accountAddress}?cluster=devnet`, '_blank')}
                className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
              >
                View on Explorer
              </button>
            )}
          </div>
        </div>

        {/* Account Information */}
        {accountData && (
          <div className="bg-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-3">Account Information</h3>
            <div className="space-y-2">
              <div>
                <span className="text-sm font-medium">Owner:</span>
                <span className="text-gray-300 ml-2 font-mono text-xs">
                  {accountData.owner.toString()}
                </span>
              </div>
              <div>
                <span className="text-sm font-medium">Lamports:</span>
                <span className="text-gray-300 ml-2">{accountData.lamports}</span>
              </div>
              <div>
                <span className="text-sm font-medium">Executable:</span>
                <span className="text-gray-300 ml-2">{accountData.executable ? 'Yes' : 'No'}</span>
              </div>
              <div>
                <span className="text-sm font-medium">Rent Epoch:</span>
                <span className="text-gray-300 ml-2">{accountData.rentEpoch}</span>
              </div>
              <div>
                <span className="text-sm font-medium">Data Length:</span>
                <span className="text-gray-300 ml-2">{accountData.data.length} bytes</span>
              </div>
            </div>
          </div>
        )}

        {/* Instructions */}
        <div className="bg-blue-900/20 border border-blue-500/30 rounded-lg p-4">
          <h3 className="text-lg font-medium mb-2 text-blue-400">Instructions</h3>
          <div className="text-sm text-gray-300 space-y-1">
            <p>• Connect your wallet to check account status</p>
            <p>• If account is not found, click "Initialize Account" to create it</p>
            <p>• Use "View on Explorer" to inspect account on Solana Explorer</p>
            <p>• Account address is derived from your wallet's public key</p>
          </div>
        </div>
      </div>
    </div>
  )
}

export default AccountTestingComponent
