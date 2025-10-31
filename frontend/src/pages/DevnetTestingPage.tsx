import React, { useState, useEffect } from 'react'
import { useConnection, useWallet } from '@solana/wallet-adapter-react'
import { PublicKey, Connection } from '@solana/web3.js'
// ProgramProvider is supplied at the route level (App.tsx DevProgramWrapper)
import WalletTestingComponent from '../components/devnet-testing/WalletTestingComponent'
import AccountTestingComponent from '../components/devnet-testing/AccountTestingComponent'
import DepositTestingComponent from '../components/devnet-testing/DepositTestingComponent'
import WithdrawTestingComponent from '../components/devnet-testing/WithdrawTestingComponent'
import BalanceDisplayComponent from '../components/devnet-testing/BalanceDisplayComponent'
import DebugPanelComponent from '../components/devnet-testing/DebugPanelComponent'
import RealServicesTestingComponent from '../components/devnet-testing/RealServicesTestingComponent'

const DevnetTestingPage: React.FC = () => {
  const { connection } = useConnection()
  const { wallet, connected, connecting } = useWallet()
  const [activeTab, setActiveTab] = useState('real-services')
  const [debugInfo, setDebugInfo] = useState<any>({})
  const [errorLog, setErrorLog] = useState<string[]>([])

  // Add error to log
  const addToErrorLog = (error: string) => {
    setErrorLog(prev => [...prev, `${new Date().toISOString()}: ${error}`])
  }

  // Update debug info
  const updateDebugInfo = (key: string, value: any) => {
    setDebugInfo(prev => ({ ...prev, [key]: value }))
  }

  // Clear error log
  const clearErrorLog = () => {
    setErrorLog([])
  }

  // Test RPC connection
  const testRPCConnection = async () => {
    try {
      const version = await connection.getVersion()
      updateDebugInfo('rpcVersion', version)
      updateDebugInfo('rpcEndpoint', connection.rpcEndpoint)
      updateDebugInfo('rpcStatus', 'connected')
      updateDebugInfo('lastRPCTest', new Date().toISOString())
    } catch (error) {
      updateDebugInfo('rpcStatus', 'error')
      addToErrorLog(`RPC Test Failed: ${error}`)
    }
  }

  // Initialize debug info
  useEffect(() => {
    updateDebugInfo('connection', connection)
    updateDebugInfo('wallet', wallet)
    updateDebugInfo('connected', connected)
    updateDebugInfo('connecting', connecting)
    // Initialize RPC status
    updateDebugInfo('rpcStatus', 'checking')
    testRPCConnection()
  }, [connection, wallet, connected, connecting])

  const tabs = [
    { id: 'real-services', label: 'Real Services', component: RealServicesTestingComponent },
    { id: 'wallet', label: 'Wallet', component: WalletTestingComponent },
    { id: 'account', label: 'Account', component: AccountTestingComponent },
    { id: 'deposit', label: 'Deposit', component: DepositTestingComponent },
    { id: 'withdraw', label: 'Withdraw', component: WithdrawTestingComponent },
    { id: 'balance', label: 'Balance', component: BalanceDisplayComponent },
    { id: 'debug', label: 'Debug', component: DebugPanelComponent }
  ]

  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-2xl font-bold text-white">Devnet Testing Interface</h1>
          <p className="text-gray-400 mt-1">
            Bare-bones interface for testing QuantDesk contract integration
          </p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto p-4">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar Navigation */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800 rounded-lg p-4">
              <h2 className="text-lg font-semibold mb-4">Testing Components</h2>
              <nav className="space-y-2">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`w-full text-left px-3 py-2 rounded-md transition-colors ${
                      activeTab === tab.id
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                    }`}
                  >
                    {tab.label}
                  </button>
                ))}
              </nav>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-3">
            <div className="bg-gray-800 rounded-lg p-6">
              {ActiveComponent && (
                <ActiveComponent
                  connection={connection}
                  wallet={wallet}
                  connected={connected}
                  connecting={connecting}
                  debugInfo={debugInfo}
                  updateDebugInfo={updateDebugInfo}
                  errorLog={errorLog}
                  addToErrorLog={addToErrorLog}
                  clearErrorLog={clearErrorLog}
                  testRPCConnection={testRPCConnection}
                />
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DevnetTestingPage
