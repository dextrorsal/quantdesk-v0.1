import React, { useState, useEffect } from 'react'
import { Connection, PublicKey, Keypair } from '@solana/web3.js'

interface RealServiceStatus {
  backend: 'checking' | 'connected' | 'error'
  mikeyAI: 'checking' | 'connected' | 'error'
  dataIngestion: 'checking' | 'connected' | 'error'
  solanaDevnet: 'checking' | 'connected' | 'error'
}

interface ServiceData {
  backend: any
  mikeyAI: any
  dataIngestion: any
  solanaDevnet: any
}

const RealServicesTestingComponent: React.FC = () => {
  const [serviceStatus, setServiceStatus] = useState<RealServiceStatus>({
    backend: 'checking',
    mikeyAI: 'checking',
    dataIngestion: 'checking',
    solanaDevnet: 'checking'
  })
  
  const [serviceData, setServiceData] = useState<ServiceData>({
    backend: null,
    mikeyAI: null,
    dataIngestion: null,
    solanaDevnet: null
  })
  
  const [testWallet, setTestWallet] = useState<Keypair | null>(null)
  const [walletBalance, setWalletBalance] = useState<number>(0)
  const [connection, setConnection] = useState<Connection | null>(null)

  useEffect(() => {
    initializeServices()
  }, [])

  const initializeServices = async () => {
    // Initialize Solana connection
    const solanaConnection = new Connection('https://api.devnet.solana.com', 'confirmed')
    setConnection(solanaConnection)
    
    // Generate test wallet
    const wallet = Keypair.generate()
    setTestWallet(wallet)
    
    // Test all services
    await Promise.all([
      testBackendService(),
      testMikeyAIService(),
      testDataIngestionService(),
      testSolanaDevnet(solanaConnection, wallet)
    ])
  }

  const testBackendService = async () => {
    try {
      setServiceStatus(prev => ({ ...prev, backend: 'checking' }))
      
      // Try /health first (correct endpoint), fallback to /api/health
      let response = await fetch('http://localhost:3002/health')
      if (!response.ok) {
        response = await fetch('http://localhost:3002/api/health')
      }
      
      if (response.ok) {
        const data = await response.json()
        setServiceData(prev => ({ ...prev, backend: data }))
        setServiceStatus(prev => ({ ...prev, backend: 'connected' }))
      } else {
        throw new Error(`Backend responded with ${response.status}`)
      }
    } catch (error) {
      console.error('Backend service error:', error)
      setServiceStatus(prev => ({ ...prev, backend: 'error' }))
    }
  }

  const testMikeyAIService = async () => {
    try {
      setServiceStatus(prev => ({ ...prev, mikeyAI: 'checking' }))
      
      // Use backend proxy to avoid CORS issues (frontend 3001 -> backend 3002 -> MIKEY-AI 3000)
      const response = await fetch('http://localhost:3002/api/mikey/health')
      if (response.ok) {
        const result = await response.json()
        // Backend proxy returns the MIKEY-AI response: { success: true, data: { status: "healthy", ... } }
        const data = result.data || result
        setServiceData(prev => ({ ...prev, mikeyAI: data }))
        setServiceStatus(prev => ({ ...prev, mikeyAI: 'connected' }))
      } else {
        throw new Error(`MIKEY-AI responded with ${response.status}`)
      }
    } catch (error) {
      console.error('MIKEY-AI service error:', error)
      setServiceStatus(prev => ({ ...prev, mikeyAI: 'error' }))
    }
  }

  const testDataIngestionService = async () => {
    try {
      setServiceStatus(prev => ({ ...prev, dataIngestion: 'checking' }))
      
      const response = await fetch('http://localhost:3003/health')
      if (response.ok) {
        const data = await response.json()
        setServiceData(prev => ({ ...prev, dataIngestion: data }))
        setServiceStatus(prev => ({ ...prev, dataIngestion: 'connected' }))
      } else {
        throw new Error(`Data Ingestion responded with ${response.status}`)
      }
    } catch (error) {
      console.error('Data Ingestion service error:', error)
      setServiceStatus(prev => ({ ...prev, dataIngestion: 'error' }))
    }
  }

  const testSolanaDevnet = async (solanaConnection: Connection, wallet: Keypair) => {
    try {
      setServiceStatus(prev => ({ ...prev, solanaDevnet: 'checking' }))
      
      // Test connection
      const version = await solanaConnection.getVersion()
      
      // Get wallet balance
      const balance = await solanaConnection.getBalance(wallet.publicKey)
      setWalletBalance(balance / 1e9)
      
      setServiceData(prev => ({ 
        ...prev, 
        solanaDevnet: { 
          version: version.version,
          balance: balance / 1e9,
          walletAddress: wallet.publicKey.toString()
        } 
      }))
      setServiceStatus(prev => ({ ...prev, solanaDevnet: 'connected' }))
    } catch (error) {
      console.error('Solana Devnet error:', error)
      setServiceStatus(prev => ({ ...prev, solanaDevnet: 'error' }))
    }
  }

  const requestDevnetSOL = async () => {
    if (!connection || !testWallet) return
    
    try {
      const signature = await connection.requestAirdrop(testWallet.publicKey, 2 * 1e9) // 2 SOL
      await connection.confirmTransaction(signature)
      
      // Update balance
      const newBalance = await connection.getBalance(testWallet.publicKey)
      setWalletBalance(newBalance / 1e9)
      
      alert(`✅ Successfully requested 2 SOL airdrop! New balance: ${(newBalance / 1e9).toFixed(4)} SOL`)
    } catch (error) {
      alert(`❌ Airdrop failed: ${error}`)
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
        <h2 className="text-xl font-semibold mb-4">Real Services Integration Testing</h2>
        <p className="text-gray-400 mb-6">
          This component tests against your actual running services instead of mocks.
        </p>
        
        {/* Service Status Overview */}
        <div className="bg-gray-700 rounded-lg p-4 mb-6">
          <h3 className="text-lg font-medium mb-4">Service Status</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(serviceStatus).map(([service, status]) => (
              <div key={service} className="text-center">
                <div className={`text-2xl mb-2 ${getStatusColor(status)}`}>
                  {getStatusIcon(status)}
                </div>
                <div className="text-sm font-medium capitalize">
                  {service.replace(/([A-Z])/g, ' $1').trim()}
                </div>
                <div className={`text-xs ${getStatusColor(status)}`}>
                  {status}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Test Wallet Information */}
        {testWallet && (
          <div className="bg-gray-700 rounded-lg p-4 mb-6">
            <h3 className="text-lg font-medium mb-4">Test Wallet</h3>
            <div className="space-y-2">
              <div>
                <span className="text-sm font-medium">Address:</span>
                <span className="text-gray-300 ml-2 font-mono text-xs break-all">
                  {testWallet.publicKey.toString()}
                </span>
              </div>
              <div>
                <span className="text-sm font-medium">Balance:</span>
                <span className="text-gray-300 ml-2">
                  {walletBalance.toFixed(4)} SOL
                </span>
              </div>
              <div className="mt-4">
                <button
                  onClick={requestDevnetSOL}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
                >
                  Request 2 SOL Airdrop
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Service Details */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Backend Service */}
          <div className="bg-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-3">Backend API (Port 3002)</h3>
            {serviceData.backend ? (
              <div className="space-y-2">
                <div>
                  <span className="text-sm font-medium">Status:</span>
                  <span className={`ml-2 ${getStatusColor(serviceStatus.backend)}`}>
                    {serviceStatus.backend}
                  </span>
                </div>
                <div>
                  <span className="text-sm font-medium">Health:</span>
                  <span className="text-gray-300 ml-2">
                    {serviceData.backend.status}
                  </span>
                </div>
                <div>
                  <span className="text-sm font-medium">Uptime:</span>
                  <span className="text-gray-300 ml-2">
                    {serviceData.backend.uptime || 'N/A'}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-gray-400">No data available</div>
            )}
          </div>

          {/* MIKEY-AI Service */}
          <div className="bg-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-3">MIKEY-AI (Port 3000)</h3>
            {serviceData.mikeyAI ? (
              <div className="space-y-2">
                <div>
                  <span className="text-sm font-medium">Status:</span>
                  <span className={`ml-2 ${getStatusColor(serviceStatus.mikeyAI)}`}>
                    {serviceStatus.mikeyAI}
                  </span>
                </div>
                <div>
                  <span className="text-sm font-medium">Health:</span>
                  <span className="text-gray-300 ml-2">
                    {serviceData.mikeyAI.status}
                  </span>
                </div>
                <div>
                  <span className="text-sm font-medium">Model:</span>
                  <span className="text-gray-300 ml-2">
                    {serviceData.mikeyAI.model || 'N/A'}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-gray-400">No data available</div>
            )}
          </div>

          {/* Data Ingestion Service */}
          <div className="bg-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-3">Data Ingestion (Port 3003)</h3>
            {serviceData.dataIngestion ? (
              <div className="space-y-2">
                <div>
                  <span className="text-sm font-medium">Status:</span>
                  <span className={`ml-2 ${getStatusColor(serviceStatus.dataIngestion)}`}>
                    {serviceStatus.dataIngestion}
                  </span>
                </div>
                <div>
                  <span className="text-sm font-medium">Health:</span>
                  <span className="text-gray-300 ml-2">
                    {serviceData.dataIngestion.status}
                  </span>
                </div>
                <div>
                  <span className="text-sm font-medium">Collectors:</span>
                  <span className="text-gray-300 ml-2">
                    {serviceData.dataIngestion.collectors || 'N/A'}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-gray-400">No data available</div>
            )}
          </div>

          {/* Solana Devnet */}
          <div className="bg-gray-700 rounded-lg p-4">
            <h3 className="text-lg font-medium mb-3">Solana Devnet</h3>
            {serviceData.solanaDevnet ? (
              <div className="space-y-2">
                <div>
                  <span className="text-sm font-medium">Status:</span>
                  <span className={`ml-2 ${getStatusColor(serviceStatus.solanaDevnet)}`}>
                    {serviceStatus.solanaDevnet}
                  </span>
                </div>
                <div>
                  <span className="text-sm font-medium">Version:</span>
                  <span className="text-gray-300 ml-2">
                    {serviceData.solanaDevnet.version}
                  </span>
                </div>
                <div>
                  <span className="text-sm font-medium">Network:</span>
                  <span className="text-gray-300 ml-2">
                    devnet
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-gray-400">No data available</div>
            )}
          </div>
        </div>

        {/* Action Buttons */}
        <div className="bg-gray-700 rounded-lg p-4">
          <h3 className="text-lg font-medium mb-4">Test Actions</h3>
          <div className="flex flex-wrap gap-4">
            <button
              onClick={() => initializeServices()}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Refresh All Services
            </button>
            <button
              onClick={testBackendService}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
            >
              Test Backend API
            </button>
            <button
              onClick={testMikeyAIService}
              className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700"
            >
              Test MIKEY-AI
            </button>
            <button
              onClick={testDataIngestionService}
              className="px-4 py-2 bg-orange-600 text-white rounded-md hover:bg-orange-700"
            >
              Test Data Ingestion
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default RealServicesTestingComponent
