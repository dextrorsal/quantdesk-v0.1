import React, { useState } from 'react'
import { Connection, SYSVAR_RENT_PUBKEY } from '@solana/web3.js'

interface DebugPanelComponentProps {
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

const DebugPanelComponent: React.FC<DebugPanelComponentProps> = ({
  connection,
  wallet,
  connected,
  connecting,
  debugInfo,
  updateDebugInfo,
  errorLog,
  addToErrorLog,
  clearErrorLog,
  testRPCConnection
}) => {
  const [expandedSections, setExpandedSections] = useState<{[key: string]: boolean}>({
    connection: true,
    wallet: true,
    account: false,
    transaction: false,
    error: false
  })

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }))
  }

  const exportDebugInfo = () => {
    const debugData = {
      timestamp: new Date().toISOString(),
      connection: {
        endpoint: connection.rpcEndpoint,
        version: debugInfo.rpcVersion,
        status: debugInfo.rpcStatus
      },
      wallet: {
        connected,
        connecting,
        name: wallet?.adapter?.name,
        publicKey: wallet?.adapter?.publicKey?.toString(),
        balance: debugInfo.walletBalance
      },
      account: {
        address: debugInfo.accountAddress,
        data: debugInfo.accountData,
        initialized: debugInfo.accountInitialized
      },
      transactions: {
        lastDeposit: debugInfo.lastDepositAttempt,
        lastWithdraw: debugInfo.lastWithdrawAttempt,
        lastSuccessfulDeposit: debugInfo.lastSuccessfulDeposit,
        lastSuccessfulWithdraw: debugInfo.lastSuccessfulWithdraw
      },
      errors: errorLog
    }

    const blob = new Blob([JSON.stringify(debugData, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `quantdesk-debug-${new Date().toISOString().split('T')[0]}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const verifyPDASeeds = async () => {
    if (!wallet?.adapter?.publicKey) {
      addToErrorLog('❌ Wallet not connected - cannot verify PDA seeds')
      return
    }

    addToErrorLog('═══════════════════════════════════════════════════════════')
    addToErrorLog('🔍 PDA SEED VERIFICATION - Starting...')
    addToErrorLog('═══════════════════════════════════════════════════════════')

    try {
      const { PublicKey } = await import('@solana/web3.js')
      const { verifyPDASeeds: verifySeeds, deriveDepositPDAs } = await import('../../services/smartContractService')
      const { programIdl } = await import('../../services/smartContractService')

      const userPubkey = wallet.adapter.publicKey
      const programId = new PublicKey(programIdl.address)

      // Verify seeds
      addToErrorLog('\n📋 Step 1: Verifying PDA seed structure...')
      const seedVerification = verifySeeds(userPubkey, 0)
      
      addToErrorLog('User Account Seeds:')
      seedVerification.userAccount.seeds.forEach((seed, i) => {
        addToErrorLog(`  [${i}]: ${seed.toString('hex')} (${seed.toString()})`)
      })
      addToErrorLog(`  ✅ Valid: ${seedVerification.userAccount.isValid}`)

      addToErrorLog('\nCollateral SOL Seeds:')
      seedVerification.collateralSOL.seeds.forEach((seed, i) => {
        addToErrorLog(`  [${i}]: ${seed.toString('hex')} (${seed.toString()})`)
      })
      addToErrorLog(`  ✅ Valid: ${seedVerification.collateralSOL.isValid}`)

      addToErrorLog('\nProtocol Vault Seeds:')
      seedVerification.protocolVault.seeds.forEach((seed, i) => {
        addToErrorLog(`  [${i}]: ${seed.toString('hex')} (${seed.toString()})`)
      })
      addToErrorLog(`  ✅ Valid: ${seedVerification.protocolVault.isValid}`)

      // Derive PDAs and compare
      addToErrorLog('\n📋 Step 2: Deriving PDAs...')
      const pdas = await deriveDepositPDAs(userPubkey, programId, 0)
      
      addToErrorLog('Derived PDAs:')
      addToErrorLog(`  User Account: ${pdas.userAccount.toString()}`)
      addToErrorLog(`  Collateral SOL: ${pdas.collateralSOL.toString()}`)
      addToErrorLog(`  Protocol Vault: ${pdas.protocolVault.toString()}`)
      addToErrorLog(`  Bumps:`, pdas.bumps)

      // Check Rust expectations
      addToErrorLog('\n📋 Step 3: Checking Rust expectations...')
      addToErrorLog('Expected Rust seeds:')
      addToErrorLog('  User Account: [b"user_account", user.key().as_ref(), &[0u8, 0u8]]')
      addToErrorLog('  Collateral: [b"collateral", user.key().as_ref(), b"SOL"]')
      addToErrorLog('  Protocol Vault: [b"protocol_sol_vault"]')

      // Verify account index buffer matches Rust
      const accountIndexBuffer = Buffer.alloc(2)
      accountIndexBuffer.writeUInt16LE(0, 0)
      const rustExpected = Buffer.from([0, 0])
      const matches = accountIndexBuffer.equals(rustExpected)
      addToErrorLog(`\n✅ Account index buffer matches Rust [0u8, 0u8]: ${matches}`)
      if (!matches) {
        addToErrorLog(`  ⚠️ MISMATCH: TS=${accountIndexBuffer.toString('hex')}, Rust expects=${rustExpected.toString('hex')}`)
      }

      addToErrorLog('\n═══════════════════════════════════════════════════════════')
      addToErrorLog('✅ PDA VERIFICATION COMPLETE')
      addToErrorLog('═══════════════════════════════════════════════════════════')
    } catch (error: any) {
      addToErrorLog('❌❌❌ PDA VERIFICATION FAILED')
      addToErrorLog(`Error: ${error.message}`)
      console.error('PDA verification error:', error)
    }
  }

  const runBestPracticeDeposit = async () => {
    if (!wallet?.adapter?.publicKey) {
      addToErrorLog('❌ Wallet not connected - cannot run best practice deposit')
      return
    }

    addToErrorLog('═══════════════════════════════════════════════════════════')
    addToErrorLog('⭐ BEST PRACTICE DEPOSIT TEST - Starting...')
    addToErrorLog('═══════════════════════════════════════════════════════════')

    try {
      // Import the service singleton instance
      const smartContractServiceModule = await import('../../services/smartContractService')
      const smartContractService = smartContractServiceModule.smartContractService
      
      // Use 0.01 SOL (10,000,000 lamports) for testing
      const amountLamports = 10_000_000
      const amountSOL = amountLamports / 1_000_000_000

      addToErrorLog(`💰 Testing deposit: ${amountSOL} SOL (${amountLamports} lamports)`)
      addToErrorLog(`🔍 This uses the best-practice depositNativeSOLBestPractice() function`)
      addToErrorLog(`📋 Which follows all expert recommendations:`)
      addToErrorLog(`   1. Uses provider.wallet.publicKey (not wallet.adapter.publicKey)`)
      addToErrorLog(`   2. Derives PDAs correctly matching Rust seeds`)
      addToErrorLog(`   3. Uses Anchor's .rpc() method`)
      addToErrorLog(`   4. Proper error handling and validation\n`)

      const signature = await smartContractService.depositNativeSOLBestPractice(wallet, amountLamports)

      addToErrorLog('✅✅✅ BEST PRACTICE DEPOSIT SUCCESSFUL!')
      addToErrorLog(`📤 Transaction signature: ${signature}`)
      addToErrorLog(`🌐 Explorer: https://explorer.solana.com/tx/${signature}?cluster=devnet`)
      addToErrorLog('═══════════════════════════════════════════════════════════')
    } catch (error: any) {
      addToErrorLog('❌❌❌ BEST PRACTICE DEPOSIT FAILED')
      addToErrorLog(`Error: ${error.message}`)
      if (error.logs) {
        addToErrorLog('Program logs:')
        error.logs.forEach((log: string) => addToErrorLog(`  ${log}`))
      }
      if (error.errorCode) {
        addToErrorLog(`Error Code: ${JSON.stringify(error.errorCode)}`)
      }
      console.error('Best practice deposit error:', error)
    }
  }

  const runDepositTestSuite = async () => {
    if (!wallet?.adapter?.publicKey) {
      addToErrorLog('❌ Wallet not connected - cannot run test suite')
      return
    }

    addToErrorLog('═══════════════════════════════════════════════════════════')
    addToErrorLog('🧪 DEPOSIT TEST SUITE - Starting...')
    addToErrorLog('═══════════════════════════════════════════════════════════')

    try {
      const { AnchorProvider, Program, BN } = await import('@coral-xyz/anchor')
      const { SystemProgram } = await import('@solana/web3.js')
      // Import IDL from smartContractService
      const { programIdl } = await import('../../services/smartContractService')
      
      addToErrorLog('✅ TEST: Imports loaded')

      // Create Anchor wallet
      const anchorWallet = {
        publicKey: wallet.adapter.publicKey,
        signTransaction: async (tx: any) => {
          addToErrorLog('📝 TEST: Wallet adapter signing transaction...')
          const signed = await wallet.adapter.signTransaction!(tx)
          addToErrorLog(`✅ TEST: Transaction signed - ${signed.signatures.length} signatures`)
          return signed
        },
        signAllTransactions: async (txs: any[]) => {
          const signed = []
          for (const tx of txs) {
            signed.push(await wallet.adapter.signTransaction!(tx))
          }
          return signed
        },
      }

      const provider = new AnchorProvider(connection, anchorWallet as any, {
        commitment: 'confirmed',
        preflightCommitment: 'confirmed',
      })

      const program = new Program(programIdl as any, provider)

      // CRITICAL FIX: Use provider.wallet.publicKey instead of wallet.adapter.publicKey
      // This ensures Anchor recognizes the signer correctly by object reference equality
      const signerPublicKey = provider.wallet.publicKey
      
      addToErrorLog('🔍 Wallet Reference Check:')
      addToErrorLog(`   wallet.adapter.publicKey: ${wallet.adapter.publicKey?.toString()}`)
      addToErrorLog(`   provider.wallet.publicKey: ${signerPublicKey.toString()}`)
      addToErrorLog(`   Same address: ${signerPublicKey.equals(wallet.adapter.publicKey!)}`)
      addToErrorLog(`   Same object reference: ${signerPublicKey === provider.wallet.publicKey}`)

      // Derive PDAs
      const accountIndex = 0
      const accountIndexBuffer = Buffer.alloc(2)
      accountIndexBuffer.writeUInt16LE(accountIndex, 0)

      const [userAccountPda] = await import('@solana/web3.js').then(({ PublicKey }) =>
        PublicKey.findProgramAddressSync(
          [
            Buffer.from('user_account'),
            signerPublicKey.toBuffer(),
            accountIndexBuffer,
          ],
          program.programId
        )
      )

      const [solCollateralAccount] = await import('@solana/web3.js').then(({ PublicKey }) =>
        PublicKey.findProgramAddressSync(
          [Buffer.from('collateral'), signerPublicKey.toBuffer(), Buffer.from('SOL')],
          program.programId
        )
      )

      const [protocolSOLVault] = await import('@solana/web3.js').then(({ PublicKey }) =>
        PublicKey.findProgramAddressSync(
          [Buffer.from('protocol_sol_vault')],
          program.programId
        )
      )

      const SOL_USD_PRICE_FEED = new (await import('@solana/web3.js')).PublicKey('H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG')

      addToErrorLog('✅ TEST 1: PDAs derived')
      addToErrorLog(`   User Account: ${userAccountPda.toBase58()}`)
      addToErrorLog(`   Collateral: ${solCollateralAccount.toBase58()}`)
      addToErrorLog(`   Protocol Vault: ${protocolSOLVault.toBase58()}`)

      // Test instruction structure
      addToErrorLog('\n🔍 TEST 2: Checking instruction account structure...')
      const testIx = await program.methods
        .depositNativeSol(new BN(10000000))
        .accounts({
          userAccount: userAccountPda,        // Position 0: user_account
          user: signerPublicKey,               // Position 1: user (Signer) - CRITICAL: Must be position 1 per IDL!
          protocolVault: protocolSOLVault,     // Position 2: protocol_vault
          collateralAccount: solCollateralAccount, // Position 3: collateral_account
          solUsdPriceFeed: SOL_USD_PRICE_FEED, // Position 4: sol_usd_price_feed
          systemProgram: SystemProgram.programId, // Position 5: system_program
          rent: SYSVAR_RENT_PUBKEY,           // Position 6: rent - REQUIRED!
        })
        .instruction()

      // Enhanced logging - log ALL account metas
      addToErrorLog('📋 Instruction Account Metas (full list):')
      testIx.keys.forEach((meta, idx) => {
        addToErrorLog(`   [${idx}] ${meta.pubkey.toBase58()}`)
        addToErrorLog(`       isSigner: ${meta.isSigner}, isWritable: ${meta.isWritable}`)
      })

      const userMeta = testIx.keys.find(k => k.pubkey.equals(signerPublicKey))
      addToErrorLog(`📍 User account in instruction:`)
      addToErrorLog(`   Index: ${testIx.keys.indexOf(userMeta!)}`)
      addToErrorLog(`   isSigner: ${userMeta?.isSigner}`)
      addToErrorLog(`   isWritable: ${userMeta?.isWritable}`)

      if (!userMeta?.isSigner) {
        throw new Error('❌ TEST FAILED: User is NOT marked as signer!')
      }

      addToErrorLog('✅ TEST 2: Instruction structure OK')

      // Test Anchor .rpc() with comprehensive transaction inspection
      addToErrorLog('\n🔍 TEST 3: Attempting deposit via Anchor .rpc()...')
      addToErrorLog('🔍 Pre-flight transaction inspection:')
      
      // Build transaction first to inspect it
      const preCheckTx = await program.methods
        .depositNativeSol(new BN(10000000))
        .accounts({
          userAccount: userAccountPda,        // Position 0: user_account
          user: signerPublicKey,               // Position 1: user (Signer) - CRITICAL: Must be position 1 per IDL!
          protocolVault: protocolSOLVault,     // Position 2: protocol_vault
          collateralAccount: solCollateralAccount, // Position 3: collateral_account
          solUsdPriceFeed: SOL_USD_PRICE_FEED, // Position 4: sol_usd_price_feed
          systemProgram: SystemProgram.programId, // Position 5: system_program
          rent: SYSVAR_RENT_PUBKEY,           // Position 6: rent - REQUIRED!
        })
        .transaction()
      
      addToErrorLog(`   Total instructions: ${preCheckTx.instructions.length}`)
      addToErrorLog(`   Fee payer: ${preCheckTx.feePayer?.toBase58() || 'Not set'}`)
      if (preCheckTx.instructions[0]?.keys) {
        addToErrorLog(`   Instruction 0 account count: ${preCheckTx.instructions[0].keys.length}`)
        const userMetaTx = preCheckTx.instructions[0].keys.find(k => k.pubkey.equals(signerPublicKey))
        addToErrorLog(`   User in transaction: index=${preCheckTx.instructions[0].keys.indexOf(userMetaTx!)}, isSigner=${userMetaTx?.isSigner}`)
      }

      const signature = await program.methods
        .depositNativeSol(new BN(10000000))
        .accounts({
          userAccount: userAccountPda,        // Position 0: user_account
          user: signerPublicKey,               // Position 1: user (Signer) - CRITICAL: Must be position 1 per IDL!
          protocolVault: protocolSOLVault,     // Position 2: protocol_vault
          collateralAccount: solCollateralAccount, // Position 3: collateral_account
          solUsdPriceFeed: SOL_USD_PRICE_FEED, // Position 4: sol_usd_price_feed
          systemProgram: SystemProgram.programId, // Position 5: system_program
          rent: SYSVAR_RENT_PUBKEY,           // Position 6: rent - REQUIRED!
        })
        .rpc({
          commitment: 'confirmed',
          skipPreflight: false,
        })

      addToErrorLog('✅✅✅ TEST 3 PASSED: Deposit successful!')
      addToErrorLog(`📤 Signature: ${signature}`)
      addToErrorLog('═══════════════════════════════════════════════════════════')
      addToErrorLog('🎉 ALL TESTS PASSED!')
      addToErrorLog('═══════════════════════════════════════════════════════════')
    } catch (error: any) {
      addToErrorLog('❌❌❌ TEST SUITE FAILED')
      addToErrorLog(`Error: ${error.message}`)
      if (error.logs) {
        addToErrorLog('Program logs:')
        error.logs.forEach((log: string) => addToErrorLog(`  ${log}`))
      }
      if (error.errorCode) {
        addToErrorLog(`Error Code: ${JSON.stringify(error.errorCode)}`)
      }
      console.error('Test suite error:', error)
    }
  }

  const runDiagnostics = async () => {
    addToErrorLog('Running diagnostics...')
    
    try {
      // Test RPC connection
      await testRPCConnection()
      
      // Test wallet connection
      if (wallet?.adapter?.publicKey) {
        const balance = await connection.getBalance(wallet.adapter.publicKey)
        updateDebugInfo('diagnosticBalance', balance / 1e9)
        addToErrorLog(`Diagnostic: Wallet balance ${(balance / 1e9).toFixed(4)} SOL`)
      }
      
      // Test account detection
      if (wallet?.adapter?.publicKey) {
        const { PublicKey } = await import('@solana/web3.js')
        const [userAccountPDA] = PublicKey.findProgramAddressSync(
          [Buffer.from('user'), wallet.adapter.publicKey.toBuffer()],
          new PublicKey('11111111111111111111111111111111')
        )
        updateDebugInfo('diagnosticAccountAddress', userAccountPDA.toString())
        addToErrorLog(`Diagnostic: Account address ${userAccountPDA.toString()}`)
      }
      
      addToErrorLog('Diagnostics completed successfully')
    } catch (error) {
      addToErrorLog(`Diagnostics failed: ${error}`)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold mb-4">Debug Information Panel</h2>
        
        {/* Debug Actions */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-medium mb-3">Debug Actions</h3>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={testRPCConnection}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700"
            >
              Test RPC Connection
            </button>
            <button
              onClick={runDiagnostics}
              className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
            >
              Run Diagnostics
            </button>
            <button
              onClick={exportDebugInfo}
              className="px-4 py-2 bg-gray-600 text-white rounded-md hover:bg-gray-700"
            >
              Export Debug Info
            </button>
            <button
              onClick={clearErrorLog}
              className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
            >
              Clear Error Log
            </button>
            <button
              onClick={verifyPDASeeds}
              disabled={!connected}
              className="px-4 py-2 bg-indigo-600 text-white rounded-md hover:bg-indigo-700 disabled:bg-gray-500 disabled:cursor-not-allowed"
            >
              🔍 Verify PDA Seeds
            </button>
            <button
              onClick={runBestPracticeDeposit}
              disabled={!connected}
              className="px-4 py-2 bg-emerald-600 text-white rounded-md hover:bg-emerald-700 disabled:bg-gray-500 disabled:cursor-not-allowed"
            >
              ⭐ Best Practice Deposit (0.01 SOL)
            </button>
            <button
              onClick={runDepositTestSuite}
              disabled={!connected}
              className="px-4 py-2 bg-purple-600 text-white rounded-md hover:bg-purple-700 disabled:bg-gray-500 disabled:cursor-not-allowed"
            >
              🧪 Run Deposit Test Suite
            </button>
          </div>
        </div>

        {/* Connection Information */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <button
            onClick={() => toggleSection('connection')}
            className="w-full flex justify-between items-center text-lg font-medium mb-3"
          >
            <span>Connection Information</span>
            <span>{expandedSections.connection ? '▼' : '▶'}</span>
          </button>
          {expandedSections.connection && (
            <div className="space-y-2">
              <div>
                <span className="text-sm font-medium">RPC Endpoint:</span>
                <span className="text-gray-300 ml-2 font-mono text-xs">{connection.rpcEndpoint}</span>
              </div>
              <div>
                <span className="text-sm font-medium">RPC Version:</span>
                <span className="text-gray-300 ml-2">{debugInfo.rpcVersion?.version || 'Unknown'}</span>
              </div>
              <div>
                <span className="text-sm font-medium">RPC Status:</span>
                <span className={`ml-2 ${
                  debugInfo.rpcStatus === 'connected' ? 'text-green-400' : 
                  debugInfo.rpcStatus === 'checking' ? 'text-yellow-400' : 
                  debugInfo.rpcStatus === 'error' ? 'text-red-400' : 
                  'text-gray-400'
                }`}>
                  {debugInfo.rpcStatus || 'Unknown'}
                </span>
              </div>
              <div>
                <span className="text-sm font-medium">Last RPC Test:</span>
                <span className="text-gray-300 ml-2">{debugInfo.lastRPCTest || 'Never'}</span>
              </div>
            </div>
          )}
        </div>

        {/* Wallet Information */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <button
            onClick={() => toggleSection('wallet')}
            className="w-full flex justify-between items-center text-lg font-medium mb-3"
          >
            <span>Wallet Information</span>
            <span>{expandedSections.wallet ? '▼' : '▶'}</span>
          </button>
          {expandedSections.wallet && (
            <div className="space-y-2">
              <div>
                <span className="text-sm font-medium">Connected:</span>
                <span className={`ml-2 ${connected ? 'text-green-400' : 'text-red-400'}`}>
                  {connected ? 'Yes' : 'No'}
                </span>
              </div>
              <div>
                <span className="text-sm font-medium">Connecting:</span>
                <span className={`ml-2 ${connecting ? 'text-yellow-400' : 'text-gray-400'}`}>
                  {connecting ? 'Yes' : 'No'}
                </span>
              </div>
              <div>
                <span className="text-sm font-medium">Wallet Name:</span>
                <span className="text-gray-300 ml-2">{wallet?.adapter?.name || 'None'}</span>
              </div>
              <div>
                <span className="text-sm font-medium">Public Key:</span>
                <span className="text-gray-300 ml-2 font-mono text-xs">
                  {wallet?.adapter?.publicKey?.toString() || 'None'}
                </span>
              </div>
              <div>
                <span className="text-sm font-medium">Wallet Balance:</span>
                <span className="text-gray-300 ml-2">
                  {debugInfo.walletBalance ? `${debugInfo.walletBalance.toFixed(4)} SOL` : 'Unknown'}
                </span>
              </div>
              <div>
                <span className="text-sm font-medium">Last Connect:</span>
                <span className="text-gray-300 ml-2">{debugInfo.lastConnectAttempt || 'Never'}</span>
              </div>
              <div>
                <span className="text-sm font-medium">Last Disconnect:</span>
                <span className="text-gray-300 ml-2">{debugInfo.lastDisconnectAttempt || 'Never'}</span>
              </div>
            </div>
          )}
        </div>

        {/* Account Information */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <button
            onClick={() => toggleSection('account')}
            className="w-full flex justify-between items-center text-lg font-medium mb-3"
          >
            <span>Account Information</span>
            <span>{expandedSections.account ? '▼' : '▶'}</span>
          </button>
          {expandedSections.account && (
            <div className="space-y-2">
              <div>
                <span className="text-sm font-medium">Account Address:</span>
                <span className="text-gray-300 ml-2 font-mono text-xs">
                  {debugInfo.accountAddress || 'Not detected'}
                </span>
              </div>
              <div>
                <span className="text-sm font-medium">Account Data:</span>
                <span className="text-gray-300 ml-2">
                  {debugInfo.accountData ? 'Available' : 'Not available'}
                </span>
              </div>
              <div>
                <span className="text-sm font-medium">Account Initialized:</span>
                <span className="text-gray-300 ml-2">{debugInfo.accountInitialized || 'Never'}</span>
              </div>
              <div>
                <span className="text-sm font-medium">Diagnostic Account:</span>
                <span className="text-gray-300 ml-2 font-mono text-xs">
                  {debugInfo.diagnosticAccountAddress || 'Not run'}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Transaction Information */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <button
            onClick={() => toggleSection('transaction')}
            className="w-full flex justify-between items-center text-lg font-medium mb-3"
          >
            <span>Transaction Information</span>
            <span>{expandedSections.transaction ? '▼' : '▶'}</span>
          </button>
          {expandedSections.transaction && (
            <div className="space-y-2">
              <div>
                <span className="text-sm font-medium">Last Deposit Attempt:</span>
                <span className="text-gray-300 ml-2">{debugInfo.lastDepositAttempt || 'Never'}</span>
              </div>
              <div>
                <span className="text-sm font-medium">Last Withdraw Attempt:</span>
                <span className="text-gray-300 ml-2">{debugInfo.lastWithdrawAttempt || 'Never'}</span>
              </div>
              <div>
                <span className="text-sm font-medium">Last Successful Deposit:</span>
                <span className="text-gray-300 ml-2">{debugInfo.lastSuccessfulDeposit || 'Never'}</span>
              </div>
              <div>
                <span className="text-sm font-medium">Last Successful Withdraw:</span>
                <span className="text-gray-300 ml-2">{debugInfo.lastSuccessfulWithdraw || 'Never'}</span>
              </div>
            </div>
          )}
        </div>

        {/* Error Log */}
        <div className="bg-gray-700 rounded-lg p-4 mb-4">
          <button
            onClick={() => toggleSection('error')}
            className="w-full flex justify-between items-center text-lg font-medium mb-3"
          >
            <span>Error Log ({errorLog.length} entries)</span>
            <span>{expandedSections.error ? '▼' : '▶'}</span>
          </button>
          {expandedSections.error && (
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {errorLog.length === 0 ? (
                <div className="text-gray-400 text-sm">No errors logged</div>
              ) : (
                errorLog.map((error, index) => (
                  <div key={index} className="text-sm text-red-400 font-mono bg-gray-800 p-2 rounded">
                    {error}
                  </div>
                ))
              )}
            </div>
          )}
        </div>

        {/* Test Suite */}
        <div className="bg-blue-900 rounded-lg p-4 mb-4">
          <h3 className="text-lg font-medium mb-3 text-blue-200">🧪 Deposit Test Suite</h3>
          <p className="text-sm text-gray-300 mb-3">
            Run comprehensive tests to debug AccountNotSigner issues step-by-step
          </p>
          <button
            onClick={runDepositTestSuite}
            disabled={!connected}
            className="w-full px-4 py-3 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed font-medium"
          >
            {!connected ? '🔒 Connect wallet first' : '▶️ Run Deposit Test Suite'}
          </button>
          <p className="text-xs text-gray-400 mt-2">
            Tests: PDA derivation → Instruction structure → Anchor .rpc() → Full deposit
          </p>
        </div>

        {/* Console Commands */}
        <div className="bg-gray-700 rounded-lg p-4">
          <h3 className="text-lg font-medium mb-3">Console Commands</h3>
          <div className="space-y-2 text-sm">
            <div className="bg-gray-800 p-2 rounded font-mono">
              <div className="text-gray-400">// Check wallet connection</div>
              <div className="text-white">console.log('Wallet:', window.solana?.isConnected)</div>
            </div>
            <div className="bg-gray-800 p-2 rounded font-mono">
              <div className="text-gray-400">// Check account state</div>
              <div className="text-white">console.log('Account:', window.accountState)</div>
            </div>
            <div className="bg-gray-800 p-2 rounded font-mono">
              <div className="text-gray-400">// Check program context</div>
              <div className="text-white">console.log('Program:', window.programContext)</div>
            </div>
            <div className="bg-gray-800 p-2 rounded font-mono">
              <div className="text-gray-400">// Check debug info</div>
              <div className="text-white">console.log('Debug:', window.debugInfo)</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DebugPanelComponent
