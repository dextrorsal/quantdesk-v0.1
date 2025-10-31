import React, { useEffect, useState } from 'react'
import { useWallet } from '@solana/wallet-adapter-react'
import { PublicKey, Connection } from '@solana/web3.js'
import { smartContractService } from '../services/smartContractService'

interface DebugPanelProps {
  open: boolean
  onClose: () => void
}

type StepResult = { label: string; ok: boolean; detail?: string }

const DebugPanel: React.FC<DebugPanelProps> = ({ open, onClose }) => {
  const { wallet, publicKey, connected } = useWallet()
  const [steps, setSteps] = useState<StepResult[]>([])
  const [busy, setBusy] = useState(false)
  const [pdas, setPdas] = useState<{ user?: string; vault?: string; collateral?: string }>({})
  const [logs, setLogs] = useState<string>('')
  const [lastRefresh, setLastRefresh] = useState(0)

  const log = (line: string) => {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => prev + (prev ? '\n' : '') + `[${timestamp}] ${line}`)
  }

  const refreshChecks = async () => {
    setSteps([])
    setLastRefresh(Date.now())
    ;(async () => {
      const s: StepResult[] = []
      try {
        s.push({ label: 'Wallet connected', ok: !!connected, detail: publicKey?.toBase58() })
        if (!connected || !publicKey) { setSteps(s); return }

        const endpoint = (import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com') as string
        s.push({ label: 'RPC endpoint', ok: true, detail: endpoint })

        const programId = smartContractService.getProgramId().toBase58()
        const conn = new Connection(endpoint, 'confirmed')
        const info = await conn.getAccountInfo(new PublicKey(programId))
        s.push({ label: 'Program account exists', ok: !!info, detail: programId })

        // Derive PDAs
        const accountIndex = Buffer.alloc(2); accountIndex.writeUInt16LE(0, 0)
        const [userPda] = await PublicKey.findProgramAddress([
          Buffer.from('user_account'), publicKey.toBuffer(), accountIndex
        ], new PublicKey(programId))
        const [vaultPda] = await PublicKey.findProgramAddress([
          Buffer.from('protocol_sol_vault')
        ], new PublicKey(programId))
        const [collPda] = await PublicKey.findProgramAddress([
          Buffer.from('collateral'), publicKey.toBuffer(), Buffer.from('SOL')
        ], new PublicKey(programId))
        setPdas({ user: userPda.toBase58(), vault: vaultPda.toBase58(), collateral: collPda.toBase58() })
        s.push({ label: 'Derived PDAs', ok: true, detail: `user=${userPda.toBase58()} vault=${vaultPda.toBase58()} sol=${collPda.toBase58()}` })

        // Existence checks
        const userInfo = await conn.getAccountInfo(userPda)
        s.push({ label: 'User account exists', ok: !!userInfo })
        const vaultInfo = await conn.getAccountInfo(vaultPda)
        s.push({ label: 'Protocol SOL vault exists', ok: !!vaultInfo })
        const solCollInfo = await conn.getAccountInfo(collPda)
        s.push({ label: 'SOL collateral account exists', ok: !!solCollInfo })
      } catch (e: any) {
        log(`Init failed: ${e?.message || e}`)
      } finally {
        setSteps(s)
      }
    })()
  }

  useEffect(() => {
    if (!open) return
    refreshChecks()
  }, [open, connected, publicKey])

  const copyLogs = () => {
    navigator.clipboard.writeText(logs)
    log('📋 Logs copied to clipboard!')
  }

  const clearLogs = () => {
    setLogs('')
    log('🧹 Logs cleared')
  }

  const doCreateAccount = async () => {
    if (!wallet) return
    setBusy(true)
    try {
      const sig = await smartContractService.createUserAccount(wallet as any)
      log(`createUserAccount: ${sig}`)
    } catch (e: any) {
      log(`createUserAccount ERROR: ${e?.message || e}`)
    } finally { setBusy(false) }
  }

  const doInitVault = async () => {
    if (!wallet) return
    setBusy(true)
    try {
      const sig = await smartContractService.initializeProtocolSOLVault(wallet as any)
      log(`initializeProtocolSOLVault: ${sig}`)
    } catch (e: any) {
      log(`initializeProtocolSOLVault ERROR: ${e?.message || e}`)
    } finally { setBusy(false) }
  }

  const doInitSOLCollateral = async () => {
    if (!wallet) return
    setBusy(true)
    try {
      log('🚀 Starting initializeSOLCollateralAccount...')
      log(`Wallet: ${wallet.adapter?.publicKey?.toBase58()}`)
      log(`⚠️ NOTE: This may fail due to Rust seed mismatch (uses [0] not "SOL")`)
      log(`💡 TIP: You can skip this - DepositNativeSol will init automatically`)
      // Initialize with 0 lamports; program should create account without deposit
      const sig = await (smartContractService as any).initializeSOLCollateralAccount(wallet as any, 0)
      log(`✅ initializeSOLCollateralAccount SUCCESS: ${sig}`)
    } catch (e: any) {
      const errMsg = e?.message || e?.toString() || String(e)
      log(`❌ initializeSOLCollateralAccount ERROR: ${errMsg}`)
      if (e?.logs) {
        log(`📋 Program logs:`)
        e.logs.forEach((l: string) => log(`  ${l}`))
      }
      if (e?.error?.errorMessage) {
        log(`🔍 Error message: ${e.error.errorMessage}`)
      }
      if (e?.error?.errorCode) {
        log(`🔍 Error code: ${JSON.stringify(e.error.errorCode)}`)
      }
      if (e?.programErrorStack) {
        log(`📚 Program error stack:`)
        e.programErrorStack.forEach((s: any) => log(`  ${JSON.stringify(s)}`))
      }
      log(`💡 RECOMMENDATION: Skip this step and use "Test Deposit" instead - it initializes automatically`)
    } finally { setBusy(false) }
  }

  const doTestDeposit = async () => {
    if (!wallet) return
    setBusy(true)
    try {
      log('🚀 Starting depositNativeSOL...')
      log(`Wallet: ${wallet.adapter?.publicKey?.toBase58()}`)
      log(`Amount: 0.01 SOL (${Math.floor(0.01 * 1e9)} lamports)`)
      const sig = await smartContractService.depositNativeSOL(wallet as any, Math.floor(0.01 * 1e9))
      log(`✅ depositNativeSOL SUCCESS: ${sig}`)
      log(`🔗 View on Explorer: https://explorer.solana.com/tx/${sig}?cluster=devnet`)
    } catch (e: any) {
      const errMsg = e?.message || e?.toString() || String(e)
      log(`❌ depositNativeSOL ERROR: ${errMsg}`)
      if (e?.logs) {
        log(`📋 Program logs:`)
        e.logs.forEach((l: string) => log(`  ${l}`))
      }
      if (e?.error?.errorMessage) {
        log(`🔍 Error details: ${e.error.errorMessage}`)
      }
      if (e?.programErrorStack) {
        log(`📚 Program error stack: ${JSON.stringify(e.programErrorStack, null, 2)}`)
      }
    } finally { setBusy(false) }
  }

  if (!open) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.6)' }}>
      <div className="bg-[var(--bg-secondary)] border border-[var(--bg-tertiary)] rounded-xl w-full max-w-3xl p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="font-semibold" style={{ color: 'var(--text-primary)' }}>Debug: Wallet → Account → Vault → Collateral → Deposit</div>
          <div className="flex gap-2">
            <button onClick={refreshChecks} disabled={busy} className="text-sm px-2 py-1 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}>Refresh Checks</button>
            <button onClick={onClose} className="text-sm px-2 py-1 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-muted)' }}>Close</button>
          </div>
        </div>

        <div className="space-y-2 mb-3">
          {steps.map((st, i) => (
            <div key={i} className="flex items-center justify-between p-2 rounded" style={{ background: 'var(--bg-primary)' }}>
              <div style={{ color: 'var(--text-primary)' }}>{st.label}</div>
              <div className={`text-sm ${st.ok ? 'text-green-400' : 'text-red-400'}`}>{st.ok ? 'OK' : 'Missing'}</div>
            </div>
          ))}
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-3">
          <button disabled={busy} onClick={doCreateAccount} className="btn-secondary">Create Account</button>
          <button disabled={busy} onClick={doInitVault} className="btn-secondary">Init Protocol SOL Vault</button>
          <button disabled={busy} onClick={doInitSOLCollateral} className="btn-secondary">Init SOL Collateral</button>
          <button disabled={busy} onClick={doTestDeposit} className="btn-primary">Test Deposit 0.01 SOL</button>
        </div>

        <div>
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm" style={{ color: 'var(--text-muted)' }}>
              PDAs: user={pdas.user} vault={pdas.vault} sol={pdas.collateral}
            </div>
            <div className="flex gap-2">
              <button onClick={copyLogs} className="text-xs px-2 py-1 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}>Copy Logs</button>
              <button onClick={clearLogs} className="text-xs px-2 py-1 rounded" style={{ background: 'var(--bg-tertiary)', color: 'var(--text-primary)' }}>Clear</button>
            </div>
          </div>
          <textarea readOnly className="w-full h-48 p-2 text-xs rounded font-mono" style={{ background: 'var(--bg-primary)', color: 'var(--text-primary)' }} value={logs} />
        </div>
      </div>
    </div>
  )
}

export default DebugPanel


