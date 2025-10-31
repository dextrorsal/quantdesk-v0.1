import React from 'react'
import { useWallet } from '@solana/wallet-adapter-react'
import { WalletReadyState } from '@solana/wallet-adapter-base'

interface CustomWalletModalProps {
  open: boolean
  onClose: () => void
}

const tag = (text: string, color: string) => (
  <span style={{
    marginLeft: 8,
    padding: '2px 6px',
    fontSize: 10,
    borderRadius: 8,
    backgroundColor: color,
    color: 'var(--text-primary)'
  }}>{text}</span>
)

const CustomWalletModal: React.FC<CustomWalletModalProps> = ({ open, onClose }) => {
  const { wallets, select, connect, connecting } = useWallet()

  if (!open) return null

  // Helper
  const byName = (name: string) => wallets.find(w => w.adapter.name === name)
  const fallbackIconDataUri = (label: string) => {
    const ch = (label || '?').slice(0, 1).toUpperCase()
    const svg = `<?xml version="1.0" encoding="UTF-8"?><svg xmlns='http://www.w3.org/2000/svg' width='20' height='20'><rect width='100%' height='100%' rx='4' ry='4' fill='%231f2937'/><text x='50%' y='58%' dominant-baseline='middle' text-anchor='middle' font-size='12' fill='white' font-family='Inter, system-ui, Arial'>${ch}</text></svg>`
    return `data:image/svg+xml;utf8,${svg}`
  }

  // Curated list (match Drift-like: Phantom, Solflare, Brave)
  const curatedNames = ['Phantom', 'Solflare', 'Backpack', 'Coinbase Wallet']
  const curatedDetected = curatedNames
    .map(byName)
    .filter(Boolean) as typeof wallets

  const moreDetected = wallets.filter(w => !curatedDetected.some(c => c.adapter.name === w.adapter.name))

  // Jupiter-style curated list of popular wallets with install links and icons
  const popular: { name: string; installUrl: string; icon?: string }[] = [
    { name: 'Backpack', installUrl: 'https://backpack.app', icon: 'https://assets.backpack.app/backpack-logo.svg' },
    { name: 'Glow', installUrl: 'https://glow.app', icon: 'https://glow.app/favicon.ico' },
    { name: 'Exodus', installUrl: 'https://www.exodus.com/download/', icon: 'https://www.exodus.com/assets/images/favicon.ico' },
    { name: 'Nightly', installUrl: 'https://nightly.app', icon: 'https://nightly.app/favicon.ico' },
    { name: 'Backpack (Mobile)', installUrl: 'https://backpack.app', icon: 'https://assets.backpack.app/backpack-logo.svg' },
    { name: 'Brave Wallet', installUrl: 'https://brave.com/wallet/', icon: 'https://brave.com/static-assets/images/brave-favicon.png' },
    { name: 'Ledger Live', installUrl: 'https://www.ledger.com/ledger-live', icon: 'https://www.ledger.com/wp-content/uploads/2021/11/favicon-32x32-1.png' },
    { name: 'OKX Wallet', installUrl: 'https://www.okx.com/web3', icon: 'https://www.okx.com/cdn/assets/imgs/236/3F2D3B4A1F1D6D53.png' },
  ]

  const moreStatic = popular.filter(p => ![...curatedNames, ...moreDetected.map(w => w.adapter.name)].includes(p.name))

  // Static entries for curated wallets not detected
  const curatedStatic = popular.filter(p => curatedNames.includes(p.name) && !curatedDetected.some(w => w.adapter.name === p.name))

  const renderStaticItem = (name: string, installUrl: string, icon?: string) => (
    <div key={name} style={{
      width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      background: 'var(--bg-primary)', border: '1px solid var(--bg-tertiary)', color: 'var(--text-primary)',
      padding: '10px 12px', borderRadius: 8
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <img
          src={icon || fallbackIconDataUri(name)}
          alt={name}
          width={20}
          height={20}
          onError={(e) => { (e.currentTarget as HTMLImageElement).src = fallbackIconDataUri(name) }}
          style={{ borderRadius: 4 }}
        />
        <span style={{ fontSize: 14 }}>{name}</span>
      </div>
      <a href={installUrl} target="_blank" rel="noreferrer" style={{ fontSize: 12, color: 'var(--primary-500)' }}>Get wallet</a>
    </div>
  )

  const renderItem = (w: (typeof wallets)[number]) => (
    <button
      key={w.adapter.name}
      onClick={async () => {
        try {
          await select(w.adapter.name)
          await connect()
          onClose()
        } catch (e) {
          console.warn('Wallet connect failed:', e)
        }
      }}
      disabled={connecting}
      style={{
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        background: 'var(--bg-primary)',
        border: '1px solid var(--bg-tertiary)',
        color: 'var(--text-primary)',
        padding: '10px 12px',
        borderRadius: 8,
        cursor: 'pointer',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
        <img
          src={w.adapter.icon || fallbackIconDataUri(w.adapter.name)}
          alt={w.adapter.name}
          width={20}
          height={20}
          onError={(e) => { (e.currentTarget as HTMLImageElement).src = fallbackIconDataUri(w.adapter.name) }}
          style={{ borderRadius: 4 }}
        />
        <span style={{ fontSize: 14 }}>{w.adapter.name}</span>
        {w.readyState === WalletReadyState.Installed && tag('Detected', 'rgba(82,196,26,0.15)')}
        {w.adapter.name === 'Phantom' && tag('Suggested', 'rgba(59,130,246,0.15)')}
      </div>
    </button>
  )

  return (
    <div style={{
      position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
      display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 99999
    }}>
      <div style={{
        background: 'var(--bg-secondary)',
        border: '1px solid var(--bg-tertiary)',
        width: 420, maxWidth: '90vw', borderRadius: 12, overflow: 'hidden'
      }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '12px 16px', borderBottom: '1px solid var(--bg-tertiary)' }}>
          <div style={{ color: 'var(--text-primary)', fontWeight: 600 }}>Connect Wallet</div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 18 }}>×</button>
        </div>
        <div style={{ padding: 16, display: 'grid', gap: 10 }}>
          <div style={{ display: 'grid', gap: 8 }}>
            {curatedDetected.map(renderItem)}
            {curatedStatic.map(w => renderStaticItem(w.name, w.installUrl, w.icon))}
          </div>
          <details open>
            <summary style={{ color: 'var(--text-muted)', cursor: 'pointer', margin: '8px 0' }}>More wallets</summary>
            <div style={{ display: 'grid', gap: 8, marginTop: 8 }}>
              {moreDetected.map(renderItem)}
              {moreStatic.map(w => renderStaticItem(w.name, w.installUrl, w.icon))}
            </div>
          </details>
        </div>
      </div>
    </div>
  )
}

export default CustomWalletModal


