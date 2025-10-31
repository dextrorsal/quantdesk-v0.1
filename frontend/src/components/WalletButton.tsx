import React, { useState } from 'react'
import { useWallet } from '@solana/wallet-adapter-react'
import CustomWalletModal from './CustomWalletModal'
import AccountSlideOut from './AccountSlideOut'

const WalletButton: React.FC = () => {
  const { connected, publicKey } = useWallet()
  const [open, setOpen] = useState(false)
  const [isAccountPanelOpen, setIsAccountPanelOpen] = useState(false)

  const label = connected && publicKey
    ? `${publicKey.toBase58().slice(0, 4)}…${publicKey.toBase58().slice(-4)}`
    : 'Connect Wallet'

  return (
    <>
      <button
        className="bg-primary-500 hover:bg-primary-600 text-white font-bold py-2 px-4 transition-colors"
        onClick={() => {
          if (connected) { setIsAccountPanelOpen(true); return }
          setOpen(true)
        }}
      >
        {label}
      </button>
      <CustomWalletModal open={open} onClose={() => setOpen(false)} />
      <AccountSlideOut isOpen={isAccountPanelOpen} onClose={() => setIsAccountPanelOpen(false)} />
    </>
  )
}

export default WalletButton
