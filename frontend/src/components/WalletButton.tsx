import React, { useState } from 'react'
import { useWallet } from '@solana/wallet-adapter-react'
import { useWalletModal } from '@solana/wallet-adapter-react-ui'
import AccountSlideOut from './AccountSlideOut'

const WalletButton: React.FC = () => {
  const { connected, publicKey } = useWallet()
  const { setVisible } = useWalletModal()
  const [isAccountPanelOpen, setIsAccountPanelOpen] = useState(false)

  const handleClick = () => {
    if (connected) {
      // If wallet is connected, open account management panel
      setIsAccountPanelOpen(true)
    } else {
      // If wallet is not connected, open wallet connection modal
      setVisible(true)
    }
  }

  return (
    <>
      <button
        className="bg-primary-500 hover:bg-primary-600 text-white font-bold py-2 px-4 rounded-lg transition-colors"
        onClick={handleClick}
      >
        {connected ? `Connected: ${publicKey?.toString().slice(0, 4)}...${publicKey?.toString().slice(-4)}` : 'Connect Wallet'}
      </button>

      {/* Account Management Slide-Out Panel */}
      <AccountSlideOut
        isOpen={isAccountPanelOpen}
        onClose={() => setIsAccountPanelOpen(false)}
      />
    </>
  )
}

export default WalletButton
