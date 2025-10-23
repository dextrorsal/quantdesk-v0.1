import React, { useState } from 'react'
import { useWallet } from '@solana/wallet-adapter-react'
import { useWalletModal } from '@solana/wallet-adapter-react-ui'
import { useWalletAuth } from '../hooks/useWalletAuth'
import AccountSlideOut from './AccountSlideOut'

const WalletButton: React.FC = () => {
  const { connected, publicKey } = useWallet()
  const { setVisible } = useWalletModal()
  const { authenticate, isAuthenticated } = useWalletAuth()
  const [isAccountPanelOpen, setIsAccountPanelOpen] = useState(false)

  const handleClick = async () => {
    if (!connected) {
      // If wallet is not connected, open wallet connection modal
      setVisible(true)
      return
    }

    console.log('🔐 Wallet button clicked - starting authentication...')
    console.log('📊 Current auth state:', { isAuthenticated, connected, publicKey: publicKey?.toString() })

    // Always open immediately when connected for snappy UX
    setIsAccountPanelOpen(true)

    // Authenticate in background (non-blocking)
    try {
      if (!isAuthenticated) {
        console.log('🚀 Starting authentication process...')
        const authResult = await authenticate()
        console.log('✅ Authentication result:', authResult)
      } else {
        console.log('✅ Already authenticated')
      }
    } catch (error) {
      console.error('❌ Authentication error:', error)
    }
  }

  return (
    <>
      <button
        className="bg-primary-500 hover:bg-primary-600 text-white font-bold py-2 px-4 transition-colors"
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
