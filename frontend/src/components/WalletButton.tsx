import React from 'react'
import { useWallet } from '@solana/wallet-adapter-react'
import { useWalletModal } from '@solana/wallet-adapter-react-ui'

const WalletButton: React.FC = () => {
  const { connected, publicKey } = useWallet()
  const { setVisible } = useWalletModal()

  return (
    <button
      className="bg-primary-500 hover:bg-primary-600 text-white font-bold py-2 px-4 rounded-lg transition-colors"
      onClick={() => setVisible(true)}
    >
      {connected ? `Connected: ${publicKey?.toString().slice(0, 4)}...${publicKey?.toString().slice(-4)}` : 'Connect Wallet'}
    </button>
  )
}

export default WalletButton
