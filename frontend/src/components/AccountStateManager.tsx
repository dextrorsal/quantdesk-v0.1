import React, { useState } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { useAccount } from '../contexts/AccountContext';
import AccountSlideOut from './AccountSlideOut';
import { User } from 'lucide-react';

// Account State Manager Component
// This component integrates with your existing trading interface
// Account management is handled by a slide-out panel (like Drift)

const AccountStateManager: React.FC = () => {
  const { connected } = useWallet();
  const { accountState, loading, error } = useAccount();
  const [isAccountPanelOpen, setIsAccountPanelOpen] = useState(false);

  // Show loading spinner while fetching account state
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
      </div>
    );
  }

  // Show error if there's an issue
  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900">
        <div className="text-center">
          <div className="text-red-500 text-xl mb-4">Error</div>
          <div className="text-gray-300 mb-4">{error}</div>
          <button 
            onClick={() => window.location.reload()}
            className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  // If wallet is not connected, show connection prompt
  if (!connected) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900 text-white p-4">
        <div className="text-center">
          <div className="w-16 h-16 bg-gradient-to-r from-purple-500 to-pink-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <User className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-2xl font-bold mb-2">Connect Your Wallet</h1>
          <p className="text-gray-400 mb-6">Connect your Solana wallet to start trading</p>
          <button
            onClick={() => setIsAccountPanelOpen(true)}
            className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white font-bold py-3 px-6 rounded-lg shadow-lg transform hover:scale-105 transition-all duration-300"
          >
            Connect Wallet
          </button>
        </div>
      </div>
    );
  }

  // If wallet is connected, show your existing trading interface
  // The account management will be handled by the slide-out panel
  return (
    <div className="relative">
      {/* Your existing trading interface goes here */}
      {/* For now, showing a placeholder that indicates the account panel is available */}
      <div className="min-h-screen bg-gray-900 text-white p-4">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold">QuantDesk Trading</h1>
          <button
            onClick={() => setIsAccountPanelOpen(true)}
            className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded-lg flex items-center space-x-2"
          >
            <User className="w-4 h-4" />
            <span>Account</span>
          </button>
        </div>
        
        <div className="text-center py-20">
          <h2 className="text-xl font-semibold mb-4">Your Trading Interface</h2>
          <p className="text-gray-400 mb-6">
            This is where your existing beautiful trading interface will be displayed.
            The account management is handled by the slide-out panel on the right.
          </p>
          <div className="bg-gray-800 rounded-lg p-6 max-w-md mx-auto">
            <div className="text-green-400 mb-2">✓ Wallet Connected</div>
            <div className="text-blue-400 mb-2">✓ Account State: {accountState?.exists ? 'Created' : 'Not Created'}</div>
            <div className="text-purple-400">✓ Ready for Trading</div>
          </div>
        </div>
      </div>

      {/* Account Slide-Out Panel */}
      <AccountSlideOut
        isOpen={isAccountPanelOpen}
        onClose={() => setIsAccountPanelOpen(false)}
      />
    </div>
  );
};

export default AccountStateManager;
