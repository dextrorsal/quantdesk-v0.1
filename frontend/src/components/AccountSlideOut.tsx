import React, { useState, useEffect, useRef } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { useAccount } from '../contexts/AccountContext';
import { X, User, DollarSign, TrendingUp, Settings, LogOut, Copy, ExternalLink } from 'lucide-react';
import { DepositModal } from './DepositModal';
import { WithdrawModal } from './WithdrawModal';

// Account Slide-Out Panel Component
// This creates a slide-out panel similar to Drift's right sidebar for account management
// Uses QuantDesk's design system with CSS variables and theme support

interface AccountSlideOutProps {
  isOpen: boolean;
  onClose: () => void;
}

const AccountSlideOut: React.FC<AccountSlideOutProps> = ({ isOpen, onClose }) => {
  const { connected, publicKey, disconnect } = useWallet();
  const { accountState, totalBalance, accountHealth, isAtRisk, createAccount, loading, fetchAccountState } = useAccount();
  const [showDepositModal, setShowDepositModal] = useState(false);
  const [showWithdrawModal, setShowWithdrawModal] = useState(false);
  const [isCreatingAccount, setIsCreatingAccount] = useState(false);
  const isCreatingAccountRef = useRef(false); // Prevent double-submission

  // Fetch account state when slide-out opens (using quick check to avoid signature prompts)
  useEffect(() => {
    if (isOpen && connected && publicKey) {
      console.log('🔍 AccountSlideOut: Panel opened, performing quick account check...');
      // Use quick check instead of full fetchAccountState to avoid signature prompts
      const quickCheck = async () => {
        try {
          const { smartContractService } = await import('../services/smartContractService');
          const quickState = await smartContractService.getQuickAccountState(publicKey.toString());
          console.log('✅ AccountSlideOut: Quick state result:', quickState);
          
          // Only fetch detailed state if we have an account and need to show balance
          if (quickState.exists && quickState.hasCollateral) {
            console.log('🔍 AccountSlideOut: Account has collateral, fetching detailed state...');
            fetchAccountState();
          }
        } catch (error) {
          console.warn('⚠️ AccountSlideOut: Quick check failed, falling back to full fetch:', error);
          fetchAccountState();
        }
      };
      
      quickCheck();
    }
  }, [isOpen, connected, publicKey, fetchAccountState]);

  // Listen for account state refresh events
  useEffect(() => {
    const handleRefresh = () => {
      console.log('🔄 AccountSlideOut: Received refresh event, updating account state...');
      fetchAccountState();
    };

    window.addEventListener('refreshAccountState', handleRefresh);
    return () => window.removeEventListener('refreshAccountState', handleRefresh);
  }, [fetchAccountState]);

  // Auto-refresh every 10 seconds when panel is open
  useEffect(() => {
    if (!isOpen || !connected || !publicKey) return;

    console.log('🔄 AccountSlideOut: Setting up auto-refresh timer...');
    const interval = setInterval(() => {
      console.log('🔄 AccountSlideOut: Auto-refreshing account state...');
      fetchAccountState();
    }, 10000); // Refresh every 10 seconds

    return () => {
      console.log('🔄 AccountSlideOut: Clearing auto-refresh timer...');
      clearInterval(interval);
    };
  }, [isOpen, connected, publicKey, fetchAccountState]);

  const handleWithdraw = () => {
    console.log('🖱️ AccountSlideOut: Withdraw button clicked');
    setShowWithdrawModal(true);
  };

  const handleCreateAccount = async () => {
    console.log('🖱️ AccountSlideOut: Create Account button clicked');
    
    // Prevent double-submission
    if (isCreatingAccountRef.current) {
      console.warn('⚠️ Account creation already in progress, ignoring duplicate call');
      return;
    }
    
    isCreatingAccountRef.current = true;
    setIsCreatingAccount(true);
    try {
      console.log('📞 AccountSlideOut: Calling createAccount...');
      await createAccount();
      console.log('✅ AccountSlideOut: Account creation completed successfully');
    } catch (error) {
      console.error('❌ AccountSlideOut: Error creating account:', error);
      console.error('Error details:', {
        message: error instanceof Error ? error.message : 'Unknown error',
        stack: error instanceof Error ? error.stack : undefined,
        name: error instanceof Error ? error.name : 'Unknown'
      });
    } finally {
      setIsCreatingAccount(false);
      isCreatingAccountRef.current = false; // Reset the guard
      console.log('🏁 AccountSlideOut: handleCreateAccount completed');
    }
  };

  const handleDisconnect = () => {
    disconnect();
    onClose();
  };

  const copyAddress = () => {
    if (publicKey) {
      navigator.clipboard.writeText(publicKey.toString());
    }
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 z-40"
        onClick={onClose}
        style={{ backgroundColor: 'rgba(0, 0, 0, 0.5)' }}
      />
      
      {/* Slide-out Panel */}
      <div 
        className="fixed right-0 top-0 h-full w-80 z-50 transform transition-transform duration-300 ease-in-out"
        style={{ 
          backgroundColor: 'var(--bg-secondary)',
          borderLeft: '1px solid var(--bg-tertiary)',
          fontFamily: 'Inter, system-ui, sans-serif'
        }}
      >
        <div className="flex flex-col h-full">
          {/* Header */}
          <div 
            className="flex items-center justify-between p-4"
            style={{ borderBottom: '1px solid var(--bg-tertiary)' }}
          >
            <h2 
              className="font-semibold"
              style={{ color: 'var(--text-primary)' }}
            >
              Account
            </h2>
            <button
              onClick={onClose}
              style={{ color: 'var(--text-muted)' }}
              onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-primary)'}
              onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Content */}
          <div className="flex-1 overflow-y-auto p-4">
            {loading ? (
              <div className="flex items-center justify-center h-32">
                <div 
                  className="w-8 h-8 border-2 border-t-transparent rounded-full animate-spin"
                  style={{ borderColor: 'var(--primary-500)' }}
                ></div>
              </div>
            ) : (
              <>
                {/* Wallet Info */}
                <div className="mb-6">
                  <div className="flex items-center space-x-3 mb-3">
                    <div 
                      className="w-10 h-10 rounded-full flex items-center justify-center"
                      style={{ 
                        background: `linear-gradient(135deg, var(--primary-500), var(--primary-600))`
                      }}
                    >
                      <User className="w-5 h-5 text-white" />
                    </div>
                    <div className="flex-1">
                      <div 
                        className="font-medium"
                        style={{ color: 'var(--text-primary)' }}
                      >
                        {publicKey?.toString().slice(0, 8)}...{publicKey?.toString().slice(-8)}
                      </div>
                      <div 
                        className="text-sm"
                        style={{ color: 'var(--success-500)' }}
                      >
                        Connected
                      </div>
                    </div>
                    <button
                      onClick={copyAddress}
                      style={{ color: 'var(--text-muted)' }}
                      onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-primary)'}
                      onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
                    >
                      <Copy className="w-4 h-4" />
                    </button>
                  </div>
                  
                  <div className="flex space-x-2">
                    <button
                      onClick={copyAddress}
                      className="btn-secondary flex-1 text-sm py-2 px-3"
                    >
                      Copy Address
                    </button>
                    <button
                      onClick={() => window.open(`https://solscan.io/account/${publicKey?.toString()}`, '_blank')}
                      className="btn-secondary p-2"
                    >
                      <ExternalLink className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Account Status */}
                {!accountState?.exists ? (
                  <div className="mb-6">
                    <div className="card">
                      <div className="text-center">
                        <div 
                          className="mb-3"
                          style={{ color: 'var(--text-secondary)' }}
                        >
                          Create your QuantDesk trading account to start trading
                        </div>
                        <button
                          onClick={handleCreateAccount}
                          disabled={isCreatingAccount}
                          className="btn-primary w-full"
                        >
                          {isCreatingAccount ? (
                            <>
                              <div 
                                className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin inline-block mr-2"
                              ></div>
                              Creating Account...
                            </>
                          ) : (
                            'Create Account'
                          )}
                        </button>
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* Balance Info */}
                    <div className="mb-6">
                      <div className="card">
                        <div className="flex items-center justify-between mb-3">
                          <span style={{ color: 'var(--text-muted)' }}>Total Collateral</span>
                          <div className="text-right">
                            <div 
                              className="font-bold"
                              style={{ color: 'var(--text-primary)' }}
                            >
                              ${(accountState?.totalCollateral || 0).toFixed(2)} USD
                            </div>
                            <div 
                              className="text-sm"
                              style={{ color: 'var(--text-muted)' }}
                            >
                              ≈{((accountState?.totalCollateral || 0) / 208).toFixed(4)} SOL
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center justify-between mb-3">
                          <span style={{ color: 'var(--text-muted)' }}>Account Health</span>
                          <span 
                            className="font-bold"
                            style={{ color: isAtRisk ? 'var(--danger-500)' : 'var(--success-500)' }}
                          >
                            {(accountHealth / 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex items-center justify-between mb-3">
                          <span style={{ color: 'var(--text-muted)' }}>Trading Status</span>
                          <span 
                            className="font-bold"
                            style={{ color: accountState?.canTrade ? 'var(--success-500)' : 'var(--text-muted)' }}
                          >
                            {accountState?.canTrade ? 'Active' : 'Inactive'}
                          </span>
                        </div>
                        <div className="flex items-center justify-between">
                          <span style={{ color: 'var(--text-muted)' }}>Positions</span>
                          <span style={{ color: 'var(--text-primary)' }}>
                            {accountState?.totalPositions || 0}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Deposit Section */}
                    {!accountState.canTrade && (
                      <div className="mb-6">
                        <div 
                          className="rounded-lg p-4"
                          style={{ 
                            background: `linear-gradient(135deg, var(--primary-500), var(--primary-600))`
                          }}
                        >
                          <div className="text-center">
                            <div className="text-white font-bold text-lg mb-2">USDC 12.23% APY</div>
                            <div className="text-white text-sm mb-3">
                              Earn APY on USDC with or without trading.
                            </div>
                            <button
                              onClick={() => setShowDepositModal(true)}
                              className="bg-white text-black font-bold py-2 px-4 rounded-lg hover:bg-gray-100 transition-colors"
                              style={{ color: 'var(--primary-600)' }}
                            >
                              Deposit Now!
                            </button>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* Account Actions */}
                    <div className="space-y-3 mb-6">
                      <button
                        onClick={() => setShowDepositModal(true)}
                        className="btn-primary w-full flex items-center justify-center space-x-2"
                      >
                        <DollarSign className="w-4 h-4" />
                        <span>Deposit</span>
                      </button>
                      
                      <button 
                        className={`w-full flex items-center justify-center space-x-2 ${
                          (accountState?.totalCollateral || 0) > 0 
                            ? 'btn-secondary' 
                            : 'btn-disabled'
                        }`}
                        onClick={handleWithdraw}
                        disabled={(accountState?.totalCollateral || 0) <= 0}
                      >
                        <TrendingUp className="w-4 h-4" />
                        <span>Withdraw</span>
                      </button>
                    </div>
                  </>
                )}

                {/* Settings */}
                <div 
                  className="pt-4"
                  style={{ borderTop: '1px solid var(--bg-tertiary)' }}
                >
                  <div className="space-y-2">
                    <button 
                      onClick={() => {
                        console.log('🔄 AccountSlideOut: Manual refresh triggered');
                        fetchAccountState();
                      }}
                      className="btn-secondary w-full flex items-center justify-center space-x-2"
                      disabled={loading}
                    >
                      <div 
                        className={`w-4 h-4 border-2 border-t-transparent rounded-full ${loading ? 'animate-spin' : ''}`}
                        style={{ borderColor: 'var(--text-primary)' }}
                      ></div>
                      <span>{loading ? 'Refreshing...' : 'Refresh Balance'}</span>
                    </button>
                    
                    <button className="btn-secondary w-full flex items-center justify-center space-x-2">
                      <Settings className="w-4 h-4" />
                      <span>Settings</span>
                    </button>
                  </div>
                </div>
              </>
            )}
          </div>

          {/* Footer */}
          <div 
            className="p-4"
            style={{ borderTop: '1px solid var(--bg-tertiary)' }}
          >
            <button
              onClick={handleDisconnect}
              className="btn-danger w-full flex items-center justify-center space-x-2"
            >
              <LogOut className="w-4 h-4" />
              <span>Disconnect</span>
            </button>
          </div>
        </div>
      </div>

      {/* Deposit Modal */}
      {showDepositModal && (
        <DepositModal
          isOpen={showDepositModal}
          onClose={() => setShowDepositModal(false)}
        />
      )}

      {/* Withdraw Modal */}
      {showWithdrawModal && (
        <WithdrawModal
          isOpen={showWithdrawModal}
          onClose={() => setShowWithdrawModal(false)}
        />
      )}
    </>
  );
};

export default AccountSlideOut;