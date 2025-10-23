import React, { useState, useEffect, useRef } from 'react';
import { X, TrendingDown, AlertCircle } from 'lucide-react';
import { useAccount } from '../contexts/AccountContext';

interface WithdrawModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const WithdrawModal: React.FC<WithdrawModalProps> = ({ isOpen, onClose }) => {
  const { wallet, accountState, fetchAccountState } = useAccount();
  const [amount, setAmount] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [solPrice, setSolPrice] = useState(208); // Default fallback price
  const [validationError, setValidationError] = useState<string | null>(null);
  const isProcessingRef = useRef(false);
  
  // Available collateral in USD
  const availableCollateral = accountState?.totalCollateral || 0;
  
  // Calculate SOL equivalent using real-time price
  const maxSOLAmount = availableCollateral / solPrice;

  useEffect(() => {
    if (!isOpen) {
      setAmount('');
      setValidationError(null);
    }
  }, [isOpen]);

  // Real-time validation
  useEffect(() => {
    if (!amount) {
      setValidationError(null);
      return;
    }

    const numAmount = parseFloat(amount);
    if (isNaN(numAmount) || numAmount <= 0) {
      setValidationError('Enter a valid amount');
      return;
    }

    if (numAmount > maxSOLAmount) {
      setValidationError('Insufficient collateral');
      return;
    }

    if (availableCollateral === 0) {
      setValidationError('No collateral available');
      return;
    }

    setValidationError(null);
  }, [amount, maxSOLAmount, availableCollateral]);

  // Fetch real-time SOL price when modal opens
  useEffect(() => {
    if (isOpen) {
      const fetchSOLPrice = async () => {
        try {
          const response = await fetch('/api/oracle/prices');
          if (response.ok) {
            const data = await response.json();
            if (data.SOL && typeof data.SOL === 'number') {
              setSolPrice(data.SOL);
              console.log('💰 Using real-time SOL price:', data.SOL);
            }
          }
        } catch (error) {
          console.warn('⚠️ Could not fetch real-time SOL price, using fallback:', error);
        }
      };
      
      fetchSOLPrice();
    }
  }, [isOpen]);

  const handleWithdraw = async () => {
    if (!amount || !wallet) return;
    
    // Prevent double-submission
    if (isProcessingRef.current) {
      console.warn('⚠️ Transaction already in progress, ignoring duplicate call');
      return;
    }
    
    const amountNum = parseFloat(amount);
    if (isNaN(amountNum) || amountNum <= 0) {
      alert('Please enter a valid amount');
      return;
    }
    
    // Check if user has enough collateral
    if (amountNum > maxSOLAmount) {
      alert(`Insufficient collateral. Maximum: ${maxSOLAmount.toFixed(6)} SOL`);
      return;
    }
    
    isProcessingRef.current = true;
    setIsLoading(true);
    
    try {
      console.log('🚀 Withdrawing SOL via smart contract...');
      
      const { smartContractService } = await import('../services/smartContractService');
      
      const amountInLamports = Math.floor(amountNum * 1e9);
      if (!Number.isFinite(amountInLamports) || amountInLamports <= 0) {
        throw new Error('Invalid withdrawal amount');
      }
      
      console.log(`💰 Withdrawing ${amountNum} SOL (${amountInLamports} lamports)`);
      
      // Withdraw native SOL using smart contract
      const signature = await smartContractService.withdrawNativeSOL(wallet, amountInLamports);
      console.log('✅ Withdrawal successful:', signature);
      
      // Wait 2 seconds for blockchain confirmation
      console.log('⏳ Waiting 2 seconds for blockchain confirmation...');
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Refresh account state to show updated collateral
      console.log('🔄 Triggering account state refresh...');
      if (fetchAccountState) {
        await fetchAccountState();
      }
      
      window.dispatchEvent(new CustomEvent('refreshAccountState', { 
        detail: { 
          signature, 
          amount: amountNum,
          type: 'withdraw',
          timestamp: Date.now()
        } 
      }));
      
      // Close modal and show success
      onClose();
      alert(`Withdrawal successful! Transaction: ${signature}\nAmount: ${amount} SOL`);
      
    } catch (error: any) {
      console.error('❌ Withdrawal failed:', error);
      
      // Show user-friendly error messages
      let errorMessage = 'Withdrawal failed. Please try again.';
      
      if (error.message?.includes('cancelled by user')) {
        errorMessage = 'Transaction was cancelled.';
      } else if (error.message?.includes('Insufficient collateral')) {
        errorMessage = 'Insufficient collateral for withdrawal. Please check your available balance.';
      } else if (error.message?.includes('expired')) {
        errorMessage = 'Transaction expired. Please try again.';
      } else if (error.message?.includes('CollateralAccountInactive')) {
        errorMessage = 'Collateral account is inactive. Please contact support.';
      } else if (error.message?.includes('Invalid withdrawal amount')) {
        errorMessage = 'Invalid withdrawal amount. Please enter a valid amount.';
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      alert(`Withdrawal failed: ${errorMessage}`);
    } finally {
      setIsLoading(false);
      isProcessingRef.current = false;
    }
  };

  const setMaxAmount = () => {
    setAmount(maxSOLAmount.toFixed(6));
  };

  // Calculate new balances after withdrawal
  const getNewAssetBalance = (): string => {
    const numAmount = parseFloat(amount) || 0;
    const newBalance = maxSOLAmount - numAmount;
    return `${newBalance.toFixed(4)} SOL`;
  };

  const getNewNetAccountBalance = (): string => {
    const numAmount = parseFloat(amount) || 0;
    const newBalanceUSD = availableCollateral - (numAmount * solPrice);
    return `$${newBalanceUSD.toFixed(2)}`;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div 
        className="rounded-2xl p-6 w-full max-w-md mx-4 shadow-2xl"
        style={{
          backgroundColor: 'var(--bg-primary)',
          borderColor: 'var(--border-color)',
          borderWidth: '1px'
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 
            className="text-xl font-semibold flex items-center"
            style={{ color: 'var(--text-primary)' }}
          >
            <TrendingDown className="mr-2 text-red-500" size={24} />
            Withdraw Funds
          </h2>
          <button
            onClick={onClose}
            className="p-2 rounded-lg transition-colors"
            style={{ 
              color: 'var(--text-secondary)',
            }}
            onMouseEnter={(e) => e.currentTarget.style.backgroundColor = 'var(--bg-secondary)'}
            onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'transparent'}
            title="Close withdraw modal"
          >
            <X size={20} />
          </button>
        </div>

        {/* Available Collateral Info */}
        <div 
          className="rounded-lg p-4 mb-4"
          style={{
            backgroundColor: 'var(--bg-secondary)',
            borderColor: 'var(--border-color)',
            borderWidth: '1px'
          }}
        >
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              Available Collateral
            </span>
            <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
              ${availableCollateral.toFixed(2)} USD
            </span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
              Max SOL Withdrawal
            </span>
            <span className="font-semibold text-green-400">
              {maxSOLAmount.toFixed(6)} SOL
            </span>
          </div>
        </div>

        {/* Warning Message */}
        {availableCollateral === 0 && (
          <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3 mb-4 flex items-start">
            <AlertCircle className="text-yellow-500 mr-2 flex-shrink-0 mt-0.5" size={18} />
            <div className="text-sm text-yellow-200">
              No collateral available. Deposit SOL first to enable withdrawals.
            </div>
          </div>
        )}

        {/* Amount Input */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-2">
            <label className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
              Amount (SOL)
            </label>
            <button
              onClick={setMaxAmount}
              className="text-xs transition-colors"
              style={{ 
                color: availableCollateral === 0 ? 'var(--text-tertiary)' : 'var(--primary-500)',
                opacity: availableCollateral === 0 ? 0.5 : 1
              }}
              disabled={availableCollateral === 0}
            >
              MAX
            </button>
          </div>
          <div className="relative">
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="0.00"
              step="0.01"
              min="0"
              max={maxSOLAmount}
              disabled={availableCollateral === 0 || isLoading}
              className="w-full rounded-lg px-4 py-3 text-lg focus:outline-none transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              style={{
                backgroundColor: 'var(--bg-secondary)',
                borderColor: 'var(--border-color)',
                borderWidth: '1px',
                color: 'var(--text-primary)'
              }}
            />
            <div className="absolute right-4 top-1/2 transform -translate-y-1/2">
              <span className="text-sm font-medium" style={{ color: 'var(--text-tertiary)' }}>
                SOL
              </span>
            </div>
          </div>
          {amount && parseFloat(amount) > 0 && (
            <div className="mt-2 text-xs" style={{ color: 'var(--text-secondary)' }}>
              ≈ ${(parseFloat(amount) * solPrice).toFixed(2)} USD
            </div>
          )}
          
          {/* Quick Amount Buttons - Like Drift */}
          <div className="flex space-x-2 mt-3">
            <button 
              onClick={() => setAmount((maxSOLAmount * 0.25).toString())}
              className="px-3 py-1 rounded hover:bg-[var(--bg-secondary)] transition-colors text-sm"
              style={{ backgroundColor: 'var(--bg-primary)' }}
              disabled={availableCollateral === 0}
            >
              25%
            </button>
            <button 
              onClick={() => setAmount((maxSOLAmount * 0.5).toString())}
              className="px-3 py-1 rounded hover:bg-[var(--bg-secondary)] transition-colors text-sm"
              style={{ backgroundColor: 'var(--bg-primary)' }}
              disabled={availableCollateral === 0}
            >
              50%
            </button>
            <button 
              onClick={() => setAmount(maxSOLAmount.toString())}
              className="px-3 py-1 rounded hover:bg-[var(--bg-secondary)] transition-colors text-sm"
              style={{ backgroundColor: 'var(--bg-primary)' }}
              disabled={availableCollateral === 0}
            >
              Max
            </button>
          </div>
        </div>

        {/* Real-time Balance Calculations - Like Drift */}
        {amount && parseFloat(amount) > 0 && (
          <div className="mb-6 space-y-3">
            <div className="flex items-center justify-between p-3 bg-[var(--bg-secondary)] rounded-lg">
              <div className="text-sm text-[var(--text-secondary)]">New Asset Balance</div>
              <div className="text-sm font-medium text-[var(--text-primary)]">{getNewAssetBalance()}</div>
            </div>
            <div className="flex items-center justify-between p-3 bg-[var(--bg-secondary)] rounded-lg">
              <div className="text-sm text-[var(--text-secondary)]">New Net Account Balance (USD)</div>
              <div className="text-sm font-medium text-[var(--text-primary)]">{getNewNetAccountBalance()}</div>
            </div>
          </div>
        )}

        {/* Account Health Warning */}
        {accountState?.accountHealth && accountState.accountHealth < 50 && (
          <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 mb-4 flex items-start">
            <AlertCircle className="text-red-500 mr-2 flex-shrink-0 mt-0.5" size={18} />
            <div className="text-sm text-red-200">
              Warning: Your account health is low ({accountState.accountHealth.toFixed(1)}%). 
              Withdrawing may increase liquidation risk if you have open positions.
            </div>
          </div>
        )}

        {/* Withdraw Button */}
        <button
          onClick={handleWithdraw}
          disabled={!amount || parseFloat(amount) <= 0 || isLoading || availableCollateral === 0 || !!validationError}
          className="w-full font-semibold py-3 px-4 rounded-lg transition-colors flex items-center justify-center disabled:cursor-not-allowed"
          style={{
            backgroundColor: (!amount || parseFloat(amount) <= 0 || isLoading || availableCollateral === 0 || !!validationError) 
              ? 'var(--bg-tertiary)' 
              : '#dc2626',
            color: 'white',
            opacity: (!amount || parseFloat(amount) <= 0 || isLoading || availableCollateral === 0 || !!validationError) ? 0.5 : 1
          }}
          onMouseEnter={(e) => {
            if (amount && parseFloat(amount) > 0 && !isLoading && availableCollateral > 0 && !validationError) {
              e.currentTarget.style.backgroundColor = '#b91c1c';
            }
          }}
          onMouseLeave={(e) => {
            if (amount && parseFloat(amount) > 0 && !isLoading && availableCollateral > 0 && !validationError) {
              e.currentTarget.style.backgroundColor = '#dc2626';
            }
          }}
        >
          {isLoading ? (
            <>
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Processing...
            </>
          ) : (
            <>
              <TrendingDown className="mr-2" size={20} />
              Withdraw SOL
            </>
          )}
        </button>

        {/* Validation Error - Like Drift */}
        {validationError && (
          <div className="mt-3 flex items-center space-x-2 text-sm text-red-400">
            <div className="w-4 h-4 rounded-full bg-red-400 flex items-center justify-center">
              <span className="text-white text-xs">!</span>
            </div>
            <span>{validationError}</span>
          </div>
        )}

        {/* Info Text */}
        <div className="mt-4 text-xs text-center" style={{ color: 'var(--text-secondary)' }}>
          Withdrawals are processed immediately on-chain.
          <br />
          Make sure you have enough SOL for gas fees (~0.001 SOL).
        </div>
      </div>
    </div>
  );
};

