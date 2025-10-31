import React, { useState, useEffect, useRef } from 'react';
import { X, ChevronDown, TrendingUp } from 'lucide-react';
import { useAccount } from '../contexts/AccountContext';
import { getDepositTokens, TokenConfig } from '../config/tokens';
import { balanceService } from '../services/balanceService';

interface DepositModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface Account {
  id: string;
  name: string;
  type: 'master' | 'sub';
  balance: number;
}

export const DepositModal: React.FC<DepositModalProps> = ({ isOpen, onClose }) => {
  const { wallet } = useAccount();
  const [selectedAccount, setSelectedAccount] = useState<Account | null>(null);
  const [selectedToken, setSelectedToken] = useState<TokenConfig | null>(null);
  const [amount, setAmount] = useState('');
  const [isAccountDropdownOpen, setIsAccountDropdownOpen] = useState(false);
  const [isTokenDropdownOpen, setIsTokenDropdownOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const isProcessingRef = useRef(false); // Prevent double-submission
  const [nativeSOLBalance, setNativeSOLBalance] = useState(0);
  const [solPrice, setSolPrice] = useState(200); // Default SOL price
  const [validationError, setValidationError] = useState<string | null>(null);

  // Get tokens from config
  const supportedTokens = getDepositTokens();

  // Simple accounts - just show master account with real balance
  const accounts: Account[] = [
    { id: 'master', name: 'Master Account', type: 'master', balance: Number(nativeSOLBalance.toFixed(9)) },
  ];

  useEffect(() => {
    if (accounts.length > 0) {
      setSelectedAccount(accounts[0]);
    }
    if (supportedTokens.length > 0) {
      setSelectedToken(supportedTokens[0]);
    }
  }, []);

  // Fetch SOL balance when modal opens
  useEffect(() => {
    if (isOpen && wallet?.adapter?.publicKey) {
      fetchSOLBalance();
      fetchSOLPrice();
    }
  }, [isOpen, wallet]);

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

    if (numAmount > nativeSOLBalance) {
      setValidationError('Not enough SOL in your wallet');
      return;
    }

    setValidationError(null);
  }, [amount, nativeSOLBalance]);

  const fetchSOLBalance = async () => {
    if (!wallet?.adapter?.publicKey) return;
    
    try {
      console.log('🔍 Fetching SOL wallet balance...');
      // Get wallet balance for depositing/withdrawing
      const solBalance = await balanceService.getNativeSOLBalance(wallet.adapter.publicKey);
      setNativeSOLBalance(solBalance);
      console.log('✅ SOL wallet balance:', solBalance);
    } catch (error) {
      console.error('❌ Error fetching SOL wallet balance:', error);
      setNativeSOLBalance(0);
    }
  };

  const fetchSOLPrice = async () => {
    try {
      // Get SOL price from our backend API
      const response = await fetch('http://localhost:3002/api/prices');
      const data = await response.json();
      if (data.SOL) {
        setSolPrice(data.SOL);
      }
    } catch (error) {
      console.error('❌ Error fetching SOL price:', error);
      // Keep default price
    }
  };

  const handleDeposit = async () => {
    if (!amount || !wallet) return;
    
    // Prevent double-submission (React Strict Mode, race conditions, etc.)
    if (isProcessingRef.current) {
      console.warn('⚠️ Transaction already in progress, ignoring duplicate call');
      return;
    }
    
    isProcessingRef.current = true;
    setIsLoading(true);
    try {
      console.log('🚀 Depositing SOL via smart contract...');
      
      // Use the smart contract for native SOL deposit
      const { smartContractService } = await import('../services/smartContractService');
      
      const amountInLamports = Math.floor(parseFloat(amount) * 1e9);
      if (!Number.isFinite(amountInLamports) || amountInLamports <= 0) {
        throw new Error('Enter a valid SOL amount');
      }
      
      // Check if user has account, create if needed
      const hasAccount = await smartContractService.checkUserAccount(wallet.adapter.publicKey.toString());
      if (!hasAccount) {
        console.log('🔄 Creating user account...');
        await smartContractService.createUserAccount(wallet);
        console.log('✅ User account created');
      }
      
      // Initialize protocol SOL vault if needed
      console.log('🔄 Checking protocol SOL vault...');
      await smartContractService.initializeProtocolSOLVault(wallet);
      
      // Check protocol vault balance before deposit
      console.log('🔍 Checking protocol vault balance before deposit...');
      const vaultBalanceBefore = await smartContractService.getProtocolVaultBalance();
      console.log('💰 Protocol vault balance before:', vaultBalanceBefore / 1e9, 'SOL');
      
      // Deposit native SOL using smart contract (now handles collateral account automatically)
      const signature = await smartContractService.depositNativeSOL(wallet, amountInLamports);
      console.log('✅ Native SOL deposit successful:', signature);
      
      // Check protocol vault balance after deposit
      console.log('🔍 Checking protocol vault balance after deposit...');
      const vaultBalanceAfter = await smartContractService.getProtocolVaultBalance();
      console.log('💰 Protocol vault balance after:', vaultBalanceAfter / 1e9, 'SOL');
      console.log('📈 Deposit amount received:', (vaultBalanceAfter - vaultBalanceBefore) / 1e9, 'SOL');
      
      // Wait 2 seconds for blockchain confirmation
      console.log('⏳ Waiting 2 seconds for blockchain confirmation...');
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // Refresh balance and account state
      console.log('🔄 Refreshing SOL balance...');
      await fetchSOLBalance();
      
      // Refresh account state to show updated collateral
      console.log('🔄 Triggering account state refresh...');
      window.dispatchEvent(new CustomEvent('refreshAccountState', { 
        detail: { 
          signature, 
          amount: parseFloat(amount),
          timestamp: Date.now()
        } 
      }));
      
      // Close modal and show success
      onClose();
      
      // Show success message with better formatting
      const successMessage = `✅ Deposit Successful!\n\nAmount: ${amount} SOL\nTransaction: ${signature && signature !== 'unknown-signature' ? `${signature.slice(0, 8)}...${signature.slice(-8)}` : 'Transaction completed'}\n\nYour collateral has been updated.`;
      alert(successMessage);
      
    } catch (error) {
      console.error('❌ Deposit failed:', error);
      alert(`Deposit failed: ${error.message}`);
    } finally {
      setIsLoading(false);
      isProcessingRef.current = false; // Reset the guard
    }
  };

  const formatBalance = (balance: number): string => {
    return balance.toFixed(6);
  };

  const formatUSDValue = (amount: string): string => {
    const numAmount = parseFloat(amount) || 0;
    return `$${(numAmount * solPrice).toFixed(2)}`;
  };

  // Calculate new balances after deposit
  const getNewAssetBalance = (): string => {
    const numAmount = parseFloat(amount) || 0;
    return `${numAmount.toFixed(4)} SOL`;
  };

  const getNewNetAccountBalance = (): string => {
    const numAmount = parseFloat(amount) || 0;
    return `$${(numAmount * solPrice).toFixed(2)}`;
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-[var(--bg-primary)] rounded-2xl p-6 w-full max-w-md mx-4 shadow-2xl border border-[var(--border-color)]">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-[var(--text-primary)]">Deposit Funds</h2>
          <button
            onClick={onClose}
            className="p-2 hover:bg-[var(--bg-secondary)] rounded-lg transition-colors"
            title="Close deposit modal"
          >
            <X className="w-5 h-5 text-[var(--text-secondary)]" />
          </button>
        </div>

        {/* Account Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
            Select Account
          </label>
          <div className="relative">
            <button
              onClick={() => setIsAccountDropdownOpen(!isAccountDropdownOpen)}
              className="w-full p-3 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg flex items-center justify-between text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] transition-colors"
            >
              <span>{selectedAccount?.name || 'Select Account'}</span>
              <ChevronDown className={`w-4 h-4 transition-transform ${isAccountDropdownOpen ? 'rotate-180' : ''}`} />
            </button>
            
            {isAccountDropdownOpen && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg shadow-lg z-10">
                {accounts.map((account) => (
                  <button
                    key={account.id}
                    onClick={() => {
                      setSelectedAccount(account);
                      setIsAccountDropdownOpen(false);
                    }}
                    className="w-full p-3 text-left hover:bg-[var(--bg-tertiary)] transition-colors first:rounded-t-lg last:rounded-b-lg"
                  >
                    <div className="flex items-center justify-between">
                      <span className="text-[var(--text-primary)]">{account.name}</span>
                      <span className="text-sm text-[var(--text-secondary)]">
                        {formatBalance(account.balance)} SOL
                      </span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Asset Selection */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
            Select Asset
          </label>
          <div className="relative">
            <button
              onClick={() => setIsTokenDropdownOpen(!isTokenDropdownOpen)}
              className="w-full p-3 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg flex items-center justify-between text-[var(--text-primary)] hover:bg-[var(--bg-tertiary)] transition-colors"
            >
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                  {selectedToken?.symbol?.charAt(0) || 'S'}
                </div>
                <div className="text-left">
                  <div className="font-medium">{selectedToken?.symbol || 'SOL'}</div>
                  <div className="text-xs text-[var(--text-secondary)]">{selectedToken?.name || 'Solana'}</div>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <div className="text-right">
                  <div className="text-sm font-medium text-green-400">{selectedToken?.apy || 8.45}% APY</div>
                  <div className="text-xs text-[var(--text-secondary)]">Earning</div>
                </div>
                <ChevronDown className={`w-4 h-4 transition-transform ${isTokenDropdownOpen ? 'rotate-180' : ''}`} />
              </div>
            </button>
            
            {isTokenDropdownOpen && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg shadow-lg z-10 max-h-64 overflow-y-auto">
                {supportedTokens.map((token) => (
                  <button
                    key={token.symbol}
                    onClick={() => {
                      setSelectedToken(token);
                      setIsTokenDropdownOpen(false);
                    }}
                    className="w-full p-3 text-left hover:bg-[var(--bg-tertiary)] transition-colors flex items-center justify-between"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-full flex items-center justify-center text-white font-bold text-sm">
                        {token.symbol.charAt(0)}
                      </div>
                      <div>
                        <div className="font-medium text-[var(--text-primary)]">{token.symbol}</div>
                        <div className="text-xs text-[var(--text-secondary)]">{token.name}</div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-medium text-green-400">{token.apy}% APY</div>
                      <div className="text-xs text-[var(--text-secondary)]">Earning</div>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Amount Input */}
        <div className="mb-6">
          <label className="block text-sm font-medium text-[var(--text-secondary)] mb-2">
            Amount
          </label>
          <div className="relative">
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="0.00"
              className="w-full p-3 bg-[var(--bg-secondary)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] placeholder-[var(--text-secondary)] focus:outline-none focus:ring-2 focus:ring-[var(--primary-500)] focus:border-transparent"
            />
            <div className="absolute right-3 top-1/2 transform -translate-y-1/2 text-sm text-[var(--text-secondary)]">
              {selectedToken?.symbol || 'SOL'}
            </div>
          </div>
          
          {/* Balance and USD Value */}
          <div className="flex items-center justify-between mt-2">
            <div className="text-sm text-[var(--text-secondary)]">
              Balance: {formatBalance(nativeSOLBalance)} SOL
            </div>
            <div className="text-sm text-[var(--text-secondary)]">
              {formatUSDValue(amount)}
            </div>
          </div>
          
          {/* Quick Amount Buttons */}
          <div className="flex space-x-2 mt-3">
            <button 
              onClick={() => setAmount((nativeSOLBalance * 0.25).toString())}
              className="px-3 py-1 rounded hover:bg-[var(--bg-secondary)] transition-colors text-sm"
              style={{ backgroundColor: 'var(--bg-primary)' }}
            >
              25%
            </button>
            <button 
              onClick={() => setAmount((nativeSOLBalance * 0.5).toString())}
              className="px-3 py-1 rounded hover:bg-[var(--bg-secondary)] transition-colors text-sm"
              style={{ backgroundColor: 'var(--bg-primary)' }}
            >
              50%
            </button>
            <button 
              onClick={() => setAmount(nativeSOLBalance.toString())}
              className="px-3 py-1 rounded hover:bg-[var(--bg-secondary)] transition-colors text-sm"
              style={{ backgroundColor: 'var(--bg-primary)' }}
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

        {/* Deposit Button */}
        <button
          onClick={handleDeposit}
          disabled={!amount || isLoading || !!validationError}
          className="w-full py-3 bg-[var(--primary-500)] text-white font-medium rounded-lg hover:bg-[var(--primary-600)] disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center space-x-2"
        >
          {isLoading ? (
            <>
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
              <span>Depositing...</span>
            </>
          ) : (
            <>
              <TrendingUp className="w-4 h-4" />
              <span>Deposit {selectedToken?.symbol || 'SOL'}</span>
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

        {/* Info */}
        <div className="mt-4 p-3 bg-[var(--bg-secondary)] rounded-lg">
          <div className="flex items-center space-x-2 text-sm text-[var(--text-secondary)]">
            <TrendingUp className="w-4 h-4 text-green-400" />
            <span>Earning {selectedToken?.apy || 8.45}% APY on your deposit</span>
          </div>
        </div>
      </div>
    </div>
  );
};