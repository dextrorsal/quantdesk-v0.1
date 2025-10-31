import React, { useState, useCallback, useEffect } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { useAccount } from '../../contexts/AccountContext';
import { SmartContractService } from '../../services/smartContractService';
import { 
  X, 
  User, 
  DollarSign, 
  TrendingUp, 
  Settings, 
  LogOut, 
  Copy, 
  ExternalLink,
  ArrowDown,
  ArrowUp,
  AlertTriangle,
  CheckCircle,
  Loader2
} from 'lucide-react';

/**
 * Enhanced Account Management Panel
 * Features:
 * - Professional deposit/withdraw flows
 * - Real-time balance updates
 * - Better error handling
 * - Transaction status tracking
 * - Professional styling
 */

interface EnhancedAccountPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

interface TransactionStatus {
  id: string;
  type: 'deposit' | 'withdraw';
  amount: number;
  status: 'pending' | 'success' | 'error';
  timestamp: number;
  error?: string;
}

const EnhancedAccountPanel: React.FC<EnhancedAccountPanelProps> = ({ isOpen, onClose }) => {
  const { connected, publicKey, disconnect } = useWallet();
  const { accountState, totalBalance, accountHealth, createAccount, loading, fetchAccountState } = useAccount();
  
  const [activeTab, setActiveTab] = useState<'overview' | 'deposit' | 'withdraw'>('overview');
  const [depositAmount, setDepositAmount] = useState('');
  const [withdrawAmount, setWithdrawAmount] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [transactions, setTransactions] = useState<TransactionStatus[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Auto-refresh account state when panel opens
  useEffect(() => {
    if (isOpen && connected && publicKey) {
      fetchAccountState();
    }
  }, [isOpen, connected, publicKey, fetchAccountState]);

  // Handle deposit
  const handleDeposit = useCallback(async () => {
    if (!publicKey || !depositAmount) return;

    const amount = parseFloat(depositAmount);
    if (amount <= 0) {
      setError('Please enter a valid amount');
      return;
    }

    setIsProcessing(true);
    setError(null);

    const transactionId = `deposit_${Date.now()}`;
    const newTransaction: TransactionStatus = {
      id: transactionId,
      type: 'deposit',
      amount,
      status: 'pending',
      timestamp: Date.now()
    };

    setTransactions(prev => [newTransaction, ...prev]);

    try {
      const smartContractService = SmartContractService.getInstance();
      const signature = await smartContractService.depositNativeSOL(
        { adapter: { publicKey } } as any,
        amount * 1e9 // Convert to lamports
      );

      // Update transaction status
      setTransactions(prev => prev.map(tx => 
        tx.id === transactionId 
          ? { ...tx, status: 'success' as const }
          : tx
      ));

      // Refresh account state
      await fetchAccountState();
      setDepositAmount('');

    } catch (err: any) {
      console.error('Deposit failed:', err);
      
      // Update transaction status
      setTransactions(prev => prev.map(tx => 
        tx.id === transactionId 
          ? { ...tx, status: 'error' as const, error: err.message }
          : tx
      ));

      setError(err.message || 'Deposit failed');
    } finally {
      setIsProcessing(false);
    }
  }, [publicKey, depositAmount, fetchAccountState]);

  // Handle withdraw
  const handleWithdraw = useCallback(async () => {
    if (!publicKey || !withdrawAmount) return;

    const amount = parseFloat(withdrawAmount);
    if (amount <= 0) {
      setError('Please enter a valid amount');
      return;
    }

    if (amount > (accountState?.totalCollateral || 0)) {
      setError('Insufficient balance');
      return;
    }

    setIsProcessing(true);
    setError(null);

    const transactionId = `withdraw_${Date.now()}`;
    const newTransaction: TransactionStatus = {
      id: transactionId,
      type: 'withdraw',
      amount,
      status: 'pending',
      timestamp: Date.now()
    };

    setTransactions(prev => [newTransaction, ...prev]);

    try {
      const smartContractService = SmartContractService.getInstance();
      const signature = await smartContractService.withdrawNativeSOL(
        { adapter: { publicKey } } as any,
        amount * 1e9 // Convert to lamports
      );

      // Update transaction status
      setTransactions(prev => prev.map(tx => 
        tx.id === transactionId 
          ? { ...tx, status: 'success' as const }
          : tx
      ));

      // Refresh account state
      await fetchAccountState();
      setWithdrawAmount('');

    } catch (err: any) {
      console.error('Withdraw failed:', err);
      
      // Update transaction status
      setTransactions(prev => prev.map(tx => 
        tx.id === transactionId 
          ? { ...tx, status: 'error' as const, error: err.message }
          : tx
      ));

      setError(err.message || 'Withdraw failed');
    } finally {
      setIsProcessing(false);
    }
  }, [publicKey, withdrawAmount, accountState, fetchAccountState]);

  // Handle account creation
  const handleCreateAccount = useCallback(async () => {
    if (!publicKey) return;

    setIsProcessing(true);
    setError(null);

    try {
      await createAccount();
    } catch (err: any) {
      setError(err.message || 'Account creation failed');
    } finally {
      setIsProcessing(false);
    }
  }, [publicKey, createAccount]);

  // Copy wallet address
  const copyWalletAddress = useCallback(() => {
    if (publicKey) {
      navigator.clipboard.writeText(publicKey.toString());
    }
  }, [publicKey]);

  // Format number
  const formatNumber = (num: number, decimals: number = 2) => {
    return num.toFixed(decimals);
  };

  // Get health color
  const getHealthColor = (health: number) => {
    if (health > 80) return 'text-green-400';
    if (health > 50) return 'text-yellow-400';
    return 'text-red-400';
  };

  // Get transaction status icon
  const getTransactionIcon = (status: TransactionStatus['status']) => {
    switch (status) {
      case 'pending': return <Loader2 className="w-4 h-4 animate-spin text-blue-400" />;
      case 'success': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'error': return <AlertTriangle className="w-4 h-4 text-red-400" />;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />
      
      {/* Panel */}
      <div className="absolute right-0 top-0 h-full w-96 bg-gray-900 border-l border-gray-700 shadow-2xl">
        {/* Header */}
        <div className="p-6 border-b border-gray-700">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-white">Account</h2>
            <button
              onClick={onClose}
              className="p-2 text-gray-400 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="h-full overflow-y-auto">
          {/* Wallet Info */}
          {publicKey && (
            <div className="p-6 border-b border-gray-700">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-blue-600 rounded-full flex items-center justify-center">
                  <User className="w-5 h-5 text-white" />
                </div>
                <div className="flex-1">
                  <p className="text-white font-semibold">
                    {publicKey.toString().slice(0, 4)}...{publicKey.toString().slice(-4)}
                  </p>
                  <button
                    onClick={copyWalletAddress}
                    className="text-gray-400 text-sm hover:text-white transition-colors flex items-center space-x-1"
                  >
                    <Copy className="w-3 h-3" />
                    <span>Copy address</span>
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Account Status */}
          {accountState ? (
            <div className="p-6 border-b border-gray-700">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Account Health</span>
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${
                      accountHealth > 80 ? 'bg-green-500' : 
                      accountHealth > 50 ? 'bg-yellow-500' : 'bg-red-500'
                    }`} />
                    <span className={`font-semibold ${getHealthColor(accountHealth)}`}>
                      {accountHealth.toFixed(1)}%
                    </span>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Total Balance</span>
                  <span className="text-white font-semibold">
                    ${formatNumber(totalBalance)}
                  </span>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-gray-400">Available Margin</span>
                  <span className="text-white font-semibold">
                    ${formatNumber((accountState.totalCollateral || 0) * 0.95)}
                  </span>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-6 border-b border-gray-700">
              <div className="text-center">
                <p className="text-gray-400 mb-4">No account found</p>
                <button
                  onClick={handleCreateAccount}
                  disabled={isProcessing}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isProcessing ? (
                    <div className="flex items-center justify-center">
                      <Loader2 className="w-4 h-4 animate-spin mr-2" />
                      Creating Account...
                    </div>
                  ) : (
                    'Create Account'
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Tab Navigation */}
          <div className="flex border-b border-gray-700">
            <button
              onClick={() => setActiveTab('overview')}
              className={`flex-1 py-3 text-sm font-semibold transition-colors ${
                activeTab === 'overview'
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setActiveTab('deposit')}
              className={`flex-1 py-3 text-sm font-semibold transition-colors ${
                activeTab === 'deposit'
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Deposit
            </button>
            <button
              onClick={() => setActiveTab('withdraw')}
              className={`flex-1 py-3 text-sm font-semibold transition-colors ${
                activeTab === 'withdraw'
                  ? 'text-blue-400 border-b-2 border-blue-400'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Withdraw
            </button>
          </div>

          {/* Tab Content */}
          <div className="p-6">
            {activeTab === 'overview' && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white">Account Overview</h3>
                
                {accountState && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                      <span className="text-gray-400">Collateral</span>
                      <span className="text-white font-semibold">
                        ${formatNumber(accountState.totalCollateral)}
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                      <span className="text-gray-400">Positions</span>
                      <span className="text-white font-semibold">
                        {accountState.totalPositions || 0}
                      </span>
                    </div>
                    
                    <div className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                      <span className="text-gray-400">Orders</span>
                      <span className="text-white font-semibold">
                        {accountState.totalOrders || 0}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'deposit' && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white">Deposit SOL</h3>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Amount (SOL)
                  </label>
                  <input
                    type="number"
                    value={depositAmount}
                    onChange={(e) => setDepositAmount(e.target.value)}
                    placeholder="0.00"
                    step="0.001"
                    min="0"
                    className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                  />
                </div>

                <button
                  onClick={handleDeposit}
                  disabled={!depositAmount || isProcessing}
                  className="w-full bg-green-600 hover:bg-green-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <ArrowDown className="w-4 h-4" />
                      <span>Deposit</span>
                    </>
                  )}
                </button>
              </div>
            )}

            {activeTab === 'withdraw' && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white">Withdraw SOL</h3>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Amount (SOL)
                  </label>
                  <input
                    type="number"
                    value={withdrawAmount}
                    onChange={(e) => setWithdrawAmount(e.target.value)}
                    placeholder="0.00"
                    step="0.001"
                    min="0"
                    max={accountState?.totalCollateral || 0}
                    className="w-full p-3 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-blue-500 focus:outline-none"
                  />
                </div>

                <button
                  onClick={handleWithdraw}
                  disabled={!withdrawAmount || isProcessing}
                  className="w-full bg-red-600 hover:bg-red-700 text-white font-semibold py-3 px-4 rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
                >
                  {isProcessing ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Processing...</span>
                    </>
                  ) : (
                    <>
                      <ArrowUp className="w-4 h-4" />
                      <span>Withdraw</span>
                    </>
                  )}
                </button>
              </div>
            )}
          </div>

          {/* Recent Transactions */}
          {transactions.length > 0 && (
            <div className="p-6 border-t border-gray-700">
              <h3 className="text-lg font-semibold text-white mb-4">Recent Transactions</h3>
              <div className="space-y-2">
                {transactions.slice(0, 5).map((tx) => (
                  <div key={tx.id} className="flex items-center justify-between p-3 bg-gray-800 rounded-lg">
                    <div className="flex items-center space-x-3">
                      {getTransactionIcon(tx.status)}
                      <div>
                        <p className="text-white text-sm font-semibold">
                          {tx.type === 'deposit' ? 'Deposit' : 'Withdraw'}
                        </p>
                        <p className="text-gray-400 text-xs">
                          {new Date(tx.timestamp).toLocaleTimeString()}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-white text-sm font-semibold">
                        {tx.type === 'deposit' ? '+' : '-'}{formatNumber(tx.amount)} SOL
                      </p>
                      {tx.error && (
                        <p className="text-red-400 text-xs">{tx.error}</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="p-6 border-t border-gray-700">
              <div className="p-3 bg-red-900/20 border border-red-500 rounded-lg">
                <div className="flex items-start space-x-2">
                  <AlertTriangle className="w-4 h-4 text-red-400 mt-0.5 flex-shrink-0" />
                  <div>
                    <p className="text-red-400 text-sm font-medium">Error</p>
                    <p className="text-red-300 text-xs mt-1">{error}</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default EnhancedAccountPanel;
