import React from 'react';
import { useAccount } from '../contexts/AccountContext';
import { useWallet } from '@solana/wallet-adapter-react';

const DebugAccountState: React.FC = () => {
  console.log('🔍 DebugAccountState: Component rendering...');
  
  const { connected, publicKey } = useWallet();
  const { 
    accountState, 
    collateralAccounts, 
    positions, 
    totalBalance, 
    accountHealth,
    loading,
    error,
    refreshData
  } = useAccount();
  
  console.log('🔍 DebugAccountState: Wallet state:', { connected, publicKey: publicKey?.toString() });
  console.log('🔍 DebugAccountState: Account state:', { accountState, loading, error });
  
  const testDeposit = async () => {
    if (!connected || !publicKey) return;
    
    try {
      const { smartContractService } = await import('../services/smartContractService');
      console.log('🧪 Testing deposit of 0.01 SOL...');
      
      // First check if account exists, if not create it
      if (!accountState?.exists) {
        console.log('🔍 Account does not exist, creating it first...');
        const { useAccount } = await import('../contexts/AccountContext');
        // We need to get the createAccount function from the context
        // For now, let's use the smart contract service directly
        const { SmartContractService } = await import('../services/smartContractService');
        const signature = await SmartContractService.getInstance().createUserAccount({ adapter: { publicKey } } as any);
        console.log('✅ Account created successfully:', signature);
      }
      
      // Now deposit the SOL
      console.log('💰 Depositing 0.01 SOL...');
      const amountInLamports = 0.01 * 1e9; // 0.01 SOL
      const signature = await smartContractService.depositNativeSOL(
        { adapter: { publicKey } } as any,
        amountInLamports
      );
      
      console.log('✅ Test deposit successful:', signature);
      
      // Refresh account state to show updated balance
      console.log('🔄 Refreshing account state...');
      await refreshData();
      console.log('✅ Account state refreshed');
      
      alert('Test deposit successful! Check your balance.');
    } catch (error) {
      console.error('❌ Test deposit failed:', error);
      alert('Test deposit failed: ' + error.message);
    }
  };

  if (!connected || !publicKey) {
    return (
      <div className="p-4 bg-red-900 text-white rounded border-4 border-yellow-400">
        <h3 className="font-bold text-xl">🔍 Debug: Wallet Not Connected</h3>
        <p>Connect your wallet to see account state</p>
        <p className="text-sm mt-2">Component is rendering but wallet is not connected</p>
      </div>
    );
  }

  return (
    <div className="p-4 bg-green-800 text-white rounded border-4 border-yellow-400">
      <h3 className="font-bold text-xl mb-2">🔍 Debug Account State</h3>
      
      <div className="space-y-2 text-sm">
        <div>
          <strong>Wallet:</strong> {publicKey.toString()}
        </div>
        
        <div>
          <strong>Loading:</strong> {loading ? 'Yes' : 'No'}
        </div>
        
        <div>
          <strong>Error:</strong> {error || 'None'}
        </div>
        
        <div>
          <strong>Account Exists:</strong> {accountState?.exists ? 'Yes' : 'No'}
        </div>
        
        <div>
          <strong>Can Trade:</strong> {accountState?.canTrade ? 'Yes' : 'No'}
        </div>
        
        <div>
          <strong>Total Collateral:</strong> ${((accountState?.totalCollateral || 0)).toFixed(2)} USD
        </div>
        
        <div>
          <strong>Total Balance:</strong> ${totalBalance}
        </div>
        
        <div>
          <strong>Account Health:</strong> {accountHealth}%
        </div>
        
        <div>
          <strong>Collateral Accounts:</strong> {collateralAccounts.length}
        </div>
        
        <div>
          <strong>Positions:</strong> {positions.length}
        </div>
        
        {collateralAccounts.length > 0 && (
          <div>
            <strong>Collateral Details:</strong>
            <ul className="ml-4">
              {collateralAccounts.map((acc, idx) => (
                <li key={idx}>
                  {acc.assetType}: {acc.amount} ({acc.valueUsd} USD)
                </li>
              ))}
            </ul>
          </div>
        )}
        
        {positions.length > 0 && (
          <div>
            <strong>Position Details:</strong>
            <ul className="ml-4">
              {positions.map((pos, idx) => (
                <li key={idx}>
                  {pos.market}: {pos.size} @ ${pos.entryPrice} (PnL: ${pos.unrealizedPnL})
                </li>
              ))}
            </ul>
          </div>
        )}
        
        <div className="mt-4">
          <button 
            onClick={testDeposit}
            className="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded text-sm"
          >
            Test Deposit 0.01 SOL
          </button>
        </div>
      </div>
    </div>
  );
};

export default DebugAccountState;
