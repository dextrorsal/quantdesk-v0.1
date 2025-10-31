import React, { useState, useEffect } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { PublicKey } from '@solana/web3.js';
import { unifiedBalanceService, UnifiedBalance } from '../services/unifiedBalanceService';
import { smartContractService } from '../services/smartContractService';

const TestBalanceWindow: React.FC = () => {
  const { connected, publicKey, wallet } = useWallet();
  const [balance, setBalance] = useState<UnifiedBalance | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [accountStatus, setAccountStatus] = useState<string>('Checking...');

  useEffect(() => {
    if (connected && publicKey) {
      fetchBalance();
    }
  }, [connected, publicKey]);

  const fetchBalance = async () => {
    if (!publicKey) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const unified = await unifiedBalanceService.getUnifiedBalance(publicKey);
      setBalance(unified);
      
      // Check if user account exists
      const accountState = await smartContractService.getQuickAccountState(publicKey.toString());
      setAccountStatus(accountState.exists ? '✅ Account Exists' : '❌ No Account');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch balance');
    } finally {
      setLoading(false);
    }
  };

  const initializeAccount = async () => {
    if (!wallet || !publicKey) {
      alert('Please connect your wallet first');
      return;
    }

    setLoading(true);
    try {
      const signature = await smartContractService.createUserAccount(wallet);
      alert(`✅ Account created! Signature: ${signature}`);
      await fetchBalance();
    } catch (err: any) {
      alert(`❌ Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const depositSOL = async (amount: number) => {
    if (!wallet || !publicKey) {
      alert('Please connect your wallet first');
      return;
    }

    setLoading(true);
    try {
      // First initialize collateral account if it doesn't exist
      try {
        await smartContractService.initializeCollateralAccount(wallet, 'SOL');
      } catch (err) {
        // Might already exist
        console.log('Collateral account might already exist');
      }

      // Then deposit
      const signature = await smartContractService.addCollateral(wallet, 'SOL', amount);
      alert(`✅ Deposited ${amount} SOL! Signature: ${signature}`);
      await fetchBalance();
    } catch (err: any) {
      alert(`❌ Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  if (!connected) {
    return (
      <div style={{ padding: '20px', color: 'var(--text-muted)' }}>
        <h3>🔌 Wallet Not Connected</h3>
        <p>Please connect your wallet to test balance fetching</p>
      </div>
    );
  }

  return (
    <div style={{ padding: '20px', height: '100%', overflow: 'auto' }}>
      <h2 style={{ color: 'var(--primary-500)', marginBottom: '20px' }}>🧪 Balance Test Window</h2>
      
      {/* Wallet Info */}
      <div style={{ 
        backgroundColor: 'var(--bg-secondary)', 
        padding: '15px', 
        borderRadius: '8px',
        marginBottom: '20px'
      }}>
        <h3 style={{ color: 'var(--text-primary)', marginBottom: '10px' }}>📱 Wallet Address</h3>
        <code style={{ color: 'var(--text-muted)', fontSize: '12px', wordBreak: 'break-all' }}>
          {publicKey?.toString()}
        </code>
        <div style={{ marginTop: '10px' }}>
          <button 
            onClick={fetchBalance}
            disabled={loading}
            style={{
              padding: '8px 16px',
              backgroundColor: 'var(--primary-500)',
              border: 'none',
              borderRadius: '4px',
              color: 'white',
              cursor: 'pointer',
              marginRight: '10px'
            }}
          >
            🔄 Refresh
          </button>
        </div>
      </div>

      {/* Account Status */}
      <div style={{ 
        backgroundColor: 'var(--bg-secondary)', 
        padding: '15px', 
        borderRadius: '8px',
        marginBottom: '20px'
      }}>
        <h3 style={{ color: 'var(--text-primary)', marginBottom: '10px' }}>📊 Account Status</h3>
        <div style={{ color: 'var(--text-muted)', fontSize: '14px' }}>
          <div>Status: {accountStatus}</div>
          <div>Can Deposit: {balance?.canDeposit ? '✅ Yes' : '❌ No'}</div>
          <div>Can Trade: {balance?.canTrade ? '✅ Yes' : '❌ No'}</div>
          <div>Account Health: {balance?.accountHealth}%</div>
        </div>
        
        {!balance?.hasUserAccount && (
          <button 
            onClick={initializeAccount}
            disabled={loading}
            style={{
              marginTop: '10px',
              padding: '8px 16px',
              backgroundColor: 'var(--success-500)',
              border: 'none',
              borderRadius: '4px',
              color: 'white',
              cursor: 'pointer'
            }}
          >
            🆕 Create Account (+ Rent: ~0.02 SOL)
          </button>
        )}
      </div>

      {/* Balance Information */}
      {loading && <div style={{ textAlign: 'center', padding: '20px' }}>Loading...</div>}
      {error && <div style={{ color: 'var(--danger-500)', padding: '10px' }}>Error: {error}</div>}
      
      {balance && (
        <>
          {/* SOL Balances */}
          <div style={{ 
            backgroundColor: 'var(--bg-secondary)', 
            padding: '15px', 
            borderRadius: '8px',
            marginBottom: '20px'
          }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '10px' }}>💰 SOL Balances</h3>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '13px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0' }}>
                <span style={{ color: 'var(--text-muted)' }}>Wallet SOL:</span>
                <span style={{ color: 'var(--text-primary)' }}>{balance.walletSOL.toFixed(4)} SOL</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0' }}>
                <span style={{ color: 'var(--text-muted)' }}>Deposited SOL:</span>
                <span style={{ color: 'var(--primary-500)' }}>{balance.depositedSOL.toFixed(4)} SOL</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderTop: '1px solid var(--bg-tertiary)', marginTop: '5px', fontWeight: 'bold' }}>
                <span style={{ color: 'var(--text-primary)' }}>Total SOL:</span>
                <span style={{ color: 'var(--success-500)' }}>{balance.totalSOL.toFixed(4)} SOL</span>
              </div>
            </div>
          </div>

          {/* USD Values */}
          <div style={{ 
            backgroundColor: 'var(--bg-secondary)', 
            padding: '15px', 
            borderRadius: '8px',
            marginBottom: '20px'
          }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '10px' }}>💵 USD Values</h3>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '13px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0' }}>
                <span style={{ color: 'var(--text-muted)' }}>Wallet Value:</span>
                <span style={{ color: 'var(--text-primary)' }}>${balance.walletSOLValueUSD.toFixed(2)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0' }}>
                <span style={{ color: 'var(--text-muted)' }}>Deposited Value:</span>
                <span style={{ color: 'var(--primary-500)' }}>${balance.depositedSOLValueUSD.toFixed(2)}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '5px 0', borderTop: '1px solid var(--bg-tertiary)', marginTop: '5px', fontWeight: 'bold' }}>
                <span style={{ color: 'var(--text-primary)' }}>Total Value:</span>
                <span style={{ color: 'var(--success-500)' }}>${balance.totalValueUSD.toFixed(2)}</span>
              </div>
            </div>
          </div>

          {/* Program Account Addresses */}
          <div style={{ 
            backgroundColor: 'var(--bg-secondary)', 
            padding: '15px', 
            borderRadius: '8px',
            marginBottom: '20px'
          }}>
            <h3 style={{ color: 'var(--text-primary)', marginBottom: '10px' }}>📍 Program Account Addresses</h3>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '11px', wordBreak: 'break-all' }}>
              <div style={{ marginBottom: '10px' }}>
                <div style={{ color: 'var(--text-muted)', marginBottom: '5px' }}>UserAccount PDA:</div>
                <code style={{ color: 'var(--text-primary)' }}>{balance.userAccountAddress || 'N/A'}</code>
              </div>
              <div>
                <div style={{ color: 'var(--text-muted)', marginBottom: '5px' }}>CollateralAccount PDA:</div>
                <code style={{ color: 'var(--text-primary)' }}>{balance.collateralAccountAddress || 'N/A'}</code>
              </div>
            </div>
          </div>

          {/* Deposit Collateral */}
          {balance.hasUserAccount && (
            <div style={{ 
              backgroundColor: 'var(--bg-secondary)', 
              padding: '15px', 
              borderRadius: '8px'
            }}>
              <h3 style={{ color: 'var(--text-primary)', marginBottom: '10px' }}>💸 Deposit SOL Collateral</h3>
              <div style={{ display: 'flex', gap: '10px', marginTop: '10px' }}>
                <button 
                  onClick={() => depositSOL(0.1)}
                  disabled={loading}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: 'var(--warning-500)',
                    border: 'none',
                    borderRadius: '4px',
                    color: 'white',
                    cursor: 'pointer'
                  }}
                >
                  Deposit 0.1 SOL
                </button>
                <button 
                  onClick={() => depositSOL(0.5)}
                  disabled={loading}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: 'var(--warning-500)',
                    border: 'none',
                    borderRadius: '4px',
                    color: 'white',
                    cursor: 'pointer'
                  }}
                >
                  Deposit 0.5 SOL
                </button>
                <button 
                  onClick={() => depositSOL(1.0)}
                  disabled={loading}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: 'var(--warning-500)',
                    border: 'none',
                    borderRadius: '4px',
                    color: 'white',
                    cursor: 'pointer'
                  }}
                >
                  Deposit 1.0 SOL
                </button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default TestBalanceWindow;

