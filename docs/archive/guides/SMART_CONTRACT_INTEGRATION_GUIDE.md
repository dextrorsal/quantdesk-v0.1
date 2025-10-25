# Smart Contract Integration Guide

## Overview
This guide explains how to integrate the QuantDesk Solana smart contracts with the frontend to create the complete account lifecycle experience similar to Drift Protocol.

## Smart Contract Architecture

### Core Programs
1. **User Account Program**: Manages user account creation and state
2. **Market Program**: Handles trading, positions, and orders
3. **Collateral Program**: Manages deposits and withdrawals
4. **Oracle Program**: Provides price feeds

### PDA Structure
```rust
// User Account PDA
let (user_account_pda, bump) = Pubkey::find_program_address(
    &[b"user_account", user_wallet.as_ref(), &account_index.to_le_bytes()],
    program_id,
);

// Market PDA
let (market_pda, bump) = Pubkey::find_program_address(
    &[b"market", base_asset.as_bytes(), quote_asset.as_bytes()],
    program_id,
);

// Position PDA
let (position_pda, bump) = Pubkey::find_program_address(
    &[b"position", user_wallet.as_ref(), market.as_ref()],
    program_id,
);

// Collateral Account PDA
let (collateral_pda, bump) = Pubkey::find_program_address(
    &[b"collateral", user_wallet.as_ref(), asset_mint.as_ref()],
    program_id,
);
```

## Account Lifecycle Implementation

### 1. Wallet Connection State
```typescript
// Check if user has QuantDesk account
const checkUserAccount = async (walletAddress: string): Promise<boolean> => {
  try {
    const [userAccountPda] = await PublicKey.findProgramAddress(
      [Buffer.from("user_account"), new PublicKey(walletAddress).toBuffer(), Buffer.from([0, 0])], // account_index = 0
      programId
    );
    
    const accountInfo = await connection.getAccountInfo(userAccountPda);
    return accountInfo !== null;
  } catch (error) {
    console.error('Error checking user account:', error);
    return false;
  }
};
```

### 2. Account Creation Flow
```typescript
// Create user account
const createUserAccount = async (wallet: Wallet): Promise<void> => {
  const [userAccountPda] = await PublicKey.findProgramAddress(
    [Buffer.from("user_account"), wallet.publicKey.toBuffer(), Buffer.from([0, 0])],
    programId
  );
  
  const transaction = new Transaction().add(
    new TransactionInstruction({
      keys: [
        { pubkey: wallet.publicKey, isSigner: true, isWritable: false },
        { pubkey: userAccountPda, isSigner: false, isWritable: true },
        { pubkey: SystemProgram.programId, isSigner: false, isWritable: false },
      ],
      programId,
      data: Buffer.from([1, 0, 0]), // create_user_account instruction with account_index = 0
    })
  );
  
  await wallet.sendTransaction(transaction, connection);
};
```

### 3. Account State Management
```typescript
// Get user account state
const getUserAccountState = async (walletAddress: string): Promise<UserAccountState> => {
  const [userAccountPda] = await PublicKey.findProgramAddress(
    [Buffer.from("user_account"), new PublicKey(walletAddress).toBuffer(), Buffer.from([0, 0])],
    programId
  );
  
  const accountInfo = await connection.getAccountInfo(userAccountPda);
  if (!accountInfo) {
    return {
      exists: false,
      canDeposit: false,
      canTrade: false,
      totalCollateral: 0,
      accountHealth: 0,
    };
  }
  
  // Decode account data
  const userAccount = UserAccount.decode(accountInfo.data);
  
  return {
    exists: true,
    canDeposit: userAccount.is_active,
    canTrade: userAccount.can_trade(),
    totalCollateral: userAccount.total_collateral,
    accountHealth: userAccount.account_health,
    liquidationPrice: userAccount.liquidation_price,
  };
};
```

### 4. Deposit Flow
```typescript
// Initialize collateral account
const initializeCollateralAccount = async (
  wallet: Wallet, 
  assetType: CollateralType
): Promise<void> => {
  const [collateralPda] = await PublicKey.findProgramAddress(
    [Buffer.from("collateral"), wallet.publicKey.toBuffer(), Buffer.from(assetType.toString())],
    programId
  );
  
  const transaction = new Transaction().add(
    new TransactionInstruction({
      keys: [
        { pubkey: wallet.publicKey, isSigner: true, isWritable: false },
        { pubkey: collateralPda, isSigner: false, isWritable: true },
        { pubkey: SystemProgram.programId, isSigner: false, isWritable: false },
      ],
      programId,
      data: Buffer.from([2, assetType]), // initialize_collateral_account instruction
    })
  );
  
  await wallet.sendTransaction(transaction, connection);
};

// Add collateral
const addCollateral = async (
  wallet: Wallet,
  assetType: CollateralType,
  amount: number
): Promise<void> => {
  const [collateralPda] = await PublicKey.findProgramAddress(
    [Buffer.from("collateral"), wallet.publicKey.toBuffer(), Buffer.from(assetType.toString())],
    programId
  );
  
  // Get user's token account
  const tokenMint = getTokenMint(assetType);
  const userTokenAccount = await getAssociatedTokenAddress(tokenMint, wallet.publicKey);
  
  const transaction = new Transaction().add(
    new TransactionInstruction({
      keys: [
        { pubkey: wallet.publicKey, isSigner: true, isWritable: false },
        { pubkey: collateralPda, isSigner: false, isWritable: true },
        { pubkey: userTokenAccount, isSigner: false, isWritable: true },
      ],
      programId,
      data: Buffer.from([3, ...new BN(amount).toArray('le', 8)]), // add_collateral instruction
    })
  );
  
  await wallet.sendTransaction(transaction, connection);
};
```

### 5. Trading Flow
```typescript
// Place order
const placeOrder = async (
  wallet: Wallet,
  market: string,
  orderType: OrderType,
  side: PositionSide,
  size: number,
  price: number,
  leverage: number
): Promise<void> => {
  const [marketPda] = await PublicKey.findProgramAddress(
    [Buffer.from("market"), Buffer.from(market.split('/')[0]), Buffer.from(market.split('/')[1])],
    programId
  );
  
  const [orderPda] = await PublicKey.findProgramAddress(
    [Buffer.from("order"), wallet.publicKey.toBuffer(), marketPda.toBuffer()],
    programId
  );
  
  const transaction = new Transaction().add(
    new TransactionInstruction({
      keys: [
        { pubkey: wallet.publicKey, isSigner: true, isWritable: false },
        { pubkey: marketPda, isSigner: false, isWritable: true },
        { pubkey: orderPda, isSigner: false, isWritable: true },
      ],
      programId,
      data: Buffer.from([
        4, // place_order instruction
        orderType,
        side,
        ...new BN(size).toArray('le', 8),
        ...new BN(price).toArray('le', 8),
        leverage,
      ]),
    })
  );
  
  await wallet.sendTransaction(transaction, connection);
};
```

## Frontend State Management

### Account Context Provider
```typescript
// frontend/src/contexts/AccountContext.tsx
export const AccountProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [accountState, setAccountState] = useState<AccountState | null>(null);
  const [loading, setLoading] = useState(false);
  const { connected, publicKey } = useWallet();
  
  const fetchAccountState = useCallback(async () => {
    if (!connected || !publicKey) {
      setAccountState(null);
      return;
    }
    
    setLoading(true);
    try {
      const state = await getUserAccountState(publicKey.toString());
      setAccountState(state);
    } catch (error) {
      console.error('Error fetching account state:', error);
    } finally {
      setLoading(false);
    }
  }, [connected, publicKey]);
  
  useEffect(() => {
    fetchAccountState();
  }, [fetchAccountState]);
  
  return (
    <AccountContext.Provider value={{
      accountState,
      loading,
      fetchAccountState,
      createAccount: createUserAccount,
      deposit: addCollateral,
      trade: placeOrder,
    }}>
      {children}
    </AccountContext.Provider>
  );
};
```

### State-Based UI Components
```typescript
// frontend/src/components/AccountStateManager.tsx
export const AccountStateManager: React.FC = () => {
  const { accountState, loading } = useAccount();
  const { connected } = useWallet();
  
  if (!connected) {
    return <WalletConnectionPrompt />;
  }
  
  if (loading) {
    return <LoadingSpinner />;
  }
  
  if (!accountState?.exists) {
    return <AccountCreationPrompt />;
  }
  
  if (!accountState.canTrade) {
    return <DepositPrompt />;
  }
  
  return <TradingInterface />;
};
```

## Backend Integration

### Smart Contract Event Sync
```typescript
// backend/src/services/smartContractSync.ts
export class SmartContractSyncService {
  async syncUserAccount(userWalletAddress: string): Promise<void> {
    // Get user account from blockchain
    const accountState = await this.getUserAccountFromBlockchain(userWalletAddress);
    
    // Update database
    await this.updateUserInDatabase(userWalletAddress, accountState);
  }
  
  async syncPositions(userWalletAddress: string): Promise<void> {
    // Get positions from blockchain
    const positions = await this.getPositionsFromBlockchain(userWalletAddress);
    
    // Update database
    await this.updatePositionsInDatabase(userWalletAddress, positions);
  }
  
  async syncOrders(userWalletAddress: string): Promise<void> {
    // Get orders from blockchain
    const orders = await this.getOrdersFromBlockchain(userWalletAddress);
    
    // Update database
    await this.updateOrdersInDatabase(userWalletAddress, orders);
  }
}
```

### API Endpoints
```typescript
// backend/src/routes/smartContract.ts
router.get('/account-state/:walletAddress', async (req, res) => {
  const { walletAddress } = req.params;
  
  try {
    // Get from blockchain
    const blockchainState = await smartContractSync.getUserAccountFromBlockchain(walletAddress);
    
    // Get from database
    const databaseState = await accountStateService.getUserAccountState(walletAddress);
    
    // Merge states
    const mergedState = {
      ...databaseState,
      blockchain: blockchainState,
    };
    
    res.json({ success: true, state: mergedState });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});
```

## Testing Strategy

### Unit Tests
```typescript
// tests/smart-contract-integration.test.ts
describe('Smart Contract Integration', () => {
  test('should create user account', async () => {
    const wallet = new Wallet();
    await createUserAccount(wallet);
    
    const accountExists = await checkUserAccount(wallet.publicKey.toString());
    expect(accountExists).toBe(true);
  });
  
  test('should deposit collateral', async () => {
    const wallet = new Wallet();
    await createUserAccount(wallet);
    await addCollateral(wallet, CollateralType.USDC, 1000);
    
    const state = await getUserAccountState(wallet.publicKey.toString());
    expect(state.totalCollateral).toBeGreaterThan(0);
  });
});
```

### Integration Tests
```typescript
// tests/account-lifecycle.test.ts
describe('Account Lifecycle', () => {
  test('complete user journey', async () => {
    const wallet = new Wallet();
    
    // 1. Connect wallet
    expect(await checkUserAccount(wallet.publicKey.toString())).toBe(false);
    
    // 2. Create account
    await createUserAccount(wallet);
    expect(await checkUserAccount(wallet.publicKey.toString())).toBe(true);
    
    // 3. Deposit funds
    await addCollateral(wallet, CollateralType.USDC, 1000);
    const state = await getUserAccountState(wallet.publicKey.toString());
    expect(state.canTrade).toBe(true);
    
    // 4. Place order
    await placeOrder(wallet, 'BTC/USDT', OrderType.Market, PositionSide.Long, 100, 0, 2);
    
    // 5. Check position
    const positions = await getPositions(wallet.publicKey.toString());
    expect(positions.length).toBeGreaterThan(0);
  });
});
```

## Deployment Checklist

### Smart Contract Deployment
- [ ] Deploy user account program
- [ ] Deploy market program
- [ ] Deploy collateral program
- [ ] Deploy oracle program
- [ ] Update program IDs in frontend
- [ ] Test all programs on devnet

### Frontend Integration
- [ ] Implement wallet connection
- [ ] Implement account creation flow
- [ ] Implement deposit flow
- [ ] Implement trading interface
- [ ] Implement position management
- [ ] Test all user flows

### Backend Integration
- [ ] Implement smart contract sync
- [ ] Implement API endpoints
- [ ] Implement event handling
- [ ] Test data synchronization
- [ ] Monitor performance

### Testing
- [ ] Unit tests for smart contracts
- [ ] Integration tests for frontend
- [ ] End-to-end tests for complete flow
- [ ] Performance tests
- [ ] Security audit

## Security Considerations

### Smart Contract Security
- [ ] Authority checks on all functions
- [ ] Input validation and sanitization
- [ ] Reentrancy protection
- [ ] Oracle price validation
- [ ] Position limit enforcement

### Frontend Security
- [ ] Wallet integration security
- [ ] Transaction signing validation
- [ ] State validation
- [ ] Error boundary handling

### Backend Security
- [ ] API authentication
- [ ] Rate limiting
- [ ] Input validation
- [ ] Audit logging

---

*This integration guide provides the foundation for building a complete Solana perpetual DEX with proper account lifecycle management.*
