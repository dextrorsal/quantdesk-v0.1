import { PublicKey, SystemProgram, SYSVAR_RENT_PUBKEY } from '@solana/web3.js';

/**
 * Get or create a collateral account for a user
 * Derives PDA: [b"collateral", user.key().as_ref()]
 */
export async function getOrCreateCollateralAccount(
  program: any,
  user: PublicKey
): Promise<PublicKey> {
  // Derive collateral PDA: [b"collateral", user.key().as_ref()]
  const [collateralPDA] = await PublicKey.findProgramAddressSync(
    [Buffer.from("collateral"), user.toBuffer()],
    program.programId
  );

  try {
    // Try to fetch existing account
    await program.account.collateralAccount.fetch(collateralPDA);
    console.log('Collateral account exists:', collateralPDA.toBase58());
    return collateralPDA;
  } catch (error) {
    console.log('Collateral account not found, creating new one...');
    
    // Account doesn't exist, create it
    try {
      const tx = await program.methods
        .initializeCollateralAccount()
        .accounts({
          collateralAccount: collateralPDA,
          user: user,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .rpc();
      
      console.log('Collateral account created:', tx);
      await program.provider.connection.confirmTransaction(tx, 'confirmed');
      
      return collateralPDA;
    } catch (createError) {
      console.error('Error creating collateral account:', createError);
      throw createError;
    }
  }
}

/**
 * Pyth devnet price feed addresses for common trading pairs
 */
const PYTH_FEEDS: Record<string, string> = {
  'SOL/USD': 'J83w4HKfqxwcq3BEMMkPFSppX3gqekLyLJBexebFVkix',
  'BTC/USD': 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J',
  'ETH/USD': 'EdVCmQ9FSPcVe5YySXDPCRmc8aDQLKJ9xvYBMZPie1Vw',
  'USDC/USD': 'Dpw1EAVrSB1ibxiDQyTAW6Zip3J4Btk2x4SgApQCeFbX',
  'USDT/USD': '3vxLXJqLqF3JG5TCbYycbKWRBbCJQLxQmBGCkyqEEefL',
};

/**
 * Get Pyth price feed PublicKey for a given symbol
 * @param symbol - Trading pair symbol (e.g., 'SOL/USD', 'BTC/USD')
 * @returns PublicKey of the Pyth price feed account
 */
export function getPythPriceFeed(symbol: string): PublicKey {
  const feed = PYTH_FEEDS[symbol];
  if (!feed) {
    throw new Error(`No Pyth price feed configured for ${symbol}. Available: ${Object.keys(PYTH_FEEDS).join(', ')}`);
  }
  return new PublicKey(feed);
}

/**
 * Get all available Pyth price feeds
 * @returns Map of symbol to PublicKey
 */
export function getAllPythPriceFeeds(): Map<string, PublicKey> {
  const feeds = new Map<string, PublicKey>();
  for (const [symbol, address] of Object.entries(PYTH_FEEDS)) {
    feeds.set(symbol, new PublicKey(address));
  }
  return feeds;
}

/**
 * Derive a user account PDA with proper encoding
 * Seeds: [b"user_account", authority.key().as_ref(), &account_index.to_le_bytes()]
 */
export function deriveUserAccountPDA(
  program: any,
  authority: PublicKey,
  accountIndex: number = 0
): [PublicKey, number] {
  // Properly encode account_index as u16 little-endian (2 bytes)
  const accountIndexBuffer = Buffer.alloc(2);
  accountIndexBuffer.writeUInt16LE(accountIndex, 0);

  return PublicKey.findProgramAddressSync(
    [
      Buffer.from("user_account"),
      authority.toBuffer(),
      accountIndexBuffer
    ],
    program.programId
  );
}

/**
 * Derive a market PDA
 * Seeds: [b"market", base_asset.as_bytes(), quote_asset.as_bytes()]
 */
export function deriveMarketPDA(
  program: any,
  baseAsset: string,
  quoteAsset: string
): [PublicKey, number] {
  return PublicKey.findProgramAddressSync(
    [
      Buffer.from("market"),
      Buffer.from(baseAsset),
      Buffer.from(quoteAsset)
    ],
    program.programId
  );
}

/**
 * Derive a position PDA
 * Seeds: [b"position", user.key().as_ref(), market.key().as_ref()]
 */
export function derivePositionPDA(
  program: any,
  user: PublicKey,
  market: PublicKey
): [PublicKey, number] {
  return PublicKey.findProgramAddressSync(
    [
      Buffer.from("position"),
      user.toBuffer(),
      market.toBuffer()
    ],
    program.programId
  );
}

/**
 * Derive a collateral PDA
 * Seeds: [b"collateral", user.key().as_ref()]
 */
export function deriveCollateralPDA(
  program: any,
  user: PublicKey
): [PublicKey, number] {
  return PublicKey.findProgramAddressSync(
    [Buffer.from("collateral"), user.toBuffer()],
    program.programId
  );
}

/**
 * Derive an order PDA
 * Seeds: [b"order", user.key().as_ref(), market.key().as_ref(), &order_index.to_le_bytes()]
 */
export function deriveOrderPDA(
  program: any,
  user: PublicKey,
  market: PublicKey,
  orderIndex: number
): [PublicKey, number] {
  const orderIndexBuffer = Buffer.alloc(2);
  orderIndexBuffer.writeUInt16LE(orderIndex, 0);

  return PublicKey.findProgramAddressSync(
    [
      Buffer.from("order"),
      user.toBuffer(),
      market.toBuffer(),
      orderIndexBuffer
    ],
    program.programId
  );
}

