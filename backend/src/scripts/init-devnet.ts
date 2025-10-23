import { Connection, PublicKey } from '@solana/web3.js';
import { databaseService } from '../services/supabaseDatabase';
import { config } from '../config/environment';
import * as anchor from '@coral-xyz/anchor';

/**
 * Initialize devnet environment
 * - Seed initial markets
 * - Verify database connectivity
 * - Check program deployment
 */

interface MarketConfig {
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
  maxLeverage: number;
  initialMarginRatio: number;
  maintenanceMarginRatio: number;
  pythFeedId: string;
}

const DEVNET_MARKETS: MarketConfig[] = [
  {
    symbol: 'SOL/USD',
    baseAsset: 'SOL',
    quoteAsset: 'USD',
    maxLeverage: 10,
    initialMarginRatio: 1000, // 10% in basis points
    maintenanceMarginRatio: 500, // 5% in basis points
    pythFeedId: config.PYTH_PRICE_FEEDS.SOL,
  },
  {
    symbol: 'BTC/USD',
    baseAsset: 'BTC',
    quoteAsset: 'USD',
    maxLeverage: 10,
    initialMarginRatio: 1000,
    maintenanceMarginRatio: 500,
    pythFeedId: config.PYTH_PRICE_FEEDS.BTC,
  },
  {
    symbol: 'ETH/USD',
    baseAsset: 'ETH',
    quoteAsset: 'USD',
    maxLeverage: 10,
    initialMarginRatio: 1000,
    maintenanceMarginRatio: 500,
    pythFeedId: config.PYTH_PRICE_FEEDS.ETH,
  },
];

async function initializeDevnet() {
  console.log('🚀 Initializing Devnet Environment...\n');

  try {
    // Step 1: Verify database connectivity
    console.log('1️⃣  Checking database connectivity...');
    const dbCheck = await databaseService.getMarkets();
    console.log(`✅ Database connected: Found ${dbCheck.length} existing markets\n`);

    // Step 2: Verify Solana connection
    console.log('2️⃣  Checking Solana devnet connection...');
    const connection = new Connection(config.RPC_URL, 'confirmed');
    const version = await connection.getVersion();
    console.log(`✅ Connected to Solana devnet: ${JSON.stringify(version)}\n`);

    // Step 3: Verify program deployment
    console.log('3️⃣  Verifying program deployment...');
    const programId = new PublicKey(config.PROGRAM_ID);
    const programInfo = await connection.getAccountInfo(programId);
    
    if (!programInfo) {
      console.log('⚠️  Program not found on devnet. Please deploy with:');
      console.log('   cd contracts && anchor build && anchor deploy --provider.cluster devnet');
      return;
    }
    
    console.log(`✅ Program deployed at: ${programId.toBase58()}`);
    console.log(`   Owner: ${programInfo.owner.toBase58()}`);
    console.log(`   Executable: ${programInfo.executable}\n`);

    // Step 4: Seed markets in database
    console.log('4️⃣  Seeding devnet markets...');
    
    // OPTIMIZATION: Create markets in parallel (3x faster)
    const marketCreationPromises = DEVNET_MARKETS.map(async (marketConfig) => {
      // Check if market already exists
      const existing = dbCheck.find(m => m.symbol === marketConfig.symbol);
      
      if (existing) {
        console.log(`   ⏭️  Market ${marketConfig.symbol} already exists (ID: ${existing.id})`);
        return { symbol: marketConfig.symbol, existed: true };
      }
      
      console.log(`   Creating ${marketConfig.symbol}...`);
      
      // Derive market PDA
      const [marketPDA] = PublicKey.findProgramAddressSync(
        [
          Buffer.from('market'),
          Buffer.from(marketConfig.baseAsset),
          Buffer.from(marketConfig.quoteAsset),
        ],
        programId
      );

      // Create market in database
      const result = await databaseService.createMarket({
        symbol: marketConfig.symbol,
        base_asset: marketConfig.baseAsset,
        quote_asset: marketConfig.quoteAsset,
        program_id: programId.toBase58(),
        market_account: marketPDA.toBase58(),
        oracle_account: marketConfig.pythFeedId,
        max_leverage: marketConfig.maxLeverage,
        initial_margin_ratio: marketConfig.initialMarginRatio,
        maintenance_margin_ratio: marketConfig.maintenanceMarginRatio,
        tick_size: 0.01,
        step_size: 0.001,
        min_order_size: 0.01,
        max_order_size: 1000000,
        funding_interval: 3600,
        current_funding_rate: 0,
        is_active: true,
        metadata: {
          pythFeedId: marketConfig.pythFeedId,
          category: 'crypto',
        },
      });
      
      console.log(`   ✅ Created ${marketConfig.symbol} (ID: ${result.id})`);
      return { symbol: marketConfig.symbol, existed: false, id: result.id };
    });
    
    await Promise.all(marketCreationPromises);
    
    console.log('\n5️⃣  Verifying market setup...');
    const finalMarkets = await databaseService.getMarkets();
    console.log(`✅ Total markets available: ${finalMarkets.length}`);
    
    finalMarkets.forEach(market => {
      const category = market.metadata?.category || 'unknown';
      console.log(`   - ${market.symbol} (${category})`);
      console.log(`     Market PDA: ${market.market_account}`);
      console.log(`     Oracle: ${market.oracle_account}`);
    });

    console.log('\n✅ Devnet initialization complete!\n');
    console.log('📋 Next steps:');
    console.log('   1. Start backend: cd backend && pnpm run dev');
    console.log('   2. Start frontend: cd frontend && pnpm run dev');
    console.log('   3. Connect wallet to devnet in browser');
    console.log('   4. Test account creation and trading\n');

  } catch (error) {
    console.error('❌ Initialization failed:', error);
    console.error('\nPlease check:');
    console.error('   - Supabase credentials in backend/.env');
    console.error('   - Program deployed to devnet');
    console.error('   - RPC endpoint is reachable');
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  initializeDevnet()
    .then(() => process.exit(0))
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}

export { initializeDevnet };

