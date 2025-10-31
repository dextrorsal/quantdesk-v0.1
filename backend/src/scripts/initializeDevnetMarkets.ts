import * as dotenv from 'dotenv';
import path from 'path';

// Load environment variables first
dotenv.config({ path: path.join(__dirname, '../../.env') });

import { databaseService } from '../services/supabaseDatabase';
import { DEVNET_CONFIG } from '../config/devnet';

interface MarketData {
  symbol: string;
  base_asset: string;
  quote_asset: string;
  max_leverage: number;
  is_active: boolean;
  initial_margin_ratio?: number;
  maintenance_margin_ratio?: number;
  funding_interval?: number;
}

async function initializeMarkets() {
  console.log('Initializing markets for devnet...');
  
  try {
    for (const market of DEVNET_CONFIG.markets) {
      const marketData: MarketData = {
        symbol: market.symbol,
        base_asset: market.baseAsset,
        quote_asset: market.quoteAsset,
        max_leverage: market.maxLeverage,
        is_active: true,
        initial_margin_ratio: 1000, // 10% initial margin
        maintenance_margin_ratio: 500, // 5% maintenance margin
        funding_interval: 3600, // 1 hour funding interval
      };
      
      try {
        await databaseService.createMarket(marketData);
        console.log(`✅ Market initialized: ${market.symbol}`);
      } catch (error) {
        console.log(`⚠️  Market ${market.symbol} may already exist:`, error);
      }
    }
    
    console.log('✅ Markets initialization completed');
  } catch (error) {
    console.error('❌ Error initializing markets:', error);
    throw error;
  }
}

async function initializeOraclePrices() {
  console.log('Initializing oracle prices for devnet...');
  
  try {
    // Initialize mock prices for devnet testing
    const mockPrices = [
      { asset: 'SOL', price: 100.50, timestamp: new Date() },
      { asset: 'BTC', price: 45000.00, timestamp: new Date() },
      { asset: 'ETH', price: 3000.00, timestamp: new Date() },
    ];
    
    for (const priceData of mockPrices) {
      try {
        const { error } = await databaseService.getClient()
          .from('oracle_prices')
          .insert({
            asset: priceData.asset,
            price: priceData.price,
            timestamp: priceData.timestamp.toISOString()
          });
        
        if (error) {
          console.log(`⚠️  Oracle price for ${priceData.asset} may already exist:`, error);
        } else {
          console.log(`✅ Oracle price initialized: ${priceData.asset} = $${priceData.price}`);
        }
      } catch (error) {
        console.log(`⚠️  Oracle price for ${priceData.asset} may already exist:`, error);
      }
    }
    
    console.log('✅ Oracle prices initialization completed');
  } catch (error) {
    console.error('❌ Error initializing oracle prices:', error);
    throw error;
  }
}

async function main() {
  try {
    console.log('🚀 Starting devnet initialization...');
    
    await initializeMarkets();
    await initializeOraclePrices();
    
    console.log('🎉 Devnet initialization completed successfully!');
    process.exit(0);
  } catch (error) {
    console.error('💥 Devnet initialization failed:', error);
    process.exit(1);
  }
}

// Run if called directly
if (require.main === module) {
  main();
}

export { initializeMarkets, initializeOraclePrices };
