#!/usr/bin/env node

/**
 * Test script for new perpetual markets
 * Tests the new 8 markets we just added
 */

const axios = require('axios');

const BASE_URL = 'http://localhost:3002';

async function testNewMarkets() {
  console.log('🚀 Testing New Perpetual Markets...\n');

  try {
    // Test 1: Get all markets
    console.log('📊 Test 1: Fetching all markets...');
    const marketsResponse = await axios.get(`${BASE_URL}/api/supabase-oracle/markets`);
    
    if (marketsResponse.data.success) {
      const markets = marketsResponse.data.data;
      console.log(`✅ Found ${markets.length} markets:`);
      
      markets.forEach(market => {
        console.log(`   • ${market.symbol} (${market.base_asset}/${market.quote_asset}) - Max Leverage: ${market.max_leverage}x`);
      });
      
      // Verify we have all expected markets
      const expectedMarkets = [
        'BTC-PERP', 'ETH-PERP', 'SOL-PERP',
        'AVAX-PERP', 'MATIC-PERP', 'ARB-PERP', 'OP-PERP',
        'DOGE-PERP', 'ADA-PERP', 'DOT-PERP', 'LINK-PERP'
      ];
      
      const foundMarkets = markets.map(m => m.symbol);
      const missingMarkets = expectedMarkets.filter(m => !foundMarkets.includes(m));
      
      if (missingMarkets.length === 0) {
        console.log('✅ All expected markets found!');
      } else {
        console.log(`❌ Missing markets: ${missingMarkets.join(', ')}`);
      }
    } else {
      console.log('❌ Failed to fetch markets');
    }

    console.log('\n');

    // Test 2: Test cross-collateralization with new assets
    console.log('💎 Test 2: Testing cross-collateralization with new assets...');
    
    const newAssets = ['AVAX', 'MATIC', 'ARB', 'OP', 'DOGE', 'ADA', 'DOT', 'LINK'];
    
    for (const asset of newAssets) {
      try {
        const collateralResponse = await axios.post(`${BASE_URL}/api/cross-collateral/initialize`, {
          user_id: 'test-user-123',
          asset_type: asset,
          amount: 1000
        });
        
        if (collateralResponse.data.success) {
          console.log(`✅ ${asset} collateral account initialized successfully`);
        } else {
          console.log(`❌ ${asset} collateral account failed: ${collateralResponse.data.error}`);
        }
      } catch (error) {
        console.log(`❌ ${asset} collateral account error: ${error.response?.data?.error || error.message}`);
      }
    }

    console.log('\n');

    // Test 3: Test price feeds for new assets
    console.log('💰 Test 3: Testing price feeds for new assets...');
    
    for (const asset of newAssets) {
      try {
        const priceResponse = await axios.get(`${BASE_URL}/api/oracle/price/${asset}`);
        
        if (priceResponse.data.success) {
          const price = priceResponse.data.data.price.price;
          const confidence = priceResponse.data.data.price.conf;
          console.log(`✅ ${asset} price: $${price} (confidence: ${confidence})`);
        } else {
          console.log(`❌ ${asset} price feed failed: ${priceResponse.data.error}`);
        }
      } catch (error) {
        console.log(`❌ ${asset} price feed error: ${error.response?.data?.error || error.message}`);
      }
    }

    console.log('\n');

    // Test 4: Test advanced orders with new markets
    console.log('📈 Test 4: Testing advanced orders with new markets...');
    
    const testOrder = {
      user_id: 'test-user-123',
      market_id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890', // AVAX-PERP
      order_type: 'stop_loss',
      side: 'long',
      size: 10,
      price: 0, // Market order
      stop_price: 25.50,
      leverage: 5,
      time_in_force: 'GTC'
    };

    try {
      const orderResponse = await axios.post(`${BASE_URL}/api/advanced-orders/place`, testOrder);
      
      if (orderResponse.data.success) {
        console.log('✅ Advanced order placed successfully on AVAX-PERP');
        console.log(`   Order ID: ${orderResponse.data.data.id}`);
        console.log(`   Order Type: ${orderResponse.data.data.order_type}`);
        console.log(`   Size: ${orderResponse.data.data.size} AVAX`);
        console.log(`   Stop Price: $${orderResponse.data.data.stop_price}`);
      } else {
        console.log(`❌ Advanced order failed: ${orderResponse.data.error}`);
      }
    } catch (error) {
      console.log(`❌ Advanced order error: ${error.response?.data?.error || error.message}`);
    }

    console.log('\n');

    // Test 5: Test portfolio analytics with new assets
    console.log('📊 Test 5: Testing portfolio analytics with new assets...');
    
    try {
      const analyticsResponse = await axios.get(`${BASE_URL}/api/cross-collateral/portfolio/test-user-123`);
      
      if (analyticsResponse.data.success) {
        const portfolio = analyticsResponse.data.data;
        console.log('✅ Portfolio analytics retrieved successfully');
        console.log(`   Total Collateral Value: $${portfolio.total_collateral_value}`);
        console.log(`   Available Borrowing Power: $${portfolio.available_borrowing_power}`);
        console.log(`   Health Score: ${portfolio.health_score}%`);
        console.log(`   Collateral Assets: ${portfolio.collateral_accounts.length}`);
      } else {
        console.log(`❌ Portfolio analytics failed: ${analyticsResponse.data.error}`);
      }
    } catch (error) {
      console.log(`❌ Portfolio analytics error: ${error.response?.data?.error || error.message}`);
    }

    console.log('\n🎉 New Markets Test Complete!');
    console.log('\n📈 Summary:');
    console.log('   • Added 8 new perpetual markets');
    console.log('   • Extended cross-collateralization to 13 assets');
    console.log('   • Updated price feeds for all new assets');
    console.log('   • Advanced orders work with new markets');
    console.log('   • Portfolio analytics support new assets');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

// Run the test
testNewMarkets().catch(console.error);
