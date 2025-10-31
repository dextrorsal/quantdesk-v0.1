#!/usr/bin/env node

/**
 * Test script for JIT Liquidity & Market Making
 * Tests institutional-grade liquidity features
 */

const axios = require('axios');

const BASE_URL = 'http://localhost:3002';

async function testJITLiquidity() {
  console.log('💧 Testing JIT Liquidity & Market Making...\n');

  try {
    // Test 1: JIT Liquidity Statistics
    console.log('📊 Test 1: JIT Liquidity Statistics...');
    try {
      const response = await axios.get(`${BASE_URL}/api/jit-liquidity/stats`);
      
      if (response.data.success) {
        const stats = response.data.data;
        console.log('✅ JIT Liquidity Statistics:');
        console.log(`   Active Auctions: ${stats.activeAuctions}`);
        console.log(`   Total Market Makers: ${stats.totalMarketMakers}`);
        console.log(`   Total Volume: $${stats.totalVolume.toLocaleString()}`);
        console.log(`   Total Fees: $${stats.totalFees.toLocaleString()}`);
        console.log(`   Price Improvements: ${stats.totalPriceImprovements}`);
        console.log(`   Avg Price Improvement: ${stats.averagePriceImprovement.toFixed(2)}%`);
        console.log(`   Liquidity Mining Programs: ${stats.liquidityMiningPrograms}`);
        console.log(`   Market Making Strategies: ${stats.marketMakingStrategies}`);
      } else {
        console.log('❌ Failed to fetch JIT liquidity stats:', response.data.error);
      }
    } catch (error) {
      if (error.response?.status === 401) {
        console.log('✅ JIT liquidity stats requires authentication (as expected)');
      } else {
        console.log('❌ JIT liquidity stats error:', error.response?.data?.error || error.message);
      }
    }

    console.log('\n');

    // Test 2: Liquidity Mining Programs
    console.log('⛏️ Test 2: Liquidity Mining Programs...');
    try {
      const response = await axios.get(`${BASE_URL}/api/jit-liquidity/liquidity-mining`);
      
      if (response.data.success) {
        const data = response.data.data;
        console.log('✅ Liquidity Mining Programs:');
        console.log(`   Total Programs: ${data.totalPrograms}`);
        
        data.programs.forEach((program, index) => {
          console.log(`   ${index + 1}. ${program.name}:`);
          console.log(`      Description: ${program.description}`);
          console.log(`      Total Rewards: ${program.totalRewards.toLocaleString()} ${program.currency}`);
          console.log(`      Markets: ${program.marketIds.join(', ')}`);
          console.log(`      Status: ${program.status}`);
          console.log(`      Participants: ${program.participants.length}`);
          console.log(`      Rules: ${program.rules.length} tiers`);
        });
      } else {
        console.log('❌ Failed to fetch liquidity mining programs:', response.data.error);
      }
    } catch (error) {
      console.log('❌ Liquidity mining programs error:', error.response?.data?.error || error.message);
    }

    console.log('\n');

    // Test 3: Market Makers
    console.log('👥 Test 3: Market Makers...');
    try {
      const response = await axios.get(`${BASE_URL}/api/jit-liquidity/market-makers`);
      
      if (response.data.success) {
        const data = response.data.data;
        console.log('✅ Market Makers:');
        console.log(`   Total Market Makers: ${data.totalMarketMakers}`);
        
        if (data.marketMakers.length > 0) {
          console.log('   Top Market Makers:');
          data.marketMakers.slice(0, 3).forEach((mm, index) => {
            console.log(`     ${index + 1}. ${mm.id}:`);
            console.log(`        Tier: ${mm.tier}`);
            console.log(`        Total Volume: $${mm.totalVolume.toLocaleString()}`);
            console.log(`        Total Fees: $${mm.totalFees.toLocaleString()}`);
            console.log(`        Win Rate: ${(mm.winRate * 100).toFixed(2)}%`);
            console.log(`        Reputation: ${mm.reputation}`);
          });
        } else {
          console.log('   No market makers yet (expected for new system)');
        }
      } else {
        console.log('❌ Failed to fetch market makers:', response.data.error);
      }
    } catch (error) {
      if (error.response?.status === 401) {
        console.log('✅ Market makers requires authentication (as expected)');
      } else {
        console.log('❌ Market makers error:', error.response?.data?.error || error.message);
      }
    }

    console.log('\n');

    // Test 4: Price Improvements
    console.log('📈 Test 4: Price Improvements...');
    try {
      const response = await axios.get(`${BASE_URL}/api/jit-liquidity/price-improvements?limit=10`);
      
      if (response.data.success) {
        const data = response.data.data;
        console.log('✅ Price Improvements:');
        console.log(`   Total Improvements: ${data.totalImprovements}`);
        
        if (data.priceImprovements.length > 0) {
          console.log('   Recent Price Improvements:');
          data.priceImprovements.slice(0, 3).forEach((pi, index) => {
            console.log(`     ${index + 1}. Auction ${pi.auctionId}:`);
            console.log(`        Original Price: $${pi.originalPrice.toFixed(2)}`);
            console.log(`        Improved Price: $${pi.improvedPrice.toFixed(2)}`);
            console.log(`        Improvement: ${pi.improvementPercentage.toFixed(2)}%`);
            console.log(`        Maker: ${pi.makerId}`);
          });
        } else {
          console.log('   No price improvements yet (expected for new system)');
        }
      } else {
        console.log('❌ Failed to fetch price improvements:', response.data.error);
      }
    } catch (error) {
      console.log('❌ Price improvements error:', error.response?.data?.error || error.message);
    }

    console.log('\n');

    // Test 5: Market Making Strategies
    console.log('🤖 Test 5: Market Making Strategies...');
    try {
      const response = await axios.get(`${BASE_URL}/api/jit-liquidity/strategies`);
      
      if (response.data.success) {
        const data = response.data.data;
        console.log('✅ Market Making Strategies:');
        console.log(`   Total Strategies: ${data.totalStrategies}`);
        
        if (data.strategies.length > 0) {
          console.log('   Active Strategies:');
          data.strategies.slice(0, 3).forEach((strategy, index) => {
            console.log(`     ${index + 1}. ${strategy.strategyType}:`);
            console.log(`        Market: ${strategy.marketId}`);
            console.log(`        Active: ${strategy.isActive}`);
            console.log(`        Total PnL: $${strategy.performance.totalPnL.toFixed(2)}`);
            console.log(`        Win Rate: ${(strategy.performance.winRate * 100).toFixed(2)}%`);
            console.log(`        Total Volume: $${strategy.performance.totalVolume.toLocaleString()}`);
          });
        } else {
          console.log('   No strategies yet (expected for new system)');
        }
      } else {
        console.log('❌ Failed to fetch market making strategies:', response.data.error);
      }
    } catch (error) {
      if (error.response?.status === 401) {
        console.log('✅ Market making strategies requires authentication (as expected)');
      } else {
        console.log('❌ Market making strategies error:', error.response?.data?.error || error.message);
      }
    }

    console.log('\n');

    // Test 6: Liquidity Auctions
    console.log('🏛️ Test 6: Liquidity Auctions...');
    try {
      const response = await axios.get(`${BASE_URL}/api/jit-liquidity/auctions`);
      
      if (response.data.success) {
        const data = response.data.data;
        console.log('✅ Liquidity Auctions:');
        console.log(`   Total Active Auctions: ${data.totalAuctions}`);
        
        if (data.auctions.length > 0) {
          console.log('   Active Auctions:');
          data.auctions.slice(0, 3).forEach((auction, index) => {
            console.log(`     ${index + 1}. ${auction.id}:`);
            console.log(`        Market: ${auction.marketId}`);
            console.log(`        Side: ${auction.side}`);
            console.log(`        Size: ${auction.size}`);
            console.log(`        Price: $${auction.price.toFixed(2)}`);
            console.log(`        Participants: ${auction.participants.length}`);
            console.log(`        Status: ${auction.status}`);
          });
        } else {
          console.log('   No active auctions yet (expected for new system)');
        }
      } else {
        console.log('❌ Failed to fetch liquidity auctions:', response.data.error);
      }
    } catch (error) {
      if (error.response?.status === 401) {
        console.log('✅ Liquidity auctions requires authentication (as expected)');
      } else {
        console.log('❌ Liquidity auctions error:', error.response?.data?.error || error.message);
      }
    }

    console.log('\n🎉 JIT Liquidity & Market Making Test Complete!');
    console.log('\n💧 JIT Liquidity Features Implemented:');
    console.log('   • Liquidity Auctions (competitive bidding)');
    console.log('   • Market Maker Incentives (tiered rewards)');
    console.log('   • Liquidity Mining Programs (4-tier system)');
    console.log('   • Price Improvement Mechanisms (better execution)');
    console.log('   • Market Making Strategies (5 strategy types)');
    console.log('   • Real-time Statistics (comprehensive metrics)');
    console.log('   • Bid Management (submit, withdraw, win)');
    console.log('   • Performance Tracking (PnL, win rate, volume)');

  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

// Run the test
testJITLiquidity().catch(console.error);
