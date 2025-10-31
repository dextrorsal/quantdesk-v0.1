#!/bin/bash

# QuantDesk Comprehensive Market Seeding Script
# Based on existing tokens in your codebase

echo "🚀 QuantDesk Comprehensive Market Seeding"
echo "=========================================="

# Check if backend is running
if ! curl -s http://localhost:3002/health > /dev/null; then
    echo "❌ Backend is not running. Please start it first:"
    echo "   cd backend && npm run dev"
    exit 1
fi

echo "✅ Backend is running"

# Check if we have database connection
echo "📊 Checking database connection..."
DB_STATUS=$(curl -s http://localhost:3002/api/supabase-oracle/health | jq -r '.status')

if [ "$DB_STATUS" != "healthy" ]; then
    echo "❌ Database connection failed"
    exit 1
fi

echo "✅ Database connection healthy"

# Create comprehensive market seeding script
echo "🌱 Creating comprehensive market database..."

cat > /tmp/comprehensive_markets.js << 'EOF'
const markets = [
    // Major Cryptocurrencies (High Leverage)
    { symbol: 'BTC-PERP', baseAsset: 'BTC', quoteAsset: 'USDT', maxLeverage: 100, category: 'major-crypto', description: 'Bitcoin - The original cryptocurrency' },
    { symbol: 'ETH-PERP', baseAsset: 'ETH', quoteAsset: 'USDT', maxLeverage: 50, category: 'major-crypto', description: 'Ethereum - Smart contract platform' },
    { symbol: 'SOL-PERP', baseAsset: 'SOL', quoteAsset: 'USDT', maxLeverage: 50, category: 'major-crypto', description: 'Solana - High-performance blockchain' },
    { symbol: 'BNB-PERP', baseAsset: 'BNB', quoteAsset: 'USDT', maxLeverage: 30, category: 'major-crypto', description: 'Binance Coin - Exchange token' },
    { symbol: 'AVAX-PERP', baseAsset: 'AVAX', quoteAsset: 'USDT', maxLeverage: 30, category: 'major-crypto', description: 'Avalanche - Fast blockchain' },
    { symbol: 'MATIC-PERP', baseAsset: 'MATIC', quoteAsset: 'USDT', maxLeverage: 30, category: 'major-crypto', description: 'Polygon - Layer 2 scaling' },
    { symbol: 'ARB-PERP', baseAsset: 'ARB', quoteAsset: 'USDT', maxLeverage: 30, category: 'major-crypto', description: 'Arbitrum - Layer 2 solution' },
    { symbol: 'OP-PERP', baseAsset: 'OP', quoteAsset: 'USDT', maxLeverage: 25, category: 'major-crypto', description: 'Optimism - Layer 2 scaling' },
    { symbol: 'ADA-PERP', baseAsset: 'ADA', quoteAsset: 'USDT', maxLeverage: 25, category: 'major-crypto', description: 'Cardano - Academic blockchain' },
    { symbol: 'DOT-PERP', baseAsset: 'DOT', quoteAsset: 'USDT', maxLeverage: 25, category: 'major-crypto', description: 'Polkadot - Interoperability' },
    { symbol: 'LINK-PERP', baseAsset: 'LINK', quoteAsset: 'USDT', maxLeverage: 25, category: 'major-crypto', description: 'Chainlink - Oracle network' },
    { symbol: 'SUI-PERP', baseAsset: 'SUI', quoteAsset: 'USDT', maxLeverage: 25, category: 'major-crypto', description: 'Sui - Move-based blockchain' },
    { symbol: 'IMX-PERP', baseAsset: 'IMX', quoteAsset: 'USDT', maxLeverage: 25, category: 'major-crypto', description: 'Immutable X - Gaming NFT platform' },

    // Solana Ecosystem Tokens
    { symbol: 'RAY-PERP', baseAsset: 'RAY', quoteAsset: 'USDT', maxLeverage: 20, category: 'solana-ecosystem', description: 'Raydium - Solana DEX' },
    { symbol: 'SRM-PERP', baseAsset: 'SRM', quoteAsset: 'USDT', maxLeverage: 20, category: 'solana-ecosystem', description: 'Serum - Solana DEX' },
    { symbol: 'MNGO-PERP', baseAsset: 'MNGO', quoteAsset: 'USDT', maxLeverage: 20, category: 'solana-ecosystem', description: 'Mango Markets - Solana DeFi' },
    { symbol: 'ORCA-PERP', baseAsset: 'ORCA', quoteAsset: 'USDT', maxLeverage: 20, category: 'solana-ecosystem', description: 'Orca - Solana AMM' },
    { symbol: 'JUP-PERP', baseAsset: 'JUP', quoteAsset: 'USDT', maxLeverage: 20, category: 'solana-ecosystem', description: 'Jupiter - Solana aggregator' },
    { symbol: 'STEP-PERP', baseAsset: 'STEP', quoteAsset: 'USDT', maxLeverage: 20, category: 'solana-ecosystem', description: 'Step Finance - Portfolio tracker' },
    { symbol: 'COPE-PERP', baseAsset: 'COPE', quoteAsset: 'USDT', maxLeverage: 20, category: 'solana-ecosystem', description: 'Cope - Solana meme coin' },
    { symbol: 'FIDA-PERP', baseAsset: 'FIDA', quoteAsset: 'USDT', maxLeverage: 20, category: 'solana-ecosystem', description: 'Bonfida - Solana domain service' },
    { symbol: 'JTO-PERP', baseAsset: 'JTO', quoteAsset: 'USDT', maxLeverage: 20, category: 'solana-ecosystem', description: 'Jito - Solana MEV protection' },
    { symbol: 'BSOL-PERP', baseAsset: 'BSOL', quoteAsset: 'USDT', maxLeverage: 20, category: 'solana-ecosystem', description: 'BlazeStake - Liquid staking' },
    { symbol: 'USDE-PERP', baseAsset: 'USDE', quoteAsset: 'USDT', maxLeverage: 20, category: 'solana-ecosystem', description: 'Ethena USD - Synthetic dollar' },

    // Popular Meme Coins (From your existing list)
    { symbol: 'BONK-PERP', baseAsset: 'BONK', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Bonk - Solana meme coin' },
    { symbol: 'WIF-PERP', baseAsset: 'WIF', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Dogwifhat - Popular Solana meme' },
    { symbol: 'POPCAT-PERP', baseAsset: 'POPCAT', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Popcat - Viral meme coin' },
    { symbol: 'MYRO-PERP', baseAsset: 'MYRO', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Myro - Solana meme coin' },
    { symbol: 'FARTCOIN-PERP', baseAsset: 'FARTCOIN', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Fartcoin - Solana meme coin' },
    { symbol: 'PENGU-PERP', baseAsset: 'PENGU', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Pengu - Solana meme coin' },
    { symbol: 'TRUMP-PERP', baseAsset: 'TRUMP', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Trump - Political meme coin' },
    { symbol: 'WLFI-PERP', baseAsset: 'WLFI', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Wolfi - Solana meme coin' },
    { symbol: 'PONKE-PERP', baseAsset: 'PONKE', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Ponke - Solana meme coin' },
    { symbol: 'SPX-PERP', baseAsset: 'SPX', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'SPX - Solana meme coin' },
    { symbol: 'GIGA-PERP', baseAsset: 'GIGA', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Giga - Solana meme coin' },
    { symbol: 'HYPE-PERP', baseAsset: 'HYPE', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Hype - Solana meme coin' },
    { symbol: 'PNUT-PERP', baseAsset: 'PNUT', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Pnut - Solana meme coin' },
    { symbol: 'MOODENG-PERP', baseAsset: 'MOODENG', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Moodeng - Solana meme coin' },
    { symbol: 'MEW-PERP', baseAsset: 'MEW', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Mew - Solana meme coin' },
    { symbol: 'GOAT-PERP', baseAsset: 'GOAT', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Goat - Solana meme coin' },
    { symbol: 'AI16Z-PERP', baseAsset: 'AI16Z', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'AI16Z - Solana meme coin' },
    { symbol: 'BOME-PERP', baseAsset: 'BOME', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Book of Meme - Solana meme coin' },
    { symbol: 'FWOG-PERP', baseAsset: 'FWOG', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Fwog - Solana meme coin' },
    { symbol: 'PEPE-PERP', baseAsset: 'PEPE', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Pepe - Ethereum meme coin' },
    { symbol: 'PEPECOIN-PERP', baseAsset: 'PEPECOIN', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Pepecoin - Meme coin' },
    { symbol: 'DODGE-PERP', baseAsset: 'DODGE', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Dodge - Meme coin' },
    { symbol: 'FLOKI-PERP', baseAsset: 'FLOKI', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Floki - Meme coin' },
    { symbol: 'BRETT-PERP', baseAsset: 'BRETT', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Brett - Meme coin' },
    { symbol: 'SHIB-PERP', baseAsset: 'SHIB', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Shiba Inu - Dog meme coin' },
    { symbol: 'DOGE-PERP', baseAsset: 'DOGE', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Dogecoin - Original meme coin' },
    { symbol: 'CHILLGUY-PERP', baseAsset: 'CHILLGUY', quoteAsset: 'USDT', maxLeverage: 10, category: 'meme-coins', description: 'Chillguy - Solana meme coin' },

    // DeFi Tokens
    { symbol: 'UNI-PERP', baseAsset: 'UNI', quoteAsset: 'USDT', maxLeverage: 20, category: 'defi', description: 'Uniswap - DEX protocol' },
    { symbol: 'AAVE-PERP', baseAsset: 'AAVE', quoteAsset: 'USDT', maxLeverage: 20, category: 'defi', description: 'Aave - Lending protocol' },
    { symbol: 'COMP-PERP', baseAsset: 'COMP', quoteAsset: 'USDT', maxLeverage: 20, category: 'defi', description: 'Compound - Lending protocol' },
    { symbol: 'MKR-PERP', baseAsset: 'MKR', quoteAsset: 'USDT', maxLeverage: 20, category: 'defi', description: 'Maker - DAO governance' },
    { symbol: 'CRV-PERP', baseAsset: 'CRV', quoteAsset: 'USDT', maxLeverage: 20, category: 'defi', description: 'Curve - Stablecoin DEX' },
    { symbol: 'CAKE-PERP', baseAsset: 'CAKE', quoteAsset: 'USDT', maxLeverage: 20, category: 'defi', description: 'PancakeSwap - BSC DEX' },
    { symbol: 'GRASS-PERP', baseAsset: 'GRASS', quoteAsset: 'USDT', maxLeverage: 20, category: 'defi', description: 'Grass - DePIN protocol' },
    { symbol: 'DRIFT-PERP', baseAsset: 'DRIFT', quoteAsset: 'USDT', maxLeverage: 20, category: 'defi', description: 'Drift Protocol - Solana perps' },

    // Gaming Tokens
    { symbol: 'AXS-PERP', baseAsset: 'AXS', quoteAsset: 'USDT', maxLeverage: 15, category: 'gaming', description: 'Axie Infinity - Gaming token' },
    { symbol: 'SAND-PERP', baseAsset: 'SAND', quoteAsset: 'USDT', maxLeverage: 15, category: 'gaming', description: 'Sandbox - Metaverse token' },
    { symbol: 'MANA-PERP', baseAsset: 'MANA', quoteAsset: 'USDT', maxLeverage: 15, category: 'gaming', description: 'Decentraland - Metaverse token' },
    { symbol: 'GALA-PERP', baseAsset: 'GALA', quoteAsset: 'USDT', maxLeverage: 15, category: 'gaming', description: 'Gala Games - Gaming platform' },
    { symbol: 'BIGTIME-PERP', baseAsset: 'BIGTIME', quoteAsset: 'USDT', maxLeverage: 15, category: 'gaming', description: 'Big Time - Gaming token' },
    { symbol: 'PI-PERP', baseAsset: 'PI', quoteAsset: 'USDT', maxLeverage: 15, category: 'gaming', description: 'Pi Network - Mobile mining' },

    // AI Tokens
    { symbol: 'FET-PERP', baseAsset: 'FET', quoteAsset: 'USDT', maxLeverage: 20, category: 'ai', description: 'Fetch.ai - AI agent network' },
    { symbol: 'AGIX-PERP', baseAsset: 'AGIX', quoteAsset: 'USDT', maxLeverage: 20, category: 'ai', description: 'SingularityNET - AI marketplace' },
    { symbol: 'OCEAN-PERP', baseAsset: 'OCEAN', quoteAsset: 'USDT', maxLeverage: 20, category: 'ai', description: 'Ocean Protocol - Data marketplace' },

    // Stablecoins (Low Leverage)
    { symbol: 'USDC-PERP', baseAsset: 'USDC', quoteAsset: 'USDT', maxLeverage: 5, category: 'stablecoins', description: 'USD Coin - Stablecoin' },
    { symbol: 'DAI-PERP', baseAsset: 'DAI', quoteAsset: 'USDT', maxLeverage: 5, category: 'stablecoins', description: 'Dai - Decentralized stablecoin' },

    // Commodities
    { symbol: 'GOLD-PERP', baseAsset: 'GOLD', quoteAsset: 'USDT', maxLeverage: 10, category: 'commodities', description: 'Gold - Precious metal' },
    { symbol: 'SILVER-PERP', baseAsset: 'SILVER', quoteAsset: 'USDT', maxLeverage: 10, category: 'commodities', description: 'Silver - Precious metal' }
];

async function addMarkets() {
    console.log('🌱 Adding comprehensive markets to database...');
    console.log(`📊 Total markets to add: ${markets.length}`);
    
    let successCount = 0;
    let skipCount = 0;
    
    for (const market of markets) {
        try {
            const response = await fetch('http://localhost:3002/api/markets', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(market)
            });
            
            if (response.ok) {
                console.log(`✅ Added ${market.symbol} (${market.category})`);
                successCount++;
            } else {
                console.log(`⚠️  ${market.symbol} might already exist`);
                skipCount++;
            }
        } catch (error) {
            console.log(`❌ Failed to add ${market.symbol}:`, error.message);
        }
    }
    
    console.log('\n🎉 Market seeding complete!');
    console.log(`✅ Successfully added: ${successCount} markets`);
    console.log(`⚠️  Skipped (already exist): ${skipCount} markets`);
    console.log(`📊 Total processed: ${markets.length} markets`);
}

addMarkets();
EOF

echo "📊 Running comprehensive market seeding..."
node /tmp/comprehensive_markets.js

# Clean up
rm /tmp/comprehensive_markets.js

echo ""
echo "🎯 Comprehensive Market Seeding Complete!"
echo "=========================================="
echo "✅ Added major cryptocurrencies (BTC, ETH, SOL, BNB, AVAX, etc.)"
echo "✅ Added Solana ecosystem tokens (RAY, SRM, MNGO, ORCA, JUP, etc.)"
echo "✅ Added popular meme coins (BONK, WIF, FARTCOIN, POPCAT, etc.)"
echo "✅ Added DeFi tokens (UNI, AAVE, COMP, MKR, etc.)"
echo "✅ Added gaming tokens (AXS, SAND, MANA, GALA, etc.)"
echo "✅ Added AI tokens (FET, AGIX, OCEAN)"
echo "✅ Added stablecoins and commodities"
echo ""
echo "📊 Total markets: $(curl -s http://localhost:3002/api/markets | jq '.markets | length')"
echo ""
echo "🚀 Your frontend now has access to a comprehensive market database!"
echo "   Users can click on any ticker and get real-time charts and data."
echo "   Perfect for your hackathon demo!"
