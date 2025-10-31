#!/bin/bash

# Quick market addition script for hackathon demo
echo "🚀 Adding Essential Markets for Hackathon Demo"
echo "=============================================="

# Essential Solana ecosystem tokens
curl -s -X POST -H "Content-Type: application/json" -d '{"symbol":"JUP-PERP","baseAsset":"JUP","quoteAsset":"USDT","maxLeverage":20,"category":"solana-ecosystem","description":"Jupiter - Solana aggregator"}' http://localhost:3002/api/market-management
curl -s -X POST -H "Content-Type: application/json" -d '{"symbol":"ORCA-PERP","baseAsset":"ORCA","quoteAsset":"USDT","maxLeverage":20,"category":"solana-ecosystem","description":"Orca - Solana AMM"}' http://localhost:3002/api/market-management
curl -s -X POST -H "Content-Type: application/json" -d '{"symbol":"MNGO-PERP","baseAsset":"MNGO","quoteAsset":"USDT","maxLeverage":20,"category":"solana-ecosystem","description":"Mango Markets - Solana DeFi"}' http://localhost:3002/api/market-management

# Popular meme coins
curl -s -X POST -H "Content-Type: application/json" -d '{"symbol":"WIF-PERP","baseAsset":"WIF","quoteAsset":"USDT","maxLeverage":10,"category":"meme-coins","description":"Dogwifhat - Popular Solana meme"}' http://localhost:3002/api/market-management
curl -s -X POST -H "Content-Type: application/json" -d '{"symbol":"PENGU-PERP","baseAsset":"PENGU","quoteAsset":"USDT","maxLeverage":10,"category":"meme-coins","description":"Pengu - Solana meme coin"}' http://localhost:3002/api/market-management
curl -s -X POST -H "Content-Type: application/json" -d '{"symbol":"TRUMP-PERP","baseAsset":"TRUMP","quoteAsset":"USDT","maxLeverage":10,"category":"meme-coins","description":"Trump - Political meme coin"}' http://localhost:3002/api/market-management

# Major altcoins
curl -s -X POST -H "Content-Type: application/json" -d '{"symbol":"BNB-PERP","baseAsset":"BNB","quoteAsset":"USDT","maxLeverage":30,"category":"major-crypto","description":"Binance Coin - Exchange token"}' http://localhost:3002/api/market-management
curl -s -X POST -H "Content-Type: application/json" -d '{"symbol":"AVAX-PERP","baseAsset":"AVAX","quoteAsset":"USDT","maxLeverage":30,"category":"major-crypto","description":"Avalanche - Fast blockchain"}' http://localhost:3002/api/market-management
curl -s -X POST -H "Content-Type: application/json" -d '{"symbol":"MATIC-PERP","baseAsset":"MATIC","quoteAsset":"USDT","maxLeverage":30,"category":"major-crypto","description":"Polygon - Layer 2 scaling"}' http://localhost:3002/api/market-management

# DeFi tokens
curl -s -X POST -H "Content-Type: application/json" -d '{"symbol":"UNI-PERP","baseAsset":"UNI","quoteAsset":"USDT","maxLeverage":20,"category":"defi","description":"Uniswap - DEX protocol"}' http://localhost:3002/api/market-management
curl -s -X POST -H "Content-Type: application/json" -d '{"symbol":"AAVE-PERP","baseAsset":"AAVE","quoteAsset":"USDT","maxLeverage":20,"category":"defi","description":"Aave - Lending protocol"}' http://localhost:3002/api/market-management

echo ""
echo "✅ Added essential markets!"
echo "📊 Total markets: $(curl -s http://localhost:3002/api/markets | jq '.markets | length')"
echo ""
echo "🎯 Your hackathon demo now has:"
echo "   • FARTCOIN, BONK, WIF, POPCAT, PENGU, TRUMP (meme coins)"
echo "   • SOL, BTC, ETH, BNB, AVAX, MATIC (major crypto)"
echo "   • RAY, JUP, ORCA, MNGO (Solana ecosystem)"
echo "   • UNI, AAVE (DeFi tokens)"
echo ""
echo "🚀 Perfect for your hackathon demo!"
