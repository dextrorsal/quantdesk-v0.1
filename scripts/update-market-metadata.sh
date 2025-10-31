#!/bin/bash

# Update market metadata with proper categories
echo "🔄 Updating market metadata with categories..."

# Update FARTCOIN metadata
curl -s -X PUT -H "Content-Type: application/json" -d '{"metadata": {"category": "meme-coins", "description": "Fartcoin - Solana meme coin", "logo_url": "https://cdn.quantdesk.com/logos/fartcoin.png"}}' "http://localhost:3002/api/market-management/fa64d833-9707-4b21-932e-b4197b899aec"

# Update BONK metadata
curl -s -X PUT -H "Content-Type: application/json" -d '{"metadata": {"category": "meme-coins", "description": "Bonk - Solana meme coin", "logo_url": "https://cdn.quantdesk.com/logos/bonk.png"}}' "http://localhost:3002/api/market-management/06fab01b-d559-4430-bcec-99855cd308a2"

# Update RAY metadata
curl -s -X PUT -H "Content-Type: application/json" -d '{"metadata": {"category": "solana-ecosystem", "description": "Raydium - Solana DEX", "logo_url": "https://cdn.quantdesk.com/logos/ray.png"}}' "http://localhost:3002/api/market-management/9d45a4aa-f82b-4f6b-a820-75c74e021109"

# Update POPCAT metadata
curl -s -X PUT -H "Content-Type: application/json" -d '{"metadata": {"category": "meme-coins", "description": "Popcat - Viral meme coin", "logo_url": "https://cdn.quantdesk.com/logos/popcat.png"}}' "http://localhost:3002/api/market-management/5152cd72-ef7b-48fc-9b3e-010895b8c719"

echo ""
echo "✅ Updated market metadata!"
echo "📊 Testing updated markets API..."

# Test the updated API
curl -s "http://localhost:3002/api/markets" | jq '.markets[] | {symbol: .symbol, baseAsset: .baseAsset, category: .category}' | head -10
