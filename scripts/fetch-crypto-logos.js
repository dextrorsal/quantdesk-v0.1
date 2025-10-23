#!/usr/bin/env node

/**
 * Crypto Logo Fetcher
 * Fetches high-quality crypto logos from various sources
 * Sources: Jupiter, DexScreener, CoinGecko, etc.
 */

const fs = require('fs');
const path = require('path');
const https = require('https');
const http = require('http');

// List of assets we need logos for
const ASSETS = [
  'SOL', 'USDT', 'BTC', 'ETH', 'BNB', 'USDC', 'JLP', 'MSOL', 'PYTH', 
  'JITOSOL', 'BSOL', 'JTO', 'JUP', 'USDE', 'FARTCOIN', 'PENGU', 'BONK', 
  'TRUMP', 'WLFI', 'USD1', 'PUMP', 'WIF', 'PONKE', 'POPCAT', 'BOME', 
  'AI16Z', 'GOAT', 'FWOG', 'DRIFT'
];

// Token mappings for different sources
const TOKEN_MAPPINGS = {
  // Solana token addresses (for Jupiter/DexScreener)
  'SOL': 'So11111111111111111111111111111111111111112',
  'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
  'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
  'BTC': '9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E',
  'ETH': '2FPyTwcZLUg1MDrwsyoP4D6s1tM7hAkHYRjkNb5w6Pxk',
  'MSOL': 'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',
  'PYTH': 'HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3',
  'JITOSOL': 'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn',
  'BSOL': 'bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1',
  'JTO': 'jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL',
  'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
  'USDE': '7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj',
  'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
  'WIF': 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
  'PONKE': 'CzLSujWBLFsSjncfkh59rUFqvafWcY5tzedWJSuypump',
  'POPCAT': '7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr',
  'BOME': 'ukHH6c7mMyiWCf1b9pnWe25TSpkDDt3H5pQZgZ74J82',
  'GOAT': 'CzLSujWBLFsSjncfkh59rUFqvafWcY5tzedWJSuypump',
  'FWOG': '5z3EqYQo9HiCEs3R84RCDMu2n7anpDMxRhdK8PSWmrRC',
  'DRIFT': 'DriFtupJYLTosbwoN8koMbEYSx54aFAVLddZsbryz27'
};

// Create logos directory
const LOGOS_DIR = path.join(__dirname, '..', 'frontend', 'public', 'logos');
if (!fs.existsSync(LOGOS_DIR)) {
  fs.mkdirSync(LOGOS_DIR, { recursive: true });
}

// Utility function to download image
function downloadImage(url, filepath) {
  return new Promise((resolve, reject) => {
    const protocol = url.startsWith('https') ? https : http;
    
    protocol.get(url, (response) => {
      if (response.statusCode === 200) {
        const file = fs.createWriteStream(filepath);
        response.pipe(file);
        file.on('finish', () => {
          file.close();
          console.log(`✅ Downloaded: ${path.basename(filepath)}`);
          resolve();
        });
      } else {
        console.log(`❌ Failed to download ${url}: ${response.statusCode}`);
        reject(new Error(`HTTP ${response.statusCode}`));
      }
    }).on('error', (err) => {
      console.log(`❌ Error downloading ${url}:`, err.message);
      reject(err);
    });
  });
}

// Fetch logo from Solana token list
async function fetchFromSolanaTokenList(symbol, tokenAddress) {
  try {
    // Try multiple logo URLs
    const logoUrls = [
      `https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/${tokenAddress}/logo.png`,
      `https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/${tokenAddress}/logo.svg`,
      `https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/${tokenAddress}/logo.jpg`,
      `https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/${tokenAddress}/logo.webp`
    ];

    for (const url of logoUrls) {
      try {
        const filepath = path.join(LOGOS_DIR, `${symbol.toLowerCase()}.png`);
        await downloadImage(url, filepath);
        return true;
      } catch (err) {
        continue;
      }
    }
    return false;
  } catch (err) {
    console.log(`❌ Failed to fetch ${symbol} from Solana token list`);
    return false;
  }
}

// Fetch logo from DexScreener
async function fetchFromDexScreener(symbol, tokenAddress) {
  try {
    const response = await fetch(`https://api.dexscreener.com/latest/dex/tokens/${tokenAddress}`);
    const data = await response.json();
    
    if (data.pairs && data.pairs.length > 0) {
      const logoUrl = data.pairs[0].baseToken?.image;
      if (logoUrl) {
        const filepath = path.join(LOGOS_DIR, `${symbol.toLowerCase()}.png`);
        await downloadImage(logoUrl, filepath);
        return true;
      }
    }
    return false;
  } catch (err) {
    console.log(`❌ Failed to fetch ${symbol} from DexScreener`);
    return false;
  }
}

// Fetch logo from CoinGecko
async function fetchFromCoinGecko(symbol) {
  try {
    // Map symbols to CoinGecko IDs
    const coingeckoIds = {
      'SOL': 'solana',
      'BTC': 'bitcoin',
      'ETH': 'ethereum',
      'USDT': 'tether',
      'USDC': 'usd-coin',
      'BNB': 'binancecoin',
      'BONK': 'bonk',
      'WIF': 'dogwifcoin',
      'PENGU': 'pengu',
      'TRUMP': 'maga',
      'PUMP': 'pump-fun',
      'PONKE': 'ponke',
      'POPCAT': 'popcat',
      'BOME': 'book-of-meme',
      'GOAT': 'goatseus-maximus',
      'FWOG': 'fwog',
      'DRIFT': 'drift-protocol'
    };

    const coingeckoId = coingeckoIds[symbol];
    if (!coingeckoId) return false;

    const response = await fetch(`https://api.coingecko.com/api/v3/coins/${coingeckoId}`);
    const data = await response.json();
    
    if (data.image && data.image.large) {
      const filepath = path.join(LOGOS_DIR, `${symbol.toLowerCase()}.png`);
      await downloadImage(data.image.large, filepath);
      return true;
    }
    return false;
  } catch (err) {
    console.log(`❌ Failed to fetch ${symbol} from CoinGecko`);
    return false;
  }
}

// Generate fallback SVG logos
function generateFallbackSVG(symbol) {
  const svg = `
<svg width="64" height="64" viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect width="64" height="64" rx="32" fill="#6366f1"/>
  <text x="32" y="40" text-anchor="middle" fill="white" font-family="Inter, sans-serif" font-size="20" font-weight="bold">${symbol}</text>
</svg>`;
  
  const filepath = path.join(LOGOS_DIR, `${symbol.toLowerCase()}.svg`);
  fs.writeFileSync(filepath, svg);
  console.log(`📝 Generated fallback SVG for ${symbol}`);
}

// Main function to fetch all logos
async function fetchAllLogos() {
  console.log('🚀 Starting crypto logo fetch...\n');
  
  for (const symbol of ASSETS) {
    console.log(`\n📥 Fetching logo for ${symbol}...`);
    
    let success = false;
    
    // Try Solana token list first (for Solana tokens)
    if (TOKEN_MAPPINGS[symbol]) {
      console.log(`  🔍 Trying Solana token list...`);
      success = await fetchFromSolanaTokenList(symbol, TOKEN_MAPPINGS[symbol]);
    }
    
    // Try DexScreener if Solana token list failed
    if (!success && TOKEN_MAPPINGS[symbol]) {
      console.log(`  🔍 Trying DexScreener...`);
      success = await fetchFromDexScreener(symbol, TOKEN_MAPPINGS[symbol]);
    }
    
    // Try CoinGecko for major tokens
    if (!success) {
      console.log(`  🔍 Trying CoinGecko...`);
      success = await fetchFromCoinGecko(symbol);
    }
    
    // Generate fallback SVG if all sources failed
    if (!success) {
      console.log(`  📝 All sources failed, generating fallback SVG...`);
      generateFallbackSVG(symbol);
    }
    
    // Small delay to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  console.log('\n🎉 Logo fetching complete!');
  console.log(`📁 Logos saved to: ${LOGOS_DIR}`);
  
  // List all downloaded files
  const files = fs.readdirSync(LOGOS_DIR);
  console.log('\n📋 Downloaded files:');
  files.forEach(file => {
    console.log(`  - ${file}`);
  });
}

// Run the script
if (require.main === module) {
  fetchAllLogos().catch(console.error);
}

module.exports = { fetchAllLogos, ASSETS, TOKEN_MAPPINGS };