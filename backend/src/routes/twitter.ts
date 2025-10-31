import { Router, Request, Response } from 'express';
import axios from 'axios';
import { Logger } from '../utils/logger';

const router = Router();
const logger = new Logger();

// X/Twitter API v2 Configuration
// Decode URL-encoded token if needed
const rawToken = process.env.TWITTER_BEARER_TOKEN || '';
const TWITTER_BEARER_TOKEN = rawToken.includes('%') 
  ? decodeURIComponent(rawToken)
  : rawToken;
const TWITTER_API_URL = 'https://api.twitter.com/2';

// Safety: Disable API calls if token is invalid to avoid wasting quota
const USE_REAL_API = true; // Using the regenerated Bearer Token

// Debug: Log token status
if (TWITTER_BEARER_TOKEN) {
  console.log('✅ TWITTER_BEARER_TOKEN loaded (length:', TWITTER_BEARER_TOKEN.length, ')');
} else {
  console.log('⚠️ TWITTER_BEARER_TOKEN is undefined');
}

interface Tweet {
  id: string;
  text: string;
  author: string;
  timestamp: string;
  likes: number;
  retweets: number;
  url: string;
}

// Top Crypto KOLs, Traders, and Accounts for October 2025
const CRYPTO_ACCOUNTS = {
  // Major Crypto News & Analysis
  'CryptoNews': '@CryptoNews',
  'CoinDesk': '@CoinDesk', 
  'TheBlock': '@TheBlock__',
  'CoinTelegraph': '@Cointelegraph',
  'Decrypt': '@decryptmedia',
  
  // Influencers & Traders
  'CryptoWendyO': '@CryptoWendyO',
  'Kaleo': '@CryptoKaleo',
  'CryptoCred': '@cryptocred',
  'RektCaptial': '@rektcapital',
  'CryptoDaku': '@Cryptodaku',
  'TheMoonCarl': '@TheMoonCarl',
  'MMCrypto': '@MMCrypto',
  'BitBoy': '@BenArmstrongCrypto',
  
  // Bitcoin Focused
  'BitcoinMagazine': '@BitcoinMagazine',
  'PrestonPysh': '@PrestonPysh',
  'JamieTout': '@JamieDutout',
  
  // Ethereum/DeFi
  'VitalikButerin': '@VitalikButerin',
  'HaydenAdams': '@haydenzadams',
  'ChrisDunn': '@ChrisDunnTV',
  'DefiLlama': '@DefiLlama',
  
  // Solana Ecosystem
  'Solana': '@solana',
  'SerumDEX': '@ProjectSerum',
  'JupiterExchange': '@JupiterExchange',
  'MagicEden': '@MagicEden',
  'Phantom': '@phantom',
  
  // Alts & Memes  
  'watcherguru': '@watcherguru',
  'CryptoQuant': '@cryptoquantcom',
  'Glassnode': '@glassnode',
  'IntoTheBlock': '@intotheblock',
  
  // Trading & Technical
  'TradingView': '@tradingview',
  'CryptoYoutube': '@cryptoYT',
  'CryptoRUs': '@cryptorus',
  
  // AI & Innovation
  'OpenAI': '@OpenAI',
  'Worldcoin': '@worldcoin',
  'SingularityNET': '@singularitynet'
};

/**
 * GET /api/twitter/search
 * Search for tweets matching keywords
 */
router.get('/search', async (req: Request, res: Response) => {
  try {
    const { q, type = 'recent', count = 20 } = req.query;
    
    if (!q) {
      return res.status(400).json({ 
        success: false,
        error: 'Query parameter "q" is required'
      });
    }

    // Fetch real tweets from X API (with fallback to simulated)
    const tweets = await fetchRealTwitterTweets(q as string, parseInt(count as string));
    
    res.json({
      success: true,
      tweets,
      query: q,
      count: tweets.length
    });
  } catch (error: any) {
    logger.error('Twitter search error:', error);
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});

/**
 * GET /api/twitter/feed
 * Get live feed of tweets matching criteria
 */
router.get('/feed', async (req: Request, res: Response) => {
  try {
    const { keywords, usernames, hashtags } = req.query;
    
    const searchTerms = [];
    if (keywords) searchTerms.push(keywords as string);
    if (usernames) searchTerms.push(`from:${usernames as string}`);
    if (hashtags) searchTerms.push(`#${hashtags as string}`);
    
    const query = searchTerms.join(' OR ') || 'bitcoin ethereum crypto';
    
    logger.info(`Fetching Twitter feed for: ${query}`);
    // CONSERVATIVE: Only fetch 5 tweets with high engagement threshold
    const tweets = await fetchRealTwitterTweets(query, 5);
    
    res.json({
      success: true,
      tweets,
      query,
      count: tweets.length
    });
  } catch (error: any) {
    logger.error('Twitter feed error:', error);
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});

/**
 * Fetch real tweets from X/Twitter API v2
 * Note: Requires TWITTER_BEARER_TOKEN environment variable
 */
async function fetchRealTwitterTweets(query: string, count: number): Promise<Tweet[]> {
  // Safety check: Don't waste API requests if disabled or no token
  if (!USE_REAL_API || !TWITTER_BEARER_TOKEN) {
    if (!USE_REAL_API) {
      logger.warn('⚠️ Real Twitter API is disabled. Using simulated tweets.');
    } else {
      logger.warn('⚠️ No Twitter Bearer Token found. Using simulated tweets.');
    }
    return getSimulatedTweets(query, count);
  }

  try {
    // Build search query with crypto keywords
    const searchQuery = buildSearchQuery(query);
    
    logger.info(`🔍 Attempting to fetch real tweets from X API: "${searchQuery}"`);
    
    const response = await axios.get(`${TWITTER_API_URL}/tweets/search/recent`, {
      headers: {
        'Authorization': `Bearer ${TWITTER_BEARER_TOKEN}`,
        'Content-Type': 'application/json'
      },
      params: {
        query: searchQuery,
        max_results: Math.min(count, 100),
        'tweet.fields': 'created_at,public_metrics,author_id,text',
        'user.fields': 'name,username',
        expansions: 'author_id'
      }
    });
    
    // Log API errors immediately
    if (response.data.errors) {
      logger.error('❌ Twitter API Error:', JSON.stringify(response.data.errors));
      logger.warn('⚠️ Falling back to simulated tweets to avoid wasting API requests.');
      return getSimulatedTweets(query, count);
    }

    if (response.data && response.data.data) {
      // Map to our Tweet interface and filter by engagement
      const tweets = response.data.data
        .map((tweet: any) => {
          const author = response.data.includes?.users?.find((u: any) => u.id === tweet.author_id);
          const engagement = (tweet.public_metrics?.like_count || 0) + (tweet.public_metrics?.retweet_count || 0);
          return {
            id: tweet.id,
            text: tweet.text,
            author: author ? `@${author.username}` : '@crypto',
            timestamp: tweet.created_at,
            likes: tweet.public_metrics?.like_count || 0,
            retweets: tweet.public_metrics?.retweet_count || 0,
            url: `https://twitter.com/i/web/status/${tweet.id}`,
            engagement: engagement
          };
        })
        // Filter: Keep only tweets with at least 10 combined likes+retweets
        .filter((tweet: any) => tweet.engagement >= 10)
        // Sort by engagement (highest first)
        .sort((a: any, b: any) => b.engagement - a.engagement)
        // Take only top 5
        .slice(0, 5);

      logger.info(`✅ Fetched ${tweets.length} high-engagement tweets from X API`);
      return tweets;
    }

    return [];
  } catch (error: any) {
    // Log the actual error details to help diagnose
    if (error.response) {
      // API returned an error response
      logger.error(`❌ Twitter API Error: ${error.response.status} ${error.response.statusText}`);
      logger.error('Response data:', JSON.stringify(error.response.data));
      
      if (error.response.status === 401) {
        logger.error('🔑 Invalid or expired Bearer Token. STOP making API requests!');
      }
    } else if (error.request) {
      // Request was made but no response
      logger.error('❌ Twitter API: No response received');
    } else {
      logger.error('❌ Twitter API Error:', error.message);
    }
    
    logger.warn('⚠️ Using simulated tweets to avoid wasting API quota.');
    return getSimulatedTweets(query, count);
  }
}

/**
 * Build search query for Twitter API
 */
function buildSearchQuery(query: string): string {
  // Add crypto keywords and filter out retweets
  const cryptoTerms = 'bitcoin OR ethereum OR solana OR crypto OR defi OR web3';
  return `${query} (${cryptoTerms}) -is:retweet lang:en`;
}

/**
 * Simulated tweets for demo/testing when no API key
 */
function getSimulatedTweets(query: string, count: number): Promise<Tweet[]> {
  
  const mockTweets: Tweet[] = [
    {
      id: '1',
      text: `Bitcoin just hit $67K! The bull run continues 🚀 $BTC #Crypto`,
      author: '@CryptoNews',
      timestamp: new Date(Date.now() - 1 * 60 * 1000).toISOString(),
      likes: 452,
      retweets: 128,
      url: 'https://twitter.com/CryptoNews/status/1'
    },
    {
      id: '2', 
      text: `Ethereum scaling solutions are gaining traction. Layer 2 adoption is accelerating ⚡ $ETH #DeFi`,
      author: '@EthPrice',
      timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
      likes: 234,
      retweets: 89,
      url: 'https://twitter.com/EthPrice/status/2'
    },
    {
      id: '3',
      text: `Solana network activity at all-time highs 📈 $SOL showing strong fundamentals`,
      author: '@Solana',
      timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
      likes: 567,
      retweets: 145,
      url: 'https://twitter.com/Solana/status/3'
    },
    {
      id: '4',
      text: `Dogwifhat ($WIF) trending on Solana! Meme coin season is here 😂 $WIF`,
      author: '@MemeCoinNews',
      timestamp: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
      likes: 890,
      retweets: 267,
      url: 'https://twitter.com/MemeCoinNews/status/4'
    },
    {
      id: '5',
      text: `Bonk breaking resistance levels! BONK to the moon 🚀 #SolanaMeme`,
      author: '@BonkNews',
      timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
      likes: 623,
      retweets: 189,
      url: 'https://twitter.com/BonkNews/status/5'
    },
    {
      id: '6',
      text: `Jupiter aggregator continues to dominate Solana DEX volumes 📊 $JUP #DeFi`,
      author: '@DefiLlama',
      timestamp: new Date(Date.now() - 75 * 60 * 1000).toISOString(),
      likes: 345,
      retweets: 98,
      url: 'https://twitter.com/DefiLlama/status/6'
    },
    {
      id: '7',
      text: `Pyth Network oracle prices being used across DeFi 🔗 $PYTH shows real-world utility`,
      author: '@PythNetwork',
      timestamp: new Date(Date.now() - 90 * 60 * 1000).toISOString(),
      likes: 412,
      retweets: 112,
      url: 'https://twitter.com/PythNetwork/status/7'
    },
    {
      id: '8',
      text: `POPCAT and PONKE leading the Solana meme surge 🎮 Community engagement off the charts`,
      author: '@SolanaNews',
      timestamp: new Date(Date.now() - 105 * 60 * 1000).toISOString(),
      likes: 756,
      retweets: 201,
      url: 'https://twitter.com/SolanaNews/status/8'
    }
  ];
  
  // Filter tweets based on query
  let filteredTweets = mockTweets;
  if (query.toLowerCase().includes('bitcoin') || query.toLowerCase().includes('btc')) {
    filteredTweets = mockTweets.filter(t => t.text.toLowerCase().includes('bitcoin') || t.text.toLowerCase().includes('btc'));
  }
  if (query.toLowerCase().includes('solana') || query.toLowerCase().includes('sol')) {
    filteredTweets = mockTweets.filter(t => t.text.toLowerCase().includes('sol') || t.text.toLowerCase().includes('sola'));
  }
  if (query.toLowerCase().includes('wif')) {
    filteredTweets = mockTweets.filter(t => t.text.toLowerCase().includes('wif'));
  }
  if (query.toLowerCase().includes('bonk')) {
    filteredTweets = mockTweets.filter(t => t.text.toLowerCase().includes('bonk'));
  }
  
  return Promise.resolve(filteredTweets.slice(0, count));
}

export default router;

