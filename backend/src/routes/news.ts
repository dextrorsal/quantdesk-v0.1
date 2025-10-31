import { Router, Request, Response } from 'express';
import axios from 'axios';
import { Logger } from '../utils/logger';

const router = Router();
const logger = new Logger();

interface NewsArticle {
  headline: string;
  date: string;
  time: string;
  ticker: string;
  source: string;
  category: string;
  url: string;
  snippet: string;
}

/**
 * RSS feeds for crypto news
 */
const NEWS_FEEDS = {
  'CoinDesk': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
  'CoinTelegraph': 'https://cointelegraph.com/rss/tag/bitcoin',
  'The Block': 'https://www.theblock.co/feed/news',
  'Decrypt': 'https://decrypt.co/feed',
  'CryptoSlate': 'https://cryptoslate.com/feed/',
  'Bitcoin Magazine': 'https://bitcoinmagazine.com/feed',
};

// NewsData.io API configuration (sign up at newsdata.io for free API key)
const NEWSDATA_API_KEY = process.env.NEWSDATA_API_KEY || '';
const NEWSDATA_API_URL = 'https://newsdata.io/api/1/news';

// CryptoPanic API configuration (sign up at cryptopanic.com/developers for free API key)
const CRYPTOPANIC_API_KEY = process.env.CRYPTOPANIC_API_KEY || '';
const CRYPTOPANIC_API_URL = 'https://cryptopanic.com/api/v1/posts/';

/**
 * Fetch crypto news from NewsData.io API (200 free requests/day)
 */
async function fetchNewsDataIO(): Promise<NewsArticle[]> {
  if (!NEWSDATA_API_KEY) {
    logger.warn('⚠️ NEWSDATA_API_KEY not set - skipping NewsData.io');
    return [];
  }

  try {
    logger.info(`📰 Fetching news from NewsData.io API...`);
    const response = await axios.get(NEWSDATA_API_URL, {
      params: {
        apikey: NEWSDATA_API_KEY,
        q: 'cryptocurrency OR blockchain OR bitcoin OR ethereum OR solana OR defi OR web3',
        category: 'business',
        language: 'en',
        full_content: '1',
      },
      timeout: 5000,
    });

    if (response.data && response.data.results) {
      const articles = response.data.results.map((article: any) => ({
        headline: article.title || '',
        date: new Date(article.pubDate || article.pub_date).toISOString().split('T')[0],
        time: new Date(article.pubDate || article.pub_date).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        ticker: extractTickersFromHeadline(article.title || '', article.description || ''),
        source: 'NewsData.io',
        category: 'Crypto News',
        url: article.link || article.guid || '',
        snippet: article.description || article.content?.substring(0, 200) || '',
      }));
      logger.info(`✅ Fetched ${articles.length} articles from NewsData.io`);
      return articles;
    }
  } catch (error: any) {
    logger.error('❌ NewsData.io API error:', error.message);
  }

  return [];
}

/**
 * Fetch crypto news from CryptoPanic API (100 free requests/day)
 */
async function fetchCryptoPanic(): Promise<NewsArticle[]> {
  if (!CRYPTOPANIC_API_KEY) {
    logger.warn('⚠️ CRYPTOPANIC_API_KEY not set - skipping CryptoPanic');
    return [];
  }

  try {
    logger.info(`📰 Fetching news from CryptoPanic API...`);
    const response = await axios.get(CRYPTOPANIC_API_URL, {
      params: {
        auth_token: CRYPTOPANIC_API_KEY,
        kind: 'news',
        filter: 'hot', // hot = trending news
        public: 'true',
      },
      timeout: 5000,
    });

    if (response.data && response.data.results) {
      const articles = response.data.results.map((article: any) => ({
        headline: article.title || '',
        date: new Date(article.created_at).toISOString().split('T')[0],
        time: new Date(article.created_at).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' }),
        ticker: extractTickersFromHeadline(article.title || '', article.url || ''),
        source: 'CryptoPanic',
        category: article.kind === 'news' ? 'News' : 'General',
        url: article.url || '',
        snippet: '', // CryptoPanic doesn't provide snippets
      }));
      logger.info(`✅ Fetched ${articles.length} articles from CryptoPanic`);
      return articles;
    }
  } catch (error: any) {
    logger.error('❌ CryptoPanic API error:', error.message);
  }

  return [];
}

/**
 * Token name to ticker mapping (case-insensitive)
 */
const TOKEN_NAMES: Record<string, string> = {
  'bitcoin': 'BTC',
  'btc': 'BTC',
  'ethereum': 'ETH',
  'eth': 'ETH',
  'solana': 'SOL',
  'sol': 'SOL',
  'binance coin': 'BNB',
  'bnb': 'BNB',
  'cardano': 'ADA',
  'ada': 'ADA',
  'ripple': 'XRP',
  'xrp': 'XRP',
  'dogecoin': 'DOGE',
  'doge': 'DOGE',
  'polkadot': 'DOT',
  'dot': 'DOT',
  'chainlink': 'LINK',
  'link': 'LINK',
  'uniswap': 'UNI',
  'uni': 'UNI',
  'avalanche': 'AVAX',
  'avax': 'AVAX',
  'polygon': 'MATIC',
  'matic': 'MATIC',
  'litecoin': 'LTC',
  'ltc': 'LTC',
  'bitcoin cash': 'BCH',
  'bch': 'BCH',
  'stellar': 'XLM',
  'xlm': 'XLM',
  'algorand': 'ALGO',
  'algo': 'ALGO',
  'cosmos': 'ATOM',
  'atom': 'ATOM',
  'filecoin': 'FIL',
  'fil': 'FIL',
  'tron': 'TRX',
  'trx': 'TRX',
  'eos': 'EOS',
  'aave': 'AAVE',
  'maker': 'MKR',
  'mkr': 'MKR',
  'synthetix': 'SNX',
  'snx': 'SNX',
  'sushi': 'SUSHI',
  'curve': 'CRV',
  'crv': 'CRV',
  'compound': 'COMP',
  'comp': 'COMP',
  'yearn': 'YFI',
  'yfi': 'YFI',
  'thorchain': 'RUNE',
  'rune': 'RUNE',
  'luna': 'LUNA',
  'terra luna': 'LUNA',
  'internet computer': 'ICP',
  'icp': 'ICP',
  'ftx token': 'FTT',
  'ftt': 'FTT',
  'bonk': 'BONK',
  'wif': 'WIF',
  'dogwifhat': 'WIF',
  'pepe': 'PEPE',
  'pepe token': 'PEPE',
  'floki': 'FLOKI',
  'floki inu': 'FLOKI',
  'shiba inu': 'SHIB',
  'shib': 'SHIB',
  'jupiter': 'JUP',
  'jup': 'JUP',
  'jito': 'JTO',
  'jto': 'JTO',
  'wen': 'WEN',
  'wen token': 'WEN',
  'pyth': 'PYTH',
  'pyth network': 'PYTH',
  'raydium': 'RAY',
  'ray': 'RAY',
  'mew': 'MEW',
  'myro': 'MYRO',
  'popcat': 'POPCAT',
  'ponke': 'PONKE',
  'oppop': 'OPPOP',
  'chillguy': 'CHILLGUY',
  'slerf': 'SLERF',
  'maneki': 'MANEKI',
  'bome': 'BOME',
  'book of memes': 'BOME',
  'noot': 'NOOT',
  'fartcoin': 'FARTCOIN',
  'flow': 'FLOW',
  'flowfi': 'FLOW',
  'met': 'MET',
  'kmno': 'KMNO',
  'punk': 'PUMP',
  'pump': 'PUMP',
  'pengu': 'PENGU',
  'meow': 'MEOW',
};

/**
 * Extract ticker symbols from headlines using multiple patterns (like GodelTerminal)
 * Patterns matched:
 * - NASDAQ:AAPL, NYSE:TSLA
 * - (NASDAQ:AAPL), (NYSE:TSLA)
 * - $AAPL, $TSLA
 * - Common crypto tickers (BTC, ETH, SOL, etc.)
 * - Tickers in parentheses: (AAPL), (TSLA)
 * - Full token names (case-insensitive): "Solana" -> "SOL"
 */
function extractTickersFromHeadline(headline: string, description: string): string {
  const text = `${headline} ${description}`.toLowerCase(); // Case-insensitive search
  
  // Pattern 1: NASDAQ:TICKER or NYSE:TICKER (but NOT just the exchange name)
  const exchangeMatch = text.match(/\b(NASDAQ|NYSE|AMEX|OTC|NASDAQ|CME|NYSE|NASDA[QK]):([A-Z]{1,5})\b/i);
  if (exchangeMatch && exchangeMatch[2]) {
    const ticker = exchangeMatch[2].toUpperCase();
    // Don't return exchange names themselves as tickers
    if (!['NASDAQ', 'NYSE', 'AMEX', 'OTC', 'CME'].includes(ticker)) {
      return ticker;
    }
  }
  
  // Pattern 2: (NASDAQ:TICKER) or (NYSE:TICKER) - extract the ticker, not the exchange
  const exchangeParensMatch = text.match(/\((?:NASDAQ|NYSE|AMEX|OTC|CME):([A-Z]{1,5})\)/i);
  if (exchangeParensMatch && exchangeParensMatch[1]) {
    const ticker = exchangeParensMatch[1].toUpperCase();
    // Don't return exchange names themselves
    if (!['NASDAQ', 'NYSE', 'AMEX', 'OTC', 'CME'].includes(ticker)) {
      return ticker;
    }
  }
  
  // Pattern 3: $TICKER (case-insensitive)
  const dollarMatch = text.match(/\$([a-z]{2,5})\b/i);
  if (dollarMatch) {
    return dollarMatch[1].toUpperCase();
  }
  
  // Pattern 4: (TICKER) in parentheses - common in financial news (case-insensitive)
  const parensMatch = text.match(/\(([a-z]{1,5})\)/i);
  if (parensMatch) {
    return parensMatch[1].toUpperCase();
  }
  
  // Pattern 5: Full token names (case-insensitive)
  for (const [name, ticker] of Object.entries(TOKEN_NAMES)) {
    // Check for full token name match
    if (text.includes(name)) {
      return ticker;
    }
    // Also check for ticker as standalone word
    if (text.match(new RegExp(`\\b${name}\\b`))) {
      return ticker;
    }
  }
  
  // Pattern 6: Common cryptocurrency tickers (case-insensitive)
  const cryptoMatch = text.match(/\b(btc|eth|sol|bnb|ada|xrp|doge|dot|link|uni|avax|matic|ltc|bch|xlm|algo|atom|fil|trx|eos|aave|mkr|snx|sushi|crv|comp|yfi|rune|luna|icp|ftt|bonk|wif|pepe|floki|shib|jup|jto|wen|pyth|ray|mew|popcat|ponke|oppop|chillguy|slerf|maneki|bome|noot|fartcoin|flow|met|kmno|pump|pengu|meow)\b/i);
  if (cryptoMatch) {
    return cryptoMatch[1].toUpperCase(); // Return uppercase ticker
  }
  
  // Pattern 7: Common stock tickers (1-5 uppercase letters followed by spaces/punctuation)
  // Look for standalone ticker-like patterns
  const standaloneMatch = text.match(/\b([A-Z]{1,5})\b(?=.*(?:stock|shares|stock price|trading|investor|earnings))/i);
  if (standaloneMatch) {
    const ticker = standaloneMatch[1].toUpperCase();
    // Filter out common words that look like tickers
    const commonWords = ['THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'GET', 'HAS', 'HIM', 'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE', 'TWO', 'WHO', 'WAY'];
    if (!commonWords.includes(ticker)) {
      return ticker;
    }
  }
  
  return 'N/A';
}

/**
 * Parse RSS feed and extract articles
 */
async function parseRSSFeed(feedUrl: string, sourceName: string): Promise<NewsArticle[]> {
  try {
    const response = await axios.get(feedUrl, {
      timeout: 5000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (QuantDesk Bot)',
      },
    });

      // Simple regex-based XML parsing (works for basic RSS feeds)
      const xml = response.data;
      const itemRegex = /<item>([\s\S]*?)<\/item>/gi;
      const items: string[] = [];
      let match;
      
      while ((match = itemRegex.exec(xml)) !== null && items.length < 10) {
        items.push(match[1]);
      }
    
    const articles: NewsArticle[] = [];
    for (const item of items) {
      // Try different patterns for title (RSS vs Atom)
      const titleMatch = item.match(/<title[^>]*><!\[CDATA\[(.*?)\]\]><\/title>/i) || 
                         item.match(/<title[^>]*>(.*?)<\/title>/i);
      const linkMatch = item.match(/<link[^>]*>(.*?)<\/link>/i) || 
                        item.match(/<link[^>]*\/>/);
      const pubDateMatch = item.match(/<pubDate[^>]*>(.*?)<\/pubDate>/i) || 
                           item.match(/<dc:date[^>]*>(.*?)<\/dc:date>/i) ||
                           item.match(/<updated[^>]*>(.*?)<\/updated>/i);
      const descMatch = item.match(/<description[^>]*><!\[CDATA\[(.*?)\]\]><\/description>/i) ||
                        item.match(/<description[^>]*>(.*?)<\/description>/i) ||
                        item.match(/<content[^>]*><!\[CDATA\[(.*?)\]\]><\/content>/i) ||
                        item.match(/<content[^>]*>(.*?)<\/content>/i);
      
      const title = titleMatch ? titleMatch[1].trim().replace(/<[^>]*>/g, '') : '';
      let link = '';
      if (linkMatch) {
        link = linkMatch[1] ? linkMatch[1].trim().replace(/<[^>]*>/g, '') : linkMatch[0];
        // Clean up self-closing link tag
        if (link.includes('href=')) {
          const hrefMatch = link.match(/href="([^"]+)"/);
          link = hrefMatch ? hrefMatch[1] : '';
        }
      }
      const pubDate = pubDateMatch ? pubDateMatch[1].trim() : new Date().toISOString();
      const description = descMatch ? descMatch[1].trim().replace(/<[^>]*>/g, '') : '';
      
      // Extract ticker symbols from headline using multiple patterns
      const ticker = extractTickersFromHeadline(title, description);
      
      // Parse date
      const date = new Date(pubDate);
      const dateStr = date.toISOString().split('T')[0];
      const timeStr = date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
      
      // Determine category
      const category = determineCategory(title, description);
      
      articles.push({
        headline: title,
        date: dateStr,
        time: timeStr,
        ticker,
        source: sourceName,
        category,
        url: link,
        snippet: description.substring(0, 200)
      });
    }
    
    return articles;
  } catch (error: any) {
    logger.error(`Failed to parse RSS feed from ${sourceName}:`, error.message);
    return [];
  }
}

/**
 * Determine article category from content
 */
function determineCategory(headline: string, description: string): string {
  const text = `${headline} ${description}`.toLowerCase();
  
  if (text.includes('defi') || text.includes('decentralized')) return 'DeFi';
  if (text.includes('nft')) return 'NFTs';
  if (text.includes('ethereum') || text.includes('eth')) return 'Technology';
  if (text.includes('bitcoin') || text.includes('btc')) return 'Market Analysis';
  if (text.includes('sec') || text.includes('regulatory') || text.includes('regulation')) return 'Regulatory';
  if (text.includes('partnership') || text.includes('deal')) return 'Partnerships';
  if (text.includes('upgrade') || text.includes('launch') || text.includes('mainnet')) return 'Technology';
  if (text.includes('ecosystem') || text.includes('announces')) return 'Ecosystem';
  
  return 'General';
}

/**
 * GET /api/news
 * Get news articles from multiple RSS feeds
 */
router.get('/', async (req: Request, res: Response) => {
  try {
    const { 
      sources = 'all',  // Comma-separated list or 'all'
      ticker,          // Filter by ticker symbol
      category,        // Filter by category
      keyword,         // Search keyword
      limit = '20'     // Max articles to return
    } = req.query;
    
    logger.info(`Fetching news: sources=${sources}, ticker=${ticker}, category=${category}`);
    
    // Determine which feeds to fetch
    const feedList = sources === 'all' 
      ? Object.entries(NEWS_FEEDS)
      : sources.toString().split(',').map(s => [s.trim(), NEWS_FEEDS[s.trim() as keyof typeof NEWS_FEEDS]])
        .filter(([, url]) => url !== undefined);
    
    // Fetch all feeds in parallel (RSS + API integrations)
    const feedPromises = feedList.map(([name, url]) => parseRSSFeed(url, name));
    const feedResults = await Promise.all(feedPromises);
    
    // Also fetch from News APIs (if configured)
    const newsDataResults = await fetchNewsDataIO();
    const cryptoPanicResults = await fetchCryptoPanic();
    
    logger.info(`📊 News Sources Summary:`);
    logger.info(`   RSS Feeds: ${feedResults.flat().length} articles`);
    logger.info(`   NewsData.io: ${newsDataResults.length} articles`);
    logger.info(`   CryptoPanic: ${cryptoPanicResults.length} articles`);
    
    // Flatten and merge all articles from all sources
    let allArticles = [...feedResults.flat(), ...newsDataResults, ...cryptoPanicResults];
    
    // Apply filters
    if (ticker) {
      allArticles = allArticles.filter(article => 
        article.ticker.toLowerCase() === ticker.toString().toLowerCase()
      );
    }
    
    if (category) {
      allArticles = allArticles.filter(article => 
        article.category.toLowerCase() === category.toString().toLowerCase()
      );
    }
    
    if (keyword) {
      const keywordLower = keyword.toString().toLowerCase();
      allArticles = allArticles.filter(article =>
        article.headline.toLowerCase().includes(keywordLower) ||
        article.snippet.toLowerCase().includes(keywordLower)
      );
    }
    
    // Sort by date (newest first) and limit
    allArticles.sort((a, b) => new Date(b.date + 'T' + b.time).getTime() - new Date(a.date + 'T' + a.time).getTime());
    allArticles = allArticles.slice(0, parseInt(limit.toString()));
    
    res.json({
      success: true,
      articles: allArticles,
      count: allArticles.length,
      sources: feedList.map(([name]) => name)
    });
    
  } catch (error: any) {
    logger.error('News API error:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

export default router;
