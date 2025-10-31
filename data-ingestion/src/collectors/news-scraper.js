const axios = require('axios');
const xml2js = require('xml2js');
const { redis, logger, STREAMS, STREAM_CONFIG } = require('../config');

class NewsScraper {
  constructor() {
    this.newsSources = [
      {
        name: 'CoinDesk',
        url: 'https://www.coindesk.com/arc/outboundfeeds/rss/?outputType=xml',
        type: 'rss'
      },
      {
        name: 'CoinTelegraph',
        url: 'https://cointelegraph.com/rss',
        type: 'rss'
      },
      {
        name: 'The Block',
        url: 'https://www.theblock.co/rss.xml',
        type: 'rss'
      }
    ];
    
    this.isRunning = false;
    this.scrapeInterval = 300000; // 5 minutes
    this.lastScrapeTimes = new Map();
  }

  async start() {
    if (this.isRunning) {
      logger.warn('News scraper already running');
      return;
    }

    logger.info('Starting news scraper...');
    this.isRunning = true;

    // Initial scrape
    await this.scrapeAllSources();

    // Set up interval for continuous scraping
    this.intervalId = setInterval(() => {
      this.scrapeAllSources();
    }, this.scrapeInterval);

    logger.info(`News scraper started, scraping every ${this.scrapeInterval / 1000} seconds`);
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping news scraper...');
    this.isRunning = false;

    if (this.intervalId) {
      clearInterval(this.intervalId);
    }

    logger.info('News scraper stopped');
  }

  async scrapeAllSources() {
    try {
      const scrapePromises = this.newsSources.map(source => 
        this.scrapeSource(source).catch(error => {
          logger.error(`Error scraping ${source.name}:`, error);
          return [];
        })
      );

      const allArticles = await Promise.all(scrapePromises);
      const articles = allArticles.flat();

      if (articles.length > 0) {
        await this.publishArticles(articles);
        logger.info(`Scraped ${articles.length} articles from ${this.newsSources.length} sources`);
      }
    } catch (error) {
      logger.error('Error scraping news sources:', error);
    }
  }

  async scrapeSource(source) {
    try {
      const lastScrapeTime = this.lastScrapeTimes.get(source.name) || new Date(0);
      
      if (source.type === 'rss') {
        return await this.scrapeRSS(source, lastScrapeTime);
      } else if (source.type === 'api') {
        return await this.scrapeAPI(source, lastScrapeTime);
      }

      return [];
    } catch (error) {
      logger.error(`Error scraping ${source.name}:`, error);
      return [];
    }
  }

  async scrapeRSS(source, lastScrapeTime) {
    try {
      const response = await axios.get(source.url, {
        timeout: 10000,
        headers: {
          'User-Agent': 'QuantDesk News Scraper 1.0'
        }
      });

      const articles = await this.parseRSSFeed(response.data, source.name);
      
      // Filter articles newer than last scrape time
      const newArticles = articles.filter(article => 
        new Date(article.published_at) > lastScrapeTime
      );

      // Update last scrape time
      this.lastScrapeTimes.set(source.name, new Date());

      return newArticles;
    } catch (error) {
      logger.error(`Error scraping RSS from ${source.name}:`, error);
      return [];
    }
  }

  async parseRSSFeed(xmlData, sourceName) {
    try {
      const parser = new xml2js.Parser({
        explicitArray: false,
        ignoreAttrs: false,
        mergeAttrs: true
      });

      const result = await parser.parseStringPromise(xmlData);
      const articles = [];

      // Handle different RSS formats
      let items = [];
      if (result.rss && result.rss.channel && result.rss.channel.item) {
        items = Array.isArray(result.rss.channel.item) ? result.rss.channel.item : [result.rss.channel.item];
      } else if (result.feed && result.feed.entry) {
        // Atom feed format
        items = Array.isArray(result.feed.entry) ? result.feed.entry : [result.feed.entry];
      }

      for (const item of items) {
        try {
          const article = {
            title: this.extractTitle(item),
            content: this.extractContent(item),
            url: this.extractUrl(item),
            published_at: this.extractPublishedDate(item),
            source: sourceName,
            sentiment: 'neutral' // Will be analyzed later
          };

          // Validate article has required fields
          if (article.title && article.url && article.published_at) {
            articles.push(article);
          }
        } catch (itemError) {
          logger.warn(`Error parsing individual article from ${sourceName}:`, itemError);
          continue;
        }
      }

      logger.info(`Parsed ${articles.length} articles from ${sourceName}`);
      return articles;
    } catch (error) {
      logger.error(`Error parsing RSS feed from ${sourceName}:`, error);
      return [];
    }
  }

  extractTitle(item) {
    return item.title || item.title_ || item['title'] || '';
  }

  extractContent(item) {
    // Try different content fields
    return item.description || 
           item.description_ || 
           item.content || 
           item.content_ || 
           item.summary || 
           item.summary_ || 
           item['description'] || 
           item['content'] || 
           '';
  }

  extractUrl(item) {
    // Try different URL fields
    return item.link || 
           item.link_ || 
           item.url || 
           item.url_ || 
           item['link'] || 
           item['url'] || 
           '';
  }

  extractPublishedDate(item) {
    // Try different date fields
    const dateStr = item.pubDate || 
                   item.published || 
                   item.published_ || 
                   item.updated || 
                   item.updated_ || 
                   item['pubDate'] || 
                   item['published'] || 
                   '';

    if (dateStr) {
      try {
        return new Date(dateStr).toISOString();
      } catch (dateError) {
        logger.warn(`Error parsing date: ${dateStr}`, dateError);
        return new Date().toISOString();
      }
    }
    
    return new Date().toISOString();
  }

  async scrapeAPI(source, lastScrapeTime) {
    try {
      // Implement API scraping logic here
      // This would depend on the specific API endpoints
      return [];
    } catch (error) {
      logger.error(`Error scraping API from ${source.name}:`, error);
      return [];
    }
  }

  async analyzeSentiment(article) {
    try {
      // Simplified sentiment analysis
      // In production, you'd use a proper NLP service
      const positiveWords = ['bullish', 'moon', 'pump', 'surge', 'rally', 'breakthrough'];
      const negativeWords = ['bearish', 'crash', 'dump', 'plunge', 'decline', 'correction'];
      
      const text = `${article.title} ${article.content}`.toLowerCase();
      
      let sentiment = 'neutral';
      let score = 0;
      
      positiveWords.forEach(word => {
        if (text.includes(word)) score += 1;
      });
      
      negativeWords.forEach(word => {
        if (text.includes(word)) score -= 1;
      });
      
      if (score > 0) sentiment = 'positive';
      else if (score < 0) sentiment = 'negative';
      
      return {
        sentiment,
        score,
        confidence: Math.min(Math.abs(score) / 5, 1) // Normalize to 0-1
      };
    } catch (error) {
      logger.error('Error analyzing sentiment:', error);
      return { sentiment: 'neutral', score: 0, confidence: 0 };
    }
  }

  async publishArticles(articles) {
    try {
      for (const article of articles) {
        // Analyze sentiment
        const sentimentAnalysis = await this.analyzeSentiment(article);
        
        const newsEvent = {
          ...article,
          sentiment: sentimentAnalysis.sentiment,
          sentiment_score: sentimentAnalysis.score,
          sentiment_confidence: sentimentAnalysis.confidence,
          scraped_at: new Date().toISOString()
        };

        await redis.xadd(
          STREAMS.NEWS_RAW,
          'MAXLEN',
          '~',
          STREAM_CONFIG.maxLen,
          '*',
          'data', JSON.stringify(newsEvent),
          'source', newsEvent.source,
          'sentiment', newsEvent.sentiment,
          'timestamp', newsEvent.scraped_at
        );
      }
    } catch (error) {
      logger.error('Error publishing articles:', error);
      throw error;
    }
  }

  async addNewsSource(source) {
    try {
      this.newsSources.push(source);
      logger.info(`Added news source: ${source.name}`);
    } catch (error) {
      logger.error('Error adding news source:', error);
      throw error;
    }
  }

  async removeNewsSource(sourceName) {
    try {
      this.newsSources = this.newsSources.filter(s => s.name !== sourceName);
      logger.info(`Removed news source: ${sourceName}`);
    } catch (error) {
      logger.error('Error removing news source:', error);
      throw error;
    }
  }
}

// Start scraper if run directly
if (require.main === module) {
  const scraper = new NewsScraper();
  
  scraper.start().catch(error => {
    logger.error('Failed to start news scraper:', error);
    process.exit(1);
  });
}

module.exports = NewsScraper;
