import axios from 'axios';

export interface PythPriceData {
  id: string;
  price: {
    price: string;
    conf: string;
    expo: number;
    publish_time: number;
  };
  ema_price: {
    price: string;
    conf: string;
    expo: number;
    publish_time: number;
  };
}

export interface PythResponse {
  [key: string]: PythPriceData;
}

export class PythService {
  private readonly PYTH_API_URL = 'https://hermes.pyth.network/v2/updates/price/latest';
  private readonly PYTH_BENCHMARKS_URL = 'https://benchmarks.pyth.network/v1/updates/price/latest';
  
  // Pyth price feed IDs for major assets
  private readonly PRICE_FEED_IDS = {
    BTC: 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J', // BTC/USD
    ETH: 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB', // ETH/USD
    SOL: 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG', // SOL/USD
    USDC: 'Gnt27xtC473ZT2Mw5u8wZ68Z3gULkSTb5DuxJy7eJotD', // USDC/USD
  };

  /**
   * Get latest price data from Pyth Network
   */
  async getLatestPrices(): Promise<PythResponse> {
    try {
      const response = await axios.get(this.PYTH_API_URL, {
        params: {
          ids: Object.values(this.PRICE_FEED_IDS).join(',')
        },
        timeout: 10000
      });

      return response.data;
    } catch (error) {
      console.error('Error fetching Pyth prices:', error);
      throw new Error('Failed to fetch price data from Pyth Network');
    }
  }

  /**
   * Get price for a specific asset
   */
  async getAssetPrice(asset: keyof typeof this.PRICE_FEED_IDS): Promise<number> {
    try {
      const prices = await this.getLatestPrices();
      const feedId = this.PRICE_FEED_IDS[asset];
      const priceData = prices[feedId];

      if (!priceData) {
        throw new Error(`Price data not found for ${asset}`);
      }

      // Convert price from Pyth format (price * 10^expo)
      const price = parseFloat(priceData.price.price) * Math.pow(10, priceData.price.expo);
      
      return price;
    } catch (error) {
      console.error(`Error fetching price for ${asset}:`, error);
      throw error;
    }
  }

  /**
   * Get all supported asset prices
   */
  async getAllPrices(): Promise<Record<string, number>> {
    try {
      const prices = await this.getLatestPrices();
      const result: Record<string, number> = {};

      for (const [asset, feedId] of Object.entries(this.PRICE_FEED_IDS)) {
        const priceData = prices[feedId];
        if (priceData) {
          const price = parseFloat(priceData.price.price) * Math.pow(10, priceData.price.expo);
          result[asset] = price;
        }
      }

      return result;
    } catch (error) {
      console.error('Error fetching all prices:', error);
      throw error;
    }
  }

  /**
   * Get price confidence (uncertainty) for an asset
   */
  async getPriceConfidence(asset: keyof typeof this.PRICE_FEED_IDS): Promise<number> {
    try {
      const prices = await this.getLatestPrices();
      const feedId = this.PRICE_FEED_IDS[asset];
      const priceData = prices[feedId];

      if (!priceData) {
        throw new Error(`Price data not found for ${asset}`);
      }

      // Convert confidence from Pyth format
      const confidence = parseFloat(priceData.price.conf) * Math.pow(10, priceData.price.expo);
      
      return confidence;
    } catch (error) {
      console.error(`Error fetching confidence for ${asset}:`, error);
      throw error;
    }
  }

  /**
   * Check if price data is fresh (within last 30 seconds)
   */
  isPriceFresh(publishTime: number): boolean {
    const now = Date.now() / 1000; // Convert to seconds
    const age = now - publishTime;
    return age < 30; // 30 seconds threshold
  }

  /**
   * Get price feed metadata
   */
  async getPriceFeedMetadata(): Promise<any> {
    try {
      const response = await axios.get('https://hermes.pyth.network/v2/updates/price/latest', {
        timeout: 10000
      });

      return response.data;
    } catch (error) {
      console.error('Error fetching price feed metadata:', error);
      throw error;
    }
  }
}

export const pythService = new PythService();
