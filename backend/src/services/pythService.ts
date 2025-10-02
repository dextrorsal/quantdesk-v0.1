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
  binary?: {
    encoding: string;
    data: string[];
  };
  parsed?: PythPriceData[];
}

export interface PythApiResponse {
  binary?: {
    encoding: string;
    data: string[];
  };
  parsed?: PythPriceData[];
}

export class PythService {
  private readonly PYTH_API_URL = 'https://hermes.pyth.network/v2/updates/price/latest';
  private readonly PYTH_BENCHMARKS_URL = 'https://benchmarks.pyth.network/v1/updates/price/latest';
  
  // Pyth price feed IDs for major assets
  // Note: Only BTC ID is currently working - other IDs need to be found
  private readonly PRICE_FEED_IDS = {
    BTC: 'e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43', // BTC/USD (working ID)
    // ETH: 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB', // ETH/USD (incorrect ID)
    // SOL: 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG', // SOL/USD (incorrect ID)
    // USDC: 'Gnt27xtC473ZT2Mw5u8wZ68Z3gULkSTb5DuxJy7eJotD', // USDC/USD (incorrect ID)
  };

  /**
   * Get latest price data from Pyth Network
   */
  async getLatestPrices(): Promise<PythResponse> {
    try {
      const ids = Object.values(this.PRICE_FEED_IDS);
      const allParsedData: PythPriceData[] = [];
      
      // Make individual requests for each price feed ID
      for (const id of ids) {
        try {
          const response = await axios.get(this.PYTH_API_URL, {
            params: {
              'ids[]': id
            },
            timeout: 10000
          });
          
          // Extract parsed data from response
          if (response.data.parsed && Array.isArray(response.data.parsed)) {
            allParsedData.push(...response.data.parsed);
          }
        } catch (error) {
          console.error(`Failed to fetch price for ID ${id}:`, error);
          // Continue with other IDs even if one fails
        }
      }
      
      return {
        parsed: allParsedData
      };
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
      const response = await this.getLatestPrices();
      const feedId = this.PRICE_FEED_IDS[asset];
      
      if (!response.parsed) {
        throw new Error(`No parsed data available`);
      }

      const priceData = response.parsed.find(item => item.id === feedId);

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
      const response = await this.getLatestPrices();
      const result: Record<string, number> = {};

      if (!response.parsed) {
        return result;
      }

      for (const [asset, feedId] of Object.entries(this.PRICE_FEED_IDS)) {
        const priceData = response.parsed.find(item => item.id === feedId);
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
