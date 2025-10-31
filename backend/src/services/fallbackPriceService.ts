import axios from 'axios';

export interface PriceData {
  symbol: string;
  price: number;
  timestamp: number;
  source: string;
}

export class FallbackPriceService {
  private readonly COINGECKO_API_URL = 'https://api.coingecko.com/api/v3/simple/price';
  
  // CoinGecko IDs for major assets
  private readonly COINGECKO_IDS = {
    BTC: 'bitcoin',
    ETH: 'ethereum', 
    SOL: 'solana',
    USDC: 'usd-coin',
  };

  /**
   * Get latest price data from CoinGecko as fallback
   */
  async getLatestPrices(): Promise<Record<string, PriceData>> {
    try {
      const ids = Object.values(this.COINGECKO_IDS).join(',');
      const response = await axios.get(this.COINGECKO_API_URL, {
        params: {
          ids: ids,
          vs_currencies: 'usd',
          include_24hr_change: true
        },
        timeout: 10000
      });

      const result: Record<string, PriceData> = {};
      
      for (const [asset, coingeckoId] of Object.entries(this.COINGECKO_IDS)) {
        const data = response.data[coingeckoId];
        if (data && data.usd) {
          result[asset] = {
            symbol: asset,
            price: data.usd,
            timestamp: Date.now(),
            source: 'CoinGecko'
          };
        }
      }

      return result;
    } catch (error) {
      console.error('Error fetching CoinGecko prices:', error);
      throw new Error('Failed to fetch price data from CoinGecko');
    }
  }

  /**
   * Get price for a specific asset
   */
  async getAssetPrice(asset: keyof typeof this.COINGECKO_IDS): Promise<number> {
    try {
      const prices = await this.getLatestPrices();
      const priceData = prices[asset];

      if (!priceData) {
        throw new Error(`Price data not found for ${asset}`);
      }

      return priceData.price;
    } catch (error) {
      console.error(`Error fetching price for ${asset}:`, error);
      throw error;
    }
  }
}

export const fallbackPriceService = new FallbackPriceService();


