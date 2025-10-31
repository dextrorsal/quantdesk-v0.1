import { PublicKey } from '@solana/web3.js';
import { TOKEN_CONFIGS, TokenConfig, getTokenBySymbol, getTokenByMintAddress } from '../config/tokens';

/**
 * Token Service for integrating token configurations with smart contracts
 * This service handles token operations, validation, and smart contract integration
 */
export class TokenService {
  private static instance: TokenService;

  private constructor() {}

  public static getInstance(): TokenService {
    if (!TokenService.instance) {
      TokenService.instance = new TokenService();
    }
    return TokenService.instance;
  }

  /**
   * Get all supported tokens
   */
  public getAllTokens(): TokenConfig[] {
    return TOKEN_CONFIGS;
  }

  /**
   * Get token by symbol
   */
  public getTokenBySymbol(symbol: string): TokenConfig | undefined {
    return getTokenBySymbol(symbol);
  }

  /**
   * Get token by mint address
   */
  public getTokenByMintAddress(mintAddress: string): TokenConfig | undefined {
    return getTokenByMintAddress(mintAddress);
  }

  /**
   * Validate if a token is supported for deposits
   */
  public isTokenSupported(mintAddress: string): boolean {
    return !!getTokenByMintAddress(mintAddress);
  }

  /**
   * Get token mint address as PublicKey for Solana operations
   */
  public getTokenMintPublicKey(symbol: string): PublicKey | null {
    const token = getTokenBySymbol(symbol);
    if (!token) return null;
    
    try {
      return new PublicKey(token.mintAddress);
    } catch (error) {
      console.error(`Invalid mint address for ${symbol}:`, token.mintAddress);
      return null;
    }
  }

  /**
   * Get token mint address as PublicKey by mint address
   */
  public getTokenMintPublicKeyByAddress(mintAddress: string): PublicKey | null {
    const token = getTokenByMintAddress(mintAddress);
    if (!token) return null;
    
    try {
      return new PublicKey(mintAddress);
    } catch (error) {
      console.error(`Invalid mint address:`, mintAddress);
      return null;
    }
  }

  /**
   * Get stablecoins only
   */
  public getStablecoins(): TokenConfig[] {
    return TOKEN_CONFIGS.filter(token => token.isStablecoin);
  }

  /**
   * Get native tokens (like SOL)
   */
  public getNativeTokens(): TokenConfig[] {
    return TOKEN_CONFIGS.filter(token => token.isNative);
  }

  /**
   * Format token amount with proper decimals
   */
  public formatTokenAmount(amount: number, symbol: string): string {
    const token = getTokenBySymbol(symbol);
    if (!token) return amount.toString();
    
    const divisor = Math.pow(10, token.decimals);
    return (amount / divisor).toFixed(token.decimals);
  }

  /**
   * Convert token amount to smallest unit (lamports/tokens)
   */
  public toSmallestUnit(amount: number, symbol: string): number {
    const token = getTokenBySymbol(symbol);
    if (!token) return Math.floor(amount);
    
    const multiplier = Math.pow(10, token.decimals);
    return Math.floor(amount * multiplier);
  }

  /**
   * Get token logo URL
   */
  public getTokenLogoUrl(symbol: string): string {
    const token = getTokenBySymbol(symbol);
    if (!token) return '';
    
    return `https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/${token.mintAddress}/logo.png`;
  }

  /**
   * Validate token configuration
   */
  public validateTokenConfig(token: TokenConfig): boolean {
    try {
      // Validate mint address
      new PublicKey(token.mintAddress);
      
      // Validate decimals
      if (token.decimals < 0 || token.decimals > 18) {
        console.error(`Invalid decimals for ${token.symbol}: ${token.decimals}`);
        return false;
      }
      
      // Validate APY
      if (token.apy < 0 || token.apy > 100) {
        console.error(`Invalid APY for ${token.symbol}: ${token.apy}`);
        return false;
      }
      
      return true;
    } catch (error) {
      console.error(`Invalid token configuration for ${token.symbol}:`, error);
      return false;
    }
  }

  /**
   * Get tokens for deposit modal in specified order
   */
  public getDepositTokens(): TokenConfig[] {
    const depositOrder = [
      'SOL', 'USDT', 'BTC', 'ETH', 'BNB', 'USDC', 'JLP', 'MSOL', 'PYTH', 'JITOSOL', 
      'BSOL', 'JTO', 'JUP', 'USDE', 'FARTCOIN', 'PENGU', 'BONK', 'TRUMP', 'WLFI', 
      'USD1', 'PUMP', 'WIF', 'PONKE', 'POPCAT', 'BOME', 'AI16Z', 'GOAT', 'FWOG', 'DRIFT'
    ];
    
    return depositOrder
      .map(symbol => getTokenBySymbol(symbol))
      .filter((token): token is TokenConfig => token !== undefined);
  }

  /**
   * Check if token is a stablecoin
   */
  public isStablecoin(symbol: string): boolean {
    const token = getTokenBySymbol(symbol);
    return token?.isStablecoin || false;
  }

  /**
   * Check if token is native (like SOL)
   */
  public isNativeToken(symbol: string): boolean {
    const token = getTokenBySymbol(symbol);
    return token?.isNative || false;
  }

  /**
   * Get token info for display
   */
  public getTokenDisplayInfo(symbol: string): {
    symbol: string;
    name: string;
    mintAddress: string;
    decimals: number;
    apy: number;
    logoUrl: string;
    isStablecoin: boolean;
    isNative: boolean;
  } | null {
    const token = getTokenBySymbol(symbol);
    if (!token) return null;
    
    return {
      symbol: token.symbol,
      name: token.name,
      mintAddress: token.mintAddress,
      decimals: token.decimals,
      apy: token.apy,
      logoUrl: this.getTokenLogoUrl(symbol),
      isStablecoin: token.isStablecoin,
      isNative: token.isNative,
    };
  }
}

// Export singleton instance
export const tokenService = TokenService.getInstance();
