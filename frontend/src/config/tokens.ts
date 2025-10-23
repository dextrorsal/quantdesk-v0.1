// Token configuration with real Solana contract addresses
// This will be used throughout the platform for token operations

export interface TokenConfig {
  symbol: string;
  name: string;
  mintAddress: string; // Contract Address (CA)
  decimals: number;
  logoUrl?: string;
  apy: number;
  isStablecoin: boolean;
  isNative: boolean;
}

// Real Solana token contract addresses
export const TOKEN_CONFIGS: TokenConfig[] = [
  // Native SOL - direct native SOL deposits like modern DEXs
  {
    symbol: 'SOL',
    name: 'Solana',
    mintAddress: 'So11111111111111111111111111111111111111112', // Display purposes only
    decimals: 9,
    apy: 8.45,
    isStablecoin: false,
    isNative: true, // Native SOL deposits only
  },
  
  // Stablecoins
  {
    symbol: 'USDT',
    name: 'Tether USD',
    mintAddress: 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
    decimals: 6,
    apy: 12.23,
    isStablecoin: true,
    isNative: false,
  },
  {
    symbol: 'USDC',
    name: 'USD Coin',
    mintAddress: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
    decimals: 6,
    apy: 12.23,
    isStablecoin: true,
    isNative: false,
  },
  {
    symbol: 'USDE',
    name: 'USD Ethena',
    mintAddress: '7dHbWXmci3dT8UFYWYZweBLXgycu7Y3iL6trKn1Y7ARj',
    decimals: 6,
    apy: 14.78,
    isStablecoin: true,
    isNative: false,
  },
  
  // Major cryptocurrencies
  {
    symbol: 'BTC',
    name: 'Bitcoin',
    mintAddress: '9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E',
    decimals: 8,
    apy: 6.78,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'ETH',
    name: 'Ethereum',
    mintAddress: '2FPyTwcZLUg1MDrwsyoP4D6s1tM7hAkHYRjkNb5w6Pxk',
    decimals: 8,
    apy: 7.89,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'BNB',
    name: 'Binance Coin',
    mintAddress: '9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM',
    decimals: 8,
    apy: 5.67,
    isStablecoin: false,
    isNative: false,
  },
  
  // Solana ecosystem tokens
  {
    symbol: 'JLP',
    name: 'Jupiter LP',
    mintAddress: '27G8MtK7VtTcCHkpASjSDdkWWYfoqT6ggEuKidVJidD4',
    decimals: 6,
    apy: 15.45,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'MSOL',
    name: 'Marinade SOL',
    mintAddress: 'mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So',
    decimals: 9,
    apy: 9.12,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'JITOSOL',
    name: 'Jito SOL',
    mintAddress: 'J1toso1uCk3RLmjorhTtrVwY9HJ7X8V9yYac6Y7kGCPn',
    decimals: 9,
    apy: 8.67,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'BSOL',
    name: 'BlazeStake SOL',
    mintAddress: 'bSo13r4TkiE4KumL71LsHTPpL2euBYLFx6h9HP3piy1',
    decimals: 9,
    apy: 7.89,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'JTO',
    name: 'Jito',
    mintAddress: 'jtojtomepa8beP8AuQc6eXt5FriJwfFMwQx2v2f9mCL',
    decimals: 9,
    apy: 13.56,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'JUP',
    name: 'Jupiter',
    mintAddress: 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
    decimals: 6,
    apy: 10.23,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'PYTH',
    name: 'Pyth Network',
    mintAddress: 'HZ1JovNiVvGrGNiiYvEozEVgZ58xaU3RKwX8eACQBCt3',
    decimals: 6,
    apy: 11.34,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'DRIFT',
    name: 'Drift Protocol',
    mintAddress: 'DriFtupJYLTosbwoN8koMbEYSx54aFAVLddZsbryz27',
    decimals: 6,
    apy: 12.67,
    isStablecoin: false,
    isNative: false,
  },
  
  // Meme coins
  {
    symbol: 'BONK',
    name: 'Bonk',
    mintAddress: 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
    decimals: 5,
    apy: 16.78,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'WIF',
    name: 'Dogwifhat',
    mintAddress: 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm',
    decimals: 4,
    apy: 21.56,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'POPCAT',
    name: 'Popcat',
    mintAddress: '7GCihgDB8fe6KNjn2MYtkzZcRjQy3t9GHdC8uHYmW2hr',
    decimals: 4,
    apy: 20.12,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'BOME',
    name: 'Book of Meme',
    mintAddress: 'ukHH6c7mMyiWCf1b9pnWe25TSpkDDt3H5pQZgZ74J82',
    decimals: 6,
    apy: 15.67,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'PENGU',
    name: 'Pengu',
    mintAddress: 'EPeUFDgHRxs9xxEPVaL6kfGQvCon7jmAWKVUHuux1Tpz',
    decimals: 6,
    apy: 18.45,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'FARTCOIN',
    name: 'Fartcoin',
    mintAddress: '9BB6NFEcjBCtnNLFko2FqVQBq8HHM13kCyYcdQbgpump',
    decimals: 6,
    apy: 25.67,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'PUMP',
    name: 'Pump',
    mintAddress: 'CzLSujWBLFsSjncfkh59rUFqvafWcY5tzedWJSuypump',
    decimals: 6,
    apy: 28.45,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'PONKE',
    name: 'Ponke',
    mintAddress: 'CzLSujWBLFsSjncfkh59rUFqvafWcY5tzedWJSuypump', // Same as PUMP - need actual address
    decimals: 6,
    apy: 17.89,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'FWOG',
    name: 'Fwog',
    mintAddress: '5z3EqYQo9HiCEs3R84RCDMu2n7anpDMxRhdK8PSWmrRC',
    decimals: 6,
    apy: 16.23,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'GOAT',
    name: 'Goat',
    mintAddress: 'CzLSujWBLFsSjncfkh59rUFqvafWcY5tzedWJSuypump', // Same as PUMP - need actual address
    decimals: 6,
    apy: 13.45,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'TRUMP',
    name: 'Trump',
    mintAddress: 'HeLp6Zqjz42kqjz42kqjz42kqjz42kqjz42kqjz42kqjz', // Placeholder - need actual address
    decimals: 6,
    apy: 22.34,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'WLFI',
    name: 'Wolfi',
    mintAddress: 'EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm', // Same as WIF - need actual address
    decimals: 6,
    apy: 19.67,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'AI16Z',
    name: 'AI16Z',
    mintAddress: 'AI16Z', // Placeholder - need actual address
    decimals: 6,
    apy: 24.78,
    isStablecoin: false,
    isNative: false,
  },
  {
    symbol: 'USD1',
    name: 'USD1',
    mintAddress: 'USD1', // Placeholder - need actual address
    decimals: 6,
    apy: 11.23,
    isStablecoin: true,
    isNative: false,
  },
];

// Helper functions
export const getTokenBySymbol = (symbol: string): TokenConfig | undefined => {
  return TOKEN_CONFIGS.find(token => token.symbol === symbol);
};

export const getTokenByMintAddress = (mintAddress: string): TokenConfig | undefined => {
  return TOKEN_CONFIGS.find(token => token.mintAddress === mintAddress);
};

export const getStablecoins = (): TokenConfig[] => {
  return TOKEN_CONFIGS.filter(token => token.isStablecoin);
};

export const getNativeTokens = (): TokenConfig[] => {
  return TOKEN_CONFIGS.filter(token => token.isNative);
};

export const getTokenLogoUrl = (token: TokenConfig): string => {
  // Use Jupiter's token list for logos
  return `https://raw.githubusercontent.com/solana-labs/token-list/main/assets/mainnet/${token.mintAddress}/logo.png`;
};

// Token list for deposit modal (in the order you specified)
export const DEPOSIT_TOKEN_ORDER = [
  'SOL', 'USDT', 'BTC', 'ETH', 'BNB', 'USDC', 'JLP', 'MSOL', 'PYTH', 'JITOSOL', 
  'BSOL', 'JTO', 'JUP', 'USDE', 'FARTCOIN', 'PENGU', 'BONK', 'TRUMP', 'WLFI', 
  'USD1', 'PUMP', 'WIF', 'PONKE', 'POPCAT', 'BOME', 'AI16Z', 'GOAT', 'FWOG', 'DRIFT'
];

export const getDepositTokens = (): TokenConfig[] => {
  return DEPOSIT_TOKEN_ORDER
    .map(symbol => getTokenBySymbol(symbol))
    .filter((token): token is TokenConfig => token !== undefined);
};
