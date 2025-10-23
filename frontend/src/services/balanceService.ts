import { Connection, PublicKey } from '@solana/web3.js';
import { TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID } from '@solana/spl-token';
import { getDepositTokens, TokenConfig } from '../config/tokens';
import { smartContractService } from './smartContractService';

// Balance Service
// Following Anchor and Solana Cookbook best practices for reading token balances
// https://solanacookbook.com/references/programs.html#token-program

export interface TokenBalance {
  mint: string;
  symbol: string;
  name: string;
  balance: number;
  decimals: number;
  uiAmount: number; // Human-readable amount
  tokenAccount?: string;
}

export interface UserBalances {
  nativeSOL: number; // SOL balance in lamports
  tokens: TokenBalance[];
  totalValueUSD: number; // Total portfolio value in USD
}

class BalanceService {
  private static instance: BalanceService;
  private connection: Connection;

  private constructor() {
    this.connection = new Connection(
      import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com',
      'confirmed'
    );
  }

  public static getInstance(): BalanceService {
    if (!BalanceService.instance) {
      BalanceService.instance = new BalanceService();
    }
    return BalanceService.instance;
  }

  /**
   * Get native SOL balance
   * Following Solana Cookbook patterns
   */
  async getNativeSOLBalance(walletAddress: PublicKey): Promise<number> {
    try {
      console.log('🔍 Fetching native SOL balance for:', walletAddress.toString());
      
      const balance = await this.connection.getBalance(walletAddress);
      const solBalance = balance / 1e9; // Convert lamports to SOL
      
      console.log('💰 Native SOL balance:', solBalance, 'SOL');
      return solBalance;
    } catch (error) {
      console.error('❌ Error fetching native SOL balance:', error);
      return 0;
    }
  }

  /**
   * Get token balance for a specific mint
   * Following Anchor best practices
   */
  async getTokenBalance(
    walletAddress: PublicKey, 
    mintAddress: PublicKey
  ): Promise<TokenBalance | null> {
    try {
      console.log('🔍 Fetching token balance for mint:', mintAddress.toString());
      
      // Get associated token account
      const tokenAccount = await this.getAssociatedTokenAddress(mintAddress, walletAddress);
      
      // Check if token account exists
      const accountInfo = await this.connection.getAccountInfo(tokenAccount);
      if (!accountInfo) {
        console.log('ℹ️ Token account does not exist for mint:', mintAddress.toString());
        return null;
      }

      // Get token account balance
      const balance = await this.connection.getTokenAccountBalance(tokenAccount);
      
      // Find token config
      const tokenConfig = getDepositTokens().find(
        token => token.mintAddress === mintAddress.toString()
      );

      const tokenBalance: TokenBalance = {
        mint: mintAddress.toString(),
        symbol: tokenConfig?.symbol || 'UNKNOWN',
        name: tokenConfig?.name || 'Unknown Token',
        balance: balance.value.amount,
        decimals: balance.value.decimals,
        uiAmount: balance.value.uiAmount || 0,
        tokenAccount: tokenAccount.toString(),
      };

      console.log('💰 Token balance:', tokenBalance.symbol, tokenBalance.uiAmount);
      return tokenBalance;
    } catch (error) {
      console.error('❌ Error fetching token balance:', error);
      return null;
    }
  }

  /**
   * Get all token balances for a user
   * Following Solana Cookbook patterns for comprehensive balance fetching
   */
  async getAllTokenBalances(walletAddress: PublicKey): Promise<TokenBalance[]> {
    try {
      console.log('🔍 Fetching all token balances for:', walletAddress.toString());
      
      // Get all token accounts owned by the user
      const tokenAccounts = await this.connection.getParsedTokenAccountsByOwner(
        walletAddress,
        {
          programId: TOKEN_PROGRAM_ID,
        }
      );

      console.log('📊 Found', tokenAccounts.value.length, 'token accounts');

      const balances: TokenBalance[] = [];
      
      for (const accountInfo of tokenAccounts.value) {
        const accountData = accountInfo.account.data.parsed.info;
        const mint = accountData.mint;
        
        // Only include tokens from our supported list
        const tokenConfig = getDepositTokens().find(
          token => token.mintAddress === mint
        );

        if (tokenConfig && accountData.tokenAmount.uiAmount > 0) {
          const tokenBalance: TokenBalance = {
            mint: mint,
            symbol: tokenConfig.symbol,
            name: tokenConfig.name,
            balance: accountData.tokenAmount.amount,
            decimals: accountData.tokenAmount.decimals,
            uiAmount: accountData.tokenAmount.uiAmount,
            tokenAccount: accountInfo.pubkey.toString(),
          };

          balances.push(tokenBalance);
          console.log('💰 Found balance:', tokenBalance.symbol, tokenBalance.uiAmount);
        }
      }

      return balances;
    } catch (error) {
      console.error('❌ Error fetching all token balances:', error);
      return [];
    }
  }

  /**
   * Get comprehensive user balances (SOL + tokens)
   * Following Anchor best practices for portfolio management
   * Updated to use real oracle prices and smart contract collateral balances
   */
  async getUserBalances(walletAddress: PublicKey): Promise<UserBalances> {
    try {
      console.log('🔍 Fetching comprehensive user balances for:', walletAddress.toString());
      
      // Get real SOL price from oracle
      const solPrice = await this.getRealSOLPrice();
      
      // Get actual collateral balance from smart contract instead of wallet balance
      const collateralBalance = await smartContractService.getSOLCollateralBalance(
        walletAddress.toString()
      );
      
      // Fetch all token balances
      const tokens = await this.getAllTokenBalances(walletAddress);
      
      // Calculate total USD value using real prices
      let totalValueUSD = collateralBalance * solPrice; // Use actual deposited amount
      for (const token of tokens) {
        // Use real oracle prices when available, fallback to mock prices
        const tokenPrice = await this.getRealTokenPrice(token.symbol) || this.getMockTokenPrice(token.symbol);
        totalValueUSD += token.uiAmount * tokenPrice;
      }

      const userBalances: UserBalances = {
        nativeSOL: collateralBalance, // Use actual deposited amount, not wallet balance
        tokens,
        totalValueUSD,
      };

      console.log('💰 Total portfolio value:', totalValueUSD, 'USD');
      console.log('💰 SOL collateral balance:', collateralBalance, 'SOL');
      console.log('💰 SOL price used:', solPrice, 'USD');
      return userBalances;
    } catch (error) {
      console.error('❌ Error fetching user balances:', error);
      return {
        nativeSOL: 0,
        tokens: [],
        totalValueUSD: 0,
      };
    }
  }

  /**
   * Get associated token address
   * Following Solana Cookbook patterns
   */
  private async getAssociatedTokenAddress(mint: PublicKey, owner: PublicKey): Promise<PublicKey> {
    const [address] = await PublicKey.findProgramAddress(
      [owner.toBuffer(), TOKEN_PROGRAM_ID.toBuffer(), mint.toBuffer()],
      ASSOCIATED_TOKEN_PROGRAM_ID
    );
    return address;
  }

  /**
   * Get real SOL price from oracle
   * Integrates with existing oracle service
   */
  private async getRealSOLPrice(): Promise<number> {
    try {
      // Try to get real SOL price from backend oracle service
      const response = await fetch('/api/oracle/prices');
      if (response.ok) {
        const data = await response.json();
        const solPrice = data.SOL || data.sol || data['SOL/USD'];
        if (solPrice && typeof solPrice === 'number') {
          console.log('💰 Real SOL price from oracle:', solPrice);
          return solPrice;
        }
      }
    } catch (error) {
      console.warn('⚠️ Could not fetch real SOL price, using fallback:', error);
    }
    
    // Fallback to mock price if oracle fails
    console.log('💰 Using fallback SOL price: $100');
    return 100;
  }

  /**
   * Get real token price from oracle
   * Integrates with existing oracle service for all tokens
   */
  private async getRealTokenPrice(symbol: string): Promise<number | null> {
    try {
      // Try to get real token price from backend oracle service
      const response = await fetch('/api/oracle/prices');
      if (response.ok) {
        const data = await response.json();
        const tokenPrice = data[symbol] || data[symbol.toUpperCase()] || data[symbol.toLowerCase()];
        if (tokenPrice && typeof tokenPrice === 'number') {
          console.log(`💰 Real ${symbol} price from oracle:`, tokenPrice);
          return tokenPrice;
        }
      }
    } catch (error) {
      console.warn(`⚠️ Could not fetch real ${symbol} price:`, error);
    }
    
    // Return null to indicate fallback should be used
    return null;
  }

  /**
   * Mock token prices for USD calculation
   * In production, this would integrate with price oracles like Pyth
   */
  private getMockTokenPrice(symbol: string): number {
    const mockPrices: Record<string, number> = {
      'SOL': 100,
      'USDT': 1,
      'USDC': 1,
      'BTC': 50000,
      'ETH': 3000,
      'BNB': 300,
      'JLP': 1.5,
      'MSOL': 105,
      'PYTH': 0.5,
      'JITOSOL': 102,
      'BSOL': 101,
      'JTO': 2,
      'JUP': 0.8,
      'USDE': 1,
      'BONK': 0.00001,
      'WIF': 0.5,
      'PONKE': 0.1,
      'POPCAT': 0.2,
      'BOME': 0.001,
      'AI16Z': 0.05,
      'GOAT': 0.3,
      'FWOG': 0.15,
      'DRIFT': 0.4,
    };
    
    return mockPrices[symbol] || 0;
  }

  /**
   * Check if user has sufficient balance for a deposit
   * Following Anchor validation patterns
   */
  async hasSufficientBalance(
    walletAddress: PublicKey,
    mintAddress: PublicKey,
    requiredAmount: number
  ): Promise<boolean> {
    try {
      const tokenBalance = await this.getTokenBalance(walletAddress, mintAddress);
      
      if (!tokenBalance) {
        console.log('❌ Token account does not exist');
        return false;
      }

      const hasBalance = tokenBalance.uiAmount >= requiredAmount;
      console.log('💰 Balance check:', tokenBalance.uiAmount, '>=', requiredAmount, '=', hasBalance);
      
      return hasBalance;
    } catch (error) {
      console.error('❌ Error checking sufficient balance:', error);
      return false;
    }
  }

  /**
   * Get connection instance
   */
  getConnection(): Connection {
    return this.connection;
  }
}

// Export singleton instance
export const balanceService = BalanceService.getInstance();
