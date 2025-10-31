// Enhanced Wallet Address Extraction Utility
// Improves parsing of Solana wallet addresses from natural language queries

export class WalletAddressExtractor {
  /**
   * Extract Solana wallet address from natural language query
   * Supports various formats and contexts
   */
  static extractWalletAddress(query: string): string | null {
    // Clean the query
    const cleanQuery = query.toLowerCase().trim();
    
    // Multiple regex patterns for different wallet address formats
    const patterns = [
      // Standard Solana address (32-44 characters, base58)
      /[1-9A-HJ-NP-Za-km-z]{32,44}/g,
      // Addresses with common prefixes
      /(?:wallet|address|pubkey|public key|account)\s*:?\s*([1-9A-HJ-NP-Za-km-z]{32,44})/gi,
      // Addresses in quotes or brackets
      /["'`]?([1-9A-HJ-NP-Za-km-z]{32,44})["'`]?/g,
      // Addresses after common phrases
      /(?:for|of|at|@)\s+([1-9A-HJ-NP-Za-km-z]{32,44})/gi
    ];

    for (const pattern of patterns) {
      const matches = cleanQuery.match(pattern);
      if (matches && matches.length > 0) {
        // Return the first valid match
        for (const match of matches) {
          const address = match.replace(/[^1-9A-HJ-NP-Za-km-z]/g, '');
          if (this.isValidSolanaAddress(address)) {
            return address;
          }
        }
      }
    }

    return null;
  }

  /**
   * Validate if a string is a valid Solana address
   */
  static isValidSolanaAddress(address: string): boolean {
    if (!address || address.length < 32 || address.length > 44) {
      return false;
    }

    // Check if it's valid base58
    const base58Regex = /^[1-9A-HJ-NP-Za-km-z]+$/;
    if (!base58Regex.test(address)) {
      return false;
    }

    // Additional validation for common invalid patterns
    const invalidPatterns = [
      /^0+$/, // All zeros
      /^1+$/, // All ones
      /^[A-Z]+$/, // All uppercase (suspicious)
      /^[a-z]+$/  // All lowercase (suspicious)
    ];

    for (const pattern of invalidPatterns) {
      if (pattern.test(address)) {
        return false;
      }
    }

    return true;
  }

  /**
   * Extract multiple wallet addresses from a query
   */
  static extractMultipleWallets(query: string): string[] {
    const addresses: string[] = [];
    const pattern = /[1-9A-HJ-NP-Za-km-z]{32,44}/g;
    let match;

    while ((match = pattern.exec(query)) !== null) {
      const address = match[0];
      if (this.isValidSolanaAddress(address) && !addresses.includes(address)) {
        addresses.push(address);
      }
    }

    return addresses;
  }

  /**
   * Extract wallet address with context (what the user wants to do)
   */
  static extractWalletWithContext(query: string): { address: string | null; context: string } {
    const address = this.extractWalletAddress(query);
    
    // Extract context/action
    const contextPatterns = [
      { pattern: /check.*balance/i, context: 'balance_check' },
      { pattern: /portfolio/i, context: 'portfolio_analysis' },
      { pattern: /transaction.*history/i, context: 'transaction_history' },
      { pattern: /risk.*analysis/i, context: 'risk_analysis' },
      { pattern: /analyze/i, context: 'analysis' },
      { pattern: /get.*data/i, context: 'data_retrieval' }
    ];

    let context = 'general';
    for (const { pattern, context: ctx } of contextPatterns) {
      if (pattern.test(query)) {
        context = ctx;
        break;
      }
    }

    return { address, context };
  }

  /**
   * Generate helpful error message for invalid addresses
   */
  static generateAddressHelp(query: string): string {
    const potentialAddresses = query.match(/[1-9A-HJ-NP-Za-km-z]{20,50}/g) || [];
    
    if (potentialAddresses.length === 0) {
      return 'No wallet address found in your query. Please provide a valid Solana wallet address (32-44 characters).';
    }

    const invalidAddresses = potentialAddresses.filter(addr => !this.isValidSolanaAddress(addr));
    if (invalidAddresses.length > 0) {
      return `Invalid wallet address format: ${invalidAddresses[0]}. Solana addresses should be 32-44 characters long and use base58 encoding.`;
    }

    return 'Wallet address format appears valid.';
  }
}
