// Solana Wallet Integration for MIKEY-AI
// Allows MIKEY to interact with Solana wallets and check balances

import { DynamicTool } from '@langchain/core/tools';
import { Connection, PublicKey, LAMPORTS_PER_SOL } from '@solana/web3.js';
import { config } from '../config';
import { WalletAddressExtractor } from '../utils/WalletAddressExtractor';

export class SolanaWalletTools {
  private connection: Connection;

  constructor() {
    // Use devnet for development
    this.connection = new Connection(
      process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com',
      'confirmed'
    );
  }

  /**
   * Check SOL wallet balance
   */
  static createCheckBalanceTool(): DynamicTool {
    return new DynamicTool({
      name: 'check_sol_balance',
      description: 'Check SOL balance for a Solana wallet address. Input can be natural language containing a wallet address.',
      func: async (input: string) => {
        try {
          // Extract wallet address from natural language
          const { address, context } = WalletAddressExtractor.extractWalletWithContext(input);
          
          if (!address) {
            const helpMessage = WalletAddressExtractor.generateAddressHelp(input);
            return JSON.stringify({
              success: false,
              error: helpMessage,
              suggestion: 'Try: "Check SOL balance for wallet [your-address]"'
            });
          }

          const connection = new Connection(
            process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com',
            'confirmed'
          );

          const publicKey = new PublicKey(address);
          const balance = await connection.getBalance(publicKey);
          const solBalance = balance / LAMPORTS_PER_SOL;

          // Get account info
          const accountInfo = await connection.getAccountInfo(publicKey);
          const isExecutable = accountInfo?.executable || false;
          const owner = accountInfo?.owner?.toString();

          return JSON.stringify({
            success: true,
            wallet: address,
            balance: {
              lamports: balance,
              sol: solBalance.toFixed(6)
            },
            accountInfo: {
              executable: isExecutable,
              owner: owner,
              dataLength: accountInfo?.data.length || 0
            },
            context: context,
            network: 'devnet',
            timestamp: new Date().toISOString()
          });
        } catch (error) {
          return JSON.stringify({
            success: false,
            error: `Error checking balance: ${error.message}`,
            input: input,
            suggestion: 'Please verify the wallet address is correct and try again'
          });
        }
      }
    });
  }

  /**
   * Get wallet transaction history
   */
  static createGetTransactionHistoryTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_transaction_history',
      description: 'Get recent transaction history for a Solana wallet. Input can be natural language containing a wallet address.',
      func: async (input: string) => {
        try {
          // Extract wallet address from natural language
          const address = WalletAddressExtractor.extractWalletAddress(input);
          
          if (!address) {
            const helpMessage = WalletAddressExtractor.generateAddressHelp(input);
            return JSON.stringify({
              success: false,
              error: helpMessage,
              suggestion: 'Try: "Get transaction history for wallet [your-address]"'
            });
          }

          const connection = new Connection(
            process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com',
            'confirmed'
          );

          const publicKey = new PublicKey(address);
          
          // Get recent signatures (last 10 transactions)
          const signatures = await connection.getSignaturesForAddress(publicKey, { limit: 10 });
          
          const transactions = [];
          for (const sig of signatures.slice(0, 5)) { // Limit to 5 for performance
            try {
              const tx = await connection.getTransaction(sig.signature, {
                commitment: 'confirmed',
                maxSupportedTransactionVersion: 0
              });
              
              if (tx) {
                transactions.push({
                  signature: sig.signature,
                  slot: sig.slot,
                  blockTime: sig.blockTime ? new Date(sig.blockTime * 1000).toISOString() : null,
                  fee: tx.meta?.fee || 0,
                  success: tx.meta?.err ? false : true,
                  error: tx.meta?.err || null
                });
              }
            } catch (txError) {
              // Skip failed transaction fetches
              continue;
            }
          }

          return JSON.stringify({
            success: true,
            wallet: address,
            transactions: transactions,
            totalSignatures: signatures.length,
            network: 'devnet',
            timestamp: new Date().toISOString()
          });
        } catch (error) {
          return JSON.stringify({
            success: false,
            error: `Error getting transaction history: ${error.message}`,
            input: input,
            suggestion: 'Please verify the wallet address is correct and try again'
          });
        }
      }
    });
  }

  /**
   * Check if wallet is a program account
   */
  static createCheckProgramAccountTool(): DynamicTool {
    return new DynamicTool({
      name: 'check_program_account',
      description: 'Check if a Solana address is a program account and get its details. Input can be natural language containing an address.',
      func: async (input: string) => {
        try {
          // Extract address from natural language
          const address = WalletAddressExtractor.extractWalletAddress(input);
          
          if (!address) {
            const helpMessage = WalletAddressExtractor.generateAddressHelp(input);
            return JSON.stringify({
              success: false,
              error: helpMessage,
              suggestion: 'Try: "Check if [address] is a program account"'
            });
          }

          const connection = new Connection(
            process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com',
            'confirmed'
          );

          const publicKey = new PublicKey(address);
          const accountInfo = await connection.getAccountInfo(publicKey);

          if (!accountInfo) {
            return JSON.stringify({
              success: true,
              address: address,
              exists: false,
              message: 'Account does not exist on devnet'
            });
          }

          return JSON.stringify({
            success: true,
            address: address,
            exists: true,
            accountInfo: {
              executable: accountInfo.executable,
              owner: accountInfo.owner.toString(),
              lamports: accountInfo.lamports,
              dataLength: accountInfo.data.length,
              rentEpoch: accountInfo.rentEpoch
            },
            isProgram: accountInfo.executable,
            network: 'devnet',
            timestamp: new Date().toISOString()
          });
        } catch (error) {
          return JSON.stringify({
            success: false,
            error: `Error checking program account: ${error.message}`,
            input: input,
            suggestion: 'Please verify the address is correct and try again'
          });
        }
      }
    });
  }

  /**
   * Get all available Solana wallet tools
   */
  static getAllTools(): DynamicTool[] {
    return [
      this.createCheckBalanceTool(),
      this.createGetTransactionHistoryTool(),
      this.createCheckProgramAccountTool()
    ];
  }
}
