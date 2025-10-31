#!/usr/bin/env tsx

/**
 * CLI Entry Point for Solana DeFi Trading Intelligence AI
 * Run with: npx tsx src/cli.ts
 */

import { tradingCLI } from '@/cli/TradingCLI';
import { systemLogger } from '@/utils/logger';

async function main(): Promise<void> {
  try {
    console.log('🚀 Starting Solana DeFi Trading Intelligence AI CLI...');
    
    // Start the CLI
    await tradingCLI.start();
    
  } catch (error) {
    console.error('❌ Failed to start CLI:', error);
    systemLogger.shutdown('CLI startup failed');
    process.exit(1);
  }
}

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught Exception:', error);
  systemLogger.shutdown('Uncaught exception');
  process.exit(1);
});

process.on('unhandledRejection', (reason) => {
  console.error('❌ Unhandled Rejection:', reason);
  systemLogger.shutdown('Unhandled rejection');
  process.exit(1);
});

// Start the application
main().catch((error) => {
  console.error('❌ Main function failed:', error);
  process.exit(1);
});
