import { PublicKey } from '@solana/web3.js';

/**
 * Pyth Network Price Feed Accounts on Devnet
 * 
 * These are the official Pyth price feed accounts for various assets.
 * Prices are updated frequently by the Pyth oracle network.
 */
export const PYTH_PRICE_FEEDS = {
  SOL_USD: new PublicKey('H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG'),
  BTC_USD: new PublicKey('HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J'),
  ETH_USD: new PublicKey('JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB'),
};

/**
 * Program Constants
 */
export const QUANTDESK_PROGRAM_ID = new PublicKey('C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw');

