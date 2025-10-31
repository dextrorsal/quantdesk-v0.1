import { Keypair, LAMPORTS_PER_SOL, PublicKey } from '@solana/web3.js';
import bs58 from 'bs58';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url'; // Added for __dirname equivalent in ESM
import * as dotenv from 'dotenv'; // ES Module import for dotenv

// __dirname equivalent for ES Modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

import { supabaseService } from '../../backend/src/services/supabaseService.js'; // Added .js
import { ReferralPayoutService } from '../../backend/src/services/referralPayout.js'; // Added .js
import { chatService } from '../../frontend/src/services/chatService.js'; // Reusing frontend chatService for sending messages, added .js

interface TestWallet {
  publicKey: string;
  secretKey: string; // Base58 encoded
}

const WALLETS_FILE = path.join(__dirname, 'test_wallets.json');

/**
 * Generates a specified number of Solana keypairs and saves them to a file.
 */
async function createTestWallets(numWallets: number): Promise<TestWallet[]> {
  console.log(`Generating ${numWallets} test wallets...`);
  const wallets: TestWallet[] = [];
  for (let i = 0; i < numWallets; i++) {
    const keypair = Keypair.generate();
    wallets.push({
      publicKey: keypair.publicKey.toBase58(),
      secretKey: bs58.encode(keypair.secretKey),
    });
  }
  fs.writeFileSync(WALLETS_FILE, JSON.stringify(wallets, null, 2));
  console.log(`Generated ${numWallets} wallets and saved to ${WALLETS_FILE}`);
  return wallets;
}

/**
 * Funds a list of given wallet public keys with a small amount of devnet SOL.
 */
async function fundWallets(wallets: TestWallet[], amountSol: number = 0.5): Promise<void> {
  console.log(`Funding ${wallets.length} wallets with ${amountSol} SOL each...`);
  const rpcUrl = process.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com';
  const payerSecret = process.env.PAYOUT_PAYER_SECRET_BASE58; // Ensure this env var is set

  if (!payerSecret) {
    console.error('PAYOUT_PAYER_SECRET_BASE58 is not set. Cannot fund wallets.');
    return;
  }

  const referralPayoutService = new ReferralPayoutService(rpcUrl, payerSecret);

  for (const wallet of wallets) {
    try {
      const existingBalance = await referralPayoutService.getConnection().getBalance(new PublicKey(wallet.publicKey));
      if (existingBalance < amountSol * LAMPORTS_PER_SOL) {
        console.log(`Funding ${wallet.publicKey}...`);
        await referralPayoutService.sendSol(wallet.publicKey, amountSol);
        console.log(`✅ Funded ${wallet.publicKey} with ${amountSol} SOL.`);
      } else {
        console.log(`Skipping funding for ${wallet.publicKey}, already has sufficient balance.`);
      }
    } catch (error) {
      console.error(`❌ Failed to fund wallet ${wallet.publicKey}:`, error);
    }
  }
  console.log('Wallet funding complete.');
}

/**
 * Inserts test users into Supabase, linking some as referrers/referees.
 */
async function seedSupabaseUsers(wallets: TestWallet[], numReferrals: number = 2): Promise<void> {
  console.log(`Seeding ${wallets.length} Supabase users and ${numReferrals} referrals...`);
  const userPromises = wallets.map(async (wallet, index) => {
    const userData = {
      wallet_pubkey: wallet.publicKey,
      username: `tester${index + 1}`,
      email: `tester${index + 1}@example.com`,
      role: 'tester',
    };
    await supabaseService.upsertUser(userData);
  });
  await Promise.all(userPromises);

  // Create some referrals
  if (wallets.length > numReferrals) {
    for (let i = 0; i < numReferrals; i++) {
      const referrer = wallets[0].publicKey; // First wallet is the referrer
      const referee = wallets[i + 1].publicKey; // Subsequent wallets are referees
      const priorRef = await supabaseService.select('referrals', '*', { referee_pubkey: referee });
      if (!priorRef?.length) {
        await supabaseService.insert('referrals', { referrer_pubkey: referrer, referee_pubkey: referee });
        await supabaseService.update('users', { referrer_pubkey: referrer }, { wallet_pubkey: referee });
        console.log(`Created referral: ${referrer} -> ${referee}`);
      }
    }
  }
  console.log('Supabase user and referral seeding complete.');
}

/**
 * Inserts a specified number of chat messages into Supabase for a given channel.
 */
async function seedChatMessages(channel: string = 'global', numMessages: number = 10, wallets: TestWallet[]): Promise<void> {
  console.log(`Seeding ${numMessages} chat messages in channel '${channel}'...`);
  const messagePromises = [];
  for (let i = 0; i < numMessages; i++) {
    const randomWallet = wallets[Math.floor(Math.random() * wallets.length)];
    const message = `Hello from ${randomWallet.publicKey.slice(0, 6)}! This is test message ${i + 1}.`;
    messagePromises.push(supabaseService.getClient().from('chat_messages').insert({
      channel,
      author_pubkey: randomWallet.publicKey,
      message,
    }));
  }
  await Promise.all(messagePromises);
  console.log('Chat message seeding complete.');
}

/**
 * Mocks trade events to trigger referral activations and earnings.
 */
async function mockTradeEvents(numEvents: number = 5, wallets: TestWallet[]): Promise<void> {
  console.log(`Mocking ${numEvents} trade events to activate referrals...`);
  for (let i = 0; i < numEvents; i++) {
    // Pick a random referee that has a referrer and is not yet activated
    const referees = await supabaseService.select('referrals', 'referee_pubkey', { activated_at: null });
    if (referees.length === 0) {
      console.log('No unactivated referees found to mock trade events for.');
      break;
    }

    const randomReferee = referees[Math.floor(Math.random() * referees.length)].referee_pubkey;
    const mockVolume = Math.random() * 100 + 10; // Random volume between 10 and 110
    
    try {
      // Directly call the backend RPC to simulate activation via trade volume
      // In a real scenario, a trading event would trigger this activation
      await supabaseService.getClient().rpc('activate_referral', {
        p_referee: randomReferee,
        p_min_volume: mockVolume,
      });
      console.log(`Mocked trade event for ${randomReferee} with volume ${mockVolume.toFixed(2)}. Attempted activation.`);
    } catch (error) {
      console.error(`❌ Failed to mock trade event for ${randomReferee}:`, error);
    }
  }
  console.log('Mock trade events complete.');
}

/**
 * Creates default chat channels in Supabase.
 */
async function seedChatChannels(): Promise<void> {
  console.log('Seeding default chat channels...');
  const defaultChannels = [
    { name: 'global', description: 'General discussion for all users', is_private: false },
    { name: 'trading', description: 'Discuss trading strategies and market movements', is_private: false },
    { name: 'alpha', description: 'Share alpha and market insights', is_private: false },
    { name: 'support', description: 'Get help and support from the community', is_private: false },
    { name: 'off-topic', description: 'Casual conversations and non-trading discussions', is_private: false },
  ];

  for (const channelData of defaultChannels) {
    const existingChannel = await supabaseService.select('chat_channels', '*', { name: channelData.name });
    if (!existingChannel || existingChannel.length === 0) {
      await supabaseService.insert('chat_channels', channelData);
      console.log(`Created channel: #${channelData.name}`);
    } else {
      console.log(`Channel #${channelData.name} already exists. Skipping.`);
    }
  }
  console.log('Default chat channels seeding complete.');
}

async function main() {
  // Ensure environment variables are loaded if running standalone
  dotenv.config({ path: path.resolve(__dirname, '../../.env') }); // Using dotenv ES module import

  console.log('Starting seed script...');

  // Connect to Supabase and Redis first
  // Redis connection is handled by backend on server boot, not directly here for seed script
  
  const numWallets = 5; // Create 5 test wallets
  const wallets = await createTestWallets(numWallets);
  await fundWallets(wallets);
  await seedSupabaseUsers(wallets);
  await seedChatChannels(); // Call new function
  await seedChatMessages('global', 10, wallets);
  await mockTradeEvents(5, wallets); // Mock 5 trade events for activation

  console.log('Seed script finished successfully!');
  process.exit(0);
}

main().catch((err) => {
  console.error('Seed script failed:', err);
  process.exit(1);
});

export { createTestWallets, fundWallets, seedSupabaseUsers, seedChatMessages, mockTradeEvents };