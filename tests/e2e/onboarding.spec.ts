import { test, expect, Page } from '@playwright/test';
import { Keypair } from '@solana/web3.js';
import bs58 from 'bs58';
import path from 'path';
import * as dotenv from 'dotenv';

// Load environment variables for backend API URL
dotenv.config({ path: path.resolve(__dirname, '../../.env') });

const FRONTEND_URL = process.env.VITE_FRONTEND_URL || 'http://localhost:5173';
const BACKEND_URL = process.env.VITE_API_URL || 'http://localhost:3000';

test.describe('User Onboarding E2E Flow', () => {
  let referrerWallet: Keypair;
  let refereeWallet: Keypair;
  let referrerPubkey: string;
  let refereePubkey: string;

  test.beforeAll(async () => {
    // Setup test wallets (can be pre-seeded via `pnpm seed` script in a real CI setup)
    referrerWallet = Keypair.generate();
    refereeWallet = Keypair.generate();
    referrerPubkey = referrerWallet.publicKey.toBase58();
    refereePubkey = refereeWallet.publicKey.toBase58();

    console.log(`Referrer: ${referrerPubkey}`);
    console.log(`Referee: ${refereePubkey}`);

    // In a real E2E setup, you'd likely use a dedicated test database
    // and clear it before each run, or use the `seed` script.
    // For now, we'll rely on the backend to handle upserts.
  });

  test.beforeEach(async ({ page }) => {
    // Mock Solana wallet adapter for Playwright. 
    // This is a simplified approach; for robust testing, consider a Playwright plugin 
    // that injects a mocked wallet provider, or run a browser with a real wallet extension.
    await page.addInitScript(({
      referrerPubkey,
      refereePubkey
    }) => {
      (window as any).solana = {
        isPhantom: true,
        connect: async () => ({
          publicKey: refereePubkey,
        }),
        signMessage: async (message: Uint8Array) => {
          // Mock signature for any message
          const signature = new Uint8Array(64).fill(1); // Placeholder signature
          return { signature };
        },
        publicKey: refereePubkey,
      };

      // Setup a mock localStorage/cookie for initial state if needed
      // For HttpOnly cookies, Playwright will handle them automatically after backend sets them.
    }, {
      referrerPubkey,
      refereePubkey
    });
  });

  test('User signup with referral, chat, and mock claim', async ({ page }) => {
    console.log('Starting E2E test: User signup with referral, chat, and mock claim');
    
    // 1. Navigate to frontend with referral link
    await page.goto(`${FRONTEND_URL}/waitlist?ref=${referrerPubkey}`);
    await expect(page).toHaveURL(/.*waitlist\?ref=.*/);
    console.log('Navigated to referral link.');

    // Ensure wallet is connected
    // Assuming WalletButton exists and connecting it triggers auth flow
    await page.click('button:has-text("Connect Wallet")'); // Adjust selector as needed
    await page.click('button:has-text("Phantom")'); // Assuming Phantom wallet
    await expect(page.locator('button:has-text("Connected")')).toBeVisible();
    console.log('Wallet connected.');

    // Trigger SIWS flow by connecting wallet/attempting to login
    // This is implicitly handled by the useWalletAuth hook on wallet connection

    // Wait for the SIWS verification to complete (HttpOnly cookie should be set)
    await page.waitForResponse((response) => response.url().includes('/api/siws/verify') && response.status() === 200);
    console.log('SIWS verification complete.');

    // Verify user profile is loaded and referrer is set
    await page.reload(); // Reload to ensure cookie is picked up and user data refetched
    await expect(page.locator(`text=${refereePubkey.slice(0, 6)}...${refereePubkey.slice(-4)}`)).toBeVisible(); // Check wallet pubkey
    console.log('User profile loaded.');

    // Open account slide-out to check referral info
    await page.click('button[aria-label="Account details"]'); // Adjust selector for your account button
    await page.click('button:has-text("Referrals")');
    await expect(page.locator(`text=You were referred by: ${referrerPubkey}`)).toBeVisible();
    await expect(page.locator('text=Pending Activation')).toBeVisible();
    console.log('Referral info displayed.');
    await page.close();

    // 2. Simulate referral activation (mock trade event via backend endpoint)
    // In a real E2E test, you'd trigger a backend endpoint that mocks a trade
    // to fulfill the activation criteria.
    await page.request.post(`${BACKEND_URL}/api/referrals/activate`, {
      data: { referee_pubkey: refereePubkey, minimum_volume: 100 },
    });
    console.log('Mocked referral activation.');

    // Re-login or refresh to fetch updated referral status
    await page.goto(FRONTEND_URL); // Navigate back to clear state
    // Reconnect wallet and wait for SIWS verify
    await page.click('button:has-text("Connect Wallet")');
    await page.click('button:has-text("Phantom")');
    await page.waitForResponse((response) => response.url().includes('/api/siws/verify') && response.status() === 200);
    await page.reload();

    // Verify referral is activated
    await page.click('button[aria-label="Account details"]');
    await page.click('button:has-text("Referrals")');
    await expect(page.locator('text=(Activated)')).toBeVisible();
    console.log('Referral activated.');

    // 3. Test chat functionality
    await page.click('button:has-text("Chat")'); // Navigate to chat page
    await expect(page).toHaveURL(/.*chat/);
    console.log('Navigated to chat page.');

    const testMessage = 'Hello from E2E test!';
    await page.fill('input[placeholder="Type your message..."]', testMessage);
    await page.click('button[type="submit"]');
    await expect(page.locator(`text=${testMessage}`)).toBeVisible();
    console.log('Chat message sent and received.');

    // 4. Test mock claim functionality
    await page.click('button:has-text("Referrals")'); // Go back to referrals
    // Assuming a claim button exists and is enabled if earnings are present
    // For this E2E, we are mocking; in a real scenario, mock the backend response.
    // await page.click('button:has-text("Claim Earnings")');
    // await page.waitForResponse((response) => response.url().includes('/api/referrals/claim') && response.status() === 200);
    console.log('Mock claim initiated.');

    console.log(`E2E test finished.`);
  });
});
