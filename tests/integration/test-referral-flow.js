const { Pool } = require('pg');
const request = require('supertest');
const { referralService } = require('../backend/src/services/referralService');
const { ReferralPayoutService } = require('../backend/src/services/referralPayout');
const { supabaseService } = require('../backend/src/services/supabaseService');
const app = require('../backend/src/server').default; // Assuming your express app is exported as default

// Mock environment variables for testing
process.env.DATABASE_URL = process.env.DATABASE_URL || 'postgresql://postgres:postgres@localhost:5432/test_db';
process.env.RPC_URL = process.env.RPC_URL || 'https://api.devnet.solana.com';
process.env.PAYOUT_PAYER_SECRET_BASE58 = process.env.PAYOUT_PAYER_SECRET_BASE58 || 'mock_payer_private_key'; // Replace with a valid mock key if needed for actual Solana interaction tests

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
});

describe('Referral System Integration Flow', () => {
  let referrerWallet;
  let refereeWallet;
  let referrerUserId;
  let refereeUserId;

  beforeAll(async () => {
    // Clear and re-seed test database if necessary, or ensure a clean state
    // For simplicity, we'll just clear relevant tables for this test
    await pool.query('DELETE FROM payouts');
    await pool.query('DELETE FROM trades');
    await pool.query('DELETE FROM referrals');
    await pool.query('DELETE FROM users');

    // Create mock users
    referrerWallet = 'referrer_wallet_' + Date.now();
    refereeWallet = 'referee_wallet_' + Date.now();

    let res = await supabaseService.getClient().from('users').insert({ wallet_pubkey: referrerWallet, username: 'referrer' }).select();
    referrerUserId = res.data[0].id;

    res = await supabaseService.getClient().from('users').insert({ wallet_pubkey: refereeWallet, username: 'referee', referrer_pubkey: referrerWallet }).select();
    refereeUserId = res.data[0].id;
  });

  afterAll(async () => {
    await pool.end();
  });

  it('should complete a full referral flow from activation to payout', async () => {
    // 1. Activate referral
    const activateRes = await request(app)
      .post('/api/referrals/activate')
      .send({ referee_pubkey: refereeWallet, minimum_volume: 100 }); // Mock minimum volume

    expect(activateRes.statusCode).toEqual(200);
    expect(activateRes.body.success).toBe(true);

    // Verify referral is activated in DB
    const { data: activatedReferral } = await supabaseService.getClient()
      .from('referrals')
      .select('activated_at, activated')
      .eq('referee_pubkey', refereeWallet)
      .single();
    expect(activatedReferral.activated).toBe(true);
    expect(activatedReferral.activated_at).not.toBeNull();

    // 2. Simulate trades by referee with fees
    const mockMarketId = 'a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11'; // Dummy UUID
    await supabaseService.getClient().from('trades').insert([
      {
        user_id: refereeUserId, market_id: mockMarketId, order_id: 'b0eebc99-9c0b-4ef8-bb6d-6bb9bd380a12', trade_account: refereeWallet,
        side: 'buy', size: 1, price: 100, fees: 0.1, created_at: new Date()
      },
      {
        user_id: refereeUserId, market_id: mockMarketId, order_id: 'c0eebc99-9c0b-4ef8-bb6d-6bb9bd380a13', trade_account: refereeWallet,
        side: 'sell', size: 0.5, price: 200, fees: 0.05, created_at: new Date()
      },
      {
        user_id: referrerUserId, market_id: mockMarketId, order_id: 'd0eebc99-9c0b-4ef8-bb6d-6bb9bd380a14', trade_account: referrerWallet,
        side: 'buy', size: 2, price: 50, fees: 0.02, created_at: new Date()
      }, // Trade by referrer, should not count for referral earnings
      {
        user_id: refereeUserId, market_id: mockMarketId, order_id: 'e0eebc99-9c0b-4ef8-bb6d-6bb9bd380a15', trade_account: refereeWallet,
        side: 'buy', size: 1, price: 100, fees: -0.01, created_at: new Date()
      }, // Maker rebate, should be ignored
    ]);

    // 3. Verify referrer earnings
    const summaryRes = await request(app)
      .get(`/api/referrals/summary?wallet=${referrerWallet}`);

    expect(summaryRes.statusCode).toEqual(200);
    expect(summaryRes.body.count).toBe(1); // One activated referral
    // (0.1 * 0.25) + (0.05 * 0.25) = 0.025 + 0.0125 = 0.0375
    expect(summaryRes.body.earnings).toBeCloseTo(0.0375);

    // 4. Simulate referrer claiming payout
    // Mock ReferralPayoutService.sendSol to prevent actual SOL transfer during test
    jest.spyOn(ReferralPayoutService.prototype, 'sendSol').mockResolvedValueOnce({
      txSig: 'mock_claim_tx_sig',
      lamports: Math.round(0.0375 * 1_000_000_000),
    });

    const claimRes = await request(app)
      .post('/api/referrals/claim')
      .send({ referrer: referrerWallet });

    expect(claimRes.statusCode).toEqual(200);
    expect(claimRes.body.success).toBe(true);
    expect(claimRes.body.tx).toEqual('mock_claim_tx_sig');
    expect(claimRes.body.lamports).toBeCloseTo(Math.round(0.0375 * 1_000_000_000));

    // Verify payout record in DB
    const { data: payoutRecord } = await supabaseService.getClient()
      .from('payouts')
      .select('amount_sol, tx_signature')
      .eq('referrer_pubkey', referrerWallet)
      .single();

    expect(payoutRecord.amount_sol).toBeCloseTo(0.0375);
    expect(payoutRecord.tx_signature).toEqual('mock_claim_tx_sig');
  });

  it('should apply referee fee discount for activated referee', async () => {
    // This part requires mocking the matching service's fee calculation
    // For a true integration test, we'd run the actual matching service.
    // Here, we simulate the outcome of getRefereeFeeDiscount being called.
    const discount = await referralService.getRefereeFeeDiscount(refereeWallet);
    expect(discount).toBe(0.10); // Expecting 10% discount for activated referee
  });
});
