import { databaseService } from './supabaseDatabase';

export class ReferralService {
  constructor() {}

  async calculateReferralEarnings(referrerWallet: string): Promise<number> {
    const referrals = await databaseService.select('referrals', '*', { referrer_pubkey: referrerWallet });
    const activatedReferrals = referrals?.filter((ref: any) => ref.activated_at !== null) || [];

    if (activatedReferrals.length === 0) {
      return 0; // No activated referrals, no earnings
    }

    // Get all trades by referees of this referrer
    const refereePubkeys = activatedReferrals.map((ref: any) => ref.referee_pubkey);
    const { data: trades, error } = await databaseService.getClient()
      .from('trades')
      .select('fees, user_id')
      .in('trade_account', refereePubkeys); // Assuming trade_account stores the user's wallet pubkey

    if (error) {
      console.error('Error fetching trades for referral earnings:', error);
      return 0;
    }

    let totalEarnings = 0;
    const REFERRAL_SHARE_RATE = 0.25; // 25% of fees for referrer

    for (const trade of trades || []) {
      // Only consider positive fees (taker fees, not maker rebates)
      if (trade.fees > 0) {
        totalEarnings += trade.fees * REFERRAL_SHARE_RATE;
      }
    }

    return totalEarnings;
  }

  async getRefereeFeeDiscount(refereeWallet: string): Promise<number> {
    const { data: referral, error } = await databaseService.getClient()
      .from('referrals')
      .select('activated_at')
      .eq('referee_pubkey', refereeWallet)
      .single();

    if (error && error.code !== 'PGRST116') { // PGRST116 means no rows found
      console.error('Error fetching referee referral status:', error);
      return 0; // No discount on error
    }

    if (referral?.activated_at) {
      // Referee is activated, apply discount
      return 0.10; // 10% discount
    } else {
      return 0; // No discount if not activated or no referral
    }
  }

  async recordPayout(referrerWallet: string, amountSol: number, txSig: string): Promise<void> {
    // Record the payout in Supabase
    await databaseService.insert('payouts', {
      referrer_pubkey: referrerWallet,
      amount_sol: amountSol,
      tx_signature: txSig,
      created_at: new Date(),
    });
  }
}

export const referralService = new ReferralService();
