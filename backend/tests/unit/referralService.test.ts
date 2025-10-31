import { ReferralService } from '../../src/services/referralService';
import { supabaseService } from '../../src/services/supabaseService';

// Mocking supabaseService for isolation
jest.mock('../../src/services/supabaseService', () => ({
  supabaseService: {
    getClient: jest.fn().mockReturnValue({
      from: jest.fn((tableName) => ({
        select: jest.fn(() => ({ in: jest.fn(() => ({ data: [], error: null })), eq: jest.fn(() => ({ single: jest.fn(() => ({ data: null, error: { code: 'PGRST116' } })) })) })), // Mock for no existing data
        insert: jest.fn(() => ({ data: [], error: null })),
      })),
    }),
    select: jest.fn(() => []), // Default mock for select returning empty array
    insert: jest.fn(() => ({ data: [], error: null })),
  },
}));

describe('ReferralService', () => {
  let referralService: ReferralService;

  beforeEach(() => {
    referralService = new ReferralService();
    jest.clearAllMocks();
  });

  describe('calculateReferralEarnings', () => {
    it('should return 0 if no activated referrals', async () => {
      (supabaseService.select as jest.Mock).mockResolvedValueOnce([]); // No referrals
      const earnings = await referralService.calculateReferralEarnings('test_referrer');
      expect(earnings).toBe(0);
    });

    it('should return 0 if no trades for activated referees', async () => {
      (supabaseService.select as jest.Mock).mockResolvedValueOnce([
        { referee_pubkey: 'referee1', activated_at: new Date() }
      ]);
      (supabaseService.getClient().from('trades').select().in as jest.Mock).mockResolvedValueOnce({ data: [], error: null });

      const earnings = await referralService.calculateReferralEarnings('test_referrer');
      expect(earnings).toBe(0);
    });

    it('should calculate earnings correctly for activated referees with positive fees', async () => {
      (supabaseService.select as jest.Mock).mockResolvedValueOnce([
        { referee_pubkey: 'referee1', activated_at: new Date() },
        { referee_pubkey: 'referee2', activated_at: new Date() },
      ]);
      (supabaseService.getClient().from('trades').select().in as jest.Mock).mockResolvedValueOnce({
        data: [
          { fees: 10, user_id: 'referee1' },
          { fees: 20, user_id: 'referee2' },
          { fees: -5, user_id: 'referee1' }, // Maker rebate, should be ignored
        ],
        error: null,
      });

      const earnings = await referralService.calculateReferralEarnings('test_referrer');
      // (10 * 0.25) + (20 * 0.25) = 2.5 + 5 = 7.5
      expect(earnings).toBe(7.5);
    });

    it('should handle error when fetching trades', async () => {
      (supabaseService.select as jest.Mock).mockResolvedValueOnce([
        { referee_pubkey: 'referee1', activated_at: new Date() }
      ]);
      (supabaseService.getClient().from('trades').select().in as jest.Mock).mockResolvedValueOnce({ data: null, error: { message: 'DB Error' } });

      const earnings = await referralService.calculateReferralEarnings('test_referrer');
      expect(earnings).toBe(0);
    });
  });

  describe('getRefereeFeeDiscount', () => {
    it('should return 0 if no referral found for referee', async () => {
      (supabaseService.getClient().from as jest.Mock).mockReturnValue({
        select: jest.fn(() => ({
          eq: jest.fn(() => ({
            single: jest.fn(() => ({ data: null, error: { code: 'PGRST116' } }))
          }))
        }))
      });
      const discount = await referralService.getRefereeFeeDiscount('non_referred_referee');
      expect(discount).toBe(0);
    });

    it('should return 0.10 if referee is activated', async () => {
      (supabaseService.getClient().from as jest.Mock).mockReturnValue({
        select: jest.fn(() => ({
          eq: jest.fn(() => ({
            single: jest.fn(() => ({ data: { activated_at: new Date() }, error: null }))
          }))
        }))
      });
      const discount = await referralService.getRefereeFeeDiscount('activated_referee');
      expect(discount).toBe(0.10);
    });

    it('should return 0 if referee is not activated', async () => {
      (supabaseService.getClient().from as jest.Mock).mockReturnValue({
        select: jest.fn(() => ({
          eq: jest.fn(() => ({
            single: jest.fn(() => ({ data: { activated_at: null }, error: null }))
          }))
        }))
      });
      const discount = await referralService.getRefereeFeeDiscount('inactive_referee');
      expect(discount).toBe(0);
    });

    it('should return 0 on database error when fetching referral status', async () => {
      (supabaseService.getClient().from as jest.Mock).mockReturnValue({
        select: jest.fn(() => ({
          eq: jest.fn(() => ({
            single: jest.fn(() => ({ data: null, error: { message: 'DB Error', code: '500' } }))
          }))
        }))
      });
      const discount = await referralService.getRefereeFeeDiscount('error_referee');
      expect(discount).toBe(0);
    });
  });

  describe('recordPayout', () => {
    it('should record payout correctly', async () => {
      const insertSpy = jest.spyOn(supabaseService, 'insert');
      await referralService.recordPayout('referrer_wallet', 0.5, 'tx_signature_123');
      expect(insertSpy).toHaveBeenCalledWith('payouts', {
        referrer_pubkey: 'referrer_wallet',
        amount_sol: 0.5,
        tx_signature: 'tx_signature_123',
        created_at: expect.any(Date),
      });
    });
  });
});
