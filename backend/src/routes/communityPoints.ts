// QuantDesk Community Points API Endpoints
// Enhanced implementation exceeding Drift's capabilities
// "More Open Than Drift" competitive positioning

import { Router, Request, Response } from 'express';
import { CommunityPointsService } from '../services/communityPointsService';
import { DatabaseService } from '../services/supabaseDatabase';
import { rateLimiter } from '../middleware/rateLimiting';
import { authenticateWallet } from '../middleware/auth';

const router = Router();
const dbService = new DatabaseService();
const communityService = new CommunityPointsService(dbService);

// ==================== USER MANAGEMENT ENDPOINTS ====================

/**
 * POST /api/community/register
 * Register a new user in the community points system
 * Enhanced with early adopter bonuses (2x points for first 100 users)
 */
router.post('/register', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { wallet_address, username, email } = req.body;

    if (!wallet_address) {
      return res.status(400).json({ error: 'Wallet address is required' });
    }

    const user = await communityService.registerUser(wallet_address, username, email);
    
    res.status(201).json({
      success: true,
      message: 'User registered successfully',
      user: {
        id: user.id,
        wallet_address: user.wallet_address,
        username: user.username,
        total_points: user.total_points,
        level: user.level,
        created_at: user.created_at
      }
    });
  } catch (error) {
    console.error('Error registering user:', error);
    res.status(500).json({ 
      error: 'Failed to register user',
      details: error.message 
    });
  }
});

/**
 * GET /api/community/user/:wallet
 * Get user information and statistics
 */
router.get('/user/:wallet', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { wallet } = req.params;
    
    const user = await communityService.getUserByWallet(wallet);
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    const stats = await communityService.getUserStats(user.id);
    
    res.json({
      success: true,
      user: {
        id: user.id,
        wallet_address: user.wallet_address,
        username: user.username,
        total_points: user.total_points,
        level: user.level,
        created_at: user.created_at
      },
      stats: {
        badge_count: stats.badge_count,
        transaction_count: stats.transaction_count,
        rank: stats.rank,
        level: stats.level
      }
    });
  } catch (error) {
    console.error('Error getting user:', error);
    res.status(500).json({ 
      error: 'Failed to get user',
      details: error.message 
    });
  }
});

// ==================== POINTS MANAGEMENT ENDPOINTS ====================

/**
 * POST /api/community/points/award
 * Award points to a user (admin/automated only)
 * Enhanced with anti-gaming measures exceeding Drift's simple approach
 */
router.post('/points/award', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { user_id, points, source, description, metadata } = req.body;

    if (!user_id || !points || !source) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Anti-gaming: Validate points amount
    if (points <= 0 || points > 10000) {
      return res.status(400).json({ error: 'Invalid points amount' });
    }

    const transaction = await communityService.awardPoints(
      user_id, 
      points, 
      'earned', 
      source, 
      description, 
      metadata
    );
    
    res.status(201).json({
      success: true,
      message: 'Points awarded successfully',
      transaction: {
        id: transaction.id,
        points: transaction.points,
        source: transaction.source,
        description: transaction.description,
        created_at: transaction.created_at
      }
    });
  } catch (error) {
    console.error('Error awarding points:', error);
    res.status(500).json({ 
      error: 'Failed to award points',
      details: error.message 
    });
  }
});

/**
 * GET /api/community/points/transactions/:userId
 * Get user's points transaction history
 */
router.get('/points/transactions/:userId', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    const { limit = 50, offset = 0 } = req.query;

    const transactions = await communityService.getRecentTransactions(userId, 24 * 30); // Last 30 days
    
    res.json({
      success: true,
      transactions: transactions.slice(Number(offset), Number(offset) + Number(limit)),
      total: transactions.length
    });
  } catch (error) {
    console.error('Error getting transactions:', error);
    res.status(500).json({ 
      error: 'Failed to get transactions',
      details: error.message 
    });
  }
});

// ==================== BADGE SYSTEM ENDPOINTS ====================

/**
 * GET /api/community/badges
 * Get all available badges
 * Enhanced gamification system exceeding Drift's simple approach
 */
router.get('/badges', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { data: badges, error } = await dbService.client
      .from('badges')
      .select('*')
      .eq('is_active', true)
      .order('points_required', { ascending: true });

    if (error) throw error;
    
    res.json({
      success: true,
      badges: badges || []
    });
  } catch (error) {
    console.error('Error getting badges:', error);
    res.status(500).json({ 
      error: 'Failed to get badges',
      details: error.message 
    });
  }
});

/**
 * GET /api/community/badges/user/:userId
 * Get user's earned badges
 */
router.get('/badges/user/:userId', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const badges = await communityService.getUserBadges(userId);
    
    res.json({
      success: true,
      badges: badges
    });
  } catch (error) {
    console.error('Error getting user badges:', error);
    res.status(500).json({ 
      error: 'Failed to get user badges',
      details: error.message 
    });
  }
});

// ==================== REDEMPTION SYSTEM ENDPOINTS ====================

/**
 * GET /api/community/redemptions
 * Get all available redemption options
 * Enhanced with MIKEY AI access tiers exceeding Drift's capabilities
 */
router.get('/redemptions', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { data: redemptions, error } = await dbService.client
      .from('redemption_options')
      .select('*')
      .eq('is_active', true)
      .order('points_cost', { ascending: true });

    if (error) throw error;
    
    res.json({
      success: true,
      redemptions: redemptions || []
    });
  } catch (error) {
    console.error('Error getting redemptions:', error);
    res.status(500).json({ 
      error: 'Failed to get redemptions',
      details: error.message 
    });
  }
});

/**
 * POST /api/community/redemptions/redeem
 * Redeem points for rewards
 * Enhanced with MIKEY AI access tiers exceeding Drift's capabilities
 */
router.post('/redemptions/redeem', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { user_id, redemption_option_id } = req.body;

    if (!user_id || !redemption_option_id) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const redemption = await communityService.redeemPoints(user_id, redemption_option_id);
    
    res.status(201).json({
      success: true,
      message: 'Points redeemed successfully',
      redemption: {
        id: redemption.id,
        points_spent: redemption.points_spent,
        redeemed_at: redemption.redeemed_at,
        expires_at: redemption.expires_at,
        status: redemption.status
      }
    });
  } catch (error) {
    console.error('Error redeeming points:', error);
    res.status(500).json({ 
      error: 'Failed to redeem points',
      details: error.message 
    });
  }
});

/**
 * GET /api/community/redemptions/user/:userId
 * Get user's redemption history
 */
router.get('/redemptions/user/:userId', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const redemptions = await communityService.getRecentRedemptions(userId, 24 * 30); // Last 30 days
    
    res.json({
      success: true,
      redemptions: redemptions
    });
  } catch (error) {
    console.error('Error getting user redemptions:', error);
    res.status(500).json({ 
      error: 'Failed to get user redemptions',
      details: error.message 
    });
  }
});

// ==================== LEADERBOARDS ENDPOINTS ====================

/**
 * GET /api/community/leaderboard/points
 * Get points leaderboard
 * Enhanced community engagement exceeding Drift's simple approach
 */
router.get('/leaderboard/points', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { limit = 10 } = req.query;
    
    const topUsers = await communityService.getTopUsers(Number(limit));
    
    res.json({
      success: true,
      leaderboard: topUsers.map((user, index) => ({
        rank: index + 1,
        wallet_address: user.wallet_address,
        username: user.username,
        total_points: user.total_points,
        level: user.level
      }))
    });
  } catch (error) {
    console.error('Error getting leaderboard:', error);
    res.status(500).json({ 
      error: 'Failed to get leaderboard',
      details: error.message 
    });
  }
});

/**
 * GET /api/community/leaderboard/badges
 * Get badge collection leaderboard
 */
router.get('/leaderboard/badges', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { limit = 10 } = req.query;
    
    const { data: badgeLeaderboard, error } = await dbService.client
      .from('user_stats')
      .select('wallet_address, username, badge_count, total_points')
      .order('badge_count', { ascending: false })
      .limit(Number(limit));

    if (error) throw error;
    
    res.json({
      success: true,
      leaderboard: (badgeLeaderboard || []).map((user, index) => ({
        rank: index + 1,
        wallet_address: user.wallet_address,
        username: user.username,
        badge_count: user.badge_count,
        total_points: user.total_points
      }))
    });
  } catch (error) {
    console.error('Error getting badge leaderboard:', error);
    res.status(500).json({ 
      error: 'Failed to get badge leaderboard',
      details: error.message 
    });
  }
});

// ==================== AIRDROP TRACKING ENDPOINTS ====================

/**
 * GET /api/community/airdrop/eligibility/:userId
 * Get user's airdrop eligibility status
 * Enhanced system exceeding Drift's simple approach
 */
router.get('/airdrop/eligibility/:userId', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const { data: airdropData, error } = await dbService.client
      .from('airdrop_tracking')
      .select('*')
      .eq('user_id', userId)
      .single();

    if (error && error.code !== 'PGRST116') throw error;
    
    if (!airdropData) {
      return res.status(404).json({ error: 'Airdrop data not found' });
    }
    
    res.json({
      success: true,
      airdrop: {
        eligibility_score: airdropData.eligibility_score,
        points_contribution: airdropData.points_contribution,
        badge_contribution: airdropData.badge_contribution,
        community_engagement: airdropData.community_engagement,
        total_score: airdropData.total_score,
        airdrop_tier: airdropData.airdrop_tier,
        last_updated: airdropData.last_updated
      }
    });
  } catch (error) {
    console.error('Error getting airdrop eligibility:', error);
    res.status(500).json({ 
      error: 'Failed to get airdrop eligibility',
      details: error.message 
    });
  }
});

// ==================== COMMUNITY CHALLENGES ENDPOINTS ====================

/**
 * GET /api/community/challenges
 * Get active community challenges
 * Enhanced community engagement exceeding Drift's simple approach
 */
router.get('/challenges', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { data: challenges, error } = await dbService.client
      .from('community_challenges')
      .select('*')
      .eq('is_active', true)
      .gte('end_date', new Date().toISOString())
      .order('start_date', { ascending: true });

    if (error) throw error;
    
    res.json({
      success: true,
      challenges: challenges || []
    });
  } catch (error) {
    console.error('Error getting challenges:', error);
    res.status(500).json({ 
      error: 'Failed to get challenges',
      details: error.message 
    });
  }
});

/**
 * POST /api/community/challenges/participate
 * Participate in a community challenge
 */
router.post('/challenges/participate', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { user_id, challenge_id } = req.body;

    if (!user_id || !challenge_id) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    // Check if user is already participating
    const { data: existingParticipation, error: checkError } = await dbService.client
      .from('user_challenge_participation')
      .select('id')
      .eq('user_id', user_id)
      .eq('challenge_id', challenge_id)
      .single();

    if (checkError && checkError.code !== 'PGRST116') throw checkError;
    
    if (existingParticipation) {
      return res.status(400).json({ error: 'User already participating in this challenge' });
    }

    // Add participation
    const { data: participation, error } = await dbService.client
      .from('user_challenge_participation')
      .insert({
        user_id,
        challenge_id,
        progress: 0,
        completed: false,
        points_earned: 0,
        joined_at: new Date().toISOString()
      })
      .select()
      .single();

    if (error) throw error;
    
    res.status(201).json({
      success: true,
      message: 'Successfully joined challenge',
      participation: {
        id: participation.id,
        challenge_id: participation.challenge_id,
        progress: participation.progress,
        joined_at: participation.joined_at
      }
    });
  } catch (error) {
    console.error('Error participating in challenge:', error);
    res.status(500).json({ 
      error: 'Failed to participate in challenge',
      details: error.message 
    });
  }
});

// ==================== ANALYTICS ENDPOINTS ====================

/**
 * GET /api/community/analytics/overview
 * Get community analytics overview
 * Enhanced metrics exceeding Drift's simple approach
 */
router.get('/analytics/overview', rateLimiter, async (req: Request, res: Response) => {
  try {
    // Get total users
    const { count: totalUsers, error: usersError } = await dbService.client
      .from('users')
      .select('*', { count: 'exact', head: true })
      .eq('is_active', true);

    if (usersError) throw usersError;

    // Get total points awarded
    const { data: pointsData, error: pointsError } = await dbService.client
      .from('points_transactions')
      .select('points')
      .eq('transaction_type', 'earned');

    if (pointsError) throw pointsError;

    const totalPointsAwarded = pointsData?.reduce((sum, transaction) => sum + transaction.points, 0) || 0;

    // Get total badges earned
    const { count: totalBadges, error: badgesError } = await dbService.client
      .from('user_badges')
      .select('*', { count: 'exact', head: true })
      .eq('is_active', true);

    if (badgesError) throw badgesError;

    // Get total redemptions
    const { count: totalRedemptions, error: redemptionsError } = await dbService.client
      .from('user_redemptions')
      .select('*', { count: 'exact', head: true })
      .eq('status', 'active');

    if (redemptionsError) throw redemptionsError;

    res.json({
      success: true,
      analytics: {
        total_users: totalUsers || 0,
        total_points_awarded: totalPointsAwarded,
        total_badges_earned: totalBadges || 0,
        total_redemptions: totalRedemptions || 0,
        average_points_per_user: totalUsers ? Math.round(totalPointsAwarded / totalUsers) : 0,
        last_updated: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('Error getting analytics:', error);
    res.status(500).json({ 
      error: 'Failed to get analytics',
      details: error.message 
    });
  }
});

export default router;
