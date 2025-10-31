// QuantDesk Referral System API Endpoints
// Enhanced implementation exceeding Drift's capabilities
// "More Open Than Drift" competitive positioning with 20% referral rewards

import { Router, Request, Response } from 'express';
import { CommunityPointsService } from '../services/communityPointsService';
import { DatabaseService } from '../services/supabaseDatabase';
import { rateLimiter } from '../middleware/rateLimiting';
import { authenticateWallet } from '../middleware/auth';

const router = Router();
const dbService = new DatabaseService();
const communityService = new CommunityPointsService(dbService);

// ==================== REFERRAL CODE MANAGEMENT ====================

/**
 * POST /api/referral/generate
 * Generate referral code for user
 * Enhanced system exceeding Drift's simple referral approach
 */
router.post('/generate', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { user_id, platform, platform_id, expires_at, max_usage } = req.body;

    if (!user_id || !platform) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    if (!['wallet', 'telegram', 'discord', 'twitter'].includes(platform)) {
      return res.status(400).json({ error: 'Invalid platform. Must be wallet, telegram, discord, or twitter' });
    }

    const referralCode = await communityService.generateReferralCode(
      user_id,
      platform,
      platform_id,
      expires_at ? new Date(expires_at) : undefined,
      max_usage
    );
    
    res.status(201).json({
      success: true,
      message: 'Referral code generated successfully',
      referral_code: {
        id: referralCode.id,
        code: referralCode.code,
        platform: referralCode.platform,
        platform_id: referralCode.platform_id,
        expires_at: referralCode.expires_at,
        max_usage: referralCode.max_usage,
        usage_count: referralCode.usage_count
      }
    });
  } catch (error) {
    console.error('Error generating referral code:', error);
    res.status(500).json({ 
      error: 'Failed to generate referral code',
      details: error.message 
    });
  }
});

/**
 * GET /api/referral/codes/:userId
 * Get user's referral codes
 */
router.get('/codes/:userId', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const referralCodes = await communityService.getUserReferralCodes(userId);
    
    res.json({
      success: true,
      referral_codes: referralCodes
    });
  } catch (error) {
    console.error('Error getting referral codes:', error);
    res.status(500).json({ 
      error: 'Failed to get referral codes',
      details: error.message 
    });
  }
});

/**
 * POST /api/referral/validate
 * Validate referral code
 */
router.post('/validate', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { code } = req.body;

    if (!code) {
      return res.status(400).json({ error: 'Referral code is required' });
    }

    const validation = await communityService.validateReferralCode(code);
    
    if (!validation || !validation.is_valid) {
      return res.status(400).json({ 
        error: 'Invalid or expired referral code',
        is_valid: false
      });
    }
    
    res.json({
      success: true,
      is_valid: true,
      validation: {
        user_id: validation.user_id,
        platform: validation.platform,
        platform_id: validation.platform_id,
        usage_count: validation.usage_count,
        max_usage: validation.max_usage,
        expires_at: validation.expires_at
      }
    });
  } catch (error) {
    console.error('Error validating referral code:', error);
    res.status(500).json({ 
      error: 'Failed to validate referral code',
      details: error.message 
    });
  }
});

// ==================== REFERRAL PROCESSING ====================

/**
 * POST /api/referral/process
 * Process referral when user registers
 * Enhanced with 20% referral rewards exceeding Drift's 15%
 */
router.post('/process', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { referred_user_id, referral_code, platform_context } = req.body;

    if (!referred_user_id || !referral_code) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const relationship = await communityService.processReferral(
      referred_user_id,
      referral_code,
      platform_context
    );
    
    res.status(201).json({
      success: true,
      message: 'Referral processed successfully',
      relationship: {
        id: relationship?.id,
        referrer_id: relationship?.referrer_id,
        referred_id: relationship?.referred_id,
        platform: relationship?.platform,
        created_at: relationship?.created_at
      }
    });
  } catch (error) {
    console.error('Error processing referral:', error);
    res.status(500).json({ 
      error: 'Failed to process referral',
      details: error.message 
    });
  }
});

/**
 * GET /api/referral/relationship/:userId
 * Get referral relationship for user
 */
router.get('/relationship/:userId', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const relationship = await communityService.getReferralRelationship(userId);
    
    if (!relationship) {
      return res.status(404).json({ error: 'No referral relationship found' });
    }
    
    res.json({
      success: true,
      relationship: relationship
    });
  } catch (error) {
    console.error('Error getting referral relationship:', error);
    res.status(500).json({ 
      error: 'Failed to get referral relationship',
      details: error.message 
    });
  }
});

// ==================== REFERRAL ANALYTICS ====================

/**
 * GET /api/referral/stats/:userId
 * Get referral statistics for user
 */
router.get('/stats/:userId', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const stats = await communityService.getReferralStats(userId);
    
    if (!stats) {
      return res.status(404).json({ error: 'Referral stats not found' });
    }
    
    res.json({
      success: true,
      stats: stats
    });
  } catch (error) {
    console.error('Error getting referral stats:', error);
    res.status(500).json({ 
      error: 'Failed to get referral stats',
      details: error.message 
    });
  }
});

/**
 * GET /api/referral/leaderboard
 * Get referral leaderboard
 */
router.get('/leaderboard', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { limit = 10 } = req.query;
    
    const leaderboard = await communityService.getReferralLeaderboard(Number(limit));
    
    res.json({
      success: true,
      leaderboard: leaderboard
    });
  } catch (error) {
    console.error('Error getting referral leaderboard:', error);
    res.status(500).json({ 
      error: 'Failed to get referral leaderboard',
      details: error.message 
    });
  }
});

/**
 * GET /api/referral/referred/:referrerId
 * Get referred users for a referrer
 */
router.get('/referred/:referrerId', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { referrerId } = req.params;
    
    const referredUsers = await communityService.getReferredUsers(referrerId);
    
    res.json({
      success: true,
      referred_users: referredUsers
    });
  } catch (error) {
    console.error('Error getting referred users:', error);
    res.status(500).json({ 
      error: 'Failed to get referred users',
      details: error.message 
    });
  }
});

// ==================== REFERRAL ANALYTICS OVERVIEW ====================

/**
 * GET /api/referral/analytics/overview
 * Get referral analytics overview
 * Enhanced metrics exceeding Drift's simple approach
 */
router.get('/analytics/overview', rateLimiter, async (req: Request, res: Response) => {
  try {
    // Get total referral relationships
    const { count: totalReferrals, error: referralsError } = await dbService.client
      .from('referral_relationships')
      .select('*', { count: 'exact', head: true })
      .eq('is_active', true);

    if (referralsError) throw referralsError;

    // Get total referral rewards
    const { data: rewardsData, error: rewardsError } = await dbService.client
      .from('referral_rewards')
      .select('points_earned');

    if (rewardsError) throw rewardsError;

    const totalReferralRewards = rewardsData?.reduce((sum, reward) => sum + reward.points_earned, 0) || 0;

    // Get platform breakdown
    const { data: platformData, error: platformError } = await dbService.client
      .from('referral_relationships')
      .select('platform')
      .eq('is_active', true);

    if (platformError) throw platformError;

    const platformBreakdown = platformData?.reduce((acc, rel) => {
      acc[rel.platform] = (acc[rel.platform] || 0) + 1;
      return acc;
    }, {} as Record<string, number>) || {};

    // Get active referral codes
    const { count: activeCodes, error: codesError } = await dbService.client
      .from('referral_codes')
      .select('*', { count: 'exact', head: true })
      .eq('is_active', true);

    if (codesError) throw codesError;

    res.json({
      success: true,
      analytics: {
        total_referrals: totalReferrals || 0,
        total_referral_rewards: totalReferralRewards,
        active_referral_codes: activeCodes || 0,
        platform_breakdown: platformBreakdown,
        average_referral_reward: totalReferrals ? Math.round(totalReferralRewards / totalReferrals) : 0,
        last_updated: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('Error getting referral analytics:', error);
    res.status(500).json({ 
      error: 'Failed to get referral analytics',
      details: error.message 
    });
  }
});

// ==================== REFERRAL COMPARISON WITH DRIFT ====================

/**
 * GET /api/referral/compare-drift
 * Compare QuantDesk referral system with Drift Protocol
 */
router.get('/compare-drift', rateLimiter, async (req: Request, res: Response) => {
  try {
    const comparison = {
      drift_protocol: {
        referral_reward: '15% of taker fees to referrers',
        user_discount: '5% fee discount for referred users',
        platform_support: 'Web only',
        tracking: 'Browser session-based',
        limitations: [
          'Limited to web platform only',
          'Simple fee sharing model',
          'No multi-platform support',
          'No advanced analytics',
          'No gamification elements'
        ]
      },
      quantdesk: {
        referral_reward: '20% of earned points to referrers (exceeds Drift)',
        user_bonus: '100 points for referred users + ongoing 20% rewards',
        platform_support: 'Wallet, Telegram, Discord, Twitter',
        tracking: 'Multi-platform with comprehensive analytics',
        advantages: [
          'Multi-platform referral support',
          'Higher referral rewards (20% vs 15%)',
          'Comprehensive analytics and tracking',
          'Gamification with badges and leaderboards',
          'Advanced referral code management',
          'Platform-specific referral codes',
          'Detailed referral statistics',
          'Referral leaderboards and competitions'
        ]
      },
      competitive_positioning: {
        message: 'More Rewarding Than Drift',
        key_differentiators: [
          '20% referral rewards vs Drift\'s 15%',
          'Multi-platform support vs web-only',
          'Advanced analytics vs basic tracking',
          'Gamification elements vs simple rewards',
          'Comprehensive referral management vs basic system'
        ]
      }
    };
    
    res.json({
      success: true,
      comparison: comparison
    });
  } catch (error) {
    console.error('Error getting referral comparison:', error);
    res.status(500).json({ 
      error: 'Failed to get referral comparison',
      details: error.message 
    });
  }
});

// ==================== REFERRAL SHARING INTEGRATION ====================

/**
 * POST /api/referral/share/telegram
 * Generate Telegram sharing link
 */
router.post('/share/telegram', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { user_id, referral_code } = req.body;

    if (!user_id || !referral_code) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const telegramShareLink = `https://t.me/share/url?url=https://quantdesk.io/ref/${referral_code}&text=Join%20QuantDesk%20-%20The%20Most%20Open%20Perpetual%20DEX%20on%20Solana!%20Use%20my%20referral%20code%20${referral_code}%20to%20get%20bonus%20points!`;
    
    res.json({
      success: true,
      share_link: telegramShareLink,
      platform: 'telegram',
      referral_code: referral_code
    });
  } catch (error) {
    console.error('Error generating Telegram share link:', error);
    res.status(500).json({ 
      error: 'Failed to generate Telegram share link',
      details: error.message 
    });
  }
});

/**
 * POST /api/referral/share/twitter
 * Generate Twitter sharing link
 */
router.post('/share/twitter', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { user_id, referral_code } = req.body;

    if (!user_id || !referral_code) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const twitterShareLink = `https://twitter.com/intent/tweet?text=Join%20QuantDesk%20-%20The%20Most%20Open%20Perpetual%20DEX%20on%20Solana!%20Use%20my%20referral%20code%20${referral_code}%20to%20get%20bonus%20points!%20More%20Open%20Than%20Drift!&url=https://quantdesk.io/ref/${referral_code}`;
    
    res.json({
      success: true,
      share_link: twitterShareLink,
      platform: 'twitter',
      referral_code: referral_code
    });
  } catch (error) {
    console.error('Error generating Twitter share link:', error);
    res.status(500).json({ 
      error: 'Failed to generate Twitter share link',
      details: error.message 
    });
  }
});

/**
 * POST /api/referral/share/discord
 * Generate Discord sharing message
 */
router.post('/share/discord', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { user_id, referral_code } = req.body;

    if (!user_id || !referral_code) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const discordMessage = `🚀 **Join QuantDesk - The Most Open Perpetual DEX on Solana!**

🎯 **More Open Than Drift Protocol!**
- Complete ecosystem transparency vs smart contracts only
- AI-powered trading assistance vs no AI features
- Advanced community system vs simple referrals
- 20% referral rewards vs Drift's 15%

💰 **Use my referral code: \`${referral_code}\`**
- Get 100 bonus points when you join
- Earn 20% of my points as referral rewards
- Access exclusive community features

🔗 **Join now:** https://quantdesk.io/ref/${referral_code}`;
    
    res.json({
      success: true,
      share_message: discordMessage,
      platform: 'discord',
      referral_code: referral_code
    });
  } catch (error) {
    console.error('Error generating Discord share message:', error);
    res.status(500).json({ 
      error: 'Failed to generate Discord share message',
      details: error.message 
    });
  }
});

export default router;
