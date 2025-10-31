// QuantDesk MIKEY AI Integration API Endpoints
// Enhanced implementation exceeding Drift's capabilities
// "More Open Than Drift" competitive positioning

import { Router, Request, Response } from 'express';
import { MIKEYAIIntegrationService } from '../services/mikeyAIIntegrationService';
import { CommunityPointsService } from '../services/communityPointsService';
import { DatabaseService } from '../services/supabaseDatabase';
import { rateLimiter } from '../middleware/rateLimiting';
import { authenticateWallet } from '../middleware/auth';

const router = Router();
const dbService = new DatabaseService();
const communityService = new CommunityPointsService(dbService);
const mikeyService = new MIKEYAIIntegrationService(dbService, communityService);

// ==================== MIKEY AI ACCESS MANAGEMENT ====================

/**
 * GET /api/mikey/access/:userId
 * Check if user has active MIKEY AI access
 * Enhanced system exceeding Drift's capabilities
 */
router.get('/access/:userId', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const session = await mikeyService.checkMIKEYAccess(userId);
    
    if (!session) {
      return res.json({
        success: true,
        has_access: false,
        message: 'No active MIKEY AI access'
      });
    }
    
    res.json({
      success: true,
      has_access: true,
      session: {
        id: session.id,
        tier: session.tier,
        started_at: session.started_at,
        expires_at: session.expires_at,
        usage_count: session.usage_count,
        max_usage: session.max_usage,
        usage_remaining: session.max_usage - session.usage_count
      }
    });
  } catch (error) {
    console.error('Error checking MIKEY access:', error);
    res.status(500).json({ 
      error: 'Failed to check MIKEY access',
      details: error.message 
    });
  }
});

/**
 * POST /api/mikey/access/grant
 * Grant MIKEY AI access based on user's points
 * Enhanced system exceeding Drift's capabilities
 */
router.post('/access/grant', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { user_id, tier } = req.body;

    if (!user_id || !tier) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    if (!['basic', 'pro', 'vip'].includes(tier)) {
      return res.status(400).json({ error: 'Invalid tier. Must be basic, pro, or vip' });
    }

    const session = await mikeyService.grantMIKEYAccess(user_id, tier);
    
    res.status(201).json({
      success: true,
      message: `MIKEY AI ${tier} access granted successfully`,
      session: {
        id: session.id,
        tier: session.tier,
        started_at: session.started_at,
        expires_at: session.expires_at,
        usage_count: session.usage_count,
        max_usage: session.max_usage
      }
    });
  } catch (error) {
    console.error('Error granting MIKEY access:', error);
    res.status(500).json({ 
      error: 'Failed to grant MIKEY access',
      details: error.message 
    });
  }
});

/**
 * GET /api/mikey/features/:userId
 * Get available MIKEY features for user's tier
 */
router.get('/features/:userId', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const features = await mikeyService.getAvailableFeatures(userId);
    
    res.json({
      success: true,
      features: features
    });
  } catch (error) {
    console.error('Error getting MIKEY features:', error);
    res.status(500).json({ 
      error: 'Failed to get MIKEY features',
      details: error.message 
    });
  }
});

/**
 * GET /api/mikey/stats/:userId
 * Get MIKEY AI usage statistics
 */
router.get('/stats/:userId', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const stats = await mikeyService.getMIKEYUsageStats(userId);
    
    if (!stats) {
      return res.status(404).json({ error: 'No MIKEY AI access found' });
    }
    
    res.json({
      success: true,
      stats: stats
    });
  } catch (error) {
    console.error('Error getting MIKEY stats:', error);
    res.status(500).json({ 
      error: 'Failed to get MIKEY stats',
      details: error.message 
    });
  }
});

// ==================== MIKEY AI FEATURE ENDPOINTS ====================

/**
 * POST /api/mikey/features/market-analysis
 * Perform market analysis using MIKEY AI
 * Enhanced AI analysis exceeding Drift's capabilities
 */
router.post('/features/market-analysis', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { user_id, market_symbol } = req.body;

    if (!user_id || !market_symbol) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const analysis = await mikeyService.performMarketAnalysis(user_id, market_symbol);
    
    res.json({
      success: true,
      message: 'Market analysis completed successfully',
      analysis: analysis.analysis,
      usage_id: analysis.usage_id,
      points_earned: analysis.points_earned
    });
  } catch (error) {
    console.error('Error performing market analysis:', error);
    res.status(500).json({ 
      error: 'Failed to perform market analysis',
      details: error.message 
    });
  }
});

/**
 * POST /api/mikey/features/trading-suggestions
 * Get AI-powered trading suggestions
 * Enhanced AI suggestions exceeding Drift's capabilities
 */
router.post('/features/trading-suggestions', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { user_id, portfolio } = req.body;

    if (!user_id) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const suggestions = await mikeyService.getTradingSuggestions(user_id, portfolio);
    
    res.json({
      success: true,
      message: 'Trading suggestions generated successfully',
      suggestions: suggestions.suggestions,
      usage_id: suggestions.usage_id,
      points_earned: suggestions.points_earned
    });
  } catch (error) {
    console.error('Error getting trading suggestions:', error);
    res.status(500).json({ 
      error: 'Failed to get trading suggestions',
      details: error.message 
    });
  }
});

/**
 * POST /api/mikey/features/portfolio-optimization
 * Optimize portfolio using MIKEY AI
 * Enhanced AI optimization exceeding Drift's capabilities
 */
router.post('/features/portfolio-optimization', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { user_id, current_portfolio } = req.body;

    if (!user_id || !current_portfolio) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const optimization = await mikeyService.optimizePortfolio(user_id, current_portfolio);
    
    res.json({
      success: true,
      message: 'Portfolio optimization completed successfully',
      optimization: optimization.optimization,
      usage_id: optimization.usage_id,
      points_earned: optimization.points_earned
    });
  } catch (error) {
    console.error('Error optimizing portfolio:', error);
    res.status(500).json({ 
      error: 'Failed to optimize portfolio',
      details: error.message 
    });
  }
});

/**
 * POST /api/mikey/features/sentiment-analysis
 * Analyze market sentiment using MIKEY AI
 * Enhanced AI sentiment analysis exceeding Drift's capabilities
 */
router.post('/features/sentiment-analysis', rateLimiter, authenticateWallet, async (req: Request, res: Response) => {
  try {
    const { user_id, query } = req.body;

    if (!user_id || !query) {
      return res.status(400).json({ error: 'Missing required fields' });
    }

    const sentiment = await mikeyService.analyzeSentiment(user_id, query);
    
    res.json({
      success: true,
      message: 'Sentiment analysis completed successfully',
      sentiment: sentiment.sentiment,
      usage_id: sentiment.usage_id,
      points_earned: sentiment.points_earned
    });
  } catch (error) {
    console.error('Error analyzing sentiment:', error);
    res.status(500).json({ 
      error: 'Failed to analyze sentiment',
      details: error.message 
    });
  }
});

// ==================== MIKEY AI ANALYTICS ENDPOINTS ====================

/**
 * GET /api/mikey/analytics/overview
 * Get MIKEY AI analytics overview
 * Enhanced metrics exceeding Drift's simple approach
 */
router.get('/analytics/overview', rateLimiter, async (req: Request, res: Response) => {
  try {
    // Get total MIKEY AI sessions
    const { count: totalSessions, error: sessionsError } = await dbService.client
      .from('mikey_sessions')
      .select('*', { count: 'exact', head: true });

    if (sessionsError) throw sessionsError;

    // Get total feature usage
    const { count: totalUsage, error: usageError } = await dbService.client
      .from('mikey_usage')
      .select('*', { count: 'exact', head: true });

    if (usageError) throw usageError;

    // Get total points earned from MIKEY AI
    const { data: pointsData, error: pointsError } = await dbService.client
      .from('mikey_usage')
      .select('points_earned');

    if (pointsError) throw pointsError;

    const totalPointsEarned = pointsData?.reduce((sum, usage) => sum + usage.points_earned, 0) || 0;

    // Get active sessions
    const { count: activeSessions, error: activeError } = await dbService.client
      .from('mikey_sessions')
      .select('*', { count: 'exact', head: true })
      .eq('is_active', true)
      .gt('expires_at', new Date().toISOString());

    if (activeError) throw activeError;

    // Get tier distribution
    const { data: tierData, error: tierError } = await dbService.client
      .from('mikey_sessions')
      .select('tier')
      .eq('is_active', true)
      .gt('expires_at', new Date().toISOString());

    if (tierError) throw tierError;

    const tierDistribution = tierData?.reduce((acc, session) => {
      acc[session.tier] = (acc[session.tier] || 0) + 1;
      return acc;
    }, {} as Record<string, number>) || {};

    res.json({
      success: true,
      analytics: {
        total_sessions: totalSessions || 0,
        active_sessions: activeSessions || 0,
        total_feature_usage: totalUsage || 0,
        total_points_earned: totalPointsEarned,
        tier_distribution: tierDistribution,
        average_usage_per_session: totalSessions ? Math.round((totalUsage || 0) / totalSessions) : 0,
        last_updated: new Date().toISOString()
      }
    });
  } catch (error) {
    console.error('Error getting MIKEY analytics:', error);
    res.status(500).json({ 
      error: 'Failed to get MIKEY analytics',
      details: error.message 
    });
  }
});

/**
 * GET /api/mikey/analytics/feature-usage
 * Get MIKEY AI feature usage statistics
 */
router.get('/analytics/feature-usage', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { data: featureUsage, error } = await dbService.client
      .from('mikey_usage')
      .select(`
        feature_used,
        points_earned,
        created_at
      `)
      .order('created_at', { ascending: false });

    if (error) throw error;

    // Group by feature
    const featureStats = featureUsage?.reduce((acc, usage) => {
      if (!acc[usage.feature_used]) {
        acc[usage.feature_used] = {
          usage_count: 0,
          total_points_earned: 0,
          last_used: usage.created_at
        };
      }
      acc[usage.feature_used].usage_count++;
      acc[usage.feature_used].total_points_earned += usage.points_earned;
      acc[usage.feature_used].last_used = usage.created_at;
      return acc;
    }, {} as Record<string, any>) || {};

    res.json({
      success: true,
      feature_usage: Object.entries(featureStats).map(([feature, stats]) => ({
        feature,
        usage_count: stats.usage_count,
        total_points_earned: stats.total_points_earned,
        last_used: stats.last_used
      })).sort((a, b) => b.usage_count - a.usage_count)
    });
  } catch (error) {
    console.error('Error getting feature usage:', error);
    res.status(500).json({ 
      error: 'Failed to get feature usage',
      details: error.message 
    });
  }
});

/**
 * GET /api/mikey/analytics/user/:userId
 * Get user's MIKEY AI usage analytics
 */
router.get('/analytics/user/:userId', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    
    const { data: userStats, error } = await dbService.client
      .from('mikey_user_stats')
      .select('*')
      .eq('user_id', userId)
      .single();

    if (error && error.code !== 'PGRST116') throw error;
    
    if (!userStats) {
      return res.status(404).json({ error: 'User MIKEY AI stats not found' });
    }
    
    res.json({
      success: true,
      user_stats: userStats
    });
  } catch (error) {
    console.error('Error getting user MIKEY stats:', error);
    res.status(500).json({ 
      error: 'Failed to get user MIKEY stats',
      details: error.message 
    });
  }
});

// ==================== MIKEY AI TIER MANAGEMENT ====================

/**
 * GET /api/mikey/tiers
 * Get MIKEY AI tier requirements and features
 */
router.get('/tiers', rateLimiter, async (req: Request, res: Response) => {
  try {
    const { data: tiers, error } = await dbService.client
      .rpc('get_mikey_tier_requirements');

    if (error) throw error;
    
    res.json({
      success: true,
      tiers: tiers || []
    });
  } catch (error) {
    console.error('Error getting MIKEY tiers:', error);
    res.status(500).json({ 
      error: 'Failed to get MIKEY tiers',
      details: error.message 
    });
  }
});

/**
 * GET /api/mikey/tiers/compare
 * Compare MIKEY AI tiers with Drift's capabilities
 */
router.get('/tiers/compare', rateLimiter, async (req: Request, res: Response) => {
  try {
    const comparison = {
      drift_protocol: {
        ai_features: 'None',
        community_rewards: 'Simple referral system (15% to referrers, 5% discount to users)',
        transparency: 'Smart contracts only',
        developer_experience: 'Good SDK and documentation',
        limitations: [
          'No AI integration',
          'No advanced community features',
          'Limited ecosystem transparency',
          'Basic referral system only'
        ]
      },
      quantdesk: {
        ai_features: 'Comprehensive MIKEY AI integration',
        community_rewards: 'Advanced points-based system with badges and gamification',
        transparency: 'Complete multi-service architecture',
        developer_experience: 'Enhanced SDK with AI integration examples',
        advantages: [
          'AI-powered trading assistance',
          'Points-based community engagement',
          'Complete ecosystem transparency',
          'Advanced gamification system',
          'MIKEY AI access tiers',
          'Comprehensive analytics'
        ]
      },
      competitive_positioning: {
        message: 'More Open Than Drift',
        key_differentiators: [
          'Complete ecosystem vs smart contracts only',
          'AI integration vs no AI features',
          'Advanced community system vs simple referrals',
          'Enhanced developer experience with AI examples'
        ]
      }
    };
    
    res.json({
      success: true,
      comparison: comparison
    });
  } catch (error) {
    console.error('Error getting tier comparison:', error);
    res.status(500).json({ 
      error: 'Failed to get tier comparison',
      details: error.message 
    });
  }
});

export default router;
