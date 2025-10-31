// QuantDesk Community Points Service
// Enhanced implementation based on Drift Protocol analysis
// "More Open Than Drift" competitive positioning

import { DatabaseService } from './supabaseDatabase';
import { v4 as uuidv4 } from 'uuid';

export interface User {
  id: string;
  wallet_address: string;
  username?: string;
  email?: string;
  total_points: number;
  level: string;
  created_at: Date;
  updated_at: Date;
  is_active: boolean;
}

export interface PointsTransaction {
  id: string;
  user_id: string;
  points: number;
  transaction_type: 'earned' | 'redeemed' | 'bonus' | 'penalty';
  source: string;
  description?: string;
  created_at: Date;
  metadata?: any;
  validated: boolean;
}

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon_url?: string;
  points_required: number;
  category: string;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
  is_active: boolean;
}

export interface UserBadge {
  id: string;
  user_id: string;
  badge_id: string;
  earned_at: Date;
  is_active: boolean;
}

export interface RedemptionOption {
  id: string;
  name: string;
  description: string;
  points_cost: number;
  redemption_type: 'mikey_access' | 'pro_membership' | 'exclusive_feature' | 'priority_support' | 'airdrop_eligibility' | 'custom_badge' | 'vip_access';
  is_active: boolean;
  cooldown_hours: number;
  expires_at?: Date;
  metadata?: any;
}

export interface UserRedemption {
  id: string;
  user_id: string;
  redemption_option_id: string;
  points_spent: number;
  redeemed_at: Date;
  status: 'active' | 'expired' | 'cancelled' | 'pending';
  expires_at?: Date;
  activated_at?: Date;
  metadata?: any;
}

export interface AirdropTracking {
  id: string;
  user_id: string;
  eligibility_score: number;
  points_contribution: number;
  badge_contribution: number;
  community_engagement: number;
  total_score: number;
  airdrop_tier: 'bronze' | 'silver' | 'gold' | 'platinum';
  last_updated: Date;
}

export interface ReferralCode {
  id: string;
  user_id: string;
  code: string;
  platform: 'wallet' | 'telegram' | 'discord' | 'twitter';
  platform_id?: string;
  is_active: boolean;
  created_at: Date;
  expires_at?: Date;
  usage_count: number;
  max_usage?: number;
  metadata?: any;
}

export interface ReferralRelationship {
  id: string;
  referrer_id: string;
  referred_id: string;
  referral_code_id: string;
  platform: 'wallet' | 'telegram' | 'discord' | 'twitter';
  platform_context?: any;
  created_at: Date;
  is_active: boolean;
}

export interface ReferralReward {
  id: string;
  referral_relationship_id: string;
  referrer_id: string;
  referred_id: string;
  points_earned: number;
  points_percentage: number;
  source_transaction_id?: string;
  reward_type: string;
  created_at: Date;
  metadata?: any;
}

export interface TradingMetrics {
  id: string;
  user_id: string;
  total_trading_volume: number;
  maker_volume: number;
  taker_volume: number;
  total_deposits: number;
  total_withdrawals: number;
  net_deposits: number;
  staking_amount: number;
  insurance_fund_stake: number;
  total_trades: number;
  active_trading_days: number;
  last_trade_at?: Date;
  created_at: Date;
  updated_at: Date;
}

export interface TradingActivityLog {
  id: string;
  user_id: string;
  activity_type: 'trade' | 'deposit' | 'withdrawal' | 'staking' | 'insurance_fund';
  amount: number;
  market_symbol?: string;
  side?: 'long' | 'short' | 'buy' | 'sell' | 'maker' | 'taker';
  leverage?: number;
  points_earned: number;
  created_at: Date;
  metadata?: any;
}

export interface CommunityChallenge {
  id: string;
  name: string;
  description: string;
  challenge_type: 'points' | 'badges' | 'engagement' | 'referral';
  target_value: number;
  points_reward: number;
  badge_reward?: string;
  start_date: Date;
  end_date: Date;
  is_active: boolean;
  metadata?: any;
}

export interface LeaderboardEntry {
  id: string;
  leaderboard_id: string;
  user_id: string;
  rank: number;
  score: number;
  period_start: Date;
  period_end: Date;
}

/**
 * Community Points Service - Enhanced implementation exceeding Drift's capabilities
 * 
 * Key Features vs Drift:
 * - Advanced points system vs simple referral system
 * - AI integration tiers vs no AI features
 * - Comprehensive gamification vs basic rewards
 * - Multi-service architecture vs smart contracts only
 */
export class CommunityPointsService {
  private db: DatabaseService;

  constructor(databaseService: DatabaseService) {
    this.db = databaseService;
  }

  // ==================== USER MANAGEMENT ====================

  /**
   * Register a new user in the community points system
   * Enhanced with early adopter bonuses (2x points for first 100 users)
   */
  async registerUser(walletAddress: string, username?: string, email?: string): Promise<User> {
    try {
      // Check if user already exists
      const existingUser = await this.getUserByWallet(walletAddress);
      if (existingUser) {
        return existingUser;
      }

      // Check if user is in first 100 (early adopter bonus)
      const userCount = await this.getTotalUserCount();
      const isEarlyAdopter = userCount < 100;

      const user: User = {
        id: uuidv4(),
        wallet_address: walletAddress,
        username,
        email,
        total_points: 0,
        level: 'newcomer',
        created_at: new Date(),
        updated_at: new Date(),
        is_active: true
      };

      // Insert user
      const { data, error } = await this.db.client
        .from('users')
        .insert(user)
        .select()
        .single();

      if (error) throw error;

      // Award early testing points with bonus
      const basePoints = 100;
      const bonusMultiplier = isEarlyAdopter ? 2 : 1;
      const totalPoints = basePoints * bonusMultiplier;

      await this.awardPoints(data.id, totalPoints, 'earned', 'early_testing', 
        `Early testing participation${isEarlyAdopter ? ' (Early Adopter Bonus!)' : ''}`);

      return data;
    } catch (error) {
      console.error('Error registering user:', error);
      throw new Error(`Failed to register user: ${error.message}`);
    }
  }

  /**
   * Get user by wallet address
   */
  async getUserByWallet(walletAddress: string): Promise<User | null> {
    try {
      const { data, error } = await this.db.client
        .from('users')
        .select('*')
        .eq('wallet_address', walletAddress)
        .eq('is_active', true)
        .single();

      if (error && error.code !== 'PGRST116') throw error;
      return data || null;
    } catch (error) {
      console.error('Error getting user by wallet:', error);
      throw new Error(`Failed to get user: ${error.message}`);
    }
  }

  /**
   * Get total user count for early adopter tracking
   */
  async getTotalUserCount(): Promise<number> {
    try {
      const { count, error } = await this.db.client
        .from('users')
        .select('*', { count: 'exact', head: true })
        .eq('is_active', true);

      if (error) throw error;
      return count || 0;
    } catch (error) {
      console.error('Error getting user count:', error);
      return 0;
    }
  }

  // ==================== POINTS MANAGEMENT ====================

  /**
   * Award points to a user with validation and anti-gaming measures
   * Enhanced with rate limiting and fraud detection
   */
  async awardPoints(
    userId: string, 
    points: number, 
    transactionType: 'earned' | 'redeemed' | 'bonus' | 'penalty',
    source: string,
    description?: string,
    metadata?: any
  ): Promise<PointsTransaction> {
    try {
      // Anti-gaming: Rate limiting check
      const recentTransactions = await this.getRecentTransactions(userId, 1); // Last hour
      if (recentTransactions.length > 10) {
        throw new Error('Rate limit exceeded: Too many transactions in the last hour');
      }

      // Anti-gaming: Validate points amount
      if (points <= 0 || points > 10000) {
        throw new Error('Invalid points amount: Must be between 1 and 10,000');
      }

      // Anti-gaming: Check for duplicate transactions
      const duplicateCheck = await this.checkDuplicateTransaction(userId, source, points);
      if (duplicateCheck) {
        throw new Error('Duplicate transaction detected');
      }

      const transaction: PointsTransaction = {
        id: uuidv4(),
        user_id: userId,
        points,
        transaction_type: transactionType,
        source,
        description,
        created_at: new Date(),
        metadata,
        validated: false
      };

      // Insert transaction
      const { data, error } = await this.db.client
        .from('points_transactions')
        .insert(transaction)
        .select()
        .single();

      if (error) throw error;

      // Update user total points (triggered by database trigger)
      await this.updateUserLevel(userId);

      // Check for badge eligibility
      await this.checkBadgeEligibility(userId);

      // Update airdrop tracking
      await this.updateAirdropEligibility(userId);

      // Process referral rewards if this is an earned transaction
      if (transactionType === 'earned' && points > 0) {
        await this.processReferralReward(userId, points);
      }

      return data;
    } catch (error) {
      console.error('Error awarding points:', error);
      throw new Error(`Failed to award points: ${error.message}`);
    }
  }

  /**
   * Get recent transactions for rate limiting
   */
  async getRecentTransactions(userId: string, hours: number = 1): Promise<PointsTransaction[]> {
    try {
      const cutoffTime = new Date(Date.now() - hours * 60 * 60 * 1000);
      
      const { data, error } = await this.db.client
        .from('points_transactions')
        .select('*')
        .eq('user_id', userId)
        .gte('created_at', cutoffTime.toISOString())
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error getting recent transactions:', error);
      return [];
    }
  }

  /**
   * Check for duplicate transactions
   */
  async checkDuplicateTransaction(userId: string, source: string, points: number): Promise<boolean> {
    try {
      const recentTime = new Date(Date.now() - 5 * 60 * 1000); // Last 5 minutes
      
      const { data, error } = await this.db.client
        .from('points_transactions')
        .select('id')
        .eq('user_id', userId)
        .eq('source', source)
        .eq('points', points)
        .gte('created_at', recentTime.toISOString())
        .limit(1);

      if (error) throw error;
      return (data && data.length > 0);
    } catch (error) {
      console.error('Error checking duplicate transaction:', error);
      return false;
    }
  }

  /**
   * Update user level based on total points
   */
  async updateUserLevel(userId: string): Promise<void> {
    try {
      const user = await this.getUserById(userId);
      if (!user) return;

      let newLevel = 'newcomer';
      if (user.total_points >= 2000) newLevel = 'legend';
      else if (user.total_points >= 1000) newLevel = 'expert';
      else if (user.total_points >= 500) newLevel = 'advanced';
      else if (user.total_points >= 100) newLevel = 'intermediate';

      if (newLevel !== user.level) {
        await this.db.client
          .from('users')
          .update({ level: newLevel, updated_at: new Date().toISOString() })
          .eq('id', userId);
      }
    } catch (error) {
      console.error('Error updating user level:', error);
    }
  }

  /**
   * Get user by ID
   */
  async getUserById(userId: string): Promise<User | null> {
    try {
      const { data, error } = await this.db.client
        .from('users')
        .select('*')
        .eq('id', userId)
        .eq('is_active', true)
        .single();

      if (error && error.code !== 'PGRST116') throw error;
      return data || null;
    } catch (error) {
      console.error('Error getting user by ID:', error);
      return null;
    }
  }

  // ==================== BADGE SYSTEM ====================

  /**
   * Check if user is eligible for new badges
   * Enhanced gamification system exceeding Drift's simple approach
   */
  async checkBadgeEligibility(userId: string): Promise<void> {
    try {
      const user = await this.getUserById(userId);
      if (!user) return;

      const userBadges = await this.getUserBadges(userId);
      const existingBadgeIds = userBadges.map(b => b.badge_id);

      // Get all available badges
      const { data: badges, error } = await this.db.client
        .from('badges')
        .select('*')
        .eq('is_active', true);

      if (error) throw error;

      // Check each badge for eligibility
      for (const badge of badges || []) {
        if (existingBadgeIds.includes(badge.id)) continue;

        let isEligible = false;

        switch (badge.name) {
          case 'Early Adopter':
            isEligible = user.total_points >= 100 && user.created_at > new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
            break;
          case 'Beta Tester':
            const testTransactions = await this.getTransactionsBySource(userId, 'beta_testing');
            isEligible = testTransactions.length >= 10;
            break;
          case 'Bug Reporter':
            const bugTransactions = await this.getTransactionsBySource(userId, 'bug_report');
            isEligible = bugTransactions.length >= 3;
            break;
          case 'Feature Suggestion':
            const featureTransactions = await this.getTransactionsBySource(userId, 'feature_suggestion');
            isEligible = featureTransactions.length >= 2;
            break;
          case 'Power User':
            isEligible = user.total_points >= 1000;
            break;
          case 'Community Leader':
            const rank = await this.getUserRank(userId);
            isEligible = rank <= 10;
            break;
          case 'QuantDesk Ambassador':
            const referralTransactions = await this.getTransactionsBySource(userId, 'referral');
            isEligible = referralTransactions.length >= 10;
            break;
        }

        if (isEligible) {
          await this.awardBadge(userId, badge.id);
        }
      }
    } catch (error) {
      console.error('Error checking badge eligibility:', error);
    }
  }

  /**
   * Award badge to user
   */
  async awardBadge(userId: string, badgeId: string): Promise<UserBadge> {
    try {
      const userBadge: UserBadge = {
        id: uuidv4(),
        user_id: userId,
        badge_id: badgeId,
        earned_at: new Date(),
        is_active: true
      };

      const { data, error } = await this.db.client
        .from('user_badges')
        .insert(userBadge)
        .select()
        .single();

      if (error) throw error;

      // Award bonus points for badge earning
      await this.awardPoints(userId, 50, 'bonus', 'badge_earned', 'Badge earned bonus');

      return data;
    } catch (error) {
      console.error('Error awarding badge:', error);
      throw new Error(`Failed to award badge: ${error.message}`);
    }
  }

  /**
   * Get user badges
   */
  async getUserBadges(userId: string): Promise<UserBadge[]> {
    try {
      const { data, error } = await this.db.client
        .from('user_badges')
        .select(`
          *,
          badges (*)
        `)
        .eq('user_id', userId)
        .eq('is_active', true)
        .order('earned_at', { ascending: false });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error getting user badges:', error);
      return [];
    }
  }

  // ==================== REDEMPTION SYSTEM ====================

  /**
   * Redeem points for rewards
   * Enhanced with MIKEY AI access tiers exceeding Drift's capabilities
   */
  async redeemPoints(userId: string, redemptionOptionId: string): Promise<UserRedemption> {
    try {
      const user = await this.getUserById(userId);
      if (!user) throw new Error('User not found');

      // Get redemption option
      const { data: option, error: optionError } = await this.db.client
        .from('redemption_options')
        .select('*')
        .eq('id', redemptionOptionId)
        .eq('is_active', true)
        .single();

      if (optionError) throw optionError;
      if (!option) throw new Error('Redemption option not found');

      // Check if user has enough points
      if (user.total_points < option.points_cost) {
        throw new Error('Insufficient points');
      }

      // Check cooldown period
      const recentRedemptions = await this.getRecentRedemptions(userId, option.cooldown_hours);
      if (recentRedemptions.length > 0) {
        throw new Error(`Cooldown period active: ${option.cooldown_hours} hours`);
      }

      // Check expiration
      if (option.expires_at && new Date(option.expires_at) < new Date()) {
        throw new Error('Redemption option has expired');
      }

      // Create redemption
      const redemption: UserRedemption = {
        id: uuidv4(),
        user_id: userId,
        redemption_option_id: redemptionOptionId,
        points_spent: option.points_cost,
        redeemed_at: new Date(),
        status: 'active',
        expires_at: this.calculateExpiration(option.redemption_type),
        activated_at: new Date(),
        metadata: option.metadata
      };

      const { data, error } = await this.db.client
        .from('user_redemptions')
        .insert(redemption)
        .select()
        .single();

      if (error) throw error;

      // Deduct points
      await this.awardPoints(userId, -option.points_cost, 'redeemed', 'redemption', 
        `Redeemed: ${option.name}`);

      return data;
    } catch (error) {
      console.error('Error redeeming points:', error);
      throw new Error(`Failed to redeem points: ${error.message}`);
    }
  }

  /**
   * Get recent redemptions for cooldown checking
   */
  async getRecentRedemptions(userId: string, hours: number): Promise<UserRedemption[]> {
    try {
      const cutoffTime = new Date(Date.now() - hours * 60 * 60 * 1000);
      
      const { data, error } = await this.db.client
        .from('user_redemptions')
        .select('*')
        .eq('user_id', userId)
        .gte('redeemed_at', cutoffTime.toISOString())
        .eq('status', 'active');

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error getting recent redemptions:', error);
      return [];
    }
  }

  /**
   * Calculate expiration date based on redemption type
   */
  private calculateExpiration(redemptionType: string): Date {
    const now = new Date();
    switch (redemptionType) {
      case 'mikey_access':
        return new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000); // 30 days
      case 'pro_membership':
        return new Date(now.getTime() + 90 * 24 * 60 * 60 * 1000); // 90 days
      case 'exclusive_feature':
        return new Date(now.getTime() + 60 * 24 * 60 * 60 * 1000); // 60 days
      case 'priority_support':
        return new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000); // 30 days
      case 'vip_access':
        return new Date(now.getTime() + 30 * 24 * 60 * 60 * 1000); // 30 days
      default:
        return new Date(now.getTime() + 365 * 24 * 60 * 60 * 1000); // 1 year
    }
  }

  // ==================== AIRDROP TRACKING ====================

  /**
   * Update airdrop eligibility based on comprehensive scoring
   * Enhanced system exceeding Drift's simple approach
   */
  async updateAirdropEligibility(userId: string): Promise<void> {
    try {
      const user = await this.getUserById(userId);
      if (!user) return;

      // Calculate points contribution (max 1000 points)
      const pointsScore = Math.min(user.total_points, 1000);

      // Calculate badge contribution (max 500 points)
      const userBadges = await this.getUserBadges(userId);
      const badgeScore = Math.min(userBadges.length * 50, 500);

      // Calculate engagement score (max 500 points)
      const daysSinceJoined = Math.floor((Date.now() - new Date(user.created_at).getTime()) / (24 * 60 * 60 * 1000));
      const engagementScore = Math.min((daysSinceJoined * 2) + (user.total_points / 10), 500);

      // Calculate total score
      const totalScore = pointsScore + badgeScore + engagementScore;

      // Determine tier
      let tier: 'bronze' | 'silver' | 'gold' | 'platinum' = 'bronze';
      if (totalScore >= 1500) tier = 'platinum';
      else if (totalScore >= 1000) tier = 'gold';
      else if (totalScore >= 500) tier = 'silver';

      // Update airdrop tracking
      const airdropData = {
        user_id: userId,
        eligibility_score: totalScore,
        points_contribution: pointsScore,
        badge_contribution: badgeScore,
        community_engagement: engagementScore,
        total_score: totalScore,
        airdrop_tier: tier,
        last_updated: new Date().toISOString()
      };

      await this.db.client
        .from('airdrop_tracking')
        .upsert(airdropData, { onConflict: 'user_id' });
    } catch (error) {
      console.error('Error updating airdrop eligibility:', error);
    }
  }

  // ==================== LEADERBOARDS ====================

  /**
   * Get user rank in points leaderboard
   */
  async getUserRank(userId: string): Promise<number> {
    try {
      const { data, error } = await this.db.client
        .from('users')
        .select('total_points')
        .eq('id', userId)
        .eq('is_active', true)
        .single();

      if (error) throw error;

      const { count, error: countError } = await this.db.client
        .from('users')
        .select('*', { count: 'exact', head: true })
        .gt('total_points', data.total_points)
        .eq('is_active', true);

      if (countError) throw countError;
      return (count || 0) + 1;
    } catch (error) {
      console.error('Error getting user rank:', error);
      return 999;
    }
  }

  /**
   * Get top users leaderboard
   */
  async getTopUsers(limit: number = 10): Promise<User[]> {
    try {
      const { data, error } = await this.db.client
        .from('users')
        .select('*')
        .eq('is_active', true)
        .order('total_points', { ascending: false })
        .limit(limit);

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error getting top users:', error);
      return [];
    }
  }

  // ==================== REFERRAL SYSTEM ====================

  /**
   * Generate referral code for user
   * Enhanced system exceeding Drift's simple referral approach
   */
  async generateReferralCode(
    userId: string, 
    platform: 'wallet' | 'telegram' | 'discord' | 'twitter',
    platformId?: string,
    expiresAt?: Date,
    maxUsage?: number
  ): Promise<ReferralCode> {
    try {
      const user = await this.getUserById(userId);
      if (!user) throw new Error('User not found');

      // Generate unique referral code
      const { data: codeData, error: codeError } = await this.db.client
        .rpc('generate_referral_code', {
          user_uuid: userId,
          platform_name: platform
        });

      if (codeError) throw codeError;

      const referralCode: ReferralCode = {
        id: uuidv4(),
        user_id: userId,
        code: codeData,
        platform,
        platform_id: platformId,
        is_active: true,
        created_at: new Date(),
        expires_at: expiresAt,
        usage_count: 0,
        max_usage: maxUsage,
        metadata: { platform_id: platformId }
      };

      const { data, error } = await this.db.client
        .from('referral_codes')
        .insert(referralCode)
        .select()
        .single();

      if (error) throw error;

      return data;
    } catch (error) {
      console.error('Error generating referral code:', error);
      throw new Error(`Failed to generate referral code: ${error.message}`);
    }
  }

  /**
   * Validate referral code
   */
  async validateReferralCode(code: string): Promise<any> {
    try {
      const { data, error } = await this.db.client
        .rpc('validate_referral_code', {
          code_to_validate: code
        });

      if (error) throw error;
      return data[0] || null;
    } catch (error) {
      console.error('Error validating referral code:', error);
      return null;
    }
  }

  /**
   * Process referral when user registers
   * Enhanced with 20% referral rewards exceeding Drift's 15%
   */
  async processReferral(
    referredUserId: string, 
    referralCode: string,
    platformContext?: any
  ): Promise<ReferralRelationship | null> {
    try {
      // Validate referral code
      const validation = await this.validateReferralCode(referralCode);
      if (!validation || !validation.is_valid) {
        throw new Error('Invalid or expired referral code');
      }

      // Check if user was already referred
      const existingReferral = await this.getReferralRelationship(referredUserId);
      if (existingReferral) {
        throw new Error('User has already been referred');
      }

      // Create referral relationship
      const relationship: ReferralRelationship = {
        id: uuidv4(),
        referrer_id: validation.user_id,
        referred_id: referredUserId,
        referral_code_id: validation.user_id, // This would need to be the actual code ID
        platform: validation.platform,
        platform_context: platformContext,
        created_at: new Date(),
        is_active: true
      };

      const { data, error } = await this.db.client
        .from('referral_relationships')
        .insert(relationship)
        .select()
        .single();

      if (error) throw error;

      // Award referral bonus to referrer (150 points)
      await this.awardPoints(
        validation.user_id,
        150,
        'earned',
        'referral_bonus',
        'Referral bonus: User successfully referred'
      );

      // Award referral bonus to referred user (100 points)
      await this.awardPoints(
        referredUserId,
        100,
        'earned',
        'referred_bonus',
        'Referral bonus: Successfully referred by community member'
      );

      return data;
    } catch (error) {
      console.error('Error processing referral:', error);
      throw new Error(`Failed to process referral: ${error.message}`);
    }
  }

  /**
   * Process referral reward when referred user earns points
   * Enhanced with 20% referral rewards exceeding Drift's 15%
   */
  async processReferralReward(referredUserId: string, pointsEarned: number): Promise<number> {
    try {
      const { data, error } = await this.db.client
        .rpc('process_referral_reward', {
          referred_user_uuid: referredUserId,
          points_amount: pointsEarned
        });

      if (error) throw error;
      return data || 0;
    } catch (error) {
      console.error('Error processing referral reward:', error);
      return 0;
    }
  }

  /**
   * Get referral relationship for user
   */
  async getReferralRelationship(userId: string): Promise<ReferralRelationship | null> {
    try {
      const { data, error } = await this.db.client
        .from('referral_relationships')
        .select('*')
        .eq('referred_id', userId)
        .eq('is_active', true)
        .single();

      if (error && error.code !== 'PGRST116') throw error;
      return data || null;
    } catch (error) {
      console.error('Error getting referral relationship:', error);
      return null;
    }
  }

  /**
   * Get user's referral codes
   */
  async getUserReferralCodes(userId: string): Promise<ReferralCode[]> {
    try {
      const { data, error } = await this.db.client
        .from('referral_codes')
        .select('*')
        .eq('user_id', userId)
        .eq('is_active', true)
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error getting user referral codes:', error);
      return [];
    }
  }

  /**
   * Get referral statistics for user
   */
  async getReferralStats(userId: string): Promise<any> {
    try {
      const { data, error } = await this.db.client
        .rpc('get_referral_stats', {
          user_uuid: userId
        });

      if (error) throw error;
      return data[0] || null;
    } catch (error) {
      console.error('Error getting referral stats:', error);
      return null;
    }
  }

  /**
   * Get referral leaderboard
   */
  async getReferralLeaderboard(limit: number = 10): Promise<any[]> {
    try {
      const { data, error } = await this.db.client
        .rpc('get_referral_leaderboard', {
          limit_count: limit
        });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error getting referral leaderboard:', error);
      return [];
    }
  }

  /**
   * Get referred users for a referrer
   */
  async getReferredUsers(referrerId: string): Promise<any[]> {
    try {
      const { data, error } = await this.db.client
        .from('referral_relationships')
        .select(`
          *,
          users!referral_relationships_referred_id_fkey (
            id,
            wallet_address,
            username,
            total_points,
            level,
            created_at
          )
        `)
        .eq('referrer_id', referrerId)
        .eq('is_active', true)
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error getting referred users:', error);
      return [];
    }
  }

  // ==================== TRADING METRICS SYSTEM (CORE POINTS) ====================

  /**
   * Process trading activity and award points
   * Based on Drift's airdrop criteria: trading volume, deposits, staking, insurance fund
   */
  async processTradingActivity(
    userId: string,
    activityType: 'trade' | 'deposit' | 'withdrawal' | 'staking' | 'insurance_fund',
    amount: number,
    marketSymbol?: string,
    side?: 'long' | 'short' | 'buy' | 'sell' | 'maker' | 'taker',
    leverage?: number
  ): Promise<number> {
    try {
      const { data, error } = await this.db.client
        .rpc('process_trading_activity', {
          user_uuid: userId,
          activity_type: activityType,
          amount: amount,
          market_symbol: marketSymbol,
          side: side,
          leverage: leverage
        });

      if (error) throw error;
      
      const pointsEarned = data || 0;
      
      // Process referral rewards if this user was referred
      if (pointsEarned > 0) {
        await this.processReferralReward(userId, pointsEarned);
      }
      
      return pointsEarned;
    } catch (error) {
      console.error('Error processing trading activity:', error);
      throw new Error(`Failed to process trading activity: ${error.message}`);
    }
  }

  /**
   * Get trading metrics for user
   */
  async getTradingMetrics(userId: string): Promise<TradingMetrics | null> {
    try {
      const { data, error } = await this.db.client
        .from('trading_metrics')
        .select('*')
        .eq('user_id', userId)
        .single();

      if (error && error.code !== 'PGRST116') throw error;
      return data || null;
    } catch (error) {
      console.error('Error getting trading metrics:', error);
      return null;
    }
  }

  /**
   * Get trading activity log for user
   */
  async getTradingActivityLog(userId: string, limit: number = 50): Promise<TradingActivityLog[]> {
    try {
      const { data, error } = await this.db.client
        .from('trading_activity_log')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false })
        .limit(limit);

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error getting trading activity log:', error);
      return [];
    }
  }

  /**
   * Get trading metrics leaderboard
   * Based on Drift's airdrop criteria
   */
  async getTradingMetricsLeaderboard(
    metricType: 'trading_volume' | 'deposits' | 'staking' | 'insurance_fund' = 'trading_volume',
    limit: number = 10
  ): Promise<any[]> {
    try {
      const { data, error } = await this.db.client
        .rpc('get_trading_metrics_leaderboard', {
          metric_type: metricType,
          limit_count: limit
        });

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error getting trading metrics leaderboard:', error);
      return [];
    }
  }

  /**
   * Get comprehensive trading analytics
   * Enhanced metrics exceeding Drift's basic approach
   */
  async getTradingAnalytics(): Promise<any> {
    try {
      const { data, error } = await this.db.client
        .rpc('get_trading_analytics');

      if (error) throw error;
      return data[0] || null;
    } catch (error) {
      console.error('Error getting trading analytics:', error);
      throw new Error(`Failed to get trading analytics: ${error.message}`);
    }
  }

  /**
   * Get trading dashboard data for user
   */
  async getTradingDashboard(userId: string): Promise<any> {
    try {
      const { data, error } = await this.db.client
        .from('trading_dashboard')
        .select('*')
        .eq('user_id', userId)
        .single();

      if (error && error.code !== 'PGRST116') throw error;
      return data || null;
    } catch (error) {
      console.error('Error getting trading dashboard:', error);
      return null;
    }
  }

  // ==================== HELPER METHODS ====================

  /**
   * Get transactions by source
   */
  async getTransactionsBySource(userId: string, source: string): Promise<PointsTransaction[]> {
    try {
      const { data, error } = await this.db.client
        .from('points_transactions')
        .select('*')
        .eq('user_id', userId)
        .eq('source', source)
        .eq('transaction_type', 'earned');

      if (error) throw error;
      return data || [];
    } catch (error) {
      console.error('Error getting transactions by source:', error);
      return [];
    }
  }

  /**
   * Get user statistics
   */
  async getUserStats(userId: string): Promise<any> {
    try {
      const user = await this.getUserById(userId);
      if (!user) return null;

      const badges = await this.getUserBadges(userId);
      const transactions = await this.getRecentTransactions(userId, 24 * 7); // Last week
      const rank = await this.getUserRank(userId);

      return {
        user,
        badge_count: badges.length,
        transaction_count: transactions.length,
        rank,
        level: user.level,
        total_points: user.total_points
      };
    } catch (error) {
      console.error('Error getting user stats:', error);
      return null;
    }
  }
}

export default CommunityPointsService;
