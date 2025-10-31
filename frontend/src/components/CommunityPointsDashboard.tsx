// QuantDesk Community Points Dashboard Component
// Enhanced implementation exceeding Drift's capabilities
// "More Open Than Drift" competitive positioning

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Trophy, 
  Star, 
  Gift, 
  Users, 
  TrendingUp, 
  Award, 
  Zap,
  Crown,
  Target,
  BarChart3
} from 'lucide-react';

interface User {
  id: string;
  wallet_address: string;
  username?: string;
  total_points: number;
  level: string;
  created_at: string;
}

interface UserStats {
  badge_count: number;
  transaction_count: number;
  rank: number;
  level: string;
}

interface Badge {
  id: string;
  name: string;
  description: string;
  icon_url?: string;
  points_required: number;
  category: string;
  rarity: 'common' | 'rare' | 'epic' | 'legendary';
}

interface UserBadge {
  id: string;
  badge_id: string;
  earned_at: string;
  badges: Badge;
}

interface RedemptionOption {
  id: string;
  name: string;
  description: string;
  points_cost: number;
  redemption_type: string;
  cooldown_hours: number;
  expires_at?: string;
}

interface LeaderboardEntry {
  rank: number;
  wallet_address: string;
  username?: string;
  total_points: number;
  level: string;
}

interface AirdropData {
  eligibility_score: number;
  points_contribution: number;
  badge_contribution: number;
  community_engagement: number;
  total_score: number;
  airdrop_tier: 'bronze' | 'silver' | 'gold' | 'platinum';
}

interface CommunityAnalytics {
  total_users: number;
  total_points_awarded: number;
  total_badges_earned: number;
  total_redemptions: number;
  average_points_per_user: number;
}

/**
 * Community Points Dashboard - Enhanced implementation exceeding Drift's capabilities
 * 
 * Key Features vs Drift:
 * - Advanced points system vs simple referral system
 * - AI integration tiers vs no AI features
 * - Comprehensive gamification vs basic rewards
 * - Multi-service architecture vs smart contracts only
 */
export const CommunityPointsDashboard: React.FC = () => {
  const [user, setUser] = useState<User | null>(null);
  const [userStats, setUserStats] = useState<UserStats | null>(null);
  const [userBadges, setUserBadges] = useState<UserBadge[]>([]);
  const [redemptionOptions, setRedemptionOptions] = useState<RedemptionOption[]>([]);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [airdropData, setAirdropData] = useState<AirdropData | null>(null);
  const [analytics, setAnalytics] = useState<CommunityAnalytics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Mock wallet address - in real implementation, get from wallet connection
  const walletAddress = 'mock_wallet_address';

  useEffect(() => {
    loadCommunityData();
  }, []);

  const loadCommunityData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load user data
      const userResponse = await fetch(`/api/community/user/${walletAddress}`);
      if (userResponse.ok) {
        const userData = await userResponse.json();
        setUser(userData.user);
        setUserStats(userData.stats);
      }

      // Load user badges
      if (user) {
        const badgesResponse = await fetch(`/api/community/badges/user/${user.id}`);
        if (badgesResponse.ok) {
          const badgesData = await badgesResponse.json();
          setUserBadges(badgesData.badges);
        }

        // Load airdrop data
        const airdropResponse = await fetch(`/api/community/airdrop/eligibility/${user.id}`);
        if (airdropResponse.ok) {
          const airdropData = await airdropResponse.json();
          setAirdropData(airdropData.airdrop);
        }
      }

      // Load redemption options
      const redemptionsResponse = await fetch('/api/community/redemptions');
      if (redemptionsResponse.ok) {
        const redemptionsData = await redemptionsResponse.json();
        setRedemptionOptions(redemptionsData.redemptions);
      }

      // Load leaderboard
      const leaderboardResponse = await fetch('/api/community/leaderboard/points?limit=10');
      if (leaderboardResponse.ok) {
        const leaderboardData = await leaderboardResponse.json();
        setLeaderboard(leaderboardData.leaderboard);
      }

      // Load analytics
      const analyticsResponse = await fetch('/api/community/analytics/overview');
      if (analyticsResponse.ok) {
        const analyticsData = await analyticsResponse.json();
        setAnalytics(analyticsData.analytics);
      }

    } catch (err) {
      setError('Failed to load community data');
      console.error('Error loading community data:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRedeem = async (redemptionOptionId: string) => {
    if (!user) return;

    try {
      const response = await fetch('/api/community/redemptions/redeem', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: user.id,
          redemption_option_id: redemptionOptionId
        })
      });

      if (response.ok) {
        const result = await response.json();
        alert(`Successfully redeemed: ${result.message}`);
        loadCommunityData(); // Reload data
      } else {
        const error = await response.json();
        alert(`Redemption failed: ${error.details}`);
      }
    } catch (err) {
      alert('Failed to redeem points');
      console.error('Error redeeming points:', err);
    }
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'legend': return 'text-purple-600';
      case 'expert': return 'text-blue-600';
      case 'advanced': return 'text-green-600';
      case 'intermediate': return 'text-yellow-600';
      default: return 'text-gray-600';
    }
  };

  const getRarityColor = (rarity: string) => {
    switch (rarity) {
      case 'legendary': return 'bg-purple-100 text-purple-800 border-purple-200';
      case 'epic': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'rare': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getTierColor = (tier: string) => {
    switch (tier) {
      case 'platinum': return 'bg-gradient-to-r from-gray-100 to-gray-300 text-gray-800';
      case 'gold': return 'bg-gradient-to-r from-yellow-100 to-yellow-300 text-yellow-800';
      case 'silver': return 'bg-gradient-to-r from-gray-100 to-gray-200 text-gray-800';
      default: return 'bg-gradient-to-r from-orange-100 to-orange-200 text-orange-800';
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="text-red-600 text-xl mb-4">⚠️ Error</div>
          <div className="text-gray-600">{error}</div>
          <Button onClick={loadCommunityData} className="mt-4">
            Retry
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            QuantDesk Community
          </h1>
          <p className="text-lg text-gray-600">
            "More Open Than Drift" - Advanced community engagement exceeding Drift's simple referral system
          </p>
        </div>

        {/* User Stats Overview */}
        {user && userStats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Total Points</p>
                    <p className="text-2xl font-bold text-blue-600">{user.total_points.toLocaleString()}</p>
                  </div>
                  <Zap className="h-8 w-8 text-blue-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Level</p>
                    <p className={`text-2xl font-bold ${getLevelColor(userStats.level)}`}>
                      {userStats.level.charAt(0).toUpperCase() + userStats.level.slice(1)}
                    </p>
                  </div>
                  <Crown className="h-8 w-8 text-yellow-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Badges</p>
                    <p className="text-2xl font-bold text-green-600">{userStats.badge_count}</p>
                  </div>
                  <Award className="h-8 w-8 text-green-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Rank</p>
                    <p className="text-2xl font-bold text-purple-600">#{userStats.rank}</p>
                  </div>
                  <Trophy className="h-8 w-8 text-purple-600" />
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Main Content */}
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="badges">Badges</TabsTrigger>
            <TabsTrigger value="redemptions">Redemptions</TabsTrigger>
            <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
            <TabsTrigger value="airdrop">Airdrop</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Recent Activity */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Recent Activity
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                        <span className="text-sm">Early testing participation</span>
                      </div>
                      <span className="text-sm font-medium text-green-600">+100 points</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-blue-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                        <span className="text-sm">Badge earned bonus</span>
                      </div>
                      <span className="text-sm font-medium text-blue-600">+50 points</span>
                    </div>
                    <div className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                        <span className="text-sm">Daily login streak</span>
                      </div>
                      <span className="text-sm font-medium text-purple-600">+10 points</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Community Analytics */}
              {analytics && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <BarChart3 className="h-5 w-5" />
                      Community Analytics
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Total Users</span>
                        <span className="text-sm font-medium">{analytics.total_users.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Points Awarded</span>
                        <span className="text-sm font-medium">{analytics.total_points_awarded.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Badges Earned</span>
                        <span className="text-sm font-medium">{analytics.total_badges_earned.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Redemptions</span>
                        <span className="text-sm font-medium">{analytics.total_redemptions.toLocaleString()}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm text-gray-600">Avg Points/User</span>
                        <span className="text-sm font-medium">{analytics.average_points_per_user.toLocaleString()}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          </TabsContent>

          {/* Badges Tab */}
          <TabsContent value="badges" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Award className="h-5 w-5" />
                  Your Badges ({userBadges.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {userBadges.map((userBadge) => (
                    <div key={userBadge.id} className="p-4 border rounded-lg hover:shadow-md transition-shadow">
                      <div className="flex items-center gap-3 mb-2">
                        <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                          <Star className="h-5 w-5 text-white" />
                        </div>
                        <div>
                          <h3 className="font-medium">{userBadge.badges.name}</h3>
                          <Badge className={`text-xs ${getRarityColor(userBadge.badges.rarity)}`}>
                            {userBadge.badges.rarity}
                          </Badge>
                        </div>
                      </div>
                      <p className="text-sm text-gray-600 mb-2">{userBadge.badges.description}</p>
                      <p className="text-xs text-gray-500">
                        Earned: {new Date(userBadge.earned_at).toLocaleDateString()}
                      </p>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Redemptions Tab */}
          <TabsContent value="redemptions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Gift className="h-5 w-5" />
                  Available Redemptions
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {redemptionOptions.map((option) => (
                    <div key={option.id} className="p-4 border rounded-lg hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium">{option.name}</h3>
                        <Badge variant="outline">{option.points_cost} points</Badge>
                      </div>
                      <p className="text-sm text-gray-600 mb-3">{option.description}</p>
                      <div className="flex items-center justify-between">
                        <span className="text-xs text-gray-500">
                          {option.redemption_type.replace('_', ' ')}
                        </span>
                        <Button 
                          size="sm" 
                          onClick={() => handleRedeem(option.id)}
                          disabled={!user || user.total_points < option.points_cost}
                        >
                          Redeem
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Leaderboard Tab */}
          <TabsContent value="leaderboard" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Trophy className="h-5 w-5" />
                  Points Leaderboard
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {leaderboard.map((entry) => (
                    <div key={entry.rank} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${
                          entry.rank === 1 ? 'bg-yellow-100 text-yellow-800' :
                          entry.rank === 2 ? 'bg-gray-100 text-gray-800' :
                          entry.rank === 3 ? 'bg-orange-100 text-orange-800' :
                          'bg-blue-100 text-blue-800'
                        }`}>
                          {entry.rank}
                        </div>
                        <div>
                          <p className="font-medium">{entry.username || entry.wallet_address.slice(0, 8) + '...'}</p>
                          <p className="text-sm text-gray-600">{entry.level}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-medium">{entry.total_points.toLocaleString()} points</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Airdrop Tab */}
          <TabsContent value="airdrop" className="space-y-6">
            {airdropData && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Target className="h-5 w-5" />
                    Airdrop Eligibility
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    {/* Tier Display */}
                    <div className="text-center">
                      <div className={`inline-block px-6 py-3 rounded-full text-lg font-bold ${getTierColor(airdropData.airdrop_tier)}`}>
                        {airdropData.airdrop_tier.charAt(0).toUpperCase() + airdropData.airdrop_tier.slice(1)} Tier
                      </div>
                      <p className="text-sm text-gray-600 mt-2">
                        Eligibility Score: {airdropData.eligibility_score}
                      </p>
                    </div>

                    {/* Score Breakdown */}
                    <div className="space-y-4">
                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium">Points Contribution</span>
                          <span className="text-sm">{airdropData.points_contribution}/1000</span>
                        </div>
                        <Progress value={(airdropData.points_contribution / 1000) * 100} className="h-2" />
                      </div>

                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium">Badge Contribution</span>
                          <span className="text-sm">{airdropData.badge_contribution}/500</span>
                        </div>
                        <Progress value={(airdropData.badge_contribution / 500) * 100} className="h-2" />
                      </div>

                      <div>
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium">Community Engagement</span>
                          <span className="text-sm">{airdropData.community_engagement}/500</span>
                        </div>
                        <Progress value={(airdropData.community_engagement / 500) * 100} className="h-2" />
                      </div>
                    </div>

                    {/* Next Tier Requirements */}
                    <div className="p-4 bg-blue-50 rounded-lg">
                      <h4 className="font-medium text-blue-900 mb-2">Next Tier Requirements</h4>
                      <p className="text-sm text-blue-800">
                        {airdropData.airdrop_tier === 'bronze' && 'Reach 500 points for Silver tier'}
                        {airdropData.airdrop_tier === 'silver' && 'Reach 1000 points for Gold tier'}
                        {airdropData.airdrop_tier === 'gold' && 'Reach 1500 points for Platinum tier'}
                        {airdropData.airdrop_tier === 'platinum' && 'You have reached the highest tier!'}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default CommunityPointsDashboard;
