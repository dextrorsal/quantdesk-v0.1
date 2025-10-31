// QuantDesk Referral System Dashboard Component
// Enhanced implementation exceeding Drift's capabilities
// "More Open Than Drift" competitive positioning with 20% referral rewards

import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Users, 
  Share2, 
  Copy, 
  ExternalLink,
  Trophy,
  TrendingUp,
  MessageCircle,
  Twitter,
  Discord,
  Wallet,
  BarChart3,
  Gift,
  Zap
} from 'lucide-react';

interface ReferralCode {
  id: string;
  code: string;
  platform: 'wallet' | 'telegram' | 'discord' | 'twitter';
  platform_id?: string;
  usage_count: number;
  max_usage?: number;
  expires_at?: string;
  created_at: string;
}

interface ReferralStats {
  total_referrals: number;
  active_referrals: number;
  total_points_earned: number;
  referral_conversion_rate: number;
  platform_breakdown: any;
  last_referral_at?: string;
}

interface ReferredUser {
  id: string;
  referred_id: string;
  platform: string;
  created_at: string;
  users: {
    wallet_address: string;
    username?: string;
    total_points: number;
    level: string;
  };
}

interface LeaderboardEntry {
  rank: number;
  user_id: string;
  wallet_address: string;
  username?: string;
  total_referrals: number;
  total_points_earned: number;
  platform_breakdown: any;
}

/**
 * Referral System Dashboard - Enhanced implementation exceeding Drift's capabilities
 * 
 * Key Features vs Drift:
 * - 20% referral rewards vs Drift's 15%
 * - Multi-platform support vs web-only
 * - Advanced analytics vs basic tracking
 * - Gamification elements vs simple rewards
 * - Comprehensive referral management vs basic system
 */
export const ReferralSystemDashboard: React.FC = () => {
  const [referralCodes, setReferralCodes] = useState<ReferralCode[]>([]);
  const [referralStats, setReferralStats] = useState<ReferralStats | null>(null);
  const [referredUsers, setReferredUsers] = useState<ReferredUser[]>([]);
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [newCodePlatform, setNewCodePlatform] = useState<'wallet' | 'telegram' | 'discord' | 'twitter'>('wallet');
  const [newCodePlatformId, setNewCodePlatformId] = useState('');

  // Mock user ID - in real implementation, get from user context
  const userId = 'mock_user_id';

  useEffect(() => {
    loadReferralData();
  }, []);

  const loadReferralData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load referral codes
      const codesResponse = await fetch(`/api/referral/codes/${userId}`);
      if (codesResponse.ok) {
        const codesData = await codesResponse.json();
        setReferralCodes(codesData.referral_codes);
      }

      // Load referral stats
      const statsResponse = await fetch(`/api/referral/stats/${userId}`);
      if (statsResponse.ok) {
        const statsData = await statsResponse.json();
        setReferralStats(statsData.stats);
      }

      // Load referred users
      const referredResponse = await fetch(`/api/referral/referred/${userId}`);
      if (referredResponse.ok) {
        const referredData = await referredResponse.json();
        setReferredUsers(referredData.referred_users);
      }

      // Load leaderboard
      const leaderboardResponse = await fetch('/api/referral/leaderboard?limit=10');
      if (leaderboardResponse.ok) {
        const leaderboardData = await leaderboardResponse.json();
        setLeaderboard(leaderboardData.leaderboard);
      }

    } catch (err) {
      setError('Failed to load referral data');
      console.error('Error loading referral data:', err);
    } finally {
      setLoading(false);
    }
  };

  const generateReferralCode = async () => {
    try {
      const response = await fetch('/api/referral/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          platform: newCodePlatform,
          platform_id: newCodePlatformId || undefined
        })
      });

      if (response.ok) {
        const result = await response.json();
        alert(`Referral code generated: ${result.referral_code.code}`);
        loadReferralData(); // Reload data
        setNewCodePlatformId(''); // Reset form
      } else {
        const error = await response.json();
        alert(`Failed to generate code: ${error.details}`);
      }
    } catch (err) {
      alert('Failed to generate referral code');
      console.error('Error generating referral code:', err);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    alert('Copied to clipboard!');
  };

  const getPlatformIcon = (platform: string) => {
    switch (platform) {
      case 'wallet': return <Wallet className="h-4 w-4" />;
      case 'telegram': return <MessageCircle className="h-4 w-4" />;
      case 'discord': return <Discord className="h-4 w-4" />;
      case 'twitter': return <Twitter className="h-4 w-4" />;
      default: return <Share2 className="h-4 w-4" />;
    }
  };

  const getPlatformColor = (platform: string) => {
    switch (platform) {
      case 'wallet': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'telegram': return 'bg-cyan-100 text-cyan-800 border-cyan-200';
      case 'discord': return 'bg-indigo-100 text-indigo-800 border-indigo-200';
      case 'twitter': return 'bg-sky-100 text-sky-800 border-sky-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const generateShareLink = async (code: string, platform: string) => {
    try {
      const response = await fetch(`/api/referral/share/${platform}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          referral_code: code
        })
      });

      if (response.ok) {
        const result = await response.json();
        return result.share_link || result.share_message;
      }
    } catch (err) {
      console.error('Error generating share link:', err);
    }
    return null;
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
          <Button onClick={loadReferralData} className="mt-4">
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
            Referral System
          </h1>
          <p className="text-lg text-gray-600">
            "More Rewarding Than Drift" - 20% referral rewards vs Drift's 15%
          </p>
        </div>

        {/* Referral Stats Overview */}
        {referralStats && (
          <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Total Referrals</p>
                    <p className="text-2xl font-bold text-blue-600">{referralStats.total_referrals}</p>
                  </div>
                  <Users className="h-8 w-8 text-blue-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Active Referrals</p>
                    <p className="text-2xl font-bold text-green-600">{referralStats.active_referrals}</p>
                  </div>
                  <TrendingUp className="h-8 w-8 text-green-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Points Earned</p>
                    <p className="text-2xl font-bold text-purple-600">{referralStats.total_points_earned.toLocaleString()}</p>
                  </div>
                  <Gift className="h-8 w-8 text-purple-600" />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardContent className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm font-medium text-gray-600">Conversion Rate</p>
                    <p className="text-2xl font-bold text-orange-600">{referralStats.referral_conversion_rate.toFixed(1)}%</p>
                  </div>
                  <BarChart3 className="h-8 w-8 text-orange-600" />
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Main Content */}
        <Tabs defaultValue="codes" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4">
            <TabsTrigger value="codes">My Codes</TabsTrigger>
            <TabsTrigger value="referred">Referred Users</TabsTrigger>
            <TabsTrigger value="leaderboard">Leaderboard</TabsTrigger>
            <TabsTrigger value="compare">vs Drift</TabsTrigger>
          </TabsList>

          {/* My Codes Tab */}
          <TabsContent value="codes" className="space-y-6">
            {/* Generate New Code */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Zap className="h-5 w-5" />
                  Generate New Referral Code
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <Label htmlFor="platform">Platform</Label>
                    <select
                      id="platform"
                      value={newCodePlatform}
                      onChange={(e) => setNewCodePlatform(e.target.value as any)}
                      className="w-full p-2 border rounded-md"
                    >
                      <option value="wallet">Wallet</option>
                      <option value="telegram">Telegram</option>
                      <option value="discord">Discord</option>
                      <option value="twitter">Twitter</option>
                    </select>
                  </div>
                  <div>
                    <Label htmlFor="platformId">Platform ID (Optional)</Label>
                    <Input
                      id="platformId"
                      value={newCodePlatformId}
                      onChange={(e) => setNewCodePlatformId(e.target.value)}
                      placeholder="Username, handle, etc."
                    />
                  </div>
                  <div className="flex items-end">
                    <Button onClick={generateReferralCode} className="w-full">
                      Generate Code
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Existing Codes */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Share2 className="h-5 w-5" />
                  My Referral Codes ({referralCodes.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {referralCodes.map((code) => (
                    <div key={code.id} className="p-4 border rounded-lg hover:shadow-md transition-shadow">
                      <div className="flex items-center justify-between mb-3">
                        <div className="flex items-center gap-3">
                          <div className={`p-2 rounded-full ${getPlatformColor(code.platform)}`}>
                            {getPlatformIcon(code.platform)}
                          </div>
                          <div>
                            <h3 className="font-medium">{code.code}</h3>
                            <Badge className={`text-xs ${getPlatformColor(code.platform)}`}>
                              {code.platform}
                            </Badge>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-gray-600">Usage: {code.usage_count}</p>
                          {code.max_usage && (
                            <p className="text-sm text-gray-600">Max: {code.max_usage}</p>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => copyToClipboard(code.code)}
                        >
                          <Copy className="h-4 w-4 mr-1" />
                          Copy Code
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => copyToClipboard(`https://quantdesk.io/ref/${code.code}`)}
                        >
                          <ExternalLink className="h-4 w-4 mr-1" />
                          Copy Link
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={async () => {
                            const shareLink = await generateShareLink(code.code, code.platform);
                            if (shareLink) {
                              copyToClipboard(shareLink);
                            }
                          }}
                        >
                          <Share2 className="h-4 w-4 mr-1" />
                          Share
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Referred Users Tab */}
          <TabsContent value="referred" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="h-5 w-5" />
                  Referred Users ({referredUsers.length})
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {referredUsers.map((user) => (
                    <div key={user.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className={`p-2 rounded-full ${getPlatformColor(user.platform)}`}>
                          {getPlatformIcon(user.platform)}
                        </div>
                        <div>
                          <p className="font-medium">{user.users.username || user.users.wallet_address.slice(0, 8) + '...'}</p>
                          <p className="text-sm text-gray-600">{user.users.level} • {user.users.total_points} points</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-sm text-gray-600">
                          {new Date(user.created_at).toLocaleDateString()}
                        </p>
                        <Badge variant="outline">{user.platform}</Badge>
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
                  Referral Leaderboard
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
                          <p className="text-sm text-gray-600">{entry.total_referrals} referrals</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-medium">{entry.total_points_earned.toLocaleString()} points</p>
                        <p className="text-sm text-gray-600">earned</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Compare with Drift Tab */}
          <TabsContent value="compare" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <BarChart3 className="h-5 w-5" />
                  QuantDesk vs Drift Protocol
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Drift Protocol */}
                  <div className="p-4 border rounded-lg">
                    <h3 className="font-bold text-lg mb-3 text-gray-700">Drift Protocol</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">Referral Reward:</span>
                        <span className="text-sm font-medium">15% of taker fees</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">User Discount:</span>
                        <span className="text-sm font-medium">5% fee discount</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Platform Support:</span>
                        <span className="text-sm font-medium">Web only</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Tracking:</span>
                        <span className="text-sm font-medium">Browser session-based</span>
                      </div>
                    </div>
                    <div className="mt-4">
                      <h4 className="font-medium text-sm text-gray-600 mb-2">Limitations:</h4>
                      <ul className="text-sm text-gray-600 space-y-1">
                        <li>• Limited to web platform only</li>
                        <li>• Simple fee sharing model</li>
                        <li>• No multi-platform support</li>
                        <li>• No advanced analytics</li>
                        <li>• No gamification elements</li>
                      </ul>
                    </div>
                  </div>

                  {/* QuantDesk */}
                  <div className="p-4 border rounded-lg bg-gradient-to-br from-blue-50 to-indigo-50">
                    <h3 className="font-bold text-lg mb-3 text-blue-700">QuantDesk</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">Referral Reward:</span>
                        <span className="text-sm font-medium text-green-600">20% of earned points</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">User Bonus:</span>
                        <span className="text-sm font-medium text-green-600">100 points + ongoing 20%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Platform Support:</span>
                        <span className="text-sm font-medium text-green-600">Multi-platform</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Tracking:</span>
                        <span className="text-sm font-medium text-green-600">Advanced analytics</span>
                      </div>
                    </div>
                    <div className="mt-4">
                      <h4 className="font-medium text-sm text-blue-600 mb-2">Advantages:</h4>
                      <ul className="text-sm text-blue-600 space-y-1">
                        <li>• Multi-platform referral support</li>
                        <li>• Higher referral rewards (20% vs 15%)</li>
                        <li>• Comprehensive analytics and tracking</li>
                        <li>• Gamification with badges and leaderboards</li>
                        <li>• Advanced referral code management</li>
                        <li>• Platform-specific referral codes</li>
                        <li>• Detailed referral statistics</li>
                        <li>• Referral leaderboards and competitions</li>
                      </ul>
                    </div>
                  </div>
                </div>

                {/* Competitive Positioning */}
                <div className="mt-6 p-4 bg-gradient-to-r from-purple-50 to-pink-50 rounded-lg">
                  <h3 className="font-bold text-lg mb-3 text-purple-700">Competitive Positioning</h3>
                  <div className="text-center">
                    <h4 className="text-xl font-bold text-purple-800 mb-2">"More Rewarding Than Drift"</h4>
                    <p className="text-purple-600 mb-4">
                      QuantDesk offers superior referral rewards and comprehensive multi-platform support
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                      <div className="text-left">
                        <h5 className="font-medium text-purple-700 mb-2">Key Differentiators:</h5>
                        <ul className="space-y-1 text-purple-600">
                          <li>• 20% referral rewards vs Drift's 15%</li>
                          <li>• Multi-platform support vs web-only</li>
                          <li>• Advanced analytics vs basic tracking</li>
                          <li>• Gamification elements vs simple rewards</li>
                        </ul>
                      </div>
                      <div className="text-left">
                        <h5 className="font-medium text-purple-700 mb-2">Technical Advantages:</h5>
                        <ul className="space-y-1 text-purple-600">
                          <li>• Comprehensive referral management</li>
                          <li>• Platform-specific referral codes</li>
                          <li>• Detailed referral statistics</li>
                          <li>• Referral leaderboards and competitions</li>
                        </ul>
                      </div>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default ReferralSystemDashboard;
