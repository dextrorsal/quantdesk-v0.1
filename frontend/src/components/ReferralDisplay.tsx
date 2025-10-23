import React, { useState, useEffect, useCallback } from 'react';
import { useWalletAuth } from '../hooks/useWalletAuth';
import { supabaseService } from '../services/supabaseService'; // Assuming frontend can access supabase directly for simplicity or create a dedicated frontend service

interface ReferralDisplayProps {
  user: any; // User object from useWalletAuth
}

const ReferralDisplay: React.FC<ReferralDisplayProps> = ({ user }) => {
  const [referralLink, setReferralLink] = useState<string>('');
  const [referralSummary, setReferralSummary] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  const fetchReferralData = useCallback(async () => {
    if (!user?.wallet_pubkey) return;

    setLoading(true);
    setError(null);
    try {
      const baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:3000';
      // Fetch summary
      const summaryResponse = await fetch(`${baseUrl}/api/referrals/summary?wallet=${user.wallet_pubkey}`);
      const summaryData = await summaryResponse.json();

      if (summaryResponse.ok) {
        setReferralSummary(summaryData);
      } else {
        setError(summaryData.error || 'Failed to fetch referral summary');
      }
      
      // Generate referral link
      setReferralLink(`${window.location.origin}/waitlist?ref=${user.wallet_pubkey}`);

    } catch (err: any) {
      console.error('Error fetching referral data:', err);
      setError(err.message || 'Network error');
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    fetchReferralData();
  }, [fetchReferralData]);

  if (loading) return <div>Loading referrals...</div>;
  if (error) return <div className="text-red-500">Error: {error}</div>;

  return (
    <div className="p-4 bg-gray-800 rounded-lg">
      <h3 className="text-xl font-bold mb-4 text-white">Your Referral Program</h3>
      <p className="text-gray-300 mb-4">
        Earn rewards by inviting new traders! Here's how it works:
      </p>
      <ul className="list-disc list-inside text-gray-300 mb-4">
        <li>
          <strong className="text-green-400">25% Level 1 Referral Share:</strong> Get 25% of net trading fees from users you directly refer.
        </li>
        <li>
          <strong className="text-blue-400">10% Trader Fee Discount:</strong> Users who sign up with your link receive a 10% discount on their trading fees.
        </li>
      </ul>

      {user.referrer_pubkey && (
        <p className="text-gray-300 mb-4">
          You were referred by: <code className="bg-gray-700 p-1 rounded text-sm text-yellow-300">{user.referrer_pubkey}</code>
          {user.is_activated ? <span className="text-green-500 ml-2">(Activated)</span> : <span className="text-yellow-500 ml-2">(Pending Activation)</span>}
        </p>
      )}

      <div className="mb-4">
        <label htmlFor="referral-link" className="block text-gray-300 text-sm font-bold mb-2">Your Referral Link:</label>
        <input
          id="referral-link"
          type="text"
          readOnly
          value={referralLink}
          className="w-full p-2 bg-gray-700 border border-gray-600 rounded text-green-300 break-all"
          onClick={(e) => (e.target as HTMLInputElement).select()}
        />
        <button
          onClick={() => navigator.clipboard.writeText(referralLink)}
          className="mt-2 bg-blue-600 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg transition-colors w-full"
        >
          Copy Link
        </button>
        {navigator.share && (
          <button
            onClick={() => navigator.share({
              title: 'QuantDesk Referral',
              text: 'Join QuantDesk and earn rewards!',
              url: referralLink,
            })}
            className="mt-2 bg-purple-600 hover:bg-purple-700 text-white font-bold py-2 px-4 rounded-lg transition-colors w-full"
          >
            Share via...
          </button>
        )}
      </div>

      {referralSummary && (
        <div>
          <h4 className="text-lg font-bold mb-2 text-white">Your Referral Stats:</h4>
          <p className="text-gray-300">Referred Users: <strong className="text-purple-400">{referralSummary.count}</strong></p>
          <p className="text-gray-300">Estimated Earnings (SOL): <strong className="text-green-400">{referralSummary.earnings.toFixed(4)}</strong></p>
          {/* Add claim button here later */}
        </div>
      )}
    </div>
  );
};

export default ReferralDisplay;
