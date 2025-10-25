## Referral Yield: How I’m Rewarding Early Testers On Solana

I’m building a crypto terminal that blends trading, chat, alpha sharing, wallet tools, and Mikey—our data-hungry AI. As we wrap devnet and approach mainnet, I want to reward real action, not idle signups. So I’m rolling out a simple, web3-native incentive that turns early testers into long-term insiders: Referral Yield.

In plain English: if you bring your crew, you earn a share of the fees their trading generates. It’s familiar (think BullX/Axiom) but tuned for our stack, our community, and Solana’s speed.

### Why Referral Yield (and not a blunt airdrop)?
- **Sustainable**: Rewards come from actual usage (fees), not inflation.
- **Viral by design**: Your link → their trades → your yield. Easy to share, easy to understand.
- **Proven**: BullX pays ~35% on direct referrals; Axiom runs 30% L1 + 3% L2 with ranks. I’m adopting the best bits and keeping it clean on-chain.
- **Judge-ready**: I can demo real wallets, real claims, and transparent accounting on devnet PDAs.

### The Model (Simple, Fair, and Fun)
- **Your earnings**
  - Level 1: 25–35% of trading fees from your direct referrals.
  - Level 2: 5–10% from your referrals’ referrals.
  - Payouts in SOL (or our future token), claimable weekly or monthly from your dashboard.
- **Early tester boost**
  - OG waitlisters get a 10% fee discount on their own trades.
  - Double referral rates for your first 30 days after joining.
- **When it counts**
  - A referral “activates” after they connect a wallet and complete a small threshold (e.g., 5 test trades or $10 simulated volume). This filters out fake signups.
- **Your link**
  - Share a unique invite: `[myapp.com]/waitlist?ref=<your_wallet>`. Each successful referral bumps you up the waitlist and grows your fee share.

### Onboarding and Accessibility (No-Headache UX)
- **Wallet-first signup**: One-click connect with Solana wallet adapter. No emails required.
- **Queue jump + instant share**: On signup, I show your invite link right away—copy, share, done.
- **Accessible dashboard**: A clean “Referrals & Earnings” view with your crew, volume, fee share, and claim status. Works on desktop and mobile.
- **Transparent and auditable**: I track referrals in a PDA on Solana for verifiability. You can query your status on-chain.
- **Sybil resistance, lightweight**: One wallet per signup, activation threshold, and anti-self-ref checks. No heavy KYC, no friction.

### Implementation Phases (Devnet → Mainnet)
- **Phase 1: Devnet pilot (4–6 hours)**
  - Extend the waitlist with referral links and wallet-based signups.
  - Track referrals off-chain for speed (Supabase) and mirror to a PDA for proof.
  - Weekly SOL “drip” payouts via a simple cron job. Showcase in the dashboard.
- **Phase 2: On-chain proof**
  - PDAs store referrer → referee relationships and cumulative fee shares.
  - Add a claim instruction for SOL/token payouts.
  - Publish a public query: “Top referrer: X invites, Y SOL earned.”
- **Phase 3: Mainnet cutover**
  - Turn on live fee-based rewards.
  - Migrate balances and keep claims smooth and cheap.

### Payout Mechanics (No Surprises)
- Fees accumulate from eligible trades across your Level 1 and Level 2 network.
- I calculate your share off-chain in real-time, anchor it on-chain, and let you claim in SOL.
- Claims are batched to keep costs low. You can set reminders or auto-claim.

### Extra Ways to Earn (Beyond Referrals)
- **Early Tester NFT Badges**: Soulbound badges (Bronze/Silver/Gold) unlock fee discounts, private chat channels, and Mikey analytics perks.
- **Alpha Bounties**: Share high-signal posts in chat; community upvotes and Mikey’s scoring pay out SOL.
- **Mikey Points (Stake to use)**: Earn points from trades, chats, and referrals; stake points to unlock premium Mikey sessions (alerts, backtests).
- **Leaderboards + Raffles**: Weekly prizes for top testers and random winners who hit activity baselines.
- **Social Boosts**: Tweet your link, earn SOL on verified posts—bonus for engagement.

### What I’ll Track (And Show You)
- Referral conversion rate and viral coefficient (aiming >1.0).
- Total testers, active wallets, cumulative referral earnings.
- Leaderboard snapshots and weekly payouts.
- All core stats are explorable via on-chain PDAs and a public dashboard.

### Guardrails and Fair Play
- One wallet per signup; self-referrals disqualified.
- Thresholds prevent farming and bot loops.
- Clear terms for fee share rates and payout schedules.
- Devnet stress-tests before mainnet turn-on.

### What You Can Expect
- A low-lift way to earn from your network’s trading.
- Tangible SOL rewards, not vague promises.
- Clean UX: connect, share, test, claim.
- Recognition for contribution—badges, ranks, and access.

If you’re early, you’re valued. Bring your crew, test hard, share alpha, and I’ll make sure you feel like an insider from day one. Devnet’s almost ready—once I flip this on, your invite link is your yield machine.

### Short-Term Plan
- Ship the referral link flow, the dashboard, and weekly SOL claims first.
- Layer in badges and alpha bounties next.
- If we want an aggressive start, I’ll lock Level 1 at 35% from day one. Otherwise, I’ll start balanced at 30% and scale.

### Open Question
- Prefer 25% or 35% for Level 1 at launch? I can tune the OG multiplier window either way.


