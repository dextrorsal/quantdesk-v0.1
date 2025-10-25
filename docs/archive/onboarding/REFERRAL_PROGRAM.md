## Referral Program (Devnet → Mainnet)

I’m rewarding real action, not idle signups. The program is simple and sustainable on Solana: a 25% fee share to referrers and a 10% fee discount to traders who join via a referral. Together, that’s effectively 35% given back—split to boost both acquisition and conversion.

### Core Terms
- **Level 1 Referral Share**: 25% of net trading fees from direct referrals.
- **Trader Discount**: 10% off trading fees when signing up via a referral link (OGs keep this for 30 days).
- **Level 2 (Optional for later)**: 5% of net trading fees from your referrals’ referrals (off by default at launch).
- **Payout Asset**: SOL (switchable to future token at mainnet discretion).
- **Payout Cadence**: Weekly claims via dashboard.
- **Net Basis**: Calculated on net platform fees after partner fees/rebates.

### Activation & Eligibility
- **Activation Threshold**: A referral “activates” after hitting a minimal bar (e.g., 5 test trades or $10 simulated volume).
- **Anti-Sybil**: One wallet per signup. Self-referrals and circular patterns are disqualified.
- **Referral Binding**: The first successful login with `?ref=<pubkey>` binds the relationship.

### Guardrails
- **Intro Window**: 25% + 10% for first 30 days for OGs, then review.
- **Tiering**: High-volume referrers can retain 25%; others may settle to 20% post-intro.
- **Caps**: Per-referrer weekly cap; per-trader discount cap to reduce wash.
- **Rate Limits**: Rewards issued only on eligible activity; paused for abuse.

### User Experience
- **Wallet-First**: Sign-In with Solana (nonce + signature). No email required.
- **Instant Share**: After signup, users see and can share `?ref=<their_pubkey>`.
- **Dashboard**: View crew, volume, fee share, and claim status. Mobile-ready.
- **Transparency**: Referral graph and earnings mirrored to a PDA for on-chain proof.

### Implementation Notes
- **Tracking**: Start off-chain (Supabase), mirror to a PDA for demo transparency.
- **Claims**: HttpOnly session → claim SOL weekly; batch transactions for cost.
- **Events**: Activation flips when trade/chat thresholds are met by referee.
- **Later**: Enable Level 2, token payouts, and volume-based multipliers.

### Rationale
Splitting benefits (25% to referrer, 10% to the trader) performs similarly to a flat 35% referral but improves conversion: traders feel immediate value, while referrers still earn meaningful yield. The platform retains ~65% of fees—sustainable for infra and growth.


