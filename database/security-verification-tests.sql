-- =============================================
-- SECURITY VERIFICATION TESTS
-- Run these as different roles to verify RLS
-- =============================================

-- Test 1: Verify admin_users table is protected
-- This should return 0 rows (not accessible to regular users)
SELECT COUNT(*) FROM admin_users;
-- Expected: Error or 0 rows for regular users

-- Test 2: Verify users can only see their own positions
-- Replace 'USER_WALLET' with test wallet address
SELECT * FROM positions;
-- Expected: Only shows positions for authenticated user's wallet

-- Test 3: Verify liquidations are protected
SELECT * FROM liquidations;
-- Expected: Only shows user's own liquidations

-- Test 4: Verify system_events is protected
SELECT * FROM system_events;
-- Expected: Error (only service_role can access)

-- Test 5: Verify views don't leak data
SELECT * FROM active_positions;
-- Expected: Error (view should not exist or be protected)

-- Test 6: Verify public data is accessible
SELECT * FROM markets;
-- Expected: Works for everyone (public data)

-- Test 7: Verify oracle prices are public
SELECT * FROM oracle_prices LIMIT 10;
-- Expected: Works for everyone (on-chain public data)

-- Test 8: Test cross-user data access (should fail)
-- User A tries to access User B's positions
SELECT p.* FROM positions p
JOIN users u ON p.user_id = u.id
WHERE u.wallet_address = 'DIFFERENT_WALLET_ADDRESS';
-- Expected: 0 rows (RLS blocks access)

-- Test 9: Verify admin audit logs are protected
SELECT * FROM admin_audit_logs;
-- Expected: Error (only service_role can access)

-- Test 10: Verify insurance fund is readable but not writable
SELECT * FROM insurance_fund;
-- Expected: Works for everyone (public financial data)
INSERT INTO insurance_fund (balance, asset) VALUES (1000, 'USDC');
-- Expected: Error for non-service_role users

-- Test 11: Verify auction data is public read
SELECT * FROM auctions LIMIT 5;
-- Expected: Works for everyone
SELECT * FROM auction_quotes LIMIT 5;
-- Expected: Works for everyone
SELECT * FROM auction_settlements LIMIT 5;
-- Expected: Works for everyone

-- Test 12: Verify secure views work correctly
SELECT * FROM public_market_summary;
-- Expected: Works for everyone (aggregated data only)
SELECT * FROM my_portfolio;
-- Expected: Only shows authenticated user's portfolio

-- Test 13: Verify functions are protected
SELECT update_user_balance('00000000-0000-0000-0000-000000000000', 'USDC', 100);
-- Expected: Error for non-service_role users

-- Test 14: Verify portfolio function auth check
SELECT * FROM get_user_portfolio('DIFFERENT_WALLET_ADDRESS');
-- Expected: Error (cannot access other users' portfolios)
