# Implementation References

## Quick Reference Guide

### Database Security Files
- **RLS Policies**: `database/security-audit-fixes.sql`
- **Security Tests**: `database/security-verification-tests.sql`
- **Schema Updates**: `database/production-schema.sql`

### Expert Documentation
- **Implementation Guide**: `docs/security/solana-expert-implementation-guide.md`
- **Expert Recommendations**: `docs/security/solana-expert-recommendations.md`
- **Consultation Log**: `docs/security/expert-consultation-log.md`

### Admin Dashboard Files
- **Main Dashboard**: `frontend/src/pages/admin/AdminDashboard.tsx`
- **Login Page**: `frontend/src/pages/admin/LoginPage.tsx`
- **Dashboard Home**: `frontend/src/pages/admin/DashboardHome.tsx`
- **System Stats**: `frontend/src/pages/admin/SystemStats.tsx`
- **User Management**: `frontend/src/pages/admin/UserManagement.tsx`
- **Trading Metrics**: `frontend/src/pages/admin/TradingMetrics.tsx`
- **System Logs**: `frontend/src/pages/admin/SystemLogs.tsx`
- **Settings**: `frontend/src/pages/admin/Settings.tsx`

### Backend Services
- **Admin Auth**: `backend/src/middleware/adminAuth.ts`
- **Admin Routes**: `backend/src/routes/admin.ts`
- **AI Service**: `backend/src/services/aiService.ts`
- **AI Routes**: `backend/src/routes/ai.ts`

### Configuration Files
- **Workspace**: `pnpm-workspace.yaml`
- **Railway**: `railway.yaml`
- **Vercel**: `frontend/vercel.json`
- **Environment**: `ENVIRONMENT_SETUP.md`

## Implementation Checklist

### Phase 1: Database Security ✅
- [x] Apply RLS policies to all sensitive tables
- [x] Create secure views for public data
- [x] Implement service role isolation
- [x] Run security verification tests

### Phase 2: Admin Dashboard ✅
- [x] Implement GitHub OAuth
- [x] Implement Google OAuth
- [x] Create admin dashboard pages
- [x] Integrate with main frontend

### Phase 3: AI Service Bridge ✅
- [x] Create AI service adapter
- [x] Implement AI proxy routes
- [x] Add health checks
- [x] Configure service communication

### Phase 4: Deployment Configuration ✅
- [x] Configure pnpm workspace
- [x] Update Railway configuration
- [x] Update Vercel configuration
- [x] Create environment setup guide

## Next Steps

### Immediate Actions Required
1. **Set up OAuth credentials** (Google & GitHub)
2. **Apply database security fixes** using `database/security-audit-fixes.sql`
3. **Configure environment variables** using `ENVIRONMENT_SETUP.md`
4. **Test admin dashboard** at `/admin` route

### Future Implementations
1. **Anchor Event Implementation** - Add events to smart contracts
2. **Event Listener Service** - Implement off-chain event processing
3. **Performance Optimization** - Add database indexes and caching
4. **Monitoring Setup** - Implement comprehensive monitoring

## Expert Consultation Commands

### Solana Expert Consultation
```bash
# Use MCP tool: Solana Expert: Ask For Help
# Question format: "In a Solana perpetual DEX using Anchor, [specific question about architecture, data separation, or best practices]"
```

### Anchor Framework Consultation
```bash
# Use MCP tool: Ask Solana Anchor Framework Expert
# Question format: "In our Anchor perpetual DEX with [specific accounts], [question about events, synchronization, or RLS policies]"
```

### Documentation Search
```bash
# Use MCP tool: Solana Documentation Search
# Query format: "perpetual DEX event-driven synchronization PostgreSQL RLS policies"
```

## Security Verification Commands

### Test RLS Policies
```sql
-- Run security verification tests
\i database/security-verification-tests.sql

-- Test as different roles
SET ROLE anon;
SELECT * FROM admin_users; -- Should fail

SET ROLE authenticated;
SELECT * FROM positions; -- Should only show user's own positions

SET ROLE service_role;
SELECT * FROM admin_users; -- Should work
```

### Verify Admin Access
```bash
# Test admin dashboard access
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  http://localhost:3002/api/admin/stats

# Test OAuth flows
curl http://localhost:3002/api/admin/auth/google
curl http://localhost:3002/api/admin/auth/github
```

## Troubleshooting Guide

### Common Issues

#### RLS Policy Errors
```sql
-- Check if RLS is enabled
SELECT tablename, rowsecurity FROM pg_tables WHERE schemaname = 'public';

-- Check policies
SELECT schemaname, tablename, policyname, permissive, roles, cmd, qual 
FROM pg_policies WHERE schemaname = 'public';
```

#### OAuth Configuration Issues
```bash
# Check environment variables
echo $GOOGLE_CLIENT_ID
echo $GITHUB_CLIENT_ID
echo $ADMIN_JWT_SECRET

# Test OAuth endpoints
curl -v http://localhost:3002/api/admin/auth/google
curl -v http://localhost:3002/api/admin/auth/github
```

#### AI Service Connection Issues
```bash
# Check AI service health
curl http://localhost:3000/health

# Test AI proxy
curl -H "Authorization: Bearer $USER_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query"}' \
  http://localhost:3002/api/ai/query
```

## Performance Monitoring

### Database Performance
```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables 
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Application Performance
```bash
# Check backend health
curl http://localhost:3002/health

# Check AI service status
curl http://localhost:3002/api/ai/health

# Monitor logs
tail -f logs/backend-dev.log
tail -f logs/MIKEY-AI.log
```

## Security Monitoring

### Admin Access Monitoring
```sql
-- Check admin login attempts
SELECT * FROM admin_audit_logs 
WHERE action = 'LOGIN' 
ORDER BY created_at DESC 
LIMIT 10;

-- Check failed admin logins
SELECT * FROM admin_audit_logs 
WHERE action = 'LOGIN_FAILED' 
ORDER BY created_at DESC 
LIMIT 10;
```

### User Activity Monitoring
```sql
-- Check user trading activity
SELECT COUNT(*) as trades, SUM(size * price) as volume
FROM trades 
WHERE created_at >= NOW() - INTERVAL '24 hours';

-- Check position changes
SELECT COUNT(*) as position_updates
FROM positions 
WHERE updated_at >= NOW() - INTERVAL '1 hour';
```

## Backup and Recovery

### Database Backup
```bash
# Create database backup
pg_dump -h localhost -U postgres -d quantdesk > backup_$(date +%Y%m%d_%H%M%S).sql

# Restore database
psql -h localhost -U postgres -d quantdesk < backup_file.sql
```

### Configuration Backup
```bash
# Backup configuration files
tar -czf config_backup_$(date +%Y%m%d_%H%M%S).tar.gz \
  pnpm-workspace.yaml \
  railway.yaml \
  frontend/vercel.json \
  .env.example \
  ENVIRONMENT_SETUP.md
```

## Maintenance Schedule

### Daily Tasks
- [ ] Check system health endpoints
- [ ] Monitor error logs
- [ ] Verify RLS policies are working
- [ ] Check admin access logs

### Weekly Tasks
- [ ] Review performance metrics
- [ ] Check database query performance
- [ ] Verify backup integrity
- [ ] Update security documentation

### Monthly Tasks
- [ ] Security audit review
- [ ] Performance optimization review
- [ ] Expert consultation planning
- [ ] Documentation updates

## Support Contacts

### Internal Resources
- **Documentation**: `docs/security/`
- **Implementation Guide**: `docs/security/solana-expert-implementation-guide.md`
- **Expert Log**: `docs/security/expert-consultation-log.md`

### External Resources
- **Solana Docs**: https://docs.solana.com/
- **Anchor Docs**: https://www.anchor-lang.com/
- **Supabase Docs**: https://supabase.com/docs
- **PostgreSQL Docs**: https://www.postgresql.org/docs/

### Expert Consultation
- **MCP Solana Expert**: Available via Cursor
- **MCP Anchor Expert**: Available via Cursor
- **MCP Documentation Search**: Available via Cursor
