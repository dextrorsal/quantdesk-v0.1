# 🔒 QuantDesk Security Guide

## ⚠️ CRITICAL: Secure Your Supabase Database

### **Current Security Issues:**
1. **Database URL exposed** in codebase
2. **No Row Level Security (RLS)** enabled
3. **Public schema** allows anyone to query your database
4. **No rate limiting** on database queries
5. **No authentication** required for most endpoints

### **Immediate Actions Required:**

#### **1. Enable Row Level Security (RLS)**
In your Supabase dashboard:
1. Go to **Authentication → Policies**
2. **Enable RLS** on ALL tables:
   - `markets`
   - `users` 
   - `positions`
   - `orders`
   - `trades`
   - `oracle_prices`
   - `funding_rates`

#### **2. Create Secure Policies**
For each table, create policies like:
```sql
-- Only authenticated users can read markets
CREATE POLICY "Users can read markets" ON markets
FOR SELECT USING (auth.role() = 'authenticated');

-- Only authenticated users can read their own positions
CREATE POLICY "Users can read own positions" ON positions
FOR SELECT USING (auth.uid()::text = user_id);

-- Only authenticated users can create orders
CREATE POLICY "Users can create orders" ON orders
FOR INSERT WITH CHECK (auth.role() = 'authenticated');
```

#### **3. Secure Your Environment Variables**
1. **Never commit** `.env` files to git
2. **Use different credentials** for development/production
3. **Rotate keys** regularly
4. **Use environment-specific** Supabase projects

#### **4. Add Rate Limiting**
Your backend already has rate limiting, but ensure it's configured:
```typescript
// In your backend config
RATE_LIMIT_MAX_REQUESTS: 100, // per window
RATE_LIMIT_WINDOW_MS: 900000, // 15 minutes
```

#### **5. Add Authentication to All Endpoints**
```typescript
// Protect all sensitive endpoints
app.use('/api/positions', authMiddleware, positionRoutes);
app.use('/api/orders', authMiddleware, orderRoutes);
app.use('/api/trades', authMiddleware, tradeRoutes);
```

### **Production Security Checklist:**
- [ ] Enable RLS on all tables
- [ ] Create proper policies
- [ ] Use different Supabase project for production
- [ ] Enable CORS restrictions
- [ ] Add request validation
- [ ] Monitor database usage
- [ ] Set up alerts for unusual activity
- [ ] Use strong JWT secrets
- [ ] Enable HTTPS only
- [ ] Add request logging

### **Database Usage Protection:**
1. **Set up usage alerts** in Supabase
2. **Monitor query patterns**
3. **Implement query timeouts**
4. **Add request validation**
5. **Use connection pooling**

### **API Security:**
1. **Validate all inputs**
2. **Sanitize user data**
3. **Use prepared statements**
4. **Add request size limits**
5. **Implement proper error handling**

## 🚨 **URGENT: Do This Now**
1. **Enable RLS** on your Supabase database
2. **Create authentication policies**
3. **Remove hardcoded credentials** from code
4. **Use environment variables** for all secrets
5. **Test with authentication** required

## 📞 **Need Help?**
- Supabase RLS Docs: https://supabase.com/docs/guides/auth/row-level-security
- Security Best Practices: https://supabase.com/docs/guides/auth/security
