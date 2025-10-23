# Security Best Practices

## Expert-Recommended Security Measures

### 1. Principle of Least Privilege
```sql
-- Revoke overly permissive grants
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM anon;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM anon;

-- Grant selective permissions
GRANT USAGE ON SCHEMA public TO anon, authenticated;

-- Public tables (read-only for anon)
GRANT SELECT ON markets TO anon, authenticated;
GRANT SELECT ON oracle_prices TO anon, authenticated;

-- Authenticated user tables (RLS enforced)
GRANT SELECT, INSERT, UPDATE ON users TO authenticated;
GRANT SELECT, INSERT, UPDATE ON positions TO authenticated;
```

### 2. Service Role Isolation
```sql
-- Service role has full access for backend operations
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO service_role;
```

### 3. Input Validation
```typescript
// Joi validation schemas
const positionSchema = Joi.object({
  market_id: Joi.string().uuid().required(),
  side: Joi.string().valid('long', 'short').required(),
  size: Joi.number().positive().required(),
  leverage: Joi.number().min(1).max(100).required()
});

const orderSchema = Joi.object({
  market_id: Joi.string().uuid().required(),
  side: Joi.string().valid('buy', 'sell').required(),
  type: Joi.string().valid('market', 'limit', 'stop_loss').required(),
  size: Joi.number().positive().required(),
  price: Joi.number().positive().optional()
});
```
