# QuantDesk Admin API Documentation

## 🔒 Admin-Only Endpoints

**WARNING**: These endpoints require admin authentication and should NEVER be exposed publicly.

### Authentication Required
- JWT token with admin role (`admin` or `super_admin`)
- Multi-factor authentication (recommended)
- IP whitelisting (optional but recommended)
- Rate limiting (100 requests/minute)

### Environment Variables
```bash
# Required for admin authentication
JWT_SECRET=your_jwt_secret_key
ADMIN_ALLOWED_IPS=192.168.1.100,10.0.0.50  # Optional IP whitelist
```

## 📋 Available Endpoints

### Contract Management

#### Get Contract Status
```http
GET /api/admin/contracts/status
Authorization: Bearer <admin_jwt_token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "programId": "C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw",
    "network": "devnet",
    "deployed": true,
    "initialized": true,
    "balance": 7.29617496,
    "lastUpdated": "2025-10-25T20:00:00Z"
  }
}
```

#### Deploy Contract (Super Admin Only)
```http
POST /api/admin/contracts/deploy
Authorization: Bearer <super_admin_jwt_token>
Content-Type: application/json

{
  "network": "devnet|testnet|mainnet",
  "programId": "optional_program_id"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "programId": "C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw",
    "network": "devnet",
    "transactionId": "mock_transaction_id_1234567890",
    "deployedAt": "2025-10-25T20:00:00Z"
  }
}
```

#### Initialize Contract (Super Admin Only)
```http
POST /api/admin/contracts/initialize
Authorization: Bearer <super_admin_jwt_token>
Content-Type: application/json

{
  "adminKey": "admin_public_key",
  "oracleProgramId": "oracle_program_id",
  "initialMarkets": [
    {
      "symbol": "SOL/USDC",
      "baseAsset": "SOL",
      "quoteAsset": "USDC",
      "maxLeverage": 20
    }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "transactionId": "mock_init_transaction_id_1234567890",
    "initializedAt": "2025-10-25T20:00:00Z",
    "adminKey": "admin_public_key",
    "oracleProgramId": "oracle_program_id",
    "initialMarkets": [...]
  }
}
```

### User Management

#### Get User Data
```http
GET /api/admin/users
Authorization: Bearer <admin_jwt_token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "totalUsers": 150,
    "activeUsers": 45,
    "users": [...]
  }
}
```

### Market Management

#### Get Market Data
```http
GET /api/admin/markets
Authorization: Bearer <admin_jwt_token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "totalMarkets": 5,
    "activeMarkets": 3,
    "markets": [...]
  }
}
```

### System Monitoring

#### Get Monitoring Data
```http
GET /api/admin/monitoring
Authorization: Bearer <admin_jwt_token>
```

**Response:**
```json
{
  "success": true,
  "data": {
    "systemHealth": "healthy",
    "uptime": "99.9%",
    "activeConnections": 25,
    "lastUpdated": "2025-10-25T20:00:00Z"
  }
}
```

## 🔐 Authentication Details

### JWT Token Structure
```json
{
  "adminId": "admin_user_id",
  "role": "admin|super_admin",
  "iat": 1640995200,
  "exp": 1641081600
}
```

### Error Responses

#### Authentication Required
```json
{
  "success": false,
  "error": "Admin authentication required",
  "code": "ADMIN_AUTH_REQUIRED"
}
```

#### Insufficient Privileges
```json
{
  "success": false,
  "error": "Super admin privileges required",
  "code": "SUPER_ADMIN_REQUIRED"
}
```

#### Rate Limit Exceeded
```json
{
  "success": false,
  "error": "Admin rate limit exceeded",
  "code": "ADMIN_RATE_LIMIT_EXCEEDED"
}
```

#### IP Not Allowed
```json
{
  "success": false,
  "error": "Access denied from this IP address",
  "code": "IP_NOT_ALLOWED"
}
```

## 🛡️ Security Features

### Authentication Layers
1. **JWT Token Validation**: Verifies token signature and expiration
2. **Role-Based Access**: Different permissions for `admin` vs `super_admin`
3. **IP Whitelisting**: Optional IP address restrictions
4. **Rate Limiting**: Prevents abuse (100 requests/minute)

### Admin Roles
- **admin**: Can view status and manage users/markets
- **super_admin**: Can deploy and initialize contracts

### Security Best Practices
- Never expose admin endpoints in public API documentation
- Use HTTPS in production
- Implement comprehensive logging
- Monitor admin access patterns
- Regular token rotation
- Multi-factor authentication for super admin

## 🚀 Deployment Notes

### Production Setup
1. Set strong `JWT_SECRET`
2. Configure `ADMIN_ALLOWED_IPS` for IP whitelisting
3. Use separate admin subdomain (e.g., `admin.quantdesk.com`)
4. Implement proper logging and monitoring
5. Set up alerting for admin access

### Development Setup
```bash
# Generate admin JWT token for testing
node -e "
const jwt = require('jsonwebtoken');
const token = jwt.sign(
  { adminId: 'dev_admin', role: 'super_admin' },
  process.env.JWT_SECRET || 'dev_secret',
  { expiresIn: '24h' }
);
console.log('Admin Token:', token);
"
```

## 📊 Monitoring and Logging

All admin actions are logged with:
- Admin ID and role
- Timestamp
- Endpoint accessed
- Request parameters (sanitized)
- Response status

This ensures full audit trail for security compliance.
