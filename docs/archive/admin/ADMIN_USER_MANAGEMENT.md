# 🔐 QuantDesk Admin User Management

## Overview

The QuantDesk admin user management system provides secure authentication and role-based access control for the admin dashboard. It integrates with Supabase for data persistence and includes comprehensive audit logging.

## Architecture

### Database Schema

#### `admin_users` Table
```sql
- id: UUID (Primary Key)
- username: VARCHAR(50) (Unique)
- password_hash: VARCHAR(255)
- role: VARCHAR(20) (founding-dev, admin, super-admin)
- permissions: JSONB
- is_active: BOOLEAN
- last_login: TIMESTAMP
- created_at: TIMESTAMP
- updated_at: TIMESTAMP
- created_by: UUID (Foreign Key)
```

#### `admin_audit_logs` Table
```sql
- id: UUID (Primary Key)
- admin_user_id: UUID (Foreign Key)
- action: VARCHAR(50)
- resource: VARCHAR(100)
- details: JSONB
- ip_address: INET
- user_agent: TEXT
- created_at: TIMESTAMP
```

### Security Features

- **Password Hashing**: bcrypt with salt rounds
- **JWT Authentication**: 24-hour token expiration
- **Row Level Security**: Supabase RLS policies
- **Audit Logging**: Complete action tracking
- **Role-Based Access**: Granular permissions system

## API Endpoints

### Authentication

#### `POST /api/admin/login`
Login with username and password.

**Request:**
```json
{
  "username": "string",
  "password": "string",
  "twoFactorCode": "string" // Optional for super-admin
}
```

**Response:**
```json
{
  "success": true,
  "token": "jwt-token",
  "user": {
    "id": "uuid",
    "username": "string",
    "role": "string",
    "permissions": ["array"]
  }
}
```

#### `GET /api/admin/verify`
Verify JWT token and get user info.

**Headers:**
```
Authorization: Bearer <jwt-token>
```

**Response:**
```json
{
  "success": true,
  "user": {
    "id": "uuid",
    "username": "string",
    "role": "string",
    "permissions": ["array"]
  }
}
```

### User Management

#### `GET /api/admin/users`
Get all admin users (requires authentication).

**Response:**
```json
{
  "success": true,
  "users": [
    {
      "id": "uuid",
      "username": "string",
      "role": "string",
      "permissions": ["array"],
      "is_active": boolean,
      "last_login": "timestamp",
      "created_at": "timestamp"
    }
  ]
}
```

#### `POST /api/admin/users`
Create new admin user (requires authentication).

**Request:**
```json
{
  "username": "string",
  "password": "string",
  "role": "founding-dev|admin|super-admin",
  "permissions": ["array"] // Optional
}
```

#### `PUT /api/admin/users/:id`
Update admin user (requires authentication).

**Request:**
```json
{
  "username": "string", // Optional
  "role": "string", // Optional
  "permissions": ["array"], // Optional
  "is_active": boolean // Optional
}
```

#### `DELETE /api/admin/users/:id`
Deactivate admin user (soft delete, requires authentication).

**Response:**
```json
{
  "success": true,
  "message": "User deactivated successfully"
}
```

### Audit Logs

#### `GET /api/admin/audit-logs`
Get audit logs with pagination.

**Query Parameters:**
- `limit`: Number of logs to return (default: 100)
- `offset`: Number of logs to skip (default: 0)

**Response:**
```json
{
  "success": true,
  "logs": [
    {
      "id": "uuid",
      "action": "string",
      "resource": "string",
      "details": {},
      "ip_address": "string",
      "user_agent": "string",
      "created_at": "timestamp",
      "admin_users": {
        "username": "string",
        "role": "string"
      }
    }
  ]
}
```

## Role System

### Roles

1. **founding-dev**
   - Full system access
   - Can manage all users
   - Access to all features
   - No 2FA requirement

2. **admin**
   - Standard admin access
   - Can manage users (except founding-dev)
   - Access to most features
   - No 2FA requirement

3. **super-admin**
   - Advanced admin access
   - Can manage users (except founding-dev)
   - Access to sensitive features
   - **Requires 2FA**

### Permissions

- `read`: View data and logs
- `write`: Modify data and settings
- `admin`: User management
- `super-admin`: Advanced system operations
- `founding-dev`: Complete system control

## Setup Instructions

### 1. Database Migration

Run the SQL migration in your Supabase database:

```sql
-- Copy and execute the contents of:
-- backend/src/migrations/create_admin_users_table.sql
```

### 2. Environment Variables

Ensure these environment variables are set:

```env
SUPABASE_URL=your-supabase-url
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
JWT_SECRET=your-jwt-secret
```

### 3. Initial User Setup

Run the setup script to create initial admin users:

```bash
cd backend
node src/scripts/setup-admin-users.js
```

### 4. Backend Dependencies

Install required packages:

```bash
npm install bcrypt jsonwebtoken @types/bcrypt @types/jsonwebtoken
```

## Frontend Integration

### Authentication Context

The admin dashboard uses React Context for authentication state management:

```typescript
// AuthContext provides:
- user: Current user info
- login: Login function
- logout: Logout function
- isAuthenticated: Auth status
```

### Protected Routes

Admin routes are protected using the `ProtectedRoute` component:

```typescript
<ProtectedRoute>
  <AdminDashboard />
</ProtectedRoute>
```

### Login Page

The login page handles:
- Username/password authentication
- 2FA for super-admin users
- Error handling and validation
- Token storage in localStorage

## Security Best Practices

### Password Security
- Minimum 8 characters
- bcrypt hashing with salt rounds
- No password storage in plain text

### Token Security
- JWT tokens with 24-hour expiration
- Secure token storage in localStorage
- Automatic token refresh on app load

### Access Control
- Role-based permissions
- Row Level Security in database
- API endpoint protection
- Audit logging for all actions

### Network Security
- HTTPS required in production
- IP address logging
- User agent tracking
- Rate limiting (recommended)

## Monitoring and Logging

### Audit Trail
- All admin actions are logged
- Includes user, action, resource, and metadata
- IP address and user agent tracking
- Timestamp for all events

### Security Monitoring
- Failed login attempts
- Permission violations
- Suspicious activity patterns
- System access logs

## Troubleshooting

### Common Issues

1. **Login Failures**
   - Check username/password
   - Verify user is active
   - Check database connection

2. **Token Issues**
   - Verify JWT_SECRET is set
   - Check token expiration
   - Clear localStorage if needed

3. **Permission Errors**
   - Verify user role and permissions
   - Check API endpoint protection
   - Review audit logs

### Debug Mode

Enable debug logging by setting:
```env
LOG_LEVEL=debug
```

## API Error Codes

- `INVALID_CREDENTIALS`: Wrong username/password
- `NO_TOKEN`: Missing authorization header
- `INVALID_TOKEN`: Invalid or expired token
- `MISSING_FIELDS`: Required fields not provided
- `CREATE_USER_ERROR`: User creation failed
- `UPDATE_USER_ERROR`: User update failed
- `DELETE_USER_ERROR`: User deletion failed
- `FETCH_USERS_ERROR`: Failed to fetch users
- `FETCH_LOGS_ERROR`: Failed to fetch audit logs

## Future Enhancements

- [ ] Two-factor authentication (TOTP)
- [ ] Password reset functionality
- [ ] Session management
- [ ] API rate limiting
- [ ] Advanced audit reporting
- [ ] User activity dashboards
- [ ] Automated security alerts
- [ ] Integration with external identity providers
