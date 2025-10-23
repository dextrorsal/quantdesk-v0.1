# OAuth Setup Instructions

## Google OAuth Setup

### Step 1: Create Google Cloud Project
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter project name: `QuantDesk Admin`
4. Click "Create"

### Step 2: Enable Google+ API
1. In the Google Cloud Console, go to "APIs & Services" → "Library"
2. Search for "Google+ API"
3. Click on it and press "Enable"

### Step 3: Create OAuth 2.0 Credentials
1. Go to "APIs & Services" → "Credentials"
2. Click "Create Credentials" → "OAuth 2.0 Client ID"
3. If prompted, configure the OAuth consent screen:
   - User Type: External
   - App name: `QuantDesk Admin Dashboard`
   - User support email: your email
   - Developer contact: your email
   - Save and continue through the steps

### Step 4: Configure OAuth Client
1. Application type: Web application
2. Name: `QuantDesk Admin Dashboard`
3. Authorized redirect URIs:
   - Development: `http://localhost:3002/api/admin/auth/google/callback`
   - Production: `https://your-backend-domain.com/api/admin/auth/google/callback`
4. Click "Create"

### Step 5: Get Credentials
- Copy the **Client ID** and **Client Secret**
- Add them to your `.env` file:
```bash
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here
```

---

## GitHub OAuth Setup

### Step 1: Create GitHub OAuth App
1. Go to [GitHub Developer Settings](https://github.com/settings/developers)
2. Click "New OAuth App"

### Step 2: Configure OAuth App
1. **Application name**: `QuantDesk Admin Dashboard`
2. **Homepage URL**: 
   - Development: `http://localhost:3001`
   - Production: `https://your-frontend-domain.com`
3. **Application description**: `Admin dashboard for QuantDesk perpetual DEX`
4. **Authorization callback URL**:
   - Development: `http://localhost:3002/api/admin/auth/github/callback`
   - Production: `https://your-backend-domain.com/api/admin/auth/github/callback`
5. Click "Register application"

### Step 3: Get Credentials
- Copy the **Client ID** and **Client Secret**
- Add them to your `.env` file:
```bash
GITHUB_CLIENT_ID=your_github_client_id_here
GITHUB_CLIENT_SECRET=your_github_client_secret_here
```

---

## Environment Variables to Add

Add these to your existing `.env` file:

```bash
# OAuth Configuration (Admin)
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here
GITHUB_CLIENT_ID=your_github_client_id_here
GITHUB_CLIENT_SECRET=your_github_client_secret_here

# Admin JWT Secret (generate a strong secret)
ADMIN_JWT_SECRET=your_strong_admin_jwt_secret_here

# Backend URL for OAuth redirects
BACKEND_URL=http://localhost:3002
FRONTEND_URL=http://localhost:3001

# AI Service URL
MIKEY_AI_URL=http://localhost:3000
```

---

## Testing OAuth Setup

### 1. Start the Backend
```bash
cd backend
pnpm run dev
```

### 2. Test OAuth Endpoints
```bash
# Test Google OAuth (should redirect to Google)
curl -v http://localhost:3002/api/admin/auth/google

# Test GitHub OAuth (should redirect to GitHub)
curl -v http://localhost:3002/api/admin/auth/github
```

### 3. Test Admin Dashboard
1. Navigate to `http://localhost:3001/admin`
2. Click "Continue with Google" or "Continue with GitHub"
3. Complete OAuth flow
4. Should redirect back to admin dashboard with access

---

## Troubleshooting

### Common Issues

#### 1. "Invalid redirect URI" Error
- Check that the redirect URI in OAuth app matches exactly
- Ensure no trailing slashes
- Check for typos in the URL

#### 2. "Client ID not found" Error
- Verify the Client ID is correct
- Check that the OAuth app is properly configured
- Ensure the project is active

#### 3. "Redirect URI mismatch" Error
- Double-check the redirect URI in both Google/GitHub settings
- Make sure the backend URL is correct
- Check for HTTP vs HTTPS mismatches

#### 4. OAuth Flow Completes but No Access
- Check that your email is in the `admin_users` table
- Verify the admin user has `is_active = true`
- Check backend logs for errors

### Debug Steps

1. **Check Environment Variables**:
```bash
echo $GOOGLE_CLIENT_ID
echo $GITHUB_CLIENT_ID
echo $ADMIN_JWT_SECRET
```

2. **Check Backend Logs**:
```bash
tail -f logs/backend-dev.log
```

3. **Test OAuth Endpoints**:
```bash
curl -v http://localhost:3002/api/admin/auth/google
curl -v http://localhost:3002/api/admin/auth/github
```

4. **Check Database Connection**:
```bash
# Test if admin_users table exists and is accessible
psql -d your_database -c "SELECT COUNT(*) FROM admin_users;"
```

---

## Production Deployment

### Update OAuth Apps for Production

#### Google OAuth
1. Go to Google Cloud Console
2. Update OAuth client settings
3. Add production redirect URI:
   - `https://your-backend-domain.com/api/admin/auth/google/callback`

#### GitHub OAuth
1. Go to GitHub Developer Settings
2. Update OAuth app settings
3. Update callback URL:
   - `https://your-backend-domain.com/api/admin/auth/github/callback`

### Environment Variables for Production
```bash
# Production URLs
BACKEND_URL=https://your-backend-domain.com
FRONTEND_URL=https://your-frontend-domain.com

# Same OAuth credentials (they work for both dev and prod)
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
GITHUB_CLIENT_ID=your_github_client_id
GITHUB_CLIENT_SECRET=your_github_client_secret
```

---

## Security Notes

1. **Never commit OAuth secrets** to version control
2. **Use strong JWT secrets** (at least 32 characters)
3. **Rotate OAuth secrets** regularly
4. **Monitor admin access logs** for suspicious activity
5. **Use HTTPS** in production
6. **Implement IP whitelisting** for admin access in production

---

## Next Steps After OAuth Setup

1. ✅ **OAuth Setup Complete**
2. ⏳ **Apply Database Security Fixes** (next step)
3. ⏳ **Test Admin Dashboard**
4. ⏳ **Deploy to Production**

Once OAuth is set up, we'll proceed with applying the database security fixes!
