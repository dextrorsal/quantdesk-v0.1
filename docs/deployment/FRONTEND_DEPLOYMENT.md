# 🌐 QuantDesk Frontend Deployment Guide

## Quick Start

Your frontend is already built and ready for deployment! Run the deployment script:

```bash
cd frontend
./deploy.sh
```

## 🚀 Deployment Options

### Option 1: Vercel (Recommended)

**Why Vercel?**
- ✅ Automatic SSL certificates
- ✅ Global CDN for fast loading
- ✅ Automatic deployments from Git
- ✅ Free tier available
- ✅ Perfect for React apps

**Steps:**
1. Install Vercel CLI:
   ```bash
   npm install -g vercel
   ```

2. Deploy:
   ```bash
   cd frontend
   vercel --prod
   ```

3. Add your custom domain:
   - Go to [Vercel Dashboard](https://vercel.com/dashboard)
   - Select your project
   - Go to Settings → Domains
   - Add your purchased domain
   - Vercel automatically handles SSL

**DNS Configuration:**
- Add a CNAME record: `www` → `cname.vercel-dns.com`
- Add an A record: `@` → `76.76.19.61` (Vercel's IP)

### Option 2: Netlify

**Why Netlify?**
- ✅ Excellent performance
- ✅ Form handling
- ✅ Serverless functions
- ✅ Free tier available

**Steps:**
1. Install Netlify CLI:
   ```bash
   npm install -g netlify-cli
   ```

2. Deploy:
   ```bash
   cd frontend
   netlify deploy --prod --dir=dist
   ```

3. Add custom domain in Netlify dashboard

### Option 3: Cloudflare Pages

**Why Cloudflare?**
- ✅ Free tier
- ✅ Excellent DDoS protection
- ✅ Global CDN
- ✅ Built-in analytics

**Steps:**
1. Go to [Cloudflare Pages](https://pages.cloudflare.com/)
2. Connect your GitHub repository
3. Build settings:
   - Build command: `npm run build`
   - Build output directory: `dist`
4. Add your custom domain

### Option 4: Traditional VPS/Server

**For full control:**

1. **Upload files to your server:**
   ```bash
   # Copy dist folder to your server
   scp -r dist/* user@your-server:/var/www/html/
   ```

2. **Configure web server (nginx example):**
   ```nginx
   server {
       listen 80;
       server_name yourdomain.com www.yourdomain.com;
       root /var/www/html;
       index index.html;

       location / {
           try_files $uri $uri/ /index.html;
       }
   }
   ```

3. **Enable SSL with Let's Encrypt:**
   ```bash
   sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
   ```

## 🔧 Environment Configuration

### For Production Deployment

Create a `.env.production` file in your frontend directory:

```bash
# API Configuration
VITE_API_URL=https://your-backend-domain.com/api
VITE_WS_URL=wss://your-backend-domain.com/ws

# Supabase Configuration
VITE_SUPABASE_URL=https://your-project.supabase.co
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key

# Solana Configuration
VITE_SOLANA_RPC_URL=https://api.mainnet-beta.solana.com
VITE_SOLANA_NETWORK=mainnet-beta

# Debug (disable in production)
VITE_DEBUG=false
```

### Update Vite Config for Production

Update your `vite.config.ts`:

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  define: {
    global: 'globalThis',
    'process.env': 'import.meta.env',
  },
  envPrefix: 'VITE_',
  server: {
    port: 3001,
    host: true,
    proxy: {
      '/api': {
        target: process.env.NODE_ENV === 'production' 
          ? 'https://your-backend-domain.com' 
          : 'http://localhost:3002',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false, // Disable sourcemaps in production
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          solana: ['@solana/web3.js', '@solana/wallet-adapter-base'],
          charts: ['lightweight-charts', 'recharts'],
        },
      },
    },
  },
  optimizeDeps: {
    include: ['buffer'],
  },
})
```

## 🌐 DNS Configuration

### For Vercel
```
Type: CNAME
Name: www
Value: cname.vercel-dns.com

Type: A
Name: @
Value: 76.76.19.61
```

### For Netlify
```
Type: CNAME
Name: www
Value: your-site-name.netlify.app

Type: A
Name: @
Value: 75.2.60.5
```

### For Cloudflare Pages
```
Type: CNAME
Name: www
Value: your-site.pages.dev

Type: A
Name: @
Value: 192.0.2.1
```

## 🔒 SSL Certificate Setup

### Automatic (Recommended)
- Vercel, Netlify, and Cloudflare automatically provide SSL certificates
- Just add your domain and SSL is handled automatically

### Manual (For VPS)
```bash
# Install Certbot
sudo apt install certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## 📊 Performance Optimization

### Build Optimization
```bash
# Analyze bundle size
npm install -g webpack-bundle-analyzer
npx vite-bundle-analyzer dist
```

### CDN Configuration
- Enable gzip compression
- Set proper cache headers
- Use image optimization
- Implement lazy loading

### Monitoring
- Set up Google Analytics
- Monitor Core Web Vitals
- Use Lighthouse for performance audits

## 🚨 Troubleshooting

### Common Issues

1. **Build Fails**
   ```bash
   # Clear cache and reinstall
   rm -rf node_modules package-lock.json
   npm install
   npm run build
   ```

2. **Environment Variables Not Working**
   - Ensure variables start with `VITE_`
   - Check `.env` file is in frontend directory
   - Restart development server

3. **API Calls Failing**
   - Check CORS settings on backend
   - Verify API URL in environment variables
   - Check network tab in browser dev tools

4. **Domain Not Working**
   - Wait 24-48 hours for DNS propagation
   - Check DNS records are correct
   - Verify SSL certificate is active

### Health Checks
```bash
# Check if site is accessible
curl -I https://yourdomain.com

# Check SSL certificate
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com
```

## 🎯 Next Steps After Deployment

1. **Test your deployment:**
   - Visit your domain
   - Test all functionality
   - Check mobile responsiveness

2. **Set up monitoring:**
   - Google Analytics
   - Error tracking (Sentry)
   - Uptime monitoring

3. **Optimize performance:**
   - Run Lighthouse audit
   - Optimize images
   - Enable caching

4. **Security:**
   - Enable HTTPS redirect
   - Set security headers
   - Regular security audits

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review deployment platform documentation
3. Check browser console for errors
4. Verify environment variables are set correctly

---

**🎉 Congratulations! Your QuantDesk frontend is ready for deployment!**
