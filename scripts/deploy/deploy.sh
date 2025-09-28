#!/bin/bash

# QuantDesk Frontend Deployment Script
# This script helps deploy your frontend to various platforms

echo "🚀 QuantDesk Frontend Deployment Script"
echo "======================================"

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "❌ Error: Please run this script from the frontend directory"
    exit 1
fi

# Build the project
echo "📦 Building frontend..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed!"
    exit 1
fi

echo "✅ Build successful!"

# Check if dist directory exists
if [ ! -d "dist" ]; then
    echo "❌ Error: dist directory not found"
    exit 1
fi

echo "📁 Build output ready in ./dist directory"
echo ""
echo "🌐 Deployment Options:"
echo "1. Vercel (Recommended)"
echo "2. Netlify"
echo "3. Cloudflare Pages"
echo "4. Manual upload to server"
echo ""

read -p "Choose deployment option (1-4): " choice

case $choice in
    1)
        echo "🚀 Deploying to Vercel..."
        if command -v vercel &> /dev/null; then
            vercel --prod
        else
            echo "❌ Vercel CLI not installed. Install with: npm install -g vercel"
            echo "📋 Manual steps:"
            echo "   1. Install Vercel CLI: npm install -g vercel"
            echo "   2. Run: vercel --prod"
            echo "   3. Add your domain in Vercel dashboard"
        fi
        ;;
    2)
        echo "🚀 Deploying to Netlify..."
        if command -v netlify &> /dev/null; then
            netlify deploy --prod --dir=dist
        else
            echo "❌ Netlify CLI not installed. Install with: npm install -g netlify-cli"
            echo "📋 Manual steps:"
            echo "   1. Install Netlify CLI: npm install -g netlify-cli"
            echo "   2. Run: netlify deploy --prod --dir=dist"
            echo "   3. Add your domain in Netlify dashboard"
        fi
        ;;
    3)
        echo "🌐 Cloudflare Pages deployment:"
        echo "📋 Manual steps:"
        echo "   1. Go to https://pages.cloudflare.com/"
        echo "   2. Connect your GitHub repository"
        echo "   3. Build command: npm run build"
        echo "   4. Build output directory: dist"
        echo "   5. Add your custom domain"
        ;;
    4)
        echo "📁 Manual deployment:"
        echo "Your built files are in ./dist directory"
        echo "Upload these files to your web server's public directory"
        echo ""
        echo "📋 Server requirements:"
        echo "   - Web server (nginx, Apache, etc.)"
        echo "   - SSL certificate (Let's Encrypt recommended)"
        echo "   - Domain pointing to your server"
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

echo ""
echo "🎉 Deployment process initiated!"
echo "📝 Next steps:"
echo "   1. Configure your domain's DNS settings"
echo "   2. Wait for SSL certificate to be issued"
echo "   3. Test your deployed application"
echo ""
echo "🔧 DNS Configuration:"
echo "   - Point your domain to the deployment platform's servers"
echo "   - Vercel: Add CNAME record pointing to cname.vercel-dns.com"
echo "   - Netlify: Add CNAME record pointing to your-site.netlify.app"
echo "   - Cloudflare: Follow their domain setup instructions"
