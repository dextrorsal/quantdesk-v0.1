# 📚 QuantDesk Documentation Site

Professional documentation site for the QuantDesk Protocol. This provides a clean, organized interface to access all documentation files in your project.

## 🚀 Quick Start

```bash
# Start the documentation server
python serve.py

# Or specify a port
python serve.py 3000

# Or use the convenience script from project root
./start-docs.sh
```

Then open http://localhost:8080 in your browser.

## 🛑 Stop the Server

```bash
# From project root
./kill-docs.sh

# Or from docs-site directory
./kill-docs-site.sh

# Or manually kill processes on port 8080
lsof -ti:8080 | xargs kill -9
```

## 🎯 Features

- **Professional Design**: Clean, modern interface
- **Search Functionality**: Search through all documentation
- **Responsive**: Works on desktop, tablet, and mobile
- **Organized Navigation**: Easy access to all docs
- **Live Updates**: Automatically reflects changes to docs

## 📁 Structure

```
docs-site/
├── index.html          # Main documentation page
├── serve.py           # Python server script
└── README.md          # This file
```

## 🔗 Links to Your Documentation

The site provides organized access to all your existing documentation:

### Core System
- [Trading System Overview](../docs/TRADING_SYSTEM_OVERVIEW.md)
- [Project Overview](../docs/Project_Overview.md)
- [CSV Storage System](../docs/CSV_STORAGE_SYSTEM.md)

### ML & Strategies
- [ML Model Documentation](../docs/ML_MODEL.md)
- [ML Trading Strategies](../docs/ml-trading-strat.md)
- [Lag-based Strategies](../docs/lag-based.md)
- [Technical Indicators](../docs/INDICATORS.md)

### Implementation
- [Implementation Checklist](../docs/IMPLEMENTATION_CHECKLIST.md)
- [Live Trading Readiness](../docs/LIVE_TRADING_READINESS_PLAN.md)
- [Troubleshooting Guide](../docs/TROUBLESHOOTING.md)

### Web UI
- [Integrated Web UI Guide](../web_ui/README_INTEGRATED.md)
- [Integration Complete Summary](../web_ui/INTEGRATION_COMPLETE.md)

## 🎨 Customization

The documentation site is built with vanilla HTML/CSS/JS, making it easy to customize:

- **Styling**: Edit the CSS in `index.html`
- **Content**: Update the HTML structure
- **Navigation**: Modify the sidebar links
- **Search**: Enhance the search functionality

## 🌐 Deployment

For production deployment, you can:

1. **Static Hosting**: Upload to GitHub Pages, Netlify, or Vercel
2. **Custom Domain**: Point your domain to the static files
3. **CDN**: Use a CDN for faster global access

## 📝 Adding New Documentation

When you add new documentation files:

1. Add links to the sidebar in `index.html`
2. Update the "Complete Documentation" section
3. Consider adding to the search functionality

## 🔄 Maintenance

- **Regular Updates**: Keep the main page current with new features
- **Link Checking**: Ensure all documentation links work
- **Content Review**: Update performance metrics and features

This documentation site provides a professional way to showcase your sophisticated QuantDesk trading system and makes it easy for users to find the information they need.
