# QuantDesk Admin Dashboard

Terminal-style admin dashboard for managing QuantDesk trading platform with demo/live mode switching.

## 🎯 Features

- **Terminal Aesthetic**: Black background with neon green/cyan accents
- **Monospace Font**: JetBrains Mono for authentic terminal feel
- **Demo/Live Mode Toggle**: Easy switching between trading modes
- **Real-time Metrics**: Live trading statistics and system health
- **User Management**: View and manage active users
- **System Monitoring**: Database, API, and security status
- **Responsive Design**: Works on desktop and mobile

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd admin-dashboard
npm install
```

### 2. Start Development Server
```bash
npm run dev
```

### 3. Access Dashboard
- **Admin Dashboard**: http://localhost:5173/admin
- **Regular Dashboard**: http://localhost:5173/

## 🎨 Terminal Theme

### Color Palette
- **Background**: `#0a0a0a` (Terminal Black)
- **Primary**: `#00ff41` (Matrix Green)
- **Secondary**: `#00ffff` (Bright Cyan)
- **Accent**: `#0080ff` (Electric Blue)
- **Warning**: `#ffff00` (Bright Yellow)
- **Danger**: `#ff0080` (Hot Pink)

### Typography
- **Font**: JetBrains Mono
- **Weight**: 400-700
- **Letter Spacing**: 0.5px-1px
- **Text Shadow**: Neon glow effects

## 🔧 Configuration

### Environment Variables
Create `.env` file:
```env
VITE_API_URL=http://localhost:3002
VITE_WS_URL=ws://localhost:3002
VITE_ADMIN_TOKEN=your_admin_token
```

### Backend Integration
The dashboard connects to QuantDesk backend API:
- **Base URL**: `http://localhost:3002`
- **WebSocket**: `ws://localhost:3002`
- **Admin Endpoints**: `/api/admin/*`

## 📊 Dashboard Sections

### 1. System Mode Control
- Toggle between Demo and Live modes
- Real-time status indicators
- Safety confirmations

### 2. Trading Metrics
- Total trades and volume
- Active users count
- Win rate and P&L
- Real-time updates

### 3. System Health
- Database status
- API server health
- WebSocket connection
- Oracle feed status

### 4. Security Status
- Rate limiting
- Authentication
- SSL/TLS
- Firewall protection

### 5. User Management
- Recent users table
- Activity status
- Trading statistics
- Last active timestamps

## 🛠️ Development

### Project Structure
```
src/
├── pages/
│   └── admin/
│       └── AdminDashboard.tsx
├── styles/
│   ├── theme/
│   │   ├── _quantdesk-variables.scss
│   │   └── _quantdesk-theme.scss
│   └── _quantdesk-admin.scss
└── components/
    └── (reusable components)
```

### Adding New Features
1. Create component in `src/components/`
2. Add styles in `src/styles/`
3. Update routing in `src/App.tsx`
4. Test with `npm run dev`

## 🔌 API Integration

### Demo/Live Mode Toggle
```typescript
const toggleMode = async () => {
  const response = await fetch('/api/admin/mode', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: newMode })
  });
};
```

### Real-time Updates
```typescript
useEffect(() => {
  const ws = new WebSocket('ws://localhost:3002/admin');
  ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateMetrics(data);
  };
}, []);
```

## 🎯 Next Steps

1. **Backend Integration**: Connect to actual QuantDesk API
2. **Authentication**: Add admin login system
3. **Real-time Data**: WebSocket integration
4. **Advanced Features**: User management, system logs
5. **Deployment**: Production build and hosting

## 📝 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

---

**Built with ❤️ for QuantDesk Trading Platform**