#!/bin/bash

# QuantDesk Development Environment Setup
echo "🚀 Setting up QuantDesk development environment..."

# Check if we're in the right directory
if [ ! -f "package.json" ] && [ ! -f "frontend/package.json" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Frontend setup
echo "📦 Setting up frontend dependencies..."
cd frontend
npm install
cd ..

# Check for required tools
echo "🔍 Checking required tools..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+ from https://nodejs.org/"
    exit 1
else
    echo "✅ Node.js $(node --version) found"
fi

# Check Rust
if ! command -v rustc &> /dev/null; then
    echo "⚠️  Rust not found. Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
    echo "✅ Rust installed"
else
    echo "✅ Rust $(rustc --version) found"
fi

# Check Solana CLI
if ! command -v solana &> /dev/null; then
    echo "⚠️  Solana CLI not found. Installing..."
    sh -c "$(curl -sSfL https://release.solana.com/v1.17.0/install)"
    export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"
    echo "✅ Solana CLI installed"
else
    echo "✅ Solana CLI $(solana --version) found"
fi

# Check Anchor
if ! command -v anchor &> /dev/null; then
    echo "⚠️  Anchor not found. Installing..."
    cargo install --git https://github.com/coral-xyz/anchor avm --locked --force
    avm install latest
    avm use latest
    echo "✅ Anchor installed"
else
    echo "✅ Anchor $(anchor --version) found"
fi

# Create environment files
echo "📝 Creating environment files..."

# Frontend .env
cat > frontend/.env.local << EOF
# Frontend Environment Variables
VITE_SOLANA_NETWORK=devnet
VITE_RPC_URL=https://api.devnet.solana.com
VITE_WS_URL=wss://api.devnet.solana.com
EOF

# Backend .env (for future use)
cat > backend/.env << EOF
# Backend Environment Variables
NODE_ENV=development
PORT=3001
SOLANA_NETWORK=devnet
RPC_URL=https://api.devnet.solana.com
WS_URL=wss://api.devnet.solana.com
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
EOF

# Contracts .env (for future use)
cat > contracts/.env << EOF
# Smart Contract Environment Variables
ANCHOR_PROVIDER_URL=https://api.devnet.solana.com
ANCHOR_WALLET=~/.config/solana/id.json
EOF

echo "✅ Environment files created"

# Create development scripts
echo "📜 Creating development scripts..."

cat > dev-frontend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting frontend development server..."
cd frontend && npm run dev
EOF

cat > dev-backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting backend development server..."
cd backend && npm run dev
EOF

cat > dev-contracts.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting local Solana validator..."
solana-test-validator --reset
EOF

chmod +x dev-*.sh

echo "✅ Development scripts created"

echo ""
echo "🎉 Setup complete! You can now:"
echo ""
echo "Frontend:"
echo "  ./dev-frontend.sh     # Start frontend dev server"
echo "  cd frontend && npm run dev"
echo ""
echo "Backend (when ready):"
echo "  ./dev-backend.sh      # Start backend dev server"
echo ""
echo "Smart Contracts (when ready):"
echo "  ./dev-contracts.sh    # Start local Solana validator"
echo ""
echo "Environment files created:"
echo "  frontend/.env.local"
echo "  backend/.env"
echo "  contracts/.env"
echo ""
echo "Happy coding! 🚀"
