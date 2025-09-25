# 🔑 Wallet Utilities

Core wallet management utilities and components for the D3X7-ALGO platform. This folder contains the foundational wallet management system that powers the D3X7-ALGO CLI wallet commands.

## 📁 Directory Structure

```
src/utils/wallet/
├── encryption.py          # Wallet encryption utilities
├── phantom_utils.py       # Phantom wallet compatibility tools
├── price_service.py       # Token price fetching service
├── sol_rpc.py            # Solana RPC connection management
├── sol_wallet.py         # Base Solana wallet implementation
├── wallet_commands.py     # Legacy wallet command implementation
├── wallet_manager.py      # Multi-wallet management system
├── wallet_migration.py    # Wallet configuration migration tools
└── watch_mode_gui.py      # Real-time wallet monitoring GUI
```

## 🔧 Core Components

### WalletManager (`wallet_manager.py`)
- Core wallet management system
- Handles wallet creation, loading, and deletion
- Manages wallet configurations and persistence
- Used by the D3X7-ALGO CLI for wallet operations

### SolanaWallet (`sol_wallet.py`)
- Base Solana wallet implementation
- Handles keypair management
- Provides transaction signing capabilities
- Supports SOL and SPL token operations

### Encryption (`encryption.py`)
- Secure wallet data encryption
- Password-based encryption for wallet configs
- Key derivation and secure storage
- Used to protect sensitive wallet data

### RPC Management (`sol_rpc.py`)
- Manages Solana RPC connections
- Handles network switching (mainnet/devnet/testnet)
- Connection pooling and retry logic
- RPC endpoint configuration

## 🛠 Supporting Utilities

### Price Service (`price_service.py`)
- Real-time token price fetching
- Supports multiple currencies (USD/CAD)
- Price caching and update intervals
- Used for balance display and monitoring

### Phantom Utils (`phantom_utils.py`)
- Phantom wallet compatibility layer
- Keypair format conversion
- Seed phrase generation
- Private key extraction

### Watch Mode GUI (`watch_mode_gui.py`)
- Real-time wallet monitoring interface
- Balance updates and price tracking
- Transaction history display
- Network status indicators

### Wallet Migration (`wallet_migration.py`)
- Configuration migration tools
- Backup and restore functionality
- Version compatibility handling
- Safe upgrade paths

## 🔒 Security Features

- Encrypted wallet storage
- Secure memory handling
- Password-based encryption
- Secure RPC connections
- Protected config files

## 🔄 Integration

These utilities are used by the D3X7-ALGO CLI (`d3x7 wallet` commands) and other platform components. For CLI usage, please refer to the main D3X7-ALGO documentation.

### Example Integration
```python
from src.utils.wallet.wallet_manager import WalletManager
from src.utils.wallet.encryption import WalletEncryption

# Initialize wallet manager
wallet_manager = WalletManager()

# Create encrypted wallet
encryption = WalletEncryption()
wallet = wallet_manager.create_wallet("MAIN", keypair_path, encryption)
```

## 📚 Development Notes

### Adding New Features
1. Implement in appropriate utility module
2. Update WalletManager if needed
3. Add to CLI through src/cli/wallet/
4. Update tests and documentation

### Security Guidelines
- Always use encryption for sensitive data
- Implement secure memory handling
- Follow Solana security best practices
- Use secure RPC connections

## 🔗 Dependencies

- `solana-py`: Solana Python SDK
- `pycryptodome`: Encryption utilities
- `base58`: Address encoding
- `rich`: Console formatting
- `click`: Command line interface (legacy)

## 🤝 Contributing

When contributing to these utilities:
1. Follow security best practices
2. Add tests for new features
3. Update documentation
4. Maintain backward compatibility

## ⚠️ Legacy Notice

The `wallet_commands.py` file contains the legacy CLI implementation. For new development, use the D3X7-ALGO CLI in `src/cli/wallet/`.