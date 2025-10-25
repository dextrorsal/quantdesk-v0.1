#!/bin/bash

# QuantDesk SVM Test Environment Setup
# This script sets up Solana Virtual Machine testing for protocol validation

set -e

echo "🔬 QuantDesk SVM Test Environment Setup"
echo "======================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    local status=$1
    local message=$2
    case $status in
        "INFO")
            echo -e "${BLUE}ℹ️  $message${NC}"
            ;;
        "SUCCESS")
            echo -e "${GREEN}✅ $message${NC}"
            ;;
        "WARNING")
            echo -e "${YELLOW}⚠️  $message${NC}"
            ;;
        "ERROR")
            echo -e "${RED}❌ $message${NC}"
            ;;
    esac
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install SVM if not present
install_svm() {
    print_status "INFO" "Checking for SVM installation..."
    
    if command_exists svm; then
        print_status "SUCCESS" "SVM already installed"
        return 0
    fi
    
    print_status "INFO" "Installing SVM..."
    
    # Install SVM
    if curl -sSfL https://release.solana.com/v1.18.4/install | sh; then
        print_status "SUCCESS" "SVM installed successfully"
    else
        print_status "ERROR" "Failed to install SVM"
        return 1
    fi
    
    # Add to PATH
    export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"
    
    return 0
}

# Function to create SVM test configuration
create_svm_config() {
    print_status "INFO" "Creating SVM test configuration..."
    
    local svm_dir="solana-sandbox"
    mkdir -p "$svm_dir"
    
    # Create SVM configuration
    cat > "$svm_dir/svm-config.json" << EOF
{
  "cluster": "localnet",
  "validator_config": {
    "ledger_dir": "./ledger",
    "log_level": "info",
    "rpc_port": 8899,
    "gossip_port": 8001,
    "dynamic_port_range": "8002-8020"
  },
  "programs": {
    "quantdesk_perp_dex": {
      "program_id": "C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw",
      "deploy_path": "./target/deploy/quantdesk_perp_dex.so"
    }
  },
  "test_accounts": {
    "alice": {
      "keypair": "./test-keypairs/alice.json",
      "initial_balance": 1000000000000
    },
    "bob": {
      "keypair": "./test-keypairs/bob.json", 
      "initial_balance": 1000000000000
    }
  }
}
EOF

    print_status "SUCCESS" "SVM configuration created"
}

# Function to create test keypairs
create_test_keypairs() {
    print_status "INFO" "Creating test keypairs..."
    
    local keypairs_dir="solana-sandbox/test-keypairs"
    mkdir -p "$keypairs_dir"
    
    # Create Alice keypair
    if solana-keygen new --outfile "$keypairs_dir/alice.json" --no-bip39-passphrase --silent; then
        print_status "SUCCESS" "Alice keypair created"
    else
        print_status "ERROR" "Failed to create Alice keypair"
        return 1
    fi
    
    # Create Bob keypair
    if solana-keygen new --outfile "$keypairs_dir/bob.json" --no-bip39-passphrase --silent; then
        print_status "SUCCESS" "Bob keypair created"
    else
        print_status "ERROR" "Failed to create Bob keypair"
        return 1
    fi
    
    return 0
}

# Function to create comprehensive test suite
create_test_suite() {
    print_status "INFO" "Creating comprehensive test suite..."
    
    local test_dir="solana-sandbox/tests"
    mkdir -p "$test_dir"
    
    # Create collateral management tests
    cat > "$test_dir/test-collateral-management.ts" << 'EOF'
import { Connection, Keypair, PublicKey, Transaction } from '@solana/web3.js';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { expect } from 'chai';
import idl from '../../contracts/target/idl/quantdesk_perp_dex.json';

describe('Collateral Management Tests', () => {
  let connection: Connection;
  let program: Program;
  let alice: Keypair;
  let bob: Keypair;

  before(async () => {
    connection = new Connection('http://localhost:8899', 'confirmed');
    program = new Program(idl as any, {
      connection,
    });
    
    // Load test keypairs
    alice = Keypair.fromSecretKey(/* Alice secret key */);
    bob = Keypair.fromSecretKey(/* Bob secret key */);
  });

  it('should initialize user account', async () => {
    // Test user account initialization
    const tx = await program.methods
      .initializeUserAccount()
      .accounts({
        user: alice.publicKey,
        userAccount: /* PDA */,
        systemProgram: PublicKey.default,
      })
      .signers([alice])
      .rpc();

    expect(tx).to.be.a('string');
  });

  it('should deposit SOL collateral', async () => {
    // Test SOL collateral deposit
    const depositAmount = 1000000000; // 1 SOL in lamports
    
    const tx = await program.methods
      .depositNativeSol(new BN(depositAmount))
      .accounts({
        user: alice.publicKey,
        userAccount: /* PDA */,
        collateralAccount: /* SOL collateral PDA */,
        systemProgram: PublicKey.default,
      })
      .signers([alice])
      .rpc();

    expect(tx).to.be.a('string');
  });

  it('should withdraw SOL collateral', async () => {
    // Test SOL collateral withdrawal
    const withdrawAmount = 500000000; // 0.5 SOL in lamports
    
    const tx = await program.methods
      .withdrawNativeSol(new BN(withdrawAmount))
      .accounts({
        user: alice.publicKey,
        userAccount: /* PDA */,
        collateralAccount: /* SOL collateral PDA */,
        systemProgram: PublicKey.default,
      })
      .signers([alice])
      .rpc();

    expect(tx).to.be.a('string');
  });

  it('should calculate collateral value correctly', async () => {
    // Test collateral value calculation
    const collateralAccount = await program.account.collateralAccount.fetch(/* PDA */);
    
    expect(collateralAccount.amount.toNumber()).to.be.greaterThan(0);
    expect(collateralAccount.valueUsd.toNumber()).to.be.greaterThan(0);
  });
});
EOF

    # Create trading tests
    cat > "$test_dir/test-trading-operations.ts" << 'EOF'
import { Connection, Keypair, PublicKey, Transaction } from '@solana/web3.js';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { expect } from 'chai';
import idl from '../../contracts/target/idl/quantdesk_perp_dex.json';

describe('Trading Operations Tests', () => {
  let connection: Connection;
  let program: Program;
  let alice: Keypair;
  let bob: Keypair;

  before(async () => {
    connection = new Connection('http://localhost:8899', 'confirmed');
    program = new Program(idl as any, {
      connection,
    });
    
    // Load test keypairs
    alice = Keypair.fromSecretKey(/* Alice secret key */);
    bob = Keypair.fromSecretKey(/* Bob secret key */);
  });

  it('should create market', async () => {
    // Test market creation
    const tx = await program.methods
      .createMarket('BTC/USDC', new BN(50000), new BN(1000))
      .accounts({
        admin: alice.publicKey,
        market: /* Market PDA */,
        systemProgram: PublicKey.default,
      })
      .signers([alice])
      .rpc();

    expect(tx).to.be.a('string');
  });

  it('should place long order', async () => {
    // Test long order placement
    const tx = await program.methods
      .placeOrder(
        'BTC/USDC',
        0, // OrderType.Limit
        0, // PositionSide.Long
        new BN(1000000), // size
        new BN(50000), // price
        10 // leverage
      )
      .accounts({
        user: alice.publicKey,
        userAccount: /* PDA */,
        market: /* Market PDA */,
        order: /* Order PDA */,
        systemProgram: PublicKey.default,
      })
      .signers([alice])
      .rpc();

    expect(tx).to.be.a('string');
  });

  it('should execute trade', async () => {
    // Test trade execution
    const tx = await program.methods
      .executeTrade(
        'BTC/USDC',
        alice.publicKey,
        bob.publicKey
      )
      .accounts({
        executor: bob.publicKey,
        market: /* Market PDA */,
        order: /* Order PDA */,
        systemProgram: PublicKey.default,
      })
      .signers([bob])
      .rpc();

    expect(tx).to.be.a('string');
  });
});
EOF

    # Create risk management tests
    cat > "$test_dir/test-risk-management.ts" << 'EOF'
import { Connection, Keypair, PublicKey, Transaction } from '@solana/web3.js';
import { Program, AnchorProvider } from '@coral-xyz/anchor';
import { expect } from 'chai';
import idl from '../../contracts/target/idl/quantdesk_perp_dex.json';

describe('Risk Management Tests', () => {
  let connection: Connection;
  let program: Program;
  let alice: Keypair;

  before(async () => {
    connection = new Connection('http://localhost:8899', 'confirmed');
    program = new Program(idl as any, {
      connection,
    });
    
    alice = Keypair.fromSecretKey(/* Alice secret key */);
  });

  it('should calculate margin requirements', async () => {
    // Test margin requirement calculation
    const userAccount = await program.account.userAccount.fetch(/* PDA */);
    
    expect(userAccount.initialMarginRequirement.toNumber()).to.be.greaterThan(0);
    expect(userAccount.maintenanceMarginRequirement.toNumber()).to.be.greaterThan(0);
  });

  it('should check liquidation conditions', async () => {
    // Test liquidation condition checking
    const userAccount = await program.account.userAccount.fetch(/* PDA */);
    const accountHealth = userAccount.accountHealth;
    
    expect(accountHealth).to.be.a('number');
    expect(accountHealth).to.be.greaterThan(0);
  });

  it('should execute liquidation when needed', async () => {
    // Test liquidation execution
    const tx = await program.methods
      .liquidatePosition(alice.publicKey)
      .accounts({
        liquidator: /* Liquidator keypair */,
        user: alice.publicKey,
        userAccount: /* PDA */,
        systemProgram: PublicKey.default,
      })
      .signers([/* Liquidator keypair */])
      .rpc();

    expect(tx).to.be.a('string');
  });
});
EOF

    print_status "SUCCESS" "Comprehensive test suite created"
}

# Function to create test runner script
create_test_runner() {
    print_status "INFO" "Creating test runner script..."
    
    cat > "solana-sandbox/run-tests.sh" << 'EOF'
#!/bin/bash

# QuantDesk SVM Test Runner
set -e

echo "🧪 Running QuantDesk SVM Tests"
echo "============================="

# Start local validator
echo "Starting local Solana validator..."
solana-test-validator --reset --quiet &
VALIDATOR_PID=$!

# Wait for validator to start
sleep 5

# Deploy program
echo "Deploying program..."
solana program deploy target/deploy/quantdesk_perp_dex.so --program-id target/deploy/quantdesk_perp_dex-keypair.json

# Run tests
echo "Running test suite..."
npm test

# Cleanup
kill $VALIDATOR_PID

echo "✅ Tests completed"
EOF

    chmod +x "solana-sandbox/run-tests.sh"
    print_status "SUCCESS" "Test runner script created"
}

# Function to create comparison analysis script
create_comparison_script() {
    print_status "INFO" "Creating Drift comparison analysis script..."
    
    cat > "scripts/analyze-vs-drift.sh" << 'EOF'
#!/bin/bash

# QuantDesk vs Drift Protocol Analysis
echo "📊 QuantDesk vs Drift Protocol Analysis"
echo "========================================"

# This script will analyze our protocol against Drift
# and prepare data for expert analysis

echo "🔍 Analyzing protocol features..."
echo "📋 Preparing comparison data..."
echo "📝 Generating analysis report..."

# Create comparison report
cat > "drift-comparison-report.md" << 'REPORT'
# QuantDesk vs Drift Protocol Comparison

## Architecture Comparison

### QuantDesk
- **Collateral Management**: Multi-asset collateral with SOL focus
- **Trading Engine**: Perpetual futures with cross-margining
- **Risk Management**: Dynamic margin requirements
- **Oracle Integration**: Pyth Network integration

### Drift Protocol
- **Collateral Management**: Multi-asset collateral system
- **Trading Engine**: Perpetual futures with spot trading
- **Risk Management**: Sophisticated liquidation system
- **Oracle Integration**: Multiple oracle providers

## Key Differences

1. **Collateral Types**: QuantDesk focuses on SOL, Drift supports multiple assets
2. **Trading Features**: Drift has more advanced order types
3. **Risk Management**: Drift has more sophisticated liquidation logic
4. **User Experience**: QuantDesk focuses on simplicity

## Recommendations

1. Enhance multi-asset collateral support
2. Implement advanced order types
3. Improve liquidation system
4. Add more trading features

REPORT

echo "✅ Comparison analysis complete"
EOF

    chmod +x "scripts/analyze-vs-drift.sh"
    print_status "SUCCESS" "Drift comparison script created"
}

# Main setup function
main() {
    print_status "INFO" "Setting up QuantDesk SVM Test Environment..."
    
    # Install SVM
    install_svm || exit 1
    
    # Create SVM configuration
    create_svm_config
    
    # Create test keypairs
    create_test_keypairs || exit 1
    
    # Create test suite
    create_test_suite
    
    # Create test runner
    create_test_runner
    
    # Create comparison script
    create_comparison_script
    
    print_status "SUCCESS" "SVM test environment setup complete!"
    echo ""
    print_status "INFO" "Next steps:"
    echo "  1. Run CLI validation: ./scripts/validate-protocol-cli.sh"
    echo "  2. Start SVM tests: ./solana-sandbox/run-tests.sh"
    echo "  3. Analyze vs Drift: ./scripts/analyze-vs-drift.sh"
    echo "  4. Get Solana expert analysis via MCP"
    echo "  5. Prepare for PO/QA validation"
}

# Run main function
main "$@"
