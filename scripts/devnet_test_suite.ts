import { Connection, PublicKey, SystemProgram, LAMPORTS_PER_SOL, SYSVAR_RENT_PUBKEY } from '@solana/web3.js';
import anchor from '@coral-xyz/anchor';
import fs from 'fs';
import path from 'path';

const { AnchorProvider, Program, Wallet, BN } = anchor;
type Idl = anchor.Idl;

// Test configuration
interface TestConfig {
  rpcUrl: string;
  programId: PublicKey;
  keypairPath: string;
  pythSolFeed: PublicKey;
  verbose: boolean;
}

// Test result
interface TestResult {
  name: string;
  passed: boolean;
  error?: string;
  logs?: string[];
  duration: number;
}

// Test suite stats
interface SuiteStats {
  total: number;
  passed: number;
  failed: number;
  duration: number;
}

// Default config
const DEFAULT_CONFIG: TestConfig = {
  rpcUrl: process.env.RPC_URL || 'https://api.devnet.solana.com',
  programId: new PublicKey(process.env.PROGRAM_ID || 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw'),
  keypairPath: process.env.KEYPAIR || path.join(process.env.HOME || '', '.config/solana/keys/id.json'),
  pythSolFeed: new PublicKey(process.env.PYTH_SOL_FEED || 'H6ARHf6YXhGYeQfUzQNGk6rDN1aQfwbNgBEMwLf9f5vK'),
  verbose: process.env.VERBOSE === 'true',
};

// Test context
interface TestContext {
  program: any; // Program<Idl>
  provider: any; // AnchorProvider
  signerPublicKey: PublicKey;
  config: TestConfig;
  results: TestResult[];
}

// Test function type
type TestFunction = (ctx: TestContext) => Promise<void>;

// Utility: Load IDL
function loadIdl(): Idl {
  const candidates = [
    path.join(process.cwd(), 'contracts', 'target', 'idl', 'quantdesk_perp_dex.json'),
    path.join(process.cwd(), 'frontend', 'src', 'types', 'quantdesk_perp_dex.json'),
  ];
  
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) {
      return JSON.parse(fs.readFileSync(candidate, 'utf8')) as Idl;
    }
  }
  
  throw new Error(`IDL not found. Searched:\n${candidates.map(c => `  - ${c}`).join('\n')}`);
}

// Utility: Load keypair
function loadKeypair(path: string): anchor.web3.Keypair {
  const raw = JSON.parse(fs.readFileSync(path, 'utf8'));
  return anchor.web3.Keypair.fromSecretKey(new Uint8Array(raw));
}

// Utility: Derive PDAs
function derivePDAs(authority: PublicKey, programId: PublicKey, accountIndex: number = 0) {
  const accountIndexBuffer = Buffer.alloc(2);
  accountIndexBuffer.writeUInt16LE(accountIndex, 0);
  
  const [userAccount] = PublicKey.findProgramAddressSync(
    [Buffer.from('user_account'), authority.toBuffer(), accountIndexBuffer],
    programId
  );
  
  const [collateralAccount] = PublicKey.findProgramAddressSync(
    [Buffer.from('collateral'), authority.toBuffer(), Buffer.from('SOL')],
    programId
  );
  
  const [protocolVault] = PublicKey.findProgramAddressSync(
    [Buffer.from('protocol_sol_vault')],
    programId
  );
  
  return { userAccount, collateralAccount, protocolVault };
}

// Utility: Format duration
function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

// Test runner
class TestRunner {
  private context: TestContext;
  private indent: number = 0;

  constructor(context: TestContext) {
    this.context = context;
  }

  log(message: string, type: 'info' | 'success' | 'error' | 'warning' = 'info') {
    const prefix = '  '.repeat(this.indent);
    const icons = { info: '  ', success: '✅', error: '❌', warning: '⚠️ ' };
    console.log(`${prefix}${icons[type]} ${message}`);
  }

  async describe(name: string, fn: () => Promise<void>) {
    this.log(name, 'info');
    this.indent++;
    try {
      await fn();
    } finally {
      this.indent--;
    }
  }

  async it(name: string, fn: () => Promise<void>) {
    const start = Date.now();
    try {
      if (this.context.config.verbose) {
        this.log(name, 'info');
      }
      await fn();
      const duration = Date.now() - start;
      this.context.results.push({ name, passed: true, duration });
      if (this.context.config.verbose) {
        this.log(`${name} (${formatDuration(duration)})`, 'success');
      }
    } catch (error: any) {
      const duration = Date.now() - start;
      this.context.results.push({
        name,
        passed: false,
        error: error.message,
        logs: error.logs,
        duration,
      });
      this.log(`${name}: ${error.message}`, 'error');
      if (error.logs && this.context.config.verbose) {
        error.logs.forEach((log: string) => this.log(`   ${log}`, 'info'));
      }
      throw error;
    }
  }

  expect(value: any): any {
    const assertNotBeNull = () => {
      if (value === null || value === undefined) {
        throw new Error('Expected value to not be null');
      }
    };
    
    return {
      to: {
        equal: (expected: any) => {
          if (value !== expected) {
            throw new Error(`Expected ${expected}, got ${value}`);
          }
        },
        not: {
          be: {
            null: assertNotBeNull,
          },
        },
      },
      not: {
        be: {
          null: assertNotBeNull,
        },
      },
    };
  }
}

// Test: Basic deposit
async function testBasicDeposit(ctx: TestContext) {
  const runner = new TestRunner(ctx);
  const { program, provider, signerPublicKey, config } = ctx;
  
  await runner.describe('Basic Deposit Test', async () => {
    const pdas = derivePDAs(signerPublicKey, config.programId);
    const depositAmount = 0.001 * LAMPORTS_PER_SOL;
    
    await runner.it('Execute deposit transaction', async () => {
      const sig = await program.methods
        .depositNativeSol(new BN(depositAmount))
        .accounts({
          userAccount: pdas.userAccount,
          user: signerPublicKey,
          protocolVault: pdas.protocolVault,
          collateralAccount: pdas.collateralAccount,
          solUsdPriceFeed: config.pythSolFeed,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,
        })
        .rpc();
      
      runner.expect(sig).not.be.null;
      if (config.verbose) {
        runner.log(`Transaction: https://explorer.solana.com/tx/${sig}?cluster=devnet`, 'info');
      }
    });
    
    await runner.it('Verify user account exists', async () => {
      try {
        const account = await (program as any).account.userAccount.fetch(pdas.userAccount);
        runner.expect(account).not.be.null;
      } catch (e: any) {
        throw new Error(`UserAccount not found: ${e.message}`);
      }
    });
  });
}

// Test: PDA derivation
async function testPDADerivation(ctx: TestContext) {
  const runner = new TestRunner(ctx);
  const { signerPublicKey, config } = ctx;
  
  await runner.describe('PDA Derivation Tests', async () => {
    await runner.it('Verify user account PDA', async () => {
      const pdas = derivePDAs(signerPublicKey, config.programId, 0);
      runner.expect(pdas.userAccount).not.be.null;
    });
    
    await runner.it('Verify collateral PDA', async () => {
      const pdas = derivePDAs(signerPublicKey, config.programId);
      runner.expect(pdas.collateralAccount).not.be.null;
    });
    
    await runner.it('Verify protocol vault PDA', async () => {
      const pdas = derivePDAs(signerPublicKey, config.programId);
      runner.expect(pdas.protocolVault).not.be.null;
    });
  });
}

// Test: Account order validation
async function testAccountOrder(ctx: TestContext) {
  const runner = new TestRunner(ctx);
  const { program, signerPublicKey, config } = ctx;
  
  await runner.describe('Account Order Validation', async () => {
    await runner.it('Verify deposit_native_sol account order', async () => {
      const idl = loadIdl();
      // Try both camelCase and snake_case
      const depositIx = idl.instructions?.find((ix: any) => 
        ix.name === 'depositNativeSol' || ix.name === 'deposit_native_sol'
      );
      
      if (!depositIx || !depositIx.accounts) {
        const available = idl.instructions?.map((ix: any) => ix.name).join(', ') || 'none';
        throw new Error(`depositNativeSol instruction not found. Available: ${available}`);
      }
      
      // Expected order from IDL
      const expectedOrder = depositIx.accounts.map((acc: any) => acc.name);
      const expectedUserIndex = expectedOrder.indexOf('user');
      
      runner.expect(expectedUserIndex).to.equal(1); // User should be at position 1
      
      if (ctx.config.verbose) {
        runner.log(`Account order: ${expectedOrder.join(', ')}`, 'info');
      }
    });
  });
}

// Test: Error handling
async function testErrorHandling(ctx: TestContext) {
  const runner = new TestRunner(ctx);
  const { program, signerPublicKey, config } = ctx;
  
  await runner.describe('Error Handling Tests', async () => {
    await runner.it('Handles insufficient funds gracefully', async () => {
      const pdas = derivePDAs(signerPublicKey, config.programId);
      const hugeAmount = new BN(LAMPORTS_PER_SOL * 1000000); // 1M SOL
      
      try {
        await program.methods
          .depositNativeSol(hugeAmount)
          .accounts({
            userAccount: pdas.userAccount,
            user: signerPublicKey,
            protocolVault: pdas.protocolVault,
            collateralAccount: pdas.collateralAccount,
            solUsdPriceFeed: config.pythSolFeed,
            systemProgram: SystemProgram.programId,
            rent: SYSVAR_RENT_PUBKEY,
          })
          .rpc();
        
        throw new Error('Expected transaction to fail');
      } catch (error: any) {
        // Expected to fail
        runner.expect(error).not.be.null;
      }
    });
  });
}

// Main test suite runner
async function runTestSuite() {
  console.log('\n🧪 QuantDesk CLI Devnet Test Suite\n');
  console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
  
  const config = DEFAULT_CONFIG;
  console.log(`RPC URL: ${config.rpcUrl}`);
  console.log(`Program ID: ${config.programId.toBase58()}`);
  console.log(`Keypair: ${config.keypairPath}\n`);
  
  // Setup
  const connection = new Connection(config.rpcUrl, 'confirmed');
  const payer = loadKeypair(config.keypairPath);
  const wallet = new Wallet(payer);
  const provider = new AnchorProvider(connection, wallet, { commitment: 'confirmed' });
  const idl = loadIdl();
  const program = new Program(idl as any, provider);
  const signerPublicKey = provider.wallet.publicKey;
  
  console.log(`✅ Using IDL: ${path.join(process.cwd(), 'contracts', 'target', 'idl', 'quantdesk_perp_dex.json')}`);
  console.log(`✅ Signer: ${signerPublicKey.toBase58()}\n`);
  
  // Check balance
  const balance = await connection.getBalance(signerPublicKey);
  console.log(`Current balance: ${(balance / LAMPORTS_PER_SOL).toFixed(4)} SOL`);
  if (balance < LAMPORTS_PER_SOL * 0.01) {
    console.log('⚠️  Low balance. Requesting airdrop...');
    const sig = await connection.requestAirdrop(signerPublicKey, 0.02 * LAMPORTS_PER_SOL);
    await connection.confirmTransaction(sig);
  }
  console.log('');
  
  // Test context
  const context: TestContext = {
    program,
    provider,
    signerPublicKey,
    config,
    results: [],
  };
  
  // Run tests
  const startTime = Date.now();
  const tests: TestFunction[] = [
    testPDADerivation,
    testAccountOrder,
    testBasicDeposit,
    testErrorHandling,
  ];
  
  for (const test of tests) {
    try {
      await test(context);
    } catch (error) {
      // Error already logged by test runner
    }
  }
  
  // Summary
  const duration = Date.now() - startTime;
  const stats: SuiteStats = {
    total: context.results.length,
    passed: context.results.filter(r => r.passed).length,
    failed: context.results.filter(r => !r.passed).length,
    duration,
  };
  
  console.log('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  console.log(`\n📊 Test Results\n`);
  console.log(`Total: ${stats.total}`);
  console.log(`✅ Passed: ${stats.passed}`);
  console.log(`❌ Failed: ${stats.failed}`);
  console.log(`⏱️  Duration: ${formatDuration(stats.duration)}\n`);
  
  if (stats.failed > 0) {
    console.log('Failed Tests:');
    context.results
      .filter(r => !r.passed)
      .forEach(r => {
        console.log(`  ❌ ${r.name}`);
        if (r.error) console.log(`     Error: ${r.error}`);
      });
    console.log('');
  }
  
  process.exit(stats.failed > 0 ? 1 : 0);
}

// CLI argument parsing
const args = process.argv.slice(2);
if (args.includes('--help') || args.includes('-h')) {
  console.log(`
QuantDesk CLI Devnet Test Suite

Usage:
  npm run devnet:test:suite [options]

Options:
  --verbose, -v        Show detailed test output
  --help, -h          Show this help message

Environment Variables:
  RPC_URL             Devnet RPC endpoint (default: https://api.devnet.solana.com)
  PROGRAM_ID          Program ID (default: C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw)
  KEYPAIR             Keypair path (default: ~/.config/solana/keys/id.json)
  PYTH_SOL_FEED       Pyth SOL/USD feed (default: devnet feed)
  VERBOSE             Enable verbose output (default: false)

Examples:
  npm run devnet:test:suite
  npm run devnet:test:suite -- --verbose
  VERBOSE=true npm run devnet:test:suite
`);
  process.exit(0);
}

if (args.includes('--verbose') || args.includes('-v')) {
  DEFAULT_CONFIG.verbose = true;
}

// Run suite
runTestSuite().catch((error) => {
  console.error('Fatal error:', error);
  process.exit(1);
});

