import { Connection, Keypair, PublicKey, SystemProgram, LAMPORTS_PER_SOL, SYSVAR_RENT_PUBKEY } from '@solana/web3.js';
import anchor from '@coral-xyz/anchor';
import fs from 'fs';
import path from 'path';

const { AnchorProvider, Program, Wallet, BN } = anchor;
type Idl = anchor.Idl;

// ENV
const RPC_URL = process.env.RPC_URL || 'https://api.devnet.solana.com';
const KEYPAIR_PATH = process.env.KEYPAIR || path.join(process.env.HOME || '', '.config/solana/id.json');
const PROGRAM_ID = new PublicKey(process.env.PROGRAM_ID || 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw');
const PYTH_SOL_FEED = new PublicKey(process.env.PYTH_SOL_FEED || 'H6ARHf6YXhGYeQfUzQNGk6rDN1aQfwbNgBEMwLf9f5vK'); // devnet pyth SOL/USD
const DEPOSIT_SOL = Number(process.env.DEPOSIT_SOL || 0.001); // 0.001 SOL
const ACCOUNT_INDEX = Number(process.env.ACCOUNT_INDEX || 0);

function loadKeypair(p: string): Keypair {
  const raw = JSON.parse(fs.readFileSync(p, 'utf8'));
  return Keypair.fromSecretKey(new Uint8Array(raw));
}

function findFirstExisting(paths: string[]): string | null {
  for (const p of paths) {
    if (fs.existsSync(p)) return p;
  }
  return null;
}

async function loadIdl(): Promise<Idl> {
  const candidates = [
    path.join(process.cwd(), 'contracts', 'target', 'idl', 'quantdesk_perp_dex.json'),
    path.join(process.cwd(), 'frontend', 'src', 'types', 'quantdesk_perp_dex.json'),
    path.join(process.cwd(), 'frontend', 'src', 'types', 'quantdesk-perp-dex.json'),
  ];
  const idlPath = findFirstExisting(candidates);
  if (!idlPath) {
    throw new Error(
      `IDL not found. Searched:\n${candidates.map(c => `  - ${c}`).join('\n')}\n\n` +
      'Build contracts: cd contracts && anchor build\n' +
      'Or ensure IDL is in frontend/src/types/quantdesk_perp_dex.json'
    );
  }
  console.log(`✅ Using IDL: ${idlPath}`);
  return JSON.parse(fs.readFileSync(idlPath, 'utf8')) as Idl;
}

function pdaUserAccount(authority: PublicKey, index: number): [PublicKey, number] {
  const indexBytes = Buffer.alloc(2);
  indexBytes.writeUInt16LE(index, 0);
  return PublicKey.findProgramAddressSync([
    Buffer.from('user_account'),
    authority.toBuffer(),
    indexBytes,
  ], PROGRAM_ID);
}

function pdaCollateral(authority: PublicKey): [PublicKey, number] {
  return PublicKey.findProgramAddressSync([
    Buffer.from('collateral'),
    authority.toBuffer(),
    Buffer.from('SOL'),
  ], PROGRAM_ID);
}

function pdaProtocolVault(): [PublicKey, number] {
  return PublicKey.findProgramAddressSync([
    Buffer.from('protocol_sol_vault'),
  ], PROGRAM_ID);
}

async function main() {
  console.log('RPC_URL', RPC_URL);
  const connection = new Connection(RPC_URL, 'confirmed');
  const payer = loadKeypair(KEYPAIR_PATH);
  const wallet = new Wallet(payer);
  const provider = new AnchorProvider(connection, wallet, { commitment: 'confirmed' });
  const idl = await loadIdl();
  // Program constructor derives program ID from IDL
  const program = new Program(idl as any, provider);

  // CRITICAL: Use provider.wallet.publicKey directly - don't assign to variable
  // Anchor uses object reference equality for signer matching
  const signerPublicKey = provider.wallet.publicKey;
  const [userAccount] = pdaUserAccount(signerPublicKey, ACCOUNT_INDEX);
  const [collateralAccount] = pdaCollateral(signerPublicKey);
  const [protocolVault] = pdaProtocolVault();

  console.log('Signer (provider.wallet.publicKey):', signerPublicKey.toBase58());
  console.log('Wallet object:', provider.wallet);
  console.log('Wallet publicKey object identity:', provider.wallet.publicKey === signerPublicKey);
  console.log('Program:', PROGRAM_ID.toBase58());
  console.log('UserAccount PDA:', userAccount.toBase58());
  console.log('Collateral PDA:', collateralAccount.toBase58());
  console.log('Protocol Vault PDA:', protocolVault.toBase58());

  // 1) Check balance and airdrop if needed
  const bal = await connection.getBalance(signerPublicKey);
  console.log(`Current balance: ${bal / LAMPORTS_PER_SOL} SOL`);
  if (bal < LAMPORTS_PER_SOL * 0.01) {
    console.log('Requesting airdrop 0.02 SOL...');
    const sig = await connection.requestAirdrop(signerPublicKey, 0.02 * LAMPORTS_PER_SOL);
    await connection.confirmTransaction(sig, 'confirmed');
    const newBal = await connection.getBalance(signerPublicKey);
    console.log(`New balance: ${newBal / LAMPORTS_PER_SOL} SOL`);
  }

  // 2) Deposit SOL (deposit_native_sol handles initialization of both user_account and collateral_account)
  const lamports = Math.floor(DEPOSIT_SOL * LAMPORTS_PER_SOL);
  console.log(`\nDepositing ${DEPOSIT_SOL} SOL (${lamports} lamports)...`);
  console.log('Account order must match IDL exactly:');
  console.log('  1. user_account (PDA)');
  console.log('  2. user (signer)');
  console.log('  3. protocol_vault (PDA)');
  console.log('  4. collateral_account (PDA)');
  console.log('  5. sol_usd_price_feed');
  console.log('  6. system_program');
  console.log('  7. rent');

  try {
    // CRITICAL: Use provider.wallet.publicKey DIRECTLY - Anchor uses object reference for signer matching
    const sig = await program.methods
      .depositNativeSol(new BN(lamports))
      .accounts({
        userAccount,
        user: provider.wallet.publicKey, // MUST be provider.wallet.publicKey directly, not a variable!
        protocolVault,
        collateralAccount,
        solUsdPriceFeed: PYTH_SOL_FEED,
        systemProgram: SystemProgram.programId,
        rent: SYSVAR_RENT_PUBKEY,
      })
      .rpc();

    console.log(`\n✅ Transaction successful!`);
    console.log(`Signature: ${sig}`);
    console.log(`Explorer: https://explorer.solana.com/tx/${sig}?cluster=devnet`);
  } catch (error: any) {
    console.error(`\n❌ Transaction failed:`);
    console.error(error);
    if (error.logs) {
      console.error('\nTransaction logs:');
      error.logs.forEach((log: string) => console.error(log));
    }
    throw error;
  }

  // Fetch user account to confirm deposit
  console.log('\nVerifying deposit...');
  try {
    // Account names may be camelCase in IDL - try both
    const accountNamespace = (program as any).account;
    if (accountNamespace.userAccount) {
      const ua = await accountNamespace.userAccount.fetch(userAccount);
      console.log('✅ UserAccount fetched successfully:');
      console.log(`   Total collateral: ${ua.totalCollateral?.toString?.() ?? JSON.stringify(ua)}`);
      console.log(`   Available margin: ${ua.availableMargin?.toString?.() ?? 'N/A'}`);
      console.log(`   Account health: ${ua.accountHealth?.toString?.() ?? 'N/A'}`);
    } else {
      console.log('⚠️  userAccount namespace not found in IDL - accounts may be differently named');
      console.log(`   Available accounts: ${Object.keys(accountNamespace).join(', ')}`);
    }
  } catch (e: any) {
    console.warn('⚠️  Could not fetch userAccount:', e.message);
  }

  try {
    const accountNamespace = (program as any).account;
    if (accountNamespace.collateralAccount) {
      const coll = await accountNamespace.collateralAccount.fetch(collateralAccount);
      console.log('✅ CollateralAccount fetched successfully:');
      console.log(`   Amount: ${coll.amount?.toString?.() ?? 'N/A'} lamports`);
      console.log(`   USD value: ${coll.valueUsd?.toString?.() ?? 'N/A'}`);
      console.log(`   Asset type: ${coll.assetType ?? 'N/A'}`);
    } else {
      console.log('⚠️  collateralAccount namespace not found in IDL');
    }
  } catch (e: any) {
    console.warn('⚠️  Could not fetch collateralAccount:', e.message);
  }
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
