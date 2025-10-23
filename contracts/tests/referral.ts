import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { QuantdeskPerpDex } from "../target/types/quantdesk_perp_dex";
import { Keypair, PublicKey, SystemProgram, LAMPORTS_PER_SOL } from "@solana/web3.js";
import { expect } from "chai";
import { BankrunProvider } from "bankrun";

describe("referral tests", () => {
  const provider = BankrunProvider.local();
  anchor.setProvider(provider);

  const program = anchor.workspace.QuantdeskPerpDex as Program<QuantdeskPerpDex>;

  const deployer = provider.wallet.payer;
  let referrer: Keypair;
  let referee: Keypair;

  before(async () => {
    // Airdrop SOL to deployer if needed (Bankrun handles this typically)
    // await provider.connection.requestAirdrop(deployer.publicKey, 100 * LAMPORTS_PER_SOL);

    referrer = Keypair.generate();
    referee = Keypair.generate();

    // Fund test wallets
    await provider.connection.requestAirdrop(referrer.publicKey, 10 * LAMPORTS_PER_SOL);
    await provider.connection.requestAirdrop(referee.publicKey, 10 * LAMPORTS_PER_SOL);
  });

  it("Initializes a user account and SOL collateral account for referrer", async () => {
    const accountIndex = 0;
    const accountIndexBuffer = Buffer.alloc(2);
    accountIndexBuffer.writeUInt16LE(accountIndex, 0);

    const [referrerUserAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("user_account"), referrer.publicKey.toBuffer(), accountIndexBuffer],
      program.programId
    );

    const [referrerSolCollateralPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("collateral"), referrer.publicKey.toBuffer(), Buffer.from("SOL")],
      program.programId
    );

    // Create user account
    await program.methods
      .createUserAccount(accountIndex)
      .accounts({
        userAccount: referrerUserAccountPda,
        authority: referrer.publicKey,
        systemProgram: SystemProgram.programId,
      })
      .signers([referrer])
      .rpc();

    // Initialize SOL collateral account
    await program.methods
      .initializeCollateralAccount({ sol: {} }, new anchor.BN(0)) // Amount 0 initially
      .accounts({
        collateralAccount: referrerSolCollateralPda,
        user: referrer.publicKey,
        systemProgram: SystemProgram.programId,
      })
      .signers([referrer])
      .rpc();

    const userAccount = await program.account.userAccount.fetch(referrerUserAccountPda);
    expect(userAccount.authority.toBase58()).to.equal(referrer.publicKey.toBase58());
    // You might want to fetch and inspect the collateral account as well
  });

  it("Initializes a user account and SOL collateral account for referee", async () => {
    const accountIndex = 0;
    const accountIndexBuffer = Buffer.alloc(2);
    accountIndexBuffer.writeUInt16LE(accountIndex, 0);

    const [refereeUserAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("user_account"), referee.publicKey.toBuffer(), accountIndexBuffer],
      program.programId
    );

    const [refereeSolCollateralPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("collateral"), referee.publicKey.toBuffer(), Buffer.from("SOL")],
      program.programId
    );

    // Create user account
    await program.methods
      .createUserAccount(accountIndex)
      .accounts({
        userAccount: refereeUserAccountPda,
        authority: referee.publicKey,
        systemProgram: SystemProgram.programId,
      })
      .signers([referee])
      .rpc();

    // Initialize SOL collateral account
    await program.methods
      .initializeCollateralAccount({ sol: {} }, new anchor.BN(0)) // Amount 0 initially
      .accounts({
        collateralAccount: refereeSolCollateralPda,
        user: referee.publicKey,
        systemProgram: SystemProgram.programId,
      })
      .signers([referee])
      .rpc();

    const userAccount = await program.account.userAccount.fetch(refereeUserAccountPda);
    expect(userAccount.authority.toBase58()).to.equal(referee.publicKey.toBase58());
  });

  it("Deposits SOL into referee's collateral account (mocking trade volume for activation)", async () => {
    const accountIndex = 0;
    const accountIndexBuffer = Buffer.alloc(2);
    accountIndexBuffer.writeUInt16LE(accountIndex, 0);

    const [refereeUserAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("user_account"), referee.publicKey.toBuffer(), accountIndexBuffer],
      program.programId
    );

    const [refereeSolCollateralPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("collateral"), referee.publicKey.toBuffer(), Buffer.from("SOL")],
      program.programId
    );

    const [protocolSolVaultPda] = PublicKey.findProgramAddressSync(
      [Buffer.from("protocol_sol_vault")],
      program.programId
    );

    // Initialize protocol SOL vault if it doesn't exist
    try {
      await program.account.protocolVault.fetch(protocolSolVaultPda);
    } catch (e) {
      await program.methods
        .initializeProtocolSolVault()
        .accounts({
          protocolVault: protocolSolVaultPda,
          authority: deployer.publicKey, // Deployer initializes the vault
          systemProgram: SystemProgram.programId,
        })
        .signers([deployer])
        .rpc();
    }

    const depositAmount = new anchor.BN(0.1 * LAMPORTS_PER_SOL); // 0.1 SOL mock deposit

    await program.methods
      .depositNativeSol(depositAmount)
      .accounts({
        userAccount: refereeUserAccountPda,
        user: referee.publicKey,
        protocolVault: protocolSolVaultPda,
        collateralAccount: refereeSolCollateralPda,
        systemProgram: SystemProgram.programId,
      })
      .signers([referee])
      .rpc();

    const collateralAccount = await program.account.collateralAccount.fetch(refereeSolCollateralPda);
    // Assuming collateralAccount stores amount as lamports
    expect(collateralAccount.amount.toNumber()).to.be.greaterThanOrEqual(depositAmount.toNumber());
  });

  // Note: The actual referral binding and activation logic (e.g., setting activated_at in Supabase)
  // is handled by the backend and Supabase RPC, which Bankrun cannot directly test.
  // These tests focus on the smart contract's role in the flow.

  // You would typically have more granular tests here for specific instruction calls
  // and their effects on program state.
});
