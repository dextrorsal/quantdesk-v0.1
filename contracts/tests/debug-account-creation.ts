import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { QuantdeskPerpDex } from "../target/types/quantdesk_perp_dex";
import { PublicKey, SystemProgram } from "@solana/web3.js";

describe("debug-account-creation", () => {
  // Configure the client to use the local cluster.
  const provider = anchor.AnchorProvider.env();
  anchor.setProvider(provider);

  const program = anchor.workspace.QuantdeskPerpDex as Program<QuantdeskPerpDex>;
  const wallet = provider.wallet as anchor.Wallet;
  const connection = provider.connection;

  console.log("🔍 Debug Test Setup:");
  console.log("📍 Program ID:", program.programId.toString());
  console.log("📍 Wallet:", wallet.publicKey.toString());
  console.log("📍 Connection:", connection.rpcEndpoint);

  it("Debug PDA derivation", async () => {
    console.log("\n🧪 Testing PDA Derivation...");
    
    const accountIndex = 0;
    const accountIndexBuffer = Buffer.alloc(2);
    accountIndexBuffer.writeUInt16LE(accountIndex, 0);
    
    const [userAccountPda] = await PublicKey.findProgramAddress(
      [
        Buffer.from('user_account'),
        wallet.publicKey.toBuffer(),
        accountIndexBuffer
      ],
      program.programId
    );
    
    console.log("📍 Derived PDA:", userAccountPda.toString());
    console.log("📍 Seeds used:");
    console.log("  - 'user_account':", Buffer.from('user_account').toString('hex'));
    console.log("  - wallet pubkey:", wallet.publicKey.toBuffer().toString('hex'));
    console.log("  - account_index:", accountIndexBuffer.toString('hex'));
    
    // Check if account exists
    const accountInfo = await connection.getAccountInfo(userAccountPda);
    console.log("📍 Account exists:", accountInfo ? "YES" : "NO");
    
    if (accountInfo) {
      console.log("📍 Account data length:", accountInfo.data.length);
      console.log("📍 Account owner:", accountInfo.owner.toString());
      console.log("📍 First 32 bytes:", accountInfo.data.slice(0, 32).toString('hex'));
    }
  });

  it("Debug program instruction structure", async () => {
    console.log("\n🧪 Testing Program Instruction Structure...");
    
    // Check if program exists
    const programInfo = await connection.getAccountInfo(program.programId);
    console.log("📍 Program exists:", programInfo ? "YES" : "NO");
    
    if (programInfo) {
      console.log("📍 Program owner:", programInfo.owner.toString());
      console.log("📍 Program executable:", programInfo.executable);
      console.log("📍 Program data length:", programInfo.data.length);
    }
    
    // Check IDL structure
    console.log("📍 IDL instructions count:", program.idl.instructions.length);
    const createUserAccountInstruction = program.idl.instructions.find(
      (ix) => ix.name === "create_user_account"
    );
    
    if (createUserAccountInstruction) {
      console.log("✅ Found create_user_account instruction");
      console.log("📍 Accounts required:", createUserAccountInstruction.accounts.length);
      createUserAccountInstruction.accounts.forEach((acc, i) => {
        console.log(`  ${i + 1}. ${acc.name} (writable: ${acc.writable}, signer: ${acc.signer})`);
      });
    } else {
      console.log("❌ create_user_account instruction not found in IDL");
    }
  });

  it("Debug account creation with detailed logging", async () => {
    console.log("\n🧪 Testing Account Creation with Detailed Logging...");
    
    try {
      const accountIndex = 0;
      const accountIndexBuffer = Buffer.alloc(2);
      accountIndexBuffer.writeUInt16LE(accountIndex, 0);
      
      const [userAccountPda] = await PublicKey.findProgramAddress(
        [
          Buffer.from('user_account'),
          wallet.publicKey.toBuffer(),
          accountIndexBuffer
        ],
        program.programId
      );
      
      console.log("📍 User Account PDA:", userAccountPda.toString());
      
      // Check if account already exists
      const accountInfo = await connection.getAccountInfo(userAccountPda);
      if (accountInfo) {
        console.log("⚠️ Account already exists, skipping creation");
        return;
      }
      
      console.log("✅ Account does not exist, proceeding with creation...");
      
      // Get wallet balance
      const balance = await connection.getBalance(wallet.publicKey);
      console.log("💰 Wallet balance:", balance / 1e9, "SOL");
      
      if (balance < 0.01 * 1e9) {
        console.log("❌ Insufficient SOL balance for account creation");
        return;
      }
      
      console.log("📞 Calling create_user_account instruction...");
      
      // Call the instruction with detailed logging
      const signature = await program.methods
        .createUserAccount(accountIndex)
        .accounts({
          userAccount: userAccountPda,
          authority: wallet.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .rpc({
          commitment: 'confirmed',
          skipPreflight: false,
        });
      
      console.log("✅ Transaction sent, signature:", signature);
      
      // Wait for confirmation
      await connection.confirmTransaction(signature, 'confirmed');
      console.log("✅ Transaction confirmed");
      
      // Verify account was created
      const newAccountInfo = await connection.getAccountInfo(userAccountPda);
      if (newAccountInfo) {
        console.log("✅ Account created successfully!");
        console.log("📍 Account data length:", newAccountInfo.data.length);
        console.log("📍 Account owner:", newAccountInfo.owner.toString());
      } else {
        console.log("❌ Account was not created");
      }
      
    } catch (error: any) {
      console.error("❌ Error during account creation:", error);
      console.error("📍 Error details:", {
        message: error?.message,
        code: error?.code,
        logs: error?.logs
      });
      
      // Log the full error for debugging
      if (error?.logs) {
        console.log("📍 Transaction logs:");
        error.logs.forEach((log: string, i: number) => {
          console.log(`  ${i + 1}. ${log}`);
        });
      }
      
      throw error;
    }
  });

  it("Debug account deserialization", async () => {
    console.log("\n🧪 Testing Account Deserialization...");
    
    const accountIndex = 0;
    const accountIndexBuffer = Buffer.alloc(2);
    accountIndexBuffer.writeUInt16LE(accountIndex, 0);
    
    const [userAccountPda] = await PublicKey.findProgramAddress(
      [
        Buffer.from('user_account'),
        wallet.publicKey.toBuffer(),
        accountIndexBuffer
      ],
      program.programId
    );
    
    try {
      // Try to fetch the account
      const account = await program.account.userAccount.fetch(userAccountPda);
      console.log("✅ Account deserialized successfully!");
      console.log("📍 Account data:", {
        authority: account.authority.toString(),
        accountIndex: account.accountIndex,
        totalCollateral: account.totalCollateral.toString(),
        totalPositions: account.totalPositions,
        totalOrders: account.totalOrders,
        accountHealth: account.accountHealth,
        isActive: account.isActive
      });
    } catch (error: any) {
      console.error("❌ Failed to deserialize account:", error);
      console.error("📍 Error details:", {
        message: error?.message,
        code: error?.code
      });
      
      // Check raw account data
      const accountInfo = await connection.getAccountInfo(userAccountPda);
      if (accountInfo) {
        console.log("📍 Raw account data length:", accountInfo.data.length);
        console.log("📍 First 64 bytes:", accountInfo.data.slice(0, 64).toString('hex'));
      }
    }
  });
});
