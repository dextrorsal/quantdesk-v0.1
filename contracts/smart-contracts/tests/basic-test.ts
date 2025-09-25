import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { QuantdeskPerpDex } from "../target/types/quantdesk_perp_dex";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";
import { expect } from "chai";

describe("quantdesk-perp-dex", () => {
  // Configure the client to use the local cluster.
  anchor.setProvider(anchor.AnchorProvider.env());

  const program = anchor.workspace.QuantdeskPerpDex as Program<QuantdeskPerpDex>;
  const provider = anchor.getProvider();

  it("Program loads correctly", () => {
    expect(program).to.not.be.undefined;
    expect(program.programId).to.not.be.undefined;
    console.log("Program ID:", program.programId.toString());
  });

  it("Provider is configured", () => {
    expect(provider).to.not.be.undefined;
    expect(provider.wallet).to.not.be.undefined;
    console.log("Provider wallet:", provider.wallet.publicKey.toString());
  });

  it("Connection is working", async () => {
    const connection = provider.connection;
    const version = await connection.getVersion();
    expect(version).to.not.be.undefined;
    console.log("Solana version:", version);
  });
});
