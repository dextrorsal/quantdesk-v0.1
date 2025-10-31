<story-context id="bmad/bmm/workflows/4-implementation/story-context/template" v="1.0">
  <metadata>
    <epicId>0</epicId>
    <storyId>3</storyId>
    <title>Expose withdraw_native_sol in Program + IDL</title>
    <status>ready-for-dev</status>
    <generatedAt>2025-10-30</generatedAt>
    <generator>BMAD Story Context Workflow</generator>
    <sourceStoryPath>bmad/stories/0-3-expose-withdraw-native-sol-program-idl.md</sourceStoryPath>
  </metadata>

  <story>
    <asA>trader</asA>
    <iWant>to withdraw SOL</iWant>
    <soThat>I can exit positions and retrieve funds</soThat>
    <tasks>
- [ ] Verify IDL contains `withdraw_native_sol` instruction [AC1]
  - [ ] Check `contracts/deployed-idl.json` for instruction
  - [ ] Verify account order matches Rust struct exactly:
    - user_account (PDA)
    - user (signer)
    - protocol_vault (PDA)
    - collateral_account (PDA)
    - sol_usd_price_feed (AccountInfo)
    - system_program (Program)
    - rent (optional Sysvar)
- [ ] Create CLI test script [AC2]
  - [ ] Script: `scripts/test-withdraw-native-sol.sh`
  - [ ] Assert PDA derivations correct
  - [ ] Verify account order matches IDL
  - [ ] Check success logs
  - [ ] Verify lamports decrease in protocol_vault
  - [ ] Verify lamports increase in user wallet
  - [ ] Return transaction signature for verification
    </tasks>
  </story>

  <acceptanceCriteria>
1. IDL contains `withdraw_native_sol` with accounts: [user_account, user signer, protocol_vault, collateral_account, sol_usd_price_feed, system_program, (rent if required)]
2. CLI test withdraw succeeds on devnet; signature returned; lamports decrease in protocol_vault and increase in user wallet
  </acceptanceCriteria>

  <artifacts>
    <docs>
      <doc path="bmad/docs/tech-spec-epic-0.md" title="Epic 0 Technical Specification" section="Smart Contract Withdraw Instruction">
        Defines requirements for exposing withdraw_native_sol instruction in IDL. Requires account structure verification and CLI testing.
      </doc>
      <doc path="contracts/ANCHOR_PROGRAM_ORGANIZATION_BEST_PRACTICES.md" title="Anchor Program Organization Best Practices" section="WithdrawNativeSol">
        Documents WithdrawNativeSol struct and instruction patterns following Solana best practices.
      </doc>
      <doc path="contracts/docs/TOKEN_OPERATIONS_STRUCT_GUIDE.md" title="Token Operations Structure Guide" section="withdraw_native_sol()">
        Documents withdraw_native_sol() function with collateral integration patterns.
      </doc>
    </docs>
    <code>
      <artifact path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs" kind="instruction" symbol="withdraw_native_sol" lines="200-245" reason="Withdraw native SOL instruction implementation. Function exists but NOT exposed in program module (lib.rs line 86-92). This is the critical gap - function exists but IDL doesn't contain it because it's not in #[program] module.">
        Current implementation:
        - Function signature: `pub fn withdraw_native_sol(ctx: Context&lt;WithdrawNativeSol&gt;, amount: u64) -&gt; Result&lt;()&gt;`
        - Validates amount &gt; 0, sufficient collateral, active collateral account
        - Calculates USD value using oracle price feed
        - Transfers SOL from protocol vault to user via system program
        - Updates collateral account and user account balances
        - **CRITICAL**: Function exists but NOT exposed in lib.rs program module
      </artifact>
      <artifact path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs" kind="context" symbol="WithdrawNativeSol" lines="471-489" reason="WithdrawNativeSol account context struct. Defines required accounts for withdraw instruction.">
        Account structure:
        - user_account: Account (mutable)
        - collateral_account: Account (mutable)
        - protocol_vault: SystemAccount (mutable)
        - user: Signer (mutable)
        - sol_usd_price_feed: AccountInfo (Pyth price feed)
        - system_program: Program (System)
        - Note: No rent sysvar in current struct
      </artifact>
      <artifact path="contracts/programs/quantdesk-perp-dex/src/lib.rs" kind="program" symbol="quantdesk_perp_dex" lines="46-93" reason="Program module that exposes instructions. Currently only exposes deposit_native_sol, NOT withdraw_native_sol. This is why IDL doesn't contain withdraw instruction.">
        Current program module:
        - Exposes: open_position, close_position, initialize_keeper_security_manager, check_security_before_trading, deposit_native_sol
        - **MISSING**: withdraw_native_sol (needs to be added)
        - Program ID: "C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw" (devnet)
      </artifact>
      <artifact path="contracts/deployed-idl.json" kind="idl" symbol="instructions" lines="1-2130" reason="Deployed IDL file. Currently does NOT contain withdraw_native_sol instruction (verified via grep - no matches found). This confirms the instruction is not exposed.">
        IDL structure:
        - Contains accounts, errors, instructions
        - Currently missing withdraw_native_sol instruction
        - Address: "5yQ3PK9ZZpLesh4dyQ5pApdkz2wzv4tkqi4RwYFmaZnB" (note: different from program ID - may be outdated)
      </artifact>
      <artifact path="contracts/Anchor.toml" kind="config" symbol="programs.devnet" lines="9-16" reason="Anchor configuration file. Defines program IDs for different networks.">
        Devnet program ID:
        - quantdesk_perp_dex = "C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw"
        - Cluster: devnet (default)
      </artifact>
    </code>
    <dependencies>
      <dependency>
        <ecosystem>Solana</ecosystem>
        <packages>
          <package>anchor-lang: 0.32.1</package>
          <package>anchor-spl: 0.32.1</package>
          <package>solana-program: ^1.18.x</package>
        </packages>
      </dependency>
      <dependency>
        <ecosystem>Build Tools</ecosystem>
        <packages>
          <package>anchor-cli: 0.32.1</package>
          <package>solana-cli: 2.3.0</package>
        </packages>
      </dependency>
    </dependencies>
  </artifacts>

  <constraints>
    - Instruction must be added to `#[program]` module in `lib.rs` to appear in IDL
    - Account order in IDL must match Rust struct order exactly
    - PDA derivations must be correct: user_account, protocol_vault, collateral_account
    - Program ID: C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw (devnet)
    - Oracle price feed account required for USD value calculation
    - System program required for SOL transfer
    - Must test on devnet before mainnet deployment
    - CLI test must verify lamport balances before and after withdrawal
  </constraints>
  <interfaces>
    <interface name="withdraw_native_sol" kind="Anchor instruction" signature="pub fn withdraw_native_sol(ctx: Context&lt;WithdrawNativeSol&gt;, amount: u64) -&gt; Result&lt;()&gt;" path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs">
      Withdraw native SOL instruction. Currently implemented but NOT exposed in program module.
    </interface>
    <interface name="WithdrawNativeSol" kind="Anchor context struct" signature="pub struct WithdrawNativeSol&lt;'info&gt; { ... }" path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs">
      Account context struct for withdraw instruction. Contains: user_account, collateral_account, protocol_vault, user, sol_usd_price_feed, system_program.
    </interface>
    <interface name="anchor build" kind="CLI command" signature="anchor build" path="contracts/">
      Builds program and generates IDL. Must be run after adding instruction to program module.
    </interface>
    <interface name="anchor deploy" kind="CLI command" signature="anchor deploy --program-name quantdesk-perp-dex" path="contracts/">
      Deploys program to devnet. Must deploy after adding instruction and rebuilding.
    </interface>
  </interfaces>
  <tests>
    <standards>
      Anchor tests use Rust test framework with anchor_lang::prelude. Test files located in `contracts/tests/` directory. CLI scripts use bash/shell with anchor and solana-cli commands. Integration tests verify on-chain execution.
    </standards>
    <locations>
      - `contracts/tests/` - Anchor integration tests
      - `contracts/scripts/` - CLI test scripts
      - `contracts/tests/integration/` - On-chain integration tests
    </locations>
    <ideas>
      <test ac="AC1" idea="Verify IDL contains withdraw_native_sol after build">
        Run `anchor build` and check `target/idl/quantdesk_perp_dex.json` contains `withdraw_native_sol` instruction. Verify account order matches WithdrawNativeSol struct: user_account, collateral_account, protocol_vault, user, sol_usd_price_feed, system_program.
      </test>
      <test ac="AC1" idea="Verify account structure in IDL matches Rust struct">
        Compare IDL account structure with Rust WithdrawNativeSol struct. Ensure all accounts present, order matches, types correct (Account vs Signer vs Program vs AccountInfo).
      </test>
      <test ac="AC2" idea="CLI test: withdraw succeeds on devnet">
        Create script `scripts/test-withdraw-native-sol.sh`. Derive PDAs for user_account, protocol_vault, collateral_account. Build transaction with withdraw_native_sol instruction. Send transaction and verify success. Check transaction signature returned.
      </test>
      <test ac="AC2" idea="CLI test: verify lamport balances after withdrawal">
        Before withdrawal: Get protocol_vault lamport balance and user wallet balance. Execute withdrawal. After withdrawal: Verify protocol_vault balance decreased by amount + fees, user wallet balance increased by amount.
      </test>
      <test ac="AC2" idea="Integration test: verify account derivations">
        Test PDA derivations using anchor_lang::prelude. Verify user_account, protocol_vault, collateral_account PDAs match expected addresses. Ensure bump seeds correct.
      </test>
    </ideas>
  </tests>
</story-context>

