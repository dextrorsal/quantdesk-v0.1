<story-context id="bmad/bmm/workflows/4-implementation/story-context/template" v="1.0">
  <metadata>
    <epicId>0</epicId>
    <storyId>8</storyId>
    <title>Deposit/Withdraw CLI Tests and UI E2E</title>
    <status>ready-for-dev</status>
    <generatedAt>2025-10-30</generatedAt>
    <generator>BMAD Story Context Workflow</generator>
    <sourceStoryPath>bmad/stories/0-8-deposit-withdraw-cli-tests-ui-e2e.md</sourceStoryPath>
  </metadata>

  <story>
    <asA>maintainer</asA>
    <iWant>automated checks to ensure deposits/withdraws function through upgrades</iWant>
    <soThat>I can validate functionality continuously</soThat>
    <tasks>
- [ ] Create CLI test script for deposit [AC1]
  - [ ] Script: `scripts/test-deposit-native-sol.sh`
  - [ ] Assert PDA derivations correct
  - [ ] Verify account order matches IDL
  - [ ] Check success logs
  - [ ] Verify SOL balance increases in protocol_vault
  - [ ] Return transaction signature
- [ ] Create CLI test script for withdraw [AC1]
  - [ ] Script: `scripts/test-withdraw-native-sol.sh`
  - [ ] Assert PDA derivations correct
  - [ ] Verify account order matches IDL
  - [ ] Check success logs
  - [ ] Verify lamports decrease in protocol_vault
  - [ ] Verify lamports increase in user wallet
  - [ ] Return transaction signature
- [ ] Create UI E2E test for deposit flow [AC2]
  - [ ] Connect wallet
  - [ ] Deposit SOL via UI
  - [ ] Verify USD balance updates in UI
  - [ ] Capture screenshots at each step
  - [ ] Store screenshots for regression testing
- [ ] Create UI E2E test for withdraw flow [AC2]
  - [ ] Withdraw SOL via UI
  - [ ] Verify balance reduces correctly
  - [ ] Capture screenshots at each step
  - [ ] Store screenshots for regression testing
    </tasks>
  </story>

  <acceptanceCriteria>
1. CLI scripts: `deposit_native_sol`, `withdraw_native_sol`; assert PDA derivations, accounts order, success logs
2. UI e2e: connect wallet → deposit → see USD balance → withdraw → balance reduces; screenshots stored
  </acceptanceCriteria>

  <artifacts>
    <docs>
      <doc path="bmad/docs/tech-spec-epic-0.md" title="Epic 0 Technical Specification" section="Deposit/Withdraw CLI Tests and UI E2E">
        Defines requirements for CLI test scripts and UI E2E tests for deposit/withdraw flows. Requires PDA derivation validation, account order verification, and screenshot capture.
      </doc>
      <doc path="scripts/DEVNET_SMOKE_TEST_README.md" title="Devnet Smoke Test Guide" section="Account Order">
        Documents CLI testing guide for devnet. Includes account order verification, PDA derivations, and transaction signature validation.
      </doc>
      <doc path="scripts/CLI_DEVNET_TESTING_GUIDE.md" title="CLI Devnet Testing Guide" section="Test Suite Structure">
        Comprehensive guide for CLI testing on devnet. Documents test suite structure, PDA tests, account order tests, and error handling tests.
      </doc>
    </docs>
    <code>
      <artifact path="scripts/devnet_smoke_test.ts" kind="test script" symbol="devnet_smoke_test" lines="1-180" reason="Existing CLI test script for deposit testing. Tests deposit_native_sol instruction with PDA derivations and account order verification.">
        Current implementation:
        - Tests deposit_native_sol instruction
        - Derives PDAs: user_account, protocol_vault, collateral_account
        - Verifies account order
        - Checks success logs
        - Returns transaction signature
        - **NEEDS**: Enhance for withdraw_native_sol testing
      </artifact>
      <artifact path="scripts/devnet_test_suite.ts" kind="test suite" symbol="devnet_test_suite" lines="1-251" reason="Comprehensive test suite runner. Includes deposit tests, PDA tests, initialization tests.">
        Test suite structure:
        - testBasicDeposit: Tests deposit flow
        - testPDADerivation: Tests PDA derivations
        - **MISSING**: testWithdrawNativeSol function
        - **NEEDS**: Withdraw test implementation
      </artifact>
      <artifact path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs" kind="instruction" symbol="deposit_native_sol" lines="140-198" reason="Deposit native SOL instruction. Used by CLI tests to verify deposit functionality.">
        Deposit instruction:
        - Function: `pub fn deposit_native_sol(ctx: Context&lt;DepositNativeSol&gt;, amount: u64) -&gt; Result&lt;()&gt;`
        - Account order: user_account, user, protocol_vault, collateral_account, sol_usd_price_feed, system_program, rent
      </artifact>
      <artifact path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs" kind="instruction" symbol="withdraw_native_sol" lines="200-245" reason="Withdraw native SOL instruction. Used by CLI tests to verify withdraw functionality.">
        Withdraw instruction:
        - Function: `pub fn withdraw_native_sol(ctx: Context&lt;WithdrawNativeSol&gt;, amount: u64) -&gt; Result&lt;()&gt;`
        - Account order: user_account, collateral_account, protocol_vault, user, sol_usd_price_feed, system_program
      </artifact>
      <artifact path="frontend/src/components/WithdrawModal.tsx" kind="component" symbol="WithdrawModal" lines="5-419" reason="Withdraw modal component for UI E2E testing. Tests should interact with this component for withdraw flow.">
        Withdraw modal:
        - Opens when user clicks withdraw button
        - Accepts amount input
        - Calls smartContractService.withdrawNativeSOL
        - Shows balance updates
        - **E2E**: Test deposit → withdraw flow through this component
      </artifact>
      <artifact path="frontend/src/components/DepositModal.tsx" kind="component" symbol="DepositModal" lines="1-419" reason="Deposit modal component for UI E2E testing. Tests should interact with this component for deposit flow.">
        Deposit modal:
        - Opens when user clicks deposit button
        - Accepts amount input
        - Calls smartContractService.depositNativeSOL
        - Shows balance updates
        - **E2E**: Test deposit flow through this component
      </artifact>
    </code>
    <dependencies>
      <dependency>
        <ecosystem>Solana</ecosystem>
        <packages>
          <package>@solana/web3.js: ^1.87.x</package>
          <package>@coral-xyz/anchor: ^0.28.x</package>
          <package>solana-cli: 2.3.0</package>
        </packages>
      </dependency>
      <dependency>
        <ecosystem>Testing</ecosystem>
        <packages>
          <package>playwright: ^1.x</package>
          <package>@playwright/test: ^1.x</package>
        </packages>
      </dependency>
    </dependencies>
  </artifacts>

  <constraints>
    - CLI scripts must run on devnet (real on-chain testing)
    - PDA derivations must match Rust struct exactly
    - Account order must match IDL order exactly
    - Transaction signatures must be returned for verification
    - E2E tests must use Playwright or Cypress
    - Screenshots must be stored in `tests/e2e/screenshots/`
    - Tests must verify balance changes (protocol_vault and user wallet)
    - E2E tests must test full flow: connect wallet → deposit → verify balance → withdraw → verify balance
    - Screenshots must be captured at each step
  </constraints>
  <interfaces>
    <interface name="deposit_native_sol" kind="Anchor instruction" signature="pub fn deposit_native_sol(ctx: Context&lt;DepositNativeSol&gt;, amount: u64) -&gt; Result&lt;()&gt;" path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs">
      Deposit native SOL instruction. Account order: user_account, user, protocol_vault, collateral_account, sol_usd_price_feed, system_program, rent.
    </interface>
    <interface name="withdraw_native_sol" kind="Anchor instruction" signature="pub fn withdraw_native_sol(ctx: Context&lt;WithdrawNativeSol&gt;, amount: u64) -&gt; Result&lt;()&gt;" path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs">
      Withdraw native SOL instruction. Account order: user_account, collateral_account, protocol_vault, user, sol_usd_price_feed, system_program.
    </interface>
    <interface name="anchor build" kind="CLI command" signature="anchor build" path="contracts/">
      Builds program and generates IDL. Must be run before CLI tests to ensure IDL is up to date.
    </interface>
    <interface name="solana balance" kind="CLI command" signature="solana balance &lt;address&gt;" path="scripts/">
      Gets account balance in SOL. Used to verify balance changes before/after deposit/withdraw.
    </interface>
  </interfaces>
  <tests>
    <standards>
      CLI tests use TypeScript with @solana/web3.js and @coral-xyz/anchor. Test scripts located in `scripts/` directory. E2E tests use Playwright or Cypress in `frontend/tests/e2e/` directory. Screenshots stored in `frontend/tests/e2e/screenshots/`.
    </standards>
    <locations>
      - `scripts/` - CLI test scripts
      - `scripts/tests/` - CLI test suite modules
      - `frontend/tests/e2e/` - UI E2E tests (may need to be created)
      - `frontend/tests/e2e/screenshots/` - Screenshot storage
    </locations>
    <ideas>
      <test ac="AC1" idea="CLI test: deposit_native_sol with PDA derivations">
        Create script `scripts/test-deposit-native-sol.sh`. Derive PDAs for user_account, protocol_vault, collateral_account. Build transaction with deposit_native_sol instruction. Verify account order matches IDL. Send transaction and verify success. Check transaction signature returned. Verify SOL balance increases in protocol_vault.
      </test>
      <test ac="AC1" idea="CLI test: withdraw_native_sol with balance verification">
        Create script `scripts/test-withdraw-native-sol.sh`. Derive PDAs. Get initial balances (protocol_vault, user wallet). Execute withdraw_native_sol. Verify lamports decrease in protocol_vault, increase in user wallet. Return transaction signature.
      </test>
      <test ac="AC2" idea="E2E test: deposit flow with screenshots">
        Open frontend. Connect wallet (Phantom). Click deposit button. Enter amount. Submit deposit. Verify USD balance updates. Capture screenshots at each step: connected, deposit modal, amount entered, balance updated. Store screenshots in `tests/e2e/screenshots/deposit-flow/`.
      </test>
      <test ac="AC2" idea="E2E test: full deposit → withdraw flow">
        Connect wallet. Deposit SOL. Verify balance. Withdraw SOL. Verify balance reduces. Capture screenshots: initial balance, after deposit, after withdraw. Verify balance returns to initial state.
      </test>
    </ideas>
  </tests>
</story-context>

