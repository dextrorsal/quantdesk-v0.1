<story-context id="bmad/bmm/workflows/4-implementation/story-context/template" v="1.0">
  <metadata>
    <epicId>0</epicId>
    <storyId>9</storyId>
    <title>Drift Parity Review for Deposit/Withdraw</title>
    <status>ready-for-dev</status>
    <generatedAt>2025-10-30</generatedAt>
    <generator>BMAD Story Context Workflow</generator>
    <sourceStoryPath>bmad/stories/0-9-drift-parity-review-deposit-withdraw.md</sourceStoryPath>
  </metadata>

  <story>
    <asA>protocol engineer</asA>
    <iWant>a gap analysis vs Drift to minimize bugs</iWant>
    <soThat>I can ensure our implementation follows industry best practices</soThat>
    <tasks>
- [ ] Research Drift Protocol deposit/withdraw implementation [AC1]
  - [ ] Document Drift account list for deposit/withdraw
  - [ ] Note WSOL vs native SOL approach
  - [ ] Document additional accounts used
  - [ ] Note price check integration
- [ ] Compare with QuantDesk implementation [AC1]
  - [ ] Document QuantDesk account list
  - [ ] Compare account order differences
  - [ ] Note any missing accounts
  - [ ] Compare price check approaches
- [ ] Make recommendation [AC1]
  - [ ] Evaluate WSOL vs native SOL pros/cons
  - [ ] Select recommended approach
  - [ ] Document rationale
- [ ] Document risks and mitigation [AC2]
  - [ ] List risks (rent, signer, account order, etc.)
  - [ ] Provide mitigation strategies for each risk
  - [ ] Document best practices learned
    </tasks>
  </story>

  <acceptanceCriteria>
1. Document compares account lists, WSOL vs native SOL, remaining accounts, price checks; recommended approach selected (native/WSOL)
2. Risks and mitigation listed (e.g., rent, signer, account order)
  </acceptanceCriteria>

  <artifacts>
    <docs>
      <doc path="bmad/docs/tech-spec-epic-0.md" title="Epic 0 Technical Specification" section="Drift Parity Review">
        Defines requirements for comparing QuantDesk with Drift Protocol deposit/withdraw implementations.
      </doc>
      <doc path="DRIFT_DEPOSIT_ANALYSIS.md" title="Drift Protocol Deposit Flow Analysis" section="Key Findings">
        Existing analysis document comparing Drift's deposit flow with QuantDesk. Documents account initialization, rent usage, wrapped SOL vs native SOL approaches.
      </doc>
      <doc path="contracts/docs/TOKEN_OPERATIONS_STRUCT_GUIDE.md" title="Token Operations Structure Guide" section="withdraw_native_sol()">
        Documents QuantDesk withdraw_native_sol() implementation with collateral integration patterns.
      </doc>
    </docs>
    <code>
      <artifact path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs" kind="instruction" symbol="deposit_native_sol" lines="140-198" reason="QuantDesk deposit native SOL instruction. Used for comparison with Drift's approach.">
        QuantDesk implementation:
        - Function: `pub fn deposit_native_sol(ctx: Context&lt;DepositNativeSol&gt;, amount: u64) -&gt; Result&lt;()&gt;`
        - Account order: user_account, user, protocol_vault, collateral_account, sol_usd_price_feed, system_program, rent
        - Uses native SOL directly (not wrapped SOL)
        - Uses `init_if_needed` for account initialization
      </artifact>
      <artifact path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs" kind="instruction" symbol="withdraw_native_sol" lines="200-245" reason="QuantDesk withdraw native SOL instruction. Used for comparison with Drift's approach.">
        QuantDesk implementation:
        - Function: `pub fn withdraw_native_sol(ctx: Context&lt;WithdrawNativeSol&gt;, amount: u64) -&gt; Result&lt;()&gt;`
        - Account order: user_account, collateral_account, protocol_vault, user, sol_usd_price_feed, system_program
        - Uses native SOL directly (not wrapped SOL)
        - Transfers SOL from protocol vault to user via system program CPI
      </artifact>
      <artifact path="frontend/src/services/smartContractService.ts" kind="service" symbol="depositNativeSOLBestPractice" lines="2046-2286" reason="Frontend service implementing best practices for deposit. Documents recommended patterns and aligns with Anchor best practices.">
        Best practices documented:
        - Always uses provider.wallet.publicKey (never wallet.adapter.publicKey) in .accounts()
        - Derives PDAs correctly matching Rust seeds exactly
        - Uses Anchor's .rpc() method (preferred over manual transaction building)
        - Proper error handling and validation
      </artifact>
    </code>
    <dependencies>
      <dependency>
        <ecosystem>Research</ecosystem>
        <packages>
          <package>Drift Protocol documentation: https://drift-labs.github.io/protocol-v2/sdk/</package>
          <package>Drift SDK GitHub: drift-labs/protocol-v2</package>
        </packages>
      </dependency>
    </dependencies>
  </artifacts>

  <constraints>
    - Output document must be comprehensive but concise
    - Must compare account lists, account order, and account types
    - Must evaluate WSOL vs native SOL approaches
    - Must document risks and mitigation strategies
    - Must provide clear recommendation with rationale
    - Must reference existing Drift analysis document
    - Must align with current QuantDesk implementation
  </constraints>
  <interfaces>
    <interface name="Drift deposit/withdraw flow" kind="external reference" signature="Drift Protocol deposit/withdraw implementation" path="https://drift-labs.github.io/protocol-v2/sdk/">
      Drift Protocol deposit/withdraw implementation. Uses wrapped SOL approach, separate initialization and deposit instructions, explicit rent handling.
    </interface>
    <interface name="QuantDesk deposit/withdraw flow" kind="implementation" signature="QuantDesk deposit/withdraw implementation" path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs">
      QuantDesk deposit/withdraw implementation. Uses native SOL approach, init_if_needed for initialization, explicit rent account in IDL.
    </interface>
  </interfaces>
  <tests>
    <standards>
      This is a research/documentation story. No automated tests required. Documentation should be validated through peer review.
    </standards>
    <locations>
      - `docs/drift-parity-analysis.md` - Output document location
    </locations>
    <ideas>
      <test ac="AC1" idea="Document validation: Account lists comparison">
        Verify output document contains:
        - Drift account list for deposit/withdraw
        - QuantDesk account list for deposit/withdraw
        - Side-by-side comparison table
        - Account order differences noted
        - Missing accounts identified
      </test>
      <test ac="AC1" idea="Document validation: WSOL vs native SOL evaluation">
        Verify output document contains:
        - WSOL approach pros/cons
        - Native SOL approach pros/cons
        - Recommended approach selected
        - Rationale documented
      </test>
      <test ac="AC2" idea="Document validation: Risks and mitigation">
        Verify output document contains:
        - Risk list (rent, signer, account order, etc.)
        - Mitigation strategy for each risk
        - Best practices section
      </test>
    </ideas>
  </tests>
</story-context>

