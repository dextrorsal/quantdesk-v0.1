<story-context id="bmad/bmm/workflows/4-implementation/story-context/template" v="1.0">
  <metadata>
    <epicId>0</epicId>
    <storyId>10</storyId>
    <title>Expert Confirmation of Native SOL Patterns</title>
    <status>ready-for-dev</status>
    <generatedAt>2025-10-30</generatedAt>
    <generator>BMAD Story Context Workflow</generator>
    <sourceStoryPath>bmad/stories/0-10-expert-confirmation-native-sol-patterns.md</sourceStoryPath>
  </metadata>

  <story>
    <asA>protocol engineer</asA>
    <iWant>expert-validated patterns to avoid regressions</iWant>
    <soThat>I can ensure our implementation follows Solana best practices</soThat>
    <tasks>
- [ ] Research expert guidance on native SOL patterns [AC1]
  - [ ] Signer position in account struct (must be first)
  - [ ] SystemAccount usage for vault
  - [ ] When to include rent Sysvar
  - [ ] invoke vs invoke_signed for PDA vault
  - [ ] Anchor 0.30+ accounts/addresses behavior
- [ ] Document expert guidance [AC1]
  - [ ] Create concise document with all patterns
  - [ ] Include examples where helpful
  - [ ] Note any version-specific behaviors
  - [ ] Document best practices
    </tasks>
  </story>

  <acceptanceCriteria>
1. Short note capturing expert guidance: signer first, SystemAccount for vault, when to include rent, invoke vs invoke_signed for PDA vault, Anchor 0.30 accounts/addresses behavior
  </acceptanceCriteria>

  <artifacts>
    <docs>
      <doc path="bmad/docs/tech-spec-epic-0.md" title="Epic 0 Technical Specification" section="Expert Confirmation of Native SOL Patterns">
        Defines requirements for documenting expert-validated native SOL patterns.
      </doc>
      <doc path="contracts/ANCHOR_PROGRAM_ORGANIZATION_BEST_PRACTICES.md" title="Anchor Program Organization Best Practices" section="WithdrawNativeSol">
        Documents WithdrawNativeSol struct and instruction patterns following Solana best practices.
      </doc>
      <doc path="contracts/CPI_ARCHITECTURE.md" title="CPI Architecture Guide" section="Native SOL Patterns">
        Documents CPI patterns for native SOL transfers and vault management.
      </doc>
      <doc path="DRIFT_DEPOSIT_ANALYSIS.md" title="Drift Protocol Deposit Flow Analysis" section="Key Takeaways">
        Existing analysis document with lessons learned from Drift Protocol implementation.
      </doc>
    </docs>
    <code>
      <artifact path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs" kind="instruction" symbol="DepositNativeSol" lines="471-489" reason="QuantDesk deposit native SOL account struct. Shows current account structure and order.">
        Current implementation:
        - Account struct: user_account, user (signer), protocol_vault, collateral_account, sol_usd_price_feed, system_program, rent
        - Signer position: user is second (after user_account)
        - Protocol vault: SystemAccount (not PDA in struct, but derived as PDA in code)
        - Rent: Explicitly included in struct
      </artifact>
      <artifact path="contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs" kind="instruction" symbol="WithdrawNativeSol" lines="471-489" reason="QuantDesk withdraw native SOL account struct. Shows current account structure and order.">
        Current implementation:
        - Account struct: user_account, collateral_account, protocol_vault, user (signer), sol_usd_price_feed, system_program
        - Signer position: user is fourth (after collateral_account, protocol_vault)
        - Protocol vault: SystemAccount
        - No rent in withdraw struct
      </artifact>
      <artifact path="frontend/src/services/smartContractService.ts" kind="service" symbol="depositNativeSOLBestPractice" lines="2046-2286" reason="Frontend service implementing best practices. Documents recommended patterns.">
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
          <package>Solana Cookbook: https://solanacookbook.com/</package>
          <package>Anchor Documentation: https://www.anchor-lang.com/docs</package>
          <package>Solana Stack Exchange: Expert Q&amp;A</package>
        </packages>
      </dependency>
    </dependencies>
  </artifacts>

  <constraints>
    - Output document must be concise (short note format)
    - Must capture expert guidance on key patterns
    - Must include examples where helpful
    - Must note version-specific behaviors (Anchor 0.30+)
    - Must document best practices learned
    - Must reference Solana/Anchor documentation and expert sources
  </constraints>
  <interfaces>
    <interface name="Solana Cookbook" kind="external reference" signature="Solana Cookbook Native SOL Patterns" path="https://solanacookbook.com/">
      Solana Cookbook documentation on native SOL patterns, account management, and CPI best practices.
    </interface>
    <interface name="Anchor Documentation" kind="external reference" signature="Anchor Framework Documentation" path="https://www.anchor-lang.com/docs">
      Anchor Framework documentation on account structs, signer handling, rent management, and CPI patterns.
    </interface>
  </interfaces>
  <tests>
    <standards>
      This is a research/documentation story. No automated tests required. Documentation should be validated through peer review and expert confirmation.
    </standards>
    <locations>
      - `docs/native-sol-patterns-expert-guidance.md` - Output document location
    </locations>
    <ideas>
      <test ac="AC1" idea="Document validation: Expert guidance captured">
        Verify output document contains:
        - Signer position guidance (must be first)
        - SystemAccount usage for vault
        - When to include rent Sysvar
        - invoke vs invoke_signed for PDA vault
        - Anchor 0.30+ accounts/addresses behavior
      </test>
      <test ac="AC1" idea="Document validation: Examples and best practices">
        Verify output document contains:
        - Code examples for each pattern
        - Best practices section
        - Version-specific notes (Anchor 0.30+)
      </test>
    </ideas>
  </tests>
</story-context>

