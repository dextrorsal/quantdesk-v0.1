<story-context id="bmad/bmm/workflows/4-implementation/story-context/template" v="1.0">
  <metadata>
    <epicId>0</epicId>
    <storyId>4</storyId>
    <title>UI Guard for Missing Withdraw Instruction</title>
    <status>ready-for-dev</status>
    <generatedAt>2025-10-30</generatedAt>
    <generator>BMAD Story Context Workflow</generator>
    <sourceStoryPath>bmad/stories/0-4-ui-guard-missing-withdraw-instruction.md</sourceStoryPath>
  </metadata>

  <story>
    <asA>user</asA>
    <iWant>not to hit runtime errors if withdraw is unavailable</iWant>
    <soThat>I have a smooth experience even during upgrades</soThat>
    <tasks>
- [ ] Detect if IDL lacks `withdraw_native_sol` instruction [AC1]
  - [ ] Check `program?.idl?.instructions` for `withdrawNativeSol`
  - [ ] Create utility function to check instruction availability
- [ ] Soft-disable withdraw button [AC1]
  - [ ] Disable button when instruction unavailable
  - [ ] Add tooltip: "Upgrade in progress"
  - [ ] Style appropriately (grayed out, not clickable)
- [ ] Prevent runtime errors [AC2]
  - [ ] Never call `program.methods.withdrawNativeSol` if unavailable
  - [ ] Add guard checks before any withdraw function calls
  - [ ] Test console for any error messages
    </tasks>
  </story>

  <acceptanceCriteria>
1. Withdraw action is soft-disabled when IDL lacks `withdraw_native_sol`; tooltip explains "Upgrade in progress"
2. No "program.methods.withdrawNativeSol is not a function" appears in console
  </acceptanceCriteria>

  <artifacts>
    <docs>
      <doc path="bmad/docs/tech-spec-epic-0.md" title="Epic 0 Technical Specification" section="UI Guard for Missing Withdraw">
        Defines requirements for soft-disabling withdraw action when instruction unavailable. Requires tooltip and graceful degradation.
      </doc>
      <doc path="bmad/docs/architecture.md" title="QuantDesk Architecture Documentation" section="Frontend Services">
        Architecture outlines frontend program context and IDL access patterns.
      </doc>
    </docs>
    <code>
      <artifact path="frontend/src/components/WithdrawModal.tsx" kind="component" symbol="WithdrawModal" lines="5-419" reason="Withdraw modal component that handles withdrawal UI. Currently calls smartContractService.withdrawNativeSOL without checking if instruction is available. Needs guard check.">
        Current implementation:
        - Modal component with withdraw button (lines 362-383)
        - Calls `smartContractService.withdrawNativeSOL()` on button click (line 80-111)
        - **MISSING**: No check for instruction availability before rendering/clicking
        - Button disabled state: checks amount, collateral, validation errors (line 364)
        - Needs: IDL availability check and soft-disable with tooltip
      </artifact>
      <artifact path="frontend/src/services/smartContractService.ts" kind="service" symbol="SmartContractService.withdrawNativeSOL" lines="1583-1696" reason="Smart contract service method that handles withdrawal. Already has guard check but throws error instead of returning gracefully.">
        Current implementation:
        - Method: `async withdrawNativeSOL(wallet: any, amountInLamports: number): Promise&lt;string&gt;`
        - Guard check exists (line 1650): `if (!((program as any)?.methods && (program as any).methods.withdrawNativeSol))`
        - **ISSUE**: Throws error instead of returning gracefully (line 1651)
        - **NEEDS**: Should return null/undefined to allow UI to handle gracefully, or check should be done at UI level
      </artifact>
      <artifact path="frontend/src/components/AccountSlideOut.tsx" kind="component" symbol="AccountSlideOut" lines="18-451" reason="Account slide-out component with withdraw button. Currently calls handleWithdraw without checking instruction availability.">
        Withdraw button (lines 364-373):
        - Calls `handleWithdraw` on click (line 368)
        - Disabled state: only checks `loading` (line 369)
        - **MISSING**: No check for instruction availability
      </artifact>
      <artifact path="frontend/src/components/wallet/EnhancedAccountPanel.tsx" kind="component" symbol="EnhancedAccountPanel" lines="45-529" reason="Enhanced account panel with withdraw tab. Withdraw button calls handleWithdraw without checking instruction availability.">
        Withdraw button (lines 458-474):
        - Calls `handleWithdraw` on click (line 459)
        - Disabled state: checks `!withdrawAmount || isProcessing` (line 460)
        - **MISSING**: No check for instruction availability
      </artifact>
    </code>
    <dependencies>
      <dependency>
        <ecosystem>React</ecosystem>
        <packages>
          <package>react: ^18.x</package>
          <package>@solana/web3.js: ^1.87.x</package>
          <package>@coral-xyz/anchor: ^0.28.x</package>
        </packages>
      </dependency>
    </dependencies>
  </artifacts>

  <constraints>
    - Withdraw button must be soft-disabled (visible but not clickable) when instruction unavailable
    - Tooltip must explain: "Upgrade in progress" or similar message
    - No console errors when instruction unavailable
    - Guard check must be done at UI level before calling service method
    - Check should verify: `program?.idl?.instructions?.some(ix =&gt; ix.name === 'withdrawNativeSol')`
    - Use theme tokens for disabled state styling (var(--bg-tertiary), opacity: 0.5)
    - All withdraw entry points must be guarded: WithdrawModal, AccountSlideOut, EnhancedAccountPanel
  </constraints>
  <interfaces>
    <interface name="program.idl.instructions" kind="Anchor IDL structure" signature="program.idl.instructions: InstructionDefinition[]" path="frontend/src/services/smartContractService.ts">
      Anchor program IDL instructions array. Check for `withdrawNativeSol` instruction: `program?.idl?.instructions?.some(ix =&gt; ix.name === 'withdrawNativeSol')`.
    </interface>
    <interface name="program.methods.withdrawNativeSol" kind="Anchor method" signature="program.methods.withdrawNativeSol(amount: BN): TransactionBuilder" path="frontend/src/services/smartContractService.ts">
      Anchor program method for withdraw instruction. Only exists if instruction is in IDL. Must check availability before calling.
    </interface>
    <interface name="SmartContractService.withdrawNativeSOL" kind="service method" signature="async withdrawNativeSOL(wallet: any, amountInLamports: number): Promise&lt;string | null&gt;" path="frontend/src/services/smartContractService.ts">
      Service method for withdrawing SOL. Currently throws error if instruction unavailable. Should return null/undefined gracefully or check should be done at UI level.
    </interface>
  </interfaces>
  <tests>
    <standards>
      Frontend tests use Vitest and React Testing Library. Test files located in `frontend/src/tests/` with subdirectories: `unit/`, `integration/`, `e2e/`. Mock Anchor program and IDL for testing. Use `@testing-library/react` for component testing.
    </standards>
    <locations>
      - `frontend/src/tests/unit/` - Unit tests for utilities and services
      - `frontend/src/tests/integration/` - Integration tests for components
      - `frontend/src/tests/e2e/` - End-to-end tests for user flows
    </locations>
    <ideas>
      <test ac="AC1" idea="Unit test: utility function checks IDL for withdraw instruction">
        Mock Anchor program with IDL. Test utility function returns true when instruction exists, false when missing. Verify checks `program?.idl?.instructions` array correctly.
      </test>
      <test ac="AC1" idea="Component test: WithdrawModal soft-disables button when instruction unavailable">
        Mock program with IDL missing withdrawNativeSol. Render WithdrawModal. Verify button is disabled. Verify tooltip shows "Upgrade in progress". Verify button styling (grayed out, opacity).
      </test>
      <test ac="AC2" idea="Component test: No runtime error when instruction unavailable">
        Mock program with IDL missing withdrawNativeSol. Click withdraw button. Verify no error in console. Verify button remains disabled. Verify no call to smartContractService.withdrawNativeSOL.
      </test>
      <test ac="AC2" idea="E2E test: User sees disabled button with tooltip">
        Open withdraw modal with program missing withdraw instruction. Verify button is disabled. Hover over button. Verify tooltip shows "Upgrade in progress". Verify no console errors.
      </test>
    </ideas>
  </tests>
</story-context>

