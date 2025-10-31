# Story 0-3: Expose withdraw_native_sol in Program + IDL

Status: ready-for-dev

## Story

As a trader, I need to withdraw SOL so I can exit positions and retrieve funds.

## Acceptance Criteria

1. IDL contains `withdraw_native_sol` with accounts: [user_account, user signer, protocol_vault, collateral_account, sol_usd_price_feed, system_program, (rent if required)]
2. CLI test withdraw succeeds on devnet; signature returned; lamports decrease in protocol_vault and increase in user wallet

## Tasks / Subtasks

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

## Dev Notes

### Architecture Alignment
- Smart contract instruction: `contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs`
- Function exists at lines 201-243
- Account structure defined at lines 471+
- Program ID: `C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw` (devnet)

### References
- [Source: bmad/docs/tech-spec-epic-0.md#Smart Contract Withdraw Instruction]
- [Source: contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs]
- [Source: contracts/deployed-idl.json - IDL verification]

## Dev Agent Record

### Context Reference

- bmad/stories/0-3-expose-withdraw-native-sol-program-idl.context.md

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List

