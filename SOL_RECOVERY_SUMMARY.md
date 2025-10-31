# SOL Recovery Analysis - Old Programs

## 🔍 Findings

### Programs Checked:
- `quantdesk_collateral`: ❌ Does not exist
- `quantdesk_core`: ❌ Does not exist  
- `quantdesk_oracle`: ❌ Does not exist
- `quantdesk_perp_dex`: ✅ Current active (0.0011 SOL - rent minimum)
- `quantdesk_security`: ❌ Does not exist
- `quantdesk_trading`: ❌ Does not exist
- `old_program_1` (Gmz5q8cad...): ✅ Exists (0.0011 SOL - rent minimum)
- `old_program_2` (GcpEyGMJ...): ✅ Exists (0.0011 SOL - rent minimum)

---

## 💰 Found Reclaimable SOL

### **OLD VAULT PDA: 3.1281 SOL** ⚠️

**Vault Address**: `FsnVEemM46kshuCzMbeYezV4EFRD1sGugSG4WkRvgp3s`  
**Program**: `GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a`  
**Balance**: 3.128086440 SOL  
**Owner**: Owned by old program (PDA)

**Program Authority**: `wgfSHTWx1woRXhsWijj1kcpCP8tmbmK2KnouFVAuoc6`

### Current Program Vault: 0.0231 SOL
- Vault: `5pXGgCZiyhRWAbR29oebssF9Cb4tsSwZppvHBuTxUBZ4`
- This is our active vault (keep it)

---

## 🛠️ How to Reclaim SOL

### Option 1: Check if We Have Authority
1. Check if your wallet (`solana address`) matches the program authority
2. If yes, you can:
   - Upgrade the old program to add a `withdraw_vault` instruction
   - OR close the entire program (if no longer needed)

### Option 2: Check Program Instructions
The old program might already have a withdrawal instruction. Check:
```bash
anchor idl fetch GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a
```

### Option 3: Close Entire Program
If you have upgrade authority and don't need the program:
1. Upgrade program with empty buffer (closes it)
2. This will refund all rent to the authority
3. **Note**: This closes the program permanently!

---

## ⚠️ Important Notes

1. **Vault is a PDA**: Cannot withdraw without program instruction
2. **Need Authority**: Must have upgrade authority to modify program
3. **Check Instructions**: Program might have existing withdraw instruction
4. **3.13 SOL**: Significant amount worth recovering!

---

## 📋 Next Steps

1. ✅ Verify wallet matches program authority
2. ✅ Check if program has withdraw/close instructions
3. ✅ Consider if program is still needed
4. ✅ If reclaimable, create instruction or upgrade

---

**Status**: 🔍 Investigation complete - **3.13 SOL found in old vault!**

