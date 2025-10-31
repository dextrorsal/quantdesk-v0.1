use anchor_lang::prelude::*;

declare_id!("GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a");

#[program]
pub mod vault_recovery {
    use super::*;

    /// Withdraw all SOL from protocol vault to authority
    pub fn withdraw_vault(ctx: Context<WithdrawVault>, amount: u64) -> Result<()> {
        let vault = &ctx.accounts.protocol_vault;
        let authority = &ctx.accounts.authority;
        
        msg!("Withdrawing {} lamports from vault to authority", amount);
        
        // Transfer SOL from vault PDA to authority
        **vault.to_account_info().try_borrow_mut_lamports()? -= amount;
        **authority.to_account_info().try_borrow_mut_lamports()? += amount;
        
        msg!("Successfully withdrew {} lamports", amount);
        Ok(())
    }

    /// Close vault account and return rent + balance to authority
    pub fn close_vault(ctx: Context<CloseVault>) -> Result<()> {
        let authority = &ctx.accounts.authority;
        let vault = &ctx.accounts.protocol_vault;
        
        // Transfer all lamports (including rent) from vault to authority
        let balance = vault.to_account_info().lamports();
        **vault.to_account_info().try_borrow_mut_lamports()? = 0;
        **authority.to_account_info().try_borrow_mut_lamports()? += balance;
        
        msg!("Closed vault, returned {} lamports to authority", balance);
        Ok(())
    }
}

#[derive(Accounts)]
pub struct WithdrawVault<'info> {
    /// CHECK: Vault PDA - verified via seeds constraint
    #[account(
        mut,
        seeds = [b"protocol_sol_vault"],
        bump
    )]
    pub protocol_vault: AccountInfo<'info>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
}

#[derive(Accounts)]
pub struct CloseVault<'info> {
    #[account(
        mut,
        seeds = [b"protocol_sol_vault"],
        bump
    )]
    pub protocol_vault: Account<'info, ProtocolSolVault>,
    
    #[account(mut)]
    pub authority: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

#[account]
pub struct ProtocolSolVault {
    pub total_deposits: u64,
    pub total_withdrawals: u64,
    pub is_active: bool,
    pub bump: u8,
}

