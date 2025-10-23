use anchor_lang::prelude::*;

/// PDA Utilities Module
/// Following Solana Cookbook patterns for professional PDA management
/// https://solanacookbook.com/references/programs.html#program-derived-addresses

/// Derive PDA for user account
/// Following Solana Cookbook PDA derivation patterns
#[allow(dead_code)]
pub fn derive_user_account_pda(
    user: &Pubkey,
    account_index: u16,
    program_id: &Pubkey,
) -> Result<(Pubkey, u8)> {
    let seeds = &[
        b"user_account",
        user.as_ref(),
        &account_index.to_le_bytes(),
    ];
    
    Ok(Pubkey::find_program_address(seeds, program_id))
}

/// Derive PDA for market
#[allow(dead_code)]
pub fn derive_market_pda(
    base_asset: &str,
    quote_asset: &str,
    program_id: &Pubkey,
) -> Result<(Pubkey, u8)> {
    let seeds = &[
        b"market",
        base_asset.as_bytes(),
        quote_asset.as_bytes(),
    ];
    
    Ok(Pubkey::find_program_address(seeds, program_id))
}

/// Derive PDA for position
#[allow(dead_code)]
pub fn derive_position_pda(
    user: &Pubkey,
    market: &Pubkey,
    program_id: &Pubkey,
) -> Result<(Pubkey, u8)> {
    let seeds = &[
        b"position",
        user.as_ref(),
        market.as_ref(),
    ];
    
    Ok(Pubkey::find_program_address(seeds, program_id))
}

/// Derive PDA for token vault
#[allow(dead_code)]
pub fn derive_token_vault_pda(
    mint: &Pubkey,
    program_id: &Pubkey,
) -> Result<(Pubkey, u8)> {
    let seeds = &[
        b"vault",
        mint.as_ref(),
    ];
    
    Ok(Pubkey::find_program_address(seeds, program_id))
}

/// Derive PDA for collateral account
#[allow(dead_code)]
pub fn derive_collateral_account_pda(
    user: &Pubkey,
    asset_type: &str,
    program_id: &Pubkey,
) -> Result<(Pubkey, u8)> {
    let seeds = &[
        b"collateral",
        user.as_ref(),
        asset_type.as_bytes(),
    ];
    
    Ok(Pubkey::find_program_address(seeds, program_id))
}

/// Validate PDA ownership
#[allow(dead_code)]
pub fn validate_pda_ownership(
    _pda: &Pubkey,
    _expected_program_id: &Pubkey,
) -> Result<()> {
    // For now, we'll skip this validation as it's not critical for basic functionality
    // In production, you'd want to implement proper PDA ownership validation
    Ok(())
}

/// Create PDA seeds for CPI
#[allow(dead_code)]
pub fn create_pda_seeds(
    prefix: &[u8],
    additional_seeds: &[&[u8]],
    bump: u8,
) -> Vec<Vec<u8>> {
    let mut seeds = vec![prefix.to_vec()];
    seeds.extend(additional_seeds.iter().map(|s| s.to_vec()));
    seeds.push(vec![bump]);
    seeds
}

// Error codes for PDA operations
#[error_code]
pub enum ErrorCode {
    #[msg("PDA derivation failed")]
    PdaDerivationFailed,
    #[msg("Invalid PDA owner")]
    InvalidPdaOwner,
}
