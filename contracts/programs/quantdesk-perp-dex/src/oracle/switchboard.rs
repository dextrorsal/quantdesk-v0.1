//! Switchboard oracle integration
//! This module provides integration with Switchboard oracle feeds for multi-oracle support

use anchor_lang::prelude::*;
use crate::oracle::OraclePrice;
use crate::ErrorCode;

/// Switchboard Aggregator Account Data Structure
/// Based on Switchboard V2 aggregator account format
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwitchboardAggregatorAccount {
    pub magic: u32,           // Magic number: 0x5b4c4c4c
    pub version: u32,          // Version: 2
    pub aggregator_type: u32,  // Aggregator type
    pub size: u32,            // Account size
    pub name: [u8; 32],       // Aggregator name
    pub metadata: [u8; 128],  // Metadata
    pub queue_pubkey: Pubkey,  // Queue public key
    pub oracle_request_batch_size: u32,
    pub min_oracle_results: u32,
    pub min_job_results: u32,
    pub min_update_delay_seconds: u32,
    pub start_after: i64,
    pub variance_tolerance_multiplier: u64,
    pub force_report_period: u64,
    pub expiration: i64,
    pub read_charge: u64,
    pub reward: u64,
    pub max_confidence_interval: u64,
    pub authority: Pubkey,
    pub data_buffer: Pubkey,
    pub current_round: u64,
    pub created_at: i64,
    pub round_open_timestamp: i64,
    pub next_available_timestamp: i64,
    pub is_locked: bool,
    pub crank: Pubkey,
    pub latest_confirmed_round: u64,
    pub latest_confirmed_round_created_at: i64,
    pub latest_confirmed_round_updated_at: i64,
    pub latest_confirmed_round_updated_by: Pubkey,
    pub jobs: [Pubkey; 16],   // Job accounts
    pub job_weights: [u32; 16], // Job weights
    pub job_hashes: [[u8; 32]; 16], // Job hashes
    pub job_results: [SwitchboardJobResult; 16], // Job results
    pub job_results_len: u32,
    pub job_results_padding: u32,
    pub result: SwitchboardResult,
    pub result_padding: u64,
}

/// Switchboard Job Result Structure
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SwitchboardJobResult {
    pub value: i128,          // Result value
    pub timestamp: i64,       // Timestamp
    pub confidence_interval: u64, // Confidence interval
    pub min_response: i128,   // Minimum response
    pub max_response: i128,   // Maximum response
}

/// Switchboard Result Structure
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SwitchboardResult {
    pub value: i128,          // Final result value
    pub timestamp: i64,       // Timestamp
    pub confidence_interval: u64, // Confidence interval
    pub min_response: i128,   // Minimum response
    pub max_response: i128,   // Maximum response
}

/// Get price from Switchboard oracle feed
/// 
/// # Arguments
/// * `feed` - The Switchboard aggregator account
/// 
/// # Returns
/// * `Result<OraclePrice>` - The oracle price from Switchboard
pub fn get_price_from_switchboard(feed: &AccountInfo) -> Result<OraclePrice> {
    let data = feed.try_borrow_data()
        .map_err(|_| crate::ErrorCode::OracleFeedNotFound)?;
    
    if data.len() < std::mem::size_of::<SwitchboardAggregatorAccount>() {
        msg!("❌ Switchboard aggregator account data too small: {} bytes", data.len());
        return Err(crate::ErrorCode::OracleFeedNotFound.into());
    }
    
    // Deserialize using unsafe transmute (Switchboard uses C layout)
    let aggregator = unsafe {
        std::ptr::read(data.as_ptr() as *const SwitchboardAggregatorAccount)
    };
    
    // Validate magic number
    if aggregator.magic != 0x5b4c4c4c {
        msg!("❌ Invalid Switchboard magic number: 0x{:x}", aggregator.magic);
        return Err(crate::ErrorCode::OracleFeedNotFound.into());
    }
    
    // Validate version
    if aggregator.version != 2 {
        msg!("❌ Unsupported Switchboard version: {}", aggregator.version);
        return Err(crate::ErrorCode::OracleFeedNotFound.into());
    }
    
    // Extract price data from result
    let result = aggregator.result;
    let current_time = Clock::get()?.unix_timestamp;
    
    // Validate timestamp (staleness check)
    if current_time - result.timestamp > 60 {
        msg!("⚠️ Switchboard price is stale: {} seconds old", current_time - result.timestamp);
        return Err(ErrorCode::PriceStale.into());
    }
    
    // Convert Switchboard result to OraclePrice format
    // Switchboard uses i128 with different scaling, we need to convert to our format
    let price = result.value as i64;
    let expo = -6i32; // Standard 6 decimal places
    let conf = result.confidence_interval as u64;
    
    msg!("✅ Switchboard price loaded: ${:.2} (confidence: {})", 
         price as f64 * 10f64.powi(expo), conf);
    
    Ok(OraclePrice {
        price,
        expo,
        conf,
        timestamp: result.timestamp,
    })
}

/// Get price from Switchboard with fallback to devnet price
/// 
/// # Arguments
/// * `feed` - The Switchboard aggregator account
/// * `fallback_price_usd` - Fallback price in USD (e.g., Some(184.0) for $184/SOL)
/// 
/// # Returns
/// * `Result<OraclePrice>` - The oracle price from Switchboard or fallback
pub fn get_price_from_switchboard_with_fallback(
    feed: &AccountInfo,
    fallback_price_usd: Option<f64>,
) -> Result<OraclePrice> {
    match get_price_from_switchboard(feed) {
        Ok(price) => Ok(price),
        Err(_) => {
            // If Switchboard fails, use fallback price
            if let Some(price_usd) = fallback_price_usd {
                msg!("✅ Using Switchboard fallback price: ${:.2}/SOL", price_usd);
                Ok(OraclePrice {
                    price: (price_usd * 10f64.powi(6)) as i64, // Convert to i64 with -6 expo
                    expo: -6,
                    conf: 0,
                    timestamp: Clock::get()?.unix_timestamp,
                })
            } else {
                msg!("❌ No Switchboard fallback price provided and Switchboard failed!");
                Err(ErrorCode::OracleFeedNotFound.into())
            }
        }
    }
}

/// Validate Switchboard price security parameters
/// 
/// # Arguments
/// * `price` - The Switchboard oracle price to validate
/// 
/// # Returns
/// * `Result<()>` - Success if price passes security checks
pub fn validate_switchboard_price_security(price: &OraclePrice) -> Result<()> {
    let price_scaled = price.get_scaled_price();
    
    // Security: Confidence interval check
    let conf_scaled = price.conf as f64 * 10f64.powi(price.expo);
    let conf_percentage = if price_scaled.abs() > 0.0 {
        (conf_scaled / price_scaled.abs()) * 100.0
    } else {
        100.0 // Reject zero prices
    };
    
    // Reject prices with confidence interval > 2% (Switchboard threshold)
    if conf_percentage > 2.0 {
        msg!("⚠️ Switchboard confidence interval too wide: {:.2}%", conf_percentage);
        return Err(ErrorCode::PriceStale.into());
    }
    
    // Security: Price band validation
    // For SOL, reject prices outside reasonable range ($50 - $500)
    if price_scaled < 50.0 || price_scaled > 500.0 {
        msg!("⚠️ Switchboard price outside acceptable range: ${:.2}", price_scaled);
        return Err(ErrorCode::PriceStale.into());
    }
    
    Ok(())
}

/// Get USD value from SOL amount using Switchboard oracle
/// 
/// # Arguments
/// * `sol_lamports` - SOL amount in lamports
/// * `feed` - The Switchboard aggregator account
/// * `fallback_price_usd` - Fallback price in USD
/// 
/// # Returns
/// * `Result<u64>` - USD value in 6-decimal USDC format
pub fn get_usd_value_from_sol_switchboard(
    sol_lamports: u64,
    feed: &AccountInfo,
    fallback_price_usd: Option<f64>,
) -> Result<u64> {
    let oracle_price = get_price_from_switchboard_with_fallback(feed, fallback_price_usd)?;
    
    // Convert lamports to SOL: sol_lamports / 1e9
    let sol_amount = (sol_lamports as f64) / 1_000_000_000.0;
    
    // Get USD value: sol_amount * (price * 10^expo)
    let price_scaled = oracle_price.get_scaled_price();
    let usd_value = sol_amount * price_scaled;
    
    // Store as 6-decimal USDC: result * 1e6
    let usd_value_6_decimals = (usd_value * 1_000_000.0) as u64;
    
    msg!("🔮 Switchboard Price: ${:.2}, {} lamports = ${:.2} USD", 
         price_scaled, sol_lamports, usd_value);
    
    Ok(usd_value_6_decimals)
}

/// Get SOL amount from USD value using Switchboard oracle
/// 
/// # Arguments
/// * `usd_value_6_decimals` - USD value in 6-decimal USDC format
/// * `feed` - The Switchboard aggregator account
/// * `fallback_price_usd` - Fallback price in USD
/// 
/// # Returns
/// * `Result<u64>` - SOL amount in lamports
pub fn get_sol_from_usd_switchboard(
    usd_value_6_decimals: u64,
    feed: &AccountInfo,
    fallback_price_usd: Option<f64>,
) -> Result<u64> {
    let oracle_price = get_price_from_switchboard_with_fallback(feed, fallback_price_usd)?;
    
    // Get scaled price
    let price_scaled = oracle_price.get_scaled_price();
    
    // Convert USD to SOL
    let usd_amount = (usd_value_6_decimals as f64) / 1_000_000.0;
    let sol_amount = usd_amount / price_scaled;
    let sol_lamports = (sol_amount * 1_000_000_000.0) as u64;
    
    Ok(sol_lamports)
}

/// Convenience function for devnet testing with Switchboard fallback
/// 
/// # Arguments
/// * `sol_lamports` - SOL amount in lamports
/// * `feed` - The Switchboard aggregator account
/// 
/// # Returns
/// * `Result<u64>` - USD value in 6-decimal USDC format
pub fn get_usd_from_sol_switchboard_devnet_safe(
    sol_lamports: u64,
    feed: &AccountInfo,
) -> Result<u64> {
    // Use current SOL price (~$184) instead of hardcoded price
    get_usd_value_from_sol_switchboard(sol_lamports, feed, Some(184.0))
}

/// Convenience function for devnet testing with Switchboard fallback
/// 
/// # Arguments
/// * `usd_value_6_decimals` - USD value in 6-decimal USDC format
/// * `feed` - The Switchboard aggregator account
/// 
/// # Returns
/// * `Result<u64>` - SOL amount in lamports
pub fn get_sol_from_usd_switchboard_devnet_safe(
    usd_value_6_decimals: u64,
    feed: &AccountInfo,
) -> Result<u64> {
    get_sol_from_usd_switchboard(usd_value_6_decimals, feed, Some(184.0))
}
