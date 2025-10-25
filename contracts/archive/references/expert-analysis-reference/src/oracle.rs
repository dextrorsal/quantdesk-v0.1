use anchor_lang::prelude::*;
use pyth_sdk_solana::state::{load_price_account, SolanaPriceAccount};

/// Oracle type enum to support multiple oracle providers
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq)]
pub enum OracleType {
    Pyth,
    FixedPrice, // For devnet testing
}

/// Standardized price data structure
#[derive(Debug, Clone)]
pub struct OraclePrice {
    pub price: i64,        // Price value
    pub expo: i32,         // Price exponent
    pub conf: u64,         // Confidence interval
    pub timestamp: i64,    // Last update timestamp
}

impl OraclePrice {
    /// Get the scaled price as f64
    pub fn get_scaled_price(&self) -> f64 {
        (self.price as f64) * 10f64.powi(self.expo)
    }
}

/// Read price from Pyth oracle with proper error handling and security checks
fn get_price_from_pyth(price_feed: &AccountInfo) -> Result<OraclePrice> {
    let price_account_data = price_feed.try_borrow_data()
        .map_err(|_| crate::ErrorCode::OracleFeedNotFound)?;
    
    // Check if account has data
    if price_account_data.len() == 0 {
        msg!("⚠️ Pyth price feed account is empty (common on devnet)");
        return Err(crate::ErrorCode::OracleFeedNotFound.into());
    }
    
    // Load price account with explicit type annotation
    let price_account: &SolanaPriceAccount = load_price_account(&price_account_data.as_ref())
        .map_err(|_| {
            msg!("⚠️ Failed to parse Pyth price account");
            crate::ErrorCode::OracleFeedNotFound
        })?;
    
    // Get price and exponent from the account
    let price = price_account.agg.price;
    let expo = price_account.expo;
    let conf = price_account.agg.conf;
    let timestamp = price_account.timestamp as i64;
    
    // Check price is not stale (last updated within 5 minutes)
    let clock = Clock::get()?;
    let price_age = clock.unix_timestamp - timestamp;
    if price_age > 300 {
        msg!("⚠️ Pyth price is stale: {} seconds old", price_age);
        return Err(crate::ErrorCode::PriceStale.into());
    }
    
    // SECURITY: Check confidence interval (Flash Trade recommendation)
    let price_scaled = price as f64 * 10f64.powi(expo);
    let conf_scaled = conf as f64 * 10f64.powi(expo);
    let conf_percentage = (conf_scaled / price_scaled.abs()) * 100.0;
    
    // Reject prices with confidence interval > 1% (Flash Trade threshold)
    if conf_percentage > 1.0 {
        msg!("⚠️ Pyth confidence interval too wide: {:.2}%", conf_percentage);
        return Err(crate::ErrorCode::PriceStale.into());
    }
    
    // SECURITY: Price band validation (Kamino recommendation)
    // For SOL, reject prices outside reasonable range ($50 - $500)
    if price_scaled < 50.0 || price_scaled > 500.0 {
        msg!("⚠️ SOL price outside acceptable range: ${:.2}", price_scaled);
        return Err(crate::ErrorCode::PriceStale.into());
    }
    
    Ok(OraclePrice {
        price,
        expo,
        conf,
        timestamp,
    })
}

/// Helper function to get USD value from SOL amount with Pyth + fixed price fallback
/// This is perfect for devnet where Pyth feeds are empty
pub fn get_usd_value_from_sol_with_fallback(
    sol_lamports: u64,
    price_feed: &AccountInfo,
    fallback_price_usd: Option<f64>, // e.g., Some(208.0) for $208/SOL
) -> Result<u64> {
    // Try Pyth first
    let oracle_price = match get_price_from_pyth(price_feed) {
        Ok(price) => price,
        Err(_) => {
            // If Pyth fails, use fixed price
            if let Some(price_usd) = fallback_price_usd {
                msg!("✅ Using fixed price fallback: ${:.2}/SOL", price_usd);
                OraclePrice {
                    price: (price_usd * 10f64.powi(6)) as i64, // Convert to i64 with -6 expo
                    expo: -6,
                    conf: 0,
                    timestamp: Clock::get()?.unix_timestamp,
                }
            } else {
                msg!("❌ No fallback price provided and Pyth failed!");
                return Err(crate::ErrorCode::OracleFeedNotFound.into());
            }
        }
    };
    
    // Convert lamports to SOL: sol_lamports / 1e9
    let sol_amount = (sol_lamports as f64) / 1_000_000_000.0;
    
    // Get USD value: sol_amount * (price * 10^expo)
    let price_scaled = oracle_price.get_scaled_price();
    let usd_value = sol_amount * price_scaled;
    
    // Store as 6-decimal USDC: result * 1e6
    let usd_value_6_decimals = (usd_value * 1_000_000.0) as u64;
    
    msg!("🔮 Oracle Price: ${:.2}, {} lamports = ${:.2} USD ({})", 
         price_scaled, sol_lamports, usd_value, 
         if oracle_price.conf == 0 { "fixed" } else { "Pyth" });
    
    Ok(usd_value_6_decimals)
}

/// Helper function to get SOL amount from USD value with Pyth + fixed price fallback
pub fn get_sol_from_usd_value_with_fallback(
    usd_value_6_decimals: u64,
    price_feed: &AccountInfo,
    fallback_price_usd: Option<f64>,
) -> Result<u64> {
    // Try Pyth first
    let oracle_price = match get_price_from_pyth(price_feed) {
        Ok(price) => price,
        Err(_) => {
            // If Pyth fails, use fixed price
            if let Some(price_usd) = fallback_price_usd {
                OraclePrice {
                    price: (price_usd * 10f64.powi(6)) as i64,
                    expo: -6,
                    conf: 0,
                    timestamp: Clock::get()?.unix_timestamp,
                }
            } else {
                return Err(crate::ErrorCode::OracleFeedNotFound.into());
            }
        }
    };
    
    // Get scaled price
    let price_scaled = oracle_price.get_scaled_price();
    
    // Convert USD to SOL
    let usd_amount = (usd_value_6_decimals as f64) / 1_000_000.0;
    let sol_amount = usd_amount / price_scaled;
    let sol_lamports = (sol_amount * 1_000_000_000.0) as u64;
    
    Ok(sol_lamports)
}

// Convenience functions for common use cases

/// Get USD from SOL using Pyth (mainnet) or current market price fallback (devnet)
pub fn get_usd_from_sol_devnet_safe(sol_lamports: u64, price_feed: &AccountInfo) -> Result<u64> {
    // Use current SOL price (~$184) instead of hardcoded $208
    get_usd_value_from_sol_with_fallback(sol_lamports, price_feed, Some(184.0))
}

/// Get SOL from USD using Pyth (mainnet) or current market price fallback (devnet)
pub fn get_sol_from_usd_devnet_safe(usd_value_6_decimals: u64, price_feed: &AccountInfo) -> Result<u64> {
    get_sol_from_usd_value_with_fallback(usd_value_6_decimals, price_feed, Some(184.0))
}

/// Get USD from SOL using only Pyth (strict mainnet mode)
pub fn get_usd_from_sol_pyth_only(sol_lamports: u64, price_feed: &AccountInfo) -> Result<u64> {
    get_usd_value_from_sol_with_fallback(sol_lamports, price_feed, None)
}
