//! Multi-oracle consensus logic
//! This module provides consensus mechanisms for multiple oracle providers
//! to ensure price reliability and prevent manipulation attacks.

use anchor_lang::prelude::*;
use crate::oracle::{OraclePrice, get_price_from_pyth};
use crate::oracle::switchboard::{get_price_from_switchboard, validate_switchboard_price_security};
use crate::ErrorCode;

/// Oracle provider enumeration
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum OracleProvider {
    Pyth,
    Switchboard,
    Chainlink, // Future support
}

/// Consensus result with metadata
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    pub price: OraclePrice,
    pub provider: OracleProvider,
    pub confidence_score: f64,
    pub consensus_method: ConsensusMethod,
}

/// Consensus method enumeration
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConsensusMethod {
    WeightedAverage,
    Median,
    SingleOracle,
    Fallback,
}

/// Context for multi-oracle consensus
#[derive(Accounts)]
pub struct MultiOracleConsensus<'info> {
    #[account(mut)]
    pub signer: Signer<'info>,
    /// CHECK: Pyth price feed account
    pub pyth_feed: AccountInfo<'info>,
    /// CHECK: Switchboard aggregator account
    pub switchboard_feed: AccountInfo<'info>,
    /// CHECK: Price cache account for storing consensus result
    #[account(mut)]
    pub price_cache: AccountInfo<'info>,
}

/// Get consensus price from multiple oracle providers
/// 
/// PERFORMANCE TARGET: <1,200 CU per consensus check
/// 
/// # Arguments
/// * `ctx` - The context containing oracle feeds
/// * `max_staleness` - Maximum allowed staleness in seconds
/// 
/// # Returns
/// * `Result<ConsensusResult>` - Consensus price result with metadata
pub fn get_consensus_price(
    ctx: Context<MultiOracleConsensus>,
    max_staleness: i64,
) -> Result<ConsensusResult> {
    let current_time = Clock::get()?.unix_timestamp;
    
    msg!("🔍 Starting multi-oracle consensus");
    
    // Fetch prices from both oracles
    let pyth_result = get_price_from_pyth(&ctx.accounts.pyth_feed);
    let switchboard_result = get_price_from_switchboard(&ctx.accounts.switchboard_feed);
    
    match (pyth_result, switchboard_result) {
        (Ok(pyth_price), Ok(switchboard_price)) => {
            msg!("✅ Both oracles available: Pyth ${:.2}, Switchboard ${:.2}", 
                 pyth_price.get_scaled_price(), switchboard_price.get_scaled_price());
            
            // Both oracles available - use consensus logic
            let consensus_price = calculate_weighted_consensus(&pyth_price, &switchboard_price)?;
            
            // Validate consensus price
            validate_consensus_price(&consensus_price, max_staleness)?;
            
            // Cache consensus result
            cache_consensus_result(&mut ctx.accounts.price_cache, &consensus_price)?;
            
            Ok(ConsensusResult {
                price: consensus_price,
                provider: OracleProvider::Pyth, // Primary provider
                confidence_score: calculate_confidence_score(&pyth_price, &switchboard_price),
                consensus_method: ConsensusMethod::WeightedAverage,
            })
        }
        (Ok(pyth_price), Err(_)) => {
            msg!("⚠️ Only Pyth oracle available: ${:.2}", pyth_price.get_scaled_price());
            
            // Only Pyth available - use with extra validation
            validate_single_oracle_price(&pyth_price, max_staleness)?;
            
            // Cache single oracle result
            cache_consensus_result(&mut ctx.accounts.price_cache, &pyth_price)?;
            
            Ok(ConsensusResult {
                price: pyth_price,
                provider: OracleProvider::Pyth,
                confidence_score: 0.8, // Reduced confidence for single oracle
                consensus_method: ConsensusMethod::SingleOracle,
            })
        }
        (Err(_), Ok(switchboard_price)) => {
            msg!("⚠️ Only Switchboard oracle available: ${:.2}", switchboard_price.get_scaled_price());
            
            // Only Switchboard available - use with extra validation
            validate_switchboard_price_security(&switchboard_price)?;
            validate_single_oracle_price(&switchboard_price, max_staleness)?;
            
            // Cache single oracle result
            cache_consensus_result(&mut ctx.accounts.price_cache, &switchboard_price)?;
            
            Ok(ConsensusResult {
                price: switchboard_price,
                provider: OracleProvider::Switchboard,
                confidence_score: 0.8, // Reduced confidence for single oracle
                consensus_method: ConsensusMethod::SingleOracle,
            })
        }
        (Err(_), Err(_)) => {
            msg!("❌ All oracles failed");
            Err(ErrorCode::AllOraclesFailed.into())
        }
    }
}

/// Calculate weighted consensus between two oracle prices
/// 
/// # Arguments
/// * `p1` - First oracle price (Pyth)
/// * `p2` - Second oracle price (Switchboard)
/// 
/// # Returns
/// * `Result<OraclePrice>` - Weighted consensus price
fn calculate_weighted_consensus(p1: &OraclePrice, p2: &OraclePrice) -> Result<OraclePrice> {
    // Check for price manipulation (outlier detection)
    let price_diff_percentage = calculate_price_difference_percentage(&p1, &p2);
    
    if price_diff_percentage > 5.0 {
        msg!("⚠️ Large price difference detected: {:.2}%", price_diff_percentage);
        
        // Use median instead of weighted average for outlier protection
        return calculate_median_consensus(p1.clone(), p2.clone());
    }
    
    // Calculate weighted average based on confidence intervals
    let total_weight = p1.conf + p2.conf;
    if total_weight == 0 {
        return Err(ErrorCode::PriceStale.into());
    }
    
    let weighted_price = (p1.price * p2.conf as i64 + p2.price * p1.conf as i64) / total_weight as i64;
    let avg_confidence = (p1.conf + p2.conf) / 2;
    let max_timestamp = std::cmp::max(p1.timestamp, p2.timestamp);
    
    msg!("✅ Weighted consensus: ${:.2} (confidence: {})", 
         weighted_price as f64 * 10f64.powi(p1.expo), avg_confidence);
    
    Ok(OraclePrice {
        price: weighted_price,
        expo: p1.expo,
        conf: avg_confidence,
        timestamp: max_timestamp,
    })
}

/// Calculate median consensus between two oracle prices
/// 
/// # Arguments
/// * `p1` - First oracle price
/// * `p2` - Second oracle price
/// 
/// # Returns
/// * `Result<OraclePrice>` - Median consensus price
fn calculate_median_consensus(p1: OraclePrice, p2: OraclePrice) -> Result<OraclePrice> {
    let price1_scaled = p1.get_scaled_price();
    let price2_scaled = p2.get_scaled_price();
    
    let median_price = if price1_scaled < price2_scaled {
        p1
    } else {
        p2
    };
    
    msg!("✅ Median consensus: ${:.2}", median_price.get_scaled_price());
    
    Ok(median_price)
}

/// Calculate price difference percentage between two prices
/// 
/// # Arguments
/// * `p1` - First oracle price
/// * `p2` - Second oracle price
/// 
/// # Returns
/// * `f64` - Price difference percentage
fn calculate_price_difference_percentage(p1: &OraclePrice, p2: &OraclePrice) -> f64 {
    let price1_scaled = p1.get_scaled_price();
    let price2_scaled = p2.get_scaled_price();
    
    let avg_price = (price1_scaled + price2_scaled) / 2.0;
    let diff = (price1_scaled - price2_scaled).abs();
    
    if avg_price > 0.0 {
        (diff / avg_price) * 100.0
    } else {
        100.0
    }
}

/// Calculate confidence score for consensus result
/// 
/// # Arguments
/// * `p1` - First oracle price
/// * `p2` - Second oracle price
/// 
/// # Returns
/// * `f64` - Confidence score (0.0 to 1.0)
fn calculate_confidence_score(p1: &OraclePrice, p2: &OraclePrice) -> f64 {
    let price_diff_percentage = calculate_price_difference_percentage(p1, p2);
    
    // Higher confidence for smaller price differences
    let price_consensus_score = if price_diff_percentage < 1.0 {
        1.0
    } else if price_diff_percentage < 3.0 {
        0.9
    } else if price_diff_percentage < 5.0 {
        0.8
    } else {
        0.7
    };
    
    // Factor in individual oracle confidence
    let avg_confidence = (p1.conf + p2.conf) as f64 / 2.0;
    let confidence_score = avg_confidence / 1000.0; // Normalize confidence
    
    // Combine scores
    (price_consensus_score + confidence_score) / 2.0
}

/// Validate consensus price for security and staleness
/// 
/// # Arguments
/// * `price` - The consensus price to validate
/// * `max_staleness` - Maximum allowed staleness in seconds
/// 
/// # Returns
/// * `Result<()>` - Success if price passes validation
fn validate_consensus_price(price: &OraclePrice, max_staleness: i64) -> Result<()> {
    let current_time = Clock::get()?.unix_timestamp;
    
    // Staleness check
    if current_time - price.timestamp > max_staleness {
        msg!("⚠️ Consensus price is stale: {} seconds old", current_time - price.timestamp);
        return Err(ErrorCode::PriceStale.into());
    }
    
    // Price band validation
    let price_scaled = price.get_scaled_price();
    if price_scaled < 50.0 || price_scaled > 500.0 {
        msg!("⚠️ Consensus price outside acceptable range: ${:.2}", price_scaled);
        return Err(ErrorCode::PriceStale.into());
    }
    
    // Confidence validation
    let conf_scaled = price.conf as f64 * 10f64.powi(price.expo);
    let conf_percentage = if price_scaled.abs() > 0.0 {
        (conf_scaled / price_scaled.abs()) * 100.0
    } else {
        100.0
    };
    
    if conf_percentage > 1.5 {
        msg!("⚠️ Consensus confidence interval too wide: {:.2}%", conf_percentage);
        return Err(ErrorCode::PriceStale.into());
    }
    
    Ok(())
}

/// Validate single oracle price with extra security checks
/// 
/// # Arguments
/// * `price` - The single oracle price to validate
/// * `max_staleness` - Maximum allowed staleness in seconds
/// 
/// # Returns
/// * `Result<()>` - Success if price passes validation
fn validate_single_oracle_price(price: &OraclePrice, max_staleness: i64) -> Result<()> {
    let current_time = Clock::get()?.unix_timestamp;
    
    // Staleness check
    if current_time - price.timestamp > max_staleness {
        msg!("⚠️ Single oracle price is stale: {} seconds old", current_time - price.timestamp);
        return Err(ErrorCode::PriceStale.into());
    }
    
    // Extra strict validation for single oracle
    let price_scaled = price.get_scaled_price();
    if price_scaled < 50.0 || price_scaled > 500.0 {
        msg!("⚠️ Single oracle price outside acceptable range: ${:.2}", price_scaled);
        return Err(ErrorCode::PriceStale.into());
    }
    
    // Stricter confidence check for single oracle
    let conf_scaled = price.conf as f64 * 10f64.powi(price.expo);
    let conf_percentage = if price_scaled.abs() > 0.0 {
        (conf_scaled / price_scaled.abs()) * 100.0
    } else {
        100.0
    };
    
    if conf_percentage > 0.8 {
        msg!("⚠️ Single oracle confidence interval too wide: {:.2}%", conf_percentage);
        return Err(ErrorCode::PriceStale.into());
    }
    
    Ok(())
}

/// Cache consensus result for future use
/// 
/// # Arguments
/// * `cache_account` - The cache account to store the result
/// * `price` - The consensus price to cache
/// 
/// # Returns
/// * `Result<()>` - Success if caching completed
fn cache_consensus_result(_cache_account: &mut AccountInfo, price: &OraclePrice) -> Result<()> {
    msg!("💾 Caching consensus result: ${:.2}", price.get_scaled_price());
    
    // TODO: Implement actual caching logic when price cache account structure is defined
    // This is a placeholder for the caching functionality
    
    Ok(())
}

/// Get consensus price with fallback mechanisms
/// 
/// # Arguments
/// * `ctx` - The context containing oracle feeds
/// * `max_staleness` - Maximum allowed staleness in seconds
/// * `fallback_price_usd` - Fallback price in USD
/// 
/// # Returns
/// * `Result<ConsensusResult>` - Consensus price result with fallback if needed
pub fn get_consensus_price_with_fallback(
    ctx: Context<MultiOracleConsensus>,
    max_staleness: i64,
    fallback_price_usd: Option<f64>,
) -> Result<ConsensusResult> {
    match get_consensus_price(ctx, max_staleness) {
        Ok(result) => Ok(result),
        Err(_) => {
            // All oracles failed, use fallback
            if let Some(price_usd) = fallback_price_usd {
                msg!("⚠️ All oracles failed, using fallback price: ${:.2}", price_usd);
                
                let fallback_price = OraclePrice {
                    price: (price_usd * 10f64.powi(6)) as i64,
                    expo: -6,
                    conf: 0,
                    timestamp: Clock::get()?.unix_timestamp,
                };
                
                Ok(ConsensusResult {
                    price: fallback_price,
                    provider: OracleProvider::Pyth, // Default provider
                    confidence_score: 0.5, // Low confidence for fallback
                    consensus_method: ConsensusMethod::Fallback,
                })
            } else {
                Err(ErrorCode::AllOraclesFailed.into())
            }
        }
    }
}
