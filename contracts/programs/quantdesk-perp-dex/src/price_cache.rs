//! Price cache account structures
//! This module provides efficient price caching mechanisms for oracle performance optimization

use anchor_lang::prelude::*;
use crate::oracle::OraclePrice;
use crate::ErrorCode;

/// Price cache account for storing validated oracle prices
#[account]
pub struct PriceCache {
    /// Cached prices for different assets
    pub prices: Vec<CachedPrice>,
    /// Last update timestamp
    pub last_update: i64,
    /// Staleness threshold in seconds
    pub staleness_threshold: i64,
    /// Total number of updates
    pub update_count: u64,
    /// Cache version for migration support
    pub version: u32,
    /// Reserved space for future extensions
    pub reserved: [u8; 32],
}

/// Individual cached price entry
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug, PartialEq)]
pub struct CachedPrice {
    /// Asset symbol (e.g., "SOL", "BTC")
    pub asset: String,
    /// Price value
    pub price: i64,
    /// Price exponent
    pub expo: i32,
    /// Confidence interval
    pub conf: u64,
    /// Timestamp when price was cached
    pub timestamp: i64,
    /// Oracle provider that provided this price
    pub provider: OracleProvider,
    /// Consensus method used
    pub consensus_method: ConsensusMethod,
}

/// Oracle provider enumeration for cache
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum OracleProvider {
    Pyth,
    Switchboard,
    Chainlink,
    Fallback,
}

/// Consensus method enumeration for cache
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Copy, PartialEq, Eq, Debug)]
pub enum ConsensusMethod {
    WeightedAverage,
    Median,
    SingleOracle,
    Fallback,
}

impl PriceCache {
    /// Calculate the space required for the account
    pub const INIT_SPACE: usize = 8 + // Account discriminator
        4 + // Vec length
        (32 * 10) + // 10 cached prices (32 bytes each)
        8 + // last_update
        8 + // staleness_threshold
        8 + // update_count
        4 + // version
        32; // reserved space

    /// Create a new price cache account
    pub fn new(staleness_threshold: i64) -> Self {
        Self {
            prices: Vec::new(),
            last_update: 0,
            staleness_threshold,
            update_count: 0,
            version: 1,
            reserved: [0u8; 32],
        }
    }

    /// Add or update a cached price
    pub fn cache_price(
        &mut self,
        asset: String,
        price: OraclePrice,
        provider: OracleProvider,
        consensus_method: ConsensusMethod,
    ) -> Result<()> {
        let current_time = Clock::get()?.unix_timestamp;
        
        // Check if price already exists
        if let Some(existing_price) = self.prices.iter_mut().find(|p| p.asset == asset) {
            // Update existing price
            existing_price.price = price.price;
            existing_price.expo = price.expo;
            existing_price.conf = price.conf;
            existing_price.timestamp = price.timestamp;
            existing_price.provider = provider;
            existing_price.consensus_method = consensus_method;
        } else {
            // Add new price
            self.prices.push(CachedPrice {
                asset,
                price: price.price,
                expo: price.expo,
                conf: price.conf,
                timestamp: price.timestamp,
                provider,
                consensus_method,
            });
        }
        
        self.last_update = current_time;
        self.update_count += 1;
        
        msg!("💾 Cached price for {}: ${:.2}", 
             self.prices.last().unwrap().asset, 
             price.get_scaled_price());
        
        Ok(())
    }

    /// Get cached price if available and fresh
    pub fn get_cached_price(&self, asset: &str) -> Option<OraclePrice> {
        let current_time = Clock::get().unwrap().unix_timestamp;
        
        if let Some(cached_price) = self.prices.iter().find(|p| p.asset == asset) {
            // Check if price is still fresh
            if current_time - cached_price.timestamp < self.staleness_threshold {
                msg!("✅ Using cached price for {}: ${:.2}", 
                     asset, 
                     cached_price.price as f64 * 10f64.powi(cached_price.expo));
                
                return Some(OraclePrice {
                    price: cached_price.price,
                    expo: cached_price.expo,
                    conf: cached_price.conf,
                    timestamp: cached_price.timestamp,
                });
            } else {
                msg!("⚠️ Cached price for {} is stale: {} seconds old", 
                     asset, current_time - cached_price.timestamp);
            }
        }
        
        None
    }

    /// Remove stale prices from cache
    pub fn cleanup_stale_prices(&mut self) -> Result<usize> {
        let current_time = Clock::get()?.unix_timestamp;
        let initial_count = self.prices.len();
        
        self.prices.retain(|price| {
            current_time - price.timestamp < self.staleness_threshold
        });
        
        let removed_count = initial_count - self.prices.len();
        if removed_count > 0 {
            msg!("🧹 Cleaned up {} stale prices from cache", removed_count);
        }
        
        Ok(removed_count)
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> CacheStats {
        let current_time = Clock::get().unwrap().unix_timestamp;
        let fresh_prices = self.prices.iter()
            .filter(|p| current_time - p.timestamp < self.staleness_threshold)
            .count();
        
        CacheStats {
            total_prices: self.prices.len(),
            fresh_prices,
            stale_prices: self.prices.len() - fresh_prices,
            last_update: self.last_update,
            update_count: self.update_count,
            staleness_threshold: self.staleness_threshold,
        }
    }

    /// Validate cache integrity
    pub fn validate_cache(&self) -> Result<()> {
        let current_time = Clock::get()?.unix_timestamp;
        
        // Check for duplicate assets
        let mut seen_assets = std::collections::HashSet::new();
        for price in &self.prices {
            if !seen_assets.insert(&price.asset) {
                return Err(ErrorCode::PriceStale.into()); // Use existing error instead of InvalidCache
            }
        }
        
        // Check for reasonable price ranges
        for price in &self.prices {
            let price_scaled = price.price as f64 * 10f64.powi(price.expo);
            if price_scaled < 0.01 || price_scaled > 1000000.0 {
                msg!("⚠️ Suspicious cached price for {}: ${:.2}", price.asset, price_scaled);
            }
        }
        
        // Check cache age
        if current_time - self.last_update > self.staleness_threshold * 2 {
            msg!("⚠️ Cache is very old: {} seconds", current_time - self.last_update);
        }
        
        Ok(())
    }
}

/// Cache statistics structure
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_prices: usize,
    pub fresh_prices: usize,
    pub stale_prices: usize,
    pub last_update: i64,
    pub update_count: u64,
    pub staleness_threshold: i64,
}

/// Context for price cache operations
#[derive(Accounts)]
pub struct PriceCacheOperations<'info> {
    #[account(mut)]
    pub signer: Signer<'info>,
    /// CHECK: Price cache account
    #[account(mut)]
    pub price_cache: AccountInfo<'info>,
}

/// Initialize price cache account
/// 
/// # Arguments
/// * `ctx` - The context containing the cache account
/// * `staleness_threshold` - Staleness threshold in seconds
/// 
/// # Returns
/// * `Result<()>` - Success if initialization completed
pub fn initialize_price_cache(
    ctx: Context<PriceCacheOperations>,
    staleness_threshold: i64,
) -> Result<()> {
    let cache = &mut ctx.accounts.price_cache;
    
    // Initialize cache account
    let mut price_cache = PriceCache::new(staleness_threshold);
    
    // Serialize and store
    let serialized = price_cache.try_to_vec()?;
    cache.try_borrow_mut_data()?[..serialized.len()].copy_from_slice(&serialized);
    
    msg!("✅ Price cache initialized with staleness threshold: {} seconds", staleness_threshold);
    
    Ok(())
}

/// Update price cache with new price
/// 
/// # Arguments
/// * `ctx` - The context containing the cache account
/// * `asset` - Asset symbol
/// * `price` - Oracle price to cache
/// * `provider` - Oracle provider
/// * `consensus_method` - Consensus method used
/// 
/// # Returns
/// * `Result<()>` - Success if update completed
pub fn update_price_cache(
    ctx: Context<PriceCacheOperations>,
    asset: String,
    price: OraclePrice,
    provider: OracleProvider,
    consensus_method: ConsensusMethod,
) -> Result<()> {
    let cache_data = &mut ctx.accounts.price_cache.try_borrow_mut_data()?;
    let mut price_cache = PriceCache::try_from_slice(cache_data)?;
    
    price_cache.cache_price(asset, price, provider, consensus_method)?;
    
    // Serialize and store updated cache
    let serialized = price_cache.try_to_vec()?;
    cache_data[..serialized.len()].copy_from_slice(&serialized);
    
    Ok(())
}

/// Get cached price if available and fresh
/// 
/// # Arguments
/// * `ctx` - The context containing the cache account
/// * `asset` - Asset symbol to retrieve
/// 
/// # Returns
/// * `Result<Option<OraclePrice>>` - Cached price if available and fresh
pub fn get_cached_price(
    ctx: Context<PriceCacheOperations>,
    asset: String,
) -> Result<Option<OraclePrice>> {
    let cache_data = &ctx.accounts.price_cache.try_borrow_data()?;
    let price_cache = PriceCache::try_from_slice(cache_data)?;
    
    Ok(price_cache.get_cached_price(&asset))
}

/// Cleanup stale prices from cache
/// 
/// # Arguments
/// * `ctx` - The context containing the cache account
/// 
/// # Returns
/// * `Result<usize>` - Number of stale prices removed
pub fn cleanup_price_cache(ctx: Context<PriceCacheOperations>) -> Result<usize> {
    let cache_data = &mut ctx.accounts.price_cache.try_borrow_mut_data()?;
    let mut price_cache = PriceCache::try_from_slice(cache_data)?;
    
    let removed_count = price_cache.cleanup_stale_prices()?;
    
    // Serialize and store updated cache
    let serialized = price_cache.try_to_vec()?;
    cache_data[..serialized.len()].copy_from_slice(&serialized);
    
    Ok(removed_count)
}

/// Get cache statistics
/// 
/// # Arguments
/// * `ctx` - The context containing the cache account
/// 
/// # Returns
/// * `Result<CacheStats>` - Cache statistics
pub fn get_cache_stats(ctx: Context<PriceCacheOperations>) -> Result<CacheStats> {
    let cache_data = &ctx.accounts.price_cache.try_borrow_data()?;
    let price_cache = PriceCache::try_from_slice(cache_data)?;
    
    Ok(price_cache.get_cache_stats())
}

/// Validate cache integrity
/// 
/// # Arguments
/// * `ctx` - The context containing the cache account
/// 
/// # Returns
/// * `Result<()>` - Success if cache is valid
pub fn validate_price_cache(ctx: Context<PriceCacheOperations>) -> Result<()> {
    let cache_data = &ctx.accounts.price_cache.try_borrow_data()?;
    let price_cache = PriceCache::try_from_slice(cache_data)?;
    
    price_cache.validate_cache()?;
    
    msg!("✅ Price cache validation passed");
    
    Ok(())
}
