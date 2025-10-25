/**
 * QuantDesk Protocol Utilities
 */

import { PublicKey } from '@solana/web3.js'

/**
 * Format price with appropriate decimal places
 */
export function formatPrice(price: number, decimals: number = 2): string {
  return price.toFixed(decimals)
}

/**
 * Format percentage with appropriate decimal places
 */
export function formatPercentage(value: number, decimals: number = 2): string {
  return `${value.toFixed(decimals)}%`
}

/**
 * Format large numbers with K, M, B suffixes
 */
export function formatLargeNumber(value: number): string {
  if (value >= 1e9) {
    return `${(value / 1e9).toFixed(1)}B`
  } else if (value >= 1e6) {
    return `${(value / 1e6).toFixed(1)}M`
  } else if (value >= 1e3) {
    return `${(value / 1e3).toFixed(1)}K`
  }
  return value.toFixed(0)
}

/**
 * Calculate PnL percentage
 */
export function calculatePnLPercent(entryPrice: number, currentPrice: number, side: 'long' | 'short'): number {
  if (side === 'long') {
    return ((currentPrice - entryPrice) / entryPrice) * 100
  } else {
    return ((entryPrice - currentPrice) / entryPrice) * 100
  }
}

/**
 * Calculate liquidation price
 */
export function calculateLiquidationPrice(
  entryPrice: number,
  side: 'long' | 'short',
  leverage: number,
  maintenanceMargin: number = 0.05
): number {
  if (side === 'long') {
    return entryPrice * (1 - (1 / leverage) + maintenanceMargin)
  } else {
    return entryPrice * (1 + (1 / leverage) - maintenanceMargin)
  }
}

/**
 * Validate Solana address
 */
export function isValidSolanaAddress(address: string): boolean {
  try {
    new PublicKey(address)
    return true
  } catch {
    return false
  }
}

/**
 * Generate random order ID
 */
export function generateOrderId(): string {
  return `order_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
}

/**
 * Calculate order size in base asset
 */
export function calculateOrderSize(quoteAmount: number, price: number): number {
  return quoteAmount / price
}

/**
 * Calculate order value in quote asset
 */
export function calculateOrderValue(baseAmount: number, price: number): number {
  return baseAmount * price
}

/**
 * Round to nearest tick size
 */
export function roundToTickSize(price: number, tickSize: number): number {
  return Math.round(price / tickSize) * tickSize
}

/**
 * Calculate margin requirement
 */
export function calculateMarginRequirement(
  positionSize: number,
  price: number,
  leverage: number
): number {
  return (positionSize * price) / leverage
}

/**
 * Calculate available margin
 */
export function calculateAvailableMargin(
  totalBalance: number,
  usedMargin: number
): number {
  return totalBalance - usedMargin
}

/**
 * Format timestamp to readable string
 */
export function formatTimestamp(timestamp: string | number): string {
  const date = new Date(timestamp)
  return date.toLocaleString()
}

/**
 * Calculate time until expiration
 */
export function getTimeUntilExpiration(expirationTime: string | number): string {
  const now = Date.now()
  const expiration = typeof expirationTime === 'string' ? new Date(expirationTime).getTime() : expirationTime
  const diff = expiration - now

  if (diff <= 0) return 'Expired'

  const days = Math.floor(diff / (1000 * 60 * 60 * 24))
  const hours = Math.floor((diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60))
  const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60))

  if (days > 0) return `${days}d ${hours}h`
  if (hours > 0) return `${hours}h ${minutes}m`
  return `${minutes}m`
}
