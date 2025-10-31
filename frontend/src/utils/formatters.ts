import Decimal from 'decimal.js';

/**
 * Currency Formatters for QuantDesk
 * Following best practices from Drift Protocol and expert guidance
 * 
 * Smart Contract Storage:
 * - total_collateral: USD value with 6 decimals (e.g., $93.00 = 93,000,000)
 * 
 * Frontend Display:
 * - Divide by 1e6 to get USD value
 * - Use Pyth oracle for SOL/USD conversions
 * - Use decimal.js to avoid floating-point errors
 */

/**
 * Format USD collateral value from on-chain representation
 * @param collateralRaw - Raw value from smart contract (6 decimals)
 * @returns Formatted USD string (e.g., "$93.00")
 */
export function formatCollateralUSD(collateralRaw: number | string): string {
  const usdValue = new Decimal(collateralRaw).dividedBy(1_000_000); // Divide by 10^6
  return `$${usdValue.toFixed(2)}`;
}

/**
 * Get USD value as a number from on-chain representation
 * @param collateralRaw - Raw value from smart contract (6 decimals)
 * @returns USD value as number (e.g., 93.00)
 */
export function getCollateralUSD(collateralRaw: number | string): number {
  const usdValue = new Decimal(collateralRaw).dividedBy(1_000_000);
  return usdValue.toNumber();
}

/**
 * Format collateral with SOL equivalent
 * @param collateralRaw - Raw value from smart contract (6 decimals)
 * @param solPrice - Current SOL/USD price from Pyth oracle
 * @returns Formatted string with both USD and SOL (e.g., "$93.00 (≈0.4500 SOL)")
 */
export function formatCollateralWithSOL(
  collateralRaw: number | string,
  solPrice: number
): string {
  const usdValue = new Decimal(collateralRaw).dividedBy(1_000_000);
  const solEquiv = usdValue.dividedBy(solPrice);
  return `$${usdValue.toFixed(2)} (≈${solEquiv.toFixed(4)} SOL)`;
}

/**
 * Get SOL equivalent from USD collateral value
 * @param collateralRaw - Raw value from smart contract (6 decimals)
 * @param solPrice - Current SOL/USD price from Pyth oracle
 * @returns SOL amount as number (e.g., 0.4500)
 */
export function getSOLEquivalent(
  collateralRaw: number | string,
  solPrice: number
): number {
  const usdValue = new Decimal(collateralRaw).dividedBy(1_000_000);
  const solEquiv = usdValue.dividedBy(solPrice);
  return solEquiv.toNumber();
}

/**
 * Format collateral for display with currency toggle
 * @param collateralRaw - Raw value from smart contract (6 decimals)
 * @param solPrice - Current SOL/USD price from Pyth oracle
 * @param displayMode - Display preference: 'usd' or 'sol'
 * @returns Formatted string based on display mode
 */
export function formatCollateral(
  collateralRaw: number | string,
  solPrice: number,
  displayMode: 'usd' | 'sol' = 'usd'
): string {
  const usdValue = new Decimal(collateralRaw).dividedBy(1_000_000);
  
  if (displayMode === 'usd') {
    return `$${usdValue.toFixed(2)} USD`;
  }
  
  const solEquiv = usdValue.dividedBy(solPrice);
  return `${solEquiv.toFixed(4)} SOL (~$${usdValue.toFixed(2)})`;
}

/**
 * Convert SOL amount to USD using Pyth price
 * @param solAmount - Amount in SOL (e.g., 0.45)
 * @param solPrice - Current SOL/USD price from Pyth oracle
 * @returns USD value as string (e.g., "$93.15")
 */
export function solToUSD(solAmount: number, solPrice: number): string {
  const usdValue = new Decimal(solAmount).times(solPrice);
  return `$${usdValue.toFixed(2)}`;
}

/**
 * Convert SOL lamports to USD
 * @param lamports - Amount in lamports (e.g., 450000000)
 * @param solPrice - Current SOL/USD price from Pyth oracle
 * @returns USD value as string (e.g., "$93.15")
 */
export function lamportsToUSD(lamports: number | string, solPrice: number): string {
  const solAmount = new Decimal(lamports).dividedBy(1_000_000_000); // 1 SOL = 1e9 lamports
  const usdValue = solAmount.times(solPrice);
  return `$${usdValue.toFixed(2)}`;
}

/**
 * Format lamports as SOL
 * @param lamports - Amount in lamports (e.g., 450000000)
 * @returns Formatted SOL string (e.g., "0.4500 SOL")
 */
export function formatLamportsAsSOL(lamports: number | string): string {
  const solAmount = new Decimal(lamports).dividedBy(1_000_000_000);
  return `${solAmount.toFixed(4)} SOL`;
}

/**
 * Format USD amount for input (used in deposit/withdraw modals)
 * @param usdAmount - USD amount as number
 * @returns Formatted string for display
 */
export function formatUSDInput(usdAmount: number): string {
  return new Decimal(usdAmount).toFixed(2);
}

